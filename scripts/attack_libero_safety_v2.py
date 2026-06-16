"""
attack_libero_safety_v2.py

Drop-in replacement for attack_libero_safety.py that uses the corrected
attack machinery in attack_crossattn_v2.py:

  * target reference frozen on the clean latent (v1 used the perturbed latent)
  * matched xt: target branch xt is built from clean_latent + same (noise, t)
  * fresh noise drawn every PGD step (v1 forced noise_seed=0 in the
    targeted path, overfitting to one trajectory)
  * new --loss_type velocity, dropping the cross-attn hooks in favor of
    MSE between v(adv, benign) and v(clean, target).detach() — single-step
    only, ignores denoise_steps / guidance_scale.

The v1 script is untouched so the two can be A/B compared.

Usage (mirrors v1, just swap the script name).

Recommended commands — copy these as single lines.

Note on --alpha when --lf_attack is set:
  v2's --lf_attack dispatches to the new pure-LAB PGD (frequency-space code
  is on hold; --freq_cutoff and --freq_pixel_eps are ignored).  In this
  mode --alpha is a *fraction of each LAB channel's budget* per step:
    step_L  = alpha * lab_budget_L
    step_ab = alpha * lab_budget_ab
  e.g. --alpha 0.1 --lab_budget_ab 20 → 2.0-unit a*/b* step.
  Without --lf_attack, --alpha keeps its raw-PGD meaning (units of [-1,1]).

# Velocity loss + LAB attack (recommended starting point)
CUDA_VISIBLE_DEVICES=1 python cosmos-predict2.5/scripts/attack_libero_safety_v2.py --scene_name all --dataset_dir datasets/ --attack_num 10 --steps 300 --loss_type velocity --lf_attack --num_latent_video_frames 9 --num_latent_conditional_frames 2 --rand_prompts --alpha 0.05 --lab_budget_L 5.0 --lab_budget_ab 20.0

# KL loss + LAB attack with frozen target reference (mid-effort A/B against v1)
CUDA_VISIBLE_DEVICES=0 python cosmos-predict2.5/scripts/attack_libero_safety_v2.py --scene_name all --dataset_dir datasets/ --attack_num 10 --denoise_steps 6 --attack_layers_start 15 --steps 600 --loss_type kl --lf_attack --num_latent_video_frames 9 --num_latent_conditional_frames 1 --rand_prompts --alpha 0.05 --lab_budget_L 2.5 --lab_budget_ab 10.0

# Velocity loss + standard pixel PGD (no LAB; raw-alpha semantics)
CUDA_VISIBLE_DEVICES=1 python cosmos-predict2.5/scripts/attack_libero_safety_v2.py --scene_name all --dataset_dir datasets/ --attack_num 10 --steps 300 --loss_type velocity --num_latent_video_frames 9 --num_latent_conditional_frames 2 --rand_prompts --alpha 0.008 --eps 0.0628

# Velocity loss + LAB + moving target reference (cancels image-noise term in the loss)
CUDA_VISIBLE_DEVICES=1 python cosmos-predict2.5/scripts/attack_libero_safety_v2.py --scene_name all --dataset_dir datasets/ --attack_num 10 --steps 300 --loss_type velocity --target_ref adv --num_latent_video_frames 16 --num_latent_conditional_frames 1 --lf_attack --alpha 0.025 --lab_budget_L 5.0 --lab_budget_ab 20.0
"""

import json
import sys
import time
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
WM_ROOT    = REPO_ROOT.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(WM_ROOT))

# Dataset, prompts, arg-parser, and image loader come from v1 unchanged.
from attack_libero_safety import (   # noqa: E402
    SCENE_TARGET,
    load_dataset,
    load_image_as_video,
    build_arg_parser as _build_arg_parser_v1,
    DEFAULT_DATASET_DIR,
    build_prompt,
    build_prompt_variants,
)

# v2 attack machinery
from attack_crossattn_v2 import attack_single_video, pad_video   # noqa: E402

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference  # noqa: E402
from attack.eval_wm import eval as eval_pixel, eval_latent   # noqa: E402
from attack.shared_config import ckpt_path, experiment_name, config_file   # noqa: E402


def build_arg_parser():
    parser = _build_arg_parser_v1()
    # Add "velocity" as a valid loss_type option.
    for action in parser._actions:
        if action.dest == "loss_type":
            action.choices = ["cosine", "l2", "kl", "velocity"]
            action.help = (
                "Attack loss.  'cosine'/'l2'/'kl' minimise a cross-attention "
                "pattern distance against a FROZEN target reference computed "
                "on the clean image + target prompt (v2 fix).  'velocity' "
                "minimises MSE between v(adv,benign;xt_adv,t) and "
                "v(clean,target;xt_tgt,t).detach() — single-step random t, "
                "no cross-attn hooks, ignores --denoise_steps and "
                "--guidance_scale."
            )
            break
    parser.description = (
        "v2 attack against Cosmos-Predict 2.5: corrected target reference "
        "(clean latent), matched xt, fresh noise per PGD step, and an "
        "additional --loss_type velocity option."
    )
    parser.add_argument(
        "--probe_every", type=int, default=25,
        help="Run a held-out diagnostic probe (frozen t and noise, "
             "canonical prompts, single-step) every N PGD steps and print "
             "its loss + percent change vs step 1.  Because the per-step "
             "loss is noisy from random (t, noise) sampling, the probe is "
             "the real convergence signal.  0 = disabled.  Default: 25.",
    )
    parser.add_argument(
        "--target_ref", choices=["clean", "adv"], default="clean",
        help="What x0 the target branch uses as gt_frames and xt seed.  "
             "'clean' (default) → frozen (clean_latent, target_prompt) "
             "reference.  Loss compares (adv, benign) against this fixed "
             "reference; grows with |δ| because the image-perturbation "
             "term is in the loss.  'adv' → moving (adv_latent.detach(), "
             "target_prompt) reference; both branches share the same "
             "image so the loss is purely the cross-prompt gap.  Try "
             "'adv' if the probe shows image-noise domination.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.extend:
        raise SystemExit("--extend is not supported for the image dataset")

    if args.ckpt_path is None:
        args.ckpt_path = ckpt_path
    if args.experiment_name is None:
        args.experiment_name = experiment_name

    if args.split_gpus:
        assert torch.cuda.device_count() >= 2, "--split_gpus requires 2 CUDA devices"
        vae_device, dit_device = "cuda:0", "cuda:1"
        print(f"split_gpus: VAE on {vae_device}, DiT on {dit_device}")
    else:
        vae_device = dit_device = "cuda"

    H, W = [int(x) for x in args.resolution.split(",")]

    # ----- Resolve scenes -----
    if args.scene_name.strip().lower() == "all":
        wanted_scenes = sorted(SCENE_TARGET.keys())
    else:
        wanted_scenes = [s.strip() for s in args.scene_name.split(",") if s.strip()]
    bad = [s for s in wanted_scenes if s not in SCENE_TARGET]
    if bad:
        raise SystemExit(f"Unknown scene_name(s): {bad}.  Valid: {sorted(SCENE_TARGET)}")

    dataset_dir = Path(args.dataset_dir)
    metadata = load_dataset(dataset_dir)
    wanted_ids = None
    if args.scene_ids:
        wanted_ids = {s.strip() for s in args.scene_ids.split(",") if s.strip()}

    rows = []
    for scene in wanted_scenes:
        scene_rows = [r for r in metadata if r["scene_name"] == scene]
        if wanted_ids is not None:
            scene_rows = [r for r in scene_rows if r["scene_id"] in wanted_ids]
        if args.attack_num > 0:
            scene_rows = scene_rows[: args.attack_num]
        rows.extend(scene_rows)

    if not rows:
        raise SystemExit(
            f"No rows matched scene_name(s)={wanted_scenes} "
            f"(scene_ids={args.scene_ids})"
        )

    # ----- Build (image, benign, target, label, variants) jobs -----
    jobs = []
    for row in rows:
        img_path = row["img_path"]
        unsafe = row["unsafe_object"]
        unsafe_x, unsafe_y = row["unsafe_object_xy"]
        safes = row["safe_objects"] if args.use_both_safe else row["safe_objects"][:1]
        safes_xy = row["safe_objects_xy"] if args.use_both_safe else row["safe_objects_xy"][:1]
        for safe, (safe_x, safe_y) in zip(safes, safes_xy):
            benign = build_prompt(row["scene_name"], safe)
            target = build_prompt(row["scene_name"], unsafe)
            benign_variants = build_prompt_variants(row["scene_name"], safe, safe_x, safe_y)
            target_variants = build_prompt_variants(row["scene_name"], unsafe, unsafe_x, unsafe_y)
            target = target_variants[-2]
            label = (
                f"{row['scene_name']}_{row['scene_id']}__"
                f"{safe.replace(' ', '_')}_vs_{unsafe.replace(' ', '_')}"
            )
            jobs.append((img_path, benign, target, label, benign_variants, target_variants))

    print(f"\nPrepared {len(jobs)} job(s) for scene_name={wanted_scenes}")
    for img_path, benign, target, label, _bv, _tv in jobs[:3]:
        print(f"  [{label}]")
        print(f"    image  : {img_path.name}")
        print(f"    benign : {benign}")
        print(f"    target : {target}")
        if args.rand_prompts:
            print(f"    variants: {len(_bv)} benign, {len(_tv)} target")
    if len(jobs) > 3:
        print(f"  ... ({len(jobs) - 3} more)")

    # ----- Load model once -----
    print("\nLoading Video2World model...")
    inference = Video2WorldInference(
        experiment_name=args.experiment_name,
        ckpt_path=args.ckpt_path,
        s3_credential_path="",
        context_parallel_size=args.context_parallel_size,
        config_file=args.config_file,
        offload_diffusion_model=not args.load_diffusion_model,
        offload_text_encoder=not args.load_text_encoder,
        offload_tokenizer=not args.load_tokenizer,
    )
    model = inference.model
    state_t = model.config.state_t
    n_blocks = len(model.net.blocks)

    T_lat_attack = min(args.num_latent_video_frames, state_t)
    required_pixel_frames = (T_lat_attack - 1) * 4 + 1
    frames_to_extract = (args.num_latent_conditional_frames - 1) * 4 + 1
    T_tok = T_lat_attack

    layer_start = args.attack_layers_start
    layer_end = args.attack_layers_end
    effective_end = layer_end if layer_end is not None else n_blocks - 1

    print(f"Model         : {n_blocks} DiT blocks, state_t={state_t}")
    print(f"Attack T_lat  : {T_lat_attack}  (pixel frames: {required_pixel_frames})")
    print(f"Loss          : {args.loss_type}  blocks=[{layer_start}, {effective_end}]")
    print(f"PGD           : steps={args.steps} alpha={args.alpha:.5f} eps={args.eps:.4f}")
    if args.loss_type == "velocity" and args.denoise_steps != 1:
        print(f"[v2] velocity loss forces single-step; --denoise_steps={args.denoise_steps} ignored.")

    # ----- Output dir -----
    save_time = int(time.time())
    sweep_tag = "_".join(wanted_scenes) if len(wanted_scenes) <= 3 else "multi"
    sweep_prefix = "libero_safety_v2_infer" if args.inference_only else "libero_safety_v2"
    out_dir = WM_ROOT / "attack" / "outputs" / f"{sweep_prefix}_{sweep_tag}_{save_time}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_config = {
        **vars(args),
        "H": H,
        "W": W,
        "T_lat_attack": T_lat_attack,
        "required_pixel_frames": required_pixel_frames,
        "frames_to_extract": frames_to_extract,
        "T_tok": T_tok,
        "save_time": save_time,
        "out_dir": str(out_dir),
        "n_jobs": len(jobs),
        "script_version": "v2",
        "jobs": [
            {"image": str(p), "benign": b, "target": t, "label": l,
             "benign_variants": bv, "target_variants": tv}
            for (p, b, t, l, bv, tv) in jobs
        ],
    }
    with open(out_dir / "sweep_config.json", "w") as f:
        json.dump(sweep_config, f, indent=2)
    print(f"Sweep config saved to: {out_dir / 'sweep_config.json'}")

    compute_dtype = torch.bfloat16

    # ----- Run -----
    for i, (image_path, benign, target, label, benign_variants, target_variants) in enumerate(jobs):
        print(f"\n{'=' * 70}")
        print(f"Job {i + 1}/{len(jobs)}: {label}")
        print(f"  image  : {image_path}")
        print(f"  benign : {benign}")
        print(f"  target : {target}")
        print(f"{'=' * 70}")

        raw_padded = load_image_as_video(
            image_path,
            resolution=(H, W),
            frames_to_extract=frames_to_extract,
            required_pixel_frames=required_pixel_frames,
            device=vae_device,
        )

        args.target_prompt = target
        args.prompt = benign

        if args.inference_only:
            use_target = args.inference_prompt == "target"
            args.prompt = target if use_target else benign
            tag = args.inference_prompt
            print(f"\nInference-only mode ({tag} prompt): "
                  f"running world model on clean input...")
            print(f"  prompt: {args.prompt}")
            placeholder = out_dir / f"{label}_{tag}.pt"
            args.adv_path = out_dir
            existing = {p.name for p in out_dir.glob("generated_*.mp4")}
            eval_pixel(inference, raw_padded.cpu(), args, placeholder)
            new_files = [p for p in out_dir.glob("generated_*.mp4")
                         if p.name not in existing]
            if new_files:
                new_files.sort(key=lambda p: p.stat().st_mtime)
                renamed = out_dir / f"{label}_{tag}.mp4"
                new_files[-1].rename(renamed)
                print(f"  Saved: {renamed}")
            continue

        pseudo_path = out_dir / f"{label}.png"

        x_adv = attack_single_video(
            inference, model, pseudo_path, benign,
            H, W, frames_to_extract, required_pixel_frames,
            T_tok, layer_start, layer_end, compute_dtype, args,
            vae_device=vae_device, dit_device=dit_device, save_time=save_time,
            preloaded_raw_padded=raw_padded,
            benign_prompt_variants=benign_variants if args.rand_prompts else None,
            target_prompt_variants=target_variants if args.rand_prompts else None,
        )

        if args.skip_eval:
            continue

        if vae_device != dit_device:
            torch.cuda.set_device(vae_device)
            model.net = model.net.to(vae_device)
            if hasattr(model, "conditioner") and model.conditioner is not None:
                model.conditioner = model.conditioner.to(vae_device)
            model.tensor_kwargs["device"] = vae_device
            if hasattr(model, "tensor_kwargs_fp32"):
                model.tensor_kwargs_fp32["device"] = vae_device
            torch.cuda.empty_cache()

        attack_out_dir = WM_ROOT / "attack" / "outputs" / f"test_bch2{save_time}"
        base_name = f"{label}_adv.pt"
        args.adv_path = attack_out_dir
        args.prompt = benign

        attack_out_dir.mkdir(parents=True, exist_ok=True)
        prompts_json = attack_out_dir / f"{label}_prompts.json"
        with open(prompts_json, "w") as f:
            json.dump({"true_prompt": benign, "target_prompt": target}, f, indent=2)
        print(f"  Prompts saved: {prompts_json}")

        print("\nEvaluating on the world model with the benign prompt...")
        if args.skip_latent:
            eval_latent(inference, x_adv, args, attack_out_dir / base_name)
        else:
            eval_pixel(inference, x_adv, args, attack_out_dir / base_name)

    print(f"\nAll done.  Artifacts in: {out_dir}")
    print(f"            attacks in : {WM_ROOT / 'attack' / 'outputs' / f'test_bch2{save_time}'}")


if __name__ == "__main__":
    main()
