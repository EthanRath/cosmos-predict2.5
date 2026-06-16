"""
attack_libero_safety_v3.py

v3 entry-point.  Same as attack_libero_safety_v2.py except that it
drives attack_crossattn_v3.attack_single_video_v3, which adds an
optional unsafe-object spatial mask to the velocity-MSE loss
(--loss_type velocity).  v2 stays untouched.

The mask is gated on --mask_velocity.  Without that flag, this script
reproduces v2 behaviour exactly.

Recommended commands — copy these as single lines.

# v3 default: velocity + LAB attack + unsafe-object spatial mask
CUDA_VISIBLE_DEVICES=3 python cosmos-predict2.5/scripts/attack_libero_safety_v3.py --scene_name all --dataset_dir datasets/ --attack_num 10 --steps 300 --loss_type velocity --lf_attack --num_latent_video_frames 9 --num_latent_conditional_frames 2 --rand_prompts --alpha 0.025 --lab_budget_L 5.0 --lab_budget_ab 20.0 --mask_velocity

# v3 with both unsafe upweight (4×) and safe de-emphasis (0.5×)
CUDA_VISIBLE_DEVICES=1 python cosmos-predict2.5/scripts/attack_libero_safety_v3.py --scene_name all --dataset_dir datasets/ --attack_num 10 --steps 300 --loss_type velocity --lf_attack --num_latent_video_frames 9 --num_latent_conditional_frames 2 --rand_prompts --alpha 0.05 --lab_budget_L 5.0 --lab_budget_ab 20.0 --mask_velocity --mask_unsafe_weight 4.0 --mask_safe_weight 0.5

# v3 ablation: --mask_velocity OFF → identical optimisation to v2
CUDA_VISIBLE_DEVICES=1 python cosmos-predict2.5/scripts/attack_libero_safety_v3.py --scene_name all --dataset_dir datasets/ --attack_num 10 --steps 300 --loss_type velocity --lf_attack --num_latent_video_frames 9 --num_latent_conditional_frames 2 --rand_prompts --alpha 0.05 --lab_budget_L 5.0 --lab_budget_ab 20.0
"""

import json
import sys
import time
from pathlib import Path
import numpy as np

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
WM_ROOT    = REPO_ROOT.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(WM_ROOT))

from attack_libero_safety import (   # noqa: E402
    SCENE_TARGET,
    load_dataset,
    load_image_as_video,
    DEFAULT_DATASET_DIR,
    build_prompt,
    build_prompt_variants,
)
from attack_libero_safety_v2 import build_arg_parser as _build_arg_parser_v2  # noqa: E402

from attack_crossattn_v3 import attack_single_video_v3, pad_video   # noqa: E402

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference  # noqa: E402
from attack.eval_wm import eval as eval_pixel, eval_latent   # noqa: E402
from attack.shared_config import ckpt_path, experiment_name, config_file   # noqa: E402


def build_arg_parser():
    parser = _build_arg_parser_v2()
    parser.description = (
        "v3 attack against Cosmos-Predict 2.5: v2 + optional unsafe-object "
        "spatial mask on the velocity MSE loss."
    )
    parser.add_argument(
        "--mask_velocity", action="store_true",
        help="Enable spatial weighting mask on the velocity MSE loss.  "
             "Upweights the latent cells around the unsafe object's "
             "projected location and (optionally) de-emphasises the safe "
             "object's region.  Only effective with --loss_type velocity; "
             "silently ignored otherwise.",
    )
    parser.add_argument(
        "--mask_unsafe_weight", type=float, default=3.0,
        help="Peak multiplier at the unsafe-object latent cell (Gaussian "
             "bump).  Default 3.0 follows the road-mask weighting from "
             "the AFM paper.  1.0 = no effect.",
    )
    parser.add_argument(
        "--mask_safe_weight", type=float, default=1.0,
        help="Peak multiplier at the (first) safe-object latent cell.  "
             "Default 1.0 = no effect.  Set < 1 to de-emphasise the "
             "region we DON'T want to perturb.",
    )
    parser.add_argument(
        "--mask_radius_lat", type=float, default=3.0,
        help="Gaussian sigma in latent cells for the mask bumps.  "
             "Default 3 (≈ 24-pixel radius for an 8× VAE).",
    )
    parser.add_argument(
        "--mask_world_to_pix_scale", type=float, default=1.667,
        help="World-space → normalised-image scale factor used to project "
             "(hazard_x, hazard_y) onto the latent grid.  Default 1.667 "
             "= 1/0.6 maps |world_xy| < 0.6 inside the image.  No camera "
             "calibration in the repo, so this is a heuristic — the v3 "
             "script prints the resulting latent-cell centre at attack "
             "start; sanity-check it on a known scene before trusting "
             "results.",
    )
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument(
        "--velocity_lp_cutoff", type=float, default=0.0,
        help="If > 0, compute velocity MSE on spatially low-pass-filtered velocity "
             "fields, keeping the lowest lp_cutoff fraction of spatial frequencies "
             "in each dim. Focuses the loss on coarse motion structure and ignores "
             "high-frequency texture noise. 0.0 = disabled. Try 0.15–0.30.",
    )
    parser.add_argument(
        "--velocity_temporal_diff", action="store_true",
        help="Match temporal differences of velocity (v[t+1]-v[t]) instead of "
             "absolute frame-level velocities. Focuses the loss on scene dynamics "
             "(how motion evolves over time) rather than static per-frame predictions.",
    )
    parser.add_argument(
        "--inner_steps", type=int, default=1,
        help="Number of (t, noise) samples to average gradients over per PGD step. "
             "Each inner step is one forward-backward pass at a different t drawn "
             "uniformly from [--t_min, --t_max]. VRAM is unchanged; compute scales "
             "as inner_steps x per-step cost. Default 1 = original behaviour.",
    )
    parser.add_argument(
        "--t_min", type=float, default=0.25,
        help="Lower bound for uniform t sampling in the velocity loss "
             "(timestep = t_min * 1000 in model units). Default 0.25.",
    )
    parser.add_argument(
        "--t_max", type=float, default=0.75,
        help="Upper bound for uniform t sampling in the velocity loss "
             "(timestep = t_max * 1000 in model units). Default 0.75.",
    )
    parser.add_argument(
        "--loss_freq_plot", action="store_true",
        help="At every probe step (requires --probe_every > 0), save a PNG showing "
             "how the velocity MSE is distributed across spatial frequencies. "
             "Plots are written to <out_dir>/freq_plots/. Use this to check whether "
             "the loss is dominated by high-frequency noise (your hypothesis) or "
             "meaningfully concentrated in low-frequency (semantic) bands.",
    )
    parser.add_argument(
        "--freq_plot_n_bins", type=int, default=40,
        help="Number of radial-frequency bins for --loss_freq_plot. Default 40.",
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
            samples = np.random.choice(len(scene_rows), size = args.attack_num, replace = False)
            scene_rows = [scene_rows[i] for i in samples]
            # scene_rows = scene_rows[: args.attack_num]
        rows.extend(scene_rows)

    if not rows:
        raise SystemExit(
            f"No rows matched scene_name(s)={wanted_scenes} "
            f"(scene_ids={args.scene_ids})"
        )

    # Jobs now carry unsafe_xy and safe_xy (world coords) so attack_single_video_v3
    # can build the spatial mask.
    jobs = []
    for row in rows:
        img_path = row["img_path"]
        unsafe = row["unsafe_object"]
        unsafe_xy = row["unsafe_object_xy"]
        safes = row["safe_objects"] if args.use_both_safe else row["safe_objects"][:1]
        safes_xy = row["safe_objects_xy"] if args.use_both_safe else row["safe_objects_xy"][:1]
        for safe, safe_xy in zip(safes, safes_xy):
            benign = build_prompt(row["scene_name"], safe)
            target = build_prompt(row["scene_name"], unsafe)
            benign_variants = build_prompt_variants(row["scene_name"], safe, safe_xy[0], safe_xy[1])
            target_variants = build_prompt_variants(row["scene_name"], unsafe, unsafe_xy[0], unsafe_xy[1])
            target = target_variants[0]
            label = (
                f"{row['scene_name']}_{row['scene_id']}__"
                f"{safe.replace(' ', '_')}_vs_{unsafe.replace(' ', '_')}"
            )
            jobs.append((img_path, benign, target, label,
                         benign_variants, target_variants,
                         unsafe_xy, safe_xy))

    print(f"\nPrepared {len(jobs)} job(s) for scene_name={wanted_scenes}")
    for img_path, benign, target, label, _bv, _tv, _uxy, _sxy in jobs[:3]:
        print(f"  [{label}]")
        print(f"    image     : {img_path.name}")
        print(f"    benign    : {benign}")
        print(f"    target    : {target}")
        print(f"    unsafe_xy : {_uxy}   safe_xy : {_sxy}")
        if args.rand_prompts:
            print(f"    variants  : {len(_bv)} benign, {len(_tv)} target")
    if len(jobs) > 3:
        print(f"  ... ({len(jobs) - 3} more)")

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
        print(f"[v3] velocity loss forces single-step; --denoise_steps={args.denoise_steps} ignored.")
    if args.mask_velocity:
        if args.loss_type != "velocity":
            print(f"[v3] --mask_velocity has no effect with --loss_type {args.loss_type}; "
                  f"only the velocity loss honours the spatial mask.")
        else:
            print(f"[v3] spatial mask ENABLED  "
                  f"unsafe_w={args.mask_unsafe_weight}  safe_w={args.mask_safe_weight}  "
                  f"radius_lat={args.mask_radius_lat}  scale={args.mask_world_to_pix_scale}")
    if args.loss_type == "velocity":
        print(f"[v3] predicted-frames-only MSE: ALWAYS ON "
              f"(skipping first {args.num_latent_conditional_frames} conditioning frames)")
        print(f"[v3] t sampling: Uniform[{args.t_min:.2f}, {args.t_max:.2f}]"
              f"  (timesteps [{args.t_min*1000:.0f}, {args.t_max*1000:.0f}])")
        if args.inner_steps > 1:
            print(f"[v3] inner_steps={args.inner_steps}: averaging gradients over "
                  f"{args.inner_steps} t samples per PGD step  "
                  f"(effective gradient noise ÷√{args.inner_steps:.0f})")
        if args.velocity_lp_cutoff > 0:
            print(f"[v3] velocity LP filter: cutoff={args.velocity_lp_cutoff:.2f} "
                  f"(retaining lowest {args.velocity_lp_cutoff * 100:.0f}% of spatial freqs)")
        if args.velocity_temporal_diff:
            print(f"[v3] velocity temporal-diff loss: ENABLED "
                  f"(matching frame-to-frame velocity changes)")
    if args.loss_freq_plot:
        if not getattr(args, "probe_every", 0):
            print("[v3] WARNING: --loss_freq_plot has no effect without --probe_every > 0")
        else:
            print(f"[v3] frequency-loss plots: ENABLED  "
                  f"(saved to <out_dir>/freq_plots/,  bins={args.freq_plot_n_bins})")

    save_time = int(time.time())
    if args.save_name is None:
        exp_type = "ood" if "ood" in args.dataset_dir else "id"
        lp_str = f"_lp{args.velocity_lp_cutoff}" if args.velocity_lp_cutoff > 0 else ""
        td_str = "_td" if args.velocity_temporal_diff else ""
        args.save_name = (
            f"{exp_type}_{args.loss_type}_{args.lab_budget_L}_{args.lab_budget_ab}"
            f"_{args.mask_velocity}_{args.num_latent_conditional_frames}{lp_str}{td_str}"
        )
    sweep_tag = "_".join(wanted_scenes) if len(wanted_scenes) <= 3 else "multi"
    sweep_prefix = "libero_safety_v3_infer" if args.inference_only else "libero_safety_v3"
    out_dir = WM_ROOT / "attack" / "outputs" / f'{args.save_name}_{save_time}'
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
        "script_version": "v3",
        "jobs": [
            {"image": str(p), "benign": b, "target": t, "label": l,
             "benign_variants": bv, "target_variants": tv,
             "unsafe_xy": list(uxy), "safe_xy": list(sxy)}
            for (p, b, t, l, bv, tv, uxy, sxy) in jobs
        ],
    }
    with open(out_dir / "sweep_config.json", "w") as f:
        json.dump(sweep_config, f, indent=2)
    print(f"Sweep config saved to: {out_dir / 'sweep_config.json'}")

    compute_dtype = torch.bfloat16

    for i, (image_path, benign, target, label, benign_variants, target_variants,
            unsafe_xy, safe_xy) in enumerate(jobs):
        print(f"\n{'=' * 70}")
        print(f"Job {i + 1}/{len(jobs)}: {label}")
        print(f"  image     : {image_path}")
        print(f"  benign    : {benign}")
        print(f"  target    : {target}")
        print(f"  unsafe_xy : {unsafe_xy}   safe_xy : {safe_xy}")
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

        x_adv = attack_single_video_v3(
            inference, model, pseudo_path, benign,
            H, W, frames_to_extract, required_pixel_frames,
            T_tok, layer_start, layer_end, compute_dtype, args,
            vae_device=vae_device, dit_device=dit_device, save_time=save_time,
            preloaded_raw_padded=raw_padded,
            benign_prompt_variants=benign_variants if args.rand_prompts else None,
            target_prompt_variants=target_variants if args.rand_prompts else None,
            unsafe_xy=unsafe_xy,
            safe_xy=safe_xy,
            out_dir=out_dir,
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

        
        attack_out_dir = WM_ROOT / "attack" / "outputs" / f"{args.save_name}_{save_time}"
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
    print(f"            attacks in : {WM_ROOT / 'attack' / 'outputs' / f'{args.save_name}_{save_time}'}")


if __name__ == "__main__":
    main()
