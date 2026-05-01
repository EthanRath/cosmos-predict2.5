"""
Sweep over contrast and sharpness modifications to test their effect on the
Video2World diffusion pipeline.

Loads a source video, applies a grid of (contrast, sharpness) perturbations in
pixel space (no gradient computation), then:

  - [always]             saves the modified input video for each sweep point
  - [if --prompt]        runs the full diffusion pipeline and saves generated videos
  - [if --target_prompt] evaluates the targeted L2 cross-attention loss from
                         attack_crossattn.py (no grad, no optimisation) and prints
                         a summary table over the full grid

The L2 loss measures how far the cross-attention pattern of (modified_input,
--prompt) is from the reference pattern of (original_input, --target_prompt).
A lower loss means the modification already pushes the model's attention toward
what an adversary would want — making those inputs more susceptible to attack.

Output structure:
    attack/outputs/sweep_<timestamp>/
        orig.mp4
        sweep_l2_loss.txt                 -- grid summary (if --target_prompt)
        c{contrast:.2f}_s{sharpness:.2f}/
            modified.mp4
            generated_<ts>.mp4            -- diffusion output (if --prompt)
            full_<ts>.mp4

Usage:
    python scripts/sweep_contrast_sharpness.py \
        --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
        --prompt "robot arm picks up a block" \
        --target_prompt "robot arm knocks over the block" \
        --resolution 432 432 \
        --num_latent_video_frames 4 \
        [--contrast_values 0.3 0.6 1.0] \
        [--sharpness_values 0.0 0.5 1.0] \
        [--attack_layers_start 0] \
        [--attack_layers_end 13]
"""

import argparse
import itertools
import sys
import time
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.functional as TF

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent          # cosmos-predict2.5/
WM_ROOT    = REPO_ROOT.parent           # WM_Poison/

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))     # needed to import attack_crossattn
sys.path.insert(0, str(WM_ROOT))

from probing.test_vae_encoder import load_and_preprocess_video, normalize_video  # noqa: E402
from attack.eval_wm import eval                                                   # noqa: E402
from attack.shared_config import vae_path                                         # noqa: E402


# ---------------------------------------------------------------------------
# Image-space transforms  (operate on float [-1, 1] tensors)
# ---------------------------------------------------------------------------

def apply_contrast(video: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Adjust contrast of each frame independently.
    video: (1, C, T, H, W) float32 in [-1, 1].
    factor=1.0 -> no change; factor=0.0 -> grey frame.
    """
    v01 = (video.clamp(-1, 1) + 1) / 2.0
    B, C, T, H, W = v01.shape
    frames = v01[0].permute(1, 0, 2, 3)            # (T, C, H, W)
    out = torch.stack([TF.adjust_contrast(f, factor) for f in frames])
    out = out.permute(1, 0, 2, 3).unsqueeze(0)      # (1, C, T, H, W)
    return out * 2.0 - 1.0


def apply_sharpness(video: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Adjust sharpness of each frame independently.
    video: (1, C, T, H, W) float32 in [-1, 1].
    factor=1.0 -> no change; factor=0.0 -> blurred.
    """
    v01 = (video.clamp(-1, 1) + 1) / 2.0
    B, C, T, H, W = v01.shape
    frames = v01[0].permute(1, 0, 2, 3)            # (T, C, H, W)
    out = torch.stack([TF.adjust_sharpness(f, factor) for f in frames])
    out = out.permute(1, 0, 2, 3).unsqueeze(0)
    return out * 2.0 - 1.0


def to_uint8_frames(t: torch.Tensor) -> torch.Tensor:
    """(1, C, T, H, W) float [-1,1] -> (T, H, W, C) uint8"""
    t = t[0].cpu().float()
    t = ((t.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    return t.permute(1, 2, 3, 0)


# ---------------------------------------------------------------------------
# Condition setup for L2 loss evaluation
# ---------------------------------------------------------------------------

def build_condition_templates(inference, model, x_padded, prompt, target_prompt,
                               num_latent_conditional_frames, device="cuda"):
    """
    Build condition and target_condition objects for L2 loss evaluation.

    Mirrors the setup in attack_single_video from attack_crossattn.py, but
    skips the uncondition (not needed for denoise_steps=1) and does not
    require gradients anywhere.

    Returns (condition, target_condition_template).
    The condition.gt_frames field will be overwritten per-call inside
    compute_crossattn_loss, so only the text embeddings need to be correct here.
    """
    import cosmos_predict2._src.predict2.inference.get_t5_emb as _t5_mod
    from cosmos_predict2._src.predict2.inference.video2world import _DEFAULT_NEGATIVE_PROMPT
    from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
        IS_PREPROCESSED_KEY,
    )

    compute_dtype = torch.bfloat16

    # Move text encoder to GPU for T5 embedding extraction
    if inference.offload_text_encoder:
        if (model.text_encoder is not None
                and hasattr(model.text_encoder, "model")
                and model.text_encoder.model is not None):
            model.text_encoder.model = model.text_encoder.model.to(device).eval()
        if _t5_mod.cosmos_encoder is not None:
            _t5_mod.cosmos_encoder.text_encoder = (
                _t5_mod.cosmos_encoder.text_encoder.to(device).eval()
            )

    video_bf16 = x_padded.to(device=device, dtype=compute_dtype)

    data_batch = inference._get_data_batch_input(
        video=video_bf16,
        prompt=prompt,
        num_conditional_frames=num_latent_conditional_frames,
        negative_prompt=_DEFAULT_NEGATIVE_PROMPT,
        use_neg_prompt=True,
    )
    data_batch["video"] = video_bf16
    data_batch[IS_PREPROCESSED_KEY] = True

    target_data_batch = inference._get_data_batch_input(
        video=video_bf16,
        prompt=target_prompt,
        num_conditional_frames=num_latent_conditional_frames,
        negative_prompt=_DEFAULT_NEGATIVE_PROMPT,
        use_neg_prompt=True,
    )
    target_data_batch["video"] = video_bf16
    target_data_batch[IS_PREPROCESSED_KEY] = True

    # Offload text encoder
    if inference.offload_text_encoder:
        if (model.text_encoder is not None
                and hasattr(model.text_encoder, "model")
                and model.text_encoder.model is not None):
            model.text_encoder.model = model.text_encoder.model.to("cpu")
        if _t5_mod.cosmos_encoder is not None:
            _t5_mod.cosmos_encoder.text_encoder = (
                _t5_mod.cosmos_encoder.text_encoder.to("cpu")
            )
        torch.cuda.empty_cache()

    # Move tokenizer + DiT to GPU
    if inference.offload_tokenizer:
        if hasattr(model.tokenizer, "encoder") and model.tokenizer.encoder is not None:
            model.tokenizer.encoder = model.tokenizer.encoder.to(device)
        torch.cuda.empty_cache()
    if inference.offload_diffusion_model:
        model.net = model.net.to(device)
        if hasattr(model, "conditioner") and model.conditioner is not None:
            model.conditioner = model.conditioner.to(device)
        torch.cuda.empty_cache()

    for p in model.net.parameters():
        p.requires_grad_(False)
    for p in model.tokenizer.model.model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        _, _, condition = model.get_data_and_condition(data_batch)

    condition = condition.edit_for_inference(
        is_cfg_conditional=True,
        num_conditional_frames=num_latent_conditional_frames,
    )

    # Build target_condition by copying the base condition (so _cond_mask and
    # all other fields are identical) and swapping in the target prompt's T5
    # embeddings — exactly as done in attack_single_video.
    tgt_dict = condition.to_dict(skip_underscore=False)
    tgt_dict["crossattn_emb"] = target_data_batch["t5_text_embeddings"]
    target_condition_template = type(condition)(**tgt_dict)

    return condition, target_condition_template


def eval_l2_loss(model, x_padded, condition, target_condition, T_tok,
                  num_latent_conditional_frames=2, layer_start=0, layer_end=None,
                  denoise_steps=36):
    """
    Compute the targeted L2 cross-attention loss for a single (possibly modified)
    video tensor — no gradients, no optimisation.

    Wraps compute_crossattn_loss from attack_crossattn.py with:
      - denoise_steps  : number of Euler denoising steps over which to aggregate
                         the attention loss (default 36 = full trajectory)
      - noise_seed=0   : deterministic across sweep points
      - loss_type="l2" : mean_l ||A_base^(l) - A_target^(l)||_F^2 / (B*H)

    Returns a float scalar.
    """
    from attack_crossattn import compute_crossattn_loss

    with torch.no_grad():
        loss, cleanup = compute_crossattn_loss(
            model=model,
            x_padded=x_padded,
            condition=condition,
            T_tok=T_tok,
            layer_start=layer_start,
            layer_end=layer_end,
            skip_latent=False,
            denoise_steps=denoise_steps,
            target_condition=target_condition,
            noise_seed=0,
            num_latent_conditional_frames=num_latent_conditional_frames,
            loss_type="l2",
        )
        cleanup()
    return loss.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", required=True, help="Path to source .mp4")
    parser.add_argument(
        "--prompt", default="",
        help="Text prompt for full diffusion evaluation. "
             "Omit to skip video generation and only compute L2 loss.",
    )
    parser.add_argument(
        "--target_prompt", default=None,
        help="Target prompt for L2 cross-attention loss evaluation. "
             "When provided, computes how each (contrast, sharpness) setting "
             "shifts the model's attention toward this prompt (no grad, no generation).",
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=[432, 432],
        metavar=("H", "W"),
    )
    parser.add_argument(
        "--num_latent_video_frames", type=int, default=4,
        help="Number of latent frames (pixel frames = (n-1)*4+1, default: 4)",
    )
    parser.add_argument(
        "--num_latent_conditional_frames", type=int, default=2,
        help="Number of latent conditioning frames (default: 2)",
    )
    parser.add_argument(
        "--contrast_values", type=float, nargs="+",
        default=[0.2, 0.5, 1.0],
        help="Contrast factors to sweep (1.0 = original)",
    )
    parser.add_argument(
        "--sharpness_values", type=float, nargs="+",
        default=[0.0, 0.5, 1.0],
        help="Sharpness factors to sweep (1.0 = original)",
    )
    parser.add_argument(
        "--attack_layers_start", type=int, default=0,
        help="First DiT block index included in the L2 loss (default: 0)",
    )
    parser.add_argument(
        "--attack_layers_end", type=int, default=None,
        help="Last DiT block index included in the L2 loss (default: all blocks)",
    )
    parser.add_argument(
        "--denoise_steps", type=int, default=36,
        help="(only active with --target_prompt) Number of Euler denoising steps used "
             "to compute the L2 cross-attention loss. Default 36 runs a full generation "
             "trajectory and also saves generated videos. Any value less than 36 computes "
             "the loss only (no video generation).",
    )
    args = parser.parse_args()

    if not args.prompt and not args.target_prompt:
        parser.error("At least one of --prompt or --target_prompt must be provided.")

    device = "cuda"
    num_pixel_frames = (args.num_latent_video_frames - 1) * 4 + 1
    T_tok = args.num_latent_video_frames   # latent temporal dimension used by DiT

    print(f"Resolution      : {args.resolution}")
    print(f"Latent T frames : {args.num_latent_video_frames}")
    print(f"Pixel T frames  : {num_pixel_frames}")
    print(f"Contrast sweep  : {args.contrast_values}")
    print(f"Sharpness sweep : {args.sharpness_values}")
    if args.target_prompt:
        print(f"Target prompt   : {args.target_prompt}")
        print(f"Denoise steps   : {args.denoise_steps}")
        print(f"Loss layers     : [{args.attack_layers_start}, "
              f"{args.attack_layers_end if args.attack_layers_end is not None else 'last'}]")

    # ------------------------------------------------------------------
    # 1. Load and normalise the source video once
    # ------------------------------------------------------------------
    source_uint8 = load_and_preprocess_video(
        args.video_path, args.resolution, num_pixel_frames,
    )
    raw_state = normalize_video(source_uint8, device=device)
    print(f"Source shape    : {raw_state.shape}")

    # ------------------------------------------------------------------
    # 2. Prepare output directory
    # ------------------------------------------------------------------
    ts_run = int(time.time())
    out_root = WM_ROOT / "attack" / "outputs" / f"sweep_{ts_run}"
    out_root.mkdir(parents=True, exist_ok=True)

    torchvision.io.write_video(
        str(out_root / "orig.mp4"), to_uint8_frames(raw_state), fps=16,
    )
    print(f"Saved original video to : {out_root / 'orig.mp4'}")

    # ------------------------------------------------------------------
    # 3. Load model (always needed — either for L2 loss or diffusion eval)
    # ------------------------------------------------------------------
    print("Loading Video2World model...")
    from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
    from attack.shared_config import (
        ckpt_path, config_file, experiment_name,
        offload_text_encoder, offload_diffusion_model, offload_tokenizer,
        context_parallel_size, num_steps as cfg_num_steps,
    )
    inference = Video2WorldInference(
        experiment_name=experiment_name,
        ckpt_path=ckpt_path,
        s3_credential_path="",
        context_parallel_size=context_parallel_size,
        config_file=config_file,
        offload_diffusion_model=offload_diffusion_model,
        offload_text_encoder=offload_text_encoder,
        offload_tokenizer=offload_tokenizer,
    )
    model = inference.model

    # ------------------------------------------------------------------
    # 4. Build condition templates once (text embeddings are prompt-level,
    #    not video-level; gt_frames is patched per-call inside compute_crossattn_loss)
    # ------------------------------------------------------------------
    condition_template = None
    target_condition_template = None
    if args.target_prompt:
        print("Building condition templates for L2 loss evaluation...")
        condition_template, target_condition_template = build_condition_templates(
            inference=inference,
            model=model,
            x_padded=raw_state,
            prompt=args.prompt if args.prompt else args.target_prompt,
            target_prompt=args.target_prompt,
            num_latent_conditional_frames=args.num_latent_conditional_frames,
            device=device,
        )
        print("Condition templates ready.")

    # ------------------------------------------------------------------
    # 5. Sweep
    # ------------------------------------------------------------------
    grid = list(itertools.product(args.contrast_values, args.sharpness_values))
    print(f"\nTotal sweep points : {len(grid)}")

    # (contrast, sharpness) -> l2_loss float
    loss_results: dict[tuple, float] = {}

    for i, (contrast, sharpness) in enumerate(grid):
        label = f"c{contrast:.2f}_s{sharpness:.2f}"
        print(f"\n[{i+1}/{len(grid)}] contrast={contrast:.2f}  sharpness={sharpness:.2f}")

        x_mod = apply_contrast(raw_state, contrast)
        x_mod = apply_sharpness(x_mod, sharpness)
        x_mod = x_mod.clamp(-1, 1)

        cell_dir = out_root / label
        cell_dir.mkdir(exist_ok=True)
        torchvision.io.write_video(
            str(cell_dir / "modified.mp4"), to_uint8_frames(x_mod), fps=16,
        )

        # -- L2 cross-attention loss (no grad) --
        if args.target_prompt:
            loss_val = eval_l2_loss(
                model=model,
                x_padded=x_mod,
                condition=condition_template,
                target_condition=target_condition_template,
                T_tok=T_tok,
                num_latent_conditional_frames=args.num_latent_conditional_frames,
                layer_start=args.attack_layers_start,
                layer_end=args.attack_layers_end,
                denoise_steps=args.denoise_steps,
            )
            loss_results[(contrast, sharpness)] = loss_val
            print(f"  L2 loss : {loss_val:.6f}")

        # -- Full diffusion generation --
        # Runs when --prompt is given AND either:
        #   - no --target_prompt (pure generation sweep), or
        #   - --denoise_steps >= cfg_num_steps (full trajectory was already computed
        #     for the loss, so we generate the video too)
        run_generation = args.prompt and (
            not args.target_prompt or args.denoise_steps >= cfg_num_steps
        )
        if run_generation:
            sweep_args = argparse.Namespace(
                prompt=args.prompt,
                num_latent_conditional_frames=args.num_latent_conditional_frames,
                adv_path=str(cell_dir / "modified.mp4"),
            )
            eval(inference, x_mod.cpu(), sweep_args)

    # ------------------------------------------------------------------
    # 6. Summary table
    # ------------------------------------------------------------------
    if loss_results:
        contrasts  = sorted(set(c for c, _ in loss_results))
        sharpnesses = sorted(set(s for _, s in loss_results))

        header = f"{'':>8s} | " + " | ".join(f"sharp={s:.2f}" for s in sharpnesses)
        sep    = "-" * len(header)
        lines  = [
            "L2 cross-attention loss  (lower = more susceptible to targeted attack)",
            sep, header, sep,
        ]
        for c in contrasts:
            row = f"cont={c:.2f} | " + " | ".join(
                f"{loss_results[(c, s)]:>12.6f}" for s in sharpnesses
            )
            lines.append(row)
        lines.append(sep)
        table_str = "\n".join(lines)

        print(f"\n{table_str}\n")
        summary_path = out_root / "sweep_l2_loss.txt"
        summary_path.write_text(
            f"source prompt  : {args.prompt}\n"
            f"target prompt  : {args.target_prompt}\n"
            f"denoise_steps  : {args.denoise_steps}\n"
            f"layers         : [{args.attack_layers_start}, "
            f"{args.attack_layers_end if args.attack_layers_end is not None else 'last'}]\n\n"
            + table_str + "\n"
        )
        print(f"Saved L2 loss summary to : {summary_path}")

    print(f"\nSweep complete. Results in: {out_root}")


if __name__ == "__main__":
    main()


"""
# Full generation + loss (default denoise_steps=36):
python cosmos-predict2.5/scripts/sweep_contrast_sharpness.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --prompt "robot arm picks up a block" \
    --target_prompt "robot arm knocks over the block" \
    --resolution 432 432 \
    --num_latent_video_frames 4 \
    --contrast_values 0.2 0.5 1.0 \
    --sharpness_values 0.0 0.5 1.0 \
    --attack_layers_end 13

# Loss only, fast (denoise_steps < 36):
python cosmos-predict2.5/scripts/sweep_contrast_sharpness.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --target_prompt "robot arm knocks over the block" \
    --resolution 432 432 \
    --num_latent_video_frames 4 \
    --contrast_values 0.2 0.5 1.0 \
    --sharpness_values 0.0 0.5 1.0 \
    --denoise_steps 1 \
    --attack_layers_end 13
"""
