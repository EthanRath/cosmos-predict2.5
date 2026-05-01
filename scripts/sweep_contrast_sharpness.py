"""
Sweep over contrast and sharpness modifications to test their effect on the
Video2World diffusion pipeline.

Loads a source video, applies a grid of (contrast, sharpness) perturbations in
pixel space (no gradient computation), then runs the full diffusion evaluation
for each setting and saves the generated videos.

The hypothesis: lower contrast / lower sharpness may make the input more
susceptible to watermark attacks, i.e., easier for an adversary to shift the
model's behaviour.

Output structure:
    attack/outputs/sweep_<timestamp>/
        orig.mp4                          -- unmodified source video
        c{contrast:.2f}_s{sharpness:.2f}/
            modified.mp4                  -- contrast/sharpness-adjusted input
            generated_<ts>.mp4            -- diffusion output
            full_<ts>.mp4                 -- prefix + generated

Usage:
    python scripts/sweep_contrast_sharpness.py \
        --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
        --prompt "robot arm picks up a block" \
        --resolution 432 432 \
        --num_latent_video_frames 4 \
        [--contrast_values 0.3 0.6 1.0] \
        [--sharpness_values 0.0 0.5 1.0]
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
    # TF.adjust_contrast expects (C, H, W) uint8 or float in [0, 1]
    v01 = (video.clamp(-1, 1) + 1) / 2.0          # [0, 1]
    B, C, T, H, W = v01.shape
    frames = v01[0].permute(1, 0, 2, 3)            # (T, C, H, W)
    out = torch.stack([TF.adjust_contrast(f, factor) for f in frames])  # (T, C, H, W)
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path",  required=True, help="Path to source .mp4")
    parser.add_argument("--prompt",      required=True, help="Text prompt for diffusion evaluation")
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
    args = parser.parse_args()

    device = "cuda"
    num_pixel_frames = (args.num_latent_video_frames - 1) * 4 + 1

    print(f"Resolution      : {args.resolution}")
    print(f"Latent T frames : {args.num_latent_video_frames}")
    print(f"Pixel T frames  : {num_pixel_frames}")
    print(f"Contrast sweep  : {args.contrast_values}")
    print(f"Sharpness sweep : {args.sharpness_values}")

    # ------------------------------------------------------------------
    # 1. Load and normalise the source video once
    # ------------------------------------------------------------------
    source_uint8 = load_and_preprocess_video(
        args.video_path, args.resolution, num_pixel_frames,
    )
    raw_state = normalize_video(source_uint8, device=device)   # (1, C, T, H, W) float32 [-1,1]
    print(f"Source shape    : {raw_state.shape}")

    # ------------------------------------------------------------------
    # 2. Prepare output directory
    # ------------------------------------------------------------------
    ts_run = int(time.time())
    out_root = WM_ROOT / "attack" / "outputs" / f"sweep_{ts_run}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Save unmodified original
    torchvision.io.write_video(
        str(out_root / "orig.mp4"),
        to_uint8_frames(raw_state),
        fps=16,
    )
    print(f"Saved original video to : {out_root / 'orig.mp4'}")

    # ------------------------------------------------------------------
    # 3. Load diffusion model once (reuse across all sweep points)
    # ------------------------------------------------------------------
    print("Loading Video2World diffusion model (once for entire sweep)...")
    from attack.eval_wm import eval
    inference = None   # lazy-init on first call; eval() handles it

    # ------------------------------------------------------------------
    # 4. Sweep
    # ------------------------------------------------------------------
    grid = list(itertools.product(args.contrast_values, args.sharpness_values))
    print(f"\nTotal sweep points : {len(grid)}")

    for i, (contrast, sharpness) in enumerate(grid):
        label = f"c{contrast:.2f}_s{sharpness:.2f}"
        print(f"\n[{i+1}/{len(grid)}] contrast={contrast:.2f}  sharpness={sharpness:.2f}")

        # Apply modifications
        x_mod = apply_contrast(raw_state, contrast)
        x_mod = apply_sharpness(x_mod, sharpness)
        x_mod = x_mod.clamp(-1, 1)

        # Save modified input
        cell_dir = out_root / label
        cell_dir.mkdir(exist_ok=True)
        torchvision.io.write_video(
            str(cell_dir / "modified.mp4"),
            to_uint8_frames(x_mod),
            fps=16,
        )

        # Run diffusion evaluation
        sweep_args = argparse.Namespace(
            prompt=args.prompt,
            num_latent_conditional_frames=args.num_latent_conditional_frames,
            adv_path=str(cell_dir / "modified.mp4"),   # used only for save-path derivation
        )

        if inference is None:
            from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
            from attack.shared_config import (
                ckpt_path, config_file, experiment_name,
                offload_text_encoder, offload_diffusion_model, offload_tokenizer,
                context_parallel_size,
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

        eval(inference, x_mod.cpu(), sweep_args)

    print(f"\nSweep complete. Results in: {out_root}")


if __name__ == "__main__":
    main()


"""
python cosmos-predict2.5/scripts/sweep_contrast_sharpness.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --prompt "robot arm picks up a block" \
    --resolution 432 432 \
    --num_latent_video_frames 4 \
    --contrast_values 0.2 0.5 1.0 \
    --sharpness_values 0.0 0.5 1.0
"""
