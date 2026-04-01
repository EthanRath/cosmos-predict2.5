"""
Minimal script to test the Wan2.1 VAE encoder in isolation.

Replicates the preprocessing and encode() call from:
  - cosmos_predict2/_src/predict2/inference/video2world.py  (read_and_process_video)
  - cosmos_predict2/_src/predict2/models/text2world_model_rectified_flow.py
      (_normalize_video_databatch_inplace, get_data_and_condition -> encode)

Only the VAE is loaded into GPU memory; the diffusion model (DIT), conditioner,
and T5 text encoder are never instantiated.

Usage:
    python scripts/test_vae_encoder.py \
        --video_path /path/to/input.mp4 \
        --vae_pth /path/to/Wan2.1_VAE.pth \
        [--resolution 720 1280] \
        [--num_latent_conditional_frames 2] \
        [--num_latent_video_frames 31] \
        [--output_path /path/to/output_latent.pt]
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.functional as TF

# ---------------------------------------------------------------------------
# Add the repo root to sys.path so cosmos_predict2 is importable.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Video preprocessing  (mirrors inference/video2world.py)
# ---------------------------------------------------------------------------

def resize_input(video: torch.Tensor, resolution: list[int]) -> torch.Tensor:
    """Resize and centre-crop to [H, W].  Input: (T, C, H, W) uint8."""
    orig_h, orig_w = video.shape[2], video.shape[3]
    target_h, target_w = resolution
    scaling_ratio = max(target_w / orig_w, target_h / orig_h)
    resizing_shape = (
        int(math.ceil(scaling_ratio * orig_h)),
        int(math.ceil(scaling_ratio * orig_w)),
    )
    video = TF.resize(video, resizing_shape)
    video = TF.center_crop(video, resolution)
    return video


def load_and_preprocess_video(
    video_path: str,
    resolution: list[int],
    num_video_frames: int,
    num_latent_conditional_frames: int = 2,
) -> torch.Tensor:
    """
    Load a video, extract the last (4*(num_latent_conditional_frames-1)+1) frames,
    pad the remainder with the last frame, resize, and return a uint8 tensor of
    shape (1, C, T, H, W) ready for the model.

    Mirrors read_and_process_video() in inference/video2world.py exactly.
    """
    from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io

    if num_latent_conditional_frames not in [1, 2]:
        raise ValueError(
            f"num_latent_conditional_frames must be 1 or 2, got {num_latent_conditional_frames}"
        )

    video_frames, video_metadata = easy_io.load(video_path)  # (T, H, W, C) numpy
    print(f"Loaded video: shape={video_frames.shape}, metadata={video_metadata}")

    # (T, H, W, C) -> (C, T, H, W) float [0, 1]
    video_tensor = torch.from_numpy(video_frames).float() / 255.0
    video_tensor = video_tensor.permute(3, 0, 1, 2)

    available_frames = video_tensor.shape[1]
    frames_to_extract = 4 * (num_latent_conditional_frames - 1) + 1

    if available_frames < frames_to_extract:
        raise ValueError(
            f"Video has only {available_frames} frames but needs at least "
            f"{frames_to_extract} frames for num_latent_conditional_frames="
            f"{num_latent_conditional_frames}"
        )

    C, _, H, W = video_tensor.shape
    full_video = torch.zeros(C, num_video_frames, H, W)

    start_idx = available_frames - frames_to_extract
    extracted = video_tensor[:, start_idx:, :, :]
    full_video[:, :frames_to_extract, :, :] = extracted

    if frames_to_extract < num_video_frames:
        last_frame = extracted[:, -1:, :, :]
        padding = num_video_frames - frames_to_extract
        full_video[:, frames_to_extract:, :, :] = last_frame.repeat(1, padding, 1, 1)

    # (C, T, H, W) -> (T, C, H, W) for resize, then back
    full_video = full_video.permute(1, 0, 2, 3)            # (T, C, H, W)
    full_video = (full_video * 255.0).to(torch.uint8)
    full_video = resize_input(full_video, resolution)       # (T, C, H, W) uint8

    # (T, C, H, W) -> (1, C, T, H, W)
    full_video = full_video.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return full_video  # (1, C, T, H, W) uint8


# ---------------------------------------------------------------------------
# Normalization  (mirrors _normalize_video_databatch_inplace)
# ---------------------------------------------------------------------------

def normalize_video(video: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Convert uint8 (1, C, T, H, W) in [0, 255] -> float32 in [-1, 1] on device."""
    assert video.dtype == torch.uint8, f"Expected uint8, got {video.dtype}"
    return video.to(device=device, dtype=torch.float32) / 127.5 - 1.0


# ---------------------------------------------------------------------------
# VAE loader
# ---------------------------------------------------------------------------

def load_vae(vae_pth: str, device: str = "cuda") -> "Wan2pt1VAEInterface":
    """Instantiate only the Wan2pt1VAEInterface (VAE encoder/decoder)."""
    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface

    print(f"Loading VAE from: {vae_pth}")
    tokenizer = Wan2pt1VAEInterface(
        vae_pth=vae_pth,
        # no s3 credentials needed for a local path
        temporal_window=4,
        is_parallel=False,
        enable_grad=True,
    )
    # The WanVAE model was already placed on device='cuda' by default in _video_vae.
    # If the device arg is not cuda, move it.
    if device != "cuda":
        tokenizer.model.model = tokenizer.model.model.to(device)
    return tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", required=True, help="Path to input .mp4 video")
    parser.add_argument("--vae_pth", required=True, help="Path to Wan2.1_VAE.pth checkpoint")
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=[720, 1280],
        metavar=("H", "W"),
        help="Target resolution [H W] (default: 720 1280)",
    )
    parser.add_argument(
        "--num_latent_conditional_frames", type=int, default=2,
        help="Number of latent conditional frames from the input video (1 or 2). "
             "Determines how many pixel frames are extracted (4*(n-1)+1).",
    )
    parser.add_argument(
        "--num_latent_video_frames", type=int, default=31,
        help="Total latent temporal length expected by the model (default: 31). "
             "Pixel frames = (num_latent_video_frames - 1) * 4 + 1.",
    )
    parser.add_argument(
        "--output_path", default=None,
        help="Optional path to save the latent tensor (.pt)",
    )
    args = parser.parse_args()

    device = "cuda"
    num_pixel_frames = (args.num_latent_video_frames - 1) * 4 + 1

    print(f"Resolution       : {args.resolution}")
    print(f"Latent T frames  : {args.num_latent_video_frames}")
    print(f"Pixel T frames   : {num_pixel_frames}")

    # 1. Preprocess video
    video_uint8 = load_and_preprocess_video(
        video_path=args.video_path,
        resolution=args.resolution,
        num_video_frames=num_pixel_frames,
        num_latent_conditional_frames=args.num_latent_conditional_frames,
    )
    print(f"Preprocessed video shape : {video_uint8.shape}, dtype={video_uint8.dtype}")

    # 2. Normalize to [-1, 1] and move to GPU  (matches _normalize_video_databatch_inplace)
    raw_state = normalize_video(video_uint8, device=device).requires_grad_(True)
    print(f"Normalised raw_state     : shape={raw_state.shape}, dtype={raw_state.dtype}, "
          f"range=[{raw_state.min():.3f}, {raw_state.max():.3f}]")

    args.vae_pth = "/home/ethan/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/6787e176dce74a101d922174a95dba29fa5f0c55/tokenizer.pth"
    # 3. Load the VAE (only model on GPU)
    tokenizer = load_vae(args.vae_pth, device=device)

    vram_before = torch.cuda.memory_allocated(device) / 1e9
    print(f"VRAM after VAE load      : {vram_before:.2f} GB")

    # 4. Encode  (mirrors text2world_model_rectified_flow.py:816)
    print("Running encoder...")
    latent = tokenizer.encode(raw_state).contiguous().float()
    loss = latent.sum()
    loss.backward()
    print("Gradients ", raw_state.grad)  # gradient w.r.t. the input pixels

    print(f"Latent shape             : {latent.shape}")   # (1, 16, T_lat, H_lat, W_lat)
    print(f"Latent dtype             : {latent.dtype}")
    print(f"Latent range             : [{latent.min():.4f}, {latent.max():.4f}]")
    print(f"Latent mean / std        : {latent.mean():.4f} / {latent.std():.4f}")

    vram_after = torch.cuda.memory_allocated(device) / 1e9
    print(f"VRAM after encode        : {vram_after:.2f} GB")

    if args.output_path:
        torch.save(latent.cpu(), args.output_path)
        print(f"Saved latent to          : {args.output_path}")


if __name__ == "__main__":
    main()
