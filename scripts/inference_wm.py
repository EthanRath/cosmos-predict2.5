"""
Run Video2World inference on one or many (video, prompt) pairs without any
adversarial attack.  Mirrors the loading / eval interface of
attack_crossattn.py so that the same --video_path / --batch_path / --extend
arguments produce comparable outputs.

Input modes
-----------
Single video  : --video_path <file> --prompt "<text>"
Batch         : --batch_path <dir>  (contains videos/ and metas/ subfolders)
                --num <N>           sample N videos from the batch (-1 = all)

Saved artefacts (attack/outputs/inference_<timestamp>/):
    <name>_clean.pt           : preprocessed conditioning tensor
    <name>_clean_gen.mp4      : generated video

Usage (single video):
    python cosmos-predict2.5/scripts/inference_wm.py \
        --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
        --prompt "Use the franka robot arm to pick up the black bowl" \
        --resolution 432,432 \
        --num_latent_conditional_frames 2

Usage (extension):
    python cosmos-predict2.5/scripts/inference_wm.py \
        --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
        --prompt "..." --extend

Usage (batch):
    python cosmos-predict2.5/scripts/inference_wm.py \
        --batch_path /mnt/ssd2/libero/eval_set/ \
        --num 10
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
WM_ROOT    = REPO_ROOT.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(WM_ROOT))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from cosmos_predict2._src.predict2.inference.video2world import (   # noqa: E402
    Video2WorldInference,
)
from probing.test_vae_encoder import load_and_preprocess_video, normalize_video, resize_input  # noqa: E402
from attack.eval_wm import eval
from attack.shared_config import (
    ckpt_path, experiment_name, config_file
)


# ---------------------------------------------------------------------------
# Video padding (same helpers as attack_crossattn.py)
# ---------------------------------------------------------------------------

def pad_video(video, frames_to_extract, required_pixel_frames):
    """
    Take the last `frames_to_extract` pixel frames from `video` and repeat
    the last of those to fill `required_pixel_frames`.
    """
    context = video[:, :, -frames_to_extract:, :, :]
    padding = required_pixel_frames - frames_to_extract
    if padding > 0:
        context = torch.cat(
            [context, context[:, :, -1:, :, :].repeat(1, 1, padding, 1, 1)], dim=2
        )
    return context


def load_full_video(video_path, resolution):
    """Load all frames of a video and resize to resolution.
    Returns (1, C, T, H, W) uint8 tensor.
    """
    from cosmos_predict2._src.imaginaire.utils.easy_io import easy_io

    video_frames, video_metadata = easy_io.load(str(video_path))
    print(f"Loaded full video: shape={video_frames.shape}, metadata={video_metadata}")

    video_tensor = torch.from_numpy(video_frames).float() / 255.0  # (T, H, W, C)
    video_tensor = video_tensor.permute(3, 0, 1, 2)                # (C, T, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)                # (T, C, H, W)
    video_tensor = (video_tensor * 255.0).to(torch.uint8)
    video_tensor = resize_input(video_tensor, resolution)           # (T, C, H, W) uint8
    video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)
    return video_tensor


# ---------------------------------------------------------------------------
# Per-video inference
# ---------------------------------------------------------------------------

def infer_single_video(
    inference, video_path, prompt,
    H, W, frames_to_extract, required_pixel_frames,
    args, out_dir, device="cuda",
):
    """
    Load + pad a single video, then run the diffusion eval pipeline.
    Returns the preprocessed conditioning tensor on CPU.
    """
    print(f"\nLoading video : {video_path}")
    if args.extend:
        full_video_uint8 = load_full_video(str(video_path), [H, W])
        raw_full = normalize_video(full_video_uint8, device=device)
        args._extend_full_video = raw_full.cpu()
        print(f"Full video shape         : {raw_full.shape}")
        raw_padded = pad_video(raw_full, frames_to_extract, required_pixel_frames)
    else:
        video_uint8 = load_and_preprocess_video(
            str(video_path), [H, W], frames_to_extract
        )
        raw_cond = normalize_video(video_uint8, device=device)
        args._extend_full_video = None
        print(f"Conditioning frames shape: {raw_cond.shape}")
        raw_padded = pad_video(raw_cond, frames_to_extract, required_pixel_frames)
    print(f"Padded video shape       : {raw_padded.shape}")

    base_name = Path(video_path).stem + "_clean"
    cond_path = out_dir / f"{base_name}.pt"
    torch.save(raw_padded.cpu(), cond_path)

    print("\nRunning diffusion inference...")
    args.adv_path = out_dir
    args.prompt   = prompt
    eval(inference, raw_padded, args, cond_path)

    return raw_padded.cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path", default=None,
                             help="Path to a single input video (.mp4)")
    input_group.add_argument("--batch_path", default=None,
                             help="Path to a batch directory containing videos/ and "
                                  "metas/ subfolders.  Each <name>.mp4 in videos/ must "
                                  "have a matching <name>.txt prompt in metas/.")
    parser.add_argument("--prompt", default=None,
                        help="Text prompt (required with --video_path)")
    parser.add_argument("--num", type=int, default=-1,
                        help="Number of videos to sample from --batch_path "
                             "(-1 = use all, default: -1)")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--ckpt_path",       type=str, default=None)
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--resolution",      default="432,432",
                        help="'H,W' (default: 432,432)")
    parser.add_argument("--num_latent_conditional_frames", type=int, default=2,
                        help="Number of conditioning latent frames. Default: 2.")
    parser.add_argument("--load_diffusion_model", action="store_true",
                        help="Keep diffusion model on GPU (default: offload to CPU)")
    parser.add_argument("--load_text_encoder",    action="store_true",
                        help="Keep text encoder on GPU (default: offload to CPU)")
    parser.add_argument("--load_tokenizer",       action="store_true",
                        help="Keep tokenizer on GPU (default: offload to CPU)")
    parser.add_argument("--context_parallel_size",   type=int, default=1)
    parser.add_argument("--config_file",
                        default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    parser.add_argument("--extend", action="store_true",
                        help="Extension mode: load all frames of the input video and use the "
                             "final frames as conditioning context.")
    args = parser.parse_args()

    if args.video_path is not None and args.prompt is None:
        parser.error("--prompt is required when using --video_path")
    if args.ckpt_path is None:
        args.ckpt_path = ckpt_path
    if args.experiment_name is None:
        args.experiment_name = experiment_name

    device = "cuda"
    H, W   = [int(x) for x in args.resolution.split(",")]

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("Loading Video2World model...")
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
    model   = inference.model
    state_t = model.config.state_t

    required_pixel_frames = (state_t - 1) * 4 + 1
    frames_to_extract     = (args.num_latent_conditional_frames - 1) * 4 + 1

    print(f"Model         : state_t={state_t}")
    print(f"Pixel frames  : required={required_pixel_frames}  to_extract={frames_to_extract}")

    # ------------------------------------------------------------------
    # Collect (video_path, prompt) pairs
    # ------------------------------------------------------------------
    if args.batch_path is not None:
        batch_path  = Path(args.batch_path)
        video_files = sorted((batch_path / "videos").glob("*.mp4"))
        if args.num > 0:
            video_files = random.sample(video_files, min(args.num, len(video_files)))
        pairs = []
        for vf in video_files:
            meta_path = batch_path / "metas" / (vf.stem + ".txt")
            pairs.append((vf, meta_path.read_text().strip()))
        print(f"\nBatch mode: {len(pairs)} video(s) from {batch_path}")
    else:
        pairs = [(Path(args.video_path), args.prompt)]

    # ------------------------------------------------------------------
    # Run inference on each video
    # ------------------------------------------------------------------
    save_time = int(time.time())
    out_dir = WM_ROOT / "attack" / "outputs" / f"inference_{save_time}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        **{k: v for k, v in vars(args).items() if not k.startswith("_")},
        "H": H,
        "W": W,
        "required_pixel_frames": required_pixel_frames,
        "frames_to_extract": frames_to_extract,
        "save_time": save_time,
        "out_dir": str(out_dir),
        "pairs": [(str(vp), pr) for vp, pr in pairs],
    }
    with open(out_dir / "inference_config.json", "w") as _f:
        json.dump(run_config, _f, indent=2)
    print(f"Inference config saved to: {out_dir / 'inference_config.json'}")

    for i, (video_path, prompt) in enumerate(pairs):
        print(f"\n{'='*60}")
        print(f"Video {i+1}/{len(pairs)}: {video_path.name}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        infer_single_video(
            inference, video_path, prompt,
            H, W, frames_to_extract, required_pixel_frames,
            args, out_dir, device=device,
        )


if __name__ == "__main__":
    main()
