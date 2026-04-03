"""
PGD adversarial attack against the Wan2.1 VAE encoder.

Loads a source video and a target video, encodes the target (no grad) to obtain
a latent target, then runs PGD on the source video so that its encoding matches
the target encoding in latent space.

The adversarial video is saved to attack/outputs/<timestamp>/.

Usage:
    python scripts/attack_vae_encoder.py \
        --video_path /path/to/source.mp4 \
        --attack_target /path/to/target.mp4 \
        --vae_pth /path/to/tokenizer.pth \
        [--resolution 432 432] \
        [--num_latent_conditional_frames 2] \
        [--num_latent_video_frames 31]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
REPO_ROOT    = SCRIPT_DIR.parent          # cosmos-predict2.5/
WM_ROOT      = REPO_ROOT.parent           # WM_Poison/

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(WM_ROOT))

# ---------------------------------------------------------------------------
# Shared helpers from test_vae_encoder
# ---------------------------------------------------------------------------
from test_vae_encoder import load_and_preprocess_video, normalize_video, load_vae  # noqa: E402

# ---------------------------------------------------------------------------
# Attack
# ---------------------------------------------------------------------------
from attack.white_box import pgd  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path",     required=True, help="Path to source input .mp4")
    parser.add_argument("--attack_target",  required=True, help="Path to target .mp4 whose encoding we optimise toward")
    parser.add_argument("--vae_pth",        required=True, help="Path to tokenizer.pth checkpoint")
    parser.add_argument(
        "--resolution", type=int, nargs=2, default=[720, 1280],
        metavar=("H", "W"),
        help="Target resolution [H W] (default: 720 1280)",
    )
    parser.add_argument(
        "--num_latent_video_frames", type=int, default=31,
        help="Number of latent frames to encode (default: 31). Pixel frames = (n-1)*4+1.",
    )
    parser.add_argument("--steps",  type=int,   default=20,      help="PGD steps (default: 20)")
    parser.add_argument("--alpha",  type=float, default=1/255,   help="PGD step size (default: 1/255)")
    parser.add_argument("--eps",    type=float, default=64/255,  help="PGD epsilon (default: 64/255)")
    args = parser.parse_args()
    args.vae_pth = "/home/ethan/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/6787e176dce74a101d922174a95dba29fa5f0c55/tokenizer.pth"

    device = "cuda"
    num_pixel_frames = (args.num_latent_video_frames - 1) * 4 + 1

    print(f"Resolution      : {args.resolution}")
    print(f"Latent T frames : {args.num_latent_video_frames}")
    print(f"Pixel T frames  : {num_pixel_frames}")
    print(f"Epsilon {args.eps}")
    inverse_attack =  args.video_path == args.attack_target
    print(f"Inverse Attack {inverse_attack}")

    # ------------------------------------------------------------------
    # 1. Load and preprocess both videos
    # ------------------------------------------------------------------
    source_uint8 = load_and_preprocess_video(
        args.video_path, args.resolution, num_pixel_frames,
    )
    target_uint8 = load_and_preprocess_video(
        args.attack_target, args.resolution, num_pixel_frames,
    )

    # ------------------------------------------------------------------
    # 2. Normalise to [-1, 1]
    # ------------------------------------------------------------------
    raw_state      = normalize_video(source_uint8, device=device)
    target_state   = normalize_video(target_uint8, device=device)

    print(f"Source shape    : {raw_state.shape}")
    print(f"Target shape    : {target_state.shape}")

    # ------------------------------------------------------------------
    # 3. Load VAE with gradients enabled (needed for PGD)
    # ------------------------------------------------------------------
    tokenizer = load_vae(args.vae_pth, device=device, enable_grad=True)

    # ------------------------------------------------------------------
    # 4. Encode target with no grad to get the latent we optimise toward
    # ------------------------------------------------------------------
    tokenizer.enable_grad = False
    with torch.no_grad():
        encoded_target = tokenizer.encode(target_state).contiguous().float()
    tokenizer.enable_grad = True

    print(f"Encoded target shape : {encoded_target.shape}")

    # ------------------------------------------------------------------
    # 5. Run PGD
    # ------------------------------------------------------------------
    cos_loss = torch.nn.CosineSimilarity(dim=1)
    if inverse_attack:
        loss_fn = lambda x, y: -cos_loss(x,y).mean()
    else:
        loss_fn = lambda x, y: cos_loss(x,y).mean()
    encode_fn = lambda x: tokenizer.encode(x).contiguous().float()

    print("Running PGD attack...")
    x_adv = pgd(
        raw_state,
        encoded_target,
        encode_fn,
        loss_fn,
        steps=args.steps,
        alpha=args.alpha,
        eps=args.eps,
        device=device,
    )

    # ------------------------------------------------------------------
    # 6. Save output
    # ------------------------------------------------------------------
    out_dir = WM_ROOT / "attack" / "outputs" / str(int(time.time()))
    out_dir.mkdir(parents=True, exist_ok=True)

    def to_uint8_frames(t):
        """(1, C, T, H, W) float [-1,1] -> (T, H, W, C) uint8"""
        t = t[0].cpu().float()                                     # (C, T, H, W)
        t = ((t.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
        return t.permute(1, 2, 3, 0)                               # (T, H, W, C)

    torch.save(x_adv.cpu(), out_dir / "x_adv.pt")
    torch.save(raw_state.cpu(), out_dir / "x_orig.pt")
    torch.save(encoded_target.cpu(), out_dir / "encoded_target.pt")

    torchvision.io.write_video(str(out_dir / "x_adv.mp4"),  to_uint8_frames(x_adv),       fps=16)
    torchvision.io.write_video(str(out_dir / "x_orig.mp4"), to_uint8_frames(raw_state),   fps=16)

    print(f"Saved adversarial video to : {out_dir / 'x_adv.pt'} / x_adv.mp4")
    print(f"Saved original video to    : {out_dir / 'x_orig.pt'} / x_orig.mp4")
    print(f"Saved encoded target to    : {out_dir / 'encoded_target.pt'}")


if __name__ == "__main__":
    main()


"""
python cosmos-predict2.5/scripts/attack_vae_encoder.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --attack_target cosmos-predict2.5/assets/attack/k_2.mp4 \
    --vae_pth /home/ethan/.cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/6787e176dce74a101d922174a95dba29fa5f0c55/tokenizer.pth \
    --resolution 432 432 \
    --num_latent_video_frames 6 \
    --eps 0.0628
"""
