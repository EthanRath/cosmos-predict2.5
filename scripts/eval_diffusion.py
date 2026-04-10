"""
Run a pre-processed adversarial video tensor through the full Video2World
diffusion model and save the generated output.

Loads x_adv.pt (float32, [-1,1], shape (1,C,T,H,W)) directly, bypassing the
normal uint8 video-loading and normalisation steps. All other infrastructure
(model loading, T5 embedding, diffusion sampler, VAE decoder) is reused from
the existing Video2WorldInference class.

Supports multi-GPU context parallelism via torchrun:

    torchrun --nproc_per_node=2 scripts/eval_diffusion.py \
        --adv_path attack/outputs/<timestamp>/x_adv.pt \
        --experiment_name <experiment_name> \
        --ckpt_path <path_or_uuid> \
        --prompt "a robot arm picks up a red block" \
        --context_parallel_size 2 \
        --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
        --offload_diffusion_model --offload_tokenizer --offload_text_encoder \
        [--resolution "432,432"] \
        [--num_latent_conditional_frames 2] \
        [--guidance 7] \
        [--num_steps 35] \
        [--seed 1]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torchvision

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent      # cosmos-predict2.5/
WM_ROOT    = REPO_ROOT.parent       # WM_Poison/

sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Imports (deferred so sys.path is set first)
# ---------------------------------------------------------------------------
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference  # noqa: E402
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (    # noqa: E402
    IS_PREPROCESSED_KEY,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adv_path",       required=True,
                        help="Path to x_adv.pt (float32, [-1,1], shape (1,C,T,H,W))")
    parser.add_argument("--experiment_name", required=True,
                        help="Experiment name passed to Video2WorldInference")
    parser.add_argument("--ckpt_path",      required=True,
                        help="Checkpoint path or UUID passed to Video2WorldInference")
    parser.add_argument("--prompt",         required=True,
                        help="Text prompt for conditioning")
    parser.add_argument("--resolution",     default="432,432",
                        help="'H,W' string passed to the model (default: 432,432)")
    parser.add_argument("--num_latent_conditional_frames", type=int, default=2)
    parser.add_argument("--guidance",       type=float, default=7.0)
    parser.add_argument("--num_steps",      type=int,   default=35)
    parser.add_argument("--seed",           type=int,   default=1)
    parser.add_argument("--negative_prompt", default=None,
                        help="Override the default negative prompt")
    parser.add_argument("--offload_diffusion_model", action="store_true",
                        help="Offload DiT to CPU between steps to save VRAM")
    parser.add_argument("--offload_text_encoder",    action="store_true")
    parser.add_argument("--offload_tokenizer",       action="store_true")
    parser.add_argument("--context_parallel_size",   type=int, default=1,
                        help="Number of GPUs for context parallelism. Must match "
                             "--nproc_per_node when using torchrun (default: 1)")
    parser.add_argument("--config_file",
                        default="cosmos_predict2/_src/predict2/configs/video2world/config.py",
                        help="Path to Hydra config file (default: video2world config)")
    args = parser.parse_args()

    # torchrun sets LOCAL_RANK; fall back to 0 for single-process runs
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_rank0   = local_rank == 0

    # ------------------------------------------------------------------
    # 1. Load adversarial tensor
    # ------------------------------------------------------------------
    if is_rank0:
        print(f"Loading adversarial tensor from: {args.adv_path}")
    x_adv = torch.load(args.adv_path, map_location="cpu")
    assert x_adv.ndim == 5 and x_adv.shape[1] == 3, (
        f"Expected shape (1,3,T,H,W), got {x_adv.shape}"
    )
    assert x_adv.dtype == torch.float32, (
        f"Expected float32 tensor, got {x_adv.dtype}"
    )
    print(f"x_adv shape : {x_adv.shape}, range [{x_adv.min():.3f}, {x_adv.max():.3f}]")

    # ------------------------------------------------------------------
    # 2. Load the full Video2World model
    # ------------------------------------------------------------------
    if is_rank0:
        print("Loading Video2World model...")
        print(args.offload_diffusion_model)
        print(args.offload_text_encoder)
        print(args.offload_tokenizer)
    inference = Video2WorldInference(
        experiment_name=args.experiment_name,
        ckpt_path=args.ckpt_path,
        s3_credential_path="",
        context_parallel_size=args.context_parallel_size,
        config_file=args.config_file,
        offload_diffusion_model=args.offload_diffusion_model,
        offload_text_encoder=args.offload_text_encoder,
        offload_tokenizer=args.offload_tokenizer,
    )

    # ------------------------------------------------------------------
    # 3. Reformat x_adv to match read_and_process_video's layout:
    #    - extract the last `frames_to_extract` pixel frames as conditioning context
    #    - place them at positions 0..frames_to_extract-1
    #    - fill the remaining positions with the last context frame repeated
    #    This mirrors the normal inference path so the diffusion model conditions
    #    on the final frames of x_adv (mid-motion) rather than the first frames.
    # ------------------------------------------------------------------
    state_t = inference.model.config.state_t
    required_pixel_frames = (state_t - 1) * 4 + 1
    current_pixel_frames = x_adv.shape[2]
    frames_to_extract = 4 * (args.num_latent_conditional_frames - 1) + 1

    context_frames = x_adv[:, :, -frames_to_extract:, :, :]       # last N frames of x_adv
    last_frame     = context_frames[:, :, -1:, :, :]
    padding        = required_pixel_frames - frames_to_extract
    x_adv_padded   = torch.cat(
        [context_frames, last_frame.repeat(1, 1, padding, 1, 1)], dim=2
    )
    if is_rank0:
        print(f"Reformatted x_adv: context {frames_to_extract} frames + {padding} repeated "
              f"-> {required_pixel_frames} total")

    # ------------------------------------------------------------------
    # 4. Build data batch via the normal path (handles T5, fps, padding_mask)
    #    We pass x_adv as the video so shape-dependent fields (H, W) are correct.
    #    The video value will be replaced below before normalization runs.
    # ------------------------------------------------------------------
    # Move to this rank's GPU (torchrun sets CUDA device via distributed.init())
    x_adv_bf16 = x_adv_padded.to(device=torch.cuda.current_device(), dtype=torch.bfloat16)

    from cosmos_predict2._src.predict2.inference.video2world import _DEFAULT_NEGATIVE_PROMPT
    negative_prompt = args.negative_prompt or _DEFAULT_NEGATIVE_PROMPT

    data_batch = inference._get_data_batch_input(
        video=x_adv_bf16,
        prompt=args.prompt,
        num_conditional_frames=args.num_latent_conditional_frames,
        negative_prompt=negative_prompt,
        use_neg_prompt=True,
    )

    # ------------------------------------------------------------------
    # 5. Overwrite the video entry with x_adv_padded and mark as pre-processed
    #    so _normalize_video_databatch_inplace skips the uint8 conversion.
    # ------------------------------------------------------------------
    data_batch["video"] = x_adv_bf16
    data_batch[IS_PREPROCESSED_KEY] = True

    # ------------------------------------------------------------------
    # 6. Run diffusion sampler + decode  (mirrors generate_vid2world internals)
    # ------------------------------------------------------------------
    if is_rank0:
        print("Running diffusion model...")
    if getattr(inference.model.config, "use_lora", False):
        generate_fn = inference.model.generate_samples_from_batch_lora
    else:
        generate_fn = inference.model.generate_samples_from_batch

    with torch.no_grad():
        sample = generate_fn(
            data_batch,
            n_sample=1,
            guidance=args.guidance,
            seed=args.seed,
            is_negative_prompt=True,
            num_steps=args.num_steps,
        )

    if is_rank0:
        print("Decoding latent sample...")
    with torch.no_grad():
        if isinstance(sample, list):
            chunks = [inference.model.decode(s) for s in sample]
            video_out = torch.cat(chunks, dim=3)
        else:
            video_out = inference.model.decode(sample)

    # ------------------------------------------------------------------
    # 7. Save on rank 0 only
    # ------------------------------------------------------------------
    if is_rank0:
        print(f"Output shape : {video_out.shape}, range [{video_out.min():.3f}, {video_out.max():.3f}]")

        def to_uint8_frames(t):
            """(1, C, T, H, W) float [-1,1] -> (T, H, W, C) uint8"""
            t = t[0].cpu().float()                                 # (C, T, H, W)
            t = ((t.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
            return t.permute(1, 2, 3, 0)                           # (T, H, W, C)

        ts = int(time.time())
        out_dir = Path(args.adv_path).parent

        # generated-only video
        frames_gen = to_uint8_frames(video_out)
        out_gen = out_dir / f"generated_{ts}.mp4"
        torchvision.io.write_video(str(out_gen), frames_gen, fps=24)
        print(f"Saved generated video to     : {out_gen}")

        # x_adv (original frames only, no padding) + generated video (concatenated along time)
        frames_adv = to_uint8_frames(x_adv[:, :, :current_pixel_frames-4, :, :].float())
        frames_full = torch.cat([frames_adv, frames_gen], dim=0)   # (T_adv+T_gen, H, W, C)
        out_full = out_dir / f"full_{ts}.mp4"
        torchvision.io.write_video(str(out_full), frames_full, fps=24)
        print(f"Saved full video to          : {out_full}")


if __name__ == "__main__":
    main()


"""
torchrun --nproc_per_node=2 cosmos-predict2.5/scripts/eval_diffusion.py \
    --adv_path attack/outputs/1775846358/x_adv.pt \
    --experiment_name predict2_video2world_training_2b_libero_480 \
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/47d14a41779c654c213600ec1c35c9ebd89dd992/model.pt \
    --prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate" \
    --resolution 432,432 \
    --num_latent_conditional_frames 2 \
    --context_parallel_size 2 \
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --offload_diffusion_model \
    --offload_tokenizer \
    --offload_text_encoder
"""
