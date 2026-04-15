"""
Compute and save cross-attention activations from the Video2World DiT for a
given input video and text prompt.

For each transformer block the script registers a forward hook on the
block.cross_attn module.  The hook captures the output tensor of the cross-
attention operation (shape: (B, T*H*W, D)) which encodes how much each video
latent spatial-temporal token attended to each text token under the learned
Q-K/V projection matrices.

A single call to model.denoise() is made at a fixed noise level (--timestep,
default 0.5), so the full diffusion loop is never run.  The saved artefacts:
  - video_latents.pt    : VAE-encoded video, shape (1, 16, T_lat, H_lat, W_lat)
  - text_emb.pt         : T5 text embedding (crossattn_emb), shape (1, N, D)
  - crossattn_out.pt    : dict {block_idx -> (1, T_lat*H_lat*W_lat, D)} cpu tensors

Usage:
    python scripts/compute_cross_attention.py \
        --video_path /path/to/video.mp4 \
        --experiment_name predict2_video2world_training_2b_libero_480 \
        --ckpt_path /path/to/model.pt \
        --prompt "a robot arm picks up a block" \
        --resolution 432,432 \
        --num_latent_video_frames 6 \
        --num_latent_conditional_frames 2 \
        --timestep 0.5 \
        --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
        [--offload_diffusion_model] [--offload_tokenizer] [--offload_text_encoder]
"""

import argparse
import os
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
sys.path.insert(0, str(SCRIPT_DIR))   # so we can import test_vae_encoder

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from cosmos_predict2._src.predict2.inference.video2world import (   # noqa: E402
    Video2WorldInference,
    _DEFAULT_NEGATIVE_PROMPT,
)
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (  # noqa: E402
    IS_PREPROCESSED_KEY,
)
from test_vae_encoder import load_and_preprocess_video, normalize_video  # noqa: E402


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------

def register_crossattn_hooks(net):
    """
    Register forward hooks on every block.cross_attn inside the DiT.

    Returns:
        hooks        : list of hook handles (call h.remove() to clean up)
        captured     : dict {block_idx -> output_tensor} populated after each
                       forward pass through the net
    """
    hooks    = []
    captured = {}

    for idx, block in enumerate(net.blocks):
        if not hasattr(block, "cross_attn"):
            continue

        def _make_hook(i):
            def _hook(_module, _input, output):
                # output may be a tuple; the attention output is always first
                out = output[0] if isinstance(output, tuple) else output
                captured[i] = out.detach().cpu()
            return _hook

        h = block.cross_attn.register_forward_hook(_make_hook(idx))
        hooks.append(h)

    return hooks, captured


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path",       required=True,
                        help="Path to input .mp4 video")
    parser.add_argument("--experiment_name",  required=True)
    parser.add_argument("--ckpt_path",        required=True)
    parser.add_argument("--prompt",           required=True)
    parser.add_argument("--negative_prompt",  default=None)
    parser.add_argument("--resolution",       default="432,432",
                        help="'H,W' resolution string (default: 432,432)")
    parser.add_argument("--num_latent_video_frames",       type=int, default=6,
                        help="Latent temporal length to encode (default: 6). "
                             "Pixel frames = (n-1)*4+1.")
    parser.add_argument("--num_latent_conditional_frames", type=int, default=2,
                        help="Number of latent conditioning frames used by the model (default: 2)")
    parser.add_argument("--timestep",         type=float, default=0.5,
                        help="Noise level for the single DiT forward pass; "
                             "0 = clean latent, 1 = pure noise (default: 0.5)")
    parser.add_argument("--offload_diffusion_model", action="store_true")
    parser.add_argument("--offload_text_encoder",    action="store_true")
    parser.add_argument("--offload_tokenizer",       action="store_true")
    parser.add_argument("--context_parallel_size",   type=int, default=1)
    parser.add_argument("--config_file",
                        default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    H, W = [int(x) for x in args.resolution.split(",")]

    # ------------------------------------------------------------------
    # 1. Load and preprocess video
    # ------------------------------------------------------------------
    num_pixel_frames = (args.num_latent_video_frames - 1) * 4 + 1
    print(f"Loading video : {args.video_path}")
    video_uint8 = load_and_preprocess_video(
        video_path=args.video_path,
        resolution=[H, W],
        num_video_frames=num_pixel_frames,
    )
    print(f"Video shape   : {video_uint8.shape}")

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print("Loading Video2World model...")
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
    model = inference.model

    # ------------------------------------------------------------------
    # 3. Reformat video to match the diffusion model's expected layout:
    #    last `frames_to_extract` frames at positions 0..frames_to_extract-1,
    #    then repeat the last frame to fill the model's required pixel count.
    # ------------------------------------------------------------------
    state_t              = model.config.state_t
    required_pixel_frames = (state_t - 1) * 4 + 1
    frames_to_extract    = 4 * (args.num_latent_conditional_frames - 1) + 1

    video_f32 = normalize_video(video_uint8)                         # float32 [-1,1] on CUDA
    context_frames = video_f32[:, :, -frames_to_extract:, :, :]
    last_frame     = context_frames[:, :, -1:, :, :]
    padding        = required_pixel_frames - frames_to_extract
    video_padded   = torch.cat(
        [context_frames, last_frame.repeat(1, 1, padding, 1, 1)], dim=2
    )
    print(f"Padded video  : {video_padded.shape}  "
          f"(context {frames_to_extract} frames + {padding} repeated)")

    video_bf16 = video_padded.to(device=torch.cuda.current_device(), dtype=torch.bfloat16)

    # ------------------------------------------------------------------
    # 4. Build data batch (computes T5 text embeddings from the prompt)
    # ------------------------------------------------------------------
    negative_prompt = args.negative_prompt or _DEFAULT_NEGATIVE_PROMPT
    data_batch = inference._get_data_batch_input(
        video=video_bf16,
        prompt=args.prompt,
        num_conditional_frames=args.num_latent_conditional_frames,
        negative_prompt=negative_prompt,
        use_neg_prompt=True,
    )
    data_batch["video"]          = video_bf16
    data_batch[IS_PREPROCESSED_KEY] = True

    # ------------------------------------------------------------------
    # 5. Replicate generate_vid2world offloading sequence so VRAM usage
    #    matches the normal inference path:
    #      Step 1 – offload text encoder now that T5 embeddings are computed
    #      Step 2 – bring tokenizer encoder to GPU for VAE encoding
    #      Step 3 – bring DiT (net) + conditioner to GPU for the forward pass
    # ------------------------------------------------------------------
    if inference.offload_text_encoder and model.text_encoder is not None:
        if hasattr(model.text_encoder, "model") and model.text_encoder.model is not None:
            model.text_encoder.model = model.text_encoder.model.to("cpu")
        torch.cuda.empty_cache()

    if inference.offload_tokenizer:
        if hasattr(model.tokenizer, "encoder") and model.tokenizer.encoder is not None:
            model.tokenizer.encoder = model.tokenizer.encoder.to("cuda")
        torch.cuda.empty_cache()

    if inference.offload_diffusion_model:
        model.net = model.net.to("cuda")
        if hasattr(model, "conditioner") and model.conditioner is not None:
            model.conditioner = model.conditioner.to("cuda")
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 6. Encode video with VAE and build condition (T5 + video mask)
    # ------------------------------------------------------------------
    print("Encoding video and building condition...")
    with torch.no_grad():
        raw_state, latent_state, condition = model.get_data_and_condition(data_batch)

    condition = condition.edit_for_inference(
        is_cfg_conditional=True,
        num_conditional_frames=args.num_latent_conditional_frames,
    )

    B, C, T_lat, H_lat, W_lat = latent_state.shape
    print(f"Latent shape  : {latent_state.shape}")
    print(f"text_emb shape: {condition.crossattn_emb.shape}")

    # ------------------------------------------------------------------
    # 7. Construct noisy latent at the requested noise level (rectified flow)
    #    xt = (1 - t) * x0  +  t * noise
    # ------------------------------------------------------------------
    t     = args.timestep
    noise = torch.randn_like(latent_state)
    xt    = ((1 - t) * latent_state + t * noise).to(**model.tensor_kwargs)

    # Replace the first conditioning frames with the clean latent (mirrors denoise())
    cond_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(xt)
    xt = latent_state.type_as(xt) * cond_mask + xt * (1 - cond_mask)

    # timesteps tensor expected by the net: shape (B, T_lat)
    timesteps = torch.full(
        (B, T_lat), t,
        device=latent_state.device, dtype=latent_state.dtype
    )

    # ------------------------------------------------------------------
    # 8. Register hooks and run ONE forward pass through the DiT
    # ------------------------------------------------------------------
    hooks, crossattn_out = register_crossattn_hooks(model.net)
    print(f"Registered hooks on {len(hooks)} cross-attention modules")

    print(f"Running single DiT forward pass at t={t}...")
    with torch.no_grad():
        _ = model.net(
            x_B_C_T_H_W=xt,
            timesteps_B_T=timesteps,
            **condition.to_dict(),
        )

    for h in hooks:
        h.remove()

    print(f"Captured cross-attn outputs from {len(crossattn_out)} blocks")
    for blk_idx, tensor in sorted(crossattn_out.items()):
        print(f"  block {blk_idx:2d}: shape={tensor.shape}  "
              f"range=[{tensor.min():.4f}, {tensor.max():.4f}]")

    # ------------------------------------------------------------------
    # 9. Save on rank 0 only
    # ------------------------------------------------------------------
    if local_rank == 0:
        out_dir = WM_ROOT / "attack" / "outputs" / f"crossattn_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(latent_state.cpu(),         out_dir / "video_latents.pt")
        torch.save(condition.crossattn_emb.cpu(), out_dir / "text_emb.pt")
        torch.save(crossattn_out,              out_dir / "crossattn_out.pt")

        print(f"\nSaved to: {out_dir}")
        print(f"  video_latents.pt : {latent_state.shape}")
        print(f"  text_emb.pt      : {condition.crossattn_emb.shape}")
        print(f"  crossattn_out.pt : {len(crossattn_out)} blocks")


if __name__ == "__main__":
    main()


"""
python cosmos-predict2.5/scripts/compute_cross_attention.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --experiment_name predict2_video2world_training_2b_libero_480 \
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/8fbc6188fa2f2e4ab585dc6aac3edd0e9d8a3670/model.pt \
    --prompt "Use the franka robot arm to turn on the stove and place the moka pot on it" \
    --resolution 432,432 \
    --num_latent_video_frames 6 \
    --num_latent_conditional_frames 2 \
    --timestep 0.5 \
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --offload_diffusion_model \
    --offload_tokenizer \
    --offload_text_encoder
"""
