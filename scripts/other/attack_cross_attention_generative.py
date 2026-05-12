"""
PGD attack on cross-attention in the Video2World DiT, with denoising steps.

Instead of a single forward pass at a fixed noise level, this script runs
`num_denoise_steps` Euler steps of rectified-flow denoising before capturing
cross-attention.  The first k-1 steps run without gradients (VRAM efficient);
the final step runs with gradients so PGD can backprop through cross-attention
→ VAE encoder → input video pixels.

Attack objective:
    minimise cosine_distance(
        crossattn_after_k_steps(VAE(x_adv),  true_prompt),   [changes per PGD step]
        crossattn_after_k_steps(VAE(x_orig), target_prompt)  [fixed target]
    )

Denoising schedule:
    t = linspace(1.0, 0.0, num_denoise_steps+1)
    Steps 0..k-2 (no_grad): Euler update from t[i] → t[i+1]
    Step  k-1   (with grad): forward at t[k-1], cross-attn captured here.
    Effective capture timestep = 1 / num_denoise_steps.

    Examples:
      --num_denoise_steps 1  →  capture at t=1.0  (full noise, no prior steps)
      --num_denoise_steps 2  →  capture at t=0.5  (≈ original fixed-t attack)
      --num_denoise_steps 5  →  capture at t=0.2  (model more committed)
      --num_denoise_steps 10 →  capture at t=0.1  (near clean)

Gradient flow:
    cross-attn output → last DiT blocks (up to max(attack_layers)) → x_in_final
    → latent (cond frames only; gen-frame portion is detached) → VAE encoder → x_adv.
    Parameters of DiT and tokenizer are frozen (no parameter gradient buffers).

Usage:
    python cosmos-predict2.5/scripts/attack_cross_attention_generative.py \
        --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
        --true_prompt "Use the franka robot arm to pick up the black bowl" \
        --target_prompt "Use the franka robot arm to open the drawer" \
        --experiment_name predict2_video2world_training_2b_libero_480 \
        --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/47d14a41779c654c213600ec1c35c9ebd89dd992/model.pt \
        --resolution 432,432 \
        --num_latent_video_frames 6 \
        --num_latent_conditional_frames 2 \
        --num_denoise_steps 5 \
        --attack_layers 14 15 16 17 18 \
        --steps 20 --alpha 0.00392 --eps 0.0628 \
        --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
        --offload_diffusion_model --offload_tokenizer --offload_text_encoder
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torchvision

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
    _DEFAULT_NEGATIVE_PROMPT,
)
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (  # noqa: E402
    IS_PREPROCESSED_KEY,
)
import cosmos_predict2._src.predict2.inference.get_t5_emb as _t5_mod  # noqa: E402
from test_vae_encoder import load_and_preprocess_video, normalize_video  # noqa: E402
from attack.white_box import pgd                                          # noqa: E402


# ---------------------------------------------------------------------------
# Video padding helper (same as attack_cross_attention.py)
# ---------------------------------------------------------------------------

def pad_video(video, frames_to_extract, required_pixel_frames):
    """
    Place the last `frames_to_extract` pixel frames at positions 0..frames_to_extract-1,
    then repeat the last frame to fill required_pixel_frames.
    """
    context = video[:, :, -frames_to_extract:, :, :]
    padding = required_pixel_frames - frames_to_extract
    return torch.cat([context, context[:, :, -1:, :, :].repeat(1, 1, padding, 1, 1)], dim=2)


def move_condition_to_device(condition, device):
    """Move all tensor attributes of a condition dataclass to `device`.
    Uses object.__setattr__ to bypass frozen-dataclass restrictions."""
    for attr in vars(condition):
        val = getattr(condition, attr)
        if isinstance(val, torch.Tensor):
            object.__setattr__(condition, attr, val.to(device))
    return condition


def cond_dict_to_device(condition, device):
    """Return condition.to_dict() with every tensor moved to `device`.
    Called at each model.net() invocation to guarantee all DiT inputs are
    on the correct device, regardless of how the condition was built."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in condition.to_dict().items()
    }


# ---------------------------------------------------------------------------
# Core: denoising steps + cross-attention capture
# ---------------------------------------------------------------------------

def get_crossattn_after_denoise(
    model,
    video,
    condition,
    layer_indices,
    vae_device="cuda",
    dit_device="cuda",
    entropy_mode=False,
):
    """
    Encode `video` through the VAE, add rectified-flow noise at a randomly
    sampled timestep t ~ Uniform(0, 1), run a single DiT forward pass, and
    return the cross-attention outputs.

    t is sampled fresh on every call, so each PGD step sees a different noise
    level (randomised-timestep training objective).

    Gradient path: cross-attn → DiT → x_in → latent → VAE encoder → video.

    Returns
    -------
    entropy_mode=False : flat (N,) float32 tensor with grad attached.
    entropy_mode=True  : list of (B, S, D) float32 tensors, one per captured layer.
    """
    kw = {"device": dit_device, "dtype": torch.bfloat16}

    # 1. VAE encode (gradient-enabled for backward pass)
    latent = model.tokenizer.encode(video).contiguous().float()
    latent = latent.to(dit_device)  # differentiable bridge to dit_device
    B, C, T_lat, H_lat, W_lat = latent.shape

    cond_mask = condition.condition_video_input_mask_B_C_T_H_W \
                    .repeat(1, C, 1, 1, 1).type_as(latent)

    # 2. Sample random timestep and noise fresh each call
    t = torch.rand(1).item()
    noise = torch.randn(latent.shape, device=dit_device, dtype=latent.dtype)

    # 3. Noisy input via rectified-flow interpolation: x_t = (1-t)*x0 + t*eps
    #    Conditioning frames stay at t=0 (clean); generated frames noised at t.
    latent_det = latent.detach()
    x_noisy = (1.0 - t) * latent_det + t * noise
    x_in = latent * cond_mask + x_noisy * (1.0 - cond_mask)

    # 4. Register cross-attn hooks
    captured  = {}
    hooks     = []
    n_blocks  = len(model.net.blocks)
    idxs      = layer_indices if layer_indices is not None else list(range(n_blocks))
    max_block = max(idxs) if idxs else n_blocks - 1

    for idx in idxs:
        if idx >= n_blocks or not hasattr(model.net.blocks[idx], "cross_attn"):
            continue
        def _make_hook(i):
            def _hook(_m, _inp, output):
                captured[i] = output[0] if isinstance(output, tuple) else output
            return _hook
        hooks.append(model.net.blocks[idx].cross_attn.register_forward_hook(_make_hook(idx)))

    # Cut gradient flow after the last captured block to save VRAM.
    if max_block < n_blocks - 1:
        def _detach_hook(_m, _inp, output):
            if isinstance(output, tuple):
                return (output[0].detach(),) + output[1:]
            return output.detach()
        hooks.append(model.net.blocks[max_block].register_forward_hook(_detach_hook))

    # 5. Single forward pass at timestep t
    t_tensor = torch.full((B, T_lat), t, device=dit_device, dtype=torch.bfloat16)
    with torch.cuda.device(dit_device):
        model.net(
            x_B_C_T_H_W  = x_in.to(**kw),
            timesteps_B_T = t_tensor,
            **cond_dict_to_device(condition, dit_device),
        )

    for h in hooks:
        h.remove()

    if not captured:
        raise RuntimeError(
            f"No cross-attn outputs captured from blocks {idxs}. "
            "Check --attack_layers and that those blocks have a cross_attn attribute."
        )

    if entropy_mode:
        return [captured[k].float() for k in sorted(captured.keys())]
    return torch.cat([captured[k].flatten() for k in sorted(captured.keys())])



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path",       required=True,
                        help="Path to source input .mp4")
    parser.add_argument("--true_prompt",      required=True,
                        help="Prompt used at inference time (what the model receives)")
    parser.add_argument("--target_prompt",    default=None,
                        help="Prompt whose cross-attention activations we optimise toward. "
                             "If omitted, switches to entropy mode: minimises cross-attention "
                             "magnitude (sum of per-layer MSE to zero) with no target prompt.")
    parser.add_argument("--experiment_name",  required=True)
    parser.add_argument("--ckpt_path",        required=True)
    parser.add_argument("--resolution",       default="432,432")
    parser.add_argument("--num_latent_video_frames",       type=int, default=6)
    parser.add_argument("--num_latent_conditional_frames", type=int, default=2)
    parser.add_argument(
        "--attack_layers", type=int, nargs="+", default=None,
        help="DiT block indices whose cross-attn to target. "
             "Gradient is also cut after the highest index to save VRAM. "
             "Default: all blocks.",
    )
    parser.add_argument("--steps",  type=int,   default=20,      help="PGD steps (default: 20)")
    parser.add_argument("--alpha",  type=float, default=1/255,   help="Pixel-space PGD step size (default: 1/255)")
    parser.add_argument("--eps",    type=float, default=16/255,  help="Pixel-space PGD epsilon (default: 16/255)")
    parser.add_argument("--offload_diffusion_model", action="store_true")
    parser.add_argument("--offload_text_encoder",    action="store_true")
    parser.add_argument("--offload_tokenizer",       action="store_true")
    parser.add_argument("--context_parallel_size",   type=int, default=1)
    parser.add_argument(
        "--config_file",
        default="cosmos_predict2/_src/predict2/configs/video2world/config.py",
    )
    parser.add_argument(
        "--split_gpus", action="store_true",
        help="Place the VAE on cuda:0 and the DiT on cuda:1.  Gradient flows across "
             "devices via differentiable .to() operations, so a single backward pass "
             "spans both GPUs.  Requires two CUDA devices.  Halves per-device peak VRAM.",
    )
    args = parser.parse_args()

    entropy_mode = args.target_prompt is None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.split_gpus:
        assert torch.cuda.device_count() >= 2, "split_gpus requires at least 2 CUDA devices"
        vae_device = "cuda:0"
        dit_device = "cuda:1"
    else:
        vae_device = device
        dit_device = device
    H, W = [int(x) for x in args.resolution.split(",")]
    num_pixel_frames  = (args.num_latent_video_frames - 1) * 4 + 1
    frames_to_extract = 4 * (args.num_latent_conditional_frames - 1) + 1

    print(f"Resolution          : {args.resolution}")
    print(f"Latent T frames     : {args.num_latent_video_frames}")
    print(f"Pixel T frames      : {num_pixel_frames}")
    print(f"Timestep            : random t ~ Uniform(0,1) per PGD step")
    print(f"Attack layers       : {args.attack_layers or 'all'}")
    print(f"PGD                 : steps={args.steps}, alpha={args.alpha:.5f}, eps={args.eps:.4f}")
    print(f"Loss mode           : {'ENTROPY (minimise cross-attn magnitude)' if entropy_mode else f'COSINE  (target: {args.target_prompt})'}")

    # ------------------------------------------------------------------
    # 1. Load and preprocess video
    # ------------------------------------------------------------------
    print(f"\nLoading video: {args.video_path}")
    video_uint8 = load_and_preprocess_video(args.video_path, [H, W], num_pixel_frames)
    raw_state   = normalize_video(video_uint8, device=vae_device)  # (1, C, T, H, W) float32 [-1,1]
    print(f"Video shape         : {raw_state.shape}")

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print("\nLoading Video2World model...")
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
    model      = inference.model
    state_t    = model.config.state_t
    required_pixel_frames = (state_t - 1) * 4 + 1

    # ------------------------------------------------------------------
    # 3. Compute T5 embeddings for both prompts
    # ------------------------------------------------------------------
    compute_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    video_bf16 = pad_video(raw_state, frames_to_extract, required_pixel_frames).to(dtype=compute_dtype)

    print("\nComputing T5 embeddings for true prompt...")
    data_batch_true = inference._get_data_batch_input(
        video=video_bf16, prompt=args.true_prompt,
        num_conditional_frames=args.num_latent_conditional_frames,
        negative_prompt=_DEFAULT_NEGATIVE_PROMPT, use_neg_prompt=True,
    )
    data_batch_true["video"]             = video_bf16
    data_batch_true[IS_PREPROCESSED_KEY] = True

    if not entropy_mode:
        print("Computing T5 embeddings for target prompt...")
        data_batch_target = inference._get_data_batch_input(
            video=video_bf16, prompt=args.target_prompt,
            num_conditional_frames=args.num_latent_conditional_frames,
            negative_prompt=_DEFAULT_NEGATIVE_PROMPT, use_neg_prompt=True,
        )
        data_batch_target["video"]             = video_bf16
        data_batch_target[IS_PREPROCESSED_KEY] = True

    # ------------------------------------------------------------------
    # 4. Offloading (mirrors generate_vid2world)
    # ------------------------------------------------------------------
    if inference.offload_text_encoder:
        if model.text_encoder is not None:
            if hasattr(model.text_encoder, "model") and model.text_encoder.model is not None:
                model.text_encoder.model = model.text_encoder.model.to("cpu")
        if _t5_mod.cosmos_encoder is not None:
            print("Offloading global T5 singleton to CPU...")
            _t5_mod.cosmos_encoder.text_encoder = _t5_mod.cosmos_encoder.text_encoder.to("cpu")
        torch.cuda.empty_cache()

    if inference.offload_tokenizer:
        if hasattr(model.tokenizer, "encoder") and model.tokenizer.encoder is not None:
            model.tokenizer.encoder = model.tokenizer.encoder.to(vae_device)
        torch.cuda.empty_cache()

    if inference.offload_diffusion_model:
        model.net = model.net.to(dit_device)
        if hasattr(model, "conditioner") and model.conditioner is not None:
            model.conditioner = model.conditioner.to(dit_device)
        torch.cuda.empty_cache()
    elif args.split_gpus:
        # No offloading requested but split_gpus: move DiT to dit_device explicitly
        print(f"split_gpus: moving DiT → {dit_device}, VAE stays on {vae_device}")
        model.net = model.net.to(dit_device)
        if hasattr(model, "conditioner") and model.conditioner is not None:
            model.conditioner = model.conditioner.to(dit_device)
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Freeze DiT and tokenizer parameters
    #    (no param gradient buffers; gradients still flow through frozen params)
    # ------------------------------------------------------------------
    for p in model.net.parameters():
        p.requires_grad_(False)
    for p in model.tokenizer.model.model.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # 6. Build condition objects
    # ------------------------------------------------------------------
    print("\nBuilding condition objects...")
    with torch.no_grad():
        _, _, condition_true = model.get_data_and_condition(data_batch_true)
        if not entropy_mode:
            _, _, condition_target = model.get_data_and_condition(data_batch_target)

    condition_true = condition_true.edit_for_inference(
        is_cfg_conditional=True,
        num_conditional_frames=args.num_latent_conditional_frames,
    )
    if not entropy_mode:
        condition_target = condition_target.edit_for_inference(
            is_cfg_conditional=True,
            num_conditional_frames=args.num_latent_conditional_frames,
        )

    # Move condition tensors to the DiT device (T5 embeddings, masks, etc.)
    move_condition_to_device(condition_true, dit_device)
    if not entropy_mode:
        move_condition_to_device(condition_target, dit_device)

    model.tokenizer.enable_grad = True   # allow gradient through VAE encoder

    n_blocks = len(model.net.blocks)
    layer_indices = args.attack_layers
    if layer_indices is not None:
        print(f"Targeting cross-attn blocks {layer_indices} of {n_blocks} total")
        print(f"Gradient cutoff after block {max(layer_indices)}")
    else:
        print(f"Targeting all {n_blocks} cross-attn blocks")

    # ------------------------------------------------------------------
    # 7. Compute fixed target activations: crossattn(orig_video, target_prompt)
    #    Skipped in entropy mode — no target prompt exists.
    # ------------------------------------------------------------------
    video_padded = pad_video(raw_state, frames_to_extract, required_pixel_frames)
    if entropy_mode:
        target_acts = None
        print("\nEntropy mode: skipping target activation computation.")
    else:
        print("\nComputing fixed target cross-attention activations...")
        with torch.no_grad():
            target_acts = get_crossattn_after_denoise(
                model, video_padded,
                condition=condition_target,
                layer_indices=layer_indices,
                vae_device=vae_device,
                dit_device=dit_device,
            ).detach().float()
        print(f"Target activations shape : {target_acts.shape}")

    # ------------------------------------------------------------------
    # 8. Define encode_fn and loss for PGD.
    #
    #    Cosine mode  (target_prompt given):
    #      encode_fn -> flat (N,) vector; loss = 1 - cosine_sim(pred, target)
    #
    #    Entropy mode (no target_prompt):
    #      encode_fn -> list of per-layer (B, S, D) tensors
    #      loss      = sum_layers( MSE(layer_output, 0) )
    #                = sum_layers( layer_output.pow(2).mean() )
    # ------------------------------------------------------------------
    def encode_fn(x):
        x_padded = pad_video(x, frames_to_extract, required_pixel_frames)
        return get_crossattn_after_denoise(
            model, x_padded,
            condition=condition_true,
            layer_indices=layer_indices,
            vae_device=vae_device,
            dit_device=dit_device,
            entropy_mode=entropy_mode,
        )

    if entropy_mode:
        loss_fn = lambda outputs, _: sum(t.pow(2).mean() for t in outputs)
    else:
        target_norm = target_acts.norm()
        loss_fn = lambda x, y: 1 - torch.dot(x.float(), y) / (x.float().norm() * target_norm)

    # ------------------------------------------------------------------
    # 9. Run attack
    # ------------------------------------------------------------------
    print(f"\nRunning PGD attack  (steps={args.steps}, alpha={args.alpha:.5f}, eps={args.eps:.4f})...")
    x_adv = pgd(
        raw_state,
        target_acts,  # None in entropy mode; ignored by loss_fn
        encode_fn,
        loss_fn,
        steps=args.steps,
        alpha=args.alpha,
        eps=args.eps,
        device=device,
    )

    # ------------------------------------------------------------------
    # 10. Save outputs
    # ------------------------------------------------------------------
    def to_uint8_frames(t):
        """(1, C, T, H, W) float32 [-1,1] -> (T, H, W, C) uint8"""
        t = t[0].cpu().float()
        t = ((t.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
        return t.permute(1, 2, 3, 0)

    out_dir = WM_ROOT / "attack" / "outputs" / str(int(time.time()))
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(x_adv.cpu(),       out_dir / "x_adv.pt")
    torch.save(raw_state.cpu(),   out_dir / "x_orig.pt")
    torch.save(target_acts.cpu(), out_dir / "target_crossattn.pt")

    torchvision.io.write_video(str(out_dir / "x_adv.mp4"),  to_uint8_frames(x_adv),     fps=16)
    torchvision.io.write_video(str(out_dir / "x_orig.mp4"), to_uint8_frames(raw_state), fps=16)

    print(f"\nSaved adversarial video to : {out_dir / 'x_adv.pt'} / x_adv.mp4")
    print(f"Saved original video to    : {out_dir / 'x_orig.pt'} / x_orig.mp4")
    print(f"Saved target activations to: {out_dir / 'target_crossattn.pt'}")


if __name__ == "__main__":
    main()


"""
# Cosine mode (match target prompt cross-attention):
python cosmos-predict2.5/scripts/attack_cross_attention_generative.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --true_prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate" \
    --target_prompt "Use the franka robot arm to open the drawer and place the cookie box in it" \
    --experiment_name predict2_video2world_training_2b_libero_480 \
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/8fbc6188fa2f2e4ab585dc6aac3edd0e9d8a3670/model.pt \
    --resolution 432,432 \
    --num_latent_video_frames 6 \
    --num_latent_conditional_frames 2 \
    --attack_layers 14 15 16 17 18 \
    --steps 20 --alpha 0.00392 --eps 0.0628 \
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --offload_diffusion_model \
    --offload_tokenizer \
    --offload_text_encoder

# Entropy mode (minimise cross-attention magnitude, no target prompt):
python cosmos-predict2.5/scripts/attack_cross_attention_generative.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --true_prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate" \
    --experiment_name predict2_video2world_training_2b_libero_480 \
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/8fbc6188fa2f2e4ab585dc6aac3edd0e9d8a3670/model.pt \
    --resolution 432,432 \
    --num_latent_video_frames 6 \
    --num_latent_conditional_frames 2 \
    --attack_layers 14 15 16 17 18 \
    --steps 20 --alpha 0.00392 --eps 0.0628 \
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --offload_diffusion_model \
    --offload_tokenizer \
    --offload_text_encoder
"""
