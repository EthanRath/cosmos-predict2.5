"""
PGD attack targeting the cross-attention mechanism of the Video2World DiT.

Perturbs the input video so that the cross-attention activations produced when
the model is conditioned on the *true prompt* match those produced when it is
conditioned on a *target prompt*.  When the adversarial video is later fed to
the full diffusion pipeline with the true prompt, the model generates futures
consistent with the target prompt instead.

Attack objective:
    minimise MSE(
        crossattn(VAE(x_adv),    true_prompt_emb),   <- changes each PGD step
        crossattn(VAE(x_orig),   target_prompt_emb)   <- fixed target
    )

The "model" passed to PGD is a callable that encodes x_adv through the VAE,
pads the latent to the required temporal size, and returns the concatenated
cross-attention activations from every DiT block in a single flat tensor.

Gradients flow: crossattn output -> DiT blocks -> VAE encoder -> x_adv.
DiT and tokenizer parameters are frozen (requires_grad=False) to avoid
accumulating large gradient buffers for the 2B-parameter network.

Usage:
    python cosmos-predict2.5/scripts/attack_cross_attention.py \
        --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
        --true_prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate" \
        --target_prompt "Use the franka robot arm to open the drawer and place the cookiie box in it" \
        --experiment_name predict2_video2world_training_2b_libero_480 \
        --ckpt_path /path/to/model.pt \
        --resolution 432,432 \
        --num_latent_video_frames 6 \
        --num_latent_conditional_frames 2 \
        --timestep 0.5 \
        --steps 20 --alpha 0.00392 --eps 0.0628 \
        --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
        --offload_diffusion_model --offload_tokenizer --offload_text_encoder
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
from test_vae_encoder import load_and_preprocess_video, normalize_video  # noqa: E402
from attack.white_box import pgd                                          # noqa: E402


# ---------------------------------------------------------------------------
# Helpers shared with compute_cross_attention.py
# ---------------------------------------------------------------------------

def pad_video(video, frames_to_extract, required_pixel_frames):
    """
    Mirrors read_and_process_video: place the last `frames_to_extract` pixel
    frames at positions 0..frames_to_extract-1, then repeat the last frame to
    fill the model's required temporal size.
    """
    context = video[:, :, -frames_to_extract:, :, :]
    padding = required_pixel_frames - frames_to_extract
    return torch.cat([context, context[:, :, -1:, :, :].repeat(1, 1, padding, 1, 1)], dim=2)


def get_crossattn_flat(model, video_padded, condition, timestep, num_attack_layers=None):
    """
    DiT forward pass returning a flat tensor of concatenated cross-attention
    outputs.  Retains the computation graph so gradients flow back through the
    DiT and VAE to x_adv.

    Parameters
    ----------
    model             : Video2WorldModelRectifiedFlow
    video_padded      : (1, C, required_pixel_frames, H, W) float32 on CUDA
    condition         : Video2WorldCondition
    timestep          : float in [0, 1]
    num_attack_layers : int or None.
        When set, a detach hook is placed on the output of block N-1 so that
        gradient flow is cut for blocks N onward.  The forward pass still runs
        in full (avoiding any shape-mismatch from calling DiT internals
        directly), but the backward graph only includes blocks 0..N-1, saving
        significant VRAM per PGD step.  None = full backward through all blocks.
    """
    # VAE encode (gradients enabled via tokenizer.enable_grad = True)
    latent = model.tokenizer.encode(video_padded).contiguous().float()

    B, C, T_lat, H_lat, W_lat = latent.shape
    t = timestep

    noise = torch.randn_like(latent)
    xt = ((1 - t) * latent + t * noise).to(**model.tensor_kwargs)

    # Replace first N latent frames with the clean latent (mirrors denoise())
    cond_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C, 1, 1, 1).type_as(xt)
    xt = latent.type_as(xt) * cond_mask + xt * (1 - cond_mask)

    timesteps = torch.full((B, T_lat), t, device=latent.device, dtype=latent.dtype)

    captured = {}
    hooks    = []

    n_blocks = len(model.net.blocks)
    cutoff   = num_attack_layers if num_attack_layers is not None else n_blocks

    for idx, block in enumerate(model.net.blocks):
        if idx >= cutoff:
            break
        if hasattr(block, "cross_attn"):
            def _make_crossattn_hook(i):
                def _hook(_m, _inp, output):
                    captured[i] = output[0] if isinstance(output, tuple) else output
                return _hook
            hooks.append(block.cross_attn.register_forward_hook(_make_crossattn_hook(idx)))

    # If limiting layers: hook the last included block so its OUTPUT is detached
    # before being fed into block[cutoff].  This severs the gradient graph for
    # all later blocks without changing the forward computation.
    if num_attack_layers is not None and cutoff < n_blocks:
        def _detach_hook(_m, _inp, output):
            if isinstance(output, tuple):
                return (output[0].detach(),) + output[1:]
            return output.detach()
        hooks.append(model.net.blocks[cutoff - 1].register_forward_hook(_detach_hook))

    model.net(x_B_C_T_H_W=xt, timesteps_B_T=timesteps, **condition.to_dict())

    for h in hooks:
        h.remove()

    return torch.cat([captured[k].flatten() for k in sorted(captured.keys())])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path",       required=True,
                        help="Path to source input .mp4")
    parser.add_argument("--true_prompt",      required=True,
                        help="Prompt that will be used at inference time (what the model receives)")
    parser.add_argument("--target_prompt",    required=True,
                        help="Prompt whose cross-attention activations we optimise toward")
    parser.add_argument("--experiment_name",  required=True)
    parser.add_argument("--ckpt_path",        required=True)
    parser.add_argument("--resolution",       default="432,432")
    parser.add_argument("--num_latent_video_frames",       type=int, default=6)
    parser.add_argument("--num_latent_conditional_frames", type=int, default=2)
    parser.add_argument("--timestep",         type=float, default=0.5,
                        help="Fixed noise level for the DiT forward pass (default: 0.5)")
    parser.add_argument("--steps",  type=int,   default=20,      help="PGD steps")
    parser.add_argument("--alpha",  type=float, default=1/255,   help="PGD step size")
    parser.add_argument("--eps",    type=float, default=16/255,  help="PGD epsilon")
    parser.add_argument("--num_attack_layers", type=int, default=None,
                        help="Only backprop through the first N DiT blocks. "
                             "Drastically reduces peak VRAM. Default: all blocks.")
    parser.add_argument("--offload_diffusion_model", action="store_true")
    parser.add_argument("--offload_text_encoder",    action="store_true")
    parser.add_argument("--offload_tokenizer",       action="store_true")
    parser.add_argument("--context_parallel_size",   type=int, default=1)
    parser.add_argument("--config_file",
                        default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    args = parser.parse_args()

    device = "cuda"
    H, W = [int(x) for x in args.resolution.split(",")]
    num_pixel_frames  = (args.num_latent_video_frames - 1) * 4 + 1
    frames_to_extract = 4 * (args.num_latent_conditional_frames - 1) + 1

    # ------------------------------------------------------------------
    # 1. Load and preprocess video
    # ------------------------------------------------------------------
    print(f"Loading video: {args.video_path}")
    video_uint8 = load_and_preprocess_video(args.video_path, [H, W], num_pixel_frames)
    raw_state   = normalize_video(video_uint8, device=device)   # (1, C, T, H, W) float32 [-1,1]
    print(f"Video shape  : {raw_state.shape}")

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
    model      = inference.model
    state_t    = model.config.state_t
    required_pixel_frames = (state_t - 1) * 4 + 1

    # ------------------------------------------------------------------
    # 3. Compute T5 embeddings for both prompts (text encoder still on GPU)
    # ------------------------------------------------------------------
    video_bf16 = pad_video(raw_state, frames_to_extract, required_pixel_frames).to(dtype=torch.bfloat16)

    print("Computing T5 embeddings for true prompt...")
    data_batch_true = inference._get_data_batch_input(
        video=video_bf16, prompt=args.true_prompt,
        num_conditional_frames=args.num_latent_conditional_frames,
        negative_prompt=_DEFAULT_NEGATIVE_PROMPT, use_neg_prompt=True,
    )
    data_batch_true["video"]          = video_bf16
    data_batch_true[IS_PREPROCESSED_KEY] = True

    print("Computing T5 embeddings for target prompt...")
    data_batch_target = inference._get_data_batch_input(
        video=video_bf16, prompt=args.target_prompt,
        num_conditional_frames=args.num_latent_conditional_frames,
        negative_prompt=_DEFAULT_NEGATIVE_PROMPT, use_neg_prompt=True,
    )
    data_batch_target["video"]          = video_bf16
    data_batch_target[IS_PREPROCESSED_KEY] = True

    # ------------------------------------------------------------------
    # 4. Offloading sequence (mirrors generate_vid2world)
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
    # 5. Freeze DiT and tokenizer parameters so no gradient buffers are
    #    accumulated for the 2B-parameter network during PGD backprop.
    #    Gradients still FLOW THROUGH frozen parameters w.r.t. the input.
    # ------------------------------------------------------------------
    for p in model.net.parameters():
        p.requires_grad_(False)
    for p in model.tokenizer.model.model.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # 6. Build condition objects (T5 emb + video conditioning mask)
    # ------------------------------------------------------------------
    print("Building condition objects...")
    with torch.no_grad():
        _, _, condition_true   = model.get_data_and_condition(data_batch_true)
        _, _, condition_target = model.get_data_and_condition(data_batch_target)

    condition_true = condition_true.edit_for_inference(
        is_cfg_conditional=True,
        num_conditional_frames=args.num_latent_conditional_frames,
    )
    condition_target = condition_target.edit_for_inference(
        is_cfg_conditional=True,
        num_conditional_frames=args.num_latent_conditional_frames,
    )

    # Enable gradient flow through the VAE encoder
    model.tokenizer.enable_grad = True

    # ------------------------------------------------------------------
    # 7. Compute fixed target: crossattn(true_video, target_prompt) [no grad]
    # ------------------------------------------------------------------
    print("Computing target cross-attention activations (true video + target prompt)...")
    if args.num_attack_layers is not None:
        print(f"Using first {args.num_attack_layers} DiT blocks (of {len(model.net.blocks)} total)")
    video_padded = pad_video(raw_state, frames_to_extract, required_pixel_frames)
    with torch.no_grad():
        target_acts = get_crossattn_flat(
            model, video_padded, condition_target, args.timestep, args.num_attack_layers
        )
    target_acts = target_acts.detach()
    print(f"Target activations shape: {target_acts.shape}")

    # ------------------------------------------------------------------
    # 8. Define encode_fn and loss for PGD
    #    encode_fn(x) -> crossattn(VAE(pad(x)), true_prompt_emb)
    # ------------------------------------------------------------------
    def encode_fn(x):
        x_padded = pad_video(x, frames_to_extract, required_pixel_frames)
        return get_crossattn_flat(
            model, x_padded, condition_true, args.timestep, args.num_attack_layers
        )

    mse_loss = nn.MSELoss()
    loss_fn  = lambda pred, tgt: mse_loss(pred, tgt)

    # ------------------------------------------------------------------
    # 9. Run PGD
    # ------------------------------------------------------------------
    print(f"Running PGD attack (steps={args.steps}, alpha={args.alpha:.5f}, eps={args.eps:.4f})...")
    x_adv = pgd(
        raw_state,
        target_acts,
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

    torch.save(x_adv.cpu(),        out_dir / "x_adv.pt")
    torch.save(raw_state.cpu(),    out_dir / "x_orig.pt")
    torch.save(target_acts.cpu(),  out_dir / "target_crossattn.pt")

    torchvision.io.write_video(str(out_dir / "x_adv.mp4"),  to_uint8_frames(x_adv),     fps=16)
    torchvision.io.write_video(str(out_dir / "x_orig.mp4"), to_uint8_frames(raw_state), fps=16)

    print(f"Saved adversarial video to : {out_dir / 'x_adv.pt'} / x_adv.mp4")
    print(f"Saved original video to    : {out_dir / 'x_orig.pt'} / x_orig.mp4")
    print(f"Saved target activations to: {out_dir / 'target_crossattn.pt'}")


if __name__ == "__main__":
    main()


"""
python cosmos-predict2.5/scripts/attack_cross_attention.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --true_prompt "Use the franka robot arm to pick up the block on the left" \
    --target_prompt "Use the franka robot arm to turn on the stove and place the moka pot on it" \
    --experiment_name predict2_video2world_training_2b_libero_480 \
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/8fbc6188fa2f2e4ab585dc6aac3edd0e9d8a3670/model.pt \
    --resolution 432,432 \
    --num_latent_video_frames 6 \
    --num_latent_conditional_frames 2 \
    --timestep 0.5 \
    --steps 20 --alpha 0.00392 --eps 0.0628 \
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --offload_diffusion_model \
    --offload_tokenizer \
    --offload_text_encoder
"""
