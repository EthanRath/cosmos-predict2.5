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


# ---------------------------------------------------------------------------
# Core: denoising steps + cross-attention capture
# ---------------------------------------------------------------------------

def get_crossattn_after_denoise(
    model,
    video,
    condition_denoise,
    condition_attn,
    num_denoise_steps,
    layer_indices,
    seed=42,
    device="cuda",
):
    """
    Encode `video` through the VAE, run `num_denoise_steps` Euler denoising
    steps, and return the concatenated cross-attention outputs from the final
    step.

    Parameters
    ----------
    model              : Video2WorldModelRectifiedFlow
    video              : (1, C, T_px, H, W) float32 on device, requires_grad or not
    condition_denoise  : Video2WorldCondition used for the k-1 no_grad Euler steps
    condition_attn     : Video2WorldCondition whose T5 emb drives the final cross-attn
                         (true_prompt for encode_fn, target_prompt for fixed target)
    num_denoise_steps  : int ≥ 1.  Total Euler steps.  First k-1 are no_grad;
                         last step runs with grad and captures cross-attn.
    layer_indices      : list[int] | None.  Block indices to capture.  None = all.
    seed               : int.  RNG seed for the initial noise sample.
    device             : str.

    Returns
    -------
    flat (N,) float32 tensor with grad attached (grad flows through VAE encoder
    via the conditional frame portion of x_in_final).
    """
    kw = model.tensor_kwargs   # e.g. {'device': 'cuda', 'dtype': torch.bfloat16}

    # --- 1. VAE encode (gradient enabled for the backward pass) ---
    latent = model.tokenizer.encode(video).contiguous().float()  # (B, C, T_lat, H_lat, W_lat)
    B, C, T_lat, H_lat, W_lat = latent.shape

    # Conditioning mask: 1 for cond frames (kept clean), 0 for generation frames
    cond_mask = condition_denoise.condition_video_input_mask_B_C_T_H_W \
                    .repeat(1, C, 1, 1, 1).type_as(latent)   # (B, C, T_lat, H_lat, W_lat)

    # --- 2. Initialise at t=1 (pure noise for gen frames, clean for cond frames) ---
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    noise = torch.randn(latent.shape, generator=gen, device=device, dtype=latent.dtype)

    latent_det = latent.detach()
    x_t = latent_det * cond_mask + noise * (1 - cond_mask)   # detached; no grad needed yet

    # Rectified flow schedule: t descends from 1 → 0 over num_denoise_steps+1 points
    ts = torch.linspace(1.0, 0.0, num_denoise_steps + 1, device=device)

    # --- 3. Early Euler steps (no grad) ---
    with torch.no_grad():
        for step_idx in range(num_denoise_steps - 1):
            t_cur  = ts[step_idx].item()
            t_next = ts[step_idx + 1].item()
            dt     = t_next - t_cur           # negative (t decreasing)

            t_tensor = torch.full((B, T_lat), t_cur, device=device, dtype=latent.dtype)
            # Pin cond frames to clean latent at every step (mirrors the model's denoise())
            x_in = latent_det * cond_mask + x_t * (1 - cond_mask)

            v = model.net(
                x_B_C_T_H_W  = x_in.to(**kw),
                timesteps_B_T = t_tensor.to(**kw),
                **condition_denoise.to_dict(),
            )
            if isinstance(v, tuple):
                v = v[0]
            v  = v.float()
            x_t = x_t + dt * v
            # Re-pin cond frames (prevent drift)
            x_t = latent_det * cond_mask + x_t * (1 - cond_mask)

    # --- 4. Final Euler step (WITH grad) — cross-attn captured here ---
    t_cur    = ts[num_denoise_steps - 1].item()
    t_tensor = torch.full((B, T_lat), t_cur, device=device, dtype=latent.dtype)

    # latent has grad (cond frames); x_t is detached (gen frames).
    # Gradient flows: cross-attn output → model.net → x_in_final
    #                 → latent * cond_mask → VAE encoder → video (x_adv)
    x_in_final = latent * cond_mask + x_t * (1 - cond_mask)

    # Register cross-attn hooks on the requested blocks
    captured    = {}
    hooks       = []
    n_blocks    = len(model.net.blocks)
    idxs        = layer_indices if layer_indices is not None else list(range(n_blocks))
    max_block   = max(idxs) if idxs else n_blocks - 1

    for idx in idxs:
        if idx >= n_blocks or not hasattr(model.net.blocks[idx], "cross_attn"):
            continue
        def _make_hook(i):
            def _hook(_m, _inp, output):
                captured[i] = output[0] if isinstance(output, tuple) else output
            return _hook
        hooks.append(model.net.blocks[idx].cross_attn.register_forward_hook(_make_hook(idx)))

    # Cut gradient flow after the last captured block to save VRAM.
    # The forward pass still runs in full (avoids shape mismatches), but the
    # autograd graph is severed so later blocks don't accumulate activation buffers.
    if max_block < n_blocks - 1:
        def _detach_hook(_m, _inp, output):
            if isinstance(output, tuple):
                return (output[0].detach(),) + output[1:]
            return output.detach()
        hooks.append(model.net.blocks[max_block].register_forward_hook(_detach_hook))

    model.net(
        x_B_C_T_H_W  = x_in_final.to(**kw),
        timesteps_B_T = t_tensor.to(**kw),
        **condition_attn.to_dict(),
    )

    for h in hooks:
        h.remove()

    if not captured:
        raise RuntimeError(
            f"No cross-attn outputs captured from blocks {idxs}. "
            "Check --attack_layers and that those blocks have a cross_attn attribute."
        )

    return torch.cat([captured[k].flatten() for k in sorted(captured.keys())])


# ---------------------------------------------------------------------------
# Alternating mode: latent-space PGD + pixel-space PGD (low VRAM)
# ---------------------------------------------------------------------------

def get_crossattn_from_latent(
    model,
    latent,
    condition_denoise,
    condition_attn,
    num_denoise_steps,
    layer_indices,
    seed=42,
    device="cuda",
):
    """
    Like get_crossattn_after_denoise but receives an already-encoded VAE latent
    instead of a pixel video.  Gradient flows from cross-attn → DiT → latent
    without touching the VAE encoder — enabling Phase 1 of the alternating attack.

    Parameters
    ----------
    latent : (B, C, T_lat, H_lat, W_lat) float32, may have requires_grad=True.
    All other parameters identical to get_crossattn_after_denoise.
    """
    kw = model.tensor_kwargs

    B, C, T_lat, H_lat, W_lat = latent.shape

    cond_mask = condition_denoise.condition_video_input_mask_B_C_T_H_W \
                    .repeat(1, C, 1, 1, 1).type_as(latent)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    noise = torch.randn(latent.shape, generator=gen, device=device, dtype=latent.dtype)

    latent_det = latent.detach()
    x_t = latent_det * cond_mask + noise * (1 - cond_mask)

    ts = torch.linspace(1.0, 0.0, num_denoise_steps + 1, device=device)

    with torch.no_grad():
        for step_idx in range(num_denoise_steps - 1):
            t_cur  = ts[step_idx].item()
            t_next = ts[step_idx + 1].item()
            dt     = t_next - t_cur

            t_tensor = torch.full((B, T_lat), t_cur, device=device, dtype=latent_det.dtype)
            x_in = latent_det * cond_mask + x_t * (1 - cond_mask)
            v = model.net(
                x_B_C_T_H_W  = x_in.to(**kw),
                timesteps_B_T = t_tensor.to(**kw),
                **condition_denoise.to_dict(),
            )
            if isinstance(v, tuple):
                v = v[0]
            v = v.float()
            x_t = x_t + dt * v
            x_t = latent_det * cond_mask + x_t * (1 - cond_mask)

    # Final step with grad — latent contributes via cond_mask portion
    t_cur    = ts[num_denoise_steps - 1].item()
    t_tensor = torch.full((B, T_lat), t_cur, device=device, dtype=latent.dtype)
    x_in_final = latent * cond_mask + x_t * (1 - cond_mask)

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

    if max_block < n_blocks - 1:
        def _detach_hook(_m, _inp, output):
            if isinstance(output, tuple):
                return (output[0].detach(),) + output[1:]
            return output.detach()
        hooks.append(model.net.blocks[max_block].register_forward_hook(_detach_hook))

    model.net(
        x_B_C_T_H_W  = x_in_final.to(**kw),
        timesteps_B_T = t_tensor.to(**kw),
        **condition_attn.to_dict(),
    )

    for h in hooks:
        h.remove()

    if not captured:
        raise RuntimeError("No cross-attn outputs captured from get_crossattn_from_latent.")

    return torch.cat([captured[k].flatten() for k in sorted(captured.keys())])


def pgd_latent(z_init, emb_target, crossattn_fn, target_norm, steps, alpha, eps):
    """
    PGD in VAE latent space.  Unlike pixel-space PGD, latents are not bounded
    to [-1, 1]; we project the perturbation to an L-inf ball of radius `eps`
    around the initial latent.

    Parameters
    ----------
    z_init       : (B, C, T, H, W) float32 detached latent tensor
    emb_target   : fixed cross-attn target embedding
    crossattn_fn : callable, takes latent → flat cross-attn embedding
    loss_fn      : callable, takes (prediction, target) → scalar loss
    steps        : int, number of PGD steps
    alpha        : float, step size in latent space
    eps          : float, L-inf ball radius in latent space

    Returns
    -------
    z_adv : (B, C, T, H, W) float32 updated latent
    """
    
    z_orig = z_init.clone().detach()
    z_adv  = z_init.clone().detach()
    #cos_loss = lambda x, y: 1.0 - torch.dot(x.float(), y) / (x.float().norm() * target_norm)
    with torch.no_grad():
        cos_sim = torch.nn.functional.cosine_similarity
        attn = crossattn_fn(z_adv)
        sim = 1-cos_sim(attn, emb_target, dim=0)
    print(sim.shape)
    loss_fn = lambda x, y: 1-cos_sim(x*sim,y, dim=0).mean()

    for _ in range(steps):
        z_adv.requires_grad_(True)

        attn = crossattn_fn(z_adv)
        cost = loss_fn(attn, emb_target)
        print(f"  [latent PGD] loss: {cost.item():.6f}", end="\r")

        grad = torch.autograd.grad(cost, z_adv, retain_graph=False)[0]

        z_adv  = z_adv.detach() - alpha * grad.sign()
        delta  = torch.clamp(z_adv - z_orig, min=-eps, max=eps)
        z_adv  = (z_orig + delta).detach()

    print()
    return z_adv


def alternating_attack(
    x_orig,
    emb_target,
    target_norm,
    model,
    condition_true,
    frames_to_extract,
    required_pixel_frames,
    num_latent_conditional_frames,
    num_denoise_steps,
    layer_indices,
    outer_steps,
    embed_steps,
    pixel_steps,
    alpha,
    eps,
    embed_alpha,
    embed_eps,
    seed,
    device,
    offload_net=False,
):
    """
    Two-phase alternating attack (low VRAM):

    Phase 1 — latent PGD (DiT backprop, no VAE backprop):
        Encode x_adv → z with no_grad.  PGD in latent space to find z_star
        whose cross-attn is closer to emb_target.
        Gradient: cross-attn → DiT → z  (VAE encoder NOT in graph).

    Phase 2 — pixel PGD (VAE backprop only, no DiT):
        Only encode the conditioning frames (last frames_to_extract pixel frames),
        giving num_latent_conditional_frames latent frames.  Match those against
        z_star's conditioning portion using cosine distance.
        Gradient: cos_dist(VAE_cond(x_adv), z_star_cond) → VAE → x_adv
        (DiT NOT in graph, and only a small fraction of frames encoded → low VRAM).

    VRAM savings vs end-to-end:
      - DiT and VAE activation buffers never coexist.
      - Phase 2 encodes only conditioning frames (e.g., 5 px frames → 2 lat frames)
        instead of the full padded video (93 px frames).  This is the main fix for
        OOM: attack_vae_encoder.py works because it encodes a small clip; Phase 2
        now matches that behaviour.
      - If offload_net=True, model.net is moved to CPU during Phase 2 and back
        before Phase 1, freeing ~4 GB (2B params @ bfloat16).
    """
    x_adv = x_orig.clone().detach()

    crossattn_loss_fn = lambda x, y: 1.0 - torch.dot(x.float(), y) / (x.float().norm() * target_norm)

    for outer in range(outer_steps):
        print(f"\nOuter iteration {outer + 1}/{outer_steps}")

        # ------------------------------------------------------------------
        # Phase 1: PGD in latent space  (DiT on GPU, VAE no-grad)
        # ------------------------------------------------------------------
        if offload_net and outer > 0:
            # Reload DiT for this iteration's Phase 1
            print("  [offload] Moving DiT → GPU")
            model.net = model.net.to(device)
            torch.cuda.empty_cache()

        with torch.no_grad():
            x_padded = pad_video(x_adv, frames_to_extract, required_pixel_frames)
            z_adv = model.tokenizer.encode(x_padded).contiguous().float()

        print(f"  Phase 1 — latent PGD ({embed_steps} steps, alpha={embed_alpha}, eps={embed_eps})")

        embed_fn = lambda z: get_crossattn_from_latent(
            model, z, condition_true, condition_true,
            num_denoise_steps, layer_indices, seed, device,
        )
        z_star = pgd_latent(
            z_adv, emb_target, embed_fn, target_norm,
            steps=embed_steps, alpha=embed_alpha, eps=embed_eps,
        )

        # ------------------------------------------------------------------
        # Phase 2: pixel-space PGD  (VAE only, DiT optionally offloaded)
        # Only match conditioning latent frames — avoids encoding 93-frame
        # padded video and dramatically cuts VRAM vs. Phase 1.
        # ------------------------------------------------------------------
        if offload_net:
            print("  [offload] Moving DiT → CPU")
            model.net = model.net.to("cpu")
            torch.cuda.empty_cache()

        print(f"  Phase 2 — pixel PGD  ({pixel_steps} steps, alpha={alpha:.5f}, eps={eps:.4f})")

        # Target: only the conditioning portion of z_star (e.g., 2 latent frames)
        z_star_cond      = z_star[:, :, :num_latent_conditional_frames, :, :].detach().float()
        z_star_cond_flat = z_star_cond.flatten()
        z_star_cond_norm = z_star_cond_flat.norm()

        def vae_encode_cond(x):
            # Extract last frames_to_extract pixel frames (the real conditioning content)
            x_cond = x[:, :, -frames_to_extract:, :, :]
            z_cond = model.tokenizer.encode(x_cond).contiguous().float()
            return z_cond[:, :, :num_latent_conditional_frames, :, :].flatten()

        vae_cos_loss = lambda z, z_t: 1.0 - torch.dot(z.float(), z_t) / (z.float().norm() * z_star_cond_norm)

        x_adv = pgd(
            x_adv, z_star_cond_flat, vae_encode_cond, vae_cos_loss,
            steps=pixel_steps, alpha=alpha, eps=eps, device=device,
        )

    # Ensure DiT is back on GPU after the last iteration
    if offload_net:
        model.net = model.net.to(device)
        torch.cuda.empty_cache()

    return x_adv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path",       required=True,
                        help="Path to source input .mp4")
    parser.add_argument("--true_prompt",      required=True,
                        help="Prompt used at inference time (what the model receives)")
    parser.add_argument("--target_prompt",    required=True,
                        help="Prompt whose cross-attention activations we optimise toward")
    parser.add_argument("--experiment_name",  required=True)
    parser.add_argument("--ckpt_path",        required=True)
    parser.add_argument("--resolution",       default="432,432")
    parser.add_argument("--num_latent_video_frames",       type=int, default=6)
    parser.add_argument("--num_latent_conditional_frames", type=int, default=2)
    parser.add_argument(
        "--num_denoise_steps", type=int, default=5,
        help="Euler denoising steps before cross-attn capture. "
             "Effective capture timestep = 1/num_denoise_steps. "
             "k=1 → t=1.0 (pure noise, equiv. to original attack); "
             "k=2 → t=0.5; k=5 → t=0.2; k=10 → t=0.1. (default: 5)",
    )
    parser.add_argument(
        "--attack_layers", type=int, nargs="+", default=None,
        help="DiT block indices whose cross-attn to target. "
             "Gradient is also cut after the highest index to save VRAM. "
             "Default: all blocks.",
    )
    parser.add_argument("--seed",   type=int,   default=42,      help="RNG seed for initial noise (default: 42)")
    parser.add_argument("--steps",  type=int,   default=20,
                        help="PGD steps (standard mode) or pixel-space PGD steps per outer "
                             "iteration (alternating mode). (default: 20)")
    parser.add_argument("--alpha",  type=float, default=1/255,   help="Pixel-space PGD step size (default: 1/255)")
    parser.add_argument("--eps",    type=float, default=16/255,  help="Pixel-space PGD epsilon (default: 16/255)")
    # Alternating mode
    parser.add_argument(
        "--alternating", action="store_true",
        help="Enable two-phase alternating attack: Phase 1 PGD in latent space (DiT only, "
             "no VAE backprop), Phase 2 PGD in pixel space (VAE only, no DiT backprop). "
             "Halves peak VRAM vs. the standard end-to-end mode.",
    )
    parser.add_argument("--outer_steps",  type=int,   default=10,
                        help="[alternating] Number of outer iterations (default: 10)")
    parser.add_argument("--embed_steps",  type=int,   default=5,
                        help="[alternating] Phase 1 latent-space PGD steps per outer iter (default: 5)")
    parser.add_argument("--embed_alpha",  type=float, default=0.05,
                        help="[alternating] Phase 1 latent-space step size (default: 0.05). "
                             "Latents are ~[-5, 5], so scale accordingly.")
    parser.add_argument("--embed_eps",    type=float, default=0.5,
                        help="[alternating] Phase 1 latent-space L-inf epsilon (default: 0.5)")
    parser.add_argument(
        "--offload_net_between_phases", action="store_true",
        help="[alternating] Move the DiT to CPU during Phase 2 and reload before Phase 1. "
             "Frees ~4 GB VRAM at the cost of extra CPU↔GPU transfer time per outer iteration.",
    )
    parser.add_argument("--offload_diffusion_model", action="store_true")
    parser.add_argument("--offload_text_encoder",    action="store_true")
    parser.add_argument("--offload_tokenizer",       action="store_true")
    parser.add_argument("--context_parallel_size",   type=int, default=1)
    parser.add_argument(
        "--config_file",
        default="cosmos_predict2/_src/predict2/configs/video2world/config.py",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    H, W = [int(x) for x in args.resolution.split(",")]
    num_pixel_frames  = (args.num_latent_video_frames - 1) * 4 + 1
    frames_to_extract = 4 * (args.num_latent_conditional_frames - 1) + 1
    eff_t = 1.0 / args.num_denoise_steps

    print(f"Resolution          : {args.resolution}")
    print(f"Latent T frames     : {args.num_latent_video_frames}")
    print(f"Pixel T frames      : {num_pixel_frames}")
    print(f"Denoising steps     : {args.num_denoise_steps}  (capture at t ≈ {eff_t:.3f})")
    print(f"Attack layers       : {args.attack_layers or 'all'}")
    print(f"PGD eps             : {args.eps:.4f}")
    if args.alternating:
        print(f"Mode                : ALTERNATING  "
              f"(outer={args.outer_steps}, embed={args.embed_steps}×α={args.embed_alpha}/ε={args.embed_eps}, "
              f"pixel={args.steps}×α={args.alpha:.5f}/ε={args.eps:.4f})")
    else:
        print(f"Mode                : standard end-to-end  (steps={args.steps})")

    # ------------------------------------------------------------------
    # 1. Load and preprocess video
    # ------------------------------------------------------------------
    print(f"\nLoading video: {args.video_path}")
    video_uint8 = load_and_preprocess_video(args.video_path, [H, W], num_pixel_frames)
    raw_state   = normalize_video(video_uint8, device=device)   # (1, C, T, H, W) float32 [-1,1]
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
        if device == "cuda":
            torch.cuda.empty_cache()

    if inference.offload_tokenizer:
        if hasattr(model.tokenizer, "encoder") and model.tokenizer.encoder is not None:
            model.tokenizer.encoder = model.tokenizer.encoder.to(device)
        if device == "cuda":
            torch.cuda.empty_cache()

    if inference.offload_diffusion_model:
        model.net = model.net.to(device)
        if hasattr(model, "conditioner") and model.conditioner is not None:
            model.conditioner = model.conditioner.to(device)
        if device == "cuda":
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
    # ------------------------------------------------------------------
    print("\nComputing fixed target cross-attention activations...")
    video_padded = pad_video(raw_state, frames_to_extract, required_pixel_frames)
    with torch.no_grad():
        target_acts = get_crossattn_after_denoise(
            model, video_padded,
            condition_denoise=condition_true,    # denoising uses true_prompt trajectory
            condition_attn=condition_target,     # but cross-attn is measured under target_prompt
            num_denoise_steps=args.num_denoise_steps,
            layer_indices=layer_indices,
            seed=args.seed,
            device=device,
        ).detach().float()
    print(f"Target activations shape : {target_acts.shape}")

    # ------------------------------------------------------------------
    # 8. Define encode_fn and loss for PGD
    #    encode_fn(x_adv) -> crossattn(VAE(pad(x_adv)), true_prompt)
    # ------------------------------------------------------------------
    def encode_fn(x):
        x_padded = pad_video(x, frames_to_extract, required_pixel_frames)
        return get_crossattn_after_denoise(
            model, x_padded,
            condition_denoise=condition_true,
            condition_attn=condition_true,
            num_denoise_steps=args.num_denoise_steps,
            layer_indices=layer_indices,
            seed=args.seed,
            device=device,
        )

    target_norm = target_acts.norm()
    loss_fn = lambda x, y: 1 - torch.dot(x.float(), y) / (x.float().norm() * target_norm)

    # ------------------------------------------------------------------
    # 9. Run attack
    # ------------------------------------------------------------------
    if args.alternating:
        print("\nRunning alternating attack...")
        x_adv = alternating_attack(
            x_orig                    = raw_state,
            emb_target                = target_acts,
            target_norm               = target_acts.norm(),
            model                     = model,
            condition_true            = condition_true,
            frames_to_extract         = frames_to_extract,
            required_pixel_frames     = required_pixel_frames,
            num_latent_conditional_frames = args.num_latent_conditional_frames,
            num_denoise_steps         = args.num_denoise_steps,
            layer_indices             = layer_indices,
            outer_steps               = args.outer_steps,
            embed_steps               = args.embed_steps,
            pixel_steps               = args.steps,
            alpha                     = args.alpha,
            eps                       = args.eps,
            embed_alpha               = args.embed_alpha,
            embed_eps                 = args.embed_eps,
            seed                      = args.seed,
            device                    = device,
            offload_net               = args.offload_net_between_phases,
        )
    else:
        print(f"\nRunning PGD attack  (steps={args.steps}, alpha={args.alpha:.5f}, eps={args.eps:.4f})...")
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
# Standard end-to-end mode (high VRAM):
python cosmos-predict2.5/scripts/attack_cross_attention_generative.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --true_prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate" \
    --target_prompt "Use the franka robot arm to open the drawer and place the cookie box in it" \
    --experiment_name predict2_video2world_training_2b_libero_480 \
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/8fbc6188fa2f2e4ab585dc6aac3edd0e9d8a3670/model.pt \
    --resolution 432,432 \
    --num_latent_video_frames 6 \
    --num_latent_conditional_frames 2 \
    --num_denoise_steps 5 \
    --attack_layers 14 15 16 17 18 \
    --steps 20 --alpha 0.00392 --eps 0.0628 \
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --offload_diffusion_model \
    --offload_tokenizer \
    --offload_text_encoder

# Alternating mode (low VRAM):
python cosmos-predict2.5/scripts/attack_cross_attention_generative.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --true_prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate" \
    --target_prompt "Use the franka robot arm to open the drawer and place the cookie box in it" \
    --experiment_name predict2_video2world_training_2b_libero_480 \
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/8fbc6188fa2f2e4ab585dc6aac3edd0e9d8a3670/model.pt \
    --resolution 432,432 \
    --num_latent_video_frames 6 \
    --num_latent_conditional_frames 2 \
    --num_denoise_steps 5 \
    --attack_layers 14 15 16 17 18 \
    --alternating \
    --outer_steps 10 \
    --embed_steps 5 --embed_alpha 0.05 --embed_eps 0.5 \
    --steps 10 --alpha 0.00392 --eps 0.0628 \
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --offload_diffusion_model \
    --offload_tokenizer \
    --offload_text_encoder
"""
