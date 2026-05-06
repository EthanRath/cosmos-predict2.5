"""
PGD attack minimising cross-attention between the text prompt and generated
video tokens in Cosmos Video2World.

Adds an imperceptible perturbation to the first conditioning frame (or to
latent frames with --skip_latent) so that, at each DiT block, the raw
cross-attention scores between video queries and text keys collapse toward
zero.  When these scores are suppressed the model effectively ignores the
prompt during generation, causing the output to drift from the intended
behaviour without any obvious artefact on the conditioning input.

Loss (cross-attention analogue of the L_freeze objective):

    L_cross = (1/L) * Σ_l  ||S^(l)||_F^2

where S^(l) is the per-layer post-softmax cross-attention weight matrix.
Because Q has shape (B, S, H, D) with S = T*H_patch*W_patch (potentially
very large), we use the same spatial-mean approximation as the freeze
attack: Q is averaged over spatial tokens within each temporal frame,
giving frame-level centroids (B, T, H, D).  K retains its full text-token
shape (B, N, H, D).  The resulting (B, T, N, H) score matrix is the
quantity we minimise.

Optimisation:

    min_{||δ||_∞ ≤ ε}  E_t [ L_cross(x_cond + δ; t) ]

where t ~ U[0,1] is resampled at every PGD step.  Only the first
conditioning pixel frame is perturbed (or the first num_latent_conditional_frames
latent frames when --skip_latent is set).

VRAM knobs
----------
--num_latent_video_frames   Reduce below the model's default state_t to shorten
                            the latent sequence.  Default: 24.
--attack_layers_start       First DiT block index to include in the loss (inclusive,
                            default: 0).
--attack_layers_end         Last DiT block index to include in the loss (inclusive).
                            Gradient graph is cut after this block. Default: all blocks.

Input modes
-----------
Single video  : --video_path <file> --prompt "<text>"
Batch         : --batch_path <dir>  (contains videos/ and metas/ subfolders)
                --attack_num <N>    sample N videos from the batch (-1 = all)

Saved artefacts (attack/outputs/crossattn_<timestamp>/):
    <name>_adv_<skip_latent>.pt : perturbed video/latent tensor
    x_adv.mp4                   : perturbed video (pixel-space attacks only)

Usage (single video):
    python cosmos-predict2.5/scripts/attack_crossattn.py \
        --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
        --prompt "Use the franka robot arm to pick up the black bowl" \
        --resolution 432,432 \
        --num_latent_video_frames 9 \
        --num_latent_conditional_frames 2 \
        --steps 100 --alpha 0.00392 --eps 0.0628 \
        --attack_layers_end 13

Usage (batch):
    python cosmos-predict2.5/scripts/attack_crossattn.py \
        --batch_path /mnt/ssd2/libero/eval_set/ \
        --attack_num 10 \
        --resolution 432,432 \
        --num_latent_video_frames 9 \
        --num_latent_conditional_frames 2 \
        --steps 100 --alpha 0.00392 --eps 0.0628 \
        --attack_layers_end 13 --skip_latent
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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
from probing.test_vae_encoder import load_and_preprocess_video, normalize_video, resize_input  # noqa: E402
from attack.white_box import pgd, freq_pgd, lab_freq_pgd
from attack.eval_wm import eval, eval_latent
from attack.shared_config import (
    ckpt_path, experiment_name, config_file
)

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _move_condition_to_device(condition, device):
    """Move all tensor attributes of a frozen condition dataclass to `device`.
    Uses object.__setattr__ to bypass the frozen-dataclass restriction."""
    for attr in vars(condition):
        val = getattr(condition, attr)
        if isinstance(val, torch.Tensor):
            object.__setattr__(condition, attr, val.to(device))
    return condition


# ---------------------------------------------------------------------------
# Video padding
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
# Cross-attention loss hooks
# ---------------------------------------------------------------------------

def install_crossattn_hooks(net, T_tok, loss_terms, layer_start=0, layer_end=None):
    """
    Wrap block.cross_attn.compute_attention on blocks [layer_start, layer_end].

    For each intercepted block the wrapper:
      1. Computes the spatial-mean Q per temporal frame → (B, T, H, D)
         (Q comes from video tokens; K comes from text tokens — no spatial
         averaging needed on K since N_text is already small)
      2. Computes the (B, T, S_per_frame, N, H) raw cross-attention score
         matrix, preserving the full spatial dimension
      3. Appends the raw score tensor to `loss_terms`

    The caller decides how to aggregate the collected tensors:
      - Untargeted attack: ||scores||_F^2 per block, then mean
      - Targeted attack  : flatten all scores into a vector, cosine similarity
                           against a precomputed reference vector

    Parameters
    ----------
    net         : the DiT net (model.net)
    T_tok       : number of temporal token-frames
    loss_terms  : list — raw (B, T, N, H) score tensors appended during forward
    layer_start : first block index to hook (inclusive, default: 0)
    layer_end   : last block index to hook (inclusive, default: last block)

    Returns
    -------
    restore : callable — removes all patches
    """
    _patches = {}
    n_blocks = len(net.blocks)
    end      = layer_end if layer_end is not None else n_blocks - 1

    for idx in range(layer_start, end + 1):
        if idx >= n_blocks:
            break
        block = net.blocks[idx]
        if not hasattr(block, "cross_attn"):
            continue

        attn    = block.cross_attn
        orig_fn = attn.compute_attention   # bound method

        def _make_wrapper(fn):
            def _wrapper(q, k, v, **kw):
                # q: (B, S, H, D)  — video spatial-temporal tokens
                # k: (B, N, H, D)  — text tokens (N = text sequence length)
                B, S, H, D  = q.shape
                S_per_frame = S // T_tok
                scale       = D ** -0.5

                # Reshape to (B, T, S_per_frame, H, D) — keep full spatial dimension
                q_spatial = q.reshape(B, T_tok, S_per_frame, H, D)

                # Cast to fp32 before einsum: bf16 has only 7 mantissa bits, so
                # small score differences get quantised away, flattening the loss.
                scores = torch.einsum(
                    "btshd,bnhd->btsnh", q_spatial.float(), k.float()
                ) * scale

                # Apply softmax over the text-token (N) dimension to get the
                # actual attention weights used in the paper's loss (Eq. 1-3, 7).
                # Raw pre-softmax scores are unbounded; the post-softmax weights
                # live in [0,1] and sum to 1 over N, which is what the paper
                # minimises (||A||_F^2) or matches (cosine similarity).
                attn_weights = F.softmax(scores, dim=3)
                loss_terms.append(attn_weights)

                return fn(q, k, v, **kw)
            return _wrapper

        attn.__dict__["compute_attention"] = _make_wrapper(orig_fn)
        _patches[idx] = (attn, orig_fn)

    def _restore():
        for _idx, (attn_mod, _orig) in _patches.items():
            attn_mod.__dict__.pop("compute_attention", None)

    return _restore


# ---------------------------------------------------------------------------
# Spatial importance mask (full-trajectory, no-grad)
# ---------------------------------------------------------------------------

def compute_sim_mask(model, latent, condition, target_condition, T_tok,
                     layer_start=0, layer_end=None, num_latent_conditional_frames=2,
                     noise_seed=0, num_steps=36):
    """
    Run a full `num_steps`-step Euler denoising trajectory (no grad) with each
    of condition and target_condition on the unmodified latent.  Cross-attention
    scores are accumulated as an online mean across all timesteps — at most two
    copies of the score tensors are live at once — then a per-spatial cosine-
    similarity mask is derived from the mean scores.

    Returns a list of (B, T, S_per_frame, H) masks — one per hooked layer —
    where high values mark spatial positions whose attention pattern differs
    most between the two prompts over the full generation trajectory.
    """
    compute_dtype = next(model.net.parameters()).dtype
    B, C, T_lat, H_lat, W_lat = latent.shape

    def _build_live(cond):
        return cond.set_video_condition(
            gt_frames=latent.to(compute_dtype),
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_latent_conditional_frames,
        )

    condition_live = _build_live(condition)
    target_live    = _build_live(target_condition)

    gen   = torch.Generator(device=latent.device).manual_seed(noise_seed)
    noise = torch.randn(latent.shape, generator=gen,
                        dtype=latent.dtype, device=latent.device)

    def _collect_mean(cond_live, label):
        """Denoising trajectory → per-layer mean score tensor."""
        # Reset scheduler state before each trajectory so the multi-step solver
        # starts fresh (its internal step counter and model-output history are
        # consumed after one full run and must be re-initialised).
        model.sample_scheduler.set_timesteps(num_steps, device=latent.device, shift=5.0)
        timesteps_schedule = model.sample_scheduler.timesteps
        seed_g = torch.Generator(device=latent.device).manual_seed(noise_seed)

        x_t      = noise.clone().float()
        step_buf: list = []
        accum    = None

        restore = install_crossattn_hooks(model.net, T_tok, step_buf, layer_start, layer_end)
        for step_i, t in enumerate(timesteps_schedule):
            xt        = x_t.to(**model.tensor_kwargs)
            # Scheduler gives scalar t in [0, 1000]; expand to (B, 1)
            timesteps = t.reshape(1, 1).expand(B, 1).to(device=latent.device, dtype=latent.dtype)

            del step_buf[:]   # free previous step's tensors before the next forward
            with torch.no_grad():
                v = model.denoise(noise=noise, xt_B_C_T_H_W=xt,
                                  timesteps_B_T=timesteps, condition=cond_live)

            if accum is None:
                accum = [s.detach().clone() for s in step_buf]
            else:
                for i, s in enumerate(step_buf):
                    accum[i].add_(s.detach())

            x_t = model.sample_scheduler.step(
                v.float().unsqueeze(0), t, x_t.unsqueeze(0),
                return_dict=False, generator=seed_g,
            )[0].squeeze(0)
            print(f"  [sim_mask:{label}] step {step_i+1}/{num_steps}", end="\r")

        restore()
        print()
        return [a.detach() / num_steps for a in accum]

    print(f"  [sim_mask] running {num_steps}-step denoising for true condition...")
    true_means = _collect_mean(condition_live, "true")
    print(f"  [sim_mask] running {num_steps}-step denoising for target condition...")
    tgt_means  = _collect_mean(target_live, "target")

    # cosine similarity over text-token dim (N) → (B, T, S, H)
    # 1 - sim: high where the two prompts differ most
    masks = [
        (1.0 - F.cosine_similarity(s_true, s_tgt, dim=3)).detach()
        for s_true, s_tgt in zip(true_means, tgt_means)
    ]
    #Normalize mask per layer
    for i in range(len(masks)):
        mi = masks[i].min()
        mx = masks[i].max()
        masks[i] = (masks[i] - mi) / (mx - mi)
    print(f"  [sim_mask] {len(masks)} layer masks, shape {masks[0].shape}, "
          f"range [{masks[0].min():.3f}, {masks[0].max():.3f}]")
    return masks


# ---------------------------------------------------------------------------
# Sim-mask dimensionality analysis (debugging)
# ---------------------------------------------------------------------------

def analyze_sim_masks(masks, spatial_grid=27, save_dir=None):
    """
    Print per-dimension statistics for a list of sim masks and optionally save
    spatial heatmaps.  Call this immediately after compute_sim_mask.

    masks      : list of (B, T, S, H) tensors — one per hooked layer
    spatial_grid : sqrt(S), i.e. 27 for 432×432 with patch size 16
    save_dir   : if set, saves one PNG heatmap per layer (spatially averaged
                 over T and H) using matplotlib

    Reports
    -------
    Layer      : mean mask value per layer — which DiT blocks are most discriminative
    Temporal   : mean over (B, S, H) → (T,) — which frames differ most
    Head       : mean over (B, T, S) → (H,) — which attention heads are most discriminative
    Spatial    : mean over (B, T, H) → (S,) reshaped to (grid, grid) — peak patches
    """
    import numpy as np

    L = len(masks)
    B, T, S, H = masks[0].shape

    print(f"\n{'='*60}")
    print(f"Sim-mask analysis  |  {L} layers, shape (B={B}, T={T}, S={S}, H={H})")
    print(f"{'='*60}")

    # -- Layer --
    layer_means = [m.mean().item() for m in masks]
    layer_vars  = [m.var().item()  for m in masks]
    top_layers  = sorted(range(L), key=lambda i: layer_means[i], reverse=True)[:5]
    print(f"\n[Layer]  mean difference per layer (top 5 most discriminative):")
    for i in top_layers:
        print(f"  layer {i:3d}  mean={layer_means[i]:.4f}  var={layer_vars[i]:.4f}")

    # -- Aggregate mask across all layers for the remaining analyses --
    agg = torch.stack(masks).mean(dim=0)   # (B, T, S, H)
    agg = agg[0]                           # drop batch → (T, S, H)

    # -- Temporal --
    temporal = agg.mean(dim=(1, 2))        # (T,)
    print(f"\n[Temporal]  mean difference per frame:")
    for t_i, val in enumerate(temporal.tolist()):
        bar = "█" * int(val / temporal.max().item() * 20)
        print(f"  frame {t_i:2d}  {val:.4f}  {bar}")

    # -- Head --
    head = agg.mean(dim=(0, 1))            # (H,)
    top_heads = head.argsort(descending=True)
    print(f"\n[Head]  mean difference per attention head (ranked):")
    for rank, h_i in enumerate(top_heads.tolist()):
        bar = "█" * int(head[h_i].item() / head.max().item() * 20)
        print(f"  rank {rank:2d}  head {h_i:2d}  {head[h_i].item():.4f}  {bar}")

    # -- Spatial --
    spatial = agg.mean(dim=(0, 2))         # (S,) — averaged over T and H
    sp_grid = spatial.reshape(spatial_grid, spatial_grid)
    sp_np   = sp_grid.cpu().float().numpy()
    top_k   = int(S * 0.05)               # top 5% of patches
    flat    = spatial.cpu()
    top_idx = flat.argsort(descending=True)[:top_k]
    top_coords = [(int(i) // spatial_grid, int(i) % spatial_grid) for i in top_idx]
    print(f"\n[Spatial]  top {top_k} patches (row, col) with highest avg difference:")
    for row, col in top_coords[:10]:
        print(f"  patch ({row:2d}, {col:2d})  val={sp_grid[row, col].item():.4f}")
    print(f"  spatial range  [{spatial.min().item():.4f}, {spatial.max().item():.4f}]")

    # -- Variance analysis: which dim has the most spread? --
    print(f"\n[Variance across dims — higher = more information in that axis]")
    print(f"  layer    var: {np.var(layer_means):.6f}")
    print(f"  temporal var: {temporal.var().item():.6f}")
    print(f"  head     var: {head.var().item():.6f}")
    print(f"  spatial  var: {spatial.var().item():.6f}")

    if save_dir is not None:
        try:
            import matplotlib.pyplot as plt
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            for l_i, m in enumerate(masks):
                # average over B, T, H → spatial heatmap
                sp = m[0].mean(dim=(0, 2)).reshape(spatial_grid, spatial_grid)
                fig, ax = plt.subplots(figsize=(5, 5))
                im = ax.imshow(sp.cpu().float().numpy(), cmap="hot", interpolation="nearest")
                ax.set_title(f"Layer {l_i}  spatial diff")
                plt.colorbar(im, ax=ax)
                fig.savefig(save_dir / f"sim_mask_layer_{l_i:03d}.png", dpi=80,
                            bbox_inches="tight")
                plt.close(fig)
            # also save the layer-aggregated map
            agg_sp = sp_grid
            fig, ax = plt.subplots(figsize=(5, 5))
            im = ax.imshow(sp_np, cmap="hot", interpolation="nearest")
            ax.set_title("All layers aggregated spatial diff")
            plt.colorbar(im, ax=ax)
            fig.savefig(save_dir / "sim_mask_agg.png", dpi=80, bbox_inches="tight")
            plt.close(fig)
            print(f"\n  Saved heatmaps to {save_dir}")
        except ImportError:
            print("  [warn] matplotlib not available — skipping heatmap save")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Multi-step denoising forward passes → L_cross scalar
# ---------------------------------------------------------------------------

def compute_crossattn_loss(model, x_padded, condition, T_tok, layer_start=0, layer_end=None,
                           skip_latent=False, max_att=False, denoise_steps=1,
                           uncondition=None, guidance_scale=7.0,
                           target_condition=None, noise_seed=None,
                           num_latent_conditional_frames=2, sim_masks=None,
                           loss_type="cosine", dit_device=None):
    """
    Run `denoise_steps` Euler denoising steps and return the cross-attention loss.

    loss_type="cosine"  (default)
        Untargeted: L = mean_l ||A^(l)||_F^2 / (B*H)
        Targeted  : L = 1 - cosine_similarity(flat(A_base * M), flat(A_target * M))
                    where M is the optional per-layer spatial mask.

    loss_type="l2"
        Untargeted: L = mean_l ||A^(l)||_F^2 / (B*H)   (identical to cosine untargeted)
        Targeted  : L = mean_l ||A_base^(l) - A_target^(l)||_F^2 / (B*H)
                    Minimises the per-block Frobenius distance between the adv and
                    target attention distributions.  More memory-efficient than cosine
                    because the difference is reduced to a scalar per block without
                    materialising large flat vectors.
    """
    compute_dtype = next(model.net.parameters()).dtype

    # Ensure the default CUDA device is dit_device for the entire loss computation.
    # TE modules (e.g. RMSNorm) use torch.cuda.current_device() to allocate internal
    # buffers; without this the buffers land on cuda:0 while model weights are on
    # dit_device, causing a device mismatch in the first block forward pass.
    if dit_device is not None:
        torch.cuda.set_device(dit_device)

    if skip_latent:
        latent = x_padded
    else:
        latent = model.tokenizer.encode(x_padded.to(compute_dtype)).contiguous().float()

    # Differentiable bridge from VAE device to DiT device for split-GPU gradient flow
    if dit_device is not None and latent.device != torch.device(dit_device):
        latent = latent.to(dit_device)

    B, C, T_lat, H_lat, W_lat = latent.shape

    def _build_live(cond):
        # Preserve all fields set by edit_for_inference (including _cond_mask)
        # by copying the full dict and patching only gt_frames, matching the
        # working self-attention approach.
        cond_dict = cond.to_dict(skip_underscore=False)
        cond_dict['gt_frames'] = latent.to(compute_dtype)
        return type(cond)(**cond_dict)

    condition_live    = _build_live(condition)
    uncondition_live  = _build_live(uncondition)  if uncondition  is not None else None
    target_cond_live  = _build_live(target_condition) if target_condition is not None else None

    n_blocks = len(model.net.blocks)

    loss_terms = []
    restore    = install_crossattn_hooks(model.net, T_tok, loss_terms, layer_start, layer_end)

    # Cut the gradient graph after layer_end so blocks beyond it don't accumulate
    # activations in the backward pass.
    detach_hook = None
    effective_end = layer_end if layer_end is not None else n_blocks - 1
    if effective_end < n_blocks - 1:
        def _detach(_m, _inp, output):
            if isinstance(output, tuple):
                return (output[0].detach(),) + output[1:]
            return output.detach()
        detach_hook = model.net.blocks[effective_end].register_forward_hook(_detach)

    if noise_seed is not None:
        gen   = torch.Generator(device=latent.device).manual_seed(noise_seed)
        noise = torch.randn(latent.shape, generator=gen,
                            dtype=latent.dtype, device=latent.device)
    else:
        noise = torch.randn_like(latent)

    target_terms = []

    if denoise_steps == 1:
        t_norm    = torch.rand(1).item()                        # ∈ [0, 1] for interpolation
        xt        = ((1 - t_norm) * latent + t_norm * noise).to(**model.tensor_kwargs)
        # Model expects timesteps in [0, 1000]; scale accordingly
        timesteps = torch.full((B, T_lat), t_norm * 1000.0, device=latent.device, dtype=latent.dtype)

        if target_cond_live is not None:
            n_before = len(loss_terms)
            with torch.no_grad():
                model.denoise(noise=noise, xt_B_C_T_H_W=xt,
                              timesteps_B_T=timesteps, condition=target_cond_live)
            target_terms = list(loss_terms[n_before:])
            del loss_terms[n_before:]

        model.denoise(noise=noise, xt_B_C_T_H_W=xt, timesteps_B_T=timesteps,
                      condition=condition_live)
    else:
        # Use the model's scheduler for properly-scaled timesteps in [0, 1000].
        # We only need its t values here; the Euler update stays as plain arithmetic
        # so that gradients can flow back through the entire x_t chain into loss_terms.
        model.sample_scheduler.set_timesteps(denoise_steps, device=latent.device, shift=5.0)
        sched_ts = model.sample_scheduler.timesteps   # shape (denoise_steps,) in [0, 1000]
        x_t = noise.float()

        for step_i in range(denoise_steps):
            t_model = sched_ts[step_i]
            t_next  = sched_ts[step_i + 1] if step_i + 1 < denoise_steps else sched_ts.new_tensor(0.0)

            xt        = x_t.to(**model.tensor_kwargs)
            # (B, 1) scalar timestep in [0, 1000]
            timesteps = t_model.reshape(1, 1).expand(B, 1).to(device=latent.device, dtype=latent.dtype)

            # Target pass at the same xt — no grad, hooks route into target_terms
            if target_cond_live is not None:
                n_before = len(loss_terms)
                with torch.no_grad():
                    model.denoise(noise=noise, xt_B_C_T_H_W=xt,
                                  timesteps_B_T=timesteps, condition=target_cond_live)
                target_terms.extend(loss_terms[n_before:])
                del loss_terms[n_before:]

            # CFG uncond pass — velocity only, discard hook terms
            if uncondition_live is not None:
                n_before = len(loss_terms)
                with torch.no_grad():
                    uncond_v = model.denoise(noise=noise, xt_B_C_T_H_W=xt,
                                             timesteps_B_T=timesteps,
                                             condition=uncondition_live)
                del loss_terms[n_before:]

            # Base cond pass — hooks collect gradient-connected score tensors
            cond_v = model.denoise(noise=noise, xt_B_C_T_H_W=xt,
                                   timesteps_B_T=timesteps, condition=condition_live)

            v_pred = (cond_v + guidance_scale * (cond_v - uncond_v)
                      if uncondition_live is not None else cond_v)
            # Differentiable Euler step: dt in normalised [0,1] space = (t_curr - t_next) / 1000
            dt_norm = (t_model - t_next).item() / 1000.0
            x_t = x_t - dt_norm * v_pred.float()

    def _cleanup():
        restore()
        if detach_hook is not None:
            detach_hook.remove()

    if target_cond_live is not None:
        if loss_type == "l2":
            # Per-block ||A_base - A_target||_F^2 / (B*H).  The difference tensor
            # is reduced to a scalar immediately so only one block's worth of
            # activations is live at a time — much lower peak VRAM than cosine.
            if sim_masks is not None:
                loss = torch.stack([
                    ((s - t.detach()) * m.unsqueeze(3)).pow(2).sum() / (s.shape[0] * s.shape[4])
                    for s, t, m in zip(loss_terms, target_terms, sim_masks)
                ]).mean()
            else:
                loss = torch.stack([
                    (s - t.detach()).pow(2).sum() / (s.shape[0] * s.shape[4])
                    for s, t in zip(loss_terms, target_terms)
                ]).mean()
        elif loss_type == "kl":
                loss = torch.stack([
                    F.kl_div(
                        s.log(),           # predicted log-probs (adv image, true prompt)
                        t.detach(),        # target probs (true image, target prompt)
                        reduction='sum'
                    ) / (s.shape[0] * s.shape[4])
                    for s, t in zip(loss_terms, target_terms)
                ]).mean()
        else:
            # cosine: build tgt_vec first, free target_terms, then build adv_vec.
            # Each score tensor is (B, T, S_per_frame, N, H) fp32 — freeing
            # target_terms before materialising adv_vec saves ~one list worth of VRAM.
            if sim_masks is not None:
                tgt_vec = torch.cat([(s * m.unsqueeze(3)).detach().flatten()
                                      for s, m in zip(target_terms, sim_masks)])
                del target_terms
                adv_vec = torch.cat([(s * m.unsqueeze(3)).flatten()
                                      for s, m in zip(loss_terms, sim_masks)])
            else:
                tgt_vec = torch.cat([s.detach().flatten() for s in target_terms])
                del target_terms
                adv_vec = torch.cat([s.flatten() for s in loss_terms])
            del loss_terms
            loss = 1.0 - F.cosine_similarity(adv_vec.unsqueeze(0), tgt_vec.unsqueeze(0)).squeeze()
    else:
        loss = torch.stack(
            [s.pow(2).sum() / (s.shape[0] * s.shape[4]) for s in loss_terms]
        ).mean() * (-1 if max_att else 1)

    return loss, _cleanup


# ---------------------------------------------------------------------------
# Per-video attack
# ---------------------------------------------------------------------------

def attack_single_video(
    inference, model, video_path, prompt,
    H, W, frames_to_extract, required_pixel_frames,
    T_tok, layer_start, layer_end, compute_dtype, args,
    vae_device="cuda", dit_device="cuda", save_time=None
):
    """
    Run the full PGD attack for a single (video_path, prompt) pair.

    Returns
    -------
    x_adv : (1, C, T, H, W) tensor on CPU
    """
    if save_time is None:
        save_time = int(time.time())

    # ------------------------------------------------------------------
    # 2. Load video
    # ------------------------------------------------------------------
    if inference.offload_text_encoder:
        if model.text_encoder is not None:
            if hasattr(model.text_encoder, "model") and model.text_encoder.model is not None:
                model.text_encoder.model = model.text_encoder.model.to(vae_device).eval()
        if _t5_mod.cosmos_encoder is not None:
            _t5_mod.cosmos_encoder.text_encoder = \
                _t5_mod.cosmos_encoder.text_encoder.to(vae_device).eval()

    print(f"\nLoading video : {video_path}")
    if args.extend:
        full_video_uint8 = load_full_video(str(video_path), [H, W])
        raw_full = normalize_video(full_video_uint8, device=vae_device)
        args._extend_full_video = raw_full.cpu()  # stored on CPU; forwarded to eval for saving
        print(f"Full video shape         : {raw_full.shape}")
        raw_padded = pad_video(raw_full, frames_to_extract, required_pixel_frames)
    else:
        video_uint8 = load_and_preprocess_video(
            str(video_path), [H, W], frames_to_extract
        )
        raw_cond = normalize_video(video_uint8, device=vae_device)
        args._extend_full_video = None
        print(f"Conditioning frames shape: {raw_cond.shape}")
        raw_padded = pad_video(raw_cond, frames_to_extract, required_pixel_frames)
    print(f"Padded video shape       : {raw_padded.shape}")

    # ------------------------------------------------------------------
    # 3. Build T5 condition(s)
    #    Text encoder is on GPU here; build all data_batch objects that
    #    need T5 embeddings before the offload step below.
    # ------------------------------------------------------------------
    video_bf16 = raw_padded.to(dtype=compute_dtype)

    negative_prompt = args.negative_prompt or _DEFAULT_NEGATIVE_PROMPT
    data_batch = inference._get_data_batch_input(
        video=video_bf16,
        prompt=prompt,
        num_conditional_frames=args.num_latent_conditional_frames,
        negative_prompt=negative_prompt,
        use_neg_prompt=True,
    )
    data_batch["video"]             = video_bf16
    data_batch[IS_PREPROCESSED_KEY] = True

    # Build target data_batch now (text encoder still on GPU).
    target_data_batch = None
    if getattr(args, "target_prompt", None) is not None:
        target_data_batch = inference._get_data_batch_input(
            video=video_bf16,
            prompt=args.target_prompt,
            num_conditional_frames=args.num_latent_conditional_frames,
            negative_prompt=negative_prompt,
            use_neg_prompt=True,
        )
        target_data_batch["video"]             = video_bf16
        target_data_batch[IS_PREPROCESSED_KEY] = True

    # ------------------------------------------------------------------
    # 4. Offloading sequence
    # ------------------------------------------------------------------
    if inference.offload_text_encoder:
        if model.text_encoder is not None:
            if hasattr(model.text_encoder, "model") and model.text_encoder.model is not None:
                model.text_encoder.model = model.text_encoder.model.to("cpu")
        if _t5_mod.cosmos_encoder is not None:
            _t5_mod.cosmos_encoder.text_encoder = \
                _t5_mod.cosmos_encoder.text_encoder.to("cpu")
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
    elif vae_device != dit_device:
        # split_gpus without full offloading: explicitly place DiT on dit_device
        print(f"split_gpus: moving DiT → {dit_device}, VAE stays on {vae_device}")
        model.net = model.net.to(dit_device)
        if hasattr(model, "conditioner") and model.conditioner is not None:
            model.conditioner = model.conditioner.to(dit_device)
        torch.cuda.empty_cache()

    # model.tensor_kwargs["device"] is hardcoded to "cuda" at model init; patch it
    # so that model.denoise() sends tensors to the correct DiT device.
    if vae_device != dit_device:
        model.tensor_kwargs["device"] = dit_device
        if hasattr(model, "tensor_kwargs_fp32"):
            model.tensor_kwargs_fp32["device"] = dit_device
        # Set the default CUDA device to dit_device.  Transformer Engine modules
        # (e.g. RMSNorm used in t_embedding_norm) call torch.cuda.current_device()
        # internally to create temporary tensors.  Without this, those tensors land
        # on cuda:0 even though model.net lives on dit_device, causing device
        # mismatch errors in the first DiT block forward pass.
        torch.cuda.set_device(dit_device)

    # ------------------------------------------------------------------
    # 5. Freeze model parameters / enable VAE gradient
    # ------------------------------------------------------------------
    for p in model.net.parameters():
        p.requires_grad_(False)
    for p in model.tokenizer.model.model.parameters():
        p.requires_grad_(False)
    model.tokenizer.enable_grad = True

    # ------------------------------------------------------------------
    # 6. Build condition and uncondition objects (VAE encode + cond mask)
    # ------------------------------------------------------------------
    print("Building condition...")
    with torch.no_grad():
        _, _, condition = model.get_data_and_condition(data_batch)
        _, uncondition = model.conditioner.get_condition_with_negative_prompt(data_batch)
    condition = condition.edit_for_inference(
        is_cfg_conditional=True,
        num_conditional_frames=args.num_latent_conditional_frames,
    )
    # uncondition.gt_frames is None because the video embedder is dropped out
    # in the unconditional branch.  edit_for_inference needs gt_frames to build
    # the conditioning mask, so copy it from condition before calling it.
    uncond_dict = uncondition.to_dict(skip_underscore=False)
    uncond_dict['gt_frames'] = condition.gt_frames
    uncondition = type(uncondition)(**uncond_dict)
    uncondition = uncondition.edit_for_inference(
        is_cfg_conditional=False,
        num_conditional_frames=args.num_latent_conditional_frames,
    )

    # ------------------------------------------------------------------
    # 6b. Build target condition template (only when --target_prompt is given)
    #     The template is rebuilt with the current perturbed gt_frames inside
    #     compute_crossattn_loss at every PGD step, so target and base always
    #     share identical conditioning.
    # ------------------------------------------------------------------
    target_condition_template = None
    if target_data_batch is not None:
        print(f"Building target condition for: \"{args.target_prompt}\"")
        # Derive from base condition so fps, gt_frames, masks are all identical.
        # TextAttr.forward() is a pass-through, so crossattn_emb == t5_text_embeddings.
        # Replace only crossattn_emb with the target prompt's text embeddings;
        # do NOT call get_condition_with_negative_prompt (that re-runs Qwen with
        # a fresh random fps, producing a different crossattn_emb even for same prompt).
        tgt_dict = condition.to_dict(skip_underscore=False)
        tgt_dict['crossattn_emb'] = target_data_batch['t5_text_embeddings']
        target_condition_template = type(condition)(**tgt_dict)
        # edit_for_inference was already applied to condition (which we copied)

    # When split_gpus is active, _get_data_batch_input moves all float tensors
    # to cuda:0 via .cuda(), but the DiT lives on dit_device.  Explicitly move
    # every tensor attribute of each condition object to dit_device so that
    # model.denoise() (which calls condition.to_dict()) sees a consistent device.
    if vae_device != dit_device:
        _move_condition_to_device(condition, dit_device)
        _move_condition_to_device(uncondition, dit_device)
        if target_condition_template is not None:
            _move_condition_to_device(target_condition_template, dit_device)

    # ------------------------------------------------------------------
    # 7. PGD optimisation
    # ------------------------------------------------------------------
    # Release any fragmented allocator cache before the gradient-connected
    # forward passes start.  The condition-building and any no-grad encodes
    # above may have left cached-but-freed blocks that prevent the large
    # contiguous allocations the VAE encoder needs during backprop.
    torch.cuda.empty_cache()

    print(f"\nStarting PGD optimisation "
          f"({'targeted' if target_condition_template is not None else 'untargeted'})...")

    # Precompute spatial importance mask on the clean input (one forward pass each,
    # no grad, no denoising trajectory — no memory accumulation).
    sim_masks = None
    if target_condition_template is not None:
        with torch.no_grad():
            latent = model.tokenizer.encode(raw_padded.to(compute_dtype)).contiguous().float()
            if vae_device != dit_device:
                latent = latent.to(dit_device)
            print(f"Latent Shape {latent.shape}")
        # print("Computing spatial sim mask...")
        if args.mask:
            sim_masks = compute_sim_mask(
                model, latent, condition, target_condition_template, T_tok,
                layer_start=layer_start,
                layer_end=layer_end,
                num_latent_conditional_frames=args.num_latent_conditional_frames,
                noise_seed=0
            )                                                                  
            analyze_sim_masks(sim_masks, spatial_grid=27, save_dir= "sim_analysis")
    

    loss_fn = lambda x, y: compute_crossattn_loss(
        model, x, condition, T_tok,
        layer_start=layer_start,
        layer_end=layer_end,
        skip_latent=args.skip_latent,
        max_att=args.max_att,
        denoise_steps=args.denoise_steps,
        uncondition=uncondition if args.denoise_steps > 1 else None,
        guidance_scale=args.guidance_scale,
        target_condition=target_condition_template,
        noise_seed=0 if target_condition_template is not None else None,
        num_latent_conditional_frames=args.num_latent_conditional_frames,
        sim_masks=sim_masks,
        loss_type=args.loss_type,
        dit_device=dit_device if vae_device != dit_device else None,
    )
    if getattr(args, "lf_attack", False):
        if args.skip_latent:
            raise ValueError("--lf_attack requires pixel-space optimisation and is incompatible with --skip_latent")
        x_adv = lab_freq_pgd(
            raw_padded, 0, lambda x: x, loss_fn,
            args.steps, args.alpha,
            frames_to_extract,
            freq_cutoff=args.freq_cutoff,
            lab_budget_L=args.lab_budget_L,
            lab_budget_ab=args.lab_budget_ab,
            freq_pixel_eps=args.freq_pixel_eps,
        )
    elif args.skip_latent:
        with torch.no_grad():
            latent = model.tokenizer.encode(raw_padded.to(compute_dtype)).contiguous().float()
            # latent = torch.empty_like(latent).uniform_(-3, 3)
        # PGD optimises delta on vae_device; compute_crossattn_loss bridges to dit_device
        x_adv = pgd(latent, 0, lambda x: x, loss_fn, args.steps, args.alpha, args.eps, args.num_latent_conditional_frames)
    else:
        x_adv = pgd(raw_padded, 0, lambda x: x, loss_fn, args.steps, args.alpha, args.eps, frames_to_extract,
                    pixel_min=-1.0, pixel_max=1.0)

    def to_uint8_frames(t):
        """(B, C, T, H, W) float32 [-1,1] → (T, H, W, C) uint8"""
        t = t[0].cpu().float()
        t = ((t.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
        return t.permute(1, 2, 3, 0)

    out_dir = WM_ROOT / "attack" / "outputs" / f"crossattn_batch_{save_time}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = str(video_path).split("/")[-1].split(".")[0] + "_adv"
    torch.save(x_adv.cpu(), out_dir / f"{base_name}_adv.pt")

    if not args.skip_latent:
        torchvision.io.write_video(
            str(out_dir / "x_adv.mp4"), to_uint8_frames(x_adv), fps=16)

    print(f"\nSaved to: {out_dir}")
    print(f"  x_adv.pt / x_adv.mp4 : perturbed video")
    return x_adv.cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    # Input: single video or batch directory (one must be provided)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path", default=None,
                             help="Path to a single input video (.mp4)")
    input_group.add_argument("--batch_path", default=None,
                             help="Path to a batch directory containing videos/ and "
                                  "metas/ subfolders.  Each <name>.mp4 in videos/ must "
                                  "have a matching <name>.txt prompt in metas/.")
    parser.add_argument("--mask", action = "store_true")
    parser.add_argument("--prompt", default=None,
                        help="Text prompt (required with --video_path)")
    parser.add_argument("--target_prompt", default=None,
                        help="Target prompt for the targeted cross-attention attack. "
                             "When provided, the attack steers the cross-attention "
                             "pattern of (x_adv, --prompt) toward the reference "
                             "pattern of (x_orig, --target_prompt), encouraging the "
                             "model to generate video as if given --target_prompt. "
                             "Omit for the untargeted (attention-suppression) attack.")
    parser.add_argument("--batch_targets", action="store_true",
                        help="Read per-video target prompts from a metas-target/ "
                             "subfolder alongside --batch_path.  Each <name>.txt in "
                             "metas-target/ must match a <name>.mp4 in videos/.  "
                             "Overrides --target_prompt for batch runs.")
    parser.add_argument("--attack_num", type=int, default=-1,
                        help="Number of videos to sample from --batch_path "
                             "(-1 = use all, default: -1)")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--ckpt_path",       type=str, default=None)
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--resolution",      default="432,432",
                        help="'H,W' (default: 432,432)")
    parser.add_argument("--num_latent_video_frames",       type=int, default=24,
                        help="Number of latent frames in the attack's forward pass. "
                             "Reduce below the model's state_t to shorten the token "
                             "sequence and save VRAM. Default: 24.")
    parser.add_argument("--num_latent_conditional_frames", type=int, default=2,
                        help="Number of conditioning latent frames. Default: 2.")
    parser.add_argument("--steps",           type=int,   default=100,
                        help="PGD iterations. Default: 100.")
    parser.add_argument("--alpha",           type=float, default=4/255,
                        help="PGD step size (default: 2/255 ≈ 0.00784)")
    parser.add_argument("--eps",             type=float, default=16/255,
                        help="L∞ perturbation budget (default: 16/255 ≈ 0.0628)")
    parser.add_argument("--attack_layers_start", type=int, default=0,
                        help="First DiT block index to include in the cross-attention "
                             "loss (inclusive). Default: 0.")
    parser.add_argument("--attack_layers_end", type=int, default=None,
                        help="Last DiT block index to include in the cross-attention "
                             "loss (inclusive). Gradient graph is cut after this block. "
                             "Default: all remaining blocks.")
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
                             "final frames as conditioning context.  The saved output video "
                             "includes all original frames followed by the generated extension.")
    parser.add_argument("--skip_latent", action="store_true",
                        help="Optimise in latent space instead of pixel space.")
    parser.add_argument("--max_att", action="store_true",
                        help="Maximise cross-attention instead of minimising it.")
    parser.add_argument("--loss_type", choices=["cosine", "l2", "kl"], default="cosine",
                        help="Loss for the targeted attack.  "
                             "'cosine' (default): minimise 1 - cosine_similarity between "
                             "flattened adv and target attention vectors.  "
                             "'l2': minimise mean_l ||A_base^(l) - A_target^(l)||_F^2 / (B*H); "
                             "more memory-efficient and uses the same Frobenius-norm form as "
                             "the self-attention freeze attack.  "
                             "Has no effect in the untargeted case (both use Frobenius norm).")
    parser.add_argument("--denoise_steps", type=int, default=1,
                        help="Number of consecutive Euler denoising steps per PGD "
                             "gradient evaluation.  1 (default) = single forward pass "
                             "at a random t.  K>1 = K-step Euler trajectory from t=1 "
                             "to t=1/K; gradient flows through the full chain.")
    parser.add_argument("--guidance_scale", type=float, default=7.0,
                        help="CFG guidance scale for the Euler denoising trajectory. "
                             "Only applied when --denoise_steps > 1. "
                             "Matches the default used during normal inference. "
                             "Default: 7.0.")
    parser.add_argument("--lf_attack", action="store_true",
                        help="Use the LAB+Frequency space PGD attack instead of standard "
                             "pixel-space PGD.  Optimises a low-frequency complex delta in "
                             "the frequency domain of the LAB image.  Incompatible with "
                             "--skip_latent.")
    parser.add_argument("--freq_cutoff", type=float, default=0.1,
                        help="Fraction of low-frequency coefficients to perturb in the "
                             "LAB+Frequency attack.  Default: 0.1.")
    parser.add_argument("--lab_budget_L", type=float, default=5.0,
                        help="Maximum L* channel perturbation (LAB units) for --lf_attack.  "
                             "Default: 5.0.")
    parser.add_argument("--lab_budget_ab", type=float, default=20.0,
                        help="Maximum a*, b* channel perturbation (LAB units) for --lf_attack.  "
                             "Default: 20.0.")
    parser.add_argument("--freq_pixel_eps", type=float, default=0.1,
                        help="Maximum per-pixel perturbation from the frequency delta in [0,1] "
                             "RGB space for --lf_attack.  The frequency delta is reconstructed "
                             "to pixel space via irfft2 and clamped to this bound before being "
                             "added to the LAB-perturbed image.  Default: 0.1.")
    parser.add_argument(
        "--split_gpus", action="store_true",
        help="Place the VAE encoder on cuda:0 and the DiT on cuda:1.  Gradient flows "
             "across devices via differentiable .to() calls so a single backward pass "
             "spans both GPUs, halving per-device peak VRAM.  Requires two CUDA devices.",
    )
    args = parser.parse_args()

    if args.video_path is not None and args.prompt is None:
        parser.error("--prompt is required when using --video_path")
    if args.ckpt_path is None:
        args.ckpt_path = ckpt_path
    if args.experiment_name is None:
        args.experiment_name = experiment_name

    device = "cuda"
    if args.split_gpus:
        assert torch.cuda.device_count() >= 2, "--split_gpus requires at least 2 CUDA devices"
        vae_device = "cuda:0"
        dit_device  = "cuda:1"
        print(f"split_gpus: VAE on {vae_device}, DiT on {dit_device}")
    else:
        vae_device = device
        dit_device  = device
    H, W   = [int(x) for x in args.resolution.split(",")]

    # ------------------------------------------------------------------
    # 1. Load model
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
    model    = inference.model
    state_t  = model.config.state_t
    n_blocks = len(model.net.blocks)

    T_lat_attack          = min(args.num_latent_video_frames, state_t)
    required_pixel_frames = (T_lat_attack - 1) * 4 + 1
    frames_to_extract     = (args.num_latent_conditional_frames - 1) * 4 + 1
    T_tok                 = T_lat_attack

    layer_start   = args.attack_layers_start
    layer_end     = args.attack_layers_end    # None = last block
    compute_dtype = torch.bfloat16

    effective_end = layer_end if layer_end is not None else n_blocks - 1

    print(f"Model         : {n_blocks} DiT blocks, state_t={state_t}")
    print(f"Attack T_lat  : {T_lat_attack}  (pixel frames: {required_pixel_frames})")
    print(f"T_tok         : {T_tok}  (frame × text-token cross-attn matrix: {T_tok}×N)")
    print(f"Loss blocks   : [{layer_start}, {effective_end}]  "
          f"(gradient cut after block {effective_end})")
    print(f"PGD           : steps={args.steps}  alpha={args.alpha:.5f}  eps={args.eps:.4f}")

    # ------------------------------------------------------------------
    # Collect (video_path, prompt, target_prompt) triples
    # ------------------------------------------------------------------
    if args.batch_targets and args.batch_path is None:
        parser.error("--batch_targets requires --batch_path")

    if args.batch_path is not None:
        batch_path  = Path(args.batch_path)
        video_files = sorted((batch_path / "videos").glob("*.mp4"))
        if args.attack_num > 0:
            video_files = random.sample(video_files, min(args.attack_num, len(video_files)))
        pairs = []
        for vf in video_files:
            meta_path = batch_path / "metas" / (vf.stem + ".txt")
            if args.batch_targets:
                tgt_path = batch_path / "metas-target" / (vf.stem + ".txt")
                target_prompt = tgt_path.read_text().strip()
            else:
                target_prompt = args.target_prompt
            pairs.append((vf, meta_path.read_text().strip(), target_prompt))
        print(f"\nBatch mode: {len(pairs)} video(s) from {batch_path}"
              + (" (per-video target prompts from metas-target/)" if args.batch_targets else ""))
    else:
        pairs = [(Path(args.video_path), args.prompt, args.target_prompt)]

    # ------------------------------------------------------------------
    # Attack each video
    # ------------------------------------------------------------------
    all_x_adv = []
    save_time = int(time.time())
    out_dir = WM_ROOT / "attack" / "outputs" / f"crossattn_batch_{save_time}"
    out_dir.mkdir(parents=True, exist_ok=True)

    attack_config = {
        **vars(args),
        "H": H,
        "W": W,
        "T_lat_attack": T_lat_attack,
        "required_pixel_frames": required_pixel_frames,
        "frames_to_extract": frames_to_extract,
        "T_tok": T_tok,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "n_blocks": n_blocks,
        "save_time": save_time,
        "out_dir": str(out_dir),
        "batch_pairs": [(str(vp), pr, tp) for vp, pr, tp in pairs],
    }
    with open(out_dir / "attack_config.json", "w") as _f:
        json.dump(attack_config, _f, indent=2)
    print(f"Attack config saved to: {out_dir / 'attack_config.json'}")

    for i, (video_path, prompt, target_prompt) in enumerate(pairs):
        print(f"\n{'='*60}")
        print(f"Video {i+1}/{len(pairs)}: {video_path.name}")
        print(f"Prompt: {prompt}")
        if target_prompt is not None:
            print(f"Target: {target_prompt}")
        print(f"{'='*60}")
        args.target_prompt = target_prompt
        x_adv = attack_single_video(
            inference, model, video_path, prompt,
            H, W, frames_to_extract, required_pixel_frames,
            T_tok, layer_start, layer_end, compute_dtype, args,
            vae_device=vae_device, dit_device=dit_device, save_time=save_time,
        )
        all_x_adv.append(x_adv)
        base_name = str(video_path).split("/")[-1].split(".")[0] + "_adv.pt"
        print("\nEvaluating Diffusion")
        args.adv_path = out_dir
        args.prompt   = prompt

        # ------------------------------------------------------------------
        # Restore single-GPU state before evaluation.
        # During the attack, torch.cuda.set_device(dit_device) was called and
        # model.net was placed on dit_device.  eval_wm.py uses
        # torch.cuda.current_device() to decide where to put tensors, so we
        # must reset the default device and move everything back to vae_device
        # before running inference.  For batch mode, attack_single_video will
        # re-apply the split at the start of the next iteration.
        # ------------------------------------------------------------------
        if vae_device != dit_device:
            torch.cuda.set_device(vae_device)
            model.net = model.net.to(vae_device)
            if hasattr(model, "conditioner") and model.conditioner is not None:
                model.conditioner = model.conditioner.to(vae_device)
            model.tensor_kwargs["device"] = vae_device
            if hasattr(model, "tensor_kwargs_fp32"):
                model.tensor_kwargs_fp32["device"] = vae_device
            torch.cuda.empty_cache()

        if args.skip_latent:
            eval_latent(inference, x_adv, args, out_dir / base_name)
        else:
            eval(inference, x_adv, args, out_dir / base_name)


if __name__ == "__main__":
    main()
