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

where S^(l) is the per-layer pre-softmax cross-attention score matrix.
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
--num_attack_layers         Cut the gradient graph after this many DiT blocks.
                            Default: all blocks.

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
        --num_attack_layers 14

Usage (batch):
    python cosmos-predict2.5/scripts/attack_crossattn.py \
        --batch_path /mnt/ssd2/libero/eval_set/ \
        --attack_num 10 \
        --resolution 432,432 \
        --num_latent_video_frames 9 \
        --num_latent_conditional_frames 2 \
        --steps 100 --alpha 0.00392 --eps 0.0628 \
        --num_attack_layers 14 --skip_latent
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
from probing.test_vae_encoder import load_and_preprocess_video, normalize_video  # noqa: E402
from attack.white_box import pgd
from attack.eval_wm import eval, eval_latent
from attack.shared_config import (
    ckpt_path, experiment_name, config_file
)

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


# ---------------------------------------------------------------------------
# Cross-attention loss hooks
# ---------------------------------------------------------------------------

def install_crossattn_hooks(net, T_tok, loss_terms, cutoff=None):
    """
    Wrap block.cross_attn.compute_attention on every block up to `cutoff`.

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
    net        : the DiT net (model.net)
    T_tok      : number of temporal token-frames
    loss_terms : list — raw (B, T, N, H) score tensors appended during forward
    cutoff     : block index beyond which hooks are not installed

    Returns
    -------
    restore : callable — removes all patches
    """
    _patches = {}
    n_blocks = len(net.blocks)
    limit    = cutoff if cutoff is not None else n_blocks

    for idx in range(limit):
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
                     num_attack_layers=None, num_latent_conditional_frames=2,
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

        restore = install_crossattn_hooks(model.net, T_tok, step_buf, num_attack_layers)
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
    print("NEW")
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

def compute_crossattn_loss(model, x_padded, condition, T_tok, num_attack_layers=None,
                           skip_latent=False, max_att=False, denoise_steps=1,
                           uncondition=None, guidance_scale=7.0,
                           target_condition=None, noise_seed=None,
                           num_latent_conditional_frames=2, sim_masks=None):
    """
    Run `denoise_steps` Euler denoising steps and return the cross-attention loss.

    Untargeted (target_condition=None):
        L = mean_l ||S^(l)||_F^2 / (B*H)

    Targeted (target_condition provided):
        At each denoising step, a no-grad pass with target_condition is run at
        the *same* xt and with the *same* perturbed gt_frames as the base pass.
        L = 1 - cosine_similarity(flat(S_base * M), flat(S_target * M))
        where M is the optional per-layer spatial mask from compute_sim_mask.
        When sim_masks is None the unweighted scores are used.
    """
    compute_dtype = next(model.net.parameters()).dtype

    if skip_latent:
        latent = x_padded
    else:
        latent = model.tokenizer.encode(x_padded.to(compute_dtype)).contiguous().float()
    B, C, T_lat, H_lat, W_lat = latent.shape

    def _build_live(cond):
        return cond.set_video_condition(
            gt_frames=latent.to(compute_dtype),
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_latent_conditional_frames,
        )

    condition_live    = _build_live(condition)
    uncondition_live  = _build_live(uncondition)  if uncondition  is not None else None
    target_cond_live  = _build_live(target_condition) if target_condition is not None else None

    n_blocks = len(model.net.blocks)
    cutoff   = num_attack_layers

    loss_terms = []
    restore    = install_crossattn_hooks(model.net, T_tok, loss_terms, cutoff)

    detach_hook = None
    if cutoff is not None and cutoff < n_blocks:
        def _detach(_m, _inp, output):
            if isinstance(output, tuple):
                return (output[0].detach(),) + output[1:]
            return output.detach()
        detach_hook = model.net.blocks[cutoff - 1].register_forward_hook(_detach)

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
        if sim_masks is not None:
            # Weight each (B,T,S,N,H) score by its per-spatial mask (B,T,S,H),
            # broadcast over the text-token dim N.
            adv_vec = torch.cat([(s * m.unsqueeze(3)).flatten()
                                  for s, m in zip(loss_terms, sim_masks)])
            tgt_vec = torch.cat([(s * m.unsqueeze(3)).detach().flatten()
                                  for s, m in zip(target_terms, sim_masks)])
        else:
            adv_vec = torch.cat([s.flatten() for s in loss_terms])
            tgt_vec = torch.cat([s.detach().flatten() for s in target_terms])
        cos_sim_val = F.cosine_similarity(adv_vec.unsqueeze(0), tgt_vec.unsqueeze(0)).squeeze()
        loss = 1.0 - cos_sim_val
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
    T_tok, cutoff_blocks, compute_dtype, args,
    device="cuda", save_time=None
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
                model.text_encoder.model = model.text_encoder.model.to(device).eval()
        if _t5_mod.cosmos_encoder is not None:
            _t5_mod.cosmos_encoder.text_encoder = \
                _t5_mod.cosmos_encoder.text_encoder.to(device).eval()

    print(f"\nLoading video : {video_path}")
    video_uint8 = load_and_preprocess_video(
        str(video_path), [H, W], frames_to_extract
    )
    raw_cond = normalize_video(video_uint8, device=device)
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
            model.tokenizer.encoder = model.tokenizer.encoder.to(device)
        torch.cuda.empty_cache()

    if inference.offload_diffusion_model:
        model.net = model.net.to(device)
        if hasattr(model, "conditioner") and model.conditioner is not None:
            model.conditioner = model.conditioner.to(device)
        torch.cuda.empty_cache()

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

    # ------------------------------------------------------------------
    # 7. PGD optimisation
    # ------------------------------------------------------------------
    print(f"\nStarting PGD optimisation "
          f"({'targeted' if target_condition_template is not None else 'untargeted'})...")

    # Precompute spatial importance mask on the clean input (one forward pass each,
    # no grad, no denoising trajectory — no memory accumulation).
    sim_masks = None
    if target_condition_template is not None:
        with torch.no_grad():
            latent = model.tokenizer.encode(raw_padded.to(compute_dtype)).contiguous().float()
            print(f"Latent Shape {latent.shape}")
        print("Computing spatial sim mask...")
        sim_masks = compute_sim_mask(
            model, latent, condition, target_condition_template, T_tok,
            num_attack_layers=cutoff_blocks,
            num_latent_conditional_frames=args.num_latent_conditional_frames,
            noise_seed=0
        )                                                                  
    #     analyze_sim_masks(sim_masks, spatial_grid=27, save_dir= "sim_analysis")
    

    loss_fn = lambda x, y: compute_crossattn_loss(
        model, x, condition, T_tok,
        num_attack_layers=cutoff_blocks,
        skip_latent=args.skip_latent,
        max_att=args.max_att,
        denoise_steps=args.denoise_steps,
        uncondition=uncondition if args.denoise_steps > 1 else None,
        guidance_scale=args.guidance_scale,
        target_condition=target_condition_template,
        noise_seed=0 if target_condition_template is not None else None,
        num_latent_conditional_frames=args.num_latent_conditional_frames,
        sim_masks=sim_masks,
    )
    if args.skip_latent:
        with torch.no_grad():
            latent = model.tokenizer.encode(raw_padded.to(compute_dtype)).contiguous().float()
            # latent = torch.empty_like(latent).uniform_(-3, 3)
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

    parser.add_argument("--prompt", default=None,
                        help="Text prompt (required with --video_path)")
    parser.add_argument("--target_prompt", default=None,
                        help="Target prompt for the targeted cross-attention attack. "
                             "When provided, the attack steers the cross-attention "
                             "pattern of (x_adv, --prompt) toward the reference "
                             "pattern of (x_orig, --target_prompt), encouraging the "
                             "model to generate video as if given --target_prompt. "
                             "Omit for the untargeted (attention-suppression) attack.")
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
    parser.add_argument("--num_attack_layers", type=int, default=None,
                        help="Only include the first N DiT blocks in the loss / "
                             "gradient graph. Drastically reduces peak VRAM. "
                             "Grad is cut at block N; forward still runs fully. "
                             "Default: all blocks.")
    parser.add_argument("--load_diffusion_model", action="store_true",
                        help="Keep diffusion model on GPU (default: offload to CPU)")
    parser.add_argument("--load_text_encoder",    action="store_true",
                        help="Keep text encoder on GPU (default: offload to CPU)")
    parser.add_argument("--load_tokenizer",       action="store_true",
                        help="Keep tokenizer on GPU (default: offload to CPU)")
    parser.add_argument("--context_parallel_size",   type=int, default=1)
    parser.add_argument("--config_file",
                        default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    parser.add_argument("--skip_latent", action="store_true",
                        help="Optimise in latent space instead of pixel space.")
    parser.add_argument("--max_att", action="store_true",
                        help="Maximise cross-attention instead of minimising it.")
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

    cutoff_blocks = args.num_attack_layers   # None = all
    compute_dtype = torch.bfloat16

    print(f"Model         : {n_blocks} DiT blocks, state_t={state_t}")
    print(f"Attack T_lat  : {T_lat_attack}  (pixel frames: {required_pixel_frames})")
    print(f"T_tok         : {T_tok}  (frame × text-token cross-attn matrix: {T_tok}×N)")
    print(f"Loss blocks   : {cutoff_blocks or n_blocks}/{n_blocks}  "
          f"(gradient cut after block {(cutoff_blocks or n_blocks) - 1})")
    print(f"PGD           : steps={args.steps}  alpha={args.alpha:.5f}  eps={args.eps:.4f}")

    # ------------------------------------------------------------------
    # Collect (video_path, prompt) pairs
    # ------------------------------------------------------------------
    if args.batch_path is not None:
        batch_path  = Path(args.batch_path)
        video_files = sorted((batch_path / "videos").glob("*.mp4"))
        if args.attack_num > 0:
            video_files = random.sample(video_files, min(args.attack_num, len(video_files)))
        pairs = []
        for vf in video_files:
            meta_path = batch_path / "metas" / (vf.stem + ".txt")
            pairs.append((vf, meta_path.read_text().strip()))
        print(f"\nBatch mode: {len(pairs)} video(s) from {batch_path}")
    else:
        pairs = [(Path(args.video_path), args.prompt)]

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
        "cutoff_blocks": cutoff_blocks,
        "n_blocks": n_blocks,
        "save_time": save_time,
        "out_dir": str(out_dir),
        "batch_pairs": [(str(vp), pr) for vp, pr in pairs],
    }
    with open(out_dir / "attack_config.json", "w") as _f:
        json.dump(attack_config, _f, indent=2)
    print(f"Attack config saved to: {out_dir / 'attack_config.json'}")

    for i, (video_path, prompt) in enumerate(pairs):
        print(f"\n{'='*60}")
        print(f"Video {i+1}/{len(pairs)}: {video_path.name}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        x_adv = attack_single_video(
            inference, model, video_path, prompt,
            H, W, frames_to_extract, required_pixel_frames,
            T_tok, cutoff_blocks, compute_dtype, args,
            device=device, save_time=save_time,
        )
        all_x_adv.append(x_adv)
        base_name = str(video_path).split("/")[-1].split(".")[0] + "_adv.pt"
        print("\nEvaluating Diffusion")
        args.adv_path = out_dir
        args.prompt   = prompt
        if args.skip_latent:
            eval_latent(inference, x_adv, args, out_dir / base_name)
        else:
            eval(inference, x_adv, args, out_dir / base_name)


if __name__ == "__main__":
    main()
