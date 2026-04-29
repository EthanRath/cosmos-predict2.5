"""
Attention visualization for Cosmos Video2World DiT.

Runs a full 36-step diffusion generation pass, then captures the self- and
cross-attention maps at the **final conditioned denoising step** and saves
heatmap plots.

Self-attention
--------------
Hooks block.self_attn.compute_attention(q, k, v).  q/k/v: (B, S, H, D)
where S = T*Sf (Sf = H_patch*W_patch).  Two views per block:

  frame_attn  : (T, T) — mean attention between temporal frames (spatial
                tokens averaged out).
  spatial_attn: (Sf, Sf) — within-frame spatial patch mixing (averaged over
                temporal frames and heads).

Note: in context-parallel mode the sequence is split across GPUs; captured
tensors are all_gathered so plots show the full sequence.

Cross-attention
---------------
Hooks block.cross_attn.compute_attention(q, k, v).  Post-softmax weights:
(B, T, Sf, N, H) where N = text-token length.  Two views per block:

  frame_text  : (T, N) — mean attention per frame per text token (spatial
                and head dims averaged out).
  spatial_heat: (H_patch, W_patch) — mean attention per spatial patch
                (text, frame, head dims averaged out).

Aggregated versions (averaged over all captured blocks) are also saved.

--video_path also accepts .pt tensors saved by attack_selfattn_freeze.py or
attack_crossattn.py.  The channel count auto-selects the path:
  C=3  → pixel-space tensor, treated like a loaded video
  C>3  → latent-space tensor, tokenizer.encode is bypassed via monkey-patch

Usage
-----
    torchrun --nproc_per_node=2 cosmos-predict2.5/scripts/visualize_attention.py \
        --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
        --prompt "Use the franka robot arm to pick up the black bowl" \
        --resolution 432,432 \
        --num_latent_video_frames 9 \
        --num_latent_conditional_frames 1 \
        --context_parallel_size 1 \
        --num_steps 36 \
        --blocks 0 7 13 \
        --out_dir attn_viz_2
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
WM_ROOT    = REPO_ROOT.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(WM_ROOT))

from cosmos_predict2._src.predict2.inference.video2world import (
    Video2WorldInference,
    _DEFAULT_NEGATIVE_PROMPT,
)
from cosmos_predict2._src.predict2.models.text2world_model_rectified_flow import (
    IS_PREPROCESSED_KEY,
)
import cosmos_predict2._src.predict2.inference.get_t5_emb as _t5_mod
from probing.test_vae_encoder import load_and_preprocess_video, normalize_video
from attack.shared_config import ckpt_path, experiment_name, config_file


# ---------------------------------------------------------------------------
# Video utilities
# ---------------------------------------------------------------------------

def pad_video(video, frames_to_extract, required_pixel_frames):
    context = video[:, :, -frames_to_extract:, :, :]
    padding = required_pixel_frames - frames_to_extract
    if padding > 0:
        context = torch.cat(
            [context, context[:, :, -1:, :, :].repeat(1, 1, padding, 1, 1)], dim=2
        )
    return context


def load_and_preprocess_image(image_path, resolution, device="cpu"):
    H, W = resolution
    img = Image.open(image_path).convert("RGB").resize((W, H), Image.BILINEAR)
    img_t = torchvision.transforms.functional.to_tensor(img)
    img_t = img_t * 2.0 - 1.0
    return img_t.unsqueeze(0).unsqueeze(2).to(device)


# ---------------------------------------------------------------------------
# Attention capture hooks
# ---------------------------------------------------------------------------

def install_selfattn_hooks(net, captured_self, captured_self_spatial, block_indices,
                           capture_active, Sf, cp_group=None):
    """
    Hook block.self_attn.compute_attention for each block in block_indices.

    Uses the spatial-mean approximation to avoid materialising the full S×S
    attention matrix (which is ~43 M elements for S=6561):
      1. Compute T_local = S // Sf  (Sf = spatial_grid^2, known globally)
      2. Average Q and K over the Sf spatial tokens within each temporal frame
         → Q_mean, K_mean: (B, T_local, H, D)
      3. All-gather across CP ranks → (B, T, H, D) on every rank
      4. Compute (B, T, T, H) frame-to-frame attention scores and softmax

    Also computes a spatial attention map using the complementary approximation:
      - Average Q and K over the T temporal frames → Q_sp, K_sp: (B, Sf, H, D)
      - Gather across CP ranks (mean of shard-local temporal means)
      - Compute (B, Sf, Sf, H) patch-to-patch attention and record attention
        received per patch: sum over query dim → (B, Sf, H), stored as (Sf, H)
        in captured_self_spatial.

    Only captures when capture_active[0] is True.
    """
    _patches = {}

    for idx in block_indices:
        if idx >= len(net.blocks):
            continue
        block = net.blocks[idx]
        if not hasattr(block, "self_attn"):
            continue

        attn    = block.self_attn
        orig_fn = attn.compute_attention

        def _make_wrapper(fn, block_idx):
            def _wrapper(q, k, v, **kw):
                result = fn(q, k, v, **kw)
                if capture_active[0]:
                    with torch.no_grad():
                        B, S, H, D = q.shape
                        # Derive local frame count from the known per-frame token count.
                        # This is robust to CP temporal splits (S_local = T_local * Sf)
                        # where T_local != T_tok (the global frame count).
                        T_local = S // Sf if S % Sf == 0 else 1
                        Sf_use  = S // T_local
                        scale   = D ** -0.5

                        qf = q.float().reshape(B, T_local, Sf_use, H, D)
                        kf = k.float().reshape(B, T_local, Sf_use, H, D)

                        # Spatial mean over patches → (B, T_local, H, D) for frame attn
                        q_m = qf.mean(dim=2)
                        k_m = kf.mean(dim=2)

                        # Temporal mean over frames → (B, Sf_use, H, D) for spatial attn
                        q_s = qf.mean(dim=1)
                        k_s = kf.mean(dim=1)

                        # Gather across CP ranks so every rank has full (B, T, H, D)
                        if cp_group is not None and torch.distributed.is_initialized():
                            ws = torch.distributed.get_world_size(cp_group)
                            if ws > 1:
                                q_shards = [torch.zeros_like(q_m) for _ in range(ws)]
                                k_shards = [torch.zeros_like(k_m) for _ in range(ws)]
                                torch.distributed.all_gather(q_shards, q_m.contiguous(),
                                                             group=cp_group)
                                torch.distributed.all_gather(k_shards, k_m.contiguous(),
                                                             group=cp_group)
                                q_m = torch.cat(q_shards, dim=1)   # (B, T, H, D)
                                k_m = torch.cat(k_shards, dim=1)

                                # Gather spatial shard-local temporal means then average
                                qs_shards = [torch.zeros_like(q_s) for _ in range(ws)]
                                ks_shards = [torch.zeros_like(k_s) for _ in range(ws)]
                                torch.distributed.all_gather(qs_shards, q_s.contiguous(),
                                                             group=cp_group)
                                torch.distributed.all_gather(ks_shards, k_s.contiguous(),
                                                             group=cp_group)
                                q_s = torch.stack(qs_shards, dim=0).mean(dim=0)  # (B, Sf, H, D)
                                k_s = torch.stack(ks_shards, dim=0).mean(dim=0)

                        # Frame-to-frame scores: (B, T, T, H)
                        scores  = torch.einsum("bthd,bshd->btsh", q_m, k_m) * scale
                        weights = F.softmax(scores, dim=2)   # softmax over key frames
                        captured_self[block_idx] = weights.cpu()

                        # Spatial patch-to-patch scores: (B, Sf, Sf, H)
                        # scores_s[b, p, q, h] = query-patch p attending to key-patch q
                        scores_s  = torch.einsum("bphd,bqhd->bpqh", q_s, k_s) * scale
                        weights_s = F.softmax(scores_s, dim=2)   # softmax over key patches
                        # Attention received by each patch: sum over query dim → (B, Sf, H)
                        attn_rcvd = weights_s.sum(dim=1).cpu()   # (B, Sf, H)
                        captured_self_spatial[block_idx] = attn_rcvd.mean(dim=0)  # (Sf, H)

                return result
            return _wrapper

        attn.__dict__["compute_attention"] = _make_wrapper(orig_fn, idx)
        _patches[idx] = (attn, orig_fn)

    def _restore():
        for _idx, (attn_mod, _orig) in _patches.items():
            attn_mod.__dict__.pop("compute_attention", None)

    return _restore


def install_crossattn_hooks(net, captured_cross, block_indices, capture_active):
    """
    Hook block.cross_attn.compute_attention for each block in block_indices.

    Stores raw post-softmax weights as (B, S_local, N, H) on GPU — no temporal
    reshaping at capture time.  In CP mode each rank may hold a shard of the
    full token sequence; gather_captured reassembles and reshapes afterward.
    """
    _patches = {}

    for idx in block_indices:
        if idx >= len(net.blocks):
            continue
        block = net.blocks[idx]
        if not hasattr(block, "cross_attn"):
            continue

        attn    = block.cross_attn
        orig_fn = attn.compute_attention

        def _make_wrapper(fn, block_idx):
            def _wrapper(q, k, v, **kw):
                result = fn(q, k, v, **kw)
                if capture_active[0]:
                    with torch.no_grad():
                        B, S, H, D = q.shape
                        scale   = D ** -0.5
                        # scores: (B, S_local, N, H)
                        scores  = torch.einsum(
                            "bshd,bnhd->bsnh", q.float(), k.float()
                        ) * scale
                        weights = F.softmax(scores, dim=2)   # softmax over N
                        captured_cross[block_idx] = weights  # keep on GPU for gather
                return result
            return _wrapper

        attn.__dict__["compute_attention"] = _make_wrapper(orig_fn, idx)
        _patches[idx] = (attn, orig_fn)

    def _restore():
        for _idx, (attn_mod, _orig) in _patches.items():
            attn_mod.__dict__.pop("compute_attention", None)

    return _restore


# ---------------------------------------------------------------------------
# All-gather helper for CP mode
# ---------------------------------------------------------------------------

def gather_along_dim(tensor, dim, group=None):
    """
    All-gather `tensor` (must be a CUDA tensor) across the process group and
    concatenate along `dim`.  Returns unchanged if distributed is not initialised.
    """
    if not torch.distributed.is_initialized():
        return tensor
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return tensor
    shards = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(shards, tensor.contiguous(), group=group)
    return torch.cat(shards, dim=dim)


def gather_captured(captured_self, captured_cross, device, T_tok, Sf):
    """
    Gather cross-attention tensors across CP ranks, trim padding, and reshape.

    Self-attn is already gathered inside the hook (Q/K all-gathered before
    score computation), so it only needs to stay on CPU.

    Cross-attn pipeline:
      Each rank holds (B, S_local, N, H) on GPU.
      1. all_gather along S dim (dim 1) → (B, S_gathered, N, H)
         S_gathered >= T_tok * Sf due to CP padding; trim to S_true = T_tok * Sf.
      2. Reshape to (B, T_tok, Sf, N, H) and move to CPU.
    """
    S_true = T_tok * Sf

    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        for k in list(captured_cross.keys()):
            w = captured_cross[k]   # (B, S_local, N, H) on GPU
            B, S, N, H = w.shape
            w = w[:, :S_true].reshape(B, T_tok, Sf, N, H)
            captured_cross[k] = w.cpu()
        return captured_self, captured_cross

    for block_idx in list(captured_cross.keys()):
        w = captured_cross[block_idx].to(device)          # (B, S_local, N, H)
        w = gather_along_dim(w, dim=1)                    # (B, S_gathered, N, H)
        B, _, N, H = w.shape
        w = w[:, :S_true].reshape(B, T_tok, Sf, N, H)    # (B, T, Sf, N, H)
        captured_cross[block_idx] = w.cpu()

    return captured_self, captured_cross


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_selfattn(captured_self, save_dir, suffix=""):
    """
    Plot frame-to-frame self-attention heatmaps.
    captured_self values: (B, T, T, H) — spatial-mean approximation.

    Saves two files per block:
      *_frame_attn  : (T, T) averaged over B and H
      *_per_head    : grid of H individual (T, T) heatmaps, one per head
    """
    import math
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    (save_dir / "self_attn").mkdir(parents=True, exist_ok=True)

    for block_idx, weights in sorted(captured_self.items()):
        # weights: (B, T, T, H)
        B, T, _, H = weights.shape
        w = weights.float().mean(dim=0)   # (T, T, H) — avg over batch

        # ---- head-averaged (T, T) ----
        fa_np = w.mean(dim=2).numpy()   # (T, T)

        fig, ax = plt.subplots(figsize=(max(5, T // 2), max(5, T // 2)))
        im = ax.imshow(fa_np, cmap="viridis", aspect="auto", interpolation="nearest")
        ax.set_title(f"Block {block_idx} — frame self-attention (T×T, head-mean)")
        ax.set_xlabel("Key frame")
        ax.set_ylabel("Query frame")
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(save_dir / "self_attn" / f"block{block_idx:03d}_frame_attn{suffix}.png",
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

        # ---- per-head grid: each head's (T, T) matrix ----
        cols = min(H, 8)
        rows = math.ceil(H / cols)
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(cols * (T // 2 + 1), rows * (T // 2 + 1)))
        axes = [axes] if H == 1 else list(axes.flat)
        for h_i in range(H):
            ax = axes[h_i]
            ax.imshow(w[:, :, h_i].numpy(), cmap="viridis", aspect="auto",
                      interpolation="nearest")
            ax.set_title(f"head {h_i}", fontsize=8)
            ax.set_xticks(range(0, T, max(1, T // 6)))
            ax.set_yticks(range(0, T, max(1, T // 6)))
            ax.tick_params(labelsize=6)
        for h_i in range(H, len(axes)):
            axes[h_i].set_visible(False)
        fig.suptitle(f"Block {block_idx} — self-attention per head (T×T)", fontsize=10)
        fig.tight_layout()
        fig.savefig(save_dir / "self_attn" / f"block{block_idx:03d}_per_head{suffix}.png",
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved self-attention plots to {save_dir / 'self_attn'}")


def plot_selfattn_spatial(captured_self_spatial, spatial_grid, save_dir, suffix=""):
    """
    Plot spatial self-attention heatmaps.
    captured_self_spatial values: (Sf, H) — attention received per patch per head,
    computed from a temporal-mean approximation of Q/K.

    Saves two files per block:
      *_spatial      : (spatial_grid, spatial_grid) head-averaged heatmap
      *_spatial_per_head : grid of H individual spatial heatmaps
    """
    import math
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    (save_dir / "self_attn").mkdir(parents=True, exist_ok=True)

    for block_idx, attn_rcvd in sorted(captured_self_spatial.items()):
        # attn_rcvd: (Sf, H)
        Sf, H = attn_rcvd.shape
        g = spatial_grid if spatial_grid * spatial_grid == Sf else int(Sf ** 0.5)

        # ---- head-averaged (spatial_grid, spatial_grid) ----
        sp_np = attn_rcvd.float().mean(dim=1).numpy().reshape(g, g)

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(sp_np, cmap="viridis", interpolation="nearest")
        ax.set_title(f"Block {block_idx} — self-attn received per patch\n"
                     f"({g}×{g} patches, head-mean, temporal-mean approx)")
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(save_dir / "self_attn" / f"block{block_idx:03d}_spatial{suffix}.png",
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

        # ---- per-head grid ----
        cols = min(H, 8)
        rows = math.ceil(H / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = [axes] if H == 1 else list(axes.flat)
        for h_i in range(H):
            ax = axes[h_i]
            ax.imshow(attn_rcvd[:, h_i].float().numpy().reshape(g, g),
                      cmap="viridis", interpolation="nearest")
            ax.set_title(f"head {h_i}", fontsize=8)
            ax.axis("off")
        for h_i in range(H, len(axes)):
            axes[h_i].set_visible(False)
        fig.suptitle(f"Block {block_idx} — self-attn received per patch (per head)", fontsize=10)
        fig.tight_layout()
        fig.savefig(save_dir / "self_attn" / f"block{block_idx:03d}_spatial_per_head{suffix}.png",
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved spatial self-attention plots to {save_dir / 'self_attn'}")


def plot_crossattn(captured_cross, T_tok, spatial_grid, save_dir, suffix=""):
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    (save_dir / "cross_attn").mkdir(parents=True, exist_ok=True)

    for block_idx, weights in sorted(captured_cross.items()):
        # weights: (B, T, Sf, N, H)
        B, T, Sf, N, H = weights.shape

        # ---- frame × text token (T, N) ----
        # Mean over B, Sf, H — auto-scale colormap so small but real variations show.
        ft    = weights.mean(dim=(0, 2, 4))   # (T, N)
        ft_np = ft.float().numpy()

        fig, ax = plt.subplots(figsize=(max(6, N // 4), max(4, T // 2)))
        im = ax.imshow(ft_np, cmap="hot", aspect="auto", interpolation="nearest")
        ax.set_title(f"Block {block_idx} — cross-attention: frame × text token")
        ax.set_xlabel("Text token index")
        ax.set_ylabel("Video frame")
        ax.set_yticks(range(T))
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(save_dir / "cross_attn" / f"block{block_idx:03d}_frame_text{suffix}.png",
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

        # ---- spatial heatmap (H_patch, W_patch) ----
        # Max over text-token dim → for each patch, peak attention to any text token.
        # Averaging over N collapses to ~1/N for every patch (uniform), so use max
        # to reveal which patches concentrate their attention budget.
        sp = weights.float().max(dim=3).values.mean(dim=(0, 1, 3))   # (Sf,)
        try:
            sp_grid = sp.reshape(spatial_grid, spatial_grid).numpy()
        except RuntimeError:
            g = int(Sf ** 0.5)
            sp_grid = sp.reshape(g, g).numpy()

        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(sp_grid, cmap="hot", interpolation="nearest")
        ax.set_title(f"Block {block_idx} — cross-attention: peak attn per patch")
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(save_dir / "cross_attn" / f"block{block_idx:03d}_spatial{suffix}.png",
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved cross-attention plots to {save_dir / 'cross_attn'}")


def plot_aggregated_selfattn(captured_self, save_dir, suffix=""):
    import matplotlib.pyplot as plt
    if not captured_self:
        return
    save_dir = Path(save_dir)

    weights_list = [v for _, v in sorted(captured_self.items())]
    # Each weight: (B, T, T, H)
    agg = torch.stack([w.mean(dim=(0, 3)) for w in weights_list]).mean(dim=0)  # (T, T)
    T   = agg.shape[0]
    agg_np = agg.float().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(agg_np, cmap="viridis", vmin=0, aspect="auto", interpolation="nearest")
    ax.set_title("All blocks aggregated — frame-level self-attention (spatial-mean)")
    ax.set_xlabel("Key frame")
    ax.set_ylabel("Query frame")
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_dir / f"self_attn_agg{suffix}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved aggregated self-attention to {save_dir / f'self_attn_agg{suffix}.png'}")


def plot_aggregated_selfattn_spatial(captured_self_spatial, spatial_grid, save_dir, suffix=""):
    import matplotlib.pyplot as plt
    if not captured_self_spatial:
        return
    save_dir = Path(save_dir)

    sp_list = [v.float().mean(dim=1) for _, v in sorted(captured_self_spatial.items())]
    agg_sp  = torch.stack(sp_list).mean(dim=0)   # (Sf,)

    g = spatial_grid if spatial_grid * spatial_grid == agg_sp.shape[0] else int(agg_sp.shape[0] ** 0.5)
    sp_grid = agg_sp.numpy().reshape(g, g)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(sp_grid, cmap="viridis", interpolation="nearest")
    ax.set_title("All blocks aggregated — self-attn received per patch\n"
                 "(head-mean, temporal-mean approx)")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_dir / f"self_attn_spatial_agg{suffix}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved aggregated spatial self-attention to "
          f"{save_dir / f'self_attn_spatial_agg{suffix}.png'}")


def plot_aggregated_crossattn(captured_cross, T_tok, spatial_grid, save_dir, suffix=""):
    import matplotlib.pyplot as plt
    if not captured_cross:
        return
    save_dir = Path(save_dir)

    weights_list = [v for _, v in sorted(captured_cross.items())]
    # Max over text-token dim then mean over B, T, H — same as per-block spatial plot
    sp_list = [w.float().max(dim=3).values.mean(dim=(0, 1, 3)) for w in weights_list]  # (Sf,)
    agg_sp  = torch.stack(sp_list).mean(dim=0)

    try:
        sp_grid = agg_sp.reshape(spatial_grid, spatial_grid).numpy()
    except RuntimeError:
        g = int(agg_sp.shape[0] ** 0.5)
        sp_grid = agg_sp.reshape(g, g).numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(sp_grid, cmap="hot", interpolation="nearest")
    ax.set_title("All blocks aggregated — cross-attention: peak attn per patch")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_dir / f"cross_attn_spatial_agg{suffix}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved aggregated cross-attention to {save_dir / f'cross_attn_spatial_agg{suffix}.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path",
                             help="Path to a conditioning video (.mp4) or a saved "
                                  "attack tensor (.pt).  .pt files are auto-detected "
                                  "as pixel-space (C=3) or latent-space (C>3).")
    input_group.add_argument("--image_path",
                             help="Path to a conditioning image (.jpg/.png)")

    parser.add_argument("--prompt", required=True)
    parser.add_argument("--resolution",     default="432,432")
    parser.add_argument("--num_latent_video_frames",       type=int, default=9)
    parser.add_argument("--num_latent_conditional_frames", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=36,
                        help="Number of diffusion denoising steps (default: 36)")
    parser.add_argument("--guidance",  type=float, default=7.0)
    parser.add_argument("--seed",      type=int,   default=1)
    parser.add_argument("--attn_type", choices=["self", "cross", "both"], default="both",
                        help="Which attention type to capture and visualise "
                             "(default: both).  Use 'cross' or 'self' to halve "
                             "peak VRAM during the capture step.")
    parser.add_argument("--blocks", type=int, nargs="*", default=None,
                        help="DiT block indices to visualise.  "
                             "Defaults to 7 evenly-spaced blocks.")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory.  "
                             "Defaults to attack/outputs/attn_viz_<timestamp>/")
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--ckpt_path",       type=str, default=None)
    parser.add_argument("--config_file",
                        default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    parser.add_argument("--context_parallel_size", type=int, default=1,
                        help="Number of GPUs for context parallelism.  "
                             "Must match --nproc_per_node when using torchrun.")
    parser.add_argument("--load_diffusion_model", action="store_true")
    parser.add_argument("--load_text_encoder",    action="store_true")
    parser.add_argument("--load_tokenizer",       action="store_true")
    args = parser.parse_args()

    if args.ckpt_path is None:
        args.ckpt_path = ckpt_path
    if args.experiment_name is None:
        args.experiment_name = experiment_name

    # torchrun sets LOCAL_RANK; fall back to 0 for single-process runs
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_rank0   = local_rank == 0

    compute_dtype = torch.bfloat16
    H, W          = [int(x) for x in args.resolution.split(",")]

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    if is_rank0:
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

    is_image         = args.image_path is not None
    is_pt            = args.video_path is not None and Path(args.video_path).suffix == ".pt"
    num_cond_frames  = 1 if is_image else args.num_latent_conditional_frames
    T_lat            = min(args.num_latent_video_frames, state_t)
    req_pixel_frames = (T_lat - 1) * 4 + 1
    frames_to_extract = 1 if is_image else (num_cond_frames - 1) * 4 + 1
    T_tok            = T_lat
    n_blocks         = len(model.net.blocks)
    spatial_grid     = H // 16   # patch size 16 → e.g. 432/16 = 27; overridden for .pt

    block_indices = (
        args.blocks if args.blocks is not None
        else [int(round(i * (n_blocks - 1) / 6)) for i in range(7)]
    )

    if is_rank0:
        print(f"Model         : {n_blocks} DiT blocks, state_t={state_t}")
        print(f"T_tok         : {T_tok}   spatial_grid: {spatial_grid}×{spatial_grid}")
        print(f"Blocks        : {block_indices}")
        print(f"Diffusion steps: {args.num_steps}")
        print(f"Attn type     : {args.attn_type}")

    # ------------------------------------------------------------------
    # 2. Text encoder on GPU to build data batch
    # ------------------------------------------------------------------
    device = torch.cuda.current_device()

    if inference.offload_text_encoder:
        if model.text_encoder is not None:
            if hasattr(model.text_encoder, "model") and model.text_encoder.model is not None:
                model.text_encoder.model = model.text_encoder.model.to(device).eval()
        if _t5_mod.cosmos_encoder is not None:
            _t5_mod.cosmos_encoder.text_encoder = \
                _t5_mod.cosmos_encoder.text_encoder.to(device).eval()

    # ------------------------------------------------------------------
    # 3. Load and preprocess input
    # ------------------------------------------------------------------
    negative_prompt = args.negative_prompt or _DEFAULT_NEGATIVE_PROMPT
    pt_latent       = None   # set to float32 GPU tensor for latent-space .pt files

    if is_image:
        if is_rank0:
            print(f"\nLoading image: {args.image_path}")
        raw_cond   = load_and_preprocess_image(args.image_path, [H, W], device="cpu")
        raw_padded = pad_video(raw_cond, frames_to_extract=1,
                               required_pixel_frames=req_pixel_frames)
        if is_rank0:
            print(f"  Padded input shape: {raw_padded.shape}")
        video_bf16 = raw_padded.to(device=device, dtype=compute_dtype)

    elif is_pt:
        if is_rank0:
            print(f"\nLoading .pt tensor: {args.video_path}")
        raw_tensor = torch.load(args.video_path, map_location="cpu")
        if raw_tensor.dim() != 5:
            raise ValueError(
                f"Expected 5-D tensor (B,C,T,H,W) in {args.video_path}, "
                f"got shape {raw_tensor.shape}"
            )
        _, C_pt, T_pt, H_pt, W_pt = raw_tensor.shape

        if C_pt == 3:
            # Pixel-space .pt — pad/truncate to req_pixel_frames (same as mp4 path)
            raw_tensor = raw_tensor.float()
            _, _, T_px, H, W = raw_tensor.shape
            if T_px < req_pixel_frames:
                pad_n = req_pixel_frames - T_px
                raw_tensor = torch.cat(
                    [raw_tensor,
                     raw_tensor[:, :, -1:, :, :].repeat(1, 1, pad_n, 1, 1)],
                    dim=2,
                )
                if is_rank0:
                    print(f"  Padded pixel .pt from T={T_px} to T={req_pixel_frames}")
            elif T_px > req_pixel_frames:
                raw_tensor = raw_tensor[:, :, :req_pixel_frames]
                if is_rank0:
                    print(f"  Truncated pixel .pt from T={T_px} to T={req_pixel_frames}")
            raw_padded   = raw_tensor
            T_tok        = T_lat
            spatial_grid = H // 16
            if is_rank0:
                print(f"  Pixel-space .pt   shape: {raw_padded.shape}  "
                      f"(T_tok={T_tok}, spatial_grid={spatial_grid}×{spatial_grid})")
            video_bf16 = raw_padded.to(device=device, dtype=compute_dtype)
        else:
            # Latent-space .pt — bypass the tokenizer via model.encode monkey-patch
            scf  = model.tokenizer.spatial_compression_factor   # e.g. 8
            H    = H_pt * scf
            W    = W_pt * scf
            spatial_grid = H // 16
            # Use min(T_pt, T_lat) frames — truncate if file is longer, keep as-is if shorter
            T_tok = min(T_pt, T_lat)
            if T_pt != T_tok:
                raw_tensor = raw_tensor[:, :, :T_tok, :, :]
                if is_rank0:
                    print(f"  Truncated latent from T={T_pt} to T={T_tok}")
            pt_latent  = raw_tensor.float().to(device=device)
            # Dummy pixel video sized to match T_tok (for T5/conditioning metadata only)
            T_pixel    = (T_tok - 1) * 4 + 1
            video_bf16 = torch.zeros(
                1, 3, T_pixel, H, W, device=device, dtype=compute_dtype
            )
            if is_rank0:
                print(f"  Latent-space .pt  shape: {pt_latent.shape}  "
                      f"(derived pixel: {H}×{W}, spatial_grid={spatial_grid}×{spatial_grid})")

    else:
        if is_rank0:
            print(f"\nLoading video: {args.video_path}")
        video_uint8 = load_and_preprocess_video(
            str(args.video_path), [H, W], frames_to_extract
        )
        raw_cond   = normalize_video(video_uint8, device="cpu")
        raw_padded = pad_video(raw_cond, frames_to_extract, req_pixel_frames)
        if is_rank0:
            print(f"  Padded input shape: {raw_padded.shape}")
        video_bf16 = raw_padded.to(device=device, dtype=compute_dtype)

    data_batch = inference._get_data_batch_input(
        video=video_bf16,
        prompt=args.prompt,
        num_conditional_frames=num_cond_frames,
        negative_prompt=negative_prompt,
        use_neg_prompt=True,
    )
    data_batch["video"]             = video_bf16
    data_batch[IS_PREPROCESSED_KEY] = True

    # ------------------------------------------------------------------
    # 4. Offload text encoder; ensure tokenizer + diffusion model on GPU
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

    for p in model.net.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # 5. Install attention hooks + denoise call counter
    #
    # generate_samples_from_batch calls velocity_fn at each step, which
    # does:  cond_pass (call 1,3,...) then uncond_pass (call 2,4,...).
    # Total = 2 * num_steps calls.  We capture only on the final cond pass
    # (call number 2*num_steps - 1).
    # ------------------------------------------------------------------
    capture_active       = [False]
    captured_self        = {}
    captured_self_spatial = {}
    captured_cross       = {}

    # Track denoise calls to identify the final cond pass
    call_counter        = [0]
    total_denoise_calls = 2 * args.num_steps  # cond + uncond per step
    orig_denoise        = model.denoise

    def _tracked_denoise(*a, **kw):
        call_counter[0] += 1
        # Odd calls = cond (1, 3, ...), even = uncond (2, 4, ...)
        is_cond  = (call_counter[0] % 2 == 1)
        is_final = (call_counter[0] >= total_denoise_calls - 1)
        capture_active[0] = is_cond and is_final
        if is_rank0 and is_final:
            label = "cond" if is_cond else "uncond"
            print(f"  [attn capture] final step {label} pass — hooks active: {capture_active[0]}")
        return orig_denoise(*a, **kw)

    model.denoise = _tracked_denoise

    cp_group = (model.get_context_parallel_group()
                if hasattr(model, "get_context_parallel_group") else None)

    do_self  = args.attn_type in ("self",  "both")
    do_cross = args.attn_type in ("cross", "both")

    Sf = spatial_grid * spatial_grid

    restore_self = (
        install_selfattn_hooks(model.net, captured_self, captured_self_spatial,
                               block_indices, capture_active, Sf, cp_group=cp_group)
        if do_self else (lambda: None)
    )
    restore_cross = (
        install_crossattn_hooks(model.net, captured_cross, block_indices,
                                capture_active)
        if do_cross else (lambda: None)
    )

    # ------------------------------------------------------------------
    # 6. Run full diffusion generation
    # ------------------------------------------------------------------
    if is_rank0:
        print(f"\nRunning {args.num_steps}-step diffusion generation...")

    if getattr(model.config, "use_lora", False):
        generate_fn = model.generate_samples_from_batch_lora
    else:
        generate_fn = model.generate_samples_from_batch

    generate_kwargs = dict(
        n_sample=1,
        guidance=args.guidance,
        seed=args.seed,
        is_negative_prompt=True,
        num_steps=args.num_steps,
    )
    if pt_latent is not None:
        # Bypass tokenizer.encode — return the pre-computed latent directly.
        # Assigning to the instance shadows the class method; del restores it.
        model.encode = lambda state: pt_latent
        generate_kwargs["state_shape"] = list(pt_latent.shape[1:])

    try:
        with torch.no_grad():
            sample = generate_fn(data_batch, **generate_kwargs)
    finally:
        # Restore hooks and denoise regardless of exceptions
        restore_self()
        restore_cross()
        del model.denoise   # removes instance attribute, restores class method
        if pt_latent is not None and "encode" in model.__dict__:
            del model.encode

    if is_rank0:
        print(f"  Captured self-attn  : {len(captured_self)} blocks")
        print(f"  Captured cross-attn : {len(captured_cross)} blocks")

    # ------------------------------------------------------------------
    # 7. All-gather attention tensors across CP ranks
    # ------------------------------------------------------------------
    Sf = spatial_grid * spatial_grid
    captured_self, captured_cross = gather_captured(
        captured_self, captured_cross, device, T_tok, Sf)

    # ------------------------------------------------------------------
    # 8. Decode generated video and save (rank 0 only)
    # ------------------------------------------------------------------
    if is_rank0:
        print("Decoding generated video...")

    with torch.no_grad():
        if isinstance(sample, list):
            chunks   = [model.decode(s) for s in sample]
            video_out = torch.cat(chunks, dim=3)
        else:
            video_out = model.decode(sample)

    ts      = int(time.time())
    out_dir = Path(args.out_dir) if args.out_dir else \
              WM_ROOT / "attack" / "outputs" / f"attn_viz_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_rank0:
        def to_uint8_frames(t):
            t = t[0].cpu().float()
            t = ((t.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
            return t.permute(1, 2, 3, 0)

        vid_path = out_dir / f"generated_{ts}.mp4"
        torchvision.io.write_video(str(vid_path), to_uint8_frames(video_out), fps=24)
        print(f"  Saved generated video: {vid_path}")

        # ------------------------------------------------------------------
        # 9. Plot attention maps
        # ------------------------------------------------------------------
        if not captured_self and not captured_cross:
            print("[warn] No attention tensors captured — check block indices and hook setup.")
        else:
            print(f"\nSaving attention plots to: {out_dir}")
            suffix = f"_step{args.num_steps}"

            if captured_self:
                for block_idx, w in sorted(captured_self.items()):
                    B, T, _, Hn = w.shape
                    sp = captured_self_spatial.get(block_idx)
                    sp_str = f"  spatial: (Sf={sp.shape[0]}, H={sp.shape[1]})" if sp is not None else ""
                    print(f"  [self  block {block_idx:3d}]  (B={B}, T={T}, T={T}, H={Hn}){sp_str}")
                plot_selfattn(captured_self, out_dir, suffix)
                plot_aggregated_selfattn(captured_self, out_dir, suffix)
                if captured_self_spatial:
                    plot_selfattn_spatial(captured_self_spatial, spatial_grid, out_dir, suffix)
                    plot_aggregated_selfattn_spatial(captured_self_spatial, spatial_grid,
                                                     out_dir, suffix)

            if captured_cross:
                for block_idx, w in sorted(captured_cross.items()):
                    B, T, Sf, N, H = w.shape
                    print(f"  [cross block {block_idx:3d}]  "
                          f"(B={B}, T={T}, Sf={Sf}, N={N}, H={H})")
                plot_crossattn(captured_cross, T_tok, spatial_grid, out_dir, suffix)
                plot_aggregated_crossattn(captured_cross, T_tok, spatial_grid, out_dir, suffix)

        print(f"\nDone.  All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
