"""
Compute self-attention saliency heatmaps from the Cosmos Video2World DiT.

For each DiT block the script registers a forward hook on block.self_attn.
The hook captures the projected output tensor (B, T_lat*H_lat*W_lat, D) and
computes the L2 norm per spatial-temporal token as a proxy for how much each
region of the video is emphasised by self-attention.  Norms are averaged
across the selected layers, reshaped to (T_lat, H_lat, W_lat), bilinearly
upsampled to the original pixel resolution, and blended onto the input video
frames as a colour heatmap.

The model is loaded first so that the required pixel frame count is read from
model.config.state_t.  The full video is loaded at that length with no
sub-sampling or repeat-padding, so every heatmap frame corresponds to real
video content.

A single DiT forward pass is made at a fixed noise level (--timestep, default
0.5), so the full diffusion loop is never run.

Saved artefacts (in attack/outputs/selfattn_<timestamp>/):
  - selfattn_saliency.pt   : per-frame saliency, shape (T_px, H_px, W_px) float32
  - heatmap.mp4            : input frames with heatmap overlay + frame annotations
  - saliency_only.mp4      : raw coloured heatmap without the original image
  - heatmap_early.mp4      : heatmap from first half of DiT blocks only
  - heatmap_late.mp4       : heatmap from second half of DiT blocks only

Usage:
    python cosmos-predict2.5/scripts/compute_selfattn_heatmap.py \
        --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
        --experiment_name predict2_video2world_training_2b_libero_480 \
        --ckpt_path /path/to/model.pt \
        --prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate"  \
        --resolution 432,432 \
        --num_latent_conditional_frames 2 \
        --timestep 0.5 \
        --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
        --layers 20 \
        --alpha_blend 0.5 \
        --offload_diffusion_model --offload_tokenizer --offload_text_encoder
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
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

def register_selfattn_hooks(net, layer_indices=None):
    """
    Register forward hooks on block.self_attn for each selected DiT block.

    Parameters
    ----------
    net           : the DiT net (net.blocks is the list of transformer blocks)
    layer_indices : list of int indices to hook, or None to hook all blocks

    Returns
    -------
    hooks    : list of hook handles (call h.remove() to clean up)
    captured : dict {block_idx -> (B, T_lat*H_lat*W_lat, D) cpu tensor}
               populated after each forward pass
    """
    hooks    = []
    captured = {}
    n_blocks = len(net.blocks)
    indices  = layer_indices if layer_indices is not None else list(range(n_blocks))

    for idx in indices:
        if idx >= n_blocks:
            continue
        block = net.blocks[idx]
        if not hasattr(block, "self_attn"):
            continue

        def _make_hook(i):
            def _hook(_module, _input, output):
                out = output[0] if isinstance(output, tuple) else output
                captured[i] = out.detach().cpu()  # (B, S, D)
            return _hook

        h = block.self_attn.register_forward_hook(_make_hook(idx))
        hooks.append(h)

    return hooks, captured


# ---------------------------------------------------------------------------
# Saliency computation
# ---------------------------------------------------------------------------

def detect_token_dims(S, T_lat, H_lat, W_lat):
    """
    Infer the DiT token grid (T_tok, H_tok, W_tok) from the sequence length S
    and the VAE latent dimensions.

    The DiT typically applies a 2D spatial patchification (patch size 1, 2, or 4)
    on top of the VAE latent before the transformer blocks, so the spatial token
    dims may differ from the latent spatial dims.  We try each candidate spatial
    patch size and prefer the factorisation where T_tok == T_lat (i.e. no
    additional temporal patchification in the DiT).

    Returns
    -------
    (T_tok, H_tok, W_tok) : int triple whose product equals S
    """
    # Prefer factorisation where T_tok == T_lat (temporal dim is unchanged)
    for p in [1, 2, 4]:
        h, w = H_lat // p, W_lat // p
        if h <= 0 or w <= 0:
            continue
        if S == T_lat * h * w:
            return T_lat, h, w
    # Fallback: any valid factorisation whose T_tok divides T_lat
    for p in [1, 2, 4]:
        h, w = H_lat // p, W_lat // p
        if h <= 0 or w <= 0:
            continue
        if h * w > 0 and S % (h * w) == 0:
            t = S // (h * w)
            if T_lat % t == 0:
                return t, h, w
    raise ValueError(
        f"Cannot factor sequence length S={S} from latent dims "
        f"(T={T_lat}, H={H_lat}, W={W_lat})"
    )


def compute_saliency(captured, T_tok, H_tok, W_tok):
    """
    Convert raw hook outputs to a per-frame saliency map.

    For each captured block output (B, S, D) the L2 norm is computed per
    token to obtain (B, S), then reshaped to (B, T_tok, H_tok, W_tok).
    The result is averaged across all captured blocks.

    Returns
    -------
    saliency : (B, T_tok, H_tok, W_tok) float32 tensor (unnormalised)
    """
    maps = []
    for idx in sorted(captured.keys()):
        out = captured[idx].float()              # (B, S, D)
        norm = out.norm(dim=-1)                  # (B, S)
        B, S = norm.shape
        assert S == T_tok * H_tok * W_tok, (
            f"Block {idx}: expected S={T_tok*H_tok*W_tok}, got {S}"
        )
        maps.append(norm.view(B, T_tok, H_tok, W_tok))

    return torch.stack(maps, dim=0).mean(dim=0)   # (B, T_tok, H_tok, W_tok)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def saliency_to_colormap(saliency_hw, colormap="viridis"):
    """
    Map a (H, W) float32 saliency tensor in [0, 1] to an (H, W, 3) uint8 RGB
    image using matplotlib's colormap.
    """
    import matplotlib.cm as cm
    cmap  = cm.get_cmap(colormap)
    rgb   = cmap(saliency_hw)[:, :, :3]           # (H, W, 3) float [0, 1]
    return (rgb * 255).astype("uint8")


def blend_heatmap(frame_hwc, heatmap_hwc, alpha=0.5):
    """
    Alpha-blend a uint8 heatmap onto a uint8 original frame.

    Parameters
    ----------
    frame_hwc   : (H, W, 3) uint8  original video frame
    heatmap_hwc : (H, W, 3) uint8  coloured saliency heatmap
    alpha       : float in [0, 1]  weight of heatmap (1-alpha for original)

    Returns
    -------
    blended : (H, W, 3) uint8
    """
    import numpy as np
    frame_f   = frame_hwc.astype("float32")
    heatmap_f = heatmap_hwc.astype("float32")
    blended   = (1.0 - alpha) * frame_f + alpha * heatmap_f
    return blended.clip(0, 255).astype("uint8")


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def add_border(frame_hwc, color_rgb, thickness=6):
    """Burn a solid border of `color_rgb` (len-3 list) onto a (H,W,3) uint8 frame."""
    frame = frame_hwc.copy()
    frame[:thickness, :] = color_rgb
    frame[-thickness:, :] = color_rgb
    frame[:, :thickness] = color_rgb
    frame[:, -thickness:] = color_rgb
    return frame


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
    parser.add_argument("--num_latent_conditional_frames", type=int, default=2,
                        help="Conditioning frames used by the model (default: 2)")
    parser.add_argument("--timestep",         type=float, default=0.0,
                        help="Noise level for the single DiT forward pass; "
                             "0 = clean latent (best for visualisation), "
                             "1 = pure noise (default: 0.0)")
    parser.add_argument("--layers",           type=int, nargs="*", default=None,
                        help="DiT block indices to average over. "
                             "Default: all blocks.")
    parser.add_argument("--alpha_blend",      type=float, default=0.5,
                        help="Heatmap opacity when blending onto the video "
                             "(0=original only, 1=heatmap only; default: 0.5)")
    parser.add_argument("--colormap",         default="viridis",
                        help="Matplotlib colormap name (default: viridis)")
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
    # 1. Load model first so we can read state_t → required frame count
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
    state_t               = model.config.state_t
    required_pixel_frames = (state_t - 1) * 4 + 1
    print(f"Model state_t : {state_t}  →  required pixel frames: {required_pixel_frames}")

    # ------------------------------------------------------------------
    # 2. Load the full video at the model's required frame count.
    #    No sub-sampling or repeat-padding — every frame is real content.
    # ------------------------------------------------------------------
    print(f"Loading video : {args.video_path}")
    video_uint8 = load_and_preprocess_video(
        video_path=args.video_path,
        resolution=[H, W],
        num_video_frames=required_pixel_frames,
    )
    print(f"Video shape   : {video_uint8.shape}")    # (1, C, T_px, H, W) uint8

    # ------------------------------------------------------------------
    # 3. Normalise and build data batch (T5 text embedding)
    # ------------------------------------------------------------------
    video_f32  = normalize_video(video_uint8)   # (1, C, T_px, H, W) float32 [-1,1] on CUDA
    video_bf16 = video_f32.to(device=torch.cuda.current_device(), dtype=torch.bfloat16)

    negative_prompt = args.negative_prompt or _DEFAULT_NEGATIVE_PROMPT
    data_batch = inference._get_data_batch_input(
        video=video_bf16,
        prompt=args.prompt,
        num_conditional_frames=args.num_latent_conditional_frames,
        negative_prompt=negative_prompt,
        use_neg_prompt=True,
    )
    data_batch["video"]            = video_bf16
    data_batch[IS_PREPROCESSED_KEY] = True

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
    # 5. VAE encode + build condition
    # ------------------------------------------------------------------
    print("Encoding video and building condition...")
    with torch.no_grad():
        raw_state, latent_state, condition = model.get_data_and_condition(data_batch)

    condition = condition.edit_for_inference(
        is_cfg_conditional=True,
        num_conditional_frames=args.num_latent_conditional_frames,
    )

    B, C_lat, T_lat, H_lat, W_lat = latent_state.shape
    print(f"Latent shape  : {latent_state.shape}")
    print(f"Latent grid   : T={T_lat}, H={H_lat}, W={W_lat} "
          f"(tokens per frame: {H_lat*W_lat}, total: {T_lat*H_lat*W_lat})")

    # ------------------------------------------------------------------
    # 6. Build noisy latent at the requested timestep
    # ------------------------------------------------------------------
    t     = args.timestep
    noise = torch.randn_like(latent_state)
    xt    = ((1 - t) * latent_state + t * noise).to(**model.tensor_kwargs)

    cond_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(1, C_lat, 1, 1, 1).type_as(xt)
    xt = latent_state.type_as(xt) * cond_mask + xt * (1 - cond_mask)

    timesteps = torch.full(
        (B, T_lat), t,
        device=latent_state.device, dtype=latent_state.dtype,
    )

    # ------------------------------------------------------------------
    # 7. Register self-attention hooks on ALL blocks regardless of --layers.
    #    We capture everything, then compute separate per-group saliencies.
    # ------------------------------------------------------------------
    n_blocks = len(model.net.blocks)
    layer_indices = args.layers  # None → all blocks
    print(f"Registering hooks on {n_blocks} DiT blocks "
          f"(layers={layer_indices if layer_indices else 'all'})")

    hooks, captured = register_selfattn_hooks(model.net, layer_indices)
    print(f"Registered hooks on {len(hooks)} self-attention modules")

    print(f"Running single DiT forward pass at t={t}...")
    with torch.no_grad():
        _ = model.net(
            x_B_C_T_H_W=xt,
            timesteps_B_T=timesteps,
            **condition.to_dict(),
        )

    for h in hooks:
        h.remove()

    print(f"Captured self-attn outputs from {len(captured)} blocks")
    for blk_idx, tensor in sorted(captured.items()):
        print(f"  block {blk_idx:2d}: shape={tensor.shape}  "
              f"norm range=[{tensor.norm(dim=-1).min():.4f}, {tensor.norm(dim=-1).max():.4f}]")

    # ------------------------------------------------------------------
    # 9. Compute per-token saliency maps
    # ------------------------------------------------------------------
    print("Computing saliency maps...")

    # Detect the actual token grid dimensions (DiT may spatially patchify the latent)
    sample_S = next(iter(captured.values())).shape[1]
    T_tok, H_tok, W_tok = detect_token_dims(sample_S, T_lat, H_lat, W_lat)
    print(f"Token grid    : T={T_tok}, H={H_tok}, W={W_tok}  "
          f"(spatial patch size ~{H_lat // H_tok}x{W_lat // W_tok}, "
          f"temporal ratio {T_lat // T_tok}x)")

    # Full saliency (user-specified layer subset or all layers)
    selected_keys = sorted(args.layers) if args.layers else sorted(captured.keys())
    saliency_all = compute_saliency(
        {k: captured[k] for k in selected_keys if k in captured},
        T_tok, H_tok, W_tok,
    )   # (1, T_tok, H_tok, W_tok)

    # Early / late block group saliency for diagnostic comparison
    all_keys  = sorted(captured.keys())
    mid       = len(all_keys) // 2
    early_sal = compute_saliency(
        {k: captured[k] for k in all_keys[:mid]}, T_tok, H_tok, W_tok,
    )
    late_sal  = compute_saliency(
        {k: captured[k] for k in all_keys[mid:]}, T_tok, H_tok, W_tok,
    )

    def upsample_to_pixel(sal_tok):
        """(1, T_tok, H_tok, W_tok) → (T_tok, H, W) bilinear upsampled."""
        return F.interpolate(
            sal_tok.view(1, T_tok, H_tok, W_tok).permute(1, 0, 2, 3),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze(1)

    sal_px_all   = upsample_to_pixel(saliency_all)    # (T_tok, H, W)
    sal_px_early = upsample_to_pixel(early_sal)
    sal_px_late  = upsample_to_pixel(late_sal)

    def normalise(sal):
        s_min, s_max = sal.min(), sal.max()
        return (sal - s_min) / (s_max - s_min + 1e-8), s_min, s_max

    sal_norm_all,   s_min_all,   s_max_all   = normalise(sal_px_all)
    sal_norm_early, s_min_early, s_max_early = normalise(sal_px_early)
    sal_norm_late,  s_min_late,  s_max_late  = normalise(sal_px_late)

    # Each token frame maps back through:
    #   token → latent: stride = T_lat // T_tok  (DiT temporal patch factor)
    #   latent → pixel: stride = 4               (WAN VAE temporal factor)
    px_per_tok = 4 * (T_lat // T_tok)

    # Conditioning token frame boundary (clean latent, no noise added)
    num_cond_tok = args.num_latent_conditional_frames * (T_tok // T_lat) \
                   if T_tok < T_lat else args.num_latent_conditional_frames

    # ------------------------------------------------------------------
    # Per-token-frame saliency statistics (diagnostic printout)
    # ------------------------------------------------------------------
    print(f"\nPer-token-frame saliency statistics (t={t}, layers={selected_keys}):")
    print(f"  {'tok':>4}  {'mean':>8}  {'max':>8}  {'type'}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*12}")
    for t_tok in range(T_tok):
        sal = sal_px_all[t_tok]
        frame_type = "COND" if t_tok < num_cond_tok else "GEN"
        print(f"  {t_tok:>4}  {sal.mean():>8.3f}  {sal.max():>8.3f}  {frame_type}")

    # ------------------------------------------------------------------
    # Build annotated video frames
    # ------------------------------------------------------------------
    pixel_frames_np = video_uint8[0].permute(1, 2, 3, 0).numpy()  # (T_px, H, W, C)
    T_px = pixel_frames_np.shape[0]

    # Border colours: green = conditioning, orange = generation
    BORDER_COND = [0, 200, 80]
    BORDER_GEN  = [255, 140, 0]

    def build_frame_list(sal_norm):
        heatmap_f, saliency_f = [], []
        for t_px in range(T_px):
            t_tok_idx = min(t_px // px_per_tok, T_tok - 1)
            sal_hw    = sal_norm[t_tok_idx].numpy()            # (H, W) [0,1]
            cmap_hwc  = saliency_to_colormap(sal_hw, args.colormap)
            orig_hwc  = pixel_frames_np[t_px]
            blended   = blend_heatmap(orig_hwc, cmap_hwc, alpha=args.alpha_blend)
            # Add border to indicate conditioning vs generation
            border_col = BORDER_COND if t_tok_idx < num_cond_tok else BORDER_GEN
            blended    = add_border(blended, border_col)
            heatmap_f.append(blended)
            saliency_f.append(cmap_hwc)
        return heatmap_f, saliency_f

    heatmap_all,   saliency_frames = build_frame_list(sal_norm_all)
    heatmap_early, _               = build_frame_list(sal_norm_early)
    heatmap_late,  _               = build_frame_list(sal_norm_late)

    saliency_out = torch.from_numpy(
        np.stack([sal_norm_all[min(t_px // px_per_tok, T_tok - 1)].numpy()
                  for t_px in range(T_px)], axis=0)
    )   # (T_px, H, W)

    # ------------------------------------------------------------------
    # 10. Save outputs (rank 0 only)
    # ------------------------------------------------------------------
    if local_rank == 0:
        out_dir = WM_ROOT / "attack" / "outputs" / f"selfattn_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(saliency_out, out_dir / "selfattn_saliency.pt")

        def frames_to_tensor(frames_list):
            """list of (H,W,3) uint8 numpy arrays -> (T, H, W, C) uint8 torch tensor"""
            return torch.from_numpy(np.stack(frames_list, axis=0))

        torchvision.io.write_video(str(out_dir / "heatmap.mp4"),
                                   frames_to_tensor(heatmap_all),   fps=16)
        torchvision.io.write_video(str(out_dir / "saliency_only.mp4"),
                                   frames_to_tensor(saliency_frames), fps=16)
        torchvision.io.write_video(str(out_dir / "heatmap_early.mp4"),
                                   frames_to_tensor(heatmap_early), fps=16)
        torchvision.io.write_video(str(out_dir / "heatmap_late.mp4"),
                                   frames_to_tensor(heatmap_late),  fps=16)

        print(f"\nSaved to: {out_dir}")
        print(f"  selfattn_saliency.pt : {saliency_out.shape}")
        print(f"  heatmap.mp4          : all layers  [{s_min_all:.2f}, {s_max_all:.2f}]  "
              f"(green border=COND, orange=GEN)")
        print(f"  heatmap_early.mp4    : blocks {all_keys[:mid]}  [{s_min_early:.2f}, {s_max_early:.2f}]")
        print(f"  heatmap_late.mp4     : blocks {all_keys[mid:]}  [{s_min_late:.2f}, {s_max_late:.2f}]")
        print(f"\nToken grid         : T={T_tok}, H={H_tok}, W={W_tok}")
        print(f"Conditioning tokens: 0–{num_cond_tok-1}  Generation tokens: {num_cond_tok}–{T_tok-1}")
        print(f"Pixels per token   : {px_per_tok}")


if __name__ == "__main__":
    main()


"""

python cosmos-predict2.5/scripts/compute_selfattn_heatmap.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --experiment_name predict2_video2world_training_2b_libero_480 \
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/8fbc6188fa2f2e4ab585dc6aac3edd0e9d8a3670/model.pt \
    --prompt "Use the franka robot arm to pick up the cookie box and place it on the plate" \
    --resolution 432,432 \
    --num_latent_conditional_frames 2 \
    --timestep 0.0 \
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --offload_diffusion_model \
    --offload_tokenizer \
    --offload_text_encoder
    
"""
