"""
Compute cross-attention saliency heatmaps from the Cosmos Video2World DiT.

For each DiT block the script registers a forward hook on block.cross_attn.
Cross-attention computes video tokens (queries) against text tokens from the
prompt (keys/values).

Two saliency modes are supported:

  norm  (default)
      L2 norm of the cross-attention output per video token.  Fast and
      requires a single forward pass, but is dominated by the magnitude of
      the text value vectors rather than the attention weight pattern.  Can
      appear flat/uninformative if all text value vectors have similar norms.

  diff  (recommended for prompt sensitivity)
      Requires --compare_prompt.  Runs two forward passes (--prompt and
      --compare_prompt) and computes ||out_A - out_B|| per video token.
      Regions with high difference are genuinely driven by the text; a flat
      diff map means cross-attention has low spatial variation in this model.
      This mode directly answers "is the heatmap noisy because of a bug, or
      because the model really doesn't spatially localise cross-attention?"

The model is loaded first so the required pixel frame count is read from
model.config.state_t.  The full video is loaded at that length with no
sub-sampling or repeat-padding.

Saved artefacts (in attack/outputs/crossattn_heatmap_<timestamp>/):
  - heatmap.mp4             : input frames with heatmap overlay + annotations
  - saliency_only.mp4       : raw coloured heatmap
  - heatmap_early.mp4       : first-half DiT blocks only
  - heatmap_late.mp4        : second-half DiT blocks only
  - crossattn_saliency.pt   : per-frame saliency (T_px, H_px, W_px) float32
  [diff mode only]
  - prompt_diff.mp4         : ||out_A - out_B|| heatmap
  - prompt_diff.pt          : raw difference saliency tensor

Usage (norm mode):
    python cosmos-predict2.5/scripts/compute_crossattn_heatmap.py \\
        --video_path /path/to/video.mp4 ... \\
        --prompt "prompt A"

Usage (diff mode — recommended):
    python cosmos-predict2.5/scripts/compute_crossattn_heatmap.py \\
        --video_path /path/to/video.mp4 ... \\
        --prompt "prompt A" \\
        --compare_prompt "a completely different prompt"
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

def register_crossattn_hooks(net, layer_indices=None):
    """
    Register forward hooks on block.cross_attn for each selected DiT block.

    Parameters
    ----------
    net           : the DiT net (net.blocks is the list of transformer blocks)
    layer_indices : list of int indices to hook, or None to hook all blocks

    Returns
    -------
    hooks    : list of hook handles (call h.remove() to clean up)
    captured : dict {block_idx -> (B, T_tok*H_tok*W_tok, D) cpu tensor}
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
        if not hasattr(block, "cross_attn"):
            continue

        def _make_hook(i):
            def _hook(_module, _input, output):
                out = output[0] if isinstance(output, tuple) else output
                captured[i] = out.detach().cpu()   # (B, S, D)
            return _hook

        h = block.cross_attn.register_forward_hook(_make_hook(idx))
        hooks.append(h)

    return hooks, captured


# ---------------------------------------------------------------------------
# Token dimension detection (shared logic with compute_selfattn_heatmap.py)
# ---------------------------------------------------------------------------

def detect_token_dims(S, T_lat, H_lat, W_lat):
    """
    Infer the DiT token grid (T_tok, H_tok, W_tok) from the sequence length S
    and the VAE latent dimensions.

    Tries spatial patch sizes 1, 2, 4 and prefers the factorisation where
    T_tok == T_lat (no additional temporal patchification in the DiT).
    """
    for p in [1, 2, 4]:
        h, w = H_lat // p, W_lat // p
        if h <= 0 or w <= 0:
            continue
        if S == T_lat * h * w:
            return T_lat, h, w
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


# ---------------------------------------------------------------------------
# Saliency computation
# ---------------------------------------------------------------------------

def compute_saliency(captured, T_tok, H_tok, W_tok):
    """
    Convert raw cross-attn hook outputs to a per-token saliency map.

    For each captured block output (B, S, D) the L2 norm is computed per
    token to obtain (B, S), reshaped to (B, T_tok, H_tok, W_tok), then
    averaged across all captured blocks.

    Returns
    -------
    saliency : (B, T_tok, H_tok, W_tok) float32 tensor (unnormalised)
    """
    maps = []
    for idx in sorted(captured.keys()):
        out  = captured[idx].float()              # (B, S, D)
        norm = out.norm(dim=-1)                   # (B, S)
        B, S = norm.shape
        assert S == T_tok * H_tok * W_tok, (
            f"Block {idx}: expected S={T_tok*H_tok*W_tok}, got {S}"
        )
        maps.append(norm.view(B, T_tok, H_tok, W_tok))

    return torch.stack(maps, dim=0).mean(dim=0)   # (B, T_tok, H_tok, W_tok)


def compute_diff_saliency(captured_a, captured_b, T_tok, H_tok, W_tok):
    """
    Contrastive saliency: mean over blocks of ||out_A - out_B|| per token.

    Parameters
    ----------
    captured_a / captured_b : dict {block_idx -> (B, S, D)} from two passes
                              with different text conditions (same video, same xt)

    Returns
    -------
    saliency : (B, T_tok, H_tok, W_tok) float32 — prompt-sensitivity map
    """
    maps = []
    for idx in sorted(captured_a.keys()):
        if idx not in captured_b:
            continue
        out_a = captured_a[idx].float()   # (B, S, D)
        out_b = captured_b[idx].float()
        diff  = (out_a - out_b).norm(dim=-1)  # (B, S)
        B, S  = diff.shape
        assert S == T_tok * H_tok * W_tok
        maps.append(diff.view(B, T_tok, H_tok, W_tok))

    return torch.stack(maps, dim=0).mean(dim=0)   # (B, T_tok, H_tok, W_tok)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def saliency_to_colormap(saliency_hw, colormap="viridis"):
    """(H, W) float32 [0,1] → (H, W, 3) uint8 via matplotlib colormap."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap(colormap)
    rgb  = cmap(saliency_hw)[:, :, :3]
    return (rgb * 255).astype("uint8")


def blend_heatmap(frame_hwc, heatmap_hwc, alpha=0.5):
    """Alpha-blend a uint8 heatmap onto a uint8 original frame."""
    blended = (1.0 - alpha) * frame_hwc.astype("float32") \
            + alpha * heatmap_hwc.astype("float32")
    return blended.clip(0, 255).astype("uint8")


def add_border(frame_hwc, color_rgb, thickness=6):
    """Burn a solid border of `color_rgb` onto a (H,W,3) uint8 frame."""
    frame = frame_hwc.copy()
    frame[:thickness, :]  = color_rgb
    frame[-thickness:, :] = color_rgb
    frame[:, :thickness]  = color_rgb
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
    parser.add_argument("--prompt",           required=True,
                        help="Text prompt — directly drives cross-attention")
    parser.add_argument("--compare_prompt",   default=None,
                        help="If given, run a second pass with this prompt and "
                             "visualise ||out_A - out_B|| per token (diff mode). "
                             "Use a prompt describing completely different content "
                             "for the clearest contrast.")
    parser.add_argument("--negative_prompt",  default=None)
    parser.add_argument("--resolution",       default="432,432",
                        help="'H,W' resolution string (default: 432,432)")
    parser.add_argument("--num_latent_conditional_frames", type=int, default=2,
                        help="Conditioning frames used by the model (default: 2)")
    parser.add_argument("--timestep",         type=float, default=0.0,
                        help="Noise level for the DiT forward pass; "
                             "0 = clean latent (best for visualisation), "
                             "1 = pure noise (default: 0.0)")
    parser.add_argument("--layers",           type=int, nargs="*", default=None,
                        help="DiT block indices to average over. Default: all.")
    parser.add_argument("--alpha_blend",      type=float, default=0.5,
                        help="Heatmap opacity (0=original only, 1=heatmap only; "
                             "default: 0.5)")
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
    # 2. Load the full video at the model's required frame count
    # ------------------------------------------------------------------
    print(f"Loading video : {args.video_path}")
    video_uint8 = load_and_preprocess_video(
        video_path=args.video_path,
        resolution=[H, W],
        num_video_frames=required_pixel_frames,
    )
    print(f"Video shape   : {video_uint8.shape}")

    # ------------------------------------------------------------------
    # 3. Normalise and build data batch (T5 text embedding)
    # ------------------------------------------------------------------
    video_f32  = normalize_video(video_uint8)
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
    # 4. Offloading sequence
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
    print(f"Latent grid   : T={T_lat}, H={H_lat}, W={W_lat}")
    print(f"Prompt        : {args.prompt}")

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
    # 7. Helper: run one forward pass and return captured cross-attn outputs
    # ------------------------------------------------------------------
    n_blocks      = len(model.net.blocks)
    layer_indices = args.layers
    print(f"Registering hooks on {n_blocks} DiT blocks "
          f"(layers={layer_indices if layer_indices else 'all'})")

    def run_pass(cond):
        hooks, cap = register_crossattn_hooks(model.net, layer_indices)
        with torch.no_grad():
            _ = model.net(
                x_B_C_T_H_W=xt,
                timesteps_B_T=timesteps,
                **cond.to_dict(),
            )
        for h in hooks:
            h.remove()
        return cap

    # Pass A — main prompt
    print(f"Forward pass A  prompt: {args.prompt}")
    captured_a = run_pass(condition)
    print(f"  captured {len(captured_a)} blocks")

    # Pass B — compare prompt (optional)
    captured_b = None
    if args.compare_prompt:
        print(f"Building condition for compare prompt...")
        data_batch_b = inference._get_data_batch_input(
            video=video_bf16,
            prompt=args.compare_prompt,
            num_conditional_frames=args.num_latent_conditional_frames,
            negative_prompt=negative_prompt,
            use_neg_prompt=True,
        )
        data_batch_b["video"]            = video_bf16
        data_batch_b[IS_PREPROCESSED_KEY] = True
        with torch.no_grad():
            _, _, condition_b = model.get_data_and_condition(data_batch_b)
        condition_b = condition_b.edit_for_inference(
            is_cfg_conditional=True,
            num_conditional_frames=args.num_latent_conditional_frames,
        )
        print(f"Forward pass B  prompt: {args.compare_prompt}")
        captured_b = run_pass(condition_b)
        print(f"  captured {len(captured_b)} blocks")

    # ------------------------------------------------------------------
    # 8. Compute per-token saliency maps
    # ------------------------------------------------------------------
    print("Computing saliency maps...")

    sample_S = next(iter(captured_a.values())).shape[1]
    T_tok, H_tok, W_tok = detect_token_dims(sample_S, T_lat, H_lat, W_lat)
    print(f"Token grid    : T={T_tok}, H={H_tok}, W={W_tok}  "
          f"(spatial patch ~{H_lat // H_tok}x{W_lat // W_tok}, "
          f"temporal ratio {T_lat // T_tok}x)")

    selected_keys = sorted(args.layers) if args.layers else sorted(captured_a.keys())
    saliency_all  = compute_saliency(
        {k: captured_a[k] for k in selected_keys if k in captured_a},
        T_tok, H_tok, W_tok,
    )

    all_keys  = sorted(captured_a.keys())
    mid       = len(all_keys) // 2
    early_sal = compute_saliency(
        {k: captured_a[k] for k in all_keys[:mid]}, T_tok, H_tok, W_tok,
    )
    late_sal  = compute_saliency(
        {k: captured_a[k] for k in all_keys[mid:]}, T_tok, H_tok, W_tok,
    )

    # Diff saliency (only when compare_prompt was provided)
    diff_sal = None
    if captured_b is not None:
        diff_sal = compute_diff_saliency(
            {k: captured_a[k] for k in selected_keys if k in captured_a},
            {k: captured_b[k] for k in selected_keys if k in captured_b},
            T_tok, H_tok, W_tok,
        )
        print(f"Diff saliency range (before normalise): "
              f"[{diff_sal.min():.4f}, {diff_sal.max():.4f}]")

    def upsample_to_pixel(sal_tok):
        return F.interpolate(
            sal_tok.view(1, T_tok, H_tok, W_tok).permute(1, 0, 2, 3),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze(1)

    sal_px_all   = upsample_to_pixel(saliency_all)
    sal_px_early = upsample_to_pixel(early_sal)
    sal_px_late  = upsample_to_pixel(late_sal)

    def normalise(sal):
        s_min, s_max = sal.min(), sal.max()
        return (sal - s_min) / (s_max - s_min + 1e-8), s_min, s_max

    sal_norm_all,   s_min_all,   s_max_all   = normalise(sal_px_all)
    sal_norm_early, s_min_early, s_max_early = normalise(sal_px_early)
    sal_norm_late,  s_min_late,  s_max_late  = normalise(sal_px_late)

    px_per_tok   = 4 * (T_lat // T_tok)
    num_cond_tok = args.num_latent_conditional_frames * (T_tok // T_lat) \
                   if T_tok < T_lat else args.num_latent_conditional_frames

    # ------------------------------------------------------------------
    # Per-token-frame saliency statistics
    # ------------------------------------------------------------------
    print(f"\nPer-token-frame cross-attn saliency (norm, t={t}, layers={selected_keys}):")
    header = f"  {'tok':>4}  {'norm_mean':>10}  {'norm_max':>10}"
    if diff_sal is not None:
        sal_px_diff = upsample_to_pixel(diff_sal)
        sal_norm_diff, s_min_diff, s_max_diff = normalise(sal_px_diff)
        header += f"  {'diff_mean':>10}  {'diff_max':>10}"
    header += "  type"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t_tok in range(T_tok):
        frame_type = "COND" if t_tok < num_cond_tok else "GEN"
        row = (f"  {t_tok:>4}  {sal_px_all[t_tok].mean():>10.3f}  "
               f"{sal_px_all[t_tok].max():>10.3f}")
        if diff_sal is not None:
            row += (f"  {sal_px_diff[t_tok].mean():>10.3f}  "
                    f"{sal_px_diff[t_tok].max():>10.3f}")
        row += f"  {frame_type}"
        print(row)

    # ------------------------------------------------------------------
    # Build annotated video frames
    # ------------------------------------------------------------------
    pixel_frames_np = video_uint8[0].permute(1, 2, 3, 0).numpy()  # (T_px, H, W, C)
    T_px = pixel_frames_np.shape[0]

    BORDER_COND = [0, 200, 80]    # green  = conditioning
    BORDER_GEN  = [255, 140, 0]   # orange = generation

    def build_frame_list(sal_norm):
        heatmap_f, saliency_f = [], []
        for t_px in range(T_px):
            t_tok_idx = min(t_px // px_per_tok, T_tok - 1)
            sal_hw    = sal_norm[t_tok_idx].numpy()
            cmap_hwc  = saliency_to_colormap(sal_hw, args.colormap)
            orig_hwc  = pixel_frames_np[t_px]
            blended   = blend_heatmap(orig_hwc, cmap_hwc, alpha=args.alpha_blend)
            border    = BORDER_COND if t_tok_idx < num_cond_tok else BORDER_GEN
            blended   = add_border(blended, border)
            heatmap_f.append(blended)
            saliency_f.append(cmap_hwc)
        return heatmap_f, saliency_f

    heatmap_all,   saliency_frames = build_frame_list(sal_norm_all)
    heatmap_early, _               = build_frame_list(sal_norm_early)
    heatmap_late,  _               = build_frame_list(sal_norm_late)

    saliency_out = torch.from_numpy(
        np.stack([sal_norm_all[min(t_px // px_per_tok, T_tok - 1)].numpy()
                  for t_px in range(T_px)], axis=0)
    )

    diff_out = None
    if diff_sal is not None:
        heatmap_diff, _ = build_frame_list(sal_norm_diff)
        diff_out = torch.from_numpy(
            np.stack([sal_norm_diff[min(t_px // px_per_tok, T_tok - 1)].numpy()
                      for t_px in range(T_px)], axis=0)
        )

    # ------------------------------------------------------------------
    # 9. Save outputs (rank 0 only)
    # ------------------------------------------------------------------
    if local_rank == 0:
        out_dir = WM_ROOT / "attack" / "outputs" / f"crossattn_heatmap_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(saliency_out, out_dir / "crossattn_saliency.pt")

        def frames_to_tensor(frames_list):
            return torch.from_numpy(np.stack(frames_list, axis=0))

        torchvision.io.write_video(str(out_dir / "heatmap.mp4"),
                                   frames_to_tensor(heatmap_all),    fps=16)
        torchvision.io.write_video(str(out_dir / "saliency_only.mp4"),
                                   frames_to_tensor(saliency_frames), fps=16)
        torchvision.io.write_video(str(out_dir / "heatmap_early.mp4"),
                                   frames_to_tensor(heatmap_early),  fps=16)
        torchvision.io.write_video(str(out_dir / "heatmap_late.mp4"),
                                   frames_to_tensor(heatmap_late),   fps=16)

        if diff_out is not None:
            torch.save(diff_out, out_dir / "prompt_diff.pt")
            torchvision.io.write_video(str(out_dir / "prompt_diff.mp4"),
                                       frames_to_tensor(heatmap_diff), fps=16)

        print(f"\nSaved to: {out_dir}")
        print(f"  heatmap.mp4      : norm saliency  [{s_min_all:.2f}, {s_max_all:.2f}]  "
              f"(green=COND, orange=GEN)")
        print(f"  heatmap_early.mp4: blocks {all_keys[:mid]}  [{s_min_early:.2f}, {s_max_early:.2f}]")
        print(f"  heatmap_late.mp4 : blocks {all_keys[mid:]}  [{s_min_late:.2f}, {s_max_late:.2f}]")
        if diff_out is not None:
            print(f"  prompt_diff.mp4  : ||out_A-out_B||  [{s_min_diff:.2f}, {s_max_diff:.2f}]")
            print(f"    prompt A: {args.prompt}")
            print(f"    prompt B: {args.compare_prompt}")
            flat_frac = (sal_px_diff < 0.05 * sal_px_diff.max()).float().mean().item()
            print(f"    {flat_frac*100:.1f}% of tokens have diff < 5% of max  "
                  f"({'very flat → model behaviour' if flat_frac > 0.8 else 'spatial structure present'})")
        print(f"\nToken grid         : T={T_tok}, H={H_tok}, W={W_tok}")
        print(f"Conditioning tokens: 0–{num_cond_tok-1}  Generation tokens: {num_cond_tok}–{T_tok-1}")
        print(f"Pixels per token   : {px_per_tok}")


if __name__ == "__main__":
    main()


"""
# norm mode (single pass)
python cosmos-predict2.5/scripts/compute_crossattn_heatmap.py \\
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \\
    --experiment_name predict2_video2world_training_2b_libero_480 \\
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/8fbc6188fa2f2e4ab585dc6aac3edd0e9d8a3670/model.pt \\
    --prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate" \\
    --resolution 432,432 \\
    --num_latent_conditional_frames 2 \\
    --timestep 0.0 \\
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \\
    --offload_diffusion_model --offload_tokenizer --offload_text_encoder

# diff mode (two passes — recommended for diagnosing prompt sensitivity)
python cosmos-predict2.5/scripts/compute_crossattn_heatmap.py \\
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \\
    --experiment_name predict2_video2world_training_2b_libero_480 \\
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/8fbc6188fa2f2e4ab585dc6aac3edd0e9d8a3670/model.pt \\
    --prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate" \\
    --compare_prompt "a cat sitting on a table in a kitchen" \\
    --resolution 432,432 \\
    --num_latent_conditional_frames 2 \\
    --timestep 0.0 \\
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \\
    --offload_diffusion_model --offload_tokenizer --offload_text_encoder
"""
