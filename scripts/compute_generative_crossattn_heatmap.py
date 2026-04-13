"""
Measure cross-attention during the full Video2World denoising loop.

Rather than a single forward pass on a fixed video, this script runs the
complete diffusion sampling loop and accumulates cross-attention outputs
from block.cross_attn at every denoising step.  The mean activation across
all steps gives a true generative attention map — showing which spatial-
temporal regions of the generated video are most driven by the text prompt
while the model is actually deciding what to generate.

Key difference from compute_crossattn_heatmap.py:
  - Runs full generation (~35 denoising steps by default)
  - Hooks accumulate across ALL conditional denoising steps
  - Heatmap is overlaid on the GENERATED video frames (not the input)
  - CFG uses two net calls per step (conditional then unconditional);
    only the conditional fires are accumulated

With --compare_prompt, a second generation is run and the heatmap shows
||mean_attn_A - mean_attn_B|| per token — the regions where the two prompts
drove the model to attend differently during generation.

Saved artefacts (attack/outputs/gen_crossattn_<timestamp>/):
  - generated.mp4          : the generated video (context + future frames)
  - heatmap.mp4            : generated video with cross-attn heatmap overlay
  - saliency_only.mp4      : raw coloured heatmap
  - heatmap_early.mp4      : first-half DiT blocks only
  - heatmap_late.mp4       : second-half DiT blocks only
  - crossattn_saliency.pt  : saliency tensor (T_px, H_px, W_px) float32
  [with --compare_prompt]
  - generated_B.mp4        : generated video for the compare prompt
  - prompt_diff.mp4        : ||attn_A - attn_B|| heatmap
  - prompt_diff.pt         : raw diff saliency tensor

Usage:
    python cosmos-predict2.5/scripts/compute_generative_crossattn_heatmap.py \
        --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
        --experiment_name predict2_video2world_training_2b_libero_480 \
        --ckpt_path /path/to/model.pt \
        --prompt "Use the franka robot arm to pick up the black bowl" \
        --resolution 432,432 \
        --num_latent_conditional_frames 2 \
        --num_steps 35 \
        --guidance 7 \
        --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
        [--compare_prompt "a cat sitting on a table"] \
        [--layers 14 15 16 17 18] \
        [--alpha_blend 0.5] \
        [--offload_diffusion_model] [--offload_tokenizer] [--offload_text_encoder]
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
# Accumulating hooks
# ---------------------------------------------------------------------------

def make_accumulating_hooks(net, layer_indices, store_full_output=False):
    """
    Register forward hooks on block.cross_attn that accumulate across every
    denoising step.

    CFG runs two net forward passes per denoising step:
      1. conditional pass  (with prompt)    ← we accumulate this
      2. unconditional pass (negative prompt) ← we discard this

    The hook tracks a per-block fire counter.  Even-indexed fires (0, 2, 4…)
    are the conditional pass; odd-indexed fires are the unconditional pass.

    Parameters
    ----------
    net               : DiT net with .blocks list
    layer_indices     : list of int block indices to hook, or None for all
    store_full_output : if True, accumulate full (B, S, D) tensors rather than
                        just the per-token L2 norm.  Required for diff mode
                        but uses much more CPU RAM (~140 MB per block).

    Returns
    -------
    hooks        : list of hook handles
    accumulated  : dict {block_idx -> running sum tensor} filled during generation
    fire_counts  : dict {block_idx -> int} number of conditional fires per block
    """
    hooks       = []
    accumulated = {}
    fire_counts = {}
    raw_counts  = {}   # total fires (conditional + unconditional)

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
                raw_counts[i] = raw_counts.get(i, 0) + 1
                # Only accumulate on conditional (even-indexed) fires
                if raw_counts[i] % 2 != 1:   # 1-indexed: 1,3,5… are conditional
                    return
                out = output[0] if isinstance(output, tuple) else output
                out = out.detach().float().cpu()  # (B, S, D)

                if store_full_output:
                    val = out
                else:
                    val = out.norm(dim=-1)          # (B, S)

                if i not in accumulated:
                    accumulated[i]  = val
                    fire_counts[i]  = 1
                else:
                    accumulated[i] += val
                    fire_counts[i] += 1
            return _hook

        h = block.cross_attn.register_forward_hook(_make_hook(idx))
        hooks.append(h)

    return hooks, accumulated, fire_counts


def finalise_accumulated(accumulated, fire_counts):
    """Divide each running sum by its fire count to get the per-step mean."""
    return {i: accumulated[i] / fire_counts[i] for i in accumulated}


# ---------------------------------------------------------------------------
# Token dimension detection (shared with other heatmap scripts)
# ---------------------------------------------------------------------------

def detect_token_dims(S, T_lat, H_lat, W_lat):
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
        f"Cannot factor S={S} from latent dims (T={T_lat}, H={H_lat}, W={W_lat})"
    )


# ---------------------------------------------------------------------------
# Saliency from accumulated data
# ---------------------------------------------------------------------------

def compute_norm_saliency(mean_accumulated, T_tok, H_tok, W_tok):
    """
    mean_accumulated: dict {block_idx -> (B, S)} mean norm per token.
    Returns (B, T_tok, H_tok, W_tok).
    """
    maps = []
    for idx in sorted(mean_accumulated.keys()):
        norm = mean_accumulated[idx].float()  # (B, S)
        B, S = norm.shape
        assert S == T_tok * H_tok * W_tok
        maps.append(norm.view(B, T_tok, H_tok, W_tok))
    return torch.stack(maps, dim=0).mean(dim=0)


def compute_diff_saliency(mean_acc_a, mean_acc_b, T_tok, H_tok, W_tok):
    """
    mean_acc_a/b: dict {block_idx -> (B, S, D)} mean full output.
    Returns (B, T_tok, H_tok, W_tok) diff norm.
    """
    maps = []
    for idx in sorted(mean_acc_a.keys()):
        if idx not in mean_acc_b:
            continue
        out_a = mean_acc_a[idx].float()   # (B, S, D)
        out_b = mean_acc_b[idx].float()
        diff  = (out_a - out_b).norm(dim=-1)  # (B, S)
        B, S  = diff.shape
        assert S == T_tok * H_tok * W_tok
        maps.append(diff.view(B, T_tok, H_tok, W_tok))
    return torch.stack(maps, dim=0).mean(dim=0)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def saliency_to_colormap(saliency_hw, colormap="viridis"):
    import matplotlib.cm as cm
    cmap = cm.get_cmap(colormap)
    rgb  = cmap(saliency_hw)[:, :, :3]
    return (rgb * 255).astype("uint8")


def blend_heatmap(frame_hwc, heatmap_hwc, alpha=0.5):
    blended = (1.0 - alpha) * frame_hwc.astype("float32") \
            + alpha * heatmap_hwc.astype("float32")
    return blended.clip(0, 255).astype("uint8")


def add_border(frame_hwc, color_rgb, thickness=6):
    frame = frame_hwc.copy()
    frame[:thickness, :]  = color_rgb
    frame[-thickness:, :] = color_rgb
    frame[:, :thickness]  = color_rgb
    frame[:, -thickness:] = color_rgb
    return frame


def latent_to_video_frames(video_tensor):
    """(1, C, T, H, W) float [-1,1] → (T, H, W, C) uint8 numpy."""
    t = video_tensor[0].cpu().float()
    t = ((t.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    return t.permute(1, 2, 3, 0).numpy()


def build_heatmap_frames(sal_norm, pixel_frames_np, T_tok, px_per_tok,
                          num_cond_tok, colormap, alpha_blend):
    """Overlay normalised saliency on pixel frames with COND/GEN border."""
    T_px = pixel_frames_np.shape[0]
    BORDER_COND = [0, 200, 80]
    BORDER_GEN  = [255, 140, 0]
    heatmap_f, saliency_f = [], []
    for t_px in range(T_px):
        t_tok_idx = min(t_px // px_per_tok, T_tok - 1)
        sal_hw    = sal_norm[t_tok_idx].numpy()
        cmap_hwc  = saliency_to_colormap(sal_hw, colormap)
        orig_hwc  = pixel_frames_np[t_px]
        blended   = blend_heatmap(orig_hwc, cmap_hwc, alpha=alpha_blend)
        border    = BORDER_COND if t_tok_idx < num_cond_tok else BORDER_GEN
        blended   = add_border(blended, border)
        heatmap_f.append(blended)
        saliency_f.append(cmap_hwc)
    return heatmap_f, saliency_f


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def run_generation(inference, data_batch, args, accumulated_hooks_fn):
    """
    Run the full denoising loop, accumulating cross-attn during generation.

    accumulated_hooks_fn() should register hooks and return
    (hooks, accumulated, fire_counts).  Hooks are removed after generation.

    Returns (sample_latent, mean_accumulated_dict).
    """
    model = inference.model
    hooks, accumulated, fire_counts = accumulated_hooks_fn()
    print(f"  Registered {len(hooks)} cross-attn hooks")

    if getattr(model.config, "use_lora", False):
        generate_fn = model.generate_samples_from_batch_lora
    else:
        generate_fn = model.generate_samples_from_batch

    with torch.no_grad():
        sample = generate_fn(
            data_batch,
            n_sample=1,
            guidance=args.guidance,
            seed=args.seed,
            is_negative_prompt=True,
            num_steps=args.num_steps,
        )

    for h in hooks:
        h.remove()

    total_fires = sum(fire_counts.values())
    steps_fired = total_fires // max(len(fire_counts), 1)
    print(f"  Hook fired {steps_fired} conditional steps across {len(fire_counts)} blocks")

    mean_acc = finalise_accumulated(accumulated, fire_counts)
    return sample, mean_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path",       required=True)
    parser.add_argument("--experiment_name",  required=True)
    parser.add_argument("--ckpt_path",        required=True)
    parser.add_argument("--prompt",           required=True,
                        help="Text prompt used for generation and attention measurement")
    parser.add_argument("--compare_prompt",   default=None,
                        help="Run a second generation with this prompt and output "
                             "a contrastive ||attn_A - attn_B|| heatmap")
    parser.add_argument("--negative_prompt",  default=None)
    parser.add_argument("--resolution",       default="432,432")
    parser.add_argument("--num_latent_conditional_frames", type=int, default=2)
    parser.add_argument("--num_steps",        type=int,   default=35)
    parser.add_argument("--guidance",         type=float, default=7.0)
    parser.add_argument("--seed",             type=int,   default=1)
    parser.add_argument("--layers",           type=int, nargs="*", default=None,
                        help="DiT block indices to accumulate. Default: all.")
    parser.add_argument("--alpha_blend",      type=float, default=0.5)
    parser.add_argument("--colormap",         default="viridis")
    parser.add_argument("--offload_diffusion_model", action="store_true")
    parser.add_argument("--offload_text_encoder",    action="store_true")
    parser.add_argument("--offload_tokenizer",       action="store_true")
    parser.add_argument("--context_parallel_size",   type=int, default=1)
    parser.add_argument("--config_file",
                        default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    H, W = [int(x) for x in args.resolution.split(",")]
    diff_mode = args.compare_prompt is not None

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
        offload_diffusion_model=args.offload_diffusion_model,
        offload_text_encoder=args.offload_text_encoder,
        offload_tokenizer=args.offload_tokenizer,
    )
    model = inference.model
    state_t               = model.config.state_t
    required_pixel_frames = (state_t - 1) * 4 + 1
    frames_to_extract     = 4 * (args.num_latent_conditional_frames - 1) + 1
    print(f"Model state_t : {state_t}  →  required pixel frames: {required_pixel_frames}")

    # ------------------------------------------------------------------
    # 2. Load video and build padded input
    #    (eval_diffusion.py layout: last frames_to_extract real frames +
    #     repeated last frame to fill required_pixel_frames)
    # ------------------------------------------------------------------
    print(f"Loading video : {args.video_path}")
    # Load enough frames to extract the conditioning context
    num_load_frames = (args.num_latent_conditional_frames - 1) * 4 + 1
    video_uint8 = load_and_preprocess_video(
        video_path=args.video_path,
        resolution=[H, W],
        num_video_frames=num_load_frames,
    )
    print(f"Video shape   : {video_uint8.shape}")

    video_f32      = normalize_video(video_uint8)
    context_frames = video_f32[:, :, -frames_to_extract:, :, :]
    last_frame     = context_frames[:, :, -1:, :, :]
    padding        = required_pixel_frames - frames_to_extract
    video_padded   = torch.cat(
        [context_frames, last_frame.repeat(1, 1, padding, 1, 1)], dim=2
    )
    video_bf16 = video_padded.to(device=torch.cuda.current_device(),
                                  dtype=torch.bfloat16)

    # ------------------------------------------------------------------
    # 3. Build data batches
    # ------------------------------------------------------------------
    negative_prompt = args.negative_prompt or _DEFAULT_NEGATIVE_PROMPT

    def make_batch(prompt):
        db = inference._get_data_batch_input(
            video=video_bf16,
            prompt=prompt,
            num_conditional_frames=args.num_latent_conditional_frames,
            negative_prompt=negative_prompt,
            use_neg_prompt=True,
        )
        db["video"]            = video_bf16
        db[IS_PREPROCESSED_KEY] = True
        return db

    data_batch_a = make_batch(args.prompt)
    data_batch_b = make_batch(args.compare_prompt) if diff_mode else None

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
    # 5. Generation A — main prompt
    # ------------------------------------------------------------------
    print(f"\nGeneration A  ({args.num_steps} steps, guidance={args.guidance})")
    print(f"  Prompt: {args.prompt}")

    def hooks_fn_a():
        return make_accumulating_hooks(
            model.net, args.layers, store_full_output=diff_mode
        )

    sample_a, mean_acc_a = run_generation(inference, data_batch_a, args, hooks_fn_a)

    print("Decoding generation A...")
    with torch.no_grad():
        if isinstance(sample_a, list):
            video_out_a = torch.cat([model.decode(s) for s in sample_a], dim=3)
        else:
            video_out_a = model.decode(sample_a)

    # ------------------------------------------------------------------
    # 6. Generation B — compare prompt (optional)
    # ------------------------------------------------------------------
    sample_b    = None
    mean_acc_b  = None
    video_out_b = None

    if diff_mode:
        print(f"\nGeneration B  ({args.num_steps} steps, guidance={args.guidance})")
        print(f"  Prompt: {args.compare_prompt}")

        def hooks_fn_b():
            return make_accumulating_hooks(
                model.net, args.layers, store_full_output=True
            )

        sample_b, mean_acc_b = run_generation(inference, data_batch_b, args, hooks_fn_b)

        print("Decoding generation B...")
        with torch.no_grad():
            if isinstance(sample_b, list):
                video_out_b = torch.cat([model.decode(s) for s in sample_b], dim=3)
            else:
                video_out_b = model.decode(sample_b)

    # ------------------------------------------------------------------
    # 7. Derive token grid dimensions from first captured block
    # ------------------------------------------------------------------
    sample_block_out = next(iter(mean_acc_a.values()))
    # For norm mode: (B, S); for full mode: (B, S, D)
    S = sample_block_out.shape[1]

    # Get latent shape from the decoded video to infer T_lat, H_lat, W_lat
    # video_out shape: (1, C, T_px_out, H, W); latent has 4x temporal compression
    T_px_out  = video_out_a.shape[2]
    # Derive latent dims from state_t
    T_lat = state_t
    H_lat = H // 8   # WAN VAE spatial compression factor
    W_lat = W // 8

    T_tok, H_tok, W_tok = detect_token_dims(S, T_lat, H_lat, W_lat)
    px_per_tok = 4 * (T_lat // T_tok)
    num_cond_tok = args.num_latent_conditional_frames * (T_tok // T_lat) \
                   if T_tok < T_lat else args.num_latent_conditional_frames

    print(f"\nToken grid    : T={T_tok}, H={H_tok}, W={W_tok}")
    print(f"Pixels per token frame : {px_per_tok}")
    print(f"Conditioning tokens    : 0–{num_cond_tok-1}")

    # ------------------------------------------------------------------
    # 8. Compute saliency maps
    # ------------------------------------------------------------------
    def upsample(sal_tok):
        return F.interpolate(
            sal_tok.view(1, T_tok, H_tok, W_tok).permute(1, 0, 2, 3),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze(1)

    def normalise(sal):
        s_min, s_max = sal.min(), sal.max()
        return (sal - s_min) / (s_max - s_min + 1e-8), s_min, s_max

    selected_keys = sorted(args.layers) if args.layers else sorted(mean_acc_a.keys())
    all_keys      = sorted(mean_acc_a.keys())
    mid           = len(all_keys) // 2

    if diff_mode:
        # In diff mode mean_acc contains full (B, S, D) — compute norm first
        def norm_from_full(acc, keys):
            return {k: acc[k].norm(dim=-1) for k in keys if k in acc}

        sal_all   = compute_norm_saliency(norm_from_full(mean_acc_a, selected_keys),
                                          T_tok, H_tok, W_tok)
        sal_early = compute_norm_saliency(norm_from_full(mean_acc_a, all_keys[:mid]),
                                          T_tok, H_tok, W_tok)
        sal_late  = compute_norm_saliency(norm_from_full(mean_acc_a, all_keys[mid:]),
                                          T_tok, H_tok, W_tok)
        sal_diff  = compute_diff_saliency(
            {k: mean_acc_a[k] for k in selected_keys if k in mean_acc_a},
            {k: mean_acc_b[k] for k in selected_keys if k in mean_acc_b},
            T_tok, H_tok, W_tok,
        )
    else:
        sal_all   = compute_norm_saliency(
            {k: mean_acc_a[k] for k in selected_keys if k in mean_acc_a},
            T_tok, H_tok, W_tok,
        )
        sal_early = compute_norm_saliency(
            {k: mean_acc_a[k] for k in all_keys[:mid]}, T_tok, H_tok, W_tok,
        )
        sal_late  = compute_norm_saliency(
            {k: mean_acc_a[k] for k in all_keys[mid:]}, T_tok, H_tok, W_tok,
        )
        sal_diff  = None

    sal_px_all,   s_min_all,   s_max_all   = normalise(upsample(sal_all))
    sal_px_early, s_min_early, s_max_early = normalise(upsample(sal_early))
    sal_px_late,  s_min_late,  s_max_late  = normalise(upsample(sal_late))
    sal_px_diff,  s_min_diff,  s_max_diff  = (normalise(upsample(sal_diff))
                                               if sal_diff is not None
                                               else (None, None, None))

    # Per-token stats
    print(f"\nPer-token-frame saliency (mean over {args.num_steps} generation steps):")
    header = f"  {'tok':>4}  {'norm_mean':>10}  {'norm_max':>10}"
    if sal_px_diff is not None:
        header += f"  {'diff_mean':>10}  {'diff_max':>10}"
    header += "  type"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for t_tok in range(T_tok):
        frame_type = "COND" if t_tok < num_cond_tok else "GEN"
        row = (f"  {t_tok:>4}  {sal_px_all[t_tok].mean():>10.3f}  "
               f"{sal_px_all[t_tok].max():>10.3f}")
        if sal_px_diff is not None:
            row += (f"  {sal_px_diff[t_tok].mean():>10.3f}  "
                    f"{sal_px_diff[t_tok].max():>10.3f}")
        row += f"  {frame_type}"
        print(row)

    # ------------------------------------------------------------------
    # 9. Build video frame arrays for overlay
    # ------------------------------------------------------------------
    gen_frames_a = latent_to_video_frames(video_out_a)   # (T_px_out, H, W, C)
    T_px_gen     = gen_frames_a.shape[0]

    def overlay(sal_norm, pixel_frames_np):
        return build_heatmap_frames(
            sal_norm, pixel_frames_np,
            T_tok, px_per_tok, num_cond_tok,
            args.colormap, args.alpha_blend,
        )

    heatmap_all,   saliency_frames = overlay(sal_px_all,   gen_frames_a)
    heatmap_early, _               = overlay(sal_px_early, gen_frames_a)
    heatmap_late,  _               = overlay(sal_px_late,  gen_frames_a)

    saliency_out = torch.from_numpy(
        np.stack([sal_px_all[min(t // px_per_tok, T_tok - 1)].numpy()
                  for t in range(T_px_gen)], axis=0)
    )

    heatmap_diff_frames = None
    diff_out            = None
    gen_frames_b        = None
    if sal_px_diff is not None:
        gen_frames_b = latent_to_video_frames(video_out_b)
        heatmap_diff_frames, _ = overlay(sal_px_diff, gen_frames_b)
        diff_out = torch.from_numpy(
            np.stack([sal_px_diff[min(t // px_per_tok, T_tok - 1)].numpy()
                      for t in range(gen_frames_b.shape[0])], axis=0)
        )

    # ------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------
    if local_rank == 0:
        out_dir = WM_ROOT / "attack" / "outputs" / f"gen_crossattn_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)

        def to_tensor(frames):
            return torch.from_numpy(np.stack(frames, axis=0))

        def write(name, frames):
            torchvision.io.write_video(str(out_dir / name), to_tensor(frames), fps=24)

        write("generated.mp4",    list(gen_frames_a))
        write("heatmap.mp4",      heatmap_all)
        write("saliency_only.mp4", saliency_frames)
        write("heatmap_early.mp4", heatmap_early)
        write("heatmap_late.mp4",  heatmap_late)
        torch.save(saliency_out, out_dir / "crossattn_saliency.pt")

        if diff_out is not None:
            write("generated_B.mp4",  list(gen_frames_b))
            write("prompt_diff.mp4",  heatmap_diff_frames)
            torch.save(diff_out, out_dir / "prompt_diff.pt")

        print(f"\nSaved to: {out_dir}")
        print(f"  heatmap.mp4      : norm saliency  [{s_min_all:.2f}, {s_max_all:.2f}]  "
              f"(green=COND, orange=GEN)")
        print(f"  heatmap_early.mp4: blocks {all_keys[:mid]}  [{s_min_early:.2f}, {s_max_early:.2f}]")
        print(f"  heatmap_late.mp4 : blocks {all_keys[mid:]}  [{s_min_late:.2f}, {s_max_late:.2f}]")
        if diff_out is not None:
            flat_frac = (upsample(sal_diff) < 0.05 * upsample(sal_diff).max()).float().mean().item()
            print(f"  prompt_diff.mp4  : [{s_min_diff:.2f}, {s_max_diff:.2f}]  "
                  f"({flat_frac*100:.1f}% tokens < 5% of max → "
                  f"{'very flat' if flat_frac > 0.8 else 'spatial structure present'})")
            print(f"    prompt A: {args.prompt}")
            print(f"    prompt B: {args.compare_prompt}")


if __name__ == "__main__":
    main()


"""
# single prompt
python cosmos-predict2.5/scripts/compute_generative_crossattn_heatmap.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --experiment_name predict2_video2world_training_2b_libero_480 \
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/8fbc6188fa2f2e4ab585dc6aac3edd0e9d8a3670/model.pt \
    --prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate" \
    --resolution 432,432 \
    --num_latent_conditional_frames 2 \
    --num_steps 35 --guidance 7 --seed 1 \
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --offload_diffusion_model --offload_tokenizer --offload_text_encoder

# contrastive diff mode
python cosmos-predict2.5/scripts/compute_generative_crossattn_heatmap.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --experiment_name predict2_video2world_training_2b_libero_480 \
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/a42439ce5f10e438aa8ba6d7fd1657a206e0d13f/model.pt \
    --prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate" \
    --compare_prompt "Use the franka robot arm to pick up the cookie box and place it on the stove" \
    --resolution 432,432 \
    --num_latent_conditional_frames 2 \
    --num_steps 35 --guidance 7 --seed 1 \
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --offload_diffusion_model --offload_tokenizer --offload_text_encoder
"""
