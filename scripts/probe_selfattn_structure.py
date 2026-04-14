"""
Probe the structural properties of self-attention in the Cosmos Video2World DiT.

Captures Q and K tensors from selected DiT blocks by temporarily wrapping
block.self_attn.compute_attention (instance-level monkey-patch, no source
modification needed).  For each probed block, the script reports:

  1. Q / K / output tensor shapes and dtypes
  2. Per-temporal-frame Q-norm and K-norm (mean over heads + spatial tokens)
  3. T_tok × T_tok frame-to-frame attention score matrix:
       frame_attn[qi, kj]  = softmax_kj( mean_head( Q̄[qi] · K̄[kj] / √D ) )
     where Q̄[qi] is the spatial-mean Q vector for token-frame qi.
     This is an approximation of the true frame-level aggregate
     (which would require materialising the full S×S matrix), but it directly
     reveals the temporal structure: which source frames each query frame attends
     to most strongly.
  4. Per-frame output L2-norm (from the output hook on block.self_attn)

After all blocks, a summary frame-to-frame matrix averaged across all probed
blocks is printed, along with four aggregate attention-flow numbers:
  COND→COND, GEN→COND, COND→GEN, GEN→GEN.

No video rendering is performed.

Saved artefacts (attack/outputs/selfattn_structure_<timestamp>/):
  selfattn_structure.pt  — dict with:
      token_grid      : {"T_tok", "H_tok", "W_tok"}
      num_cond_tok    : int
      latent_shape    : (B, C, T, H, W)
      mean_frame_attn : (T_tok, T_tok) float32  — average across probed blocks
      blocks          : {block_idx: {"frame_attn": (T,T), "q_norm": (T,),
                                     "k_norm": (T,), "out_norm": (T,)}}
      [--save_qk only]
      blocks[idx]["q"] : (B, S, H, D) float32
      blocks[idx]["k"] : (B, S, H, D) float32

Usage:
    python cosmos-predict2.5/scripts/probe_selfattn_structure.py \\
        --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \\
        --experiment_name predict2_video2world_training_2b_libero_480 \\
        --ckpt_path /path/to/model.pt \\
        --prompt "Use the franka robot arm to pick up the black bowl" \\
        --resolution 432,432 \\
        --num_latent_conditional_frames 2 \\
        --timestep 0.0 \\
        --layers 0 5 10 15 20 25 \\
        --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \\
        --offload_diffusion_model --offload_tokenizer --offload_text_encoder
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
# Q / K capture via instance-level compute_attention wrap
# ---------------------------------------------------------------------------

def wrap_selfattn_qk(net, layer_indices=None):
    """
    Temporarily replace block.self_attn.compute_attention on each selected
    block with a closure that records (q, k) before delegating to the original.

    Works by writing to the instance __dict__, so:
      • The class method is untouched.
      • 'self' is NOT auto-prepended when the instance attr is called
        (non-data descriptor rule), so the wrapper signature (q, k, v, **kw)
        matches the real call site.
      • orig_fn is a bound method, so orig_fn(q, k, v, **kw) works correctly.

    Returns
    -------
    captured_qk : dict {block_idx -> (q, k)}
                  q, k : (B, S, H, D) float32 cpu tensors
                  Populated after a forward pass.
    restore     : callable — removes all patches.
    """
    n_blocks = len(net.blocks)
    indices  = layer_indices if layer_indices is not None else list(range(n_blocks))

    captured_qk = {}
    _patches    = {}   # {idx: (attn_module, original_bound_method)}

    for idx in indices:
        if idx >= n_blocks:
            continue
        block = net.blocks[idx]
        if not hasattr(block, "self_attn"):
            continue

        attn    = block.self_attn
        orig_fn = attn.compute_attention  # bound method (has self baked in)

        def _make_wrapper(i, fn):
            def _wrapper(q, k, v, **kw):
                captured_qk[i] = (
                    q.detach().cpu().float(),
                    k.detach().cpu().float(),
                )
                return fn(q, k, v, **kw)
            return _wrapper

        # Write to instance dict — shadows the class method for this instance only
        attn.__dict__["compute_attention"] = _make_wrapper(idx, orig_fn)
        _patches[idx] = (attn, orig_fn)

    def _restore():
        for _idx, (attn_mod, _orig) in _patches.items():
            attn_mod.__dict__.pop("compute_attention", None)

    return captured_qk, _restore


# ---------------------------------------------------------------------------
# Self-attn output hook (same approach as compute_selfattn_heatmap.py)
# ---------------------------------------------------------------------------

def register_selfattn_output_hooks(net, layer_indices=None):
    """Register forward hooks on block.self_attn to capture the output tensor."""
    n_blocks     = len(net.blocks)
    indices      = layer_indices if layer_indices is not None else list(range(n_blocks))
    hooks        = []
    captured_out = {}

    for idx in indices:
        if idx >= n_blocks:
            continue
        block = net.blocks[idx]
        if not hasattr(block, "self_attn"):
            continue

        def _make_hook(i):
            def _hook(_m, _inp, output):
                out = output[0] if isinstance(output, tuple) else output
                captured_out[i] = out.detach().cpu().float()   # (B, S, D)
            return _hook

        h = block.self_attn.register_forward_hook(_make_hook(idx))
        hooks.append(h)

    return hooks, captured_out


# ---------------------------------------------------------------------------
# Token-grid detection (shared with other scripts in this directory)
# ---------------------------------------------------------------------------

def detect_token_dims(S, T_lat, H_lat, W_lat):
    """
    Infer (T_tok, H_tok, W_tok) from sequence length S and VAE latent dims.
    Tries spatial patch sizes 1, 2, 4; prefers T_tok == T_lat.
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
        f"Cannot factor S={S} from latent dims (T={T_lat}, H={H_lat}, W={W_lat})"
    )


# ---------------------------------------------------------------------------
# Frame-to-frame attention approximation
# ---------------------------------------------------------------------------

def compute_frame_attn(q, k, T_tok):
    """
    Compute a (B, T_tok, T_tok) frame-to-frame attention score matrix.

    q, k : (B, S, H, D) float32  where S = T_tok * H_tok * W_tok

    Method
    ------
    Reshape Q and K to (B, T_tok, S_frame, H, D), average over spatial tokens
    within each frame to get frame-level centroids (B, T_tok, H, D), compute
    the scaled dot-product (T_tok, T_tok) matrix, head-average, then softmax
    over key frames.

    This approximates the true frame-level aggregate (mean over all token pairs
    per frame pair) without materialising the full S×S attention matrix.

    frame_attn[b, qi, kj] ≈ softmax_{kj}( mean_h( Q̄_h[qi] · K̄_h[kj] / √D ) )
    """
    B, S, H, D = q.shape
    S_per_frame = S // T_tok
    scale       = D ** -0.5

    # Mean Q and K per temporal-frame: (B, T_tok, H, D)
    q_frame = q.view(B, T_tok, S_per_frame, H, D).mean(dim=2)
    k_frame = k.view(B, T_tok, S_per_frame, H, D).mean(dim=2)

    # (B, T_qi, T_kj, H) → average over heads → (B, T_qi, T_kj)
    scores = torch.einsum("bihd,bjhd->bijh", q_frame, k_frame) * scale
    scores = scores.mean(dim=-1)            # (B, T_tok, T_tok)
    return torch.softmax(scores, dim=-1)    # softmax over key frames


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_frame_attn(frame_attn, num_cond_tok, prefix=""):
    """
    Print a (T_tok, T_tok) float tensor as a labelled matrix.
    C = conditioning frame, G = generation frame.
    """
    T = frame_attn.shape[0]
    col_labels = [f"k{j}({'C' if j < num_cond_tok else 'G'})" for j in range(T)]
    header = prefix + f"{'':>7}  " + "  ".join(f"{lbl:>7}" for lbl in col_labels)
    print(header)
    for i in range(T):
        tag   = "C" if i < num_cond_tok else "G"
        label = f"q{i}({tag})"
        vals  = "  ".join(f"{frame_attn[i, j].item():>7.4f}" for j in range(T))
        print(prefix + f"{label:>7}  {vals}")


def print_bar(values, label, width=30):
    """Print a small ASCII bar chart for a 1-D tensor."""
    vmin, vmax = values.min().item(), values.max().item()
    span = max(vmax - vmin, 1e-8)
    for i, v in enumerate(values.tolist()):
        filled = int((v - vmin) / span * width)
        bar = "█" * filled + "░" * (width - filled)
        print(f"  {label}{i:>2}: {bar} {v:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path",       required=True)
    parser.add_argument("--experiment_name",  required=True)
    parser.add_argument("--ckpt_path",        required=True)
    parser.add_argument("--prompt",           required=True)
    parser.add_argument("--negative_prompt",  default=None)
    parser.add_argument("--resolution",       default="432,432",
                        help="'H,W' resolution (default: 432,432)")
    parser.add_argument("--num_latent_conditional_frames", type=int, default=2)
    parser.add_argument("--timestep",         type=float, default=0.0,
                        help="Noise level: 0=clean latent, 1=pure noise (default: 0.0)")
    parser.add_argument("--layers",           type=int, nargs="*", default=None,
                        help="DiT block indices to probe. "
                             "Default: 5 evenly-spaced blocks across all layers.")
    parser.add_argument("--save_qk",          action="store_true",
                        help="Also save raw Q and K tensors (can be several hundred MB).")
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
    model    = inference.model
    state_t  = model.config.state_t
    n_blocks = len(model.net.blocks)
    required_pixel_frames = (state_t - 1) * 4 + 1

    # Default: 5 evenly-spaced blocks
    if args.layers:
        layer_indices = args.layers
    else:
        step = max(1, n_blocks // 5)
        layer_indices = list(range(0, n_blocks, step))[:5]
        layer_indices[-1] = n_blocks - 1  # always include last

    print(f"Model         : {n_blocks} DiT blocks, state_t={state_t}")
    print(f"Pixel frames  : {required_pixel_frames}")
    print(f"Probing blocks: {layer_indices}")

    # ------------------------------------------------------------------
    # 2. Load video
    # ------------------------------------------------------------------
    print(f"\nLoading video : {args.video_path}")
    video_uint8 = load_and_preprocess_video(
        video_path=args.video_path,
        resolution=[H, W],
        num_video_frames=required_pixel_frames,
    )
    print(f"Video shape   : {video_uint8.shape}")  # (1, C, T_px, H, W)

    # ------------------------------------------------------------------
    # 3. Build data batch
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
    data_batch["video"]             = video_bf16
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
    print(f"Latent shape  : {tuple(latent_state.shape)}")
    print(f"Latent grid   : T={T_lat}, H={H_lat}, W={W_lat}  "
          f"(tokens per frame = {H_lat*W_lat}, total = {T_lat*H_lat*W_lat})")

    # ------------------------------------------------------------------
    # 6. Build noisy latent at the requested timestep
    # ------------------------------------------------------------------
    t     = args.timestep
    noise = torch.randn_like(latent_state)
    xt    = ((1 - t) * latent_state + t * noise).to(**model.tensor_kwargs)

    cond_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(
        1, C_lat, 1, 1, 1
    ).type_as(xt)
    xt = latent_state.type_as(xt) * cond_mask + xt * (1 - cond_mask)

    timesteps = torch.full(
        (B, T_lat), t, device=latent_state.device, dtype=latent_state.dtype,
    )

    # ------------------------------------------------------------------
    # 7. Install probes
    # ------------------------------------------------------------------
    print(f"\nInstalling Q/K capture wrappers on blocks {layer_indices}...")
    captured_qk,  restore_qk  = wrap_selfattn_qk(model.net, layer_indices)
    hooks_out,    captured_out = register_selfattn_output_hooks(model.net, layer_indices)

    # ------------------------------------------------------------------
    # 8. Single forward pass
    # ------------------------------------------------------------------
    print(f"Running DiT forward pass at t={t}...")
    with torch.no_grad():
        _ = model.net(
            x_B_C_T_H_W=xt,
            timesteps_B_T=timesteps,
            **condition.to_dict(),
        )

    restore_qk()
    for h in hooks_out:
        h.remove()

    print(f"Captured Q/K  : {len(captured_qk)} blocks")
    print(f"Captured out  : {len(captured_out)} blocks")

    # ------------------------------------------------------------------
    # 9. Detect token grid from output sequence length
    # ------------------------------------------------------------------
    sample_S = next(iter(captured_out.values())).shape[1]  # S = T_tok*H_tok*W_tok
    T_tok, H_tok, W_tok = detect_token_dims(sample_S, T_lat, H_lat, W_lat)
    num_cond_tok = args.num_latent_conditional_frames

    print(f"\n{'='*72}")
    print(f"  TOKEN GRID   : T_tok={T_tok}  H_tok={H_tok}  W_tok={W_tok}")
    print(f"  Sequence len : {T_tok*H_tok*W_tok}  ({H_tok*W_tok} tokens/frame)")
    print(f"  Frames       : {num_cond_tok} conditioning (0..{num_cond_tok-1})  "
          f"+ {T_tok-num_cond_tok} generation ({num_cond_tok}..{T_tok-1})")
    print(f"{'='*72}")

    # ------------------------------------------------------------------
    # 10. Per-block structural analysis
    # ------------------------------------------------------------------
    blocks_save = {}
    all_frame_attns = []

    for blk_idx in sorted(captured_qk.keys()):
        q, k = captured_qk[blk_idx]   # (B, S, H, D) float32 cpu
        out  = captured_out.get(blk_idx)  # (B, S, D) float32 cpu

        B_loc, S, n_heads, d_head = q.shape
        S_per_frame = S // T_tok

        print(f"\n{'─'*72}")
        print(f"  BLOCK {blk_idx}")
        print(f"{'─'*72}")
        print(f"  Q  : {tuple(q.shape)}  dtype={q.dtype}")
        print(f"  K  : {tuple(k.shape)}")
        print(f"  out: {tuple(out.shape) if out is not None else 'not captured'}")
        print(f"  heads={n_heads}  d_head={d_head}")

        # Per-frame Q-norm and K-norm  (mean over heads and spatial tokens)
        q_frames = q.view(B_loc, T_tok, S_per_frame, n_heads, d_head)  # (B,T,Sf,H,D)
        k_frames = k.view(B_loc, T_tok, S_per_frame, n_heads, d_head)
        q_norm   = q_frames.norm(dim=-1).mean(dim=(-1, -2))[0]  # (T_tok,)
        k_norm   = k_frames.norm(dim=-1).mean(dim=(-1, -2))[0]

        print(f"\n  Per-frame Q-norm (mean over heads + spatial tokens):")
        print_bar(q_norm, "  q-norm frame")
        print(f"\n  Per-frame K-norm:")
        print_bar(k_norm, "  k-norm frame")

        # Per-frame output norm
        out_norm_frame = None
        if out is not None:
            out_frames     = out.view(B_loc, T_tok, S_per_frame, out.shape[-1])
            out_norm_frame = out_frames.norm(dim=-1).mean(dim=-1)[0]  # (T_tok,)
            print(f"\n  Per-frame self-attn output norm:")
            print_bar(out_norm_frame, "  out-norm  frame")

        # Frame-to-frame attention matrix
        frame_attn = compute_frame_attn(q[0:1], k[0:1], T_tok)[0]  # (T_tok, T_tok)
        all_frame_attns.append(frame_attn)

        print(f"\n  Frame-to-frame attention score matrix")
        print(f"  (rows=query frame, cols=key frame, softmax over key frames, head-averaged)")
        print(f"  [C=conditioning, G=generation]")
        print_frame_attn(frame_attn, num_cond_tok, prefix="  ")

        # Conditioning↔generation breakdown for this block
        if num_cond_tok > 0 and num_cond_tok < T_tok:
            c2c = frame_attn[:num_cond_tok, :num_cond_tok].mean().item()
            g2c = frame_attn[num_cond_tok:, :num_cond_tok].mean().item()
            c2g = frame_attn[:num_cond_tok, num_cond_tok:].mean().item()
            g2g = frame_attn[num_cond_tok:, num_cond_tok:].mean().item()
            print(f"\n  Attention flow (mean weight):  "
                  f"COND→COND {c2c:.4f}  |  GEN→COND {g2c:.4f}  |  "
                  f"COND→GEN {c2g:.4f}  |  GEN→GEN {g2g:.4f}")

        # Accumulate for saving
        entry = {
            "frame_attn": frame_attn,
            "q_norm":     q_norm,
            "k_norm":     k_norm,
        }
        if out_norm_frame is not None:
            entry["out_norm"] = out_norm_frame
        if args.save_qk:
            entry["q"] = q
            entry["k"] = k
        blocks_save[blk_idx] = entry

    # ------------------------------------------------------------------
    # 11. Summary: average frame-to-frame attention across all probed blocks
    # ------------------------------------------------------------------
    mean_frame_attn = torch.stack(all_frame_attns, dim=0).mean(dim=0)  # (T_tok, T_tok)

    print(f"\n{'='*72}")
    print(f"  SUMMARY — frame-to-frame attention averaged over {len(all_frame_attns)} blocks")
    print(f"{'='*72}")
    print_frame_attn(mean_frame_attn, num_cond_tok)

    if num_cond_tok > 0 and num_cond_tok < T_tok:
        c2c = mean_frame_attn[:num_cond_tok, :num_cond_tok].mean().item()
        g2c = mean_frame_attn[num_cond_tok:, :num_cond_tok].mean().item()
        c2g = mean_frame_attn[:num_cond_tok, num_cond_tok:].mean().item()
        g2g = mean_frame_attn[num_cond_tok:, num_cond_tok:].mean().item()
        print(f"\nAttention flow (mean weight over frame pairs):")
        print(f"  COND→COND : {c2c:.4f}")
        print(f"  GEN→COND  : {g2c:.4f}  ← how strongly generation attends to conditioning")
        print(f"  COND→GEN  : {c2g:.4f}")
        print(f"  GEN→GEN   : {g2g:.4f}")

    # ------------------------------------------------------------------
    # 12. Save (rank 0 only)
    # ------------------------------------------------------------------
    if local_rank == 0:
        out_dir = WM_ROOT / "attack" / "outputs" / f"selfattn_structure_{int(time.time())}"
        out_dir.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "token_grid":      {"T_tok": T_tok, "H_tok": H_tok, "W_tok": W_tok},
            "num_cond_tok":    num_cond_tok,
            "latent_shape":    tuple(latent_state.shape),
            "probed_blocks":   list(blocks_save.keys()),
            "mean_frame_attn": mean_frame_attn,
            "blocks":          blocks_save,
        }
        pt_path = out_dir / "selfattn_structure.pt"
        torch.save(save_dict, pt_path)

        print(f"\nSaved: {pt_path}")
        print(f"  Keys: token_grid, num_cond_tok, latent_shape, probed_blocks, "
              f"mean_frame_attn, blocks")
        print(f"  blocks[idx] keys: frame_attn (T,T), q_norm (T,), k_norm (T,), "
              f"out_norm (T,)" + (", q (B,S,H,D), k (B,S,H,D)" if args.save_qk else ""))
        print(f"\nTo load:")
        print(f"  import torch")
        print(f"  d = torch.load('{pt_path}', weights_only=False)")
        print(f"  d['mean_frame_attn']           # ({T_tok}, {T_tok}) frame attention matrix")
        print(f"  d['blocks'][<idx>]['frame_attn']  # per-block version")


if __name__ == "__main__":
    main()


"""
# Example run — 5 evenly-spaced blocks (default)
python cosmos-predict2.5/scripts/probe_selfattn_structure.py \\
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \\
    --experiment_name predict2_video2world_training_2b_libero_480 \\
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/8fbc6188fa2f2e4ab585dc6aac3edd0e9d8a3670/model.pt \\
    --prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate" \\
    --resolution 432,432 \\
    --num_latent_conditional_frames 2 \\
    --timestep 0.0 \\
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \\
    --offload_diffusion_model --offload_tokenizer --offload_text_encoder

# Specific blocks + save raw Q/K tensors
python cosmos-predict2.5/scripts/probe_selfattn_structure.py \\
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \\
    --experiment_name predict2_video2world_training_2b_libero_480 \\
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/8fbc6188fa2f2e4ab585dc6aac3edd0e9d8a3670/model.pt \\
    --prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate" \\
    --resolution 432,432 \\
    --num_latent_conditional_frames 2 \\
    --timestep 0.0 \\
    --layers 0 5 10 15 20 25 \\
    --save_qk \\
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \\
    --offload_diffusion_model --offload_tokenizer --offload_text_encoder
"""
