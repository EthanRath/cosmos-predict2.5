"""
Probe the structural properties of self-attention in the Cosmos Video2World DiT.

Captures Q and K tensors from selected DiT blocks by temporarily wrapping
block.self_attn.compute_attention (instance-level monkey-patch, no source
modification needed).  For each probed block the script reports:

  1. Q / K / output tensor shapes and dtypes
  2. Per-temporal-frame Q-norm and K-norm (mean over heads + spatial tokens)
  3. T_tok × T_tok frame-to-frame attention score matrix:
       frame_attn[qi, kj]  = softmax_kj( mean_head( Q̄[qi] · K̄[kj] / √D ) )
     where Q̄[qi] is the spatial-mean Q vector for token-frame qi.
     This approximates the true frame-level aggregate without materialising
     the full S×S attention matrix.
  4. Per-frame output L2-norm

After all blocks a summary frame-to-frame matrix averaged across all probed
blocks is printed, along with four aggregate attention-flow numbers:
  COND→COND, GEN→COND, COND→GEN, GEN→GEN.

Memory design
-------------
Per-frame statistics (frame_attn, q_norm, k_norm, out_norm) are computed
inline as each block fires, so peak CPU RAM is bounded to one block's Q/K
tensors at a time (~140 MB for the 2B model) regardless of how many blocks
are probed.  Raw Q/K tensors are discarded immediately unless --save_qk is
set (adds ~136 MB × n_probed_blocks to peak CPU RAM).

No video rendering is performed.

Saved artefacts (attack/outputs/selfattn_structure_<timestamp>/):
  selfattn_structure.pt — dict with:
      token_grid      : {"T_tok", "H_tok", "W_tok"}
      num_cond_tok    : int
      latent_shape    : (B, C, T, H, W)
      probed_blocks   : list[int]
      mean_frame_attn : (T_tok, T_tok) float32
      blocks          : {block_idx: {"frame_attn": (T,T), "q_norm": (T,),
                                     "k_norm": (T,), "out_norm": (T,),
                                     ["q": (B,S,H,D), "k": (B,S,H,D)]}}

Usage:
    python cosmos-predict2.5/scripts/probe_selfattn_structure.py \
        --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
        --experiment_name predict2_video2world_training_2b_libero_480 \
        --ckpt_path /path/to/model.pt \
        --prompt "Use the franka robot arm to pick up the black bowl" \
        --resolution 432,432 \
        --num_latent_conditional_frames 2 \
        --timestep 0.0 \
        --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
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
# Token-grid detection
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

    q, k : (B, S, H, D) float32  where S = T_tok * S_per_frame

    Spatial tokens within each frame are averaged to get frame-level Q/K
    centroids (B, T_tok, H, D).  The (T_tok, T_tok) scaled dot-product matrix
    is head-averaged then softmaxed over key frames.

    frame_attn[b, qi, kj] ≈ softmax_{kj}( mean_h( Q̄_h[qi] · K̄_h[kj] / √D ) )

    This avoids materialising the full (S × S) attention matrix.
    """
    B, S, H, D = q.shape
    S_per_frame = S // T_tok
    scale       = D ** -0.5

    q_frame = q.view(B, T_tok, S_per_frame, H, D).mean(dim=2)  # (B, T, H, D)
    k_frame = k.view(B, T_tok, S_per_frame, H, D).mean(dim=2)

    scores = torch.einsum("bihd,bjhd->bijh", q_frame, k_frame) * scale  # (B, T, T, H)
    return torch.softmax(scores.mean(dim=-1), dim=-1)   # (B, T_tok, T_tok)


# ---------------------------------------------------------------------------
# Streaming probes — compute stats inline, discard raw Q/K immediately
# ---------------------------------------------------------------------------

def install_selfattn_probes(net, T_tok, layer_indices=None, save_qk=False):
    """
    Install Q/K wrappers and output hooks on selected DiT blocks.

    Both wrappers and hooks write into the same `captured` dict keyed by
    block index.  Statistics are computed inline as each block fires so
    that peak CPU RAM is bounded to one block's tensors at a time.

    Returns
    -------
    captured : dict {block_idx -> stats_entry}  — populated during a forward pass
               stats_entry keys: frame_attn (T,T), q_norm (T,), k_norm (T,),
               out_norm (T,) [added by output hook], plus q/k if save_qk.
    restore  : callable — removes all patches and hooks.
    """
    n_blocks = len(net.blocks)
    indices  = layer_indices if layer_indices is not None else list(range(n_blocks))

    captured  = {}
    _patches  = {}   # {idx: (attn_mod, orig_bound_method)}
    _hooks    = []

    for idx in indices:
        if idx >= n_blocks:
            continue
        block = net.blocks[idx]
        if not hasattr(block, "self_attn"):
            continue

        attn    = block.self_attn
        orig_fn = attn.compute_attention  # bound method

        # ── Q/K wrapper ─────────────────────────────────────────────────────
        # Written into instance __dict__ so Python returns it directly (no
        # auto-binding), giving signature (q, k, v, **kw) at the call site.
        def _make_qk_wrapper(i, fn):
            def _wrapper(q, k, v, **kw):
                q_cpu = q.detach().cpu().float()   # move to CPU while still hot
                k_cpu = k.detach().cpu().float()

                B_loc, S, H, D = q_cpu.shape
                S_per_frame = S // T_tok

                # Per-frame Q/K norms  (discard spatial/head detail immediately)
                q_f    = q_cpu.view(B_loc, T_tok, S_per_frame, H, D)
                k_f    = k_cpu.view(B_loc, T_tok, S_per_frame, H, D)
                q_norm = q_f.norm(dim=-1).mean(dim=(-1, -2))[0]   # (T_tok,)
                k_norm = k_f.norm(dim=-1).mean(dim=(-1, -2))[0]

                # Frame-to-frame attention matrix
                frame_attn = compute_frame_attn(q_cpu[:1], k_cpu[:1], T_tok)[0]

                entry = {
                    "frame_attn": frame_attn,
                    "q_norm":     q_norm,
                    "k_norm":     k_norm,
                    "shape_q":    tuple(q_cpu.shape),
                }
                if save_qk:
                    entry["q"] = q_cpu
                    entry["k"] = k_cpu
                # q_cpu / k_cpu go out of scope here if not saved → freed
                captured[i] = entry

                return fn(q, k, v, **kw)
            return _wrapper

        attn.__dict__["compute_attention"] = _make_qk_wrapper(idx, orig_fn)
        _patches[idx] = (attn, orig_fn)

        # ── Output hook ─────────────────────────────────────────────────────
        def _make_out_hook(i):
            def _hook(_m, _inp, output):
                out = (output[0] if isinstance(output, tuple) else output
                       ).detach().cpu().float()   # (B, S, D)
                B_loc, S, D = out.shape
                out_f    = out.view(B_loc, T_tok, S // T_tok, D)
                out_norm = out_f.norm(dim=-1).mean(dim=-1)[0]  # (T_tok,)
                if i in captured:
                    captured[i]["out_norm"] = out_norm
                else:
                    captured[i] = {"out_norm": out_norm}
            return _hook

        _hooks.append(attn.register_forward_hook(_make_out_hook(idx)))

    def _restore():
        for _idx, (attn_mod, _orig) in _patches.items():
            attn_mod.__dict__.pop("compute_attention", None)
        for h in _hooks:
            h.remove()

    return captured, _restore


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_frame_attn(frame_attn, num_cond_tok, prefix=""):
    """Print a (T_tok, T_tok) float tensor as a labelled matrix."""
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
                        help="DiT block indices to probe. Default: all blocks.")
    parser.add_argument("--save_qk",          action="store_true",
                        help="Also save raw Q and K tensors (~136 MB × n_blocks).")
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

    layer_indices = args.layers  # None → all blocks
    print(f"Model         : {n_blocks} DiT blocks, state_t={state_t}")
    print(f"Pixel frames  : {required_pixel_frames}")
    print(f"Probing blocks: {layer_indices if layer_indices else 'all'}")

    # ------------------------------------------------------------------
    # 2. Load video
    # ------------------------------------------------------------------
    print(f"\nLoading video : {args.video_path}")
    video_uint8 = load_and_preprocess_video(
        video_path=args.video_path,
        resolution=[H, W],
        num_video_frames=required_pixel_frames,
    )
    print(f"Video shape   : {video_uint8.shape}")

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

    # ------------------------------------------------------------------
    # 6. Detect token grid from latent dims (no extra pass needed)
    #    The DiT receives the latent directly with no additional temporal
    #    patchification, so T_tok == T_lat in the typical case.
    # ------------------------------------------------------------------
    T_tok, H_tok, W_tok = detect_token_dims(
        T_lat * H_lat * W_lat, T_lat, H_lat, W_lat
    )
    num_cond_tok = args.num_latent_conditional_frames

    print(f"Token grid    : T={T_tok}  H={H_tok}  W={W_tok}  "
          f"(S={T_tok*H_tok*W_tok}, {H_tok*W_tok} per frame)")
    print(f"Frames        : {num_cond_tok} conditioning (0..{num_cond_tok-1})  "
          f"+ {T_tok-num_cond_tok} generation ({num_cond_tok}..{T_tok-1})")

    # ------------------------------------------------------------------
    # 7. Build noisy latent at the requested timestep
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
    # 8. Install streaming probes (stats computed inline → bounded RAM)
    # ------------------------------------------------------------------
    print(f"\nInstalling self-attention probes...")
    captured, restore = install_selfattn_probes(
        model.net, T_tok,
        layer_indices=layer_indices,
        save_qk=args.save_qk,
    )

    # ------------------------------------------------------------------
    # 9. Single forward pass
    # ------------------------------------------------------------------
    print(f"Running DiT forward pass at t={t}...")
    with torch.no_grad():
        _ = model.net(
            x_B_C_T_H_W=xt,
            timesteps_B_T=timesteps,
            **condition.to_dict(),
        )
    restore()
    print(f"Captured stats from {len(captured)} blocks")

    # ------------------------------------------------------------------
    # 10. Print per-block structural analysis
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"  TOKEN GRID   : T_tok={T_tok}  H_tok={H_tok}  W_tok={W_tok}")
    print(f"  Sequence len : {T_tok*H_tok*W_tok}  ({H_tok*W_tok} tokens/frame)")
    print(f"  Frames       : {num_cond_tok} conditioning  "
          f"+ {T_tok-num_cond_tok} generation")
    print(f"{'='*72}")

    all_frame_attns = []

    for blk_idx in sorted(captured.keys()):
        entry      = captured[blk_idx]
        frame_attn = entry["frame_attn"]   # (T_tok, T_tok)
        q_norm     = entry["q_norm"]       # (T_tok,)
        k_norm     = entry["k_norm"]       # (T_tok,)
        out_norm   = entry.get("out_norm") # (T_tok,) or None
        shape_q    = entry.get("shape_q")

        print(f"\n{'─'*72}")
        print(f"  BLOCK {blk_idx}")
        print(f"{'─'*72}")
        if shape_q:
            B_loc, S, H, D = shape_q
            print(f"  Q shape : {shape_q}  (heads={H}, d_head={D})")

        print(f"\n  Per-frame Q-norm  (mean over heads + spatial tokens):")
        print_bar(q_norm, "  q-norm  frame")
        print(f"\n  Per-frame K-norm:")
        print_bar(k_norm, "  k-norm  frame")
        if out_norm is not None:
            print(f"\n  Per-frame self-attn output norm:")
            print_bar(out_norm, "  out-norm frame")

        print(f"\n  Frame-to-frame attention score matrix")
        print(f"  (rows=query frame, cols=key frame; softmax over key frames, head-averaged)")
        print(f"  [C=conditioning, G=generation]")
        print_frame_attn(frame_attn, num_cond_tok, prefix="  ")

        if num_cond_tok > 0 and num_cond_tok < T_tok:
            c2c = frame_attn[:num_cond_tok, :num_cond_tok].mean().item()
            g2c = frame_attn[num_cond_tok:, :num_cond_tok].mean().item()
            c2g = frame_attn[:num_cond_tok, num_cond_tok:].mean().item()
            g2g = frame_attn[num_cond_tok:, num_cond_tok:].mean().item()
            print(f"\n  Attention flow:  "
                  f"C→C {c2c:.4f}  |  G→C {g2c:.4f}  |  "
                  f"C→G {c2g:.4f}  |  G→G {g2g:.4f}")

        all_frame_attns.append(frame_attn)

    # ------------------------------------------------------------------
    # 11. Summary averaged across all probed blocks
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
        print(f"  C→C : {c2c:.4f}")
        print(f"  G→C : {g2c:.4f}  ← how strongly generation attends to conditioning")
        print(f"  C→G : {c2g:.4f}")
        print(f"  G→G : {g2g:.4f}")

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
            "probed_blocks":   sorted(captured.keys()),
            "mean_frame_attn": mean_frame_attn,
            "blocks":          captured,
        }
        pt_path = out_dir / "selfattn_structure.pt"
        torch.save(save_dict, pt_path)

        print(f"\nSaved: {pt_path}")
        print(f"  blocks[idx] keys: frame_attn (T,T), q_norm (T,), k_norm (T,), out_norm (T,)"
              + (", q (B,S,H,D), k (B,S,H,D)" if args.save_qk else ""))
        print(f"\nTo load:")
        print(f"  import torch")
        print(f"  d = torch.load('{pt_path}', weights_only=False)")
        print(f"  d['mean_frame_attn']              # ({T_tok},{T_tok}) frame attention matrix")
        print(f"  d['blocks'][<idx>]['frame_attn']  # per-block version")


if __name__ == "__main__":
    main()


"""
# All blocks (default) — bounded RAM regardless of model depth
python cosmos-predict2.5/scripts/probe_selfattn_structure.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --experiment_name predict2_video2world_training_2b_libero_480 \
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/47d14a41779c654c213600ec1c35c9ebd89dd992/model.pt \
    --prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate" \
    --resolution 432,432 \
    --num_latent_conditional_frames 2 \
    --timestep 1.0 \
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --offload_diffusion_model --offload_tokenizer --offload_text_encoder

# Specific blocks + save raw Q/K tensors for further analysis
python cosmos-predict2.5/scripts/probe_selfattn_structure.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --experiment_name predict2_video2world_training_2b_libero_480 \
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/8fbc6188fa2f2e4ab585dc6aac3edd0e9d8a3670/model.pt \
    --prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate" \
    --resolution 432,432 \
    --num_latent_conditional_frames 2 \
    --timestep 0.0 \
    --layers 0 5 10 15 20 25 \
    --save_qk \
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --offload_diffusion_model --offload_tokenizer --offload_text_encoder
"""
