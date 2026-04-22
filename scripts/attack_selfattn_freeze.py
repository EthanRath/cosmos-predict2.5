"""
PGD attack implementing the Vid-Freeze L_freeze objective for Cosmos Video2World.

Adds an imperceptible perturbation to the first conditioning frame so that the
model's self-attention collapses toward uniform weights across all frames.  When
attention is uniform each generated frame copies the conditioning content equally,
producing a temporally frozen (static) output video.

Loss (Eq. 7 from Vid-Freeze, coupled spatio-temporal case):

    L_freeze = (1/L) * Σ_l  ||A^(l)||_F^2

where A^(l) is the per-layer attention tensor.  Because the full S×S attention
matrix is infeasible to materialise (flash attention does not return weights), we
use a spatial-mean approximation: Q and K are averaged over spatial tokens within
each temporal frame, giving frame-level centroids (B, T, H, D).  The resulting
(B, T, T, H) score matrix after softmax is the quantity we minimise.

Optimisation (Eq. 8):

    min_{||δ||_∞ ≤ ε}  E_t [ L_freeze(x_cond + δ; t) ]

where t ~ U[0,1] is resampled at every PGD step.  Only the first conditioning
pixel frame is perturbed (the "input image" in the Vid-Freeze sense).

VRAM knobs
----------
--num_latent_video_frames   Reduce below the model's default state_t to shorten
                            the latent sequence (fewer tokens → less memory and
                            faster backward pass).  Default: 24.
--num_attack_layers         Cut the gradient graph after this many DiT blocks.
                            The forward pass still runs fully; backward only
                            reaches the first N blocks.  Default: all blocks.

Input modes
-----------
Single video  : --video_path <file> --prompt "<text>"
Batch         : --batch_path <dir>  (contains videos/ and metas/ subfolders)
                --attack_num <N>    sample N videos from the batch (-1 = all)

Saved artefacts (attack/outputs/selfattn_freeze_<timestamp>/):
    x_adv.pt        : perturbed padded video (B, C, T_px, H, W) float32 in [-1,1]
    x_orig.pt       : original padded video
    delta.pt        : perturbation on first frame (B, C, 1, H, W)
    x_adv_frame0.pt : perturbed first frame only (B, C, 1, H, W)
    x_adv.mp4       : perturbed video
    x_orig.mp4      : original video

Usage (single video):
    python cosmos-predict2.5/scripts/attack_selfattn_freeze.py \
        --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
        --prompt "Use the franka robot arm to pick up the black bowl" \
        --resolution 432,432 \
        --num_latent_video_frames 9 \
        --num_latent_conditional_frames 2 \
        --steps 100 --alpha 0.00392 --eps 0.0628 \
        --num_attack_layers 14 \
        # (offloading is on by default; pass --load_diffusion_model/tokenizer/text_encoder to disable)

Usage (batch):
    python cosmos-predict2.5/scripts/attack_selfattn_freeze.py \
        --batch_path /mnt/ssd2/libero/eval_set/ \
        --attack_num 10 \
        --resolution 432,432 \
        --num_latent_video_frames 9 \
        --num_latent_conditional_frames 2 \
        --steps 100 --alpha 0.00392 --eps 0.0628 \
        --num_attack_layers 14 --skip_latent
"""

import argparse
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
# Video padding (mirrors attack_cross_attention.py)
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
# Freeze-loss hooks: compute ||frame_attn||_F^2 per block with grad graph
# ---------------------------------------------------------------------------

def install_freeze_hooks(net, T_tok, loss_terms, cutoff=None):
    """
    Wrap block.self_attn.compute_attention on every block up to `cutoff`.

    For each intercepted block the wrapper:
      1. Computes spatial-mean Q and K per temporal frame → (B, T, H, D)
      2. Computes the (B, T, T, H) frame-to-frame attention score matrix
      3. Applies softmax over the key-frame dimension
      4. Accumulates ||attn||_F^2 / (B*H) into `loss_terms`

    Gradients are retained on Q and K so they flow back through the DiT
    and VAE encoder to the perturbed input.

    Parameters
    ----------
    net        : the DiT net (model.net)
    T_tok      : number of temporal token-frames
    loss_terms : list — appended to during the forward pass
    cutoff     : block index beyond which hooks are not installed
                 (gradient graph is severed at that boundary separately)

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
        if not hasattr(block, "self_attn"):
            continue

        attn    = block.self_attn
        orig_fn = attn.compute_attention   # bound method

        def _make_wrapper(fn):
            def _wrapper(q, k, v, **kw):
                # q, k: (B, S, H, D) — keep grad graph, stay on GPU
                B, S, H, D  = q.shape
                S_per_frame = S // T_tok
                scale       = D ** -0.5

                # Spatial mean per temporal frame: (B, T, H, D)
                q_frame = q.view(B, T_tok, S_per_frame, H, D).mean(dim=2)
                k_frame = k.view(B, T_tok, S_per_frame, H, D).mean(dim=2)

                # (B, T_qi, T_kj, H) attention weights — softmax over key-frame dim
                scores = torch.einsum("bihd,bjhd->bijh", q_frame, k_frame) * scale
                attn_weights = F.softmax(scores, dim=2)  # softmax over j (key-frame)

                # ||A||_F^2 / (B*H): minimising post-softmax weights → uniform attention
                block_loss = attn_weights.pow(2).sum() / (B * H)
                loss_terms.append(block_loss)

                return fn(q, k, v, **kw)
            return _wrapper

        attn.__dict__["compute_attention"] = _make_wrapper(orig_fn)
        _patches[idx] = (attn, orig_fn)

    def _restore():
        for _idx, (attn_mod, _orig) in _patches.items():
            attn_mod.__dict__.pop("compute_attention", None)

    return _restore


# ---------------------------------------------------------------------------
# Multi-step denoising forward passes → L_freeze scalar
# ---------------------------------------------------------------------------

def compute_freeze_loss(model, x_padded, condition, T_tok, num_attack_layers=None,
                        skip_latent=False, max_att=False, denoise_steps=1,
                        uncondition=None, guidance_scale=7.0,
                        num_latent_conditional_frames=2):
    """
    Run `denoise_steps` consecutive Euler denoising steps through the DiT,
    starting from pure noise at t=1, and return the mean L_freeze loss
    accumulated across all steps and blocks.

    Routes through model.denoise() (not model.net() directly) so that:
      - condition.gt_frames is always the current perturbed latent
      - timesteps are correctly overridden for conditioning positions
        (via model.config.conditional_frame_timestep)
    This matches the exact inference code path.

    denoise_steps=1 (default): a single forward pass at a randomly drawn
    t ~ U[0,1].  CFG is not applied for the single-step case.

    denoise_steps=K>1 runs a short rectified-flow Euler trajectory:
        t_i = 1 - i*(1/K)   for i = 0, 1, …, K-1
        x_{t_{i+1}} = x_{t_i} - (1/K) * v_pred(x_{t_i}, t_i)
    When `uncondition` is provided, v_pred is computed with CFG:
        v_pred = cond_v + guidance_scale * (cond_v - uncond_v)
    matching the formula used during normal Video2World inference.
    The unconditional pass runs under torch.no_grad().

    Parameters
    ----------
    model             : Video2WorldModelRectifiedFlow
    x_padded          : (B, C, T, H, W) float32 on CUDA, first frame has grad
    condition         : Video2WorldCondition (template — gt_frames updated each call)
    T_tok             : number of temporal token-frames in the DiT sequence
    num_attack_layers : int or None — gradient cut after this many blocks
    denoise_steps     : int — number of consecutive Euler steps (default: 1)
    uncondition       : Video2WorldCondition or None — when provided, CFG is
                        applied for the Euler updates (denoise_steps > 1 only)
    guidance_scale    : float — CFG guidance weight (default: 7.0)
    num_latent_conditional_frames : int — number of conditioning latent frames

    Returns
    -------
    (l_freeze, cleanup) : scalar tensor with gradient graph, and a callable that
                          removes all hooks — must be called after backward().
    """
    compute_dtype = next(model.net.parameters()).dtype

    if skip_latent:
        latent = x_padded
    else:
        latent = model.tokenizer.encode(x_padded.to(compute_dtype)).contiguous().float()
    B, C, T_lat, H_lat, W_lat = latent.shape

    # ── Rebuild condition with current (possibly perturbed) gt_frames ────────
    # CRITICAL: model.denoise() replaces conditioning positions in xt with
    # condition.gt_frames.  The condition template was built from the original
    # (unperturbed) video, so gt_frames is stale.  Rebuild it here so the DiT
    # always sees the current x_adv latent at conditioning positions — matching
    # what eval() does when it builds a fresh condition from x_adv.
    cond_dict = condition.to_dict(skip_underscore=False)
    cond_dict['gt_frames'] = latent.to(compute_dtype)
    condition_live = type(condition)(**cond_dict)

    uncondition_live = None
    if uncondition is not None:
        ucond_dict = uncondition.to_dict(skip_underscore=False)
        ucond_dict['gt_frames'] = latent.to(compute_dtype)
        uncondition_live = type(uncondition)(**ucond_dict)

    n_blocks = len(model.net.blocks)
    cutoff   = num_attack_layers

    # ── Install hooks once — they must remain registered across all K forward
    #    passes AND the entire backward (for gradient-checkpoint recomputation).
    loss_terms = []
    restore    = install_freeze_hooks(model.net, T_tok, loss_terms, cutoff)

    detach_hook = None
    if cutoff is not None and cutoff < n_blocks:
        def _detach(_m, _inp, output):
            if isinstance(output, tuple):
                return (output[0].detach(),) + output[1:]
            return output.detach()
        detach_hook = model.net.blocks[cutoff - 1].register_forward_hook(_detach)

    # Sample noise once — used for all denoising steps.
    noise = torch.randn_like(latent)

    if denoise_steps == 1:
        # Single pass at a randomly drawn t.
        # model.denoise() will:
        #   1. Replace xt[cond] with condition_live.gt_frames (= current latent)
        #   2. Override timesteps[cond] with conditional_frame_timestep (if set)
        #   3. Call model.net(xt, ...) — hooks fire here
        t_norm    = torch.rand(1).item()   # ∈ [0, 1] for interpolation
        xt        = ((1 - t_norm) * latent + t_norm * noise).to(**model.tensor_kwargs)
        timesteps = torch.full((B, T_lat), t_norm * 1000.0, device=latent.device, dtype=latent.dtype)
        model.denoise(noise=noise, xt_B_C_T_H_W=xt, timesteps_B_T=timesteps,
                      condition=condition_live)
    else:
        # K-step Euler trajectory using the model's own scheduler for timesteps.
        # Use differentiable manual Euler (not sample_scheduler.step()) so gradients
        # flow back through the full x_t chain to the perturbed input.
        model.sample_scheduler.set_timesteps(denoise_steps, device=latent.device, shift=5.0)
        sched_ts = model.sample_scheduler.timesteps   # in [0, 1000], length = denoise_steps
        x_t = noise.float()   # start from pure noise; denoise() pins cond frames each step

        for step_i in range(denoise_steps):
            t_model = sched_ts[step_i]
            t_next  = sched_ts[step_i + 1] if step_i + 1 < denoise_steps else sched_ts.new_tensor(0.0)
            xt        = x_t.to(**model.tensor_kwargs)
            timesteps = t_model.reshape(1, 1).expand(B, 1).to(device=latent.device, dtype=latent.dtype)

            if uncondition_live is not None:
                # CFG: uncond pass under no_grad — trim spurious hook terms.
                n_before = len(loss_terms)
                with torch.no_grad():
                    uncond_v = model.denoise(noise=noise, xt_B_C_T_H_W=xt,
                                             timesteps_B_T=timesteps,
                                             condition=uncondition_live)
                del loss_terms[n_before:]

            # Cond pass — freeze hooks collect loss terms here.
            cond_v = model.denoise(noise=noise, xt_B_C_T_H_W=xt,
                                   timesteps_B_T=timesteps, condition=condition_live)

            # CFG combination (matches video2world_model_rectified_flow.py)
            if uncondition_live is not None:
                v_pred = cond_v + guidance_scale * (cond_v - uncond_v)
            else:
                v_pred = cond_v

            # Differentiable Euler step: dt in normalised [0,1] space
            dt_norm = (t_model - t_next).item() / 1000.0
            x_t = x_t - dt_norm * v_pred.float()
            # No need to re-pin cond frames manually: denoise() handles it next step

    # Do NOT call restore() or detach_hook.remove() here.
    #
    # PyTorch gradient checkpointing recomputes each block's forward during
    # cost.backward().  For that recomputation to produce the same number of
    # saved tensors as the original forward, the hooks must still be registered:
    #
    #   • freeze hooks  – add ops on q/k (einsum, softmax, …) that cause extra
    #                     tensors to be saved inside the checkpointed segment.
    #   • detach_hook   – changes the tensor passed between block[cutoff-1] and
    #                     block[cutoff].
    #
    # The caller (pgd) must invoke _cleanup() after cost.backward() returns.

    def _cleanup():
        restore()
        if detach_hook is not None:
            detach_hook.remove()

    return torch.stack(loss_terms).mean() * (-1 if max_att else 1), _cleanup


# ---------------------------------------------------------------------------
# Per-video attack
# ---------------------------------------------------------------------------

def attack_single_video(
    inference, model, video_path, prompt,
    H, W, frames_to_extract, required_pixel_frames,
    T_tok, cutoff_blocks, compute_dtype, args,
    device="cuda", save_time = None
):
    """
    Run the full PGD attack for a single (video_path, prompt) pair.

    Handles its own offloading / model-setup so calls can be chained in a
    batch loop without needing to reload the model.  If the text encoder was
    offloaded during a previous iteration it is restored to GPU before the
    T5 embeddings are computed, then offloaded again afterwards.

    Returns
    -------
    x_adv : (1, C, T, H, W) tensor on CPU
    """
    if save_time is None: save_time = int(time.time())
    # ------------------------------------------------------------------
    # 2. Load video (just the conditioning pixel frames)
    # ------------------------------------------------------------------
    # If text encoder was offloaded by a previous iteration, restore it
    # before building the T5 condition.
    if inference.offload_text_encoder:
        if model.text_encoder is not None:
            if hasattr(model.text_encoder, "model") and model.text_encoder.model is not None:
                model.text_encoder.model = model.text_encoder.model.to(device)
        if _t5_mod.cosmos_encoder is not None:
            _t5_mod.cosmos_encoder.text_encoder = \
                _t5_mod.cosmos_encoder.text_encoder.to(device)

    print(f"\nLoading video : {video_path}")
    video_uint8 = load_and_preprocess_video(
        str(video_path), [H, W], frames_to_extract
    )
    raw_cond = normalize_video(video_uint8, device=device)   # (1, C, T_cond_px, H, W) float32 [-1,1]
    print(f"Conditioning frames shape: {raw_cond.shape}")

    # Pad to required_pixel_frames by repeating the last conditioning frame
    raw_padded = pad_video(raw_cond, frames_to_extract, required_pixel_frames)
    print(f"Padded video shape       : {raw_padded.shape}")

    # ------------------------------------------------------------------
    # 3. Build T5 condition
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
    #    (parameters are frozen so no gradient buffers accumulate for them;
    #     gradients still *flow through* frozen params w.r.t. the input)
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
    # 7. PGD optimisation
    #
    #    delta : (B, C, 1, H, W) — applied only to the first pixel frame.
    #    The perturbation is kept in this shape throughout so all gradient
    #    steps affect only frame 0 (the conditioning "image").
    # ------------------------------------------------------------------
    print(f"\nStarting PGD optimisation...")

    loss_fn = lambda x, y: compute_freeze_loss(
        model, x, condition, T_tok,
        num_attack_layers=cutoff_blocks,
        skip_latent=args.skip_latent, max_att=args.max_att,
        denoise_steps=args.denoise_steps,
        uncondition=uncondition if args.denoise_steps > 1 else None,
        guidance_scale=args.guidance_scale,
        num_latent_conditional_frames=args.num_latent_conditional_frames,
    )
    if args.skip_latent:
        with torch.no_grad():
            latent = model.tokenizer.encode(raw_padded.to(compute_dtype)).contiguous().float()
        x_adv = pgd(latent, 0, lambda x: x, loss_fn, args.steps, args.alpha, args.eps, args.num_latent_conditional_frames)
    else:
        x_adv = pgd(raw_padded, 0, lambda x: x, loss_fn, args.steps, args.alpha, args.eps, frames_to_extract,
                    pixel_min=-1.0, pixel_max=1.0)

    def to_uint8_frames(t):
        """(B, C, T, H, W) float32 [-1,1] → (T, H, W, C) uint8"""
        t = t[0].cpu().float()
        t = ((t.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
        return t.permute(1, 2, 3, 0)

    out_dir = WM_ROOT / "attack" / "outputs" / f"selfattn_freeze_{save_time}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = str(video_path).split("/")[-1].split(".")[0] + "_adv"
    torch.save(x_adv.cpu(), out_dir / f"{base_name}_{args.skip_latent}.pt")

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
    parser.add_argument("--alpha",           type=float, default=2/255,
                        help="PGD step size (default: 1/255 ≈ 0.00392)")
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
    parser.add_argument("--skip_latent", action="store_true")
    parser.add_argument("--max_att", action="store_true")
    parser.add_argument("--denoise_steps", type=int, default=1,
                        help="Number of consecutive Euler denoising steps per PGD "
                             "gradient evaluation.  1 (default) = single forward pass "
                             "at a random t.  K>1 = K-step Euler trajectory from t=1 "
                             "to t=1/K; gradient flows through the full chain, giving "
                             "richer signal with modest additional VRAM cost.")
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
    model   = inference.model
    state_t = model.config.state_t
    n_blocks = len(model.net.blocks)

    # Effective temporal length for the attack (may be < state_t to save VRAM)
    T_lat_attack = min(args.num_latent_video_frames, state_t)
    required_pixel_frames = (T_lat_attack - 1) * 4 + 1
    frames_to_extract     = (args.num_latent_conditional_frames - 1) * 4 + 1
    T_tok = T_lat_attack   # DiT has no additional temporal patchification

    cutoff_blocks  = args.num_attack_layers  # None = all
    compute_dtype  = torch.bfloat16

    print(f"Model         : {n_blocks} DiT blocks, state_t={state_t}")
    print(f"Attack T_lat  : {T_lat_attack}  (pixel frames: {required_pixel_frames})")
    print(f"T_tok         : {T_tok}  (frame-to-frame attn matrix: {T_tok}×{T_tok})")
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
    # Attack each video, collect results
    # ------------------------------------------------------------------
    all_x_adv = []
    save_time = int(time.time())
    out_dir = WM_ROOT / "attack" / "outputs" / f"selfattn_freeze_batch_{save_time}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_t = args.num_latent_video_frames
    for i, (video_path, prompt) in enumerate(pairs):
        print(f"\n{'='*60}")
        print(f"Video {i+1}/{len(pairs)}: {video_path.name}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        x_adv = attack_single_video(
            inference, model, video_path, prompt,
            H, W, frames_to_extract, required_pixel_frames,
            T_tok, cutoff_blocks, compute_dtype, args,
            device=device, save_time= save_time
        )
        all_x_adv.append(x_adv)
        base_name = str(video_path).split("/")[-1].split(".")[0] + "_adv.pt"
        print("\nEvaluating Diffusion")
        args.adv_path = out_dir
        args.prompt   = prompt   # make the per-video prompt visible to eval
        if args.skip_latent:
            eval_latent(inference, x_adv, args, out_dir / base_name)
        else:
            eval(inference, x_adv, args, out_dir / base_name)


    # ------------------------------------------------------------------
    # Start Evaluation
    # ------------------------------------------------------------------
    
    


if __name__ == "__main__":
    main()
