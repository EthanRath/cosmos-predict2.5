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

Saved artefacts (attack/outputs/selfattn_freeze_<timestamp>/):
    x_adv.pt        : perturbed padded video (B, C, T_px, H, W) float32 in [-1,1]
    x_orig.pt       : original padded video
    delta.pt        : perturbation on first frame (B, C, 1, H, W)
    x_adv_frame0.pt : perturbed first frame only (B, C, 1, H, W)
    x_adv.mp4       : perturbed video
    x_orig.mp4      : original video

Usage:
    python cosmos-predict2.5/scripts/attack_selfattn_freeze.py \
        --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
        --experiment_name predict2_video2world_training_2b_libero_480 \
        --ckpt_path /path/to/model.pt \
        --prompt "Use the franka robot arm to pick up the black bowl" \
        --resolution 432,432 \
        --num_latent_video_frames 9 \
        --num_latent_conditional_frames 2 \
        --steps 100 --alpha 0.00392 --eps 0.0628 \
        --num_attack_layers 14 \
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
from probing.test_vae_encoder import load_and_preprocess_video, normalize_video  # noqa: E402
from attack.white_box import pgd
from attack.eval_wm import eval, eval_latent

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

                # (B, T_qi, T_kj, H) score matrix
                scores = torch.einsum("bihd,bjhd->bijh", q_frame, k_frame) * scale

                # Softmax over key-frame dimension, then ||A||_F^2 normalised per B, H
                attn_w     = torch.softmax(scores, dim=2)   # (B, T, T, H)
                block_loss = attn_w.pow(2).sum() / (B * H)
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
# Single forward pass → L_freeze scalar
# ---------------------------------------------------------------------------

def compute_freeze_loss(model, x_padded, condition, T_tok, num_attack_layers=None, skip_latent = False):
    """
    Encode `x_padded` through the VAE, build a noisy latent at timestep `t`,
    run the DiT with freeze hooks, and return the L_freeze scalar.

    Gradient graph is live from the scalar back through the DiT and VAE to
    `x_padded` (which must have requires_grad via its dependency on delta).

    Parameters
    ----------
    model             : Video2WorldModelRectifiedFlow
    x_padded          : (B, C, T_px, H, W) float32 on CUDA, first frame has grad
    condition         : Video2WorldCondition (frozen)
    t                 : float in [0, 1] — diffusion timestep
    T_tok             : number of temporal token-frames in the DiT sequence
    num_attack_layers : int or None — gradient is cut after this many blocks

    Returns
    -------
    l_freeze : scalar tensor with gradient graph
    """
    t = torch.rand(1).item()
    compute_dtype = next(model.net.parameters()).dtype

    # VAE encode (grads flow through when model.tokenizer.enable_grad = True)
    if skip_latent: 
        latent = x_padded
    else:
        latent = model.tokenizer.encode(x_padded.to(compute_dtype)).contiguous().float()
    B, C, T_lat, H_lat, W_lat = latent.shape

    noise = torch.randn_like(latent)
    xt    = ((1 - t) * latent + t * noise).to(**model.tensor_kwargs)

    cond_mask = condition.condition_video_input_mask_B_C_T_H_W.repeat(
        1, C, 1, 1, 1
    ).type_as(xt)
    xt = latent.type_as(xt) * cond_mask + xt * (1 - cond_mask)

    timesteps = torch.full((B, T_lat), t, device=latent.device, dtype=latent.dtype)

    # ── Install freeze hooks ─────────────────────────────────────────────
    loss_terms = []
    n_blocks   = len(model.net.blocks)
    cutoff     = num_attack_layers   # None → all blocks

    restore = install_freeze_hooks(model.net, T_tok, loss_terms, cutoff)

    # Detach hook at the cutoff boundary — severs the backward graph for
    # blocks [cutoff, n_blocks) while keeping the full forward computation.
    detach_hook = None
    if cutoff is not None and cutoff < n_blocks:
        def _detach(_m, _inp, output):
            if isinstance(output, tuple):
                return (output[0].detach(),) + output[1:]
            return output.detach()
        detach_hook = model.net.blocks[cutoff - 1].register_forward_hook(_detach)

    model.net(x_B_C_T_H_W=xt, timesteps_B_T=timesteps, **condition.to_dict())

    restore()
    if detach_hook is not None:
        detach_hook.remove()

    # L_freeze = mean over blocks of normalised ||A^(l)||_F^2
    return torch.stack(loss_terms).mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path",      required=True,
                        help="Path to input video (.mp4)")
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--ckpt_path",       required=True)
    parser.add_argument("--prompt",          required=True,
                        help="Text prompt used during inference (used to build the "
                             "condition; null-prompt or caption both work)")
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
    parser.add_argument("--alpha",           type=float, default=1/255,
                        help="PGD step size (default: 1/255 ≈ 0.00392)")
    parser.add_argument("--eps",             type=float, default=16/255,
                        help="L∞ perturbation budget (default: 16/255 ≈ 0.0628)")
    parser.add_argument("--num_attack_layers", type=int, default=None,
                        help="Only include the first N DiT blocks in the loss / "
                             "gradient graph. Drastically reduces peak VRAM. "
                             "Grad is cut at block N; forward still runs fully. "
                             "Default: all blocks.")
    parser.add_argument("--offload_diffusion_model", action="store_true")
    parser.add_argument("--offload_text_encoder",    action="store_true")
    parser.add_argument("--offload_tokenizer",       action="store_true")
    parser.add_argument("--context_parallel_size",   type=int, default=1)
    parser.add_argument("--config_file",
                        default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    parser.add_argument("--skip_latent", action="store_true")
    args = parser.parse_args()

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
        offload_diffusion_model=args.offload_diffusion_model,
        offload_text_encoder=args.offload_text_encoder,
        offload_tokenizer=args.offload_tokenizer,
    )
    model   = inference.model
    state_t = model.config.state_t
    n_blocks = len(model.net.blocks)

    # Effective temporal length for the attack (may be < state_t to save VRAM)
    T_lat_attack = min(args.num_latent_video_frames, state_t)
    required_pixel_frames = (T_lat_attack - 1) * 4 + 1
    frames_to_extract     = (args.num_latent_conditional_frames - 1) * 4 + 1
    T_tok = T_lat_attack   # DiT has no additional temporal patchification

    cutoff_blocks = args.num_attack_layers  # None = all

    print(f"Model         : {n_blocks} DiT blocks, state_t={state_t}")
    print(f"Attack T_lat  : {T_lat_attack}  (pixel frames: {required_pixel_frames})")
    print(f"T_tok         : {T_tok}  (frame-to-frame attn matrix: {T_tok}×{T_tok})")
    print(f"Loss blocks   : {cutoff_blocks or n_blocks}/{n_blocks}  "
          f"(gradient cut after block {(cutoff_blocks or n_blocks) - 1})")
    print(f"PGD           : steps={args.steps}  alpha={args.alpha:.5f}  eps={args.eps:.4f}")

    # ------------------------------------------------------------------
    # 2. Load video (just the conditioning pixel frames)
    # ------------------------------------------------------------------
    print(f"\nLoading video : {args.video_path}")
    video_uint8 = load_and_preprocess_video(
        args.video_path, [H, W], frames_to_extract
    )
    raw_cond = normalize_video(video_uint8, device=device)   # (1, C, T_cond_px, H, W) float32 [-1,1]
    print(f"Conditioning frames shape: {raw_cond.shape}")

    # Pad to required_pixel_frames by repeating the last conditioning frame
    raw_padded = pad_video(raw_cond, frames_to_extract, required_pixel_frames)
    print(f"Padded video shape       : {raw_padded.shape}")

    # ------------------------------------------------------------------
    # 3. Build T5 condition
    # ------------------------------------------------------------------
    compute_dtype = torch.bfloat16
    video_bf16    = raw_padded.to(dtype=compute_dtype)

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
    # ---------------------------------------------s---------------------
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
    # 6. Build condition object (VAE encode + cond mask)
    # ------------------------------------------------------------------
    print("Building condition...")
    with torch.no_grad():
        _, _, condition = model.get_data_and_condition(data_batch)
    condition = condition.edit_for_inference(
        is_cfg_conditional=True,
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

    t = torch.rand(1).item()

    loss_fn =lambda x, y: compute_freeze_loss(
        model, x, condition, T_tok,
        num_attack_layers=cutoff_blocks,
    )
    if args.skip_latent:
        with torch.no_grad():
            latent = model.tokenizer.encode(raw_padded.to(compute_dtype)).contiguous().float()
        x_adv = pgd(latent, 0, lambda x: x, loss_fn, args.steps, args.alpha, args.eps, frames_to_extract)
    else:
        x_adv = pgd(raw_padded, 0, lambda x: x, loss_fn, args.steps, args.alpha, args.eps, frames_to_extract)

    def to_uint8_frames(t):
        """(B, C, T, H, W) float32 [-1,1] → (T, H, W, C) uint8"""
        t = t[0].cpu().float()
        t = ((t.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
        return t.permute(1, 2, 3, 0)

    out_dir = WM_ROOT / "attack" / "outputs" / f"selfattn_freeze_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(x_adv.cpu(), out_dir / f"x_adv{args.skip_latent}.pt")
    # torch.save(raw_padded.cpu(), out_dir / "x_orig.pt")

    if not args.skip_latent:
        torchvision.io.write_video(
            str(out_dir / "x_adv.mp4"),  to_uint8_frames(x_adv),     fps=16)
        # torchvision.io.write_video(
        #     str(out_dir / "x_orig.mp4"), to_uint8_frames(raw_padded), fps=16)

    print(f"\nSaved to: {out_dir}")
    print(f"  x_adv.pt / x_adv.mp4 : perturbed video")
    print("Evaluating Diffusion")
    if args.skip_latent:
        eval_latent(inference, x_adv, args)
    else:
        eval(inference, x_adv, args)


if __name__ == "__main__":
    main()


"""
# Full sequence (24 frames), all blocks — highest quality, most VRAM
python cosmos-predict2.5/scripts/attack_selfattn_freeze.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --experiment_name predict2_video2world_training_2b_libero_480 \
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/47d14a41779c654c213600ec1c35c9ebd89dd992/model.pt \
    --prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate" \
    --resolution 432,432 \
    --num_latent_video_frames 24 \
    --num_latent_conditional_frames 2 \
    --steps 100 --alpha 0.00392 --eps 0.0628 \
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --offload_diffusion_model --offload_tokenizer --offload_text_encoder

# Reduced sequence + gradient cutoff — lower VRAM
python cosmos-predict2.5/scripts/attack_selfattn_freeze.py \
    --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \
    --experiment_name predict2_video2world_training_2b_libero_480 \
    --ckpt_path /home/ethan/.cache/huggingface/hub/models--EthanRath--cosmos-predict2-libero/snapshots/47d14a41779c654c213600ec1c35c9ebd89dd992/model.pt \
    --prompt "Use the franka robot arm to pick up the black bowl next to the cookie box and place it on the plate" \
    --resolution 432,432 \
    --num_latent_video_frames 9 \
    --num_latent_conditional_frames 2 \
    --steps 100 --alpha 0.00392 --eps 0.0628 \
    --num_attack_layers 14 \
    --config_file cosmos_predict2/_src/predict2/configs/video2world/config.py \
    --offload_diffusion_model --offload_tokenizer --offload_text_encoder --skip_latent
"""
