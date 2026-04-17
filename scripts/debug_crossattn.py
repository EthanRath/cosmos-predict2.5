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
      2. Computes the (B, T, N, H) raw cross-attention score matrix
         between frame-mean video queries and text keys
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

                # Spatial mean per temporal frame: (B, T, H, D)
                q_frame = q.reshape(B, T_tok, S_per_frame, H, D).mean(dim=2)

                # Cross-attention score matrix: (B, T, N, H)
                # Each temporal frame query attends to every text token.
                scores = torch.einsum("bthd,bnhd->btnh", q_frame, k) * scale

                # Store raw scores; the loss is computed from these in
                # compute_crossattn_loss depending on the attack mode.
                loss_terms.append(scores)

                return fn(q, k, v, **kw)
            return _wrapper

        attn.__dict__["compute_attention"] = _make_wrapper(orig_fn)
        _patches[idx] = (attn, orig_fn)

    def _restore():
        for _idx, (attn_mod, _orig) in _patches.items():
            attn_mod.__dict__.pop("compute_attention", None)

    return _restore


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------

_DIAG_DONE = False   # run field comparison only on the first loss call


def _diag_compare_conditions(cond_a, cond_b, label_a="base", label_b="target"):
    """Field-by-field comparison of two condition objects."""
    print(f"\n[DIAG] Comparing {label_a} vs {label_b}:")
    dict_a = cond_a.to_dict(skip_underscore=False)
    dict_b = cond_b.to_dict(skip_underscore=False)
    all_keys = sorted(set(list(dict_a) + list(dict_b)))
    any_diff = False
    for k in all_keys:
        va, vb = dict_a.get(k), dict_b.get(k)
        if va is None and vb is None:
            print(f"  {k}: both None")
            continue
        if va is None or vb is None:
            print(f"  {k}: ONE IS NONE  ({label_a}={va}, {label_b}={vb})")
            any_diff = True
            continue
        if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor):
            if va.shape != vb.shape:
                print(f"  {k}: SHAPE MISMATCH  {label_a}={tuple(va.shape)}  {label_b}={tuple(vb.shape)}")
                any_diff = True
            else:
                md = (va.float() - vb.float()).abs().max().item()
                eq = "EQUAL" if md < 1e-5 else "DIFFERENT"
                if md >= 1e-5:
                    any_diff = True
                print(f"  {k}: shape={tuple(va.shape)}  max_diff={md:.4e}  [{eq}]")
        elif isinstance(va, bool) and isinstance(vb, bool):
            eq = "EQUAL" if va == vb else "DIFFERENT"
            if va != vb:
                any_diff = True
            print(f"  {k}: {label_a}={va}  {label_b}={vb}  [{eq}]")
        else:
            eq = "EQUAL" if va == vb else "DIFFERENT"
            if va != vb:
                any_diff = True
            print(f"  {k}: {label_a}={va}  {label_b}={vb}  [{eq}]")
    print(f"[DIAG] Summary: {'DIFFERENCES FOUND' if any_diff else 'ALL FIELDS MATCH'}\n")


def _diag_compare_score_lists(terms_a, terms_b, label_a="base", label_b="target"):
    """Compare collected score tensors from two forward passes."""
    print(f"\n[DIAG] Comparing score tensors ({label_a} vs {label_b}):")
    if len(terms_a) != len(terms_b):
        print(f"  MISMATCH: {len(terms_a)} vs {len(terms_b)} blocks captured")
        return
    for i, (sa, sb) in enumerate(zip(terms_a, terms_b)):
        md = (sa.detach().float() - sb.detach().float()).abs().max().item()
        cos = F.cosine_similarity(
            sa.detach().flatten().unsqueeze(0),
            sb.detach().flatten().unsqueeze(0),
        ).item()
        print(f"  block {i:2d}: shape={tuple(sa.shape)}  max_diff={md:.4e}  cosine={cos:.6f}")


# ---------------------------------------------------------------------------
# Multi-step denoising forward passes → L_cross scalar
# ---------------------------------------------------------------------------

def compute_crossattn_loss(model, x_padded, condition, T_tok, num_attack_layers=None,
                           skip_latent=False, max_att=False, denoise_steps=1,
                           uncondition=None, guidance_scale=7.0,
                           target_condition=None, noise_seed=None,
                           num_latent_conditional_frames=2):
    """
    Run `denoise_steps` Euler denoising steps and return the cross-attention loss.

    Untargeted (target_condition=None):
        L = mean_l ||S^(l)||_F^2 / (B*H)

    Targeted (target_condition provided):
        At each denoising step, a no-grad pass with target_condition is run at
        the *same* xt and with the *same* perturbed gt_frames as the base pass.
        L = 1 - cosine_similarity(flat(S_base), flat(S_target))
        Both vectors are computed from identical conditioning, so same-prompt
        gives L≈0 and the gradient is well-defined from the start.
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

    # ---- Diagnostic: compare conditions once on first call ----
    global _DIAG_DONE
    if not _DIAG_DONE and target_cond_live is not None:
        _DIAG_DONE = True
        _diag_compare_conditions(condition_live, target_cond_live,
                                 label_a="condition_live", label_b="target_cond_live")

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
        t         = torch.rand(1).item()
        xt        = ((1 - t) * latent + t * noise).to(**model.tensor_kwargs)
        timesteps = torch.full((B, T_lat), t, device=latent.device, dtype=latent.dtype)

        # ---- Diagnostic: self-consistency check (run base cond twice) ----
        _self_check = (not _DIAG_DONE) or (target_cond_live is not None)
        _self_check = False   # toggle True to enable
        if _self_check:
            n_before = len(loss_terms)
            with torch.no_grad():
                model.denoise(noise=noise, xt_B_C_T_H_W=xt,
                              timesteps_B_T=timesteps, condition=condition_live)
            run1 = list(loss_terms[n_before:])
            del loss_terms[n_before:]
            n_before = len(loss_terms)
            with torch.no_grad():
                model.denoise(noise=noise, xt_B_C_T_H_W=xt,
                              timesteps_B_T=timesteps, condition=condition_live)
            run2 = list(loss_terms[n_before:])
            del loss_terms[n_before:]
            print("\n[DIAG] Self-consistency check (same condition, two identical passes):")
            _diag_compare_score_lists(run1, run2, label_a="run1", label_b="run2")

        if target_cond_live is not None:
            n_before = len(loss_terms)
            with torch.no_grad():
                model.denoise(noise=noise, xt_B_C_T_H_W=xt,
                              timesteps_B_T=timesteps, condition=target_cond_live)
            target_terms = list(loss_terms[n_before:])
            del loss_terms[n_before:]

        model.denoise(noise=noise, xt_B_C_T_H_W=xt, timesteps_B_T=timesteps,
                      condition=condition_live)

        # ---- Diagnostic: compare base vs target score tensors ----
        if target_cond_live is not None and not getattr(compute_crossattn_loss, "_score_diag_done", False):
            compute_crossattn_loss._score_diag_done = True
            _diag_compare_score_lists(loss_terms, target_terms,
                                      label_a="base_scores", label_b="target_scores")
    else:
        dt  = 1.0 / denoise_steps
        x_t = noise.float()

        for step_i in range(denoise_steps):
            t         = 1.0 - step_i * dt
            xt        = x_t.to(**model.tensor_kwargs)
            timesteps = torch.full((B, T_lat), t, device=latent.device, dtype=latent.dtype)

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
            x_t = x_t - dt * v_pred.float()

    def _cleanup():
        restore()
        if detach_hook is not None:
            detach_hook.remove()

    if target_cond_live is not None:
        adv_vec = torch.cat([s.flatten() for s in loss_terms])
        tgt_vec = torch.cat([s.detach().flatten() for s in target_terms])
        cos_sim = F.cosine_similarity(adv_vec.unsqueeze(0), tgt_vec.unsqueeze(0)).squeeze()
        loss = 1.0 - cos_sim
    else:
        loss = torch.stack(
            [s.pow(2).sum() / (s.shape[0] * s.shape[3]) for s in loss_terms]
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
    # 6c. Diagnostic: compare condition vs target_condition_template fields
    #     before _build_live is applied (raw templates, not yet gt_frames-updated)
    # ------------------------------------------------------------------
    if target_condition_template is not None:
        print("\n[DIAG] Pre-_build_live: condition template vs target_condition_template")
        _diag_compare_conditions(condition, target_condition_template,
                                 label_a="condition", label_b="target_condition_template")

    # ------------------------------------------------------------------
    # 7. PGD optimisation
    # ------------------------------------------------------------------
    print(f"\nStarting PGD optimisation "
          f"({'targeted' if target_condition_template is not None else 'untargeted'})...")

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

    out_dir = WM_ROOT / "attack" / "outputs" / f"crossattn_{save_time}"
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
    parser.add_argument("--alpha",           type=float, default=2/255,
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
