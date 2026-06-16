"""
attack_crossattn_v2.py

Corrected attack machinery for the LIBERO Safety setup.  Sits alongside
attack_crossattn.py so the v1 attack stays byte-identical and the two can
be A/B compared.

Differences from attack_crossattn.py
------------------------------------
1.  Frozen target reference.  v1 built the target attention pattern from
    (adv_latent, target_prompt) — both branches moved with the perturbation
    every PGD step, so the optimizer was effectively chasing "make
    attention insensitive to the prompt" rather than "make it look like
    the target prompt".  v2 builds the target reference from
    (clean_latent, target_prompt), recomputed fresh at the new (noise, t)
    each step but never touched by delta.  Applied to all cross-attn loss
    types (cosine, l2, kl).

2.  Matched xt construction.  In the multi-prompt forward passes, the adv
    branch's xt is now built from adv_latent and the target branch's xt
    from clean_latent, both with the same (noise, t).  v1 used adv_latent
    for both.

3.  Noise resampling.  Fresh noise is drawn every PGD step.  v1's
    targeted path forced noise_seed=0 so the perturbation overfit to one
    trajectory.

4.  New loss type "velocity".  Drops cross-attn hooks entirely:
        L = MSE( v(adv_latent, benign_prompt; xt_adv, t),
                 v(clean_latent, target_prompt; xt_tgt, t).detach() )
    Single-step random-t only (denoise_steps and CFG are ignored for this
    loss).  Significantly lower peak VRAM than the cross-attn paths
    because there are no fp32 (B,T,S,N,H) activation tensors live in
    the backward graph.

Reused from attack_crossattn.py: install_crossattn_hooks, pad_video,
load_full_video, _move_condition_to_device, compute_sim_mask,
analyze_sim_masks.
"""

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
from probing.test_vae_encoder import load_and_preprocess_video, normalize_video, resize_input  # noqa: E402
from attack.white_box import pgd, freq_pgd, lab_freq_pgd, lab_pgd
from attack.eval_wm import eval, eval_latent
from attack.shared_config import (
    ckpt_path, experiment_name, config_file
)

# Reused machinery from v1
from attack_crossattn import (  # noqa: E402
    install_crossattn_hooks,
    pad_video,
    load_full_video,
    _move_condition_to_device,
    compute_sim_mask,
    analyze_sim_masks,
)


# ---------------------------------------------------------------------------
# Corrected attack loss
# ---------------------------------------------------------------------------

def compute_attack_loss(
    model, x_padded, condition, T_tok,
    clean_latent, target_condition,
    layer_start=0, layer_end=None,
    skip_latent=False, denoise_steps=1,
    uncondition=None, guidance_scale=7.0,
    num_latent_conditional_frames=2, sim_masks=None,
    loss_type="cosine", dit_device=None,
    probe_state=None,
    target_ref="clean",
):
    """
    Returns (loss, cleanup_fn).

    Required (vs v1):
      clean_latent       : precomputed VAE encode of the clean raw_padded.
                           Float32 tensor on the DiT device.  Used as
                           gt_frames AND as the x0 for xt construction on
                           the target reference branch.
      target_condition   : condition template for the target prompt.
                           Its gt_frames will be overwritten with
                           clean_latent inside this function.

    loss_type:
      cosine, l2, kl     : cross-attention-pattern matching (v1 loss
                           formulas), but with the target reference now
                           computed from (clean_latent, target_prompt)
                           and a matched xt.
      velocity           : MSE between v(adv, benign; xt_adv, t) and
                           v(clean, target; xt_tgt, t).detach().  Forces
                           single-step semantics; denoise_steps and
                           guidance_scale are ignored.

    probe_state : optional (t_norm, noise) tuple.  When provided, those
                  values are used in place of fresh random samples and
                  cross-attn loss is forced to single-step semantics.
                  Used by the held-out diagnostic probe in pgd / lab_pgd.

    target_ref  : "clean" (default) → target branch uses clean_latent as
                  gt_frames and as the x0 in xt_tgt construction.  The
                  loss compares (adv, benign) against the frozen
                  (clean, target) reference.
                  "adv" → target branch uses adv_latent.detach() instead,
                  i.e. both branches share the same image and the loss
                  is purely the cross-prompt velocity/attention gap.
                  Recommended when the v2 loss is dominated by
                  image-perturbation noise as δ grows.
    """
    compute_dtype = next(model.net.parameters()).dtype

    if dit_device is not None:
        torch.cuda.set_device(dit_device)

    # Encode adv input (gradient flows through tokenizer when skip_latent=False)
    if skip_latent:
        adv_latent = x_padded
    else:
        adv_latent = model.tokenizer.encode(x_padded.to(compute_dtype)).contiguous().float()

    if dit_device is not None and adv_latent.device != torch.device(dit_device):
        adv_latent = adv_latent.to(dit_device)
    if clean_latent.device != adv_latent.device:
        clean_latent = clean_latent.to(adv_latent.device)

    B, C, T_lat, H_lat, W_lat = adv_latent.shape

    def _patch_gt_frames(cond, lat):
        cond_dict = cond.to_dict(skip_underscore=False)
        cond_dict['gt_frames'] = lat.to(compute_dtype)
        return type(cond)(**cond_dict)

    # target_ref selects what the target branch is conditioned on:
    #   "clean" → frozen (clean_latent, target_prompt) reference
    #   "adv"   → moving (adv_latent.detach(), target_prompt) reference,
    #             cancels the image-perturbation component of the loss
    if target_ref == "adv":
        target_x0 = adv_latent.detach()
    else:
        target_x0 = clean_latent

    condition_live   = _patch_gt_frames(condition,        adv_latent)
    target_cond_live = _patch_gt_frames(target_condition, target_x0)
    uncondition_live = _patch_gt_frames(uncondition,      adv_latent) if uncondition is not None else None

    # Fresh noise every call — gives EoT-style averaging across PGD steps.
    # If probe_state is set, pin (t, noise) instead so the diagnostic series
    # is a stable function of delta alone.
    if probe_state is not None:
        probe_t_norm, probe_noise = probe_state
        noise = probe_noise.to(adv_latent.device)
    else:
        probe_t_norm = None
        noise = torch.randn_like(adv_latent)

    # -------------------------------------------------------------------
    # Velocity loss: single-step, no hooks
    # -------------------------------------------------------------------
    if loss_type == "velocity":
        t_norm = probe_t_norm if probe_t_norm is not None else float(torch.rand(1).item())
        timesteps = torch.full(
            (B, T_lat), t_norm * 1000.0,
            device=adv_latent.device, dtype=adv_latent.dtype,
        )

        # Target velocity at (clean_latent, target_prompt), no grad
        with torch.no_grad():
            xt_tgt = ((1 - t_norm) * target_x0 + t_norm * noise).to(**model.tensor_kwargs)
            v_tgt = model.denoise(
                noise=noise, xt_B_C_T_H_W=xt_tgt,
                timesteps_B_T=timesteps, condition=target_cond_live,
            ).float().detach()

        # Adv velocity at (adv_latent, benign_prompt), with grad
        xt_adv = ((1 - t_norm) * adv_latent + t_norm * noise).to(**model.tensor_kwargs)
        v_adv = model.denoise(
            noise=noise, xt_B_C_T_H_W=xt_adv,
            timesteps_B_T=timesteps, condition=condition_live,
        ).float()

        loss = F.mse_loss(v_adv, v_tgt)

        def _cleanup():
            pass
        return loss, _cleanup

    # -------------------------------------------------------------------
    # Cross-attention pattern loss (cosine / l2 / kl)
    # -------------------------------------------------------------------
    loss_terms = []
    restore = install_crossattn_hooks(model.net, T_tok, loss_terms, layer_start, layer_end)

    n_blocks = len(model.net.blocks)
    detach_hook = None
    effective_end = layer_end if layer_end is not None else n_blocks - 1
    if effective_end < n_blocks - 1:
        def _detach(_m, _inp, output):
            if isinstance(output, tuple):
                return (output[0].detach(),) + output[1:]
            return output.detach()
        detach_hook = model.net.blocks[effective_end].register_forward_hook(_detach)

    target_terms = []

    # Probe always runs single-step so the diagnostic is self-consistent
    # regardless of --denoise_steps.
    use_single_step = (denoise_steps == 1) or (probe_state is not None)

    if use_single_step:
        t_norm = probe_t_norm if probe_t_norm is not None else float(torch.rand(1).item())
        timesteps = torch.full(
            (B, T_lat), t_norm * 1000.0,
            device=adv_latent.device, dtype=adv_latent.dtype,
        )

        xt_adv = ((1 - t_norm) * adv_latent + t_norm * noise).to(**model.tensor_kwargs)
        xt_tgt = ((1 - t_norm) * target_x0  + t_norm * noise).to(**model.tensor_kwargs)

        # Target ref pass — gt_frames=target_x0 + target prompt — hooks → target_terms
        n_before = len(loss_terms)
        with torch.no_grad():
            model.denoise(
                noise=noise, xt_B_C_T_H_W=xt_tgt,
                timesteps_B_T=timesteps, condition=target_cond_live,
            )
        target_terms = list(loss_terms[n_before:])
        del loss_terms[n_before:]

        # Adv pass — adv latent + benign prompt — hooks → loss_terms
        model.denoise(
            noise=noise, xt_B_C_T_H_W=xt_adv,
            timesteps_B_T=timesteps, condition=condition_live,
        )
    else:
        # Two parallel Euler trajectories sharing (noise, scheduler timesteps).
        # adv chain: gradient-connected, conditioned on adv_latent + benign.
        # tgt chain: no_grad, conditioned on clean_latent + target.
        model.sample_scheduler.set_timesteps(denoise_steps, device=adv_latent.device, shift=5.0)
        sched_ts = model.sample_scheduler.timesteps

        # Initialise both chains at t=sched_ts[0] from their respective x0's.
        # Without this, x_t starts at pure noise, which is inconsistent with how
        # denoise_steps==1 interpolates between latent and noise.
        t0 = (sched_ts[0] / 1000.0).item()
        adv_xt = ((1 - t0) * adv_latent + t0 * noise).float()
        with torch.no_grad():
            tgt_xt = ((1 - t0) * target_x0 + t0 * noise).float()

        for step_i in range(denoise_steps):
            t_model = sched_ts[step_i]
            t_next  = sched_ts[step_i + 1] if step_i + 1 < denoise_steps else sched_ts.new_tensor(0.0)
            timesteps = t_model.reshape(1, 1).expand(B, 1).to(
                device=adv_latent.device, dtype=adv_latent.dtype,
            )

            # Target ref pass (no grad) — uses clean trajectory
            n_before = len(loss_terms)
            with torch.no_grad():
                v_tgt = model.denoise(
                    noise=noise, xt_B_C_T_H_W=tgt_xt.to(**model.tensor_kwargs),
                    timesteps_B_T=timesteps, condition=target_cond_live,
                )
            target_terms.extend(loss_terms[n_before:])
            del loss_terms[n_before:]

            # CFG uncond pass on adv chain (velocity only, hooks discarded)
            uncond_v = None
            if uncondition_live is not None:
                n_before = len(loss_terms)
                with torch.no_grad():
                    uncond_v = model.denoise(
                        noise=noise, xt_B_C_T_H_W=adv_xt.to(**model.tensor_kwargs),
                        timesteps_B_T=timesteps, condition=uncondition_live,
                    )
                del loss_terms[n_before:]

            # Adv pass on adv chain (gradient-connected) — hooks → loss_terms
            cond_v = model.denoise(
                noise=noise, xt_B_C_T_H_W=adv_xt.to(**model.tensor_kwargs),
                timesteps_B_T=timesteps, condition=condition_live,
            )

            v_pred_adv = (cond_v + guidance_scale * (cond_v - uncond_v)
                          if uncond_v is not None else cond_v)
            dt_norm = (t_model - t_next).item() / 1000.0
            adv_xt = adv_xt - dt_norm * v_pred_adv.float()
            with torch.no_grad():
                tgt_xt = tgt_xt - dt_norm * v_tgt.float()

    def _cleanup():
        restore()
        if detach_hook is not None:
            detach_hook.remove()

    # ---- Reduce to scalar loss --------------------------------------
    if loss_type == "l2":
        if sim_masks is not None:
            loss = torch.stack([
                ((s - t.detach()) * m.unsqueeze(3)).pow(2).sum() / (s.shape[0] * s.shape[4])
                for s, t, m in zip(loss_terms, target_terms, sim_masks)
            ]).mean()
        else:
            loss = torch.stack([
                (s - t.detach()).pow(2).sum() / (s.shape[0] * s.shape[4])
                for s, t in zip(loss_terms, target_terms)
            ]).mean()
    elif loss_type == "kl":
        loss = torch.stack([
            F.kl_div(s.log(), t.detach(), reduction='sum') / (s.shape[0] * s.shape[4])
            for s, t in zip(loss_terms, target_terms)
        ]).mean()
    else:  # cosine
        if sim_masks is not None:
            tgt_vec = torch.cat([(s * m.unsqueeze(3)).detach().flatten()
                                  for s, m in zip(target_terms, sim_masks)])
            del target_terms
            adv_vec = torch.cat([(s * m.unsqueeze(3)).flatten()
                                  for s, m in zip(loss_terms, sim_masks)])
        else:
            tgt_vec = torch.cat([s.detach().flatten() for s in target_terms])
            del target_terms
            adv_vec = torch.cat([s.flatten() for s in loss_terms])
        del loss_terms
        loss = 1.0 - F.cosine_similarity(
            adv_vec.unsqueeze(0), tgt_vec.unsqueeze(0)
        ).squeeze()

    return loss, _cleanup


# ---------------------------------------------------------------------------
# Per-video attack (v2)
# ---------------------------------------------------------------------------

def attack_single_video(
    inference, model, video_path, prompt,
    H, W, frames_to_extract, required_pixel_frames,
    T_tok, layer_start, layer_end, compute_dtype, args,
    vae_device="cuda", dit_device="cuda", save_time=None,
    preloaded_raw_padded=None,
    benign_prompt_variants=None,
    target_prompt_variants=None,
):
    """
    v2 attack:
      - Precomputes clean_latent once and freezes the target reference.
      - Resamples noise each PGD step (EoT-style).
      - Supports loss_type="velocity" in addition to cosine/l2/kl.
    """
    if save_time is None:
        save_time = int(time.time())

    if inference.offload_text_encoder:
        if model.text_encoder is not None:
            if hasattr(model.text_encoder, "model") and model.text_encoder.model is not None:
                model.text_encoder.model = model.text_encoder.model.to(vae_device).eval()
        if _t5_mod.cosmos_encoder is not None:
            _t5_mod.cosmos_encoder.text_encoder = \
                _t5_mod.cosmos_encoder.text_encoder.to(vae_device).eval()

    if preloaded_raw_padded is not None:
        raw_padded = preloaded_raw_padded.to(vae_device)
        args._extend_full_video = None
        print(f"\nUsing preloaded input : {video_path}")
    else:
        print(f"\nLoading video : {video_path}")
        if args.extend:
            full_video_uint8 = load_full_video(str(video_path), [H, W])
            raw_full = normalize_video(full_video_uint8, device=vae_device)
            args._extend_full_video = raw_full.cpu()
            print(f"Full video shape         : {raw_full.shape}")
            raw_padded = pad_video(raw_full, frames_to_extract, required_pixel_frames)
        else:
            video_uint8 = load_and_preprocess_video(
                str(video_path), [H, W], frames_to_extract
            )
            raw_cond = normalize_video(video_uint8, device=vae_device)
            args._extend_full_video = None
            print(f"Conditioning frames shape: {raw_cond.shape}")
            raw_padded = pad_video(raw_cond, frames_to_extract, required_pixel_frames)
    print(f"Padded video shape       : {raw_padded.shape}")

    # ------------------------------------------------------------------
    # Build T5 conditions (text encoder still on GPU here)
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

    benign_var_data_batches: list = []
    target_var_data_batches: list = []
    if getattr(args, "rand_prompts", False):
        for tag, variants, store in [
            ("benign", benign_prompt_variants or [], benign_var_data_batches),
            ("target", target_prompt_variants or [], target_var_data_batches),
        ]:
            if variants:
                print(f"Building T5 embeddings for {len(variants)} {tag} prompt variants...")
                for vp in variants:
                    db = inference._get_data_batch_input(
                        video=video_bf16,
                        prompt=vp,
                        num_conditional_frames=args.num_latent_conditional_frames,
                        negative_prompt=negative_prompt,
                        use_neg_prompt=True,
                    )
                    db["video"]             = video_bf16
                    db[IS_PREPROCESSED_KEY] = True
                    store.append(db)

    # ------------------------------------------------------------------
    # Offload text encoder, ensure tokenizer & DiT are on their devices
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
            model.tokenizer.encoder = model.tokenizer.encoder.to(vae_device)
        torch.cuda.empty_cache()

    if inference.offload_diffusion_model:
        model.net = model.net.to(dit_device)
        if hasattr(model, "conditioner") and model.conditioner is not None:
            model.conditioner = model.conditioner.to(dit_device)
        torch.cuda.empty_cache()
    elif vae_device != dit_device:
        print(f"split_gpus: moving DiT → {dit_device}, VAE stays on {vae_device}")
        model.net = model.net.to(dit_device)
        if hasattr(model, "conditioner") and model.conditioner is not None:
            model.conditioner = model.conditioner.to(dit_device)
        torch.cuda.empty_cache()

    if vae_device != dit_device:
        model.tensor_kwargs["device"] = dit_device
        if hasattr(model, "tensor_kwargs_fp32"):
            model.tensor_kwargs_fp32["device"] = dit_device
        torch.cuda.set_device(dit_device)

    # ------------------------------------------------------------------
    # Freeze model params; enable VAE grad for pixel-space attacks
    # ------------------------------------------------------------------
    for p in model.net.parameters():
        p.requires_grad_(False)
    for p in model.tokenizer.model.model.parameters():
        p.requires_grad_(False)
    model.tokenizer.enable_grad = True

    # ------------------------------------------------------------------
    # Build base condition / uncondition
    # ------------------------------------------------------------------
    print("Building condition...")
    with torch.no_grad():
        _, _, condition = model.get_data_and_condition(data_batch)
        _, uncondition = model.conditioner.get_condition_with_negative_prompt(data_batch)
    condition = condition.edit_for_inference(
        is_cfg_conditional=True,
        num_conditional_frames=args.num_latent_conditional_frames,
    )
    uncond_dict = uncondition.to_dict(skip_underscore=False)
    uncond_dict['gt_frames'] = condition.gt_frames
    uncondition = type(uncondition)(**uncond_dict)
    uncondition = uncondition.edit_for_inference(
        is_cfg_conditional=False,
        num_conditional_frames=args.num_latent_conditional_frames,
    )

    # Build target condition template (gt_frames will be overwritten with
    # clean_latent inside compute_attack_loss).
    target_condition_template = None
    if target_data_batch is not None:
        print(f"Building target condition for: \"{args.target_prompt}\"")
        tgt_dict = condition.to_dict(skip_underscore=False)
        tgt_dict['crossattn_emb'] = target_data_batch['t5_text_embeddings']
        target_condition_template = type(condition)(**tgt_dict)

    benign_cond_variants: list = []
    target_cond_variants: list = []
    if getattr(args, "rand_prompts", False):
        for db in benign_var_data_batches:
            var_dict = condition.to_dict(skip_underscore=False)
            var_dict["crossattn_emb"] = db["t5_text_embeddings"]
            benign_cond_variants.append(type(condition)(**var_dict))
        for db in target_var_data_batches:
            var_dict = condition.to_dict(skip_underscore=False)
            var_dict["crossattn_emb"] = db["t5_text_embeddings"]
            target_cond_variants.append(type(condition)(**var_dict))
        if benign_cond_variants:
            print(f"Built {len(benign_cond_variants)} benign + "
                  f"{len(target_cond_variants)} target condition variants.")

    if vae_device != dit_device:
        _move_condition_to_device(condition,   dit_device)
        _move_condition_to_device(uncondition, dit_device)
        if target_condition_template is not None:
            _move_condition_to_device(target_condition_template, dit_device)
        for vc in benign_cond_variants:
            _move_condition_to_device(vc, dit_device)
        for vc in target_cond_variants:
            _move_condition_to_device(vc, dit_device)

    torch.cuda.empty_cache()
    print(f"\nStarting PGD optimisation "
          f"({'targeted' if target_condition_template is not None else 'untargeted'}, "
          f"loss={args.loss_type})...")

    # ------------------------------------------------------------------
    # Precompute clean_latent — the frozen reference x0 for the target
    # branch.  Computed once here so that the target attention/velocity
    # never moves with delta.
    # ------------------------------------------------------------------
    clean_latent = None
    sim_masks = None
    if target_condition_template is not None:
        with torch.no_grad():
            clean_latent = model.tokenizer.encode(
                raw_padded.to(compute_dtype)
            ).contiguous().float()
            if vae_device != dit_device:
                clean_latent = clean_latent.to(dit_device)
        print(f"Clean latent (frozen target ref) shape: {clean_latent.shape}")

        # ----------------------------------------------------------
        # One-shot calibration: compute ||v(clean, benign) - v(clean, target)||²
        # at (t=0.4, fresh noise).  This is the "pure prompt effect" floor —
        # the cross-prompt velocity gap at the clean image, with no
        # perturbation in play.  Useful for interpreting the probe series:
        #   - With --target_ref clean, the probe will start near this value
        #     and may grow as the image-perturbation term takes over.
        #   - With --target_ref adv, the probe stays bounded near this
        #     magnitude because the image term cancels.
        # ----------------------------------------------------------
        with torch.no_grad():
            _cal_t = 0.4
            _cal_noise = torch.randn_like(clean_latent)
            _B, _, _T_lat, _, _ = clean_latent.shape
            _cal_ts = torch.full(
                (_B, _T_lat), _cal_t * 1000.0,
                device=clean_latent.device, dtype=clean_latent.dtype,
            )
            _xt_cal = ((1 - _cal_t) * clean_latent + _cal_t * _cal_noise).to(**model.tensor_kwargs)

            def _live_for_cal(cond, lat):
                cd = cond.to_dict(skip_underscore=False)
                cd['gt_frames'] = lat.to(compute_dtype)
                return type(cond)(**cd)

            _v_b = model.denoise(
                noise=_cal_noise, xt_B_C_T_H_W=_xt_cal,
                timesteps_B_T=_cal_ts,
                condition=_live_for_cal(condition, clean_latent),
            ).float()
            _v_t = model.denoise(
                noise=_cal_noise, xt_B_C_T_H_W=_xt_cal,
                timesteps_B_T=_cal_ts,
                condition=_live_for_cal(target_condition_template, clean_latent),
            ).float()
            _cross_floor = F.mse_loss(_v_b, _v_t).item()
        print(f"[probe calibration] ||v(clean, benign) - v(clean, target)||² = "
              f"{_cross_floor:.6f}  (at t=0.4, fresh noise)")
        print(f"[probe calibration] target_ref={getattr(args, 'target_ref', 'clean')}  "
              f"— if probe loss quickly exceeds the floor, image-noise dominates "
              f"(try --target_ref adv)")

        # Sim masks are only meaningful for cross-attn losses.
        if args.mask and args.loss_type in ("cosine", "l2", "kl"):
            sim_masks = compute_sim_mask(
                model, clean_latent, condition, target_condition_template, T_tok,
                layer_start=layer_start,
                layer_end=layer_end,
                num_latent_conditional_frames=args.num_latent_conditional_frames,
                noise_seed=0,
            )
            analyze_sim_masks(sim_masks, spatial_grid=27, save_dir="sim_analysis")

    # Velocity loss can't use sim_masks (no attention tensors).  Warn and
    # ignore --mask if requested with --loss_type velocity.
    if args.loss_type == "velocity" and args.mask:
        print("[v2] --mask has no effect with --loss_type velocity (no attention tensors).")

    # ------------------------------------------------------------------
    # Held-out diagnostic probe (fixed t & noise, canonical prompts).
    # The per-step loss is noisy because (t, noise) are resampled every
    # step.  The probe re-evaluates the loss at a frozen (t, noise) pair
    # every `probe_every` steps, so its series shows real convergence
    # (monotone decrease if the gradient is doing useful work) without
    # the per-step variance.
    # ------------------------------------------------------------------
    probe_every = int(getattr(args, "probe_every", 0))
    probe_fn = None
    if probe_every > 0 and target_condition_template is not None:
        probe_t_norm = 0.4   # mid-trajectory; semantic work happens here
        _probe_noise_holder = {"noise": None}

        def probe_fn(x_pert_in):
            if _probe_noise_holder["noise"] is None:
                if args.skip_latent:
                    shape_ref = x_pert_in
                else:
                    with torch.no_grad():
                        shape_ref = model.tokenizer.encode(
                            x_pert_in.to(compute_dtype)
                        ).contiguous().float()
                if vae_device != dit_device:
                    shape_ref = shape_ref.to(dit_device)
                _probe_noise_holder["noise"] = torch.randn_like(shape_ref)

            with torch.no_grad():
                loss, cleanup = compute_attack_loss(
                    model, x_pert_in, condition, T_tok,
                    clean_latent=clean_latent,
                    target_condition=target_condition_template,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    skip_latent=args.skip_latent,
                    denoise_steps=1,
                    uncondition=None,
                    guidance_scale=args.guidance_scale,
                    num_latent_conditional_frames=args.num_latent_conditional_frames,
                    sim_masks=sim_masks,
                    loss_type=args.loss_type,
                    dit_device=dit_device if vae_device != dit_device else None,
                    probe_state=(probe_t_norm, _probe_noise_holder["noise"]),
                    target_ref=getattr(args, "target_ref", "clean"),
                )
                val = float(loss.item())
            if cleanup is not None:
                cleanup()
            return val

        print(f"[probe] enabled — fixed t={probe_t_norm}, fresh noise on first call, "
              f"cadence every {probe_every} PGD steps")
    elif probe_every > 0:
        print("[probe] disabled (no target_condition to compare against)")

    _use_variants = getattr(args, "rand_prompts", False) and bool(benign_cond_variants)

    def loss_fn(x, y):
        c = random.choice(benign_cond_variants) if _use_variants else condition
        t = (random.choice(target_cond_variants)
             if (_use_variants and target_cond_variants)
             else target_condition_template)
        # uncondition only matters for the multi-step CFG path on cross-attn losses.
        use_uncond = (
            args.loss_type in ("cosine", "l2", "kl")
            and args.denoise_steps > 1
        )
        return compute_attack_loss(
            model, x, c, T_tok,
            clean_latent=clean_latent,
            target_condition=t,
            layer_start=layer_start,
            layer_end=layer_end,
            skip_latent=args.skip_latent,
            denoise_steps=args.denoise_steps,
            uncondition=uncondition if use_uncond else None,
            guidance_scale=args.guidance_scale,
            num_latent_conditional_frames=args.num_latent_conditional_frames,
            sim_masks=sim_masks,
            loss_type=args.loss_type,
            dit_device=dit_device if vae_device != dit_device else None,
            target_ref=getattr(args, "target_ref", "clean"),
        )

    if getattr(args, "lf_attack", False):
        if args.skip_latent:
            raise ValueError("--lf_attack requires pixel-space optimisation")
        # v2: pure LAB-space PGD.  Frequency-space machinery is intentionally
        # bypassed for now (the v1 paths had a `freq_oly` typo and other
        # issues); --freq_cutoff and --freq_pixel_eps are silently ignored.
        # NOTE: --alpha here is interpreted as a fraction of each LAB
        # channel's budget (e.g. alpha=0.1 with lab_budget_ab=20 → step=2.0).
        if args.freq_cutoff != 0.1 or args.freq_pixel_eps != 0.1:
            print("[v2] --freq_cutoff / --freq_pixel_eps are ignored by lab_pgd; "
                  "frequency-space attacks are on hold.")
        x_adv = lab_pgd(
            raw_padded, 0, lambda x: x, loss_fn,
            args.steps, args.alpha,
            num_frames=frames_to_extract,
            momentum=0.9,
            lab_budget_L=args.lab_budget_L,
            lab_budget_ab=args.lab_budget_ab,
            probe_fn=probe_fn,
            probe_every=probe_every,
        )
    elif args.skip_latent:
        with torch.no_grad():
            latent = model.tokenizer.encode(
                raw_padded.to(compute_dtype)
            ).contiguous().float()
        x_adv = pgd(
            latent, 0, lambda x: x, loss_fn,
            args.steps, args.alpha, args.eps,
            args.num_latent_conditional_frames,
            probe_fn=probe_fn,
            probe_every=probe_every,
        )
    else:
        x_adv = pgd(
            raw_padded, 0, lambda x: x, loss_fn,
            args.steps, args.alpha, args.eps, frames_to_extract,
            pixel_min=-1.0, pixel_max=1.0,
            probe_fn=probe_fn,
            probe_every=probe_every,
        )

    def to_uint8_frames(t):
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
