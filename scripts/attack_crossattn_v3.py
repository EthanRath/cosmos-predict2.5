"""
attack_crossattn_v3.py

Experimental extension of attack_crossattn_v2.py that adds a *spatial
weighting mask* to the velocity loss.  v2's velocity MSE weights every
latent cell equally; v3 lets us upweight the region where the unsafe
object lives (and optionally de-emphasise the safe object's region),
ported from the road-mask idea in the AFM paper (arxiv 2605.00880).

v2 behaviour is preserved exactly:
  * loss_type in {cosine, l2, kl} → delegated to v2.compute_attack_loss.
  * loss_type == "velocity" with no mask (velocity_mask=None) → identical
    to v2's velocity branch.
  * loss_type == "velocity" with a mask → weighted MSE
        loss = ((v_adv - v_tgt)**2 * mask).mean()
    where `mask` is broadcast to (B, C, T_lat, H_lat, W_lat) and is
    normalised so mean(mask)=1 (probe values stay comparable to v2).

Recommended commands — copy these as single lines.

# v3 velocity + LAB attack with unsafe-object mask (default weight 3.0, radius 3 latent cells)
CUDA_VISIBLE_DEVICES=1 python cosmos-predict2.5/scripts/attack_libero_safety_v3.py --scene_name all --dataset_dir datasets/ --attack_num 10 --steps 300 --loss_type velocity --lf_attack --num_latent_video_frames 9 --num_latent_conditional_frames 2 --rand_prompts --alpha 0.05 --lab_budget_L 5.0 --lab_budget_ab 20.0 --mask_velocity

# v3 velocity + LAB + de-emphasise safe object region too
CUDA_VISIBLE_DEVICES=1 python cosmos-predict2.5/scripts/attack_libero_safety_v3.py --scene_name all --dataset_dir datasets/ --attack_num 10 --steps 300 --loss_type velocity --lf_attack --num_latent_video_frames 9 --num_latent_conditional_frames 2 --rand_prompts --alpha 0.05 --lab_budget_L 5.0 --lab_budget_ab 20.0 --mask_velocity --mask_unsafe_weight 4.0 --mask_safe_weight 0.5

# v3 ablation: no mask (should reproduce v2 velocity behaviour bit-for-bit modulo RNG)
CUDA_VISIBLE_DEVICES=1 python cosmos-predict2.5/scripts/attack_libero_safety_v3.py --scene_name all --dataset_dir datasets/ --attack_num 10 --steps 300 --loss_type velocity --lf_attack --num_latent_video_frames 9 --num_latent_conditional_frames 2 --rand_prompts --alpha 0.05 --lab_budget_L 5.0 --lab_budget_ab 20.0
"""

import json
import random
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
from attack.white_box import pgd, lab_pgd
from attack.eval_wm import eval, eval_latent
from attack.shared_config import (
    ckpt_path, experiment_name, config_file
)

# Reused machinery
from attack_crossattn import (  # noqa: E402
    install_crossattn_hooks,
    pad_video,
    load_full_video,
    _move_condition_to_device,
    compute_sim_mask,
    analyze_sim_masks,
)
from attack_crossattn_v2 import (  # noqa: E402
    compute_attack_loss as _compute_attack_loss_v2,
)


# ---------------------------------------------------------------------------
# Spatial mask construction
# ---------------------------------------------------------------------------
def build_unsafe_spatial_mask(
    T_lat, H_lat, W_lat,
    unsafe_xy=None,
    safe_xy=None,
    unsafe_weight=3.0,
    safe_weight=1.0,
    radius_lat=3.0,
    world_to_pix_scale=1.667,
    device="cuda",
    dtype=torch.float32,
):
    """Build a (1, 1, T_lat, H_lat, W_lat) latent-cell weight tensor with
    Gaussian bumps centred on the world-projected pixel locations of the
    unsafe (and optional safe) object.

    The returned tensor is normalised to have mean 1.0, so plugging it
    into a weighted MSE yields a value on the same scale as plain MSE
    when the bumps are small.  Returns (mask, diag) where diag is a
    dict with the projected latent-cell centres for logging.

    World → image mapping (LIBERO default front-view camera, *uncalibrated*):
        norm_px = 0.5 + wy * scale     (world y → image x;   wy > 0 → right)
        norm_py = 0.5 + wx * scale     (world x → image y;   wx > 0 → near
                                         camera → bottom of image)
    The default scale 1.667 ≈ 1 / 0.6 maps world coords of ±0.6 to the
    image edge, which roughly matches the spread of values in the
    LIBERO Safety metadata (|wx|, |wy| ≲ 0.4).  If the projected centre
    is off-screen we clip into [0, 1].

    Args:
      T_lat, H_lat, W_lat : latent grid dimensions of clean_latent.
      unsafe_xy, safe_xy  : (world_x, world_y) tuples or None.
      unsafe_weight       : peak multiplier at the unsafe-object centre.
                            1.0 ⇒ no effect.
      safe_weight         : peak multiplier at the safe-object centre.
                            <1.0 ⇒ de-emphasise that region.
                            1.0 ⇒ no effect.
      radius_lat          : Gaussian sigma in latent cells (NOT pixels).
      world_to_pix_scale  : world-space → normalised-image scale factor.
    """
    def world_to_lat(wx, wy):
        norm_px = 0.5 + wy * world_to_pix_scale
        norm_py = 0.5 + wx * world_to_pix_scale
        norm_px = float(max(0.0, min(1.0, norm_px)))
        norm_py = float(max(0.0, min(1.0, norm_py)))
        cx = norm_px * (W_lat - 1)
        cy = norm_py * (H_lat - 1)
        return cx, cy

    ys = torch.arange(H_lat, dtype=dtype, device=device).view(H_lat, 1)
    xs = torch.arange(W_lat, dtype=dtype, device=device).view(1, W_lat)

    mask_2d = torch.ones((H_lat, W_lat), dtype=dtype, device=device)

    diag = {"unsafe_lat": None, "safe_lat": None,
            "unsafe_weight": unsafe_weight, "safe_weight": safe_weight,
            "radius_lat": radius_lat, "H_lat": H_lat, "W_lat": W_lat}

    if unsafe_xy is not None and unsafe_weight != 1.0:
        ux, uy = world_to_lat(*unsafe_xy)
        bump = torch.exp(-((ys - uy) ** 2 + (xs - ux) ** 2) / (2 * radius_lat ** 2))
        mask_2d = mask_2d + (unsafe_weight - 1.0) * bump
        diag["unsafe_lat"] = (ux, uy)

    if safe_xy is not None and safe_weight != 1.0:
        sx, sy = world_to_lat(*safe_xy)
        bump = torch.exp(-((ys - sy) ** 2 + (xs - sx) ** 2) / (2 * radius_lat ** 2))
        mask_2d = mask_2d + (safe_weight - 1.0) * bump
        diag["safe_lat"] = (sx, sy)

    mask_2d = mask_2d.clamp(min=1e-6)
    mask_2d = mask_2d / mask_2d.mean()

    mask = mask_2d.view(1, 1, 1, H_lat, W_lat).expand(1, 1, T_lat, H_lat, W_lat).contiguous()
    return mask, diag


# ---------------------------------------------------------------------------
# Spatial low-pass filter for velocity MSE
# ---------------------------------------------------------------------------
def _lowpass_velocity_mse(v1, v2, lp_cutoff=0.25, velocity_mask=None):
    """MSE computed only on spatially low-pass-filtered velocity fields.

    Filters v1 and v2 in the 2D spatial frequency domain, retaining only the
    lowest lp_cutoff fraction of spatial frequencies in each spatial dimension.
    This focuses the loss on coarse structural motion and ignores high-frequency
    texture/noise differences that are uncorrelated with prompt semantics.

    v1, v2        : (B, C, T, H, W) velocity fields
    lp_cutoff     : fraction of spatial frequencies to retain (0 < lp_cutoff <= 1)
    velocity_mask : optional (1, 1, T, H, W) spatial weight tensor with mean≈1
    """
    B, C, T, H, W = v1.shape
    N = B * C * T
    V1 = torch.fft.rfft2(v1.reshape(N, H, W))  # (N, H, W//2+1) complex
    V2 = torch.fft.rfft2(v2.reshape(N, H, W))

    H_fft, W_fft = V1.shape[-2], V1.shape[-1]  # H, W//2+1
    h_cut = max(1, int(H_fft * lp_cutoff))
    w_cut = max(1, int(W_fft * lp_cutoff))

    # Real mask broadcast over complex: keeps DC + positive low-freq rows,
    # plus the negative low-freq rows that wrap at the end of the H dimension.
    lp_mask = torch.zeros(H_fft, W_fft, dtype=v1.dtype, device=v1.device)
    lp_mask[:h_cut, :w_cut] = 1.0
    lp_mask[-h_cut:, :w_cut] = 1.0

    v1_lp = torch.fft.irfft2(V1 * lp_mask, s=(H, W)).reshape(B, C, T, H, W)
    v2_lp = torch.fft.irfft2(V2 * lp_mask, s=(H, W)).reshape(B, C, T, H, W)

    if velocity_mask is not None:
        m = velocity_mask.to(device=v1_lp.device, dtype=v1_lp.dtype)
        return ((v1_lp - v2_lp) ** 2 * m).mean()
    return F.mse_loss(v1_lp, v2_lp)


# ---------------------------------------------------------------------------
# Frequency-loss diagnostic  (--loss_freq_plot)
# ---------------------------------------------------------------------------
def _compute_freq_profile(v_adv, v_tgt, n_bins=40):
    """Bin velocity-MSE power by radial spatial frequency.

    Returns (bin_centers, mean_power, total_power, cumulative):
      bin_centers  : centre of each radial-frequency bin (normalised; 0 = DC)
      mean_power   : mean |Δv_FFT|² per FFT component in the bin
      total_power  : total |Δv_FFT|² summed over all components in the bin
      cumulative   : fraction [0,1] of total MSE energy up to and including
                     each bin (useful for reading off "X% of MSE is below freq F")
    All returned as np.ndarray on CPU.
    """
    diff = v_adv.float() - v_tgt.float()   # (B, C, T, H, W) already on CPU
    B, C, T, H, W = diff.shape
    N = B * C * T

    D = torch.fft.rfft2(diff.reshape(N, H, W))           # (N, H, W//2+1) complex
    power = (D.real ** 2 + D.imag ** 2).mean(dim=0)      # (H, W//2+1) mean over N

    # Normalised signed frequency for each rfft2 output cell
    ks_h = torch.arange(H, dtype=torch.float32)
    ks_h[H // 2 + 1:] -= H                               # wrap to signed [-H/2, H/2)
    freq_h = ks_h / H                                     # → [-0.5, 0.5)
    freq_w = torch.arange(W // 2 + 1, dtype=torch.float32) / W  # [0, 0.5]

    radial = torch.sqrt(freq_h.view(H, 1) ** 2 + freq_w.view(1, W // 2 + 1) ** 2)
    radial_np = radial.reshape(-1).numpy()
    power_np  = power.reshape(-1).numpy()

    max_r = float(radial_np.max())
    bin_edges   = np.linspace(0, max_r, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_power  = np.zeros(n_bins)
    total_power = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (radial_np >= lo) & (radial_np < hi)
        if i == n_bins - 1:
            in_bin |= (radial_np == hi)     # include exact max in last bin
        if in_bin.sum() > 0:
            mean_power[i]  = power_np[in_bin].mean()
            total_power[i] = power_np[in_bin].sum()

    cumulative = np.cumsum(total_power) / (total_power.sum() + 1e-12)
    return bin_centers, mean_power, total_power, cumulative


def _plot_freq_loss_profile(v_adv, v_tgt, step, out_dir, label="probe", n_bins=40):
    """Compute and save a two-panel frequency-profile PNG.

    Left  — mean |Δv_FFT|² per component vs. normalised radial frequency.
             A flat or rising curve here means high-freq noise dominates.
    Right — cumulative fraction of total MSE up to each frequency.
             If the curve reaches 90% well before the Nyquist corner, most
             MSE energy is in low-frequency (semantically meaningful) bands.
    File is saved to <out_dir>/freq_plots/freq_profile_<label>_step<NNNN>.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [freq_plot] matplotlib not available — skipping")
        return

    bin_centers, mean_power, total_power, cumulative = _compute_freq_profile(
        v_adv, v_tgt, n_bins=n_bins
    )
    bar_w = (bin_centers[1] - bin_centers[0]) * 0.85 if len(bin_centers) > 1 else 0.01

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Velocity-loss frequency profile  |  step {step}  |  {label}",
        fontsize=9, y=1.01,
    )

    ax1.bar(bin_centers, mean_power, width=bar_w, color="steelblue",
            alpha=0.75, edgecolor="none")
    ax1.set_xlabel("Normalised radial spatial frequency  (0 = DC,  ~0.7 = Nyquist corner)")
    ax1.set_ylabel("Mean |Δv_FFT|² per component")
    ax1.set_title("MSE density by frequency")
    ax1.grid(True, alpha=0.25, axis="y")

    ax2.plot(bin_centers, cumulative * 100, color="crimson", linewidth=1.5)
    ax2.fill_between(bin_centers, cumulative * 100, alpha=0.15, color="crimson")
    ax2.axhline(50, color="gray", linestyle="--", linewidth=0.8, label="50%")
    ax2.axhline(90, color="gray", linestyle=":", linewidth=0.8, label="90%")
    ax2.set_xlabel("Normalised radial spatial frequency")
    ax2.set_ylabel("Cumulative % of total MSE")
    ax2.set_title("Cumulative MSE fraction")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_dir = Path(out_dir) / "freq_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"freq_profile_{label}_step{step:04d}.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [freq_plot] saved: {out_path}")


# ---------------------------------------------------------------------------
# Loss with optional spatial mask
# ---------------------------------------------------------------------------
def compute_attack_loss_v3(
    model, x_padded, condition, T_tok,
    clean_latent, target_condition,
    layer_start=0, layer_end=None,
    skip_latent=False, denoise_steps=1,
    uncondition=None, guidance_scale=7.0,
    num_latent_conditional_frames=2, sim_masks=None,
    loss_type="cosine", dit_device=None,
    probe_state=None,
    target_ref="clean",
    velocity_mask=None,
    velocity_lp_cutoff=0.0,
    velocity_temporal_diff=False,
    velocity_callback=None,
    t_min=0.25,
    t_max=0.75,
):
    """Wrapper around v2.compute_attack_loss that adds an optional
    spatial weighting mask to the velocity-MSE.  Non-velocity loss types
    are delegated unchanged to v2.

    velocity_mask : optional (1, 1, T_lat, H_lat, W_lat) tensor with
                    mean ≈ 1, broadcast across batch/channel dims.
                    None ⇒ behaviour matches v2 velocity loss exactly.
    """
    if loss_type != "velocity":
        if velocity_mask is not None:
            # Mask only meaningful for velocity loss; silently ignored
            # for cross-attn losses (which already have sim_masks).
            pass
        return _compute_attack_loss_v2(
            model, x_padded, condition, T_tok,
            clean_latent=clean_latent,
            target_condition=target_condition,
            layer_start=layer_start, layer_end=layer_end,
            skip_latent=skip_latent, denoise_steps=denoise_steps,
            uncondition=uncondition, guidance_scale=guidance_scale,
            num_latent_conditional_frames=num_latent_conditional_frames,
            sim_masks=sim_masks,
            loss_type=loss_type,
            dit_device=dit_device,
            probe_state=probe_state,
            target_ref=target_ref,
        )

    # -- Velocity branch (replicates v2 with optional masked MSE) -------
    compute_dtype = next(model.net.parameters()).dtype
    if dit_device is not None:
        torch.cuda.set_device(dit_device)

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

    target_x0 = adv_latent.detach() if target_ref == "adv" else clean_latent
    condition_live   = _patch_gt_frames(condition,        adv_latent)
    target_cond_live = _patch_gt_frames(target_condition, target_x0)

    if probe_state is not None:
        probe_t_norm, probe_noise = probe_state
        noise = probe_noise.to(adv_latent.device)
        t_norm = probe_t_norm
    else:
        noise = torch.randn_like(adv_latent)
        t_norm = float(t_min + torch.rand(1).item() * (t_max - t_min))

    timesteps = torch.full(
        (B, T_lat), t_norm * 1000.0,
        device=adv_latent.device, dtype=adv_latent.dtype,
    )

    with torch.no_grad():
        xt_tgt = ((1 - t_norm) * target_x0 + t_norm * noise).to(**model.tensor_kwargs)
        v_tgt = model.denoise(
            noise=noise, xt_B_C_T_H_W=xt_tgt,
            timesteps_B_T=timesteps, condition=target_cond_live,
        ).float().detach()

    xt_adv = ((1 - t_norm) * adv_latent + t_norm * noise).to(**model.tensor_kwargs)
    v_adv = model.denoise(
        noise=noise, xt_B_C_T_H_W=xt_adv,
        timesteps_B_T=timesteps, condition=condition_live,
    ).float()

    # Validate spatial mask shape before slicing
    if velocity_mask is not None:
        if velocity_mask.shape[-3:] != (T_lat, H_lat, W_lat):
            raise ValueError(
                f"velocity_mask shape {tuple(velocity_mask.shape)} does not "
                f"match latent grid (..., {T_lat}, {H_lat}, {W_lat})"
            )

    # Always restrict to predicted (non-conditioning) frames — conditioning frames
    # use the same gt_frames regardless of prompt, so their velocity difference is
    # prompt-independent noise that pollutes the gradient signal.
    cond_f = num_latent_conditional_frames
    v_adv_use = v_adv[:, :, cond_f:]
    v_tgt_use = v_tgt[:, :, cond_f:]
    mask_use = velocity_mask[:, :, cond_f:] if velocity_mask is not None else None

    # Optional: match temporal differences (v[t+1]-v[t]) instead of absolute values.
    # Focuses the loss on how the predicted scene dynamics change across frames
    # rather than the static frame-level velocity predictions.
    if velocity_temporal_diff and v_adv_use.shape[2] >= 2:
        v_adv_use = v_adv_use[:, :, 1:] - v_adv_use[:, :, :-1]
        v_tgt_use = v_tgt_use[:, :, 1:] - v_tgt_use[:, :, :-1]
        if mask_use is not None and mask_use.shape[2] >= 2:
            mask_use = (mask_use[:, :, 1:] + mask_use[:, :, :-1]) / 2

    # Expose (v_adv_use, v_tgt_use) to the caller before any filtering.
    # Used by the probe to collect data for --loss_freq_plot.
    if velocity_callback is not None:
        velocity_callback(v_adv_use.detach(), v_tgt_use.detach())

    # Compute loss: spatially LP-filtered, spatially masked, or plain MSE
    if velocity_lp_cutoff > 0:
        loss = _lowpass_velocity_mse(v_adv_use, v_tgt_use,
                                      lp_cutoff=velocity_lp_cutoff,
                                      velocity_mask=mask_use)
    elif mask_use is not None:
        m = mask_use.to(device=v_adv_use.device, dtype=v_adv_use.dtype)
        loss = ((v_adv_use - v_tgt_use) ** 2 * m).mean()
    else:
        loss = F.mse_loss(v_adv_use, v_tgt_use)

    return loss, lambda: None


# ---------------------------------------------------------------------------
# Attack driver
# ---------------------------------------------------------------------------
def attack_single_video_v3(
    inference, model, video_path, prompt,
    H, W, frames_to_extract, required_pixel_frames,
    T_tok, layer_start, layer_end, compute_dtype, args,
    vae_device="cuda", dit_device="cuda", save_time=None,
    preloaded_raw_padded=None,
    benign_prompt_variants=None,
    target_prompt_variants=None,
    unsafe_xy=None,
    safe_xy=None,
    out_dir=None,
):
    """v3 attack driver — mirrors v2 step-for-step but additionally
    constructs a spatial weighting mask from `unsafe_xy` / `safe_xy`
    after clean_latent is computed, and threads it into both the PGD
    loss closure and the held-out probe.

    Mask construction is gated on `args.mask_velocity`.  Without that
    flag, behaviour is identical to v2 (modulo RNG ordering, which is
    unchanged).
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

    for p in model.net.parameters():
        p.requires_grad_(False)
    for p in model.tokenizer.model.model.parameters():
        p.requires_grad_(False)
    model.tokenizer.enable_grad = True

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

    clean_latent = None
    sim_masks = None
    velocity_mask = None
    if target_condition_template is not None:
        with torch.no_grad():
            clean_latent = model.tokenizer.encode(
                raw_padded.to(compute_dtype)
            ).contiguous().float()
            if vae_device != dit_device:
                clean_latent = clean_latent.to(dit_device)
        print(f"Clean latent (frozen target ref) shape: {clean_latent.shape}")

        # ----- v3 spatial mask -----
        if getattr(args, "mask_velocity", False) and args.loss_type == "velocity":
            _B, _C, _T_lat, _H_lat, _W_lat = clean_latent.shape
            velocity_mask, mask_diag = build_unsafe_spatial_mask(
                _T_lat, _H_lat, _W_lat,
                unsafe_xy=unsafe_xy,
                safe_xy=safe_xy,
                unsafe_weight=args.mask_unsafe_weight,
                safe_weight=args.mask_safe_weight,
                radius_lat=args.mask_radius_lat,
                world_to_pix_scale=args.mask_world_to_pix_scale,
                device=clean_latent.device,
                dtype=torch.float32,
            )
            print(
                f"[v3 mask] latent grid (T,H,W)=({_T_lat},{_H_lat},{_W_lat})  "
                f"unsafe_xy_world={unsafe_xy} → lat_cell={mask_diag['unsafe_lat']}  "
                f"safe_xy_world={safe_xy} → lat_cell={mask_diag['safe_lat']}  "
                f"weights (u,s)=({mask_diag['unsafe_weight']},{mask_diag['safe_weight']})  "
                f"radius={mask_diag['radius_lat']}  scale={args.mask_world_to_pix_scale}"
            )

        # cross-prompt floor calibration (unchanged from v2)
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
            if velocity_mask is not None:
                _m = velocity_mask.to(device=_v_b.device, dtype=_v_b.dtype)
                _cross_floor = ((_v_b - _v_t) ** 2 * _m).mean().item()
                _cross_floor_plain = F.mse_loss(_v_b, _v_t).item()
                print(f"[probe calibration] masked ||v(clean,benign)-v(clean,target)||² = "
                      f"{_cross_floor:.6f}  (plain MSE = {_cross_floor_plain:.6f}) at t=0.4")
            else:
                _cross_floor = F.mse_loss(_v_b, _v_t).item()
                print(f"[probe calibration] ||v(clean, benign) - v(clean, target)||² = "
                      f"{_cross_floor:.6f}  (at t=0.4, fresh noise)")
        print(f"[probe calibration] target_ref={getattr(args, 'target_ref', 'clean')}  "
              f"— if probe loss quickly exceeds the floor, image-noise dominates "
              f"(try --target_ref adv)")

        if args.mask and args.loss_type in ("cosine", "l2", "kl"):
            sim_masks = compute_sim_mask(
                model, clean_latent, condition, target_condition_template, T_tok,
                layer_start=layer_start,
                layer_end=layer_end,
                num_latent_conditional_frames=args.num_latent_conditional_frames,
                noise_seed=0,
            )
            analyze_sim_masks(sim_masks, spatial_grid=27, save_dir="sim_analysis")

    if args.loss_type == "velocity" and args.mask:
        print("[v3] --mask (cross-attn sim mask) has no effect with --loss_type velocity; "
              "use --mask_velocity for the spatial mask instead.")

    probe_every = int(getattr(args, "probe_every", 0))
    probe_fn = None
    if probe_every > 0 and target_condition_template is not None:
        probe_t_norm    = 0.4
        _probe_noise_holder = {"noise": None}
        _probe_call_idx = [0]
        _do_freq_plot   = getattr(args, "loss_freq_plot", False)
        _freq_n_bins    = int(getattr(args, "freq_plot_n_bins", 40))

        def probe_fn(x_pert_in):
            call_idx = _probe_call_idx[0]
            _probe_call_idx[0] += 1
            # call 0 fires at step 0 (before any update); call k fires at step k*probe_every
            est_step = call_idx * probe_every

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

            _vel_data = {}
            def _vel_cb(va, vt):
                _vel_data["v_adv"] = va.cpu()
                _vel_data["v_tgt"] = vt.cpu()

            with torch.no_grad():
                loss, cleanup = compute_attack_loss_v3(
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
                    velocity_mask=velocity_mask,
                    velocity_lp_cutoff=getattr(args, "velocity_lp_cutoff", 0.0),
                    velocity_temporal_diff=getattr(args, "velocity_temporal_diff", False),
                    velocity_callback=_vel_cb if _do_freq_plot else None,
                )
                val = float(loss.item())
            if cleanup is not None:
                cleanup()

            if _do_freq_plot and "v_adv" in _vel_data and out_dir is not None:
                _plot_freq_loss_profile(
                    _vel_data["v_adv"], _vel_data["v_tgt"],
                    step=est_step,
                    out_dir=out_dir,
                    label=video_path.stem,
                    n_bins=_freq_n_bins,
                )

            return val

        freq_note = "  freq_plot=ON" if _do_freq_plot else ""
        print(f"[probe] enabled — fixed t={probe_t_norm}, fresh noise on first call, "
              f"cadence every {probe_every} PGD steps{freq_note}")
    elif probe_every > 0:
        print("[probe] disabled (no target_condition to compare against)")

    _use_variants = getattr(args, "rand_prompts", False) and bool(benign_cond_variants)

    def loss_fn(x, y):
        c = random.choice(benign_cond_variants) if _use_variants else condition
        t = (random.choice(target_cond_variants)
             if (_use_variants and target_cond_variants)
             else target_condition_template)
        use_uncond = (
            args.loss_type in ("cosine", "l2", "kl")
            and args.denoise_steps > 1
        )
        return compute_attack_loss_v3(
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
            velocity_mask=velocity_mask,
            velocity_lp_cutoff=getattr(args, "velocity_lp_cutoff", 0.0),
            velocity_temporal_diff=getattr(args, "velocity_temporal_diff", False),
            t_min=getattr(args, "t_min", 0.25),
            t_max=getattr(args, "t_max", 0.75),
        )

    if getattr(args, "lf_attack", False):
        if args.skip_latent:
            raise ValueError("--lf_attack requires pixel-space optimisation")
        if args.freq_cutoff != 0.1 or args.freq_pixel_eps != 0.1:
            print("[v3] --freq_cutoff / --freq_pixel_eps are ignored by lab_pgd; "
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
            inner_steps=getattr(args, "inner_steps", 1),
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

    if out_dir is None:
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


# Re-export for convenience (so callers can `from attack_crossattn_v3 import pad_video`)
__all__ = [
    "attack_single_video_v3",
    "compute_attack_loss_v3",
    "build_unsafe_spatial_mask",
    "pad_video",
]
