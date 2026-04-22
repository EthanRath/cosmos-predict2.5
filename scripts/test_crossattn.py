"""
Cross-attention injection test for Cosmos Video2World.

Tests the hypothesis that if one can make
    cross_attention(x, true_prompt) == cross_attention(x, adversarial_prompt)
then the generated video will follow the adversarial prompt semantics.

To test this directly, this script runs a full diffusion generation pass
conditioned on --prompt, but at every denoising step injects the cross-attention
outputs that *would* have been produced by --target_prompt into the actual
forward pass.  If cross-attention drives prompt semantics, the resulting video
should look as if it were generated from --target_prompt, even though the model
received --prompt as its conditioning signal.

Procedure
---------
At each Euler denoising step:
  1. Run a no-grad forward with target_condition_live at the current xt.
     Install capture hooks that record the output tensor from each block's
     cross_attn.compute_attention call.
  2. Run the real forward with condition_live at the same xt.
     Install inject hooks that *replace* each block's cross_attn.compute_attention
     output with the tensor captured in step 1, bypassing the true-prompt K/V.
  3. Use the velocity from the injected forward to step the Euler solver.

Two videos are saved for comparison:
  baseline_<ts>.mp4  — normal generation from --prompt (no injection)
  injected_<ts>.mp4  — generation from --prompt with target cross-attn injected

Usage
-----
    python cosmos-predict2.5/scripts/test_crossattn.py \\
        --video_path cosmos-predict2.5/assets/attack/k_1.mp4 \\
        --prompt "Use the franka robot arm to pick up the black bowl" \\
        --target_prompt "Use the franka robot arm to push the red block to the right" \\
        --resolution 432,432 \\
        --num_latent_video_frames 9 \\
        --num_latent_conditional_frames 2 \\
        --num_steps 35 \\
        --num_inject_layers 14
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torchvision
from PIL import Image

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
from attack.shared_config import (
    ckpt_path, experiment_name, config_file
)


# ---------------------------------------------------------------------------
# Video padding  (same as attack_crossattn.py)
# ---------------------------------------------------------------------------

def pad_video(video, frames_to_extract, required_pixel_frames):
    context = video[:, :, -frames_to_extract:, :, :]
    padding = required_pixel_frames - frames_to_extract
    if padding > 0:
        context = torch.cat(
            [context, context[:, :, -1:, :, :].repeat(1, 1, padding, 1, 1)], dim=2
        )
    return context


def load_and_preprocess_image(image_path, resolution, device="cpu"):
    """
    Load a single JPG/PNG image and return a normalized (1, C, 1, H, W)
    float32 tensor in [-1, 1].  The temporal dimension is 1; pad_video
    will replicate the frame to fill required_pixel_frames.
    """
    H, W = resolution
    img = Image.open(image_path).convert("RGB").resize((W, H), Image.BILINEAR)
    img_t = torchvision.transforms.functional.to_tensor(img)   # (C, H, W) float [0,1]
    img_t = (img_t * 2.0 - 1.0)                                # → [-1, 1]
    return img_t.unsqueeze(0).unsqueeze(2).to(device)           # (1, C, 1, H, W)


# ---------------------------------------------------------------------------
# Capture hooks — record cross_attn output tensors during the target pass
# ---------------------------------------------------------------------------

def install_capture_hooks(net, captured, cutoff=None):
    """
    For each block up to `cutoff`, wrap cross_attn.compute_attention so that
    its return value is stored in captured[block_idx] and then returned normally.

    captured : dict  block_idx -> attention output tensor
    Returns  : restore callable
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
        orig_fn = attn.compute_attention

        def _make_capture(fn, block_idx):
            def _wrapper(*args, **kw):
                result = fn(*args, **kw)
                captured[block_idx] = result.detach()
                return result
            return _wrapper

        attn.__dict__["compute_attention"] = _make_capture(orig_fn, idx)
        _patches[idx] = (attn, orig_fn)

    def _restore():
        for _idx, (attn_mod, _orig) in _patches.items():
            attn_mod.__dict__.pop("compute_attention", None)

    return _restore


# ---------------------------------------------------------------------------
# Inject hooks — replace cross_attn output with the captured target tensors
# ---------------------------------------------------------------------------

def install_inject_hooks(net, captured, cutoff=None):
    """
    For each block up to `cutoff`, wrap cross_attn.compute_attention so that
    it returns the tensor stored in captured[block_idx] instead of computing
    attention with the true-prompt K/V.

    captured : dict  block_idx -> attention output tensor (from target pass)
    Returns  : restore callable
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
        orig_fn = attn.compute_attention

        def _make_inject(fn, block_idx):
            def _wrapper(*args, **kw):
                # Ignore the true-prompt K/V and return the target output instead.
                # Cast to match the input dtype (bf16 in the DiT forward).
                return captured[block_idx].to(dtype=args[0].dtype)
            return _wrapper

        attn.__dict__["compute_attention"] = _make_inject(orig_fn, idx)
        _patches[idx] = (attn, orig_fn)

    def _restore():
        for _idx, (attn_mod, _orig) in _patches.items():
            attn_mod.__dict__.pop("compute_attention", None)

    return _restore


# ---------------------------------------------------------------------------
# Single injected denoising step
# ---------------------------------------------------------------------------

def injected_denoise_step(model, xt, timesteps, noise, condition_live, target_cond_live,
                           uncondition_live, guidance_scale, cutoff):
    """
    One denoising step with cross-attention injection.

    1. target pass (no grad): capture cross-attn outputs per block
    2. uncond pass  (no grad): velocity for CFG denominator
    3. cond pass    (no grad): velocity with injected target cross-attn

    Returns the CFG velocity prediction.
    """
    # -- 1. Capture target cross-attn outputs --
    captured = {}
    restore_capture = install_capture_hooks(model.net, captured, cutoff)
    with torch.no_grad():
        model.denoise(noise=noise, xt_B_C_T_H_W=xt,
                      timesteps_B_T=timesteps, condition=target_cond_live)
    restore_capture()

    # -- 2. Uncond pass for CFG --
    with torch.no_grad():
        uncond_v = model.denoise(noise=noise, xt_B_C_T_H_W=xt,
                                 timesteps_B_T=timesteps, condition=uncondition_live)

    # -- 3. Cond pass with injected cross-attn --
    restore_inject = install_inject_hooks(model.net, captured, cutoff)
    with torch.no_grad():
        cond_v = model.denoise(noise=noise, xt_B_C_T_H_W=xt,
                               timesteps_B_T=timesteps, condition=condition_live)
    restore_inject()

    return cond_v + guidance_scale * (cond_v - uncond_v)


# ---------------------------------------------------------------------------
# Baseline denoising step (no injection)
# ---------------------------------------------------------------------------

def baseline_denoise_step(model, xt, timesteps, noise, condition_live, uncondition_live,
                           guidance_scale):
    with torch.no_grad():
        cond_v   = model.denoise(noise=noise, xt_B_C_T_H_W=xt,
                                 timesteps_B_T=timesteps, condition=condition_live)
        uncond_v = model.denoise(noise=noise, xt_B_C_T_H_W=xt,
                                 timesteps_B_T=timesteps, condition=uncondition_live)
    return cond_v + guidance_scale * (cond_v - uncond_v)


# ---------------------------------------------------------------------------
# Full generation loop
# ---------------------------------------------------------------------------

def run_generation(model, latent, condition_live, uncondition_live,
                   target_cond_live, guidance_scale, num_steps, noise_seed,
                   cutoff, inject=True):
    """
    Run a full denoising trajectory using the model's official scheduler.

    inject=True  : inject target cross-attn at each step
    inject=False : standard generation (baseline)

    Returns the final denoised latent.
    """
    B, C, T_lat, H_lat, W_lat = latent.shape

    # Use the model's scheduler to get properly-scaled timesteps in [0, 1000]
    model.sample_scheduler.set_timesteps(num_steps, device=latent.device, shift=5.0)
    timesteps_schedule = model.sample_scheduler.timesteps

    seed_g = torch.Generator(device=latent.device).manual_seed(noise_seed)
    noise = torch.randn(latent.shape, generator=seed_g,
                        dtype=torch.float32, device=latent.device)
    x_t = noise.clone()

    for step_i, t in enumerate(timesteps_schedule):
        xt = x_t.to(**model.tensor_kwargs)
        # Scheduler provides scalar t; expand to (B, 1) as the model expects
        timesteps = t.reshape(1, 1).expand(B, 1).to(device=latent.device, dtype=latent.dtype)

        tag = "inject" if inject else "baseline"
        print(f"  [{tag}] step {step_i+1}/{num_steps}  t={t.item():.1f}", end="\r")

        if inject:
            v_pred = injected_denoise_step(
                model, xt, timesteps, noise.to(**model.tensor_kwargs),
                condition_live, target_cond_live,
                uncondition_live, guidance_scale, cutoff,
            )
        else:
            v_pred = baseline_denoise_step(
                model, xt, timesteps, noise.to(**model.tensor_kwargs),
                condition_live, uncondition_live, guidance_scale,
            )

        x_t = model.sample_scheduler.step(
            v_pred.float().unsqueeze(0), t, x_t.unsqueeze(0),
            return_dict=False, generator=seed_g,
        )[0].squeeze(0)

    print()
    return x_t.to(**model.tensor_kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video_path",
                        help="Path to the input conditioning video (.mp4)")
    input_group.add_argument("--image_path",
                        help="Path to a single conditioning image (.jpg/.png). "
                             "Forces num_latent_conditional_frames=1.")
    parser.add_argument("--prompt",         required=True,
                        help="True text prompt (model conditioning)")
    parser.add_argument("--target_prompt",  required=True,
                        help="Adversarial target prompt whose cross-attention is injected")
    parser.add_argument("--resolution",     default="432,432")
    parser.add_argument("--num_latent_video_frames",       type=int, default=9)
    parser.add_argument("--num_latent_conditional_frames", type=int, default=2,
                        help="Ignored when --image_path is used (forced to 1).")
    parser.add_argument("--num_steps",      type=int,   default=35,
                        help="Euler denoising steps (default: 35)")
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--noise_seed",     type=int,   default=0)
    parser.add_argument("--num_inject_layers", type=int, default=None,
                        help="Number of DiT blocks whose cross-attn is injected. "
                             "None = all blocks.")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--ckpt_path",       type=str, default=None)
    parser.add_argument("--negative_prompt", default=None)
    parser.add_argument("--config_file",
                        default="cosmos_predict2/_src/predict2/configs/video2world/config.py")
    parser.add_argument("--load_diffusion_model", action="store_true")
    parser.add_argument("--load_text_encoder",    action="store_true")
    parser.add_argument("--load_tokenizer",       action="store_true")
    parser.add_argument("--context_parallel_size", type=int, default=1)
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip the baseline generation (save time).")
    args = parser.parse_args()

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

    is_image_input        = args.image_path is not None
    # Image mode: only 1 conditioning latent frame (the single image).
    num_cond_frames       = 1 if is_image_input else args.num_latent_conditional_frames
    T_lat_attack          = min(args.num_latent_video_frames, state_t)
    required_pixel_frames = (T_lat_attack - 1) * 4 + 1
    frames_to_extract     = 1 if is_image_input else (num_cond_frames - 1) * 4 + 1
    T_tok                 = T_lat_attack
    cutoff                = args.num_inject_layers
    compute_dtype         = torch.bfloat16

    n_blocks = len(model.net.blocks)
    input_label = "image" if is_image_input else "video"
    print(f"Model         : {n_blocks} DiT blocks, state_t={state_t}")
    print(f"Input mode    : {input_label}  (num_cond_frames={num_cond_frames})")
    print(f"T_lat         : {T_lat_attack}  (pixel frames: {required_pixel_frames})")
    print(f"Inject layers : {cutoff or n_blocks}/{n_blocks}")
    print(f"Steps         : {args.num_steps}")
    print(f"Guidance      : {args.guidance_scale}")

    # ------------------------------------------------------------------
    # 2. Load input (image or video)
    # ------------------------------------------------------------------
    if inference.offload_text_encoder:
        if model.text_encoder is not None:
            if hasattr(model.text_encoder, "model") and model.text_encoder.model is not None:
                model.text_encoder.model = model.text_encoder.model.to(device).eval()
        if _t5_mod.cosmos_encoder is not None:
            _t5_mod.cosmos_encoder.text_encoder = \
                _t5_mod.cosmos_encoder.text_encoder.to(device).eval()

    if is_image_input:
        print(f"\nLoading image: {args.image_path}")
        # Returns (1, C, 1, H, W) float32 in [-1, 1]; pad_video repeats the
        # single frame to fill required_pixel_frames.
        raw_cond   = load_and_preprocess_image(args.image_path, [H, W], device=device)
        raw_padded = pad_video(raw_cond, frames_to_extract=1, required_pixel_frames=required_pixel_frames)
    else:
        print(f"\nLoading video: {args.video_path}")
        video_uint8 = load_and_preprocess_video(
            str(args.video_path), [H, W], frames_to_extract
        )
        raw_cond   = normalize_video(video_uint8, device=device)
        raw_padded = pad_video(raw_cond, frames_to_extract, required_pixel_frames)
    print(f"  Conditioning frames : {raw_cond.shape}")
    print(f"  Padded input shape  : {raw_padded.shape}")

    # ------------------------------------------------------------------
    # 3. Build T5 embeddings for both prompts (text encoder on GPU)
    # ------------------------------------------------------------------
    video_bf16     = raw_padded.to(dtype=compute_dtype)
    negative_prompt = args.negative_prompt or _DEFAULT_NEGATIVE_PROMPT

    data_batch = inference._get_data_batch_input(
        video=video_bf16,
        prompt=args.prompt,
        num_conditional_frames=num_cond_frames,
        negative_prompt=negative_prompt,
        use_neg_prompt=True,
    )
    data_batch["video"]             = video_bf16
    data_batch[IS_PREPROCESSED_KEY] = True

    target_data_batch = inference._get_data_batch_input(
        video=video_bf16,
        prompt=args.target_prompt,
        num_conditional_frames=num_cond_frames,
        negative_prompt=negative_prompt,
        use_neg_prompt=True,
    )
    target_data_batch["video"]             = video_bf16
    target_data_batch[IS_PREPROCESSED_KEY] = True

    # ------------------------------------------------------------------
    # 4. Offload text encoder, load tokenizer / diffusion model
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
    # 5. Freeze model parameters
    # ------------------------------------------------------------------
    for p in model.net.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # 6. Build condition objects
    # ------------------------------------------------------------------
    print("Building conditions...")
    with torch.no_grad():
        _, _, condition = model.get_data_and_condition(data_batch)
        _, uncondition  = model.conditioner.get_condition_with_negative_prompt(data_batch)

    condition = condition.edit_for_inference(
        is_cfg_conditional=True,
        num_conditional_frames=num_cond_frames,
    )
    uncond_dict = uncondition.to_dict(skip_underscore=False)
    uncond_dict["gt_frames"] = condition.gt_frames
    uncondition = type(uncondition)(**uncond_dict)
    uncondition = uncondition.edit_for_inference(
        is_cfg_conditional=False,
        num_conditional_frames=num_cond_frames,
    )

    # Target condition: same as base but with target T5 embeddings.
    print(f"Building target condition for: \"{args.target_prompt}\"")
    tgt_dict = condition.to_dict(skip_underscore=False)
    tgt_dict["crossattn_emb"] = target_data_batch["t5_text_embeddings"]
    target_condition = type(condition)(**tgt_dict)

    # ------------------------------------------------------------------
    # 7. Encode video to latent
    # ------------------------------------------------------------------
    with torch.no_grad():
        latent = model.tokenizer.encode(raw_padded.to(compute_dtype)).contiguous().float()
    print(f"Latent shape: {latent.shape}")

    def _build_live(cond):
        return cond.set_video_condition(
            gt_frames=latent.to(compute_dtype),
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=num_cond_frames,
        )

    condition_live       = _build_live(condition)
    uncondition_live     = _build_live(uncondition)
    target_cond_live     = _build_live(target_condition)

    # ------------------------------------------------------------------
    # 8. Output directory
    # ------------------------------------------------------------------
    ts      = int(time.time())
    out_dir = WM_ROOT / "attack" / "outputs" / f"test_crossattn_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def to_uint8_frames(t):
        """(1, C, T, H, W) float [-1,1] -> (T, H, W, C) uint8"""
        t = t[0].cpu().float()
        t = ((t.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
        return t.permute(1, 2, 3, 0)

    def decode_and_save(latent_out, name):
        with torch.no_grad():
            video_out = model.decode(latent_out.to(compute_dtype))
        frames = to_uint8_frames(video_out)
        path   = out_dir / name
        torchvision.io.write_video(str(path), frames, fps=24)
        print(f"  Saved: {path}")
        return video_out

    # ------------------------------------------------------------------
    # 9. Baseline generation (--prompt, no injection)
    # ------------------------------------------------------------------
    if not args.skip_baseline:
        print(f"\n{'='*60}")
        print(f"Baseline generation (prompt: \"{args.prompt}\")")
        print(f"{'='*60}")
        latent_baseline = run_generation(
            model, latent, condition_live, uncondition_live,
            target_cond_live=None,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            noise_seed=args.noise_seed,
            cutoff=cutoff,
            inject=False,
        )
        decode_and_save(latent_baseline, f"baseline_{ts}.mp4")

    # ------------------------------------------------------------------
    # 10. Injected generation (--prompt video, --target_prompt cross-attn)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Injected generation")
    print(f"  Prompt        : \"{args.prompt}\"")
    print(f"  Target prompt : \"{args.target_prompt}\"")
    print(f"{'='*60}")
    latent_injected = run_generation(
        model, latent, condition_live, uncondition_live,
        target_cond_live=target_cond_live,
        guidance_scale=args.guidance_scale,
        num_steps=args.num_steps,
        noise_seed=args.noise_seed,
        cutoff=cutoff,
        inject=True,
    )
    decode_and_save(latent_injected, f"injected_{ts}.mp4")

    print(f"\nAll outputs saved to: {out_dir}")
    print("Compare baseline vs injected to evaluate whether cross-attention")
    print("injection successfully transfers the adversarial prompt semantics.")


if __name__ == "__main__":
    main()
