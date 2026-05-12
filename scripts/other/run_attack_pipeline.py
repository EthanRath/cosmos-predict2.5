"""
Batch adversarial attack pipeline for Video2World.

Reads a JSON experiment file, runs the chosen attack (vae or crossattn) for
each experiment, then evaluates the resulting x_adv through the full diffusion
model.  All experiments share a common model / attack / eval configuration;
per-experiment fields specify the video paths and prompts.

JSON schema
-----------
{
  "attack_type": "vae" | "crossattn",      // required

  "model": {                               // required
    // shared by crossattn attack + eval:
    "experiment_name": "predict2_...",
    "ckpt_path": "/path/to/model.pt",
    "config_file": "cosmos_predict2/_src/predict2/configs/video2world/config.py",
    // shared by vae attack:
    "vae_pth": "/path/to/tokenizer.pth",
    // shared by all:
    "resolution": "432,432",
    "num_latent_video_frames": 6,
    "num_latent_conditional_frames": 2,    // crossattn + eval only
    "offload_diffusion_model": true,
    "offload_text_encoder": true,
    "offload_tokenizer": true
  },

  "attack_params": {                       // optional, all have defaults
    "steps": 20,
    "alpha": 0.00392,
    "eps": 0.0628,
    "timestep": 0.5,                       // crossattn only
    "num_attack_layers": 4                 // crossattn only; null = all layers
  },

  "eval_params": {                         // optional, all have defaults
    "guidance": 7.0,
    "num_steps": 35,
    "seed": 1
  },

  "experiments": [
    {
      // both attack types:
      "video_path":    "cosmos-predict2.5/assets/attack/k_1.mp4",
      // vae attack only:
      "attack_target": "cosmos-predict2.5/assets/attack/k_2.mp4",
      // crossattn attack only:
      "true_prompt":   "...",   // prompt used at inference / eval time
      "target_prompt": "...",   // prompt whose cross-attn activations we target
      // eval_prompt overrides true_prompt for the diffusion eval step (optional):
      "eval_prompt":   "..."
    }
  ]
}

Usage:
    python cosmos-predict2.5/scripts/run_attack_pipeline.py \\
        --experiments cosmos-predict2.5/assets/attack/experiments_vae.json

    # skip the attack step (e.g. re-eval existing outputs):
    python ... --skip_attack

    # multi-GPU eval:
    python ... --nproc_per_node 2

    # dry run (print commands only):
    python ... --dry_run
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent          # cosmos-predict2.5/
WM_ROOT    = REPO_ROOT.parent           # WM_Poison/

_ATTACK_SCRIPTS = {
    "vae":       SCRIPT_DIR / "attack_vae_encoder.py",
    "crossattn": SCRIPT_DIR / "attack_cross_attention.py",
}
EVAL_SCRIPT = SCRIPT_DIR / "eval_diffusion.py"


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def _bool_flags(cfg, keys):
    """Return --flag for every key that is True in cfg."""
    return [f"--{k}" for k in keys if cfg.get(k, False)]


def build_attack_cmd(attack_type, experiment, config):
    model  = config.get("model", {})
    params = config.get("attack_params", {})
    res    = model.get("resolution", "432,432")

    shared_pgd = [
        "--steps", str(params.get("steps", 20)),
        "--alpha", str(params.get("alpha", 1 / 255)),
        "--eps",   str(params.get("eps",   16 / 255)),
    ]

    if attack_type == "vae":
        # resolution is two ints for the VAE script
        h, w = res.split(",")
        return [
            "python", str(_ATTACK_SCRIPTS["vae"]),
            "--video_path",    experiment["video_path"],
            "--attack_target", experiment["attack_target"],
            "--vae_pth",       model.get("vae_pth", ""),
            "--resolution",    h, w,
            "--num_latent_video_frames", str(model.get("num_latent_video_frames", 6)),
        ] + shared_pgd

    elif attack_type == "crossattn":
        cmd = [
            "python", str(_ATTACK_SCRIPTS["crossattn"]),
            "--video_path",    experiment["video_path"],
            "--true_prompt",   experiment["true_prompt"],
            "--target_prompt", experiment["target_prompt"],
            "--experiment_name", model["experiment_name"],
            "--ckpt_path",       model["ckpt_path"],
            "--resolution",    res,
            "--num_latent_video_frames",       str(model.get("num_latent_video_frames", 6)),
            "--num_latent_conditional_frames", str(model.get("num_latent_conditional_frames", 2)),
            "--timestep",      str(params.get("timestep", 0.5)),
        ] + shared_pgd + _bool_flags(model, [
            "offload_diffusion_model", "offload_text_encoder", "offload_tokenizer",
        ])
        if params.get("num_attack_layers") is not None:
            cmd += ["--num_attack_layers", str(params["num_attack_layers"])]
        if model.get("config_file"):
            cmd += ["--config_file", model["config_file"]]
        return cmd

    raise ValueError(f"Unknown attack_type: {attack_type!r}")


def build_eval_cmd(adv_path, experiment, config, nproc_per_node=1):
    model  = config.get("model", {})
    params = config.get("eval_params", {})

    # eval_prompt > true_prompt (crossattn) > required for vae
    eval_prompt = (experiment.get("eval_prompt")
                   or experiment.get("true_prompt")
                   or experiment["eval_prompt"])   # raises KeyError with a clear message

    launcher = (["torchrun", f"--nproc_per_node={nproc_per_node}"]
                if nproc_per_node > 1 else ["python"])

    cmd = launcher + [
        str(EVAL_SCRIPT),
        "--adv_path",        str(adv_path),
        "--experiment_name", model["experiment_name"],
        "--ckpt_path",       model["ckpt_path"],
        "--prompt",          eval_prompt,
        "--resolution",      model.get("resolution", "432,432"),
        "--num_latent_conditional_frames", str(model.get("num_latent_conditional_frames", 2)),
        "--guidance",    str(params.get("guidance",  7.0)),
        "--num_steps",   str(params.get("num_steps", 35)),
        "--seed",        str(params.get("seed",      1)),
        "--context_parallel_size", str(nproc_per_node),
    ] + _bool_flags(model, [
        "offload_diffusion_model", "offload_text_encoder", "offload_tokenizer",
    ])
    if model.get("config_file"):
        cmd += ["--config_file", model["config_file"]]
    return cmd


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def parse_adv_path(stdout):
    """
    Extract the x_adv.pt path printed by both attack scripts:
        Saved adversarial video to : /some/path/x_adv.pt / x_adv.mp4
    """
    m = re.search(r"Saved adversarial video to\s*:\s*(\S+x_adv\.pt)", stdout)
    return Path(m.group(1)) if m else None


def run_cmd(cmd, cwd, label, dry_run=False):
    """Run cmd, stream stdout in real time, return (success, full_stdout)."""
    print(f"\n{'='*60}")
    print(f"[{label}]")
    if dry_run:
        print("  " + " \\\n    ".join(str(c) for c in cmd))
        print(f"{'='*60}", flush=True)
        return True, ""

    print("  " + " ".join(str(c) for c in cmd[:6]) + (" ..." if len(cmd) > 6 else ""))
    print(f"{'='*60}", flush=True)

    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    stdout = "".join(lines)
    ok = proc.returncode == 0
    if not ok:
        print(f"\n[ERROR] {label} exited with code {proc.returncode}")
    return ok, stdout


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments",    required=True,
                        help="Path to JSON experiments file")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="Number of GPUs for eval_diffusion via torchrun (default: 1)")
    parser.add_argument("--skip_attack",    action="store_true",
                        help="Skip attack; each experiment must include 'adv_path'")
    parser.add_argument("--skip_eval",      action="store_true",
                        help="Skip diffusion evaluation")
    parser.add_argument("--dry_run",        action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    with open(args.experiments) as f:
        config = json.load(f)

    attack_type = config["attack_type"]
    experiments = config["experiments"]
    print(f"Pipeline: {len(experiments)} experiment(s), attack_type={attack_type}")

    results = []

    for i, exp in enumerate(experiments):
        tag = f"exp {i+1}/{len(experiments)}"
        print(f"\n{'#'*60}")
        print(f"# {tag}  |  {exp['video_path']}")
        print(f"{'#'*60}")

        adv_path = None

        # ----------------------------------------------------------------
        # Attack step
        # ----------------------------------------------------------------
        if args.skip_attack:
            adv_path = Path(exp["adv_path"])
            print(f"Skipping attack; adv_path = {adv_path}")
        else:
            cmd = build_attack_cmd(attack_type, exp, config)
            ok, stdout = run_cmd(cmd, WM_ROOT, f"attack  [{tag}]", dry_run=args.dry_run)

            if not args.dry_run:
                if not ok:
                    results.append({"index": i + 1, "status": "attack_failed",
                                    "video_path": exp["video_path"]})
                    continue
                adv_path = parse_adv_path(stdout)
                if adv_path is None:
                    print("[ERROR] Could not parse x_adv.pt path from attack output")
                    results.append({"index": i + 1, "status": "parse_failed",
                                    "video_path": exp["video_path"]})
                    continue
                print(f"x_adv.pt -> {adv_path}")

        # ----------------------------------------------------------------
        # Eval step
        # ----------------------------------------------------------------
        if args.skip_eval:
            results.append({"index": i + 1, "status": "attack_ok",
                             "adv_path": str(adv_path), "video_path": exp["video_path"]})
            continue

        if adv_path is None and args.dry_run:
            # dummy path so eval command can be printed
            adv_path = Path("attack/outputs/<timestamp>/x_adv.pt")

        cmd = build_eval_cmd(adv_path, exp, config, nproc_per_node=args.nproc_per_node)
        ok, _ = run_cmd(cmd, WM_ROOT, f"eval    [{tag}]", dry_run=args.dry_run)

        results.append({
            "index":      i + 1,
            "status":     "ok" if (ok or args.dry_run) else "eval_failed",
            "adv_path":   str(adv_path),
            "video_path": exp["video_path"],
        })

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    for r in results:
        suffix = f"  ->  {r['adv_path']}" if r.get("adv_path") else ""
        print(f"  [{r['index']:>2}] {r['status']:<20} {r['video_path']}{suffix}")

    if not args.dry_run:
        ts = int(time.time())
        summary_path = WM_ROOT / "attack" / "outputs" / f"pipeline_{ts}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump({"config": config, "results": results}, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()


"""
# VAE attack batch
python cosmos-predict2.5/scripts/run_attack_pipeline.py \\
    --experiments cosmos-predict2.5/assets/attack/experiments_vae.json

# Cross-attention attack batch, 2-GPU eval
python cosmos-predict2.5/scripts/run_attack_pipeline.py \\
    --experiments cosmos-predict2.5/assets/attack/experiments_crossattn.json \\
    --nproc_per_node 2

# Dry run to preview commands
python cosmos-predict2.5/scripts/run_attack_pipeline.py \\
    --experiments cosmos-predict2.5/assets/attack/experiments_vae.json \\
    --dry_run
"""
