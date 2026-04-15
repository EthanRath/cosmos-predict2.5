"""
Build a run_attack_pipeline.py-compatible JSON from a dataset directory.

Expected directory layout:
    <dataset_dir>/
        videos/   demo_1.mp4, demo_2.mp4, ...
        metas/    demo_1.txt, demo_2.txt, ...

Each video is paired with the text file that shares its stem.  For every
pair a target is sampled uniformly at random from the rest of the dataset:
  - crossattn: target_prompt  = prompt of the sampled pair
  - vae:       attack_target  = video  of the sampled pair

The `model`, `attack_params`, and `eval_params` sections are taken from a
base config JSON (--base_config), so you can reuse an existing
experiments_crossattn.json or experiments_vae.json as a template without
duplicating all the model / hyperparameter settings.

Usage:
    python cosmos-predict2.5/scripts/build_experiment_json.py \\
        --dataset_dir /path/to/dataset \\
        --attack_type crossattn \\
        --base_config cosmos-predict2.5/assets/attack/experiments_crossattn.json \\
        --output experiments_batch.json \\
        [--seed 42] \\
        [--video_ext mp4] \\
        [--meta_ext txt]
"""

import argparse
import json
import random
from pathlib import Path


def read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", required=True,
                        help="Directory containing 'videos/' and 'metas/' subdirectories")
    parser.add_argument("--attack_type", required=True, choices=["vae", "crossattn"],
                        help="Attack type for the generated JSON")
    parser.add_argument("--base_config", required=True,
                        help="JSON file whose model/attack_params/eval_params sections are reused")
    parser.add_argument("--output", required=True,
                        help="Path to write the generated experiment JSON")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for target sampling (default: 42)")
    parser.add_argument("--video_ext", default="mp4",
                        help="Video file extension without dot (default: mp4)")
    parser.add_argument("--meta_ext", default="txt",
                        help="Prompt file extension without dot (default: txt)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    videos_dir  = dataset_dir / "videos"
    metas_dir   = dataset_dir / "metas"

    if not videos_dir.is_dir():
        raise FileNotFoundError(f"videos/ directory not found: {videos_dir}")
    if not metas_dir.is_dir():
        raise FileNotFoundError(f"metas/ directory not found: {metas_dir}")

    # ------------------------------------------------------------------
    # Collect paired (video_path, prompt) samples
    # ------------------------------------------------------------------
    samples = []
    missing_metas = []

    for video_path in sorted(videos_dir.glob(f"*.{args.video_ext}")):
        meta_path = metas_dir / f"{video_path.stem}.{args.meta_ext}"
        if not meta_path.exists():
            missing_metas.append(video_path.stem)
            continue
        samples.append({
            "video_path": str(video_path),
            "prompt":     read_prompt(meta_path),
        })

    if missing_metas:
        print(f"Warning: skipped {len(missing_metas)} video(s) with no matching meta file:")
        for stem in missing_metas:
            print(f"  {stem}")

    n = len(samples)
    if n < 2:
        raise ValueError(f"Need at least 2 paired samples to build attack experiments, found {n}")

    print(f"Found {n} paired samples in {dataset_dir}")

    # ------------------------------------------------------------------
    # Sample targets (each sample gets a different random target)
    # ------------------------------------------------------------------
    rng = random.Random(args.seed)
    indices = list(range(n))

    experiments = []
    for i, src in enumerate(samples):
        other_indices = [j for j in indices if j != i]
        tgt_idx = rng.choice(other_indices)
        tgt = samples[tgt_idx]

        if args.attack_type == "crossattn":
            experiments.append({
                "video_path":    src["video_path"],
                "true_prompt":   src["prompt"],
                "target_prompt": tgt["prompt"],
            })
        else:  # vae
            experiments.append({
                "video_path":    src["video_path"],
                "attack_target": tgt["video_path"],
                "eval_prompt":   src["prompt"],
            })

    # ------------------------------------------------------------------
    # Load base config and replace experiments
    # ------------------------------------------------------------------
    with open(args.base_config) as f:
        config = json.load(f)

    config["attack_type"]  = args.attack_type
    config["experiments"]  = experiments

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote {len(experiments)} experiments to: {output_path}")
    print(f"  attack_type : {args.attack_type}")
    print(f"  seed        : {args.seed}")
    print(f"  base_config : {args.base_config}")


if __name__ == "__main__":
    main()


"""
# Cross-attention batch from a dataset
python cosmos-predict2.5/scripts/build_experiment_json.py \\
    --dataset_dir /path/to/libero_dataset \\
    --attack_type crossattn \\
    --base_config cosmos-predict2.5/assets/attack/experiments_crossattn.json \\
    --output attack/batch_crossattn.json \\
    --seed 42

# VAE batch from the same dataset
python cosmos-predict2.5/scripts/build_experiment_json.py \\
    --dataset_dir /path/to/libero_dataset \\
    --attack_type vae \\
    --base_config cosmos-predict2.5/assets/attack/experiments_vae.json \\
    --output attack/batch_vae.json \\
    --seed 42

# Then run the pipeline on the generated JSON:
python cosmos-predict2.5/scripts/run_attack_pipeline.py \\
    --experiments attack/batch_crossattn.json \\
    --nproc_per_node 2
"""
