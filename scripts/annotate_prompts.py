"""
annotate_prompts.py

For attack output directories that predate automatic prompt JSON saving,
interactively create the missing _prompts.json files.

The script scans a directory for .pt and .mp4 files, derives a label for
each unique run, skips any label that already has a _prompts.json, then
prompts the user for the true (benign) and target prompts and writes:

  {label}_prompts.json  →  {"true_prompt": "...", "target_prompt": "..."}

# Usage
python cosmos-predict2.5/scripts/annotate_prompts.py <output_dir>
python cosmos-predict2.5/scripts/annotate_prompts.py  # prompts for directory
"""

import json
import re
import sys
from pathlib import Path

# Suffixes appended to the label before the file extension — strip these to
# recover the bare label.
_STRIP_SUFFIXES = re.compile(
    r"(_adv|_benign|_target|_clean|_generated)$", re.IGNORECASE
)


def derive_label(stem: str) -> str:
    """Strip known trailing tags from a file stem to get the run label."""
    return _STRIP_SUFFIXES.sub("", stem)


def collect_labels(directory: Path) -> list[tuple[str, str]]:
    """Return sorted (label, filename) pairs that lack a _prompts.json file."""
    label_to_file: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix == ".pt":
            label = derive_label(path.stem)
            label_to_file.setdefault(label, path.name)

    return [
        (label, fname)
        for label, fname in sorted(label_to_file.items())
        if not (directory / f"{label}_prompts.json").exists()
    ]


def ask(prompt_text: str) -> str:
    try:
        value = input(prompt_text).strip()
    except EOFError:
        raise SystemExit("\nAborted (EOF).")
    if not value:
        raise SystemExit("\nAborted (empty input).")
    return value


def main() -> None:
    if len(sys.argv) > 1:
        directory = Path(sys.argv[1])
    else:
        raw = input("Output directory: ").strip()
        if not raw:
            raise SystemExit("No directory provided.")
        directory = Path(raw)

    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    labels = collect_labels(directory)

    if not labels:
        print("No missing prompt files found — nothing to do.")
        return

    print(f"\nFound {len(labels)} .pt file(s) without _prompts.json in: {directory}\n")

    for i, (label, filename) in enumerate(labels, 1):
        print(f"[{i}/{len(labels)}] {filename}")
        true_prompt   = ask("  true prompt   (benign): ")
        target_prompt = ask("  target prompt         : ")

        out_path = directory / f"{label}_prompts.json"
        with open(out_path, "w") as f:
            json.dump({"true_prompt": true_prompt, "target_prompt": target_prompt}, f, indent=2)
        print(f"  Saved: {out_path}\n")

    print(f"Done. Wrote {len(labels)} prompt file(s).")


if __name__ == "__main__":
    main()
