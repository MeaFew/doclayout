"""Windows-compatible one-shot pipeline runner for doclayout.

Replaces `make all` on systems without GNU Make. Runnable pipeline:
samples → detect → visualize.

NOTE: download_data.py and evaluate.py are currently blocked by missing
PubLayNet val data and pycocotools; they raise NotImplementedError if run
directly. See README.md / CONTRIBUTING.md for manual setup instructions.

Each step runs as an explicit argv list (no shell=True) to avoid shell
injection and Windows quoting pitfalls. Halts on the first nonzero exit.

Usage:
    python run_all.py              # full runnable pipeline
    python run_all.py --quick      # subset (500 images) smoke run
    python run_all.py --from detect  # resume from a step
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Force UTF-8 for child processes on Windows.
os.environ.setdefault("PYTHONUTF8", "1")

# Pipeline stages, in order. Keep in sync with the Makefile `all` target.
# download_data.py / evaluate.py are excluded because they require manual
# PubLayNet data + pycocotools and currently raise NotImplementedError.
STEPS = [
    ("Generate samples", ["python", "-m", "doclayout.make_samples"]),
    ("Detect layouts", ["python", "-m", "doclayout.detect", "--batch", "samples"]),
    ("Visualize", ["python", "-m", "doclayout.visualize"]),
]


def run(cmd: list[str], cwd: Path, *, quick: bool = False) -> bool:
    """Run a command, print a banner, return False on nonzero exit."""
    cmd = list(cmd)
    if quick:
        cmd.append("--quick")
    print(f"\n{'-' * 60}\n> {cmd}\n{'-' * 60}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full doclayout pipeline.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the 500-image subset for a fast end-to-end smoke run.",
    )
    parser.add_argument(
        "--from",
        dest="start_at",
        default=None,
        choices=[s[0] for s in STEPS],
        help="Resume from a named stage (skip earlier stages).",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    print("doclayout - document layout analysis")
    print("=" * 60)

    started = args.start_at is None
    for name, cmd in STEPS:
        if not started:
            if name == args.start_at:
                started = True
            else:
                print(f"(skip) {name}")
                continue
        if not run(cmd, cwd=here, quick=args.quick):
            print(f"\nPipeline stopped at step: {name}")
            sys.exit(1)

    print("\nOK: Pipeline complete. Run `make dashboard` to launch the UI.")


if __name__ == "__main__":
    main()
