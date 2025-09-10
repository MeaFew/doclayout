"""Download PubLayNet val data + sample documents for doclayout.

Stage B of the pipeline. Fetches:
  - PubLayNet val (COCO format val.json + val images) — for mAP evaluation
  - A few sample document images into samples/ — for dashboard demos

PubLayNet's original IBM hosting is unreliable; we try sources in order:
  1. HuggingFace datasets (jordanparker6/publaynet) — parquet, via `datasets`
  2. Manual fallback note (OpenDataLab / Kaggle)

In --quick mode only the val.json + a 500-image subset is kept.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    PUBLAYNET_VAL_IMAGES_DIR,
    PUBLAYNET_VAL_JSON,
    RAW_DATA_DIR,
    ensure_dirs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download PubLayNet data for doclayout.")
    p.add_argument("--quick", action="store_true", help="Subset (500 images) only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    print("doclayout — data download")
    print("=" * 60)
    print(f"  quick mode  : {args.quick}")
    print(f"  raw dir     : {RAW_DATA_DIR}")
    print(f"  val json    : {PUBLAYNET_VAL_JSON}")
    print(f"  val images  : {PUBLAYNET_VAL_IMAGES_DIR}")

    # ── Stage B implementation ──────────────────────────────────
    # TODO(stage-b): fetch PubLayNet val.json + images.
    #   1. try HF datasets (jordanparker6/publaynet) → reconstruct COCO json
    #   2. fallback: instruct user to download from OpenDataLab/Kaggle
    #   3. write val.json + extract val images
    #   4. copy/extract a few sample docs into samples/
    print("\n[stub] download_data is a skeleton in stage A.")
    print("      Run stage B to implement the real download.")
    sys.exit(0)


if __name__ == "__main__":
    main()
