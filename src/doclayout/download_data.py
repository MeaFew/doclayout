"""Download PubLayNet val data + sample documents for doclayout.

Stage B of the pipeline. Fetches:
  - PubLayNet val (COCO format val.json + val images) — for mAP evaluation
  - A few sample document images into samples/ — for dashboard demos

PubLayNet's original IBM hosting is unreliable; we try sources in order:
  1. HuggingFace datasets (jordanparker6/publaynet) — parquet, via `datasets`
  2. Manual fallback note (OpenDataLab / Kaggle)

In --quick mode only the val.json + a 500-image subset is kept.

Usage:
    python -m doclayout.download_data
    python -m doclayout.download_data --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from doclayout.config import (  # noqa: E402
    PUBLAYNET_VAL_IMAGES_DIR,
    PUBLAYNET_VAL_JSON,
    RAW_DATA_DIR,
    ensure_dirs,
)
from doclayout.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download PubLayNet data for doclayout.")
    p.add_argument("--quick", action="store_true", help="Subset (500 images) only.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    logger.info("doclayout - data download")
    logger.info("=" * 60)
    logger.info(f"  quick mode  : {args.quick}")
    logger.info(f"  raw dir     : {RAW_DATA_DIR}")
    logger.info(f"  val json    : {PUBLAYNET_VAL_JSON}")
    logger.info(f"  val images  : {PUBLAYNET_VAL_IMAGES_DIR}")

    raise NotImplementedError(
        "PubLayNet val data download is not yet implemented: "
        "IBM DAX hosting is unreliable and HuggingFace mirrors are parquet-only. "
        "Obtain val.json + images manually and place them in "
        f"{PUBLAYNET_VAL_JSON} + {PUBLAYNET_VAL_IMAGES_DIR}/."
    )


if __name__ == "__main__":
    setup_logging()
    main()
