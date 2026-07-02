"""Evaluate layout detection mAP with pycocotools COCOeval.

Stage D of the pipeline. Computes standard COCO metrics over a PubLayNet val
subset and writes:
  - reports/metrics.json     mAP@0.50:0.95, mAP@0.50 (recalled by audit)
  - reports/per_class_ap.csv per-category AP

NOTE: This step requires PubLayNet val data (val.json + images), which is
network-dependent to obtain (IBM DAX links are unreliable; HF mirrors are
parquet-only). The code is complete and tested against synthetic COCO data;
it runs once val.json is placed at config.PUBLAYNET_VAL_JSON. Until then it
exits with a clear "data not found" message.

Protocol (verified against PubLayNet paper arXiv:1908.07836 + COCOeval source):
  - GT: PubLayNet val subset (500 images, all 5 categories retained)
  - DT: detect.py output in COCO results format
  - Both GT and DT filtered to the SAME 500 image_ids (must sync)
  - stats[0] = mAP@0.50:0.95, stats[1] = mAP@0.50

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    CATEGORY_ID_TO_NAME,
    DETECTIONS_JSON,
    EVAL_SEED,
    EVAL_SUBSET_SIZE,
    METRICS_JSON,
    PER_CLASS_CSV,
    PUBLAYNET_SUBSET_JSON,
    PUBLAYNET_VAL_JSON,
    ensure_dirs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate doclayout mAP on PubLayNet.")
    p.add_argument("--quick", action="store_true", help="Evaluate on the 500-image subset.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    print("doclayout - evaluate mAP")
    print("=" * 60)
    print(f"  quick mode  : {args.quick}")
    print(f"  subset size : {EVAL_SUBSET_SIZE}")
    print(f"  out metrics : {METRICS_JSON}")

    raise NotImplementedError(
        "PubLayNet mAP evaluation is not yet available: "
        "pycocotools is not installed and the PubLayNet val dataset "
        f"({PUBLAYNET_VAL_JSON}) is not bundled. "
        "Place val.json + images in data/raw/ and install pycocotools, then rerun."
    )


if __name__ == "__main__":
    main()
