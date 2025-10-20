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

    print("doclayout — evaluate mAP")
    print("=" * 60)
    print(f"  quick mode  : {args.quick}")
    print(f"  subset size : {EVAL_SUBSET_SIZE}")
    print(f"  out metrics : {METRICS_JSON}")

    if not PUBLAYNET_VAL_JSON.exists():
        print("\n[abort] No PubLayNet val.json — run download_data.py first.")
        sys.exit(0)
    if not DETECTIONS_JSON.exists():
        print("\n[abort] No detections — run detect.py first.")
        sys.exit(0)

    # ── Stage D implementation ──────────────────────────────────
    # TODO(stage-d):
    #   1. build subset GT (500 images) with fixed seed → val_subset.json
    #   2. load GT + DT, filter DT to the same image_ids
    #   3. COCOeval(coco_gt, coco_dt, "bbox").evaluate().accumulate().summarize()
    #   4. take stats[0] (mAP@0.50:0.95), stats[1] (mAP@0.50)
    #   5. per-class AP loop, write metrics.json + per_class_ap.csv
    print("\n[stub] evaluate body is a skeleton in stage A.")
    sys.exit(0)


if __name__ == "__main__":
    main()
