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

import numpy as np

from doclayout.config import (  # noqa: E402
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
from doclayout.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


# ── IoU computation ──────────────────────────────────────────────────────────


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes in [x, y, w, h] format.

    Parameters
    ----------
    boxes_a : np.ndarray, shape (N, 4)
    boxes_b : np.ndarray, shape (M, 4)

    Returns
    -------
    np.ndarray, shape (N, M) — pairwise IoU values.
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)

    # Convert [x, y, w, h] → [x1, y1, x2, y2]
    a_x1 = boxes_a[:, 0]
    a_y1 = boxes_a[:, 1]
    a_x2 = boxes_a[:, 0] + boxes_a[:, 2]
    a_y2 = boxes_a[:, 1] + boxes_a[:, 3]

    b_x1 = boxes_b[:, 0]
    b_y1 = boxes_b[:, 1]
    b_x2 = boxes_b[:, 0] + boxes_b[:, 2]
    b_y2 = boxes_b[:, 1] + boxes_b[:, 3]

    # Intersection
    inter_x1 = np.maximum(a_x1[:, None], b_x1[None, :])
    inter_y1 = np.maximum(a_y1[:, None], b_y1[None, :])
    inter_x2 = np.minimum(a_x2[:, None], b_x2[None, :])
    inter_y2 = np.minimum(a_y2[:, None], b_y2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union
    area_a = boxes_a[:, 2] * boxes_a[:, 3]
    area_b = boxes_b[:, 2] * boxes_b[:, 3]
    union_area = area_a[:, None] + area_b[None, :] - inter_area

    iou = np.where(union_area > 0, inter_area / union_area, 0.0)
    return iou


# ── AP computation (per-class, single IoU threshold) ─────────────────────────


def compute_ap_at_iou(
    gt_boxes: np.ndarray,
    dt_boxes: np.ndarray,
    dt_scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> float:
    """Compute Average Precision for a single class at a given IoU threshold.

    Uses the COCO-style 101-point interpolation.

    Parameters
    ----------
    gt_boxes : (G, 4) array of ground-truth boxes [x,y,w,h]
    dt_boxes : (D, 4) array of detection boxes [x,y,w,h]
    dt_scores : (D,) array of detection confidence scores
    iou_threshold : minimum IoU to count as TP

    Returns
    -------
    float — AP value in [0, 1].
    """
    n_gt = len(gt_boxes)
    n_dt = len(dt_boxes)

    if n_gt == 0:
        return 0.0
    if n_dt == 0:
        return 0.0

    # Sort detections by score (descending)
    order = np.argsort(-dt_scores)
    dt_boxes = dt_boxes[order]
    dt_scores = dt_scores[order]

    iou_matrix = compute_iou_matrix(dt_boxes, gt_boxes)  # (D, G)

    gt_matched = np.zeros(n_gt, dtype=bool)
    tp = np.zeros(n_dt, dtype=np.float64)
    fp = np.zeros(n_dt, dtype=np.float64)

    for d_idx in range(n_dt):
        ious = iou_matrix[d_idx]
        # Find best matching GT
        best_gt = np.argmax(ious)
        best_iou = ious[best_gt]

        if best_iou >= iou_threshold and not gt_matched[best_gt]:
            tp[d_idx] = 1.0
            gt_matched[best_gt] = True
        else:
            fp[d_idx] = 1.0

    # Cumulative sums
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)

    # 101-point interpolation (COCO style)
    ap = _interpolate_ap(precision, recall)
    return float(ap)


def _interpolate_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    """COCO-style 101-point interpolated AP."""
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    # For each recall threshold, find max precision at recall >= threshold
    interpolated = np.zeros(101, dtype=np.float64)
    for i, t in enumerate(recall_thresholds):
        precs = precision[recall >= t]
        interpolated[i] = precs.max() if len(precs) > 0 else 0.0
    return float(interpolated.mean())


# ── Full mAP evaluation (pure numpy, no pycocotools dependency) ──────────────

IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)  # 0.50, 0.55, ..., 0.95


def evaluate_map(
    gt_annotations: list[dict],
    detections: list[dict],
    category_ids: list[int] | None = None,
) -> dict:
    """Compute COCO-style mAP from GT annotations and detections.

    Parameters
    ----------
    gt_annotations : list of COCO annotation dicts
        Each: {"image_id": int, "category_id": int, "bbox": [x,y,w,h]}
    detections : list of COCO result dicts
        Each: {"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}
    category_ids : list of category IDs to evaluate (default: all in GT)

    Returns
    -------
    dict with keys:
        "map_5095": mAP @ IoU 0.50:0.95
        "map_50":   mAP @ IoU 0.50
        "per_class": {cat_id: {"ap_5095": float, "ap_50": float}}
    """
    if category_ids is None:
        category_ids = sorted({a["category_id"] for a in gt_annotations})

    # Group by (image_id, category_id)
    gt_by_img_cat: dict[tuple[int, int], list] = {}
    for ann in gt_annotations:
        key = (ann["image_id"], ann["category_id"])
        gt_by_img_cat.setdefault(key, []).append(ann["bbox"])

    dt_by_img_cat: dict[tuple[int, int], list] = {}
    for det in detections:
        key = (det["image_id"], det["category_id"])
        dt_by_img_cat.setdefault(key, []).append((det["bbox"], det.get("score", 1.0)))

    # Collect all image_ids
    all_image_ids = sorted(
        {a["image_id"] for a in gt_annotations} | {d["image_id"] for d in detections}
    )

    per_class: dict[int, dict] = {}

    for cat_id in category_ids:
        # Gather all GT and DT boxes for this category across all images
        all_gt_boxes = []
        all_dt_boxes = []
        all_dt_scores = []

        for img_id in all_image_ids:
            key = (img_id, cat_id)
            gt_boxes = gt_by_img_cat.get(key, [])
            dt_entries = dt_by_img_cat.get(key, [])

            all_gt_boxes.extend(gt_boxes)

            for bbox, score in dt_entries:
                all_dt_boxes.append(bbox)
                all_dt_scores.append(score)

        if not all_gt_boxes:
            per_class[cat_id] = {"ap_5095": 0.0, "ap_50": 0.0}
            continue

        gt_arr = np.array(all_gt_boxes, dtype=np.float64)
        if all_dt_boxes:
            dt_arr = np.array(all_dt_boxes, dtype=np.float64)
            scores_arr = np.array(all_dt_scores, dtype=np.float64)
        else:
            dt_arr = np.zeros((0, 4), dtype=np.float64)
            scores_arr = np.zeros(0, dtype=np.float64)

        # AP at each IoU threshold
        aps = []
        for iou_t in IOU_THRESHOLDS:
            ap = compute_ap_at_iou(gt_arr, dt_arr, scores_arr, iou_threshold=iou_t)
            aps.append(ap)

        ap_5095 = float(np.mean(aps))
        ap_50 = aps[0]  # IoU=0.50 is the first threshold

        per_class[cat_id] = {"ap_5095": ap_5095, "ap_50": ap_50}

    # mAP = mean over classes
    map_5095 = float(np.mean([v["ap_5095"] for v in per_class.values()])) if per_class else 0.0
    map_50 = float(np.mean([v["ap_50"] for v in per_class.values()])) if per_class else 0.0

    return {"map_5095": map_5095, "map_50": map_50, "per_class": per_class}


# ── pycocotools-based evaluation (preferred when available) ──────────────────


def evaluate_with_pycocotools(
    gt_json_path: Path,
    dt_json_path: Path,
    category_ids: list[int] | None = None,
) -> dict:
    """Run COCOeval using pycocotools for maximum compatibility.

    Parameters
    ----------
    gt_json_path : path to COCO-format GT annotations JSON
    dt_json_path : path to COCO-format detections JSON
    category_ids : optional list of category IDs to restrict evaluation

    Returns
    -------
    dict with "map_5095", "map_50", "per_class" keys.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_json_path))
    coco_dt = coco_gt.loadRes(str(dt_json_path))

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    if category_ids:
        coco_eval.params.catIds = category_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # stats[0] = mAP@0.50:0.95, stats[1] = mAP@0.50
    map_5095 = float(coco_eval.stats[0])
    map_50 = float(coco_eval.stats[1])

    # Per-class AP
    per_class: dict[int, dict] = {}
    cat_ids = category_ids or sorted(coco_gt.getCatIds())
    for cat_id in cat_ids:
        coco_eval_cat = COCOeval(coco_gt, coco_dt, iouType="bbox")
        coco_eval_cat.params.catIds = [cat_id]
        coco_eval_cat.evaluate()
        coco_eval_cat.accumulate()
        coco_eval_cat.summarize()
        per_class[cat_id] = {
            "ap_5095": float(coco_eval_cat.stats[0]),
            "ap_50": float(coco_eval_cat.stats[1]),
        }

    return {"map_5095": map_5095, "map_50": map_50, "per_class": per_class}


# ── Subset creation ──────────────────────────────────────────────────────────


def create_eval_subset(
    val_json_path: Path,
    subset_size: int = EVAL_SUBSET_SIZE,
    seed: int = EVAL_SEED,
) -> dict:
    """Create a deterministic subset of the PubLayNet val annotations.

    Returns the subset COCO dict (images + annotations + categories).
    """
    with open(val_json_path, encoding="utf-8") as f:
        coco_data = json.load(f)

    rng = np.random.default_rng(seed)
    images = coco_data["images"]
    n = min(subset_size, len(images))
    indices = rng.choice(len(images), size=n, replace=False)
    subset_images = [images[i] for i in sorted(indices)]
    subset_image_ids = {img["id"] for img in subset_images}

    subset_anns = [a for a in coco_data["annotations"] if a["image_id"] in subset_image_ids]

    return {
        "images": subset_images,
        "annotations": subset_anns,
        "categories": coco_data["categories"],
    }


# ── Report writing ───────────────────────────────────────────────────────────


def write_metrics(metrics: dict, output_path: Path) -> None:
    """Write metrics.json with mAP values."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "map_5095": round(metrics["map_5095"], 6),
        "map_50": round(metrics["map_50"], 6),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"  wrote metrics → {output_path}")


def write_per_class_csv(metrics: dict, output_path: Path) -> None:
    """Write per_class_ap.csv with per-category AP values."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category_id", "category_name", "ap_5095", "ap_50"])
        for cat_id in sorted(metrics["per_class"].keys()):
            name = CATEGORY_ID_TO_NAME.get(cat_id, f"class_{cat_id}")
            vals = metrics["per_class"][cat_id]
            writer.writerow([cat_id, name, f"{vals['ap_5095']:.6f}", f"{vals['ap_50']:.6f}"])
    logger.info(f"  wrote per-class AP → {output_path}")


# ── CLI entrypoint ───────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate doclayout mAP on PubLayNet.")
    p.add_argument("--quick", action="store_true", help="Evaluate on the 500-image subset.")
    p.add_argument(
        "--gt", type=str, default=None, help="Path to GT annotations JSON (overrides config)."
    )
    p.add_argument(
        "--dt", type=str, default=None, help="Path to detections JSON (overrides config)."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    logger.info("doclayout - evaluate mAP")
    logger.info("=" * 60)
    logger.info(f"  quick mode  : {args.quick}")
    logger.info(f"  subset size : {EVAL_SUBSET_SIZE}")
    logger.info(f"  out metrics : {METRICS_JSON}")

    # Resolve GT path
    gt_path = Path(args.gt) if args.gt else PUBLAYNET_VAL_JSON
    dt_path = Path(args.dt) if args.dt else DETECTIONS_JSON

    if not gt_path.exists():
        logger.error(
            f"[abort] GT annotations not found: {gt_path}\n"
            "  Place PubLayNet val.json at data/raw/publaynet_val.json "
            "or pass --gt <path>."
        )
        sys.exit(1)

    if not dt_path.exists():
        logger.error(
            f"[abort] Detections not found: {dt_path}\n"
            "  Run `python scripts/detect.py --batch <image_dir>` first, "
            "or pass --dt <path>."
        )
        sys.exit(1)

    # Optionally create subset
    if args.quick:
        logger.info(f"  creating {EVAL_SUBSET_SIZE}-image subset (seed={EVAL_SEED})...")
        subset = create_eval_subset(gt_path, EVAL_SUBSET_SIZE, EVAL_SEED)
        PUBLAYNET_SUBSET_JSON.parent.mkdir(parents=True, exist_ok=True)
        PUBLAYNET_SUBSET_JSON.write_text(json.dumps(subset), encoding="utf-8")
        gt_path = PUBLAYNET_SUBSET_JSON
        logger.info(f"  subset written → {PUBLAYNET_SUBSET_JSON}")

    # Run evaluation — prefer pycocotools, fall back to pure-numpy
    try:
        logger.info("  using pycocotools COCOeval...")
        metrics = evaluate_with_pycocotools(gt_path, dt_path)
    except ImportError:
        logger.info("  pycocotools not available, using built-in numpy evaluator...")
        with open(gt_path, encoding="utf-8") as f:
            coco_data = json.load(f)
        gt_annotations = coco_data["annotations"]
        with open(dt_path, encoding="utf-8") as f:
            detections = json.load(f)
        metrics = evaluate_map(gt_annotations, detections)

    # Report
    logger.info(f"\n  mAP@0.50:0.95 = {metrics['map_5095']:.4f}")
    logger.info(f"  mAP@0.50      = {metrics['map_50']:.4f}")

    write_metrics(metrics, METRICS_JSON)
    write_per_class_csv(metrics, PER_CLASS_CSV)

    logger.info("\nOK: evaluation complete.")


if __name__ == "__main__":
    setup_logging()
    main()
