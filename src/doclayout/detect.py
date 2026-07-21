"""Detect document layouts + table structure with PP-StructureV3.

The core inference step. Runs PaddleOCR's PP-StructureV3 over document images
and returns structured layout regions (text/title/table/figure/list + table HTML).

VERIFIED SCHEMA (paddleocr 3.7.0, probed on a real run):
  page = list(pipeline.predict(img))[0]        # LayoutParsingResultV2
  page.json                                     # dict
  page.json["res"]["parsing_res_list"]          # list of region dicts
  each region: {
      "block_label":  str,        # "text"/"title"/"table"/"figure"/"list"/...
      "block_bbox":   [x1,y1,x2,y2],   # pixels, top-left + bottom-right
      "block_content": str,        # OCR'd text or table HTML
      "block_id": int,
  }

KEY POINTS:
  - PP-StructureV3 bbox is [x1,y1,x2,y2]; COCO wants [x,y,w,h] → convert.
  - PP emits a richer label set than PubLayNet's 5 → map via config.
  - oneDNN MUST be disabled on paddle 3.3.x (PIR bug) — see config.ENABLE_MKLDNN.
  - CPU memory env vars set at import time (issue #17955).

Usage:
    python -m doclayout.detect --image path/to/doc.png    # single image → print regions
    python -m doclayout.detect --batch dir/               # batch → detections.json
    python -m doclayout.detect --batch dir/ --quick       # smoke: first image only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image  # noqa: E402

import doclayout.config as config  # noqa: E402
from doclayout.config import (  # noqa: E402
    CATEGORY_NAME_TO_ID,
    DEFAULT_DEVICE,
    DETECTIONS_JSON,
    ENABLE_MKLDNN,
    PP_TYPE_TO_PUBLAYNET,
    apply_paddle_env,
    ensure_dirs,
)
from doclayout.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

# CPU memory guard + oneDNN disable — MUST precede any paddle import.
# Single source of truth: config.apply_paddle_env() reads config.ENABLE_MKLDNN.
apply_paddle_env()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PP-StructureV3 layout detection.")
    p.add_argument("--image", type=str, help="Single image path → print regions.")
    p.add_argument("--batch", type=str, help="Directory of images → detections.json (COCO format).")
    p.add_argument(
        "--quick", action="store_true", help="Batch smoke run: process only the first image."
    )
    p.add_argument("--device", default=DEFAULT_DEVICE, choices=["cpu", "gpu"])
    return p.parse_args()


def _set_env() -> None:
    """Ensure paddle env vars are set (idempotent, safe before paddle import).

    Thin wrapper around the canonical config.apply_paddle_env() so callers that
    invoke _set_env() explicitly still go through one source of truth.
    """
    apply_paddle_env()


def load_pipeline(device: str = DEFAULT_DEVICE):
    """Load the PP-StructureV3 pipeline.

    oneDNN is disabled to work around paddlepaddle 3.3.x's PIR bug on Windows.
    """
    _set_env()
    from paddleocr import PPStructureV3

    return PPStructureV3(
        device=device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=ENABLE_MKLDNN,
    )


def extract_regions(page) -> list[dict]:
    """Pull the layout regions from a PP-StructureV3 page result.

    Returns a normalized list of {label, bbox:[x1,y1,x2,y2], content, table_html}.
    Robust to the verified schema (res.parsing_res_list).
    """
    data = page.json
    res = data.get("res", data) if isinstance(data, dict) else {}
    regions = res.get("parsing_res_list", []) if isinstance(res, dict) else []
    out = []
    for r in regions:
        label = r.get("block_label") or r.get("layout_type") or "unknown"
        bbox = r.get("block_bbox") or r.get("layout_bbox")
        content = r.get("block_content") or r.get("text") or ""
        table_html = ""
        if label == "table":
            table_res = r.get("table_res") or {}
            table_html = table_res.get("pred_html", "") if isinstance(table_res, dict) else ""
        if bbox and len(bbox) == 4:
            out.append(
                {
                    "label": label,
                    "bbox": [int(v) for v in bbox],
                    "content": content,
                    "table_html": table_html,
                }
            )
    return out


def compute_detection_score(
    bbox_xyxy: list[int], image_size: tuple[int, int], rank: int = 0, total: int = 1
) -> float:
    """Compute a heuristic confidence score for a detection.

    PP-StructureV3 does not expose per-block confidence scores, so we derive a
    proxy score from two signals:
      1. Area ratio — larger boxes relative to the image are more likely to be
         correct detections (score component in [0.3, 0.85]).
      2. Rank decay — earlier detections in the pipeline output tend to be more
         prominent; a mild linear decay penalizes later boxes (factor in [0.85, 1.0]).

    The final score is clamped to [0.1, 0.99] to keep precision-recall curves
    meaningful for mAP computation.

    Parameters
    ----------
    bbox_xyxy : [x1, y1, x2, y2] in pixels (already sorted/clamped).
    image_size : (width, height) of the source image.
    rank : 0-based position of this detection in the output list.
    total : total number of detections for this image.

    Returns
    -------
    float in [0.1, 0.99].
    """
    img_w, img_h = image_size
    img_area = max(img_w * img_h, 1)

    x1, y1, x2, y2 = bbox_xyxy
    box_area = max((x2 - x1) * (y2 - y1), 0)
    area_ratio = box_area / img_area  # in [0, 1]

    # Area component: sqrt to compress range, then scale to [0.3, 0.85]
    area_score = 0.3 + 0.55 * min(area_ratio**0.5, 1.0)

    # Rank decay: first detection gets 1.0, last gets 0.85
    if total > 1:
        rank_factor = 1.0 - 0.15 * (rank / (total - 1))
    else:
        rank_factor = 1.0

    score = area_score * rank_factor
    return float(max(0.1, min(0.99, score)))


def regions_to_coco(
    regions: list[dict],
    image_id: int,
    image_size: tuple[int, int],
    scores: list[float] | None = None,
) -> list[dict]:
    """Convert normalized regions to COCO results format for mAP evaluation.

    Maps PP labels to PubLayNet category_id (drops unmapped types), converts
    bbox from [x1,y1,x2,y2] to COCO's [x,y,w,h], and clamps coordinates to the
    image bounds. Degenerate boxes (zero width/height) are dropped.

    Parameters
    ----------
    scores : optional pre-computed confidence scores (one per input region).
        If None, a heuristic area+rank based score is computed automatically.
    """
    img_w, img_h = image_size
    out = []
    # First pass: collect valid entries to know total count for rank scoring
    valid_entries: list[tuple[dict, list[int], int]] = []  # (region, clamped_bbox, orig_idx)
    for idx, r in enumerate(regions):
        pl_name = PP_TYPE_TO_PUBLAYNET.get(r["label"])
        if pl_name is None:
            continue  # drop non-PubLayNet types (formula/header/footer/...)
        x1, y1, x2, y2 = r["bbox"]
        # Sort corners (tolerates swapped x1/x2 or y1/y2), then clamp to image bounds.
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue  # drop degenerate boxes
        valid_entries.append((r, [x1, y1, x2, y2], idx))

    total = len(valid_entries)
    for rank, (r, clamped_bbox, orig_idx) in enumerate(valid_entries):
        pl_name = PP_TYPE_TO_PUBLAYNET[r["label"]]
        x1, y1, x2, y2 = clamped_bbox
        w = x2 - x1
        h = y2 - y1

        # Use provided score or compute heuristic
        if scores is not None and orig_idx < len(scores):
            score = float(max(0.01, min(1.0, scores[orig_idx])))
        else:
            score = compute_detection_score(clamped_bbox, image_size, rank=rank, total=total)

        out.append(
            {
                "image_id": image_id,
                "category_id": CATEGORY_NAME_TO_ID[pl_name],
                "bbox": [x1, y1, w, h],  # → [x,y,w,h]
                "score": round(score, 6),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if not args.image and not args.batch:
        logger.info("Specify --image <path> or --batch <dir>. See --help.")
        sys.exit(1)

    logger.info("doclayout - detect")
    logger.info("=" * 60)
    pipeline = load_pipeline(args.device)
    logger.info(f"  device   : {args.device}  (mkldnn={ENABLE_MKLDNN})")

    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            logger.error(f"[abort] image not found: {img_path}")
            sys.exit(1)
        logger.info(f"  image    : {img_path}")
        pages = list(pipeline.predict(str(img_path)))
        regions = extract_regions(pages[0]) if pages else []
        logger.info(f"\n  detected {len(regions)} regions:")
        for r in regions:
            extra = " [table: HTML available]" if r["table_html"] else ""
            logger.info(f"    {r['label']:10s} bbox={r['bbox']}{extra}")
        logger.info("\nOK: single-image detection done.")
        return

    # ── Batch mode → COCO detections.json ──────────────────────
    img_dir = Path(args.batch)
    images = sorted([*img_dir.glob("*.png"), *img_dir.glob("*.jpg"), *img_dir.glob("*.jpeg")])
    if not images:
        logger.error(f"[abort] no images in {img_dir}")
        sys.exit(1)
    if args.quick:
        images = images[:1]
        logger.info("  quick mode: processing only the first image")
    logger.info(f"  batch    : {len(images)} images in {img_dir}")

    import json

    all_dets = []
    for i, img_path in enumerate(images):
        try:
            image_id = int(img_path.stem)
        except ValueError:
            image_id = i + 1
        with Image.open(img_path) as img:
            image_size = img.size
        pages = list(pipeline.predict(str(img_path)))
        regions = extract_regions(pages[0]) if pages else []
        all_dets.extend(regions_to_coco(regions, image_id, image_size))
        logger.info(f"    [{i + 1}/{len(images)}] {img_path.name}: {len(regions)} regions")

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DETECTIONS_JSON.write_text(json.dumps(all_dets), encoding="utf-8")
    logger.info(f"\n  wrote {len(all_dets)} detections → {DETECTIONS_JSON.name}")
    logger.info("OK: batch detection done. Next: `python -m doclayout.evaluate`")


if __name__ == "__main__":
    setup_logging()
    main()
