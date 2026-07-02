"""Render layout-detection visualizations (annotated PNGs).

Stage E helper. Runs PP-StructureV3 on the sample documents and draws colored
bounding boxes over them, one color per PubLayNet class (config.CATEGORY_COLORS).
Regions whose PP label doesn't map to PubLayNet (e.g. "chart", "formula") are
drawn in gray so the visualization stays faithful to what the model found.

Output: images/<sample>_layout.png — used by the README and as a dashboard fallback.

Usage: python scripts/visualize.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import detect  # noqa: E402  (reuse load_pipeline + extract_regions)
from PIL import Image, ImageDraw  # noqa: E402

from config import (  # noqa: E402
    CATEGORY_COLORS,
    IMAGES_DIR,
    PP_TYPE_TO_PUBLAYNET,
    SAMPLES_DIR,
    ensure_dirs,
)

# Color for regions PP found that don't map to PubLayNet (chart/formula/header/...).
UNMAPPED_COLOR = (128, 128, 128)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize doclayout detections on samples.")
    p.add_argument("--samples-dir", type=str, default=str(SAMPLES_DIR))
    return p.parse_args()


def _color_for(label: str) -> tuple[int, int, int]:
    """Map a PP label to its display color (PubLayNet class color, or gray)."""
    pl_name = PP_TYPE_TO_PUBLAYNET.get(label)
    if pl_name and pl_name in CATEGORY_COLORS:
        return CATEGORY_COLORS[pl_name]
    return UNMAPPED_COLOR


def annotate(image_path: Path, regions: list[dict], out_path: Path) -> None:
    """Draw labeled bboxes on the image and save."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for r in regions:
        x1, y1, x2, y2 = r["bbox"]
        color = _color_for(r["label"])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        # label tag in the top-left corner of the box
        tag = r["label"]
        draw.rectangle([x1, max(0, y1 - 18), x1 + 8 * len(tag) + 8, y1], fill=color)
        draw.text((x1 + 4, max(0, y1 - 16)), tag, fill="white")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    samples_dir = Path(args.samples_dir)
    images = sorted([*samples_dir.glob("*.png"), *samples_dir.glob("*.jpg")])
    # exclude the synthetic _test_doc if present
    images = [p for p in images if not p.name.startswith("_")]

    if not images:
        print(f"[abort] no sample images in {samples_dir}")
        print("        run `python scripts/make_samples.py` first.")
        sys.exit(1)

    print("doclayout - visualize")
    print("=" * 60)
    pipeline = detect.load_pipeline()

    for img_path in images:
        pages = list(pipeline.predict(str(img_path)))
        regions = detect.extract_regions(pages[0]) if pages else []
        out_path = IMAGES_DIR / f"{img_path.stem}_layout.png"
        annotate(img_path, regions, out_path)
        label_counts: dict[str, int] = {}
        for r in regions:
            label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
        summary = ", ".join(f"{k}:{v}" for k, v in sorted(label_counts.items()))
        print(f"  {img_path.name}: {len(regions)} regions ({summary}) → {out_path.name}")

    print(f"\nOK: visualizations in {IMAGES_DIR}")


if __name__ == "__main__":
    main()
