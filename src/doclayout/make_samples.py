"""Generate sample document images for dashboard demos and visualization.

Creates realistic-looking document pages with REAL rendered text (using system
fonts) so PP-StructureV3 can detect text/title/table/figure regions properly —
synthetic gray-bar placeholders get misclassified since OCR sees no text.

Saved to samples/.

Usage: python -m doclayout.make_samples
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from doclayout.config import SAMPLES_DIR, ensure_dirs  # noqa: E402
from doclayout.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

W, H = 1000, 1300
MARGIN = 60

_FONT_CANDIDATES = {
    "regular": [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ],
    "bold": [
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ],
}


def _find_font(bold: bool) -> str:
    """Return the first existing font from the cross-platform candidate list."""
    kind = "bold" if bold else "regular"
    for path in _FONT_CANDIDATES[kind]:
        if Path(path).exists():
            return path
    raise FileNotFoundError(
        f"doclayout: no suitable {kind} font found. "
        f"Tried: {', '.join(_FONT_CANDIDATES[kind])}. "
        "Install DejaVu/Liberation/Helvetica/Arial fonts."
    )


_FONT_PATHS: dict[str, str] = {}


def _font_path(bold: bool) -> str:
    """Resolve (and cache) the font path lazily — importing this module must
    not probe the filesystem (font lookup happens on first use)."""
    kind = "bold" if bold else "regular"
    if kind not in _FONT_PATHS:
        _FONT_PATHS[kind] = _find_font(bold)
    return _FONT_PATHS[kind]


def __getattr__(name: str) -> str:
    # Lazily-resolved module-level constants, kept for backward compatibility
    # (existing callers import FONT_REGULAR / FONT_BOLD directly).
    if name == "FONT_REGULAR":
        return _font_path(bold=False)
    if name == "FONT_BOLD":
        return _font_path(bold=True)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_font_path(bold), size)


def _wrapped_lines(
    text: str, font: ImageFont.FreeTypeFont, max_w: int, draw: ImageDraw.ImageDraw
) -> list[str]:
    """Greedy word-wrap to fit max_w pixels."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textlength(test, font=font) <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _draw_text(
    draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill: str = "black"
):
    draw.text(xy, text, font=font, fill=fill)


def _paragraph(draw, x, y, w, text, font, fill="black", line_gap: int = 6) -> int:
    for line in _wrapped_lines(text, font, w, draw):
        _draw_text(draw, (x, y), line, font, fill=fill)
        y += font.size + line_gap
    return y


def sample_paper() -> Image.Image:
    """Academic-paper-like page: title + abstract + two columns + table."""
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    title_f = _font(26, bold=True)
    body_f = _font(14)
    small_f = _font(13)

    # Title
    d.text(
        (MARGIN, MARGIN), "Deep Learning for Document Layout Analysis", font=title_f, fill="black"
    )
    d.text(
        (MARGIN, MARGIN + 34),
        "A Comparative Study of Detection Models",
        font=_font(16),
        fill=(80, 80, 80),
    )

    # Abstract
    y = MARGIN + 70
    d.text((MARGIN, y), "Abstract", font=_font(15, bold=True), fill="black")
    y += 24
    abs_text = (
        "Document layout analysis is the task of decomposing a document image into "
        "structural regions such as text, titles, tables, and figures. In this paper "
        "we compare several detection architectures on the PubLayNet benchmark and "
        "demonstrate that transformer-based detectors achieve competitive accuracy."
    )
    y = _paragraph(d, MARGIN, y, W - 2 * MARGIN, abs_text, small_f, line_gap=4)

    # Two-column body
    y += 20
    col_w = (W - 2 * MARGIN - 30) // 2
    left_text = (
        "Introduction. The rapid growth of digital documents has created a strong "
        "demand for automatic document understanding. Layout analysis is the first "
        "step in many document processing pipelines, including information extraction "
        "and table recognition. Traditional approaches relied on hand-crafted "
        "features and heuristics, but modern methods formulate the problem as object "
        "detection."
    )
    right_text = (
        "Related Work. Object detection has been revolutionized by deep learning. "
        "The COCO benchmark established mAP as the standard evaluation metric. For "
        "documents specifically, PubLayNet provides over three hundred thousand "
        "annotated pages. Detection transformers and YOLO variants have both been "
        "applied to this task with promising results in recent years."
    )
    _paragraph(d, MARGIN, y, col_w, left_text, body_f)
    _paragraph(d, MARGIN + col_w + 30, y, col_w, right_text, body_f)

    # Table
    ty = y + 230
    d.text(
        (MARGIN, ty - 24),
        "Table 1. Detection results on PubLayNet val.",
        font=small_f,
        fill=(80, 80, 80),
    )
    d.rectangle([MARGIN, ty, W - MARGIN, ty + 150], outline=(40, 40, 40), width=2)
    d.rectangle([MARGIN, ty, W - MARGIN, ty + 28], fill=(220, 220, 220))
    cols = [MARGIN, MARGIN + 250, MARGIN + 450, MARGIN + 650]
    headers = ["Model", "mAP@0.5", "mAP@0.5:0.95", "FPS"]
    for cx, hdr in zip(cols, headers):
        d.text((cx + 8, ty + 6), hdr, font=_font(13, bold=True))
    for r in range(1, 5):
        d.line([(MARGIN, ty + r * 28), (W - MARGIN, ty + r * 28)], fill=(160, 160, 160))
    for cx in cols[1:]:
        d.line([(cx, ty), (cx, ty + 150)], fill=(160, 160, 160))
    rows = [
        ["DETR-R50", "0.921", "0.701", "12"],
        ["YOLOv8-L", "0.934", "0.732", "85"],
        ["LayoutLMv3", "0.951", "0.764", "8"],
        ["PP-Structure", "0.918", "0.690", "41"],
    ]
    for i, row in enumerate(rows):
        for cx, val in zip(cols, row):
            d.text((cx + 8, ty + 32 + i * 28), val, font=small_f)
    return img


def sample_report() -> Image.Image:
    """Business-report-like page: title + body + figure (bar chart) + table."""
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text(
        (MARGIN, MARGIN),
        "Quarterly Revenue Report 2025",
        font=_font(24, bold=True),
        fill=(180, 30, 40),
    )
    y = MARGIN + 44
    body = (
        "This report summarizes the quarterly performance across four product "
        "categories. Overall revenue grew by eighteen percent year over year, driven "
        "primarily by strong sales in the electronics and home goods segments. The "
        "chart below visualizes the revenue distribution, and the table provides "
        "detailed figures for each category including growth rates."
    )
    y = _paragraph(d, MARGIN, y, W - 2 * MARGIN, body, _font(14))

    # Figure: bar chart
    fy = y + 20
    d.text(
        (MARGIN, fy - 20),
        "Figure 1. Revenue by category (millions USD)",
        font=_font(13),
        fill=(80, 80, 80),
    )
    d.rectangle([MARGIN, fy, W - MARGIN, fy + 240], outline=(60, 60, 60), width=1)
    chart_x = MARGIN + 60
    bar_w = 90
    bars = [
        (160, "Electronics", (31, 119, 180)),
        (120, "Home", (255, 127, 14)),
        (180, "Apparel", (44, 160, 44)),
        (95, "Sports", (214, 39, 40)),
    ]
    for bar_h, label, color in bars:
        d.rectangle([chart_x, fy + 240 - bar_h - 20, chart_x + bar_w, fy + 220], fill=color)
        d.text((chart_x, fy + 224), label, font=_font(12), fill=(60, 60, 60))
        chart_x += bar_w + 40

    # Table
    ty = fy + 270
    d.rectangle([MARGIN, ty, W - MARGIN, ty + 140], outline=(40, 40, 40), width=2)
    d.rectangle([MARGIN, ty, W - MARGIN, ty + 26], fill=(220, 220, 220))
    for cx, hdr in zip(
        [MARGIN, MARGIN + 280, MARGIN + 520, MARGIN + 720],
        ["Category", "Revenue ($M)", "Growth", "Share"],
    ):
        d.text((cx + 8, ty + 5), hdr, font=_font(13, bold=True))
    for r in range(1, 5):
        d.line([(MARGIN, ty + r * 28), (W - MARGIN, ty + r * 28)], fill=(160, 160, 160))
    for cx in [MARGIN + 280, MARGIN + 520, MARGIN + 720]:
        d.line([(cx, ty), (cx, ty + 140)], fill=(160, 160, 160))
    rows = [
        ["Electronics", "160", "+22%", "31%"],
        ["Home Goods", "120", "+15%", "23%"],
        ["Apparel", "180", "+18%", "35%"],
        ["Sports", "95", "+8%", "18%"],
    ]
    for i, row in enumerate(rows):
        for cx, val in zip([MARGIN, MARGIN + 280, MARGIN + 520, MARGIN + 720], row):
            d.text((cx + 8, ty + 30 + i * 28), val, font=_font(13))
    return img


def sample_invoice() -> Image.Image:
    """Invoice-like page: header + key-value + line-item table."""
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((MARGIN, MARGIN), "INVOICE #2025-0418", font=_font(28, bold=True), fill=(30, 30, 30))
    d.line([(MARGIN, MARGIN + 44), (W - MARGIN, MARGIN + 44)], fill=(30, 30, 30), width=2)

    y = MARGIN + 60
    fields = [
        ("Bill To:", "Acme Corporation, 123 Market Street, San Francisco CA 94103"),
        ("Date:", "April 18, 2025"),
        ("Due Date:", "May 18, 2025"),
        ("Terms:", "Net 30 days"),
    ]
    lbl_f = _font(13, bold=True)
    val_f = _font(13)
    for label, value in fields:
        d.text((MARGIN, y), label, font=lbl_f, fill=(100, 100, 100))
        d.text((MARGIN + 90, y), value, font=val_f)
        y += 24

    # Line-item table
    ty = y + 30
    d.rectangle([MARGIN, ty, W - MARGIN, ty + 380], outline=(40, 40, 40), width=2)
    d.rectangle([MARGIN, ty, W - MARGIN, ty + 30], fill=(240, 240, 240))
    for cx, hdr in zip(
        [MARGIN, MARGIN + 300, MARGIN + 480, W - MARGIN - 150],
        ["Description", "Qty", "Unit Price", "Amount"],
    ):
        d.text((cx + 8, ty + 7), hdr, font=lbl_f)
    for r in range(1, 9):
        d.line([(MARGIN, ty + r * 38), (W - MARGIN, ty + r * 38)], fill=(180, 180, 180))
    for cx in [MARGIN + 300, MARGIN + 480, W - MARGIN - 150]:
        d.line([(cx, ty), (cx, ty + 380)], fill=(180, 180, 180))
    items = [
        ["Cloud computing services (monthly)", "1", "2,400.00", "2,400.00"],
        ["Premium support package", "12", "500.00", "6,000.00"],
        ["Additional storage 5TB", "1", "800.00", "800.00"],
        ["API calls (overage)", "1", "1,150.00", "1,150.00"],
        ["Training session onsite", "2", "1,500.00", "3,000.00"],
        ["Custom integration work", "8", "225.00", "1,800.00"],
        ["Backup and disaster recovery", "1", "650.00", "650.00"],
        ["Documentation and reporting", "1", "400.00", "400.00"],
    ]
    for i, row in enumerate(items):
        for cx, val in zip([MARGIN, MARGIN + 300, MARGIN + 480, W - MARGIN - 150], row):
            d.text((cx + 8, ty + 40 + i * 38), val, font=val_f)
    return img


def main() -> None:
    ensure_dirs()
    samples = {
        "sample_paper.png": sample_paper,
        "sample_report.png": sample_report,
        "sample_invoice.png": sample_invoice,
    }
    for name, fn in samples.items():
        img = fn()
        img.save(SAMPLES_DIR / name)
        logger.info(f"  wrote {name} ({img.size[0]}x{img.size[1]})")
    logger.info(f"\nOK: {len(samples)} sample documents in {SAMPLES_DIR}")


if __name__ == "__main__":
    setup_logging()
    main()
