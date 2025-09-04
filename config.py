"""Centralized configuration for doclayout — document layout analysis.

All paths, category mappings, and evaluation parameters live here so scripts
never hardcode them. Import from config in any script:

    from config import RAW_IMAGE_DIR, CATEGORY_MAP, RANDOM_STATE
"""

from pathlib import Path

# ── Base directories (resolved relative to this file) ────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
IMAGES_DIR = BASE_DIR / "images"
SAMPLES_DIR = BASE_DIR / "samples"

# ── PubLayNet data ───────────────────────────────────────────────
# val.json is COCO format; images live in a sibling directory.
PUBLAYNET_VAL_JSON = RAW_DATA_DIR / "publaynet_val.json"
PUBLAYNET_VAL_IMAGES_DIR = RAW_DATA_DIR / "publaynet_val_images"
# Eval subset (kept small for fast CPU runs; full val = 11,245 images).
PUBLAYNET_SUBSET_JSON = PROCESSED_DATA_DIR / "publaynet_val_subset.json"

# ── Detection outputs (COCO results format) ─────────────────────
DETECTIONS_JSON = PROCESSED_DATA_DIR / "detections.json"
DETECTIONS_SUBSET_JSON = PROCESSED_DATA_DIR / "detections_subset.json"

# ── Reports ──────────────────────────────────────────────────────
METRICS_JSON = REPORTS_DIR / "metrics.json"
PER_CLASS_CSV = REPORTS_DIR / "per_class_ap.csv"

# ── PubLayNet categories (verified: arXiv:1908.07836) ───────────
# category_id 1-5 = Text/Title/List/Table/Figure. bbox format [x,y,w,h].
PUBLAYNET_CATEGORIES = [
    {"id": 1, "name": "text"},
    {"id": 2, "name": "title"},
    {"id": 3, "name": "list"},
    {"id": 4, "name": "table"},
    {"id": 5, "name": "figure"},
]
CATEGORY_NAME_TO_ID = {c["name"]: c["id"] for c in PUBLAYNET_CATEGORIES}
CATEGORY_ID_TO_NAME = {c["id"]: c["name"] for c in PUBLAYNET_CATEGORIES}

# PP-StructureV3 layout_type → PubLayNet category name.
# PP-StructureV3 emits a RICHER label set than PubLayNet's 5 classes. Labels
# observed in real runs (paddleocr 3.7): text, doc_title, paragraph_title,
# figure_title, table, figure, chart, list, formula, header, footer, reference,
# caption, ... We map the PubLayNet-overlapping ones; non-matching types
# (chart, formula, header, footer, *_title variants that aren't "title") are
# dropped so mAP only scores the 5 PubLayNet classes.
PP_TYPE_TO_PUBLAYNET = {
    # text
    "text": "text",
    # title — PP splits titles into doc_title / paragraph_title variants
    "title": "title",
    "doc_title": "title",
    "paragraph_title": "title",
    # list
    "list": "list",
    # table
    "table": "table",
    # figure
    "figure": "figure",
}

# ── Evaluation ───────────────────────────────────────────────────
EVAL_SUBSET_SIZE = 500  # images sampled for the quick mAP run
EVAL_SEED = 42

# ── Visualization colors (one per PubLayNet class) ───────────────
CATEGORY_COLORS = {
    "text": (31, 119, 180),    # blue
    "title": (255, 127, 14),   # orange
    "list": (44, 160, 44),     # green
    "table": (214, 39, 40),    # red
    "figure": (148, 103, 189), # purple
}

# ── Device ───────────────────────────────────────────────────────
# PP-StructureV3 runs on CPU by default; GPU is optional via --device flag.
DEFAULT_DEVICE = "cpu"

# ── Modeling constants ───────────────────────────────────────────
RANDOM_STATE = 42

# ── CPU memory guard (PaddleOCR 3.x issue #17955) ────────────────
# PaddlePaddle 3.x aggressively pre-allocates CPU memory (~43GB reported).
# These env vars MUST be set before importing paddle. Scripts set them at the
# top of detect.py / dashboard before importing paddleocr.
PADDLE_CPU_MEMORY_FRACTION = "0.3"
PADDLE_CPU_THREADS = "6"

# ── oneDNN disabled (paddlepaddle 3.3.x PIR bug on Windows CPU) ──
# paddlepaddle 3.3.1's new PIR executor + oneDNN hits
# `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
# [pir::ArrayAttribute<pir::DoubleAttribute>]` during layout detection
# inference on Windows CPU. Disabling oneDNN (enable_mkldnn=False) works
# around it at a modest speed cost. Revisit once paddle fixes the PIR path.
ENABLE_MKLDNN = False


def ensure_dirs() -> None:
    """Create the project's output directories. Call once at startup."""
    for d in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        IMAGES_DIR,
        SAMPLES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
