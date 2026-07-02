"""doclayout — interactive document layout analysis dashboard.

Upload a document image (or pick a sample) → PP-StructureV3 segments it into
text/title/table/figure regions, drawn as colored boxes. Table regions also
render their recognized HTML structure.

Tabs:
  - 📄 Analyze : upload/sample → annotated layout + table HTML
  - 📊 Metrics : PubLayNet mAP results (when available)

The pipeline is loaded once via st.cache_resource. oneDNN is disabled
(config.ENABLE_MKLDNN) to work around paddlepaddle 3.3.x's PIR bug.

Run:  streamlit run dashboard/app.py   (or `make dashboard`)
"""

from __future__ import annotations

import html
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

repo_root = Path(__file__).parents[1].resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "scripts") not in sys.path:
    sys.path.insert(0, str(repo_root / "scripts"))

import config  # noqa: E402
from config import METRICS_JSON, SAMPLES_DIR  # noqa: E402

# CPU memory guard + oneDNN disable — before any paddle import downstream.
os.environ.setdefault("FLAGS_fraction_of_cpu_memory_to_use", config.PADDLE_CPU_MEMORY_FRACTION)
os.environ.setdefault("FLAGS_cpu_threads", config.PADDLE_CPU_THREADS)
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import detect  # noqa: E402

st.set_page_config(page_title="doclayout", layout="wide", page_icon="📄")
st.title("📄 doclayout — document layout analysis")
st.caption(
    f"PP-StructureV3 · layout segmentation + table recognition · device `{config.DEFAULT_DEVICE}`"
)


@st.cache_resource(show_spinner="Loading PP-StructureV3 models (first run downloads weights)...")
def _load_pipeline():
    return detect.load_pipeline(config.DEFAULT_DEVICE)


def _annotate(img: Image.Image, regions: list[dict]) -> Image.Image:
    """Draw colored bboxes on a copy of the image."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for r in regions:
        x1, y1, x2, y2 = r["bbox"]
        pl = config.PP_TYPE_TO_PUBLAYNET.get(r["label"])
        color = config.CATEGORY_COLORS.get(pl or "", (128, 128, 128))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.rectangle([x1, max(0, y1 - 18), x1 + 8 * len(r["label"]) + 8, y1], fill=color)
        draw.text((x1 + 4, max(0, y1 - 16)), r["label"], fill="white")
    return img


def _render_table_html(table_html: str) -> None:
    """Render table HTML safely: native dataframe if parsable, else escaped code."""
    try:
        dfs = pd.read_html(table_html)
    except Exception:
        dfs = []
    if dfs:
        st.dataframe(dfs[0], use_container_width=True)
    else:
        st.code(html.escape(table_html), language="html")


def _list_samples() -> list[Path]:
    if not SAMPLES_DIR.exists():
        return []
    return sorted([*SAMPLES_DIR.glob("*.png"), *SAMPLES_DIR.glob("*.jpg")])


# ── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Legend (PubLayNet colors)")
    for name, color in config.CATEGORY_COLORS.items():
        st.color_picker(name, f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}", disabled=True)
    st.caption("gray = unmapped PP type (chart/formula/header/...)")

tab_analyze, tab_metrics = st.tabs(["📄 Analyze", "📊 Metrics"])

# ── Tab: Analyze ────────────────────────────────────────────────
with tab_analyze:
    st.subheader("Analyze a document")

    col_src1, col_src2 = st.columns(2)
    with col_src1:
        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    with col_src2:
        samples = _list_samples()
        sample_names = ["(none)"] + [p.name for p in samples]
        chosen = st.selectbox("...or pick a sample", sample_names)

    source_img: Image.Image | None = None
    if uploaded is not None:
        source_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    elif chosen != "(none)":
        source_img = Image.open(SAMPLES_DIR / chosen).convert("RGB")

    if source_img is not None:
        st.image(source_img, caption="input", use_container_width=True)
        if st.button("Run layout analysis", type="primary"):
            pipeline = _load_pipeline()
            with st.spinner("Running PP-StructureV3..."):
                # PP-StructureV3 accepts a file path or numpy array; use a temp path.
                # delete=False is required on Windows because the file must be closed
                # before PP-StructureV3 reads it; we unlink explicitly afterwards.
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                    source_img.save(tmp_path)
                try:
                    pages = list(pipeline.predict(tmp_path))
                finally:
                    os.unlink(tmp_path)
                regions = detect.extract_regions(pages[0]) if pages else []
            annotated = _annotate(source_img, regions)
            st.image(
                annotated, caption=f"{len(regions)} regions detected", use_container_width=True
            )

            # region detail table
            st.write("**Detected regions:**")
            df = pd.DataFrame(
                [
                    {
                        "label": r["label"],
                        "bbox": str(r["bbox"]),
                        "maps_to": config.PP_TYPE_TO_PUBLAYNET.get(r["label"], "—"),
                        "content_preview": r["content"][:60],
                    }
                    for r in regions
                ]
            )
            st.dataframe(df, use_container_width=True)

            # table HTML rendering (safe: no raw HTML injection)
            tables = [r for r in regions if r["table_html"]]
            if tables:
                st.write(f"**Table structure ({len(tables)} found):**")
                for i, t in enumerate(tables):
                    with st.expander(f"Table {i + 1}"):
                        _render_table_html(t["table_html"])
    else:
        st.info("Upload an image or pick a sample to begin.")

# ── Tab: Metrics ────────────────────────────────────────────────
with tab_metrics:
    st.subheader("PubLayNet mAP")
    if METRICS_JSON.exists():
        metrics = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
        st.json(metrics)
    else:
        st.info("No metrics yet. mAP evaluation requires PubLayNet val data.")
        st.caption(
            "PubLayNet val data download is pending (network-dependent). "
            "Once available, run `python scripts/evaluate.py`."
        )
