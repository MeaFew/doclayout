<div align="center">

# doclayout · Document Intelligence

**Document layout segmentation + table recognition with PP-StructureV3**

<img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/PaddleOCR-3.7-2855d6" alt="PaddleOCR">
<img src="https://img.shields.io/badge/PP--StructureV3-layout-green" alt="PP-StructureV3">
<img src="https://img.shields.io/badge/code%20style-ruff-000000?logo=ruff&logoColor=white" alt="Ruff">

</div>

---

## Overview

`doclayout` analyzes document images: it automatically identifies structural regions (titles, body text, tables, figures, lists) and recognizes table structure (as HTML). Powered by **PaddleOCR PP-StructureV3** (2025), a single pipeline handles both layout segmentation and table structure recognition.

This is the author's **document-intelligence** project, complementing visual retrieval (vizseek), time-series forecasting (foresight), and GNN fraud detection (graphguard).

## Highlights

- **Unified layout + table engine**: PP-StructureV3 outputs both region segmentation and table HTML in one pipeline — no model stitching.
- **Fine-grained region types**: PP-StructureV3 3.7 distinguishes `doc_title`/`paragraph_title`/`figure_title`/`chart`/`header` — richer than PubLayNet's 5 classes.
- **Interactive dashboard**: Streamlit — upload any document image → live inference → colored boxes + table HTML.
- **Production engineering**: CI (lint + test), dual entry points, `--quick` mode, category-mapping + bbox-conversion adapter.
- **Real engineering pitfalls solved**: paddlepaddle 3.3 oneDNN PIR bug (auto-disabled), CPU memory regression (env-limited), PP-StructureV3 version-sensitive schema (probed + adapted).

## Quick Start

```bash
pip install -r requirements.txt
make all              # samples → detect → visualize
make dashboard        # interactive UI
python scripts/detect.py --image samples/sample_paper.png   # single image
python scripts/evaluate.py --quick   # requires PubLayNet val data (see below)
```

## Why PP-StructureV3?

- **LayoutLMv3 can't detect**: HF transformers' LayoutLMv3 only has classification/NER heads, no bbox output.
- **Generic DETR lacks doc classes**: `facebook/detr-resnet-50` is COCO 80-class, no document layout categories.
- **PP-StructureV3 is turnkey**: one pipeline for layout + table + OCR, CPU-runnable, actively maintained in 2025.

## Evaluation Status

mAP quantification requires PubLayNet val data (11K images, COCO format). Acquisition is currently network-limited (IBM DAX unreliable, HF mirrors are parquet + unstable) and `pycocotools` is not installed in the current environment.

- **`evaluate.py` is currently a stub**: it raises `NotImplementedError` with instructions to manually obtain the data and install `pycocotools`.
- **To run evaluation**: place `val.json` at `data/raw/publaynet_val.json`, images in `data/raw/publaynet_val_images/`, install `pycocotools`, then run `python scripts/evaluate.py --quick`.

## License

MIT
