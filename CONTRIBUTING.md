# Contributing Guide

感谢你对本项目的兴趣。本指南面向希望本地运行、调试或扩展 doclayout 文档版面分析项目的开发者。

## 环境准备

```bash
# 1. 克隆仓库
git clone https://github.com/MeaFew/doclayout.git
cd doclayout

# 2. 创建虚拟环境 (Python 3.11, paddlepaddle 暂不支持 3.12)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖（CPU）
pip install -r requirements.txt
# 注意：paddleocr 会自动安装 paddlex，不要单独装 paddlex（版本冲突）
# PP-StructureV3 需要额外装 OCR 子依赖：
pip install "paddlex[ocr]==$(python -c 'import paddlex; print(paddlex.__version__)')"
```

## 数据准备

```bash
# 生成合成样本文档（含真实渲染文字，用于 dashboard 演示）
python -m doclayout.make_samples

# PubLayNet val（mAP 评估用）——当前需手动获取
# 放到 data/raw/publaynet_val.json + data/raw/publaynet_val_images/
# download_data.py 为占位实现，会自动失败并提示手动获取。
```

## 开发工作流

```bash
make verify    # lint (ruff) + format-check + test + audit
make all       # samples → detect → visualize
make dashboard # 启动看板
make evaluate  # 需先手动准备 PubLayNet val 数据（见上文"数据准备"）
```

- **代码风格**：ruff（lint + format），pre-commit hook 已配置。
- **测试**：`pytest tests/`。CI 不装 paddle（太重），测试只覆盖配置接线和纯 Python 的区域→COCO 映射逻辑（刻意不 import paddle，故无需 importorskip）；推理/评估脚本在本地运行。

## 重要约定

### oneDNN 必须禁用

paddlepaddle 3.3.x 在 Windows CPU 上启用 oneDNN 会导致 PIR 执行器崩溃（`ConvertPirAttribute2RuntimeAttribute not support`）。所有加载 PP-StructureV3 的代码必须：
- import 前设 `FLAGS_use_mkldnn=0`
- 构造时传 `enable_mkldnn=False`（见 `config.ENABLE_MKLDNN`）

### PP-StructureV3 schema 适配

结果结构版本敏感（issue #15283）。实测（paddleocr 3.7）：
- 区域在 `page.json["res"]["parsing_res_list"]`（非文档说的顶层）
- 字段名是 `block_label`/`block_bbox`（非文档说的 `layout_type`/`layout_bbox`）

`detect.py` 的 `extract_regions()` 用 `res.` 嵌套 + 双字段名 fallback 适配。改动推理逻辑时保留这个容错。

### 类别映射

PP-StructureV3 的 label 集（`doc_title`/`paragraph_title`/`chart`/`formula`/...）比 PubLayNet 5 类丰富。`config.PP_TYPE_TO_PUBLAYNET` 做映射，未映射的类型（chart/formula/header）在 mAP 评估时丢弃，但**可视化时保留**（灰色框）以忠实反映模型输出。
