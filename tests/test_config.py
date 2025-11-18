"""Tests for doclayout configuration + category mapping logic.

These run in CI without paddle/pycocotools — they validate the config wiring
and the PP-StructureV3 → PubLayNet category mapping that detect.py relies on.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402


def test_base_dir_is_this_repo():
    assert config.BASE_DIR.name == "doclayout"
    assert config.BASE_DIR.is_dir()


def test_publaynet_categories_are_five():
    assert len(config.PUBLAYNET_CATEGORIES) == 5
    names = [c["name"] for c in config.PUBLAYNET_CATEGORIES]
    assert names == ["text", "title", "list", "table", "figure"]


def test_category_id_name_roundtrip():
    for cat in config.PUBLAYNET_CATEGORIES:
        assert config.CATEGORY_ID_TO_NAME[cat["id"]] == cat["name"]
        assert config.CATEGORY_NAME_TO_ID[cat["name"]] == cat["id"]


def test_pp_mapping_covers_all_publaynet_classes():
    """Every PubLayNet class must be reachable via the PP-StructureV3 mapping."""
    publaynet_names = {c["name"] for c in config.PUBLAYNET_CATEGORIES}
    mapped_targets = set(config.PP_TYPE_TO_PUBLAYNET.values())
    assert publaynet_names <= mapped_targets, f"missing: {publaynet_names - mapped_targets}"


def test_pp_mapping_targets_only_valid_publaynet():
    """PP → PubLayNet mapping must not invent nonexistent classes."""
    valid = {c["name"] for c in config.PUBLAYNET_CATEGORIES}
    for pp_type, target in config.PP_TYPE_TO_PUBLAYNET.items():
        assert target in valid, f"{pp_type} -> {target} is not a valid PubLayNet class"


def test_category_colors_cover_all_classes():
    names = {c["name"] for c in config.PUBLAYNET_CATEGORIES}
    assert names <= set(config.CATEGORY_COLORS)


def test_ensure_dirs_creates_outputs():
    config.ensure_dirs()
    for d in (
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.MODELS_DIR,
        config.REPORTS_DIR,
    ):
        assert d.exists(), f"{d} not created"
