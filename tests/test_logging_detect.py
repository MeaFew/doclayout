"""Tests for logging_setup.py and detect.extract_regions.

These run in CI without paddle — they validate logging wiring and the
region extraction logic from PP-StructureV3 page results.
"""

import logging

from doclayout.logging_setup import get_logger, setup_logging


def test_get_logger_returns_logger():
    """get_logger should return a standard logging.Logger."""
    lg = get_logger("test.module")
    assert isinstance(lg, logging.Logger)
    assert lg.name == "test.module"


def test_setup_logging_idempotent():
    """Calling setup_logging multiple times should not add duplicate handlers."""
    setup_logging()
    root = logging.getLogger()
    handler_count = len(root.handlers)
    setup_logging()
    assert len(root.handlers) == handler_count


def test_setup_logging_sets_level():
    """setup_logging should set the root logger level."""
    setup_logging(level=logging.DEBUG)
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    # Reset to INFO for other tests
    setup_logging(level=logging.INFO)


# ── extract_regions tests ────────────────────────────────────────────────────


class FakePage:
    """Mimics a PP-StructureV3 LayoutParsingResultV2 page object."""

    def __init__(self, json_data):
        self.json = json_data


class TestExtractRegions:
    def test_basic_extraction(self):
        """Should extract regions from standard parsing_res_list format."""
        from doclayout.detect import extract_regions

        page = FakePage({
            "res": {
                "parsing_res_list": [
                    {"block_label": "text", "block_bbox": [10, 20, 100, 200], "block_content": "hello"},
                    {"block_label": "table", "block_bbox": [50, 60, 150, 250], "block_content": "<table></table>"},
                ]
            }
        })
        regions = extract_regions(page)
        assert len(regions) == 2
        assert regions[0]["label"] == "text"
        assert regions[0]["bbox"] == [10, 20, 100, 200]
        assert regions[0]["content"] == "hello"
        assert regions[1]["label"] == "table"

    def test_empty_page(self):
        """Should return empty list for page with no regions."""
        from doclayout.detect import extract_regions

        page = FakePage({"res": {"parsing_res_list": []}})
        regions = extract_regions(page)
        assert regions == []

    def test_missing_res_key(self):
        """Should handle missing 'res' key gracefully."""
        from doclayout.detect import extract_regions

        page = FakePage({"other_key": "value"})
        regions = extract_regions(page)
        assert regions == []

    def test_non_dict_data(self):
        """Should handle non-dict page.json gracefully."""
        from doclayout.detect import extract_regions

        page = FakePage("not a dict")
        regions = extract_regions(page)
        assert regions == []

    def test_drops_invalid_bbox(self):
        """Should drop regions with missing or invalid bbox."""
        from doclayout.detect import extract_regions

        page = FakePage({
            "res": {
                "parsing_res_list": [
                    {"block_label": "text", "block_bbox": None, "block_content": "x"},
                    {"block_label": "text", "block_bbox": [10, 20], "block_content": "y"},
                    {"block_label": "text", "block_bbox": [10, 20, 30, 40], "block_content": "z"},
                ]
            }
        })
        regions = extract_regions(page)
        assert len(regions) == 1
        assert regions[0]["content"] == "z"

    def test_table_html_extraction(self):
        """Should extract table HTML from table_res.pred_html."""
        from doclayout.detect import extract_regions

        page = FakePage({
            "res": {
                "parsing_res_list": [
                    {
                        "block_label": "table",
                        "block_bbox": [10, 20, 100, 200],
                        "block_content": "",
                        "table_res": {"pred_html": "<table><tr><td>cell</td></tr></table>"},
                    }
                ]
            }
        })
        regions = extract_regions(page)
        assert len(regions) == 1
        assert regions[0]["table_html"] == "<table><tr><td>cell</td></tr></table>"

    def test_fallback_layout_keys(self):
        """Should fall back to layout_type/layout_bbox keys."""
        from doclayout.detect import extract_regions

        page = FakePage({
            "res": {
                "parsing_res_list": [
                    {"layout_type": "figure", "layout_bbox": [5, 5, 50, 50], "text": "img"},
                ]
            }
        })
        regions = extract_regions(page)
        assert len(regions) == 1
        assert regions[0]["label"] == "figure"
        assert regions[0]["bbox"] == [5, 5, 50, 50]
