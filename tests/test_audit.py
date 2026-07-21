"""Tests for audit_consistency.py — README vs metrics cross-check.

These run in CI without paddle — they validate the metric extraction and
comparison logic.
"""

from pathlib import Path

from doclayout.audit_consistency import check, read_readme_metric


def test_read_readme_metric_found(tmp_path):
    """Should extract numeric metric from README-style text."""
    readme = tmp_path / "README.md"
    readme.write_text("# Results\nmAP@0.50 = 0.9234\nmAP@0.50:0.95 0.7012\n", encoding="utf-8")
    val = read_readme_metric(readme, "mAP@0.50")
    assert val is not None
    assert abs(val - 0.9234) < 1e-4


def test_read_readme_metric_not_found(tmp_path):
    """Should return None when metric is absent."""
    readme = tmp_path / "README.md"
    readme.write_text("# No metrics here\n", encoding="utf-8")
    val = read_readme_metric(readme, "mAP@0.50")
    assert val is None


def test_read_readme_metric_map5095(tmp_path):
    """Should extract mAP@0.50:0.95 correctly."""
    readme = tmp_path / "README.md"
    readme.write_text("mAP@0.50:0.95 = 0.6789\n", encoding="utf-8")
    val = read_readme_metric(readme, "mAP@0.50:0.95")
    assert val is not None
    assert abs(val - 0.6789) < 1e-4


def test_read_readme_metric_unordered(tmp_path):
    """mAP@0.50 must not capture the 0.95 from a mAP@0.50:0.95 line listed first."""
    readme = tmp_path / "README.md"
    readme.write_text("# Results\nmAP@0.50:0.95 = 0.7012\nmAP@0.50 = 0.9234\n", encoding="utf-8")
    val_50 = read_readme_metric(readme, "mAP@0.50")
    val_5095 = read_readme_metric(readme, "mAP@0.50:0.95")
    assert val_50 is not None
    assert abs(val_50 - 0.9234) < 1e-4
    assert val_5095 is not None
    assert abs(val_5095 - 0.7012) < 1e-4


def test_check_pass():
    """check() returns True for passing condition."""
    assert check(True, "test pass") is True


def test_check_fail():
    """check() returns False for failing condition."""
    assert check(False, "test fail") is False
