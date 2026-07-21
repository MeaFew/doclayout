"""Cross-reference audit: README claims vs. actual pipeline outputs.

Verifies that the mAP numbers declared in README.md match the values in
reports/metrics.json. Run after `make all`.

Usage: python -m doclayout.audit_consistency
"""

import json
import logging
import re
import sys
from pathlib import Path

from doclayout.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def read_readme_metric(readme_path: Path, metric_name: str) -> float | None:
    """Extract a numeric metric from README.md.

    Matches patterns like `mAP@0.5 = 0.123` or `mAP@0.50:0.95 0.123`.
    The `(?!:?\\d)` lookahead prevents a short name from prefix-matching a
    longer one: `mAP@0.50` must not capture the `0.95` of a `mAP@0.50:0.95`
    line that happens to appear first.
    """
    text = readme_path.read_text(encoding="utf-8")
    pattern = rf"{re.escape(metric_name)}(?!:?\d)[^\d]*?(\d+\.\d+)"
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None


def check(condition: bool, msg: str) -> bool:
    if condition:
        logger.info(f"  PASS: {msg}")
    else:
        logger.info(f"  FAIL: {msg}")
    return condition


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    readme = root / "README.md"
    metrics_json = root / "reports" / "metrics.json"

    # No-op during scaffolding (keeps `make verify` green before the pipeline runs).
    if not metrics_json.exists():
        logger.info("No reports/metrics.json found — run `make all` first.")
        logger.info("(Skipping README-vs-output audit; nothing to cross-check.)")
        return
    if not readme.exists():
        logger.info("No README.md found — skipping audit.")
        return

    with open(metrics_json, encoding="utf-8") as f:
        actual = json.load(f)

    passed = 0
    failed = 0

    checks = [
        ("mAP@0.50", actual.get("map_50")),
        ("mAP@0.50:0.95", actual.get("map_5095")),
    ]
    for label, actual_val in checks:
        if actual_val is None:
            continue
        readme_val = read_readme_metric(readme, label)
        if readme_val is None:
            logger.info(f"  SKIP: {label} not found in README (nothing to verify)")
            continue
        ok = check(
            abs(readme_val - actual_val) < 0.005,
            f"{label}: README={readme_val:.4f}, actual={actual_val:.4f}",
        )
        if ok:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    logger.info(f"\n{'=' * 40}")
    logger.info(f"Results: {passed}/{total} passed, {failed} failed")
    if failed > 0:
        logger.info("ACTION: Update README.md or pipeline to resolve mismatches.")
        sys.exit(1)


if __name__ == "__main__":
    setup_logging()
    main()
