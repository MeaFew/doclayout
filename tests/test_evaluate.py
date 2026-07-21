"""Tests for evaluate.py — mAP computation logic.

Validates the COCO-style mAP evaluation using synthetic data with known
ground truth. These tests run without pycocotools or PubLayNet data.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from doclayout.evaluate import (
    IOU_THRESHOLDS,
    _interpolate_ap,
    compute_ap_at_iou,
    compute_iou_matrix,
    create_eval_subset,
    evaluate_map,
    evaluate_with_pycocotools,
    write_metrics,
    write_per_class_csv,
)

# ── IoU tests ────────────────────────────────────────────────────────────────


class TestComputeIoU:
    def test_identical_boxes(self):
        """Identical boxes should have IoU = 1.0."""
        boxes = np.array([[10, 10, 20, 20]], dtype=np.float64)  # [x,y,w,h]
        iou = compute_iou_matrix(boxes, boxes)
        assert iou.shape == (1, 1)
        assert abs(iou[0, 0] - 1.0) < 1e-6

    def test_no_overlap(self):
        """Non-overlapping boxes should have IoU = 0.0."""
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[20, 20, 10, 10]], dtype=np.float64)
        iou = compute_iou_matrix(a, b)
        assert iou[0, 0] == 0.0

    def test_partial_overlap(self):
        """Partially overlapping boxes: known IoU value."""
        # Box A: [0,0,10,10] → area=100, covers (0,0)-(10,10)
        # Box B: [5,5,10,10] → area=100, covers (5,5)-(15,15)
        # Intersection: (5,5)-(10,10) → 5*5=25
        # Union: 100+100-25=175
        # IoU = 25/175 ≈ 0.142857
        a = np.array([[0, 0, 10, 10]], dtype=np.float64)
        b = np.array([[5, 5, 10, 10]], dtype=np.float64)
        iou = compute_iou_matrix(a, b)
        expected = 25.0 / 175.0
        assert abs(iou[0, 0] - expected) < 1e-6

    def test_contained_box(self):
        """A box fully inside another: IoU = small_area / large_area."""
        # Outer: [0,0,20,20] area=400
        # Inner: [5,5,10,10] area=100, fully inside
        # Intersection = 100, Union = 400
        # IoU = 100/400 = 0.25
        a = np.array([[0, 0, 20, 20]], dtype=np.float64)
        b = np.array([[5, 5, 10, 10]], dtype=np.float64)
        iou = compute_iou_matrix(a, b)
        assert abs(iou[0, 0] - 0.25) < 1e-6

    def test_empty_inputs(self):
        """Empty box arrays should return empty IoU matrix."""
        a = np.zeros((0, 4), dtype=np.float64)
        b = np.array([[10, 10, 5, 5]], dtype=np.float64)
        iou = compute_iou_matrix(a, b)
        assert iou.shape == (0, 1)

    def test_pairwise_matrix(self):
        """Multiple boxes produce correct pairwise matrix shape."""
        a = np.array([[0, 0, 10, 10], [20, 20, 10, 10]], dtype=np.float64)
        b = np.array([[0, 0, 10, 10], [50, 50, 10, 10], [20, 20, 10, 10]], dtype=np.float64)
        iou = compute_iou_matrix(a, b)
        assert iou.shape == (2, 3)
        # a[0] vs b[0] = identical → 1.0
        assert abs(iou[0, 0] - 1.0) < 1e-6
        # a[1] vs b[2] = identical → 1.0
        assert abs(iou[1, 2] - 1.0) < 1e-6
        # a[0] vs b[1] = no overlap → 0.0
        assert iou[0, 1] == 0.0


# ── AP computation tests ─────────────────────────────────────────────────────


class TestComputeAP:
    def test_perfect_detection(self):
        """Perfect detection (all GT matched exactly) → AP = 1.0."""
        gt = np.array([[10, 10, 20, 20], [50, 50, 30, 30]], dtype=np.float64)
        dt = np.array([[10, 10, 20, 20], [50, 50, 30, 30]], dtype=np.float64)
        scores = np.array([0.9, 0.8], dtype=np.float64)
        ap = compute_ap_at_iou(gt, dt, scores, iou_threshold=0.5)
        assert abs(ap - 1.0) < 1e-6

    def test_no_detections(self):
        """No detections → AP = 0.0."""
        gt = np.array([[10, 10, 20, 20]], dtype=np.float64)
        dt = np.zeros((0, 4), dtype=np.float64)
        scores = np.zeros(0, dtype=np.float64)
        ap = compute_ap_at_iou(gt, dt, scores, iou_threshold=0.5)
        assert ap == 0.0

    def test_no_ground_truth(self):
        """No ground truth → AP = 0.0 (no positives to find)."""
        gt = np.zeros((0, 4), dtype=np.float64)
        dt = np.array([[10, 10, 20, 20]], dtype=np.float64)
        scores = np.array([0.9], dtype=np.float64)
        ap = compute_ap_at_iou(gt, dt, scores, iou_threshold=0.5)
        assert ap == 0.0

    def test_false_positives_reduce_ap(self):
        """FP ranked BEFORE TP (higher score) should reduce AP below 1.0."""
        gt = np.array([[10, 10, 20, 20]], dtype=np.float64)
        # FP has higher score → ranked first → reduces AP
        dt = np.array([[80, 80, 10, 10], [10, 10, 20, 20]], dtype=np.float64)
        scores = np.array([0.9, 0.8], dtype=np.float64)  # FP scored higher
        ap = compute_ap_at_iou(gt, dt, scores, iou_threshold=0.5)
        assert 0.0 < ap < 1.0

    def test_score_ordering_matters(self):
        """Higher-scoring TP before FP should give better AP than reversed."""
        gt = np.array([[10, 10, 20, 20]], dtype=np.float64)
        dt = np.array([[10, 10, 20, 20], [80, 80, 10, 10]], dtype=np.float64)

        # TP has higher score → good ordering
        scores_good = np.array([0.9, 0.3], dtype=np.float64)
        ap_good = compute_ap_at_iou(gt, dt, scores_good, iou_threshold=0.5)

        # FP has higher score → bad ordering
        scores_bad = np.array([0.3, 0.9], dtype=np.float64)
        ap_bad = compute_ap_at_iou(gt, dt, scores_bad, iou_threshold=0.5)

        assert ap_good > ap_bad

    def test_iou_threshold_sensitivity(self):
        """Stricter IoU threshold should give lower or equal AP."""
        gt = np.array([[10, 10, 20, 20]], dtype=np.float64)
        # Detection slightly offset — IoU < 1.0 but > 0.5
        dt = np.array([[12, 12, 20, 20]], dtype=np.float64)
        scores = np.array([0.9], dtype=np.float64)

        ap_50 = compute_ap_at_iou(gt, dt, scores, iou_threshold=0.5)
        ap_95 = compute_ap_at_iou(gt, dt, scores, iou_threshold=0.95)
        assert ap_50 >= ap_95

    def test_second_best_gt_retry(self):
        """Best GT already claimed → detection must try the next-best GT.

        Two identical detections over two heavily overlapping GTs: the first
        claims GT0, the second must match GT1 (IoU ≈ 0.82 ≥ 0.5) instead of
        being scored FP.
        """
        gt = np.array([[10, 10, 20, 20], [11, 11, 20, 20]], dtype=np.float64)
        dt = np.array([[10, 10, 20, 20], [10, 10, 20, 20]], dtype=np.float64)
        scores = np.array([0.9, 0.8], dtype=np.float64)
        ap = compute_ap_at_iou(gt, dt, scores, iou_threshold=0.5)
        assert abs(ap - 1.0) < 1e-6


# ── Interpolation tests ──────────────────────────────────────────────────────


class TestInterpolateAP:
    def test_perfect_precision_recall(self):
        """All precision=1.0 at all recall levels → AP = 1.0."""
        precision = np.array([1.0, 1.0, 1.0])
        recall = np.array([0.33, 0.66, 1.0])
        ap = _interpolate_ap(precision, recall)
        assert abs(ap - 1.0) < 1e-6

    def test_zero_precision(self):
        """All precision=0 → AP = 0."""
        precision = np.array([0.0, 0.0])
        recall = np.array([0.0, 0.0])
        ap = _interpolate_ap(precision, recall)
        assert ap == 0.0


# ── Full mAP evaluation tests ────────────────────────────────────────────────


class TestEvaluateMAP:
    def _make_gt(self, image_id, cat_id, bbox):
        return {"image_id": image_id, "category_id": cat_id, "bbox": bbox}

    def _make_dt(self, image_id, cat_id, bbox, score):
        return {"image_id": image_id, "category_id": cat_id, "bbox": bbox, "score": score}

    def test_perfect_single_class(self):
        """Perfect detections for one class → mAP = 1.0."""
        gt = [self._make_gt(1, 1, [10, 10, 20, 20])]
        dt = [self._make_dt(1, 1, [10, 10, 20, 20], 0.9)]
        result = evaluate_map(gt, dt, category_ids=[1])
        assert abs(result["map_50"] - 1.0) < 1e-6
        assert abs(result["map_5095"] - 1.0) < 1e-6

    def test_perfect_multi_class(self):
        """Perfect detections for multiple classes → mAP = 1.0."""
        gt = [
            self._make_gt(1, 1, [10, 10, 20, 20]),
            self._make_gt(1, 2, [50, 50, 30, 30]),
            self._make_gt(2, 1, [5, 5, 15, 15]),
        ]
        dt = [
            self._make_dt(1, 1, [10, 10, 20, 20], 0.95),
            self._make_dt(1, 2, [50, 50, 30, 30], 0.90),
            self._make_dt(2, 1, [5, 5, 15, 15], 0.85),
        ]
        result = evaluate_map(gt, dt, category_ids=[1, 2])
        assert abs(result["map_50"] - 1.0) < 1e-6

    def test_no_detections_gives_zero(self):
        """No detections → mAP = 0.0."""
        gt = [self._make_gt(1, 1, [10, 10, 20, 20])]
        dt = []
        result = evaluate_map(gt, dt, category_ids=[1])
        assert result["map_50"] == 0.0
        assert result["map_5095"] == 0.0

    def test_false_positives_reduce_map(self):
        """FP with higher score than TP should reduce mAP."""
        gt = [self._make_gt(1, 1, [10, 10, 20, 20])]
        dt = [
            self._make_dt(1, 1, [80, 80, 10, 10], 0.9),  # FP scored higher
            self._make_dt(1, 1, [10, 10, 20, 20], 0.5),  # TP scored lower
        ]
        result = evaluate_map(gt, dt, category_ids=[1])
        assert 0.0 < result["map_50"] < 1.0

    def test_per_class_ap_populated(self):
        """per_class dict should have entries for all requested categories."""
        gt = [
            self._make_gt(1, 1, [10, 10, 20, 20]),
            self._make_gt(1, 2, [50, 50, 30, 30]),
        ]
        dt = [self._make_dt(1, 1, [10, 10, 20, 20], 0.9)]
        result = evaluate_map(gt, dt, category_ids=[1, 2])
        assert 1 in result["per_class"]
        assert 2 in result["per_class"]
        # Class 1 detected perfectly
        assert result["per_class"][1]["ap_50"] > 0.9
        # Class 2 not detected
        assert result["per_class"][2]["ap_50"] == 0.0

    def test_map_5095_leq_map_50(self):
        """mAP@0.50:0.95 should be <= mAP@0.50 (stricter thresholds)."""
        gt = [self._make_gt(1, 1, [10, 10, 20, 20])]
        # Slightly offset detection
        dt = [self._make_dt(1, 1, [12, 12, 20, 20], 0.9)]
        result = evaluate_map(gt, dt, category_ids=[1])
        assert result["map_5095"] <= result["map_50"] + 1e-9

    def test_multiple_images(self):
        """Evaluation across multiple images works correctly."""
        gt = [
            self._make_gt(1, 1, [10, 10, 20, 20]),
            self._make_gt(2, 1, [30, 30, 25, 25]),
            self._make_gt(3, 1, [5, 5, 40, 40]),
        ]
        dt = [
            self._make_dt(1, 1, [10, 10, 20, 20], 0.95),
            self._make_dt(2, 1, [30, 30, 25, 25], 0.90),
            self._make_dt(3, 1, [5, 5, 40, 40], 0.85),
        ]
        result = evaluate_map(gt, dt, category_ids=[1])
        assert abs(result["map_50"] - 1.0) < 1e-6

    def test_no_cross_image_matching(self):
        """Regression: a detection must never match GT from another image.

        Same coordinates on different images: with cross-image pooling this
        would score a perfect TP; correct behavior is FP (image 2 has no GT)
        plus a missed GT on image 1 → mAP = 0.
        """
        gt = [self._make_gt(1, 1, [10, 10, 20, 20])]
        dt = [self._make_dt(2, 1, [10, 10, 20, 20], 0.9)]
        result = evaluate_map(gt, dt, category_ids=[1])
        assert result["map_50"] == 0.0
        assert result["map_5095"] == 0.0

    def test_cross_image_fp_does_not_consume_gt(self):
        """Regression: FPs on other images must not consume this image's GT."""
        gt = [self._make_gt(1, 1, [10, 10, 20, 20])]
        dt = [
            self._make_dt(2, 1, [10, 10, 20, 20], 0.95),  # wrong image → FP
            self._make_dt(1, 1, [10, 10, 20, 20], 0.90),  # correct image → TP
        ]
        result = evaluate_map(gt, dt, category_ids=[1])
        # Recall reaches 1.0 via the second detection; mAP > 0 but < 1 (FP first).
        assert 0.0 < result["map_50"] < 1.0

    def test_empty_gt_and_dt(self):
        """Both empty → mAP = 0."""
        result = evaluate_map([], [], category_ids=[1])
        assert result["map_50"] == 0.0
        assert result["map_5095"] == 0.0


# ── pycocotools integration tests ────────────────────────────────────────────


class TestSubsetFiltering:
    """main() must filter detections to the GT image set (docstring protocol)."""

    def test_detections_outside_gt_images_not_counted(self, tmp_path, monkeypatch):
        """Detections on out-of-subset images must not count as false positives.

        GT covers image 1 only; the detections file also has a perfect-box
        detection on image 2. Without filtering, that detection would match
        image 1's GT (same coordinates) and inflate mAP.
        """
        from doclayout import evaluate as ev

        gt_data = {
            "images": [{"id": 1, "width": 100, "height": 100}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": "text"}],
        }
        dt_data = [
            {"image_id": 2, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.99},
            {"image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10], "score": 0.5},
        ]
        gt_path = tmp_path / "gt.json"
        dt_path = tmp_path / "dt.json"
        gt_path.write_text(json.dumps(gt_data), encoding="utf-8")
        dt_path.write_text(json.dumps(dt_data), encoding="utf-8")

        metrics_out = tmp_path / "metrics.json"
        monkeypatch.setattr(ev, "METRICS_JSON", metrics_out)
        monkeypatch.setattr(ev, "PER_CLASS_CSV", tmp_path / "per_class.csv")

        def _raise_import_error(*args, **kwargs):
            raise ImportError("forced numpy fallback")

        monkeypatch.setattr(ev, "evaluate_with_pycocotools", _raise_import_error)
        monkeypatch.setattr(sys, "argv", ["evaluate", "--gt", str(gt_path), "--dt", str(dt_path)])

        ev.main()

        data = json.loads(metrics_out.read_text(encoding="utf-8"))
        assert data["map_50"] == 0.0
        assert data["map_5095"] == 0.0


class TestPycocotoolsEval:
    @pytest.fixture()
    def synthetic_coco_files(self, tmp_path):
        """Create synthetic COCO GT + DT files for testing."""
        gt_data = {
            "images": [
                {"id": 1, "width": 100, "height": 100},
                {"id": 2, "width": 100, "height": 100},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 2, "bbox": [50, 50, 30, 30], "area": 900, "iscrowd": 0},
                {"id": 3, "image_id": 2, "category_id": 1, "bbox": [5, 5, 40, 40], "area": 1600, "iscrowd": 0},
            ],
            "categories": [
                {"id": 1, "name": "text"},
                {"id": 2, "name": "title"},
            ],
        }
        dt_data = [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "score": 0.95},
            {"image_id": 1, "category_id": 2, "bbox": [50, 50, 30, 30], "score": 0.90},
            {"image_id": 2, "category_id": 1, "bbox": [5, 5, 40, 40], "score": 0.85},
        ]
        gt_path = tmp_path / "gt.json"
        dt_path = tmp_path / "dt.json"
        gt_path.write_text(json.dumps(gt_data), encoding="utf-8")
        dt_path.write_text(json.dumps(dt_data), encoding="utf-8")
        return gt_path, dt_path

    def test_perfect_detections_pycocotools(self, synthetic_coco_files):
        """pycocotools eval with perfect detections → mAP = 1.0."""
        gt_path, dt_path = synthetic_coco_files
        result = evaluate_with_pycocotools(gt_path, dt_path)
        assert abs(result["map_50"] - 1.0) < 1e-4
        assert abs(result["map_5095"] - 1.0) < 1e-4

    def test_per_class_populated_pycocotools(self, synthetic_coco_files):
        """pycocotools eval populates per-class AP."""
        gt_path, dt_path = synthetic_coco_files
        result = evaluate_with_pycocotools(gt_path, dt_path)
        assert 1 in result["per_class"]
        assert 2 in result["per_class"]


# ── Subset creation tests ────────────────────────────────────────────────────


class TestCreateSubset:
    def test_subset_size(self, tmp_path):
        """Subset should have at most subset_size images."""
        coco_data = {
            "images": [{"id": i, "width": 100, "height": 100} for i in range(1, 21)],
            "annotations": [
                {"id": i, "image_id": i, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}
                for i in range(1, 21)
            ],
            "categories": [{"id": 1, "name": "text"}],
        }
        val_path = tmp_path / "val.json"
        val_path.write_text(json.dumps(coco_data), encoding="utf-8")

        subset = create_eval_subset(val_path, subset_size=5, seed=42)
        assert len(subset["images"]) == 5
        # Annotations should only reference subset images
        subset_ids = {img["id"] for img in subset["images"]}
        for ann in subset["annotations"]:
            assert ann["image_id"] in subset_ids

    def test_subset_deterministic(self, tmp_path):
        """Same seed → same subset."""
        coco_data = {
            "images": [{"id": i, "width": 100, "height": 100} for i in range(1, 51)],
            "annotations": [],
            "categories": [{"id": 1, "name": "text"}],
        }
        val_path = tmp_path / "val.json"
        val_path.write_text(json.dumps(coco_data), encoding="utf-8")

        s1 = create_eval_subset(val_path, subset_size=10, seed=42)
        s2 = create_eval_subset(val_path, subset_size=10, seed=42)
        ids1 = [img["id"] for img in s1["images"]]
        ids2 = [img["id"] for img in s2["images"]]
        assert ids1 == ids2


# ── Report writing tests ─────────────────────────────────────────────────────


class TestReportWriting:
    def test_write_metrics(self, tmp_path):
        """write_metrics creates valid JSON with expected keys."""
        metrics = {"map_5095": 0.75, "map_50": 0.92, "per_class": {}}
        out = tmp_path / "reports" / "metrics.json"
        write_metrics(metrics, out)
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["map_5095"] == 0.75
        assert data["map_50"] == 0.92

    def test_write_per_class_csv(self, tmp_path):
        """write_per_class_csv creates CSV with correct structure."""
        metrics = {
            "map_5095": 0.75,
            "map_50": 0.92,
            "per_class": {
                1: {"ap_5095": 0.80, "ap_50": 0.95},
                2: {"ap_5095": 0.70, "ap_50": 0.89},
            },
        }
        out = tmp_path / "reports" / "per_class_ap.csv"
        write_per_class_csv(metrics, out)
        assert out.exists()
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3  # header + 2 classes
        assert "category_id" in lines[0]


# ── IOU_THRESHOLDS sanity ────────────────────────────────────────────────────


def test_iou_thresholds_range():
    """IOU_THRESHOLDS should span 0.50 to 0.95 in steps of 0.05."""
    assert len(IOU_THRESHOLDS) == 10
    assert abs(IOU_THRESHOLDS[0] - 0.5) < 1e-6
    assert abs(IOU_THRESHOLDS[-1] - 0.95) < 1e-6
