"""Tests for detect.py conversion/validation helpers.

These run in CI without paddle — they validate the bbox normalization and
COCO-format conversion logic that detect.py relies on.
"""

from doclayout import detect


def test_regions_to_coco_clamps_and_converts():
    regions = [
        {"label": "text", "bbox": [-10, -20, 50, 60]},
        {"label": "table", "bbox": [90, 95, 120, 130]},  # partially outside 100x100
        {"label": "figure", "bbox": [30, 30, 20, 20]},  # inverted corners
        {"label": "list", "bbox": [30, 30, -5, 40]},  # x2 negative, x1 positive
    ]
    dets = detect.regions_to_coco(regions, image_id=1, image_size=(100, 100))
    assert len(dets) == 4
    # Negative coordinates clamped to zero.
    assert dets[0]["bbox"] == [0, 0, 50, 60]
    # Coordinates beyond image dimensions clamped to image size.
    assert dets[1]["bbox"] == [90, 95, 10, 5]
    # Swapped corners sorted so width/height are non-negative.
    assert dets[2]["bbox"] == [20, 20, 10, 10]
    # Negative x2 is sorted to x1 and then clamped to zero.
    assert dets[3]["bbox"] == [0, 30, 30, 10]


def test_regions_to_coco_drops_degenerate_and_unmapped():
    regions = [
        {"label": "text", "bbox": [10, 10, 10, 20]},  # zero width
        {"label": "text", "bbox": [10, 10, 20, 10]},  # zero height
        {"label": "chart", "bbox": [10, 10, 20, 20]},  # unmapped PP type
    ]
    dets = detect.regions_to_coco(regions, image_id=1, image_size=(100, 100))
    assert dets == []


def test_regions_to_coco_scores_not_hardcoded():
    """Scores must NOT be hardcoded to 1.0 — they should vary by area/rank."""
    regions = [
        {"label": "text", "bbox": [0, 0, 90, 90]},  # large box
        {"label": "title", "bbox": [10, 10, 20, 15]},  # small box
    ]
    dets = detect.regions_to_coco(regions, image_id=1, image_size=(100, 100))
    assert len(dets) == 2
    # Scores should be in valid range and NOT all 1.0
    for d in dets:
        assert 0.1 <= d["score"] <= 0.99
    # Larger box should have higher score than smaller box
    assert dets[0]["score"] > dets[1]["score"]


def test_regions_to_coco_with_explicit_scores():
    """When explicit scores are provided, they should be used directly."""
    regions = [
        {"label": "text", "bbox": [10, 10, 50, 50]},
        {"label": "table", "bbox": [60, 60, 90, 90]},
    ]
    scores = [0.95, 0.72]
    dets = detect.regions_to_coco(regions, image_id=1, image_size=(100, 100), scores=scores)
    assert len(dets) == 2
    assert dets[0]["score"] == 0.95
    assert dets[1]["score"] == 0.72


def test_compute_detection_score_range():
    """Heuristic score must be in [0.1, 0.99]."""
    # Very small box in large image
    score = detect.compute_detection_score([0, 0, 1, 1], (1000, 1000), rank=0, total=1)
    assert 0.1 <= score <= 0.99
    # Very large box
    score = detect.compute_detection_score([0, 0, 999, 999], (1000, 1000), rank=0, total=1)
    assert 0.1 <= score <= 0.99
    assert score > 0.8  # large box should score high


def test_compute_detection_score_rank_decay():
    """Later detections should have lower scores (rank decay)."""
    bbox = [10, 10, 50, 50]
    img_size = (100, 100)
    score_first = detect.compute_detection_score(bbox, img_size, rank=0, total=5)
    score_last = detect.compute_detection_score(bbox, img_size, rank=4, total=5)
    assert score_first > score_last
