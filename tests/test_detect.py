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
