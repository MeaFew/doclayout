"""Tests for visualize.py and make_samples.py helper functions.

These run in CI without paddle — they validate color mapping, annotation
rendering, font discovery, and text wrapping logic.
"""

from pathlib import Path

from PIL import Image

from doclayout.visualize import UNMAPPED_COLOR, _color_for, annotate

# ── _color_for tests ─────────────────────────────────────────────────────────


class TestColorFor:
    def test_mapped_publaynet_label(self):
        """PP labels that map to PubLayNet should return class color."""
        color = _color_for("text")
        assert color == (31, 119, 180)  # blue from CATEGORY_COLORS

    def test_mapped_title(self):
        color = _color_for("title")
        assert color == (255, 127, 14)  # orange

    def test_mapped_table(self):
        color = _color_for("table")
        assert color == (214, 39, 40)  # red

    def test_mapped_figure(self):
        color = _color_for("figure")
        assert color == (148, 103, 189)  # purple

    def test_mapped_list(self):
        color = _color_for("list")
        assert color == (44, 160, 44)  # green

    def test_unmapped_label_returns_gray(self):
        """PP labels not in PubLayNet mapping should return gray."""
        color = _color_for("chart")
        assert color == UNMAPPED_COLOR

    def test_unknown_label_returns_gray(self):
        color = _color_for("nonexistent_label")
        assert color == UNMAPPED_COLOR

    def test_doc_title_maps_to_title_color(self):
        """doc_title maps to PubLayNet 'title' → orange."""
        color = _color_for("doc_title")
        assert color == (255, 127, 14)


# ── annotate tests ───────────────────────────────────────────────────────────


class TestAnnotate:
    def test_annotate_creates_output(self, tmp_path):
        """annotate() should create an output image file."""
        # Create a simple test image
        img = Image.new("RGB", (200, 200), "white")
        img_path = tmp_path / "test_input.png"
        img.save(img_path)

        regions = [
            {"label": "text", "bbox": [10, 10, 100, 50]},
            {"label": "table", "bbox": [20, 60, 180, 150]},
        ]
        out_path = tmp_path / "output" / "annotated.png"
        annotate(img_path, regions, out_path)

        assert out_path.exists()
        result = Image.open(out_path)
        assert result.size == (200, 200)

    def test_annotate_empty_regions(self, tmp_path):
        """annotate() with no regions should still produce valid output."""
        img = Image.new("RGB", (100, 100), "white")
        img_path = tmp_path / "test_input.png"
        img.save(img_path)

        out_path = tmp_path / "annotated.png"
        annotate(img_path, [], out_path)
        assert out_path.exists()

    def test_annotate_unmapped_label(self, tmp_path):
        """annotate() should handle unmapped labels (gray boxes)."""
        img = Image.new("RGB", (100, 100), "white")
        img_path = tmp_path / "test_input.png"
        img.save(img_path)

        regions = [{"label": "formula", "bbox": [5, 5, 50, 50]}]
        out_path = tmp_path / "annotated.png"
        annotate(img_path, regions, out_path)
        assert out_path.exists()


# ── make_samples tests ───────────────────────────────────────────────────────


class TestMakeSamples:
    def test_find_font_regular(self):
        """_find_font should locate a regular font on this system."""
        from doclayout.make_samples import _find_font

        path = _find_font(bold=False)
        assert Path(path).exists()

    def test_find_font_bold(self):
        """_find_font should locate a bold font on this system."""
        from doclayout.make_samples import _find_font

        path = _find_font(bold=True)
        assert Path(path).exists()

    def test_wrapped_lines_basic(self):
        """_wrapped_lines should wrap text to fit max width."""
        from PIL import ImageDraw, ImageFont

        from doclayout.make_samples import FONT_REGULAR, _wrapped_lines

        img = Image.new("RGB", (500, 100))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(FONT_REGULAR, 14)

        text = "This is a test sentence that should be wrapped at some point"
        lines = _wrapped_lines(text, font, max_w=200, draw=draw)
        assert len(lines) >= 2
        # Each line should fit within max_w
        for line in lines:
            assert draw.textlength(line, font=font) <= 200

    def test_wrapped_lines_short_text(self):
        """Short text that fits in one line should return single line."""
        from PIL import ImageDraw, ImageFont

        from doclayout.make_samples import FONT_REGULAR, _wrapped_lines

        img = Image.new("RGB", (500, 100))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(FONT_REGULAR, 14)

        lines = _wrapped_lines("Hi", font, max_w=400, draw=draw)
        assert lines == ["Hi"]

    def test_sample_paper_generates_image(self):
        """sample_paper() should return a valid PIL Image."""
        from doclayout.make_samples import sample_paper

        img = sample_paper()
        assert isinstance(img, Image.Image)
        assert img.size == (1000, 1300)
        assert img.mode == "RGB"

    def test_sample_report_generates_image(self):
        """sample_report() should return a valid PIL Image."""
        from doclayout.make_samples import sample_report

        img = sample_report()
        assert isinstance(img, Image.Image)
        assert img.size == (1000, 1300)

    def test_sample_invoice_generates_image(self):
        """sample_invoice() should return a valid PIL Image."""
        from doclayout.make_samples import sample_invoice

        img = sample_invoice()
        assert isinstance(img, Image.Image)
        assert img.size == (1000, 1300)
