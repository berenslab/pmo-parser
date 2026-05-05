"""Tests for page/base_page.py — convert_to_string and sort_bboxes_in_reading_order."""

from __future__ import annotations

from pmo_parser.bounding_boxes import BBox, ReadingOrder, TextBox
from pmo_parser.page.base_page import convert_to_string, sort_bboxes_in_reading_order

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tbox(x0, y0, x1, y1, text) -> TextBox:
    return TextBox(
        x0=x0, y0=y0, x1=x1, y1=y1, text=text, reading_order=ReadingOrder.LTR_TD
    )


def bbox(x0, y0, x1, y1) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


# Reasonable letter metrics used throughout
LW = 8.0  # mean_letter_width
LH = 12.0  # mean_letter_height


# ===========================================================================
# convert_to_string
# ===========================================================================


def test_convert_single_word():
    """A single TextBox returns its text unchanged."""
    boxes = [tbox(0, 0, 40, 12, "hello")]
    assert convert_to_string(boxes, LW, LH) == "hello"


def test_convert_touching_words():
    """Two boxes that are closer than 0.25 * letter_width are joined without any separator."""
    # distance = 0 (touching), threshold = 0.25 * 8 = 2
    b1 = tbox(0, 0, 20, 12, "foo")
    b2 = tbox(20, 0, 40, 12, "bar")  # distance 0
    assert convert_to_string([b1, b2], LW, LH) == "foobar"


def test_convert_spaced_words():
    """Two boxes with a gap between 0.25*lw and lw (or close y rows) get a space."""
    # distance = 4, which is 0.25*8=2 < 4 < 8=lw → space
    b1 = tbox(0, 0, 20, 12, "foo")
    b2 = tbox(24, 0, 44, 12, "bar")  # x-gap = 4
    result = convert_to_string([b1, b2], LW, LH)
    assert result == "foo bar"


def test_convert_far_words():
    """Two boxes farther than letter_width apart (and not vertically close) get a tab."""
    # x-gap = 50 > 8 = lw, y-gap large → tab
    b1 = tbox(0, 0, 20, 12, "col1")
    b2 = tbox(70, 30, 110, 42, "col2")  # large x-gap, y-distance also large
    result = convert_to_string([b1, b2], LW, LH)
    assert result == "col1\tcol2"


# ===========================================================================
# sort_bboxes_in_reading_order
# ===========================================================================


def test_sort_empty():
    """An empty list returns an empty list."""
    assert sort_bboxes_in_reading_order([]) == []


def test_sort_single():
    """A single-element list is returned as-is."""
    b = bbox(10, 10, 50, 30)
    assert sort_bboxes_in_reading_order([b]) == [b]


def test_sort_top_left_before_bottom_right():
    """An element with a smaller y0 comes before one with a larger y0."""
    top = bbox(10, 0, 50, 20)
    bottom = bbox(10, 100, 50, 120)
    result = sort_bboxes_in_reading_order([bottom, top])
    assert result[0] is top
    assert result[1] is bottom


def test_sort_same_row_left_to_right():
    """Within the same row (y within margin), elements are sorted by x0."""
    left = bbox(0, 10, 40, 22)
    right = bbox(100, 10, 140, 22)
    result = sort_bboxes_in_reading_order([right, left])
    assert result[0] is left
    assert result[1] is right


def test_sort_margin_groups_close_rows():
    """
    Two elements whose y0 values differ by less than margin are treated as the same row,
    then sorted by x0 within that row.
    """
    # margin default = 3; Δy = 2 → same row
    left = bbox(0, 10, 40, 22)
    right = bbox(100, 12, 140, 24)  # y0 differs by 2 < 3
    result = sort_bboxes_in_reading_order([right, left])
    assert result[0] is left
    assert result[1] is right
