"""Tests for bounding_boxes.py — BBox, ImageBBox, TextBox, join_in_reading_order."""

from __future__ import annotations

import pytest
from PIL import Image

from pmo_parser.bounding_boxes import (
    BBox,
    ImageBBox,
    ReadingOrder,
    TextBox,
    join_in_reading_order,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bbox(x0, y0, x1, y1) -> BBox:
    """Shorthand constructor so test bodies stay readable."""
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def tbox(x0, y0, x1, y1, text="t", reading_order=ReadingOrder.LTR_TD) -> TextBox:
    """Shorthand TextBox constructor with sensible defaults."""
    return TextBox(x0=x0, y0=y0, x1=x1, y1=y1, text=text, reading_order=reading_order)


def make_image(width: int, height: int) -> Image.Image:
    """Create a plain RGB image with no file I/O required."""
    return Image.new("RGB", (width, height))


# ===========================================================================
# BBox
# ===========================================================================


def test_bbox_width_height():
    """Width and height are derived from the corner coordinates."""
    b = bbox(10, 20, 30, 50)
    assert b.width == 20
    assert b.height == 30


def test_bbox_area():
    """Area equals width * height."""
    b = bbox(0, 0, 4, 5)
    assert b.area == 20


def test_bbox_center():
    """Center is the midpoint of the box."""
    b = bbox(0, 0, 10, 20)
    assert b.center == (5.0, 10.0)


def test_bbox_width_setter():
    """Setting width extends x1; x0 is left unchanged."""
    b = bbox(10, 0, 20, 10)
    b.width = 15
    assert b.x1 == 25
    assert b.x0 == 10


def test_bbox_height_setter():
    """Setting height extends y1; y0 is left unchanged."""
    b = bbox(0, 10, 10, 20)
    b.height = 5
    assert b.y1 == 15
    assert b.y0 == 10


def test_bbox_shift():
    """Shift adds the given offsets to all four coordinates."""
    b = bbox(1, 2, 3, 4)
    b.shift(10, 20)
    assert b.x0 == 11
    assert b.y0 == 22
    assert b.x1 == 13
    assert b.y1 == 24


def test_bbox_copy_is_independent():
    """copy() returns a new object; mutating it does not affect the original."""
    original = bbox(1, 2, 3, 4)
    copy = original.copy()
    copy.x0 = 99
    assert original.x0 == 1


def test_bbox_to_dict_from_dict_roundtrip():
    """Serializing and deserializing via to_dict/from_dict preserves all coordinates."""
    b = bbox(1.5, 2.5, 3.5, 4.5)
    assert BBox.from_dict(b.to_dict()).is_equal(b)


def test_bbox_union_two_boxes():
    """union() with a single BBox returns the tightest enclosing box."""
    b1 = bbox(0, 0, 5, 5)
    b2 = bbox(3, 3, 10, 10)
    u = b1.union(b2)
    assert u.x0 == 0 and u.y0 == 0
    assert u.x1 == 10 and u.y1 == 10


def test_bbox_union_boxes_static():
    """BBox.union_boxes([b1, b2]) gives the same result as b1.union(b2)."""
    b1 = bbox(0, 0, 5, 5)
    b2 = bbox(3, 3, 10, 10)
    u = BBox.union_boxes([b1, b2])
    assert u.is_equal(b1.union(b2))


def test_bbox_union_list_overload():
    """union() accepts a list of boxes and covers all of them."""
    b1 = bbox(0, 0, 5, 5)
    b2 = bbox(3, 3, 8, 8)
    b3 = bbox(7, 7, 10, 10)
    u_instance = b1.union([b2, b3])
    u_static = BBox.union_boxes([b1, b2, b3])
    assert u_instance.is_equal(u_static)


def test_bbox_is_equal_true():
    """is_equal() returns True for boxes with identical coordinates."""
    assert bbox(1, 2, 3, 4).is_equal(bbox(1, 2, 3, 4))


def test_bbox_is_equal_false():
    """is_equal() returns False when any coordinate differs."""
    assert not bbox(1, 2, 3, 4).is_equal(bbox(1, 2, 3, 5))


def test_bbox_eq_raises():
    """
    Documents a known bug: __eq__ raises DeprecationWarning
    instead of returning a bool, making == unusable. Update once §1.4 is fixed.
    """
    b1 = bbox(1, 2, 3, 4)
    b2 = bbox(1, 2, 3, 4)
    with pytest.raises(DeprecationWarning):
        b1 == b2


def test_bbox_overlap_ratio_no_overlap():
    """Completely disjoint boxes have an overlap ratio of 0.0."""
    b1 = bbox(0, 0, 5, 5)
    b2 = bbox(10, 10, 20, 20)
    assert b1.overlap_ratio(b2) == 0.0


def test_bbox_overlap_ratio_full_containment():
    """A box fully inside another has an overlap ratio of 1.0 (ratio is relative to self)."""
    outer = bbox(0, 0, 10, 10)
    inner = bbox(2, 2, 8, 8)
    assert inner.overlap_ratio(outer) == 1.0


def test_bbox_overlap_ratio_partial():
    """Partially overlapping boxes produce a ratio equal to intersection_area / self.area."""
    b1 = bbox(0, 0, 4, 4)  # area = 16
    b2 = bbox(2, 2, 6, 6)
    # intersection is (2,2)→(4,4), area = 4
    assert b1.overlap_ratio(b2) == pytest.approx(4 / 16)


def test_bbox_distance_zero_when_touching():
    """Boxes that share an edge have distance 0."""
    b1 = bbox(0, 0, 5, 5)
    b2 = bbox(5, 0, 10, 5)
    assert b1.distance(b2) == 0


def test_bbox_distance_horizontal_gap():
    """Pure horizontal gap returns only the x component."""
    b1 = bbox(0, 0, 5, 5)
    b2 = bbox(8, 0, 13, 5)
    assert b1.distance(b2) == 3


def test_bbox_distance_vertical_gap():
    """Pure vertical gap returns only the y component."""
    b1 = bbox(0, 0, 5, 5)
    b2 = bbox(0, 7, 5, 12)
    assert b1.distance(b2) == 2


def test_bbox_distance_diagonal():
    """Diagonal separation returns the sum of the x and y gaps (Manhattan distance)."""
    b1 = bbox(0, 0, 5, 5)
    b2 = bbox(8, 9, 13, 14)
    # x-gap = 3, y-gap = 4 → 7
    assert b1.distance(b2) == 7


def test_bbox_distance_vector_intersecting():
    """Intersecting boxes have a distance vector of (0, 0)."""
    b1 = bbox(0, 0, 10, 10)
    b2 = bbox(3, 3, 7, 7)
    assert b1.distance_vector(b2) == (0.0, 0.0)


def test_bbox_intersect_overlapping():
    """intersect() returns the correct overlapping rectangle."""
    b1 = bbox(0, 0, 10, 10)
    b2 = bbox(5, 5, 15, 15)
    result = b1.intersect(b2)
    assert result is not None
    assert result.is_equal(bbox(5, 5, 10, 10))


def test_bbox_intersect_touching_edge():
    """Boxes that share only an edge do not intersect (strict < check)."""
    b1 = bbox(0, 0, 5, 5)
    b2 = bbox(5, 0, 10, 5)
    assert b1.intersect(b2) is None


def test_bbox_intersect_disjoint():
    """Completely separated boxes return None from intersect()."""
    b1 = bbox(0, 0, 5, 5)
    b2 = bbox(10, 10, 20, 20)
    assert b1.intersect(b2) is None


def test_bbox_get_intermediate_box_horizontal():
    """
    Side-by-side boxes produce a corridor spanning the gap between them.

    The corridor's x range covers the gap; its y range is the shared overlap.
    """
    b1 = bbox(0, 0, 5, 10)
    b2 = bbox(8, 2, 13, 8)
    corridor = b1.get_intermediate_box(b2)
    assert corridor is not None
    assert corridor.x0 == 5
    assert corridor.x1 == 8
    assert corridor.y0 == 2
    assert corridor.y1 == 8


def test_bbox_get_intermediate_box_vertical():
    """
    Vertically stacked boxes produce a corridor spanning the gap between them.

    The corridor's y range covers the gap; its x range is the shared overlap.
    """
    b1 = bbox(0, 0, 10, 5)
    b2 = bbox(2, 8, 8, 13)
    corridor = b1.get_intermediate_box(b2)
    assert corridor is not None
    assert corridor.y0 == 5
    assert corridor.y1 == 8
    assert corridor.x0 == 2
    assert corridor.x1 == 8


def test_bbox_get_intermediate_box_diagonal():
    """Diagonally placed boxes have no shared axis, so no corridor exists."""
    b1 = bbox(0, 0, 5, 5)
    b2 = bbox(8, 8, 13, 13)
    assert b1.get_intermediate_box(b2) is None


def test_bbox_get_intermediate_box_intersecting():
    """Overlapping boxes have no gap to fill, so get_intermediate_box returns None."""
    b1 = bbox(0, 0, 10, 10)
    b2 = bbox(5, 5, 15, 15)
    assert b1.get_intermediate_box(b2) is None


# ===========================================================================
# ImageBBox
# ===========================================================================


def test_imagebbox_dpi_from_image():
    """
    DPI is derived from image pixel height relative to bbox point height.

    Formula: dpi = round(72 * image.height / bbox.height).
    A 144 px tall image inside a 72 pt tall box → 144 DPI.
    """
    img = make_image(100, 144)
    ib = ImageBBox(x0=0, y0=0, x1=100, y1=72, image=img)
    assert ib.dpi == 144


def test_imagebbox_dpi_no_image_raises():
    """Accessing dpi without an image and without a virtual DPI raises ValueError."""
    ib = ImageBBox(x0=0, y0=0, x1=10, y1=10, image=None)
    with pytest.raises(ValueError):
        _ = ib.dpi


def test_imagebbox_dpi_virtual():
    """When image is None but _virtual_dpi is set, dpi returns the virtual value."""
    ib = ImageBBox(x0=0, y0=0, x1=10, y1=10, image=None)
    ib._virtual_dpi = 150
    assert ib.dpi == 150


def test_imagebbox_calc_virtual_size():
    """
    calc_virtual_size() computes pixel dimensions from bbox size and virtual DPI.

    Formula: (int(width * dpi / 72), int(height * dpi / 72)).
    A 72×72 pt box at 144 DPI → 144×144 px.
    """
    ib = ImageBBox(x0=0, y0=0, x1=72, y1=72, image=None)
    ib._virtual_dpi = 144
    ib.calc_virtual_size()
    assert ib._virtual_size == (144, 144)


def test_imagebbox_calc_virtual_size_no_dpi_raises():
    """calc_virtual_size() raises ValueError when _virtual_dpi has not been set."""
    ib = ImageBBox(x0=0, y0=0, x1=72, y1=72, image=None)
    with pytest.raises(ValueError):
        ib.calc_virtual_size()


def test_imagebbox_copy_independent():
    """copy() produces an independent object; mutating the original does not affect the copy."""
    img = make_image(50, 50)
    ib = ImageBBox(x0=0, y0=0, x1=10, y1=10, image=img)
    copy = ib.copy()
    ib.x0 = 99
    assert copy.x0 == 0


def test_imagebbox_get_bbox_type():
    """get_bbox() strips the image and returns a plain BBox, not an ImageBBox."""
    img = make_image(50, 50)
    ib = ImageBBox(x0=1, y0=2, x1=3, y1=4, image=img)
    result = ib.get_bbox()
    assert type(result) is BBox


def test_imagebbox_from_bbox():
    """from_bbox() copies all four coordinates from the source BBox."""
    b = bbox(1, 2, 3, 4)
    img = make_image(50, 50)
    ib = ImageBBox.from_bbox(b, img)
    assert ib.x0 == 1 and ib.y0 == 2 and ib.x1 == 3 and ib.y1 == 4


# ===========================================================================
# TextBox
# ===========================================================================


def test_textbox_from_bbox():
    """from_bbox() copies coordinates from the source BBox and attaches the given text."""
    b = bbox(1, 2, 3, 4)
    tb = TextBox.from_bbox(b, text="hello")
    assert tb.x0 == 1 and tb.y0 == 2 and tb.x1 == 3 and tb.y1 == 4
    assert tb.text == "hello"


def test_textbox_get_text_single_box():
    """A single box returns its text unchanged."""
    tb = tbox(0, 0, 10, 10, text="hello")
    result = TextBox.get_text_from_boxes([tb])
    assert result == "hello"


def test_textbox_get_text_same_line():
    """Touching boxes on the same line are joined without any separator."""
    tb1 = tbox(0, 0, 5, 10, text="foo")
    tb2 = tbox(5, 0, 10, 10, text="bar")
    result = TextBox.get_text_from_boxes([tb1, tb2])
    assert result == "foobar"


def test_textbox_get_text_same_line_with_gap():
    """Boxes on the same line separated by a gap wider than epsilon get a space inserted."""
    tb1 = tbox(0, 0, 5, 10, text="foo")
    tb2 = tbox(20, 0, 25, 10, text="bar")
    result = TextBox.get_text_from_boxes([tb1, tb2])
    assert result == "foo bar"


def test_textbox_get_text_new_line():
    """Boxes separated vertically beyond epsilon produce a newline between their texts."""
    tb1 = tbox(0, 0, 10, 10, text="line1")
    tb2 = tbox(0, 50, 10, 60, text="line2")
    result = TextBox.get_text_from_boxes([tb1, tb2])
    assert "\n" in result
    assert "line1" in result and "line2" in result


@pytest.mark.xfail(
    reason=(
        "sort_text_in_reading_order builds a y-keyed dict in insertion order without sorting "
        "the keys at the end. Boxes given in reverse order are therefore not re-sorted. "
    ),
    strict=True,
)
def test_textbox_sort_ltr_td_simple():
    """
    LTR_TD sort should place the top-left box before the bottom-right box.

    Boxes are deliberately passed in reverse order to exercise the sort.
    Currently xfail: the implementation relies on dict insertion order instead
    of sorting the y-keys, so this case is broken.
    """
    bottom_right = tbox(50, 50, 60, 60, text="B")
    top_left = tbox(0, 0, 10, 10, text="A")
    sorted_boxes = TextBox.sort_text_in_reading_order([bottom_right, top_left])
    assert sorted_boxes[0].text == "A"
    assert sorted_boxes[1].text == "B"


def test_textbox_sort_ltr_td_same_row_epsilon():
    """
    Boxes whose y-centers differ by less than epsilon are treated as the same row
    and sorted by x0 within that row.
    """
    left = tbox(0, 10, 10, 20, text="LEFT")
    right = tbox(50, 11, 60, 21, text="RIGHT")  # Δy = 1, within epsilon
    sorted_boxes = TextBox.sort_text_in_reading_order([right, left])
    assert sorted_boxes[0].text == "LEFT"
    assert sorted_boxes[1].text == "RIGHT"


def test_textbox_sort_other_reading_order_raises():
    """sort_text_in_reading_order raises NotImplementedError for any order other than LTR_TD."""
    tb1 = tbox(0, 0, 10, 10, text="A", reading_order=ReadingOrder.RTL_TD)
    tb2 = tbox(20, 0, 30, 10, text="B", reading_order=ReadingOrder.RTL_TD)
    with pytest.raises(NotImplementedError):
        TextBox.sort_text_in_reading_order([tb1, tb2])


def test_textbox_union_text_boxes():
    """
    union_text_boxes() returns a TextBox whose bbox encloses all inputs
    and whose text is the reading-order join of their individual texts.
    """
    tb1 = tbox(0, 0, 10, 10, text="hello")
    tb2 = tbox(20, 0, 30, 10, text="world")
    result = TextBox.union_text_boxes([tb1, tb2])
    assert result.x0 == 0 and result.x1 == 30
    assert "hello" in result.text and "world" in result.text


def test_textbox_to_dict_from_dict_roundtrip():
    """Serialisation via to_dict/from_dict preserves coordinates, text and reading order."""
    tb = tbox(1, 2, 3, 4, text="test")
    restored = TextBox.from_dict(tb.to_dict())
    assert restored.x0 == 1 and restored.y0 == 2
    assert restored.text == "test"
    assert restored.reading_order == ReadingOrder.LTR_TD


def test_textbox_union_raises():
    """TextBox.union() raises NotImplementedError (not yet implemented)."""
    tb1 = tbox(0, 0, 5, 5, text="A")
    tb2 = tbox(10, 10, 15, 15, text="B")
    with pytest.raises(NotImplementedError):
        tb1.union(tb2)


# ===========================================================================
# join_in_reading_order
# ===========================================================================


def test_join_single_box():
    """A single input box is returned as a one-element list, text unchanged."""
    tb = tbox(0, 0, 10, 10, text="only")
    result = join_in_reading_order([tb])
    assert len(result) == 1
    assert result[0].text == "only"


def test_join_close_boxes_merged():
    """Boxes closer than cut_off_distance are merged into a single TextBox."""
    tb1 = tbox(0, 0, 10, 10, text="hello")
    tb2 = tbox(0, 15, 10, 25, text="world")
    result = join_in_reading_order([tb1, tb2], cut_off_distance=100)
    assert len(result) == 1
    assert "hello" in result[0].text and "world" in result[0].text


def test_join_far_boxes_separate():
    """Boxes further apart than cut_off_distance remain as separate TextBox items."""
    tb1 = tbox(0, 0, 10, 10, text="A")
    tb2 = tbox(0, 500, 10, 510, text="B")
    result = join_in_reading_order([tb1, tb2], cut_off_distance=50)
    assert len(result) == 2


def test_join_multi_column():
    """
    Boxes in two widely separated columns produce two independent groups,
    even when their y-ranges overlap.
    """
    col1_row1 = tbox(0, 0, 50, 10, text="C1R1")
    col1_row2 = tbox(0, 15, 50, 25, text="C1R2")
    col2_row1 = tbox(500, 0, 550, 10, text="C2R1")
    col2_row2 = tbox(500, 15, 550, 25, text="C2R2")
    result = join_in_reading_order(
        [col1_row1, col1_row2, col2_row1, col2_row2], cut_off_distance=100
    )
    assert len(result) == 2


def test_join_invalid_reading_order_raises():
    """A box whose reading_order is not a ReadingOrder enum member causes ValueError."""
    tb = tbox(0, 0, 10, 10, text="x")
    tb.reading_order = "not_a_reading_order"
    with pytest.raises(ValueError):
        join_in_reading_order([tb])


def test_join_non_ltr_td_raises():
    """Any reading order other than LTR_TD raises NotImplementedError."""
    tb = tbox(0, 0, 10, 10, text="x", reading_order=ReadingOrder.RTL_TD)
    with pytest.raises(NotImplementedError):
        join_in_reading_order([tb])
