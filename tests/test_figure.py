"""Tests for figure.py — Figure and OutputFigure."""

from __future__ import annotations

from PIL import Image

from pmo_parser.bounding_boxes import BBox, ReadingOrder, TextBox
from pmo_parser.figure import Figure, FigureType, OutputFigure

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bbox(x0, y0, x1, y1) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def tbox(x0, y0, x1, y1, text) -> TextBox:
    return TextBox(
        x0=x0, y0=y0, x1=x1, y1=y1, text=text, reading_order=ReadingOrder.LTR_TD
    )


def simple_figure(captions=None, name=None):
    return Figure(
        page=0, figure_bbox=bbox(0, 0, 100, 100), captions=captions, name=name
    )


def simple_output_figure(caption_text="Figure 1. Caption.", image=None):
    cap = tbox(0, 110, 100, 130, caption_text)
    return OutputFigure(
        page=0,
        figure_bbox=bbox(0, 0, 100, 100),
        captions=[cap],
        image=image,
    )


# ===========================================================================
# Figure
# ===========================================================================


def test_figure_has_caption_true():
    """has_caption() returns True when captions is a non-empty list."""
    fig = simple_figure(captions=[tbox(0, 110, 100, 130, "Figure 1.")])
    assert fig.has_caption() is True


def test_figure_has_caption_false():
    """has_caption() returns False when captions is None."""
    fig = simple_figure(captions=None)
    assert fig.has_caption() is False


def test_figure_caption_bbox_single():
    """caption_bbox returns the bbox of the single caption TextBox."""
    cap = tbox(5, 110, 95, 130, "Figure 1.")
    fig = simple_figure(captions=[cap])
    result = fig.caption_bbox
    assert result is not None
    assert result.is_equal(BBox.union_boxes([cap]))


def test_figure_caption_bbox_multiple():
    """caption_bbox returns the union of all caption bboxes."""
    cap1 = tbox(5, 110, 95, 125, "Figure 1.")
    cap2 = tbox(5, 126, 95, 140, "continued text")
    fig = simple_figure(captions=[cap1, cap2])
    result = fig.caption_bbox
    assert result is not None
    expected = BBox.union_boxes([cap1, cap2])
    assert result.is_equal(expected)


def test_figure_caption_bbox_none():
    """caption_bbox returns None when captions is None."""
    fig = simple_figure(captions=None)
    assert fig.caption_bbox is None


def test_figure_contains_caption_present():
    """contains_caption() returns True for a caption that is in the list."""
    cap = tbox(0, 110, 100, 130, "Figure 1.")
    fig = simple_figure(captions=[cap])
    assert fig.contains_caption(cap) is True


def test_figure_contains_caption_absent():
    """contains_caption() returns False for a TextBox not in the captions list."""
    cap = tbox(0, 110, 100, 130, "Figure 1.")
    other = tbox(0, 200, 100, 220, "Unrelated text")
    fig = simple_figure(captions=[cap])
    assert fig.contains_caption(other) is False


def test_figure_contains_caption_no_captions():
    """contains_caption() returns False when captions is None."""
    other = tbox(0, 110, 100, 130, "Figure 1.")
    fig = simple_figure(captions=None)
    assert fig.contains_caption(other) is False


def test_figure_to_dict_keys():
    """to_dict() result contains the five expected keys."""
    cap = tbox(0, 110, 100, 130, "Figure 1.")
    fig = simple_figure(captions=[cap], name="fig_1")
    d = fig.to_dict()
    assert {"caption", "type", "figure_bbox", "page", "name"} <= set(d.keys())


def test_figure_copy_independence():
    """copy() is a deep copy: mutating the copy does not affect the original."""
    cap = tbox(0, 110, 100, 130, "Figure 1.")
    fig = simple_figure(captions=[cap])
    copy = fig.copy()
    copy.page = 99
    copy.figure_bbox.x0 = 99
    copy.captions[0].x0 = 99
    assert fig.page == 0
    assert fig.figure_bbox.x0 == 0
    assert fig.captions[0].x0 == 0


# ===========================================================================
# OutputFigure
# ===========================================================================


def test_output_figure_dpi_from_image():
    """_dpi is set from image pixel height and bbox point height on construction."""
    img = Image.new("RGB", (100, 200))  # 200 px tall
    fig = OutputFigure(page=0, figure_bbox=bbox(0, 0, 100, 100), image=img)
    # dpi = round(72 * 200 / 100) = 144
    assert fig._dpi == 144


def test_output_figure_estimated_figure_ids_match():
    """A caption starting with 'Figure 3' extracts [3]."""
    cap = tbox(0, 0, 100, 20, "Figure 3. A caption")
    fig = OutputFigure(page=0, figure_bbox=bbox(0, 0, 10, 10), captions=[cap])
    assert fig.estimated_figure_ids == [3]


def test_output_figure_estimated_figure_ids_no_match():
    """A caption with no figure number returns an empty list."""
    cap = tbox(0, 0, 100, 20, "Some unrelated text without a number")
    fig = OutputFigure(page=0, figure_bbox=bbox(0, 0, 10, 10), captions=[cap])
    assert fig.estimated_figure_ids == []


def test_output_figure_estimated_figure_ids_deduped():
    """The same figure ID appearing in multiple captions is returned only once."""
    cap1 = tbox(0, 0, 100, 20, "Figure 2. First line")
    cap2 = tbox(0, 22, 100, 42, "Figure 2. continued")
    fig = OutputFigure(page=0, figure_bbox=bbox(0, 0, 10, 10), captions=[cap1, cap2])
    ids = fig.estimated_figure_ids
    assert ids.count(2) == 1
    assert len(ids) == 1


def test_output_figure_serialize_roundtrip():
    """serialize() dict contains all expected keys with correct values."""
    cap = tbox(0, 110, 100, 130, "Figure 1. A caption.")
    fig = OutputFigure(page=1, figure_bbox=bbox(0, 0, 100, 100), captions=[cap])
    d, img = fig.serialize()
    for key in (
        "caption",
        "figure_bbox",
        "page",
        "type",
        "name",
        "base_figure_bbox",
        "used_screenshot",
        "dpi",
    ):
        assert key in d
    assert d["page"] == 1
    assert img is None  # no image supplied


def test_output_figure_from_base_figure():
    """from_base_figure() copies page, type, figure_bbox, and captions from the base Figure."""
    cap = tbox(0, 110, 100, 130, "Figure 1.")
    base = Figure(
        page=2,
        figure_bbox=bbox(10, 20, 110, 120),
        captions=[cap],
        figure_type=FigureType.FIGURE,
    )
    out = OutputFigure.from_base_figure(base)
    assert out.page == 2
    assert out.figure_bbox.is_equal(base.figure_bbox)
    assert out.captions is base.captions
    assert out.type == FigureType.FIGURE
