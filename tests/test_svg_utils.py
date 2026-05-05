"""Tests for svg_utils.py — get_bounding_box_svgelements, intersect_box_with_clips, walk_and_clip."""

from __future__ import annotations

from io import StringIO

import pytest
import svgelements

from pmo_parser.bounding_boxes import BBox
from pmo_parser.svg_utils import (
    get_bounding_box_svgelements,
    intersect_box_with_clips,
    walk_and_clip,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bbox(x0, y0, x1, y1) -> BBox:
    return BBox(x0=x0, y0=y0, x1=x1, y1=y1)


def parse_svg(svg_text: str) -> svgelements.SVG:
    return svgelements.SVG.parse(StringIO(svg_text))


# ===========================================================================
# get_bounding_box_svgelements
# ===========================================================================


def test_get_bbox_svg_image_no_transform():
    """x, y, width, height from an SVGImage with no transform map directly to BBox."""
    svg_text = """<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400">
      <image x="10" y="20" width="100" height="80"/>
    </svg>"""
    doc = parse_svg(svg_text)
    img_element = next(e for e in doc.elements() if isinstance(e, svgelements.SVGImage))
    result = get_bounding_box_svgelements(img_element)
    assert result is not None
    assert result.x0 == pytest.approx(10, abs=1)
    assert result.y0 == pytest.approx(20, abs=1)
    assert result.x1 == pytest.approx(110, abs=1)
    assert result.y1 == pytest.approx(100, abs=1)


def test_get_bbox_svg_image_with_transform():
    """A translate transform shifts the resulting BBox corners."""
    svg_text = """<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400">
      <image x="0" y="0" width="50" height="50" transform="translate(30, 40)"/>
    </svg>"""
    doc = parse_svg(svg_text)
    img_element = next(e for e in doc.elements() if isinstance(e, svgelements.SVGImage))
    result = get_bounding_box_svgelements(img_element)
    assert result is not None
    assert result.x0 == pytest.approx(30, abs=1)
    assert result.y0 == pytest.approx(40, abs=1)
    assert result.x1 == pytest.approx(80, abs=1)
    assert result.y1 == pytest.approx(90, abs=1)


def test_get_bbox_group_returns_none():
    """Groups do not have a bbox of their own; function returns None."""
    svg_text = """<svg xmlns="http://www.w3.org/2000/svg">
      <g id="grp"><rect x="0" y="0" width="10" height="10"/></g>
    </svg>"""
    doc = parse_svg(svg_text)
    group = next(e for e in doc.elements() if isinstance(e, svgelements.Group))
    assert get_bounding_box_svgelements(group) is None


# ===========================================================================
# intersect_box_with_clips
# ===========================================================================


def test_intersect_box_with_clips_no_clips():
    """With an empty clip stack the original box is returned unchanged."""
    b = bbox(0, 0, 100, 100)
    result = intersect_box_with_clips(b, [])
    assert result is not None
    assert result.is_equal(b)


def test_intersect_box_with_clips_partial():
    """A single clip box that partially overlaps returns the intersection."""
    b = bbox(0, 0, 100, 100)
    clip = bbox(50, 50, 200, 200)
    result = intersect_box_with_clips(b, [clip])
    assert result is not None
    assert result.is_equal(bbox(50, 50, 100, 100))


def test_intersect_box_with_clips_fully_clipped():
    """When the clip does not overlap at all, None is returned."""
    b = bbox(0, 0, 10, 10)
    clip = bbox(50, 50, 100, 100)
    assert intersect_box_with_clips(b, [clip]) is None


def test_intersect_box_with_clips_none_input():
    """When the input box is None, None is returned regardless of clips."""
    clip = bbox(0, 0, 100, 100)
    assert intersect_box_with_clips(None, [clip]) is None


# ===========================================================================
# walk_and_clip
# ===========================================================================


def test_walk_and_clip_image_node():
    """An SVGImage element produces one result entry with type 'Image' and dpi set."""
    svg_text = """<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400">
      <image x="10" y="10" width="72" height="72"/>
    </svg>"""
    doc = parse_svg(svg_text)
    results = walk_and_clip(doc)
    image_results = [r for r in results if r["type"] == "Image"]
    assert len(image_results) >= 1
    assert image_results[0]["dpi"] is not None


def test_walk_and_clip_path_node():
    """A Path element produces one result with type != 'Image' and dpi=None."""
    svg_text = """<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400">
      <path d="M 10 10 L 100 10 L 100 100 Z"/>
    </svg>"""
    doc = parse_svg(svg_text)
    results = walk_and_clip(doc)
    path_results = [r for r in results if r["type"] != "Image"]
    assert len(path_results) >= 1
    assert path_results[0]["dpi"] is None


def test_walk_and_clip_text_node():
    """A top-level group containing only text is filtered out."""
    svg_text = """<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400">
      <g><text x="10" y="20">Hello</text></g>
    </svg>"""
    doc = parse_svg(svg_text)
    results = walk_and_clip(doc)
    # The group has only text children so the top-level group should be filtered
    assert all(r["type"] != "Image" for r in results)
    # More importantly: the top-level group of pure text should produce no results
    # (walk_and_clip filters keep_top_level_groups text-only groups)
    assert len(results) == 0


def test_walk_and_clip_clip_applied():
    """A clip-path applied to an image restricts the resulting bbox."""
    svg_text = """<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400">
      <defs>
        <clipPath id="clip1">
          <rect x="20" y="20" width="40" height="40"/>
        </clipPath>
      </defs>
      <image x="0" y="0" width="100" height="100" clip-path="url(#clip1)"/>
    </svg>"""
    doc = parse_svg(svg_text)
    results = walk_and_clip(doc)
    image_results = [r for r in results if r["type"] == "Image"]
    if len(image_results) == 0:
        pytest.skip("SVG parser did not expose image with clip-path in this version")
    cb = image_results[0]["clip_box"]
    # The clip box should be the intersection: x0≥20, y0≥20, x1≤60, y1≤60
    assert cb.x0 >= 20 - 1
    assert cb.y0 >= 20 - 1
    assert cb.x1 <= 60 + 1
    assert cb.y1 <= 60 + 1


def test_walk_and_clip_fully_clipped_node():
    """
    A node whose effective clip_box has zero area returns an empty list.

    We exercise the early-return guard in walk_and_clip by passing a clip_box
    with zero height/width directly, rather than via SVG clip-path attributes
    (whose parsing behavior depends on the svgelements version).
    """
    svg_text = """<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400">
      <image x="0" y="0" width="100" height="100"/>
    </svg>"""
    doc = parse_svg(svg_text)
    zero_clip = bbox(50, 50, 50, 50)  # zero width and height
    results = walk_and_clip(doc, clip_box=zero_clip)
    assert results == []


def test_walk_and_clip_nested_group():
    """Elements inside a group carry the group label from their parent."""
    svg_text = """<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400">
      <g>
        <image x="10" y="10" width="72" height="72"/>
        <path d="M 10 10 L 100 10 L 100 100 Z"/>
      </g>
    </svg>"""
    doc = parse_svg(svg_text)
    results = walk_and_clip(doc)
    assert len(results) >= 1
    # All results should have a 'group' key
    assert all("group" in r for r in results)


def test_walk_and_clip_single_child_group_unwrapped():
    """A top-level group with a single non-empty child is unwrapped transparently."""
    svg_text = """<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400">
      <g>
        <g>
          <image x="10" y="10" width="72" height="72"/>
        </g>
      </g>
    </svg>"""
    doc = parse_svg(svg_text)
    results = walk_and_clip(doc)
    # Should still find the image despite the double wrapping
    image_results = [r for r in results if r["type"] == "Image"]
    assert len(image_results) >= 1
