"""Tests for renderer.py — render_page and render_figures."""

from __future__ import annotations

import pymupdf
import pytest
from PIL import Image

from pmo_parser.bounding_boxes import BBox
from pmo_parser.figure import OutputFigure
from pmo_parser.renderer import render_figures, render_page

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def out_fig(page, x0, y0, x1, y1, image=None, dpi=150):
    """Create a minimal OutputFigure for render_figures tests."""
    fig = OutputFigure(
        page=page, figure_bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1), image=image
    )
    fig._dpi = dpi
    return fig


# ===========================================================================
# render_page
# ===========================================================================


def test_render_page_returns_image(minimal_pdf_path):
    """render_page returns a PIL Image object."""
    img = render_page(minimal_pdf_path, 0)
    assert isinstance(img, Image.Image)


def test_render_page_size_matches_dpi(minimal_pdf_path):
    """The rendered image height is approximately page_height * dpi / 72."""
    dpi = 72
    img = render_page(minimal_pdf_path, 0, dpi=dpi)
    with pymupdf.open(minimal_pdf_path) as doc:
        page_height = doc[0].rect.height
    # Allow a small rounding tolerance
    assert abs(img.height - page_height) < 2


def test_render_page_bbox_crop(minimal_pdf_path):
    """When a BBox is supplied, the returned image matches the requested region size."""
    dpi = 72
    crop = BBox(x0=50, y0=50, x1=150, y1=150)  # 100×100 pt region
    img = render_page(minimal_pdf_path, 0, dpi=dpi, bbox=crop)
    # At 72 dpi: 1 pt ≈ 1 px; allow ±2 px tolerance for rounding
    assert abs(img.width - 100) <= 2
    assert abs(img.height - 100) <= 2


def test_render_page_from_path(minimal_pdf_path):
    """render_page accepts a file path and returns a valid image."""
    img = render_page(minimal_pdf_path, 0, dpi=72)
    assert img.width > 0 and img.height > 0


def test_render_page_from_bytes_io(fresh_pdf_bytes):
    """render_page accepts a BytesIO stream and returns a valid image."""
    img = render_page(fresh_pdf_bytes, 0, dpi=72)
    assert img.width > 0 and img.height > 0


@pytest.mark.xfail(
    reason=(
        "render_page calls pdf_document.close() explicitly but "
        "skips it when an exception is raised. After the fix, a try/finally or "
        "context manager should guarantee close() is called regardless."
    ),
    strict=True,
)
def test_render_page_document_closed_on_exception(minimal_pdf_path):
    """
    close() must be called on the document even when rendering raises.

    We patch DocumentWrapper so we can spy on close(). Then we trigger an
    exception by requesting a non-existent page. With the current bug, close()
    is never called. After the fix it must be called even in the error path.
    """
    from unittest.mock import patch  # noqa: PLC0415

    import pmo_parser.renderer as renderer_module  # noqa: PLC0415

    close_calls = []

    real_dw = renderer_module.DocumentWrapper

    class SpyWrapper(real_dw):
        def close(self):
            close_calls.append(True)
            super().close()

    with patch.object(renderer_module, "DocumentWrapper", SpyWrapper):
        with pytest.raises(Exception):
            render_page(minimal_pdf_path, 9999, dpi=72)

    assert len(close_calls) > 0, "close() was never called after the exception"


# ===========================================================================
# render_figures
# ===========================================================================


def test_render_figures_skips_existing(minimal_pdf_path):
    """A figure that already has an image is not re-rendered."""
    sentinel = Image.new("RGB", (10, 10), color=(255, 0, 0))
    fig = out_fig(0, 100, 100, 300, 300, image=sentinel)
    render_figures(minimal_pdf_path, [fig])
    # The sentinel image must be unchanged
    assert fig.image is sentinel


def test_render_figures_fills_missing(minimal_pdf_path):
    """A figure with image=None gets a PIL Image assigned after render_figures."""
    fig = out_fig(0, 100, 100, 300, 300, image=None, dpi=72)
    render_figures(minimal_pdf_path, [fig])
    assert isinstance(fig.image, Image.Image)
    assert fig.image.width > 0
