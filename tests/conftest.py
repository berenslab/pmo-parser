"""Shared fixtures and helpers for all test modules."""

from __future__ import annotations

import io

import pymupdf
import pytest
from PIL import Image

from pmo_parser.bounding_boxes import BBox, ImageBBox, ReadingOrder, TextBox
from pmo_parser.page.base_page import ImageCluster, convert_to_string

# ---------------------------------------------------------------------------
# Minimal PDF fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def minimal_pdf_path(tmp_path_factory):
    """One-page PDF with a gray rectangle and a caption line below it."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.draw_rect(
        pymupdf.Rect(100, 100, 300, 300),
        color=(0.5, 0.5, 0.5),
        fill=(0.8, 0.8, 0.8),
        width=1,
    )
    page.insert_text((100, 320), "Figure 1. A test rectangle.", fontsize=10)
    tmp = tmp_path_factory.mktemp("fixtures")
    path = tmp / "sample.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture(scope="session")
def minimal_pdf_bytes(minimal_pdf_path):
    """Same PDF as minimal_pdf_path but returned as a fresh BytesIO."""
    with open(minimal_pdf_path, "rb") as f:
        return io.BytesIO(f.read())


@pytest.fixture
def fresh_pdf_bytes(minimal_pdf_path):
    """Fresh BytesIO each time (for tests that consume/seek the stream)."""
    with open(minimal_pdf_path, "rb") as f:
        return io.BytesIO(f.read())


# ---------------------------------------------------------------------------
# FakePage — synthetic Page stub for pipeline unit tests
# ---------------------------------------------------------------------------


class FakePage:
    """
    Minimal Page-compatible object for pipeline unit tests.

    Constructs the page attributes directly without any PDF or pymupdf
    involvement, making the pipeline functions fully testable in isolation.
    """

    def __init__(
        self,
        page_figures: list[ImageBBox],
        page_texts: list[list[TextBox]],
        figure_clusters: list[ImageCluster] | None = None,
        remaining_paths: list[BBox] | None = None,
        page_width: float = 595.0,
        page_height: float = 842.0,
        page_num: int = 0,
        mean_letter_width: float = 8.0,
        mean_letter_height: float = 12.0,
    ):
        self.page_figures = page_figures
        self.page_texts = page_texts
        self.figure_clusters = figure_clusters if figure_clusters is not None else []
        self.remaining_paths = remaining_paths if remaining_paths is not None else []
        self.page_width = page_width
        self.page_height = page_height
        self.page_num = page_num
        self.mean_letter_width = mean_letter_width
        self.mean_letter_height = mean_letter_height

    def get_string_texts(self) -> list[str]:
        """Mirror of Page.get_string_texts using the real convert_to_string."""
        return [
            convert_to_string(texts, self.mean_letter_width, self.mean_letter_height)
            for texts in self.page_texts
        ]

    def get_cluster_index(self, image_id: int) -> int:
        """Mirror of Page.get_cluster_index."""
        for i, cluster in enumerate(self.figure_clusters):
            if image_id in cluster.image_ids:
                return i
        return -1


# ---------------------------------------------------------------------------
# Small factory helpers
# ---------------------------------------------------------------------------


def make_image_bbox(
    x0: float, y0: float, x1: float, y1: float, dpi: int = 150
) -> ImageBBox:
    """ImageBBox backed by a real (tiny) PIL image so that .dpi works."""
    h = max(1, int((y1 - y0) * dpi / 72))
    w = max(1, int((x1 - x0) * dpi / 72))
    img = Image.new("RGB", (w, h), color=(200, 200, 200))
    return ImageBBox(x0=x0, y0=y0, x1=x1, y1=y1, image=img)


def make_text_box(x0: float, y0: float, x1: float, y1: float, text: str) -> TextBox:
    """TextBox with LTR_TD reading order."""
    return TextBox(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
        reading_order=ReadingOrder.LTR_TD,
    )
