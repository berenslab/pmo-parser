"""
Render PDF pages and figure crops to PIL images.

Provides :func:`render_page` for a single page (optionally cropped to a
:class:`BBox`) and :func:`render_figures` for batch-rendering all
:class:`OutputFigure` objects that do not yet have an image attached.
"""

import io

import pymupdf
from PIL import Image

from pmo_parser.bounding_boxes import BBox
from pmo_parser.figure import OutputFigure
from pmo_parser.util import DocumentWrapper


def render_page(
    pdf_file: str | pymupdf.Document | io.BytesIO,
    page_number,
    dpi=300,
    bbox: BBox | None = None,
) -> Image.Image:
    """
    Render a single PDF page to a PIL image.

    When ``pdf_file`` is a path or stream, a temporary :class:`DocumentWrapper`
    is opened and closed for the call; an already opened document is left
    open and used directly.

    Args:
        pdf_file (str | pymupdf.Document | io.BytesIO): PDF source. Either a
            path, an in-memory stream, or an already opened MuPDF document.
        page_number (int): Zero-based page index to render.
        dpi (int, optional): DPI used for rendering. Defaults to ``300``.
        bbox (BBox | None, optional): Crop rectangle in PDF coordinates. When
            ``None``, the full page is returned. Defaults to ``None``.

    Returns:
        PIL.Image.Image: Rendered page (cropped to ``bbox`` when supplied).

    """
    pdf_document = (
        pdf_file
        if isinstance(pdf_file, pymupdf.Document)
        else DocumentWrapper(pdf_file)
    )

    page = pdf_document[page_number]
    pix = page.get_pixmap(dpi=dpi)

    png_ratio = float(pix.height) / page.rect.height

    if not isinstance(pdf_file, pymupdf.Document):
        pdf_document.close()

    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    if bbox is not None:
        image = image.crop(
            (
                int(bbox.x0 * png_ratio),
                int(bbox.y0 * png_ratio),
                int(bbox.x1 * png_ratio),
                int(bbox.y1 * png_ratio),
            )
        )
    return image


def render_figures(
    pdf_file: str | pymupdf.Document | io.BytesIO, figure_bboxes: list[OutputFigure]
):
    """
    Render and attach an image to every figure that does not yet have one.

    Figures whose ``image`` attribute is already set are left untouched.
    The DPI used for each figure is taken from its ``_dpi`` attribute.

    Args:
        pdf_file (str | pymupdf.Document | io.BytesIO): PDF source. Either a
            path, an in-memory stream, or an already opened MuPDF document.
        figure_bboxes (list[OutputFigure]): Figures to render. The list is
            mutated in place; each rendered figure has ``image`` populated.

    """
    pdf_document = (
        pdf_file
        if isinstance(pdf_file, pymupdf.Document)
        else DocumentWrapper(pdf_file)
    )

    for figure in figure_bboxes:
        if figure.image is not None:
            continue
        page = pdf_document[figure.page]
        figure_dpi = figure._dpi
        pix = page.get_pixmap(dpi=figure_dpi)  # pyright: ignore[reportAttributeAccessIssue]

        png_ratio = float(pix.height) / page.rect.height

        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        figure.image = image.crop(
            (
                int(figure.figure_bbox.x0 * png_ratio),
                int(figure.figure_bbox.y0 * png_ratio),
                int(figure.figure_bbox.x1 * png_ratio),
                int(figure.figure_bbox.y1 * png_ratio),
            )
        )

    if not isinstance(pdf_file, pymupdf.Document):
        pdf_document.close()
