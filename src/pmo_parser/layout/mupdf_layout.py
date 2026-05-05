"""
MuPDF-based layout backend.

Uses ``pymupdf`` directly (no model required). Pages can optionally be parsed
in parallel via :class:`multiprocessing.Pool`.
"""

from collections.abc import Sequence
from multiprocessing import Pool

from pmo_parser.page import Page
from pmo_parser.page.mupdf_page import MuPDFPage
from pmo_parser.util import DocumentWrapper

from .base_layout import PDFLayout
from .layout_registry import LAYOUT_REGISTRY


def _loading_worker(args):
    """
    Build a single :class:`MuPDFPage` in a worker process.

    Args:
        args (tuple): ``(pdf_document_source, page_num,
            always_create_screenshots)`` triple. The PDF source must be
            re-openable in the worker, hence the explicit re-open inside.

    Returns:
        MuPDFPage: Parsed page at ``page_num``.

    """
    pdf_document_source, page_num, always_create_screenshots = args

    with DocumentWrapper(pdf_document_source) as pdf_document:
        return MuPDFPage(
            pdf_document,
            page_num,
            always_create_screenshots=always_create_screenshots,
        )


@LAYOUT_REGISTRY.register()
class MuPDFLayout(PDFLayout):
    """Layout backend that uses ``pymupdf`` to parse each page directly."""

    def load(
        self,
        pdf_document: DocumentWrapper,
        always_create_screenshots: bool = False,
        num_processes: int = 1,
    ) -> Sequence[Page]:
        """
        Parse ``pdf_document`` into :class:`MuPDFPage` objects.

        When ``num_processes`` is greater than one and the document has more
        than one page, pages are parsed in parallel via
        :class:`multiprocessing.Pool`; otherwise the loop runs in the calling
        process.

        Args:
            pdf_document (DocumentWrapper): Already opened PDF document.
            always_create_screenshots (bool, optional): When ``True``, page
                screenshots are rendered eagerly. Defaults to ``False``.
            num_processes (int, optional): Maximum number of worker processes
                used for parsing. Defaults to ``1``.

        Returns:
            list[Page]: One :class:`MuPDFPage` per page of ``pdf_document``.

        """
        if num_processes <= 1 or pdf_document.page_count < 2:
            mupdf_pages = []
            for page_num in range(pdf_document.page_count):
                res = MuPDFPage(
                    pdf_document,
                    page_num,
                    always_create_screenshots=always_create_screenshots,
                )
                mupdf_pages.append(res)
        else:
            num_processes = min(num_processes, pdf_document.page_count)

            with Pool(processes=num_processes) as pool:
                mupdf_pages = pool.map(
                    _loading_worker,
                    [
                        (pdf_document._source, i, always_create_screenshots)
                        for i in range(pdf_document.page_count)
                    ],
                )
        return mupdf_pages
