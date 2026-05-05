"""
Abstract base class for PDF layout backends.

A layout backend turns a PDF document into a list of :class:`Page` objects;
concrete implementations live in :mod:`pmo_parser.layout.mupdf_layout` and
:mod:`pmo_parser.layout.dl_layout`.
"""

import io
from abc import ABC, abstractmethod
from collections.abc import Sequence

from pmo_parser.page import Page
from pmo_parser.util import DocumentWrapper


class PDFLayout(ABC):
    """
    Backend that turns a PDF into a list of :class:`Page` objects.

    Attributes:
        pages (list[Page]): Pages of the parsed document, populated by
            :meth:`load`.

    """

    def __init__(
        self,
        pdf_path: str | io.BytesIO,
        always_create_screenshots: bool = False,
        num_processes: int = 1,
    ):
        """
        Open ``pdf_path`` and populate :attr:`pages` via :meth:`load`.

        Args:
            pdf_path (str | io.BytesIO): File path or in-memory stream of the
                PDF to open.
            always_create_screenshots (bool, optional): When ``True``, page
                screenshots are rendered even for pages that do not need
                them. Defaults to ``False``.
            num_processes (int, optional): Maximum number of worker processes
                used by :meth:`load`. Defaults to ``1``.

        """
        with DocumentWrapper(pdf_path) as pdf_document:
            self.pages = self.load(
                pdf_document,
                always_create_screenshots=always_create_screenshots,
                num_processes=num_processes,
            )

    @abstractmethod
    def load(
        self,
        pdf_document: DocumentWrapper,
        always_create_screenshots: bool = False,
        num_processes: int = 1,
    ) -> Sequence[Page]:
        """
        Parse ``pdf_document`` into a list of :class:`Page` objects.

        Args:
            pdf_document (DocumentWrapper): Already opened PDF document.
            always_create_screenshots (bool, optional): Forwarded from
                :meth:`__init__`. Defaults to ``False``.
            num_processes (int, optional): Maximum number of worker processes
                used during parsing. Defaults to ``1``.

        Returns:
            list[Page]: One page per page of ``pdf_document``.

        """
