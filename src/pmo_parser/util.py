"""
Small utilities used across the parser.

Provides :class:`DocumentWrapper` (a unified wrapper around
:class:`pymupdf.Document` for both file paths and in-memory streams) and a
handful of geometry/list helpers.
"""

import io
from typing import Callable

import pymupdf


class DocumentWrapper(pymupdf.Document):
    """
    Wraps :class:`pymupdf.Document` for a unified file-or-stream interface.

    Attributes:
        _source (str | io.BytesIO): The original source the document was
            opened from. Kept so that the document can be re-opened in a
            worker process without forcing the caller to remember it.

    """

    def __init__(
        self,
        pdf_path: str | io.BytesIO,
        loading_function: Callable[[str | io.BytesIO], str | io.BytesIO] | None = None,
    ):
        """
        Open the PDF at ``pdf_path``.

        Args:
            pdf_path (str | io.BytesIO): File path or in-memory stream of the
                PDF to open.
            loading_function (Callable | None, optional): Optional hook
                applied to ``pdf_path`` before the document is opened (for
                example to fetch the file from a remote location). Defaults
                to ``None``.

        """
        self._source = (
            pdf_path if loading_function is None else loading_function(pdf_path)
        )
        if isinstance(pdf_path, io.BytesIO):
            super().__init__(stream=pdf_path, filetype="pdf")
        else:
            super().__init__(pdf_path)
