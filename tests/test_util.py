"""Tests for util.py — DocumentWrapper."""

from __future__ import annotations

import io

import pytest

from pmo_parser.util import DocumentWrapper

# ===========================================================================
# DocumentWrapper
# ===========================================================================


def test_document_wrapper_from_path(minimal_pdf_path):
    """Opening from a file path produces a document with at least one page."""
    with DocumentWrapper(minimal_pdf_path) as doc:
        assert doc.page_count > 0


def test_document_wrapper_from_bytes_io(fresh_pdf_bytes):
    """Opening from a BytesIO stream produces a document with at least one page."""
    with DocumentWrapper(fresh_pdf_bytes) as doc:
        assert doc.page_count > 0


@pytest.mark.xfail(
    reason=(
        "DocumentWrapper passes the original pdf_path to pymupdf "
        "instead of the loading_function result. When the original path is invalid "
        "(e.g. a URL or synthetic identifier) and only the BytesIO returned by the "
        "loading_function is usable, the document fails to open. "
        "Fix: use `source` (= loading_function result) when calling pymupdf.Document."
    ),
    strict=True,
)
def test_document_wrapper_loading_function(minimal_pdf_path):
    """
    loading_function result must be used as the PDF source, not the original path.

    The test passes a fake/non-existent identifier as pdf_path, and a
    loading_function that ignores it and returns real PDF bytes. After the fix,
    DocumentWrapper should open from those bytes. With the current bug it tries to
    open the fake path directly → FileNotFoundError → xfail demonstrates the bug.
    """
    with open(minimal_pdf_path, "rb") as f:
        real_pdf_bytes = f.read()

    def fetch_from_virtual_source(_fake_path: str) -> io.BytesIO:
        # Simulates fetching a PDF from a remote/virtual source.
        return io.BytesIO(real_pdf_bytes)

    fake_identifier = "/does/not/exist/virtual.pdf"
    with DocumentWrapper(
        fake_identifier,
        loading_function=fetch_from_virtual_source,  # pyright: ignore[reportArgumentType]
    ) as doc:
        assert doc.page_count > 0
