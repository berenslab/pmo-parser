"""
Deep-learning-based layout backend.

Uses :mod:`layoutparser` to detect figure regions; falls back to a stub
class that raises a clear :class:`ImportError` when ``layoutparser`` is not
installed.
"""

import io
from collections.abc import Sequence

from pmo_parser.page import Page
from pmo_parser.util import DocumentWrapper

from .base_layout import PDFLayout
from .layout_registry import LAYOUT_REGISTRY

try:
    from layoutparser.models import AutoLayoutModel  # pyright: ignore

    from pmo_parser.page.dl_page import DLPage

    @LAYOUT_REGISTRY.register()
    class PDFLayoutDL(PDFLayout):
        """Layout backend driven by a layoutparser model on each page."""

        def load(
            self,
            pdf_document: DocumentWrapper,
            always_create_screenshots: bool = False,
            num_processes: int = 1,  # noqa: ARG002
        ) -> Sequence[Page]:
            """
            Load the PDF structure using :class:`DLPage` for each page.

            Args:
                pdf_document (DocumentWrapper): Already opened PDF document.
                always_create_screenshots (bool, optional): When ``True``,
                    page screenshots are rendered eagerly. Defaults to
                    ``False``.
                num_processes (int, optional): Currently unused. Defaults to
                    ``1``.

            Returns:
                list[Page]: One :class:`DLPage` per page of ``pdf_document``.

            """
            model = AutoLayoutModel(config_path="lp://efficientdet/PubLayNet")
            return [
                DLPage(
                    pdf_document,
                    page_num,
                    model=model,
                    always_create_screenshots=always_create_screenshots,
                )
                for page_num in range(pdf_document.page_count)
            ]
except ImportError:
    # layoutparser is not installed
    # Add dummy class to handle the case

    @LAYOUT_REGISTRY.register(name="PDFLayoutDL")
    class DummyPDFLayoutDL(PDFLayout):
        """Stub raised when ``layoutparser`` is not installed."""

        def __init__(
            self,
            pdf_path: str | io.BytesIO,  # noqa: ARG002
            always_create_screenshots: bool = False,  # noqa: ARG002
            num_processes: int = 1,  # noqa: ARG002
        ):
            """
            Raise a clear :class:`ImportError` indicating the missing extra.

            Args:
                pdf_path (str | io.BytesIO): Unused.
                always_create_screenshots (bool, optional): Unused. Defaults
                    to ``False``.
                num_processes (int, optional): Unused. Defaults to ``1``.

            Raises:
                ImportError: Always; ``layoutparser`` is not installed.

            """
            raise ImportError(
                "Missing dependency layoutparser. "
                "Reinstall the package with pmo-parser[dl]."
            )

        def load(
            self,
            pdf_document: DocumentWrapper,  # noqa: ARG002
            always_create_screenshots: bool = False,  # noqa: ARG002
            num_processes: int = 1,  # noqa: ARG002
        ) -> list[Page]:
            """
            Raise a clear :class:`ImportError` indicating the missing extra.

            Args:
                pdf_document (DocumentWrapper): Unused.
                always_create_screenshots (bool, optional): Unused. Defaults
                    to ``False``.
                num_processes (int, optional): Unused. Defaults to ``1``.

            Returns:
                list[Page]: Never returns; always raises.

            Raises:
                ImportError: Always; ``layoutparser`` is not installed.

            """
            raise ImportError(
                "Missing dependency layoutparser. "
                "Reinstall the package with pmo-parser[dl]."
            )
