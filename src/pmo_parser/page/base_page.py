"""
Abstract :class:`Page` and helpers shared by the page backends.

Defines the :class:`Page` ABC, the :class:`ImageCluster` data class, and the
:func:`convert_to_string` / :func:`sort_bboxes_in_reading_order` helpers used by the
concrete backends in :mod:`pmo_parser.page.mupdf_page` and
:mod:`pmo_parser.page.dl_page`.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeVar

import pymupdf
from PIL import Image

from pmo_parser.bounding_boxes import BBox, ImageBBox, TextBox

BBoxType = TypeVar("BBoxType", bound="BBox")
PageText = Sequence[TextBox]


def convert_to_string(
    text_boxes: PageText, letter_width: float, letter_height: float
) -> str:
    """
    Concatenate ``text_boxes`` into a single string with separators.

    Adjacent boxes are joined directly when their distance is below a quarter
    of ``letter_width``; far-apart boxes get a tab. Boxes within a line are
    separated by a single space.

    Args:
        text_boxes (PageText): Boxes to join, in the desired output
            order.
        letter_width (float): Mean letter width on the page; used as the unit
            for horizontal-distance thresholds.
        letter_height (float): Mean letter height on the page; used as the
            unit for vertical-distance thresholds.

    Returns:
        str: Joined text with spaces and tabs inserted between boxes.

    """
    current_text = ""
    for i in range(len(text_boxes)):
        if i == 0 or text_boxes[i].distance(text_boxes[i - 1]) < 0.25 * letter_width:
            current_text += text_boxes[i].text
        elif (
            text_boxes[i].distance(text_boxes[i - 1]) < letter_width
            or abs(text_boxes[i].y0 - text_boxes[i - 1].y1) < 1.3 * letter_height
        ):
            current_text += " " + text_boxes[i].text
        else:
            current_text += "\t" + text_boxes[i].text

    return current_text


def sort_bboxes_in_reading_order(
    bbox_list: Sequence[BBoxType], margin: float = 3
) -> Sequence[BBoxType]:
    """
    Sort ``bbox_list`` left-to-right, top-to-bottom, snapping near coordinates.

    Two boxes whose ``x0`` (resp. ``y0``) values differ by less than ``margin``
    are treated as having the same x (resp. y) coordinate so that they stay in
    their original relative order within a row or column.

    Args:
        bbox_list (list[BBox]): Boxes to sort.
        margin (float, optional): Coordinate tolerance below which boxes are
            considered to share the same row or column. Defaults to ``3``.

    Returns:
        list[BBox]: Boxes sorted in reading order.

    """
    if len(bbox_list) == 0:
        return []
    if len(bbox_list) == 1:
        return bbox_list
    x_values_inds = sorted(range(len(bbox_list)), key=lambda i: bbox_list[i].x0)
    y_values_inds = sorted(range(len(bbox_list)), key=lambda i: bbox_list[i].y0)

    x_values = [bbox_list[i].x0 for i in x_values_inds]
    y_values = [bbox_list[i].y0 for i in y_values_inds]

    y_values_within_margin = []

    for y in y_values:
        if len(y_values_within_margin) == 0:
            y_values_within_margin.append(y)
        elif abs(y - y_values_within_margin[-1]) > margin:
            y_values_within_margin.append(y)
        else:
            y_values_within_margin.append(y_values_within_margin[-1])

    x_values_within_margin = []

    for x in x_values:
        if len(x_values_within_margin) == 0:
            x_values_within_margin.append(x)
        elif abs(x - x_values_within_margin[-1]) > margin:
            x_values_within_margin.append(x)
        else:
            x_values_within_margin.append(x_values_within_margin[-1])

    # Bring x_values_within_margin and y_values_within_margin back into the
    # order of the input.
    x_values_within_margin = [
        x_values_within_margin[x_values_inds.index(i)]
        for i in range(len(x_values_inds))
    ]
    y_values_within_margin = [
        y_values_within_margin[y_values_inds.index(i)]
        for i in range(len(y_values_inds))
    ]

    arg_sorted = sorted(
        range(len(bbox_list)),
        key=lambda i: (y_values_within_margin[i], x_values_within_margin[i]),
    )

    return [bbox_list[i] for i in arg_sorted]


class ImageCluster:
    """
    Group of image ids backed by a single rendered screenshot.

    Attributes:
        image_ids (list[int]): Indices of the images that belong to the
            cluster, referencing :attr:`Page.page_figures`.
        screenshot (PIL.Image.Image | None): Rendered screenshot of the
            whole cluster, or ``None`` when no screenshot was created.

    """

    def __init__(self, image_ids: Sequence[int], screenshot: Image.Image | None):
        """
        Initialize the cluster.

        Args:
            image_ids (list[int]): Indices of the images belonging to the
                cluster.
            screenshot (PIL.Image.Image | None): Rendered screenshot of the
                cluster, or ``None`` when no screenshot was created.

        """
        self.image_ids = image_ids
        self.screenshot = screenshot


class PageBlocks:
    """Blocks parsed from a page, returned by :meth:`Page.parse_blocks`."""

    def __init__(
        self,
        page_texts: Sequence[PageText],
        page_figures: Sequence[ImageBBox],
        figure_clusters: Sequence[ImageCluster],
        remaining_paths: Sequence[BBox],
    ):
        """
        Create a PageBlocks object with the given attributes.

        Args:
            page_texts (Sequence[PageText]): Text boxes of the page grouped per text
                assignment (paragraph/line block).
            page_figures (Sequence[ImageBBox]): Detected figure regions on the page.
            figure_clusters (Sequence[ImageCluster]): Clusters joining
                :attr:`page_figures` entries that belong to the same compound figure.
            remaining_paths (Sequence[BBox]): Vector-drawing paths that were not
                absorbed into a figure region.

        """
        self.page_texts = page_texts
        self.page_figures = page_figures
        self.figure_clusters = figure_clusters
        self.remaining_paths = remaining_paths


class Page(ABC):
    """
    Abstract page parsed from a PDF document.

    Concrete subclasses implement :meth:`parse_blocks` to extract texts,
    figures and clusters from a single PDF page.

    Attributes:
        page_num (int): Zero-based page index in the source document.
        page_width (float): Width of the page in PDF units.
        page_height (float): Height of the page in PDF units.
        page_texts (list[PageText]): Text boxes of the page grouped per
            text assignment (paragraph/line block).
        page_figures (list[ImageBBox]): Detected figure regions on the page.
        figure_clusters (list[ImageCluster]): Clusters joining
            :attr:`page_figures` entries that belong to the same compound
            figure.
        remaining_paths (list[BBox]): Vector-drawing paths that were not
            absorbed into a figure region.
        mean_letter_width (float): Mean width of a letter on the page; used
            as the unit for horizontal-distance thresholds.
        mean_letter_height (float): Mean height of a letter on the page;
            used as the unit for vertical-distance thresholds.

    """

    def __init__(
        self,
        document: pymupdf.Document,
        page_num: int,
        dpi: int = 300,
        always_create_screenshots: bool = False,
        **kwargs,
    ):
        """
        Parse a single page of ``document``.

        Calls :meth:`parse_blocks` to populate :attr:`page_texts`,
        :attr:`page_figures`, :attr:`figure_clusters` and
        :attr:`remaining_paths`, then derives :attr:`mean_letter_width` and
        :attr:`mean_letter_height` from the resulting texts.

        Args:
            document (pymupdf.Document): Already opened PDF document.
            page_num (int): Zero-based page index to parse.
            dpi (int, optional): DPI used for any rendering performed during
                parsing. Defaults to ``300``.
            always_create_screenshots (bool, optional): When ``True``, page
                screenshots are rendered eagerly. Defaults to ``False``.
            **kwargs: Backend-specific arguments forwarded to
                :meth:`parse_blocks`.

        """
        self.page_num = page_num

        self.page_width = document[page_num].rect.width
        self.page_height = document[page_num].rect.height
        parsed_blocks = self.parse_blocks(
            document,
            dpi=dpi,
            always_create_screenshots=always_create_screenshots,
            **kwargs,
        )

        self.page_texts = parsed_blocks.page_texts
        self.page_figures = parsed_blocks.page_figures
        self.figure_clusters = parsed_blocks.figure_clusters
        self.remaining_paths = parsed_blocks.remaining_paths

        self.mean_letter_width = (
            4
            if len(self.page_texts) == 0
            else sum(
                [w.x1 - w.x0 for assignment in self.page_texts for w in assignment]
            )
            / sum([len(w.text) for assignment in self.page_texts for w in assignment])
        )
        self.mean_letter_height = (
            11
            if len(self.page_texts) == 0
            else sum(
                [w.y1 - w.y0 for assignment in self.page_texts for w in assignment]
            )
            / sum([1 for assignment in self.page_texts for w in assignment])
        )

    @abstractmethod
    def parse_blocks(self, doc: pymupdf.Document, dpi=300, **kwargs) -> PageBlocks:
        """
        Parse the page into texts, figures, clusters and remaining paths.

        Args:
            doc (pymupdf.Document): Already opened PDF document.
            dpi (int, optional): DPI used for any rendering performed during
                parsing. Defaults to ``300``.
            **kwargs: Backend-specific arguments.

        Returns:
            PageBlocks: Blocks parsed from the page as a single object with attributes
                ``(page_texts, page_figures, figure_clusters,
                remaining_paths)`` matching the same-named attributes on
                :class:`Page`.

        Raises:
            NotImplementedError: Subclasses must override this method.

        """
        raise NotImplementedError("parse_blocks not implemented")

    def get_string_texts(self) -> list[str]:
        """
        Return the joined text of every text assignment on the page.

        Returns:
            list[str]: One joined string per entry of :attr:`page_texts`.

        """
        joined_text = []

        for assignment in self.page_texts:
            current_text = convert_to_string(
                assignment, self.mean_letter_width, self.mean_letter_height
            )

            joined_text.append(current_text)
        return joined_text

    def get_cluster_index(self, image_id: int) -> int:
        """
        Return the index of the cluster containing ``image_id``.

        If no cluster contains the given id, ``-1`` is returned.

        Args:
            image_id (int): Index into :attr:`page_figures`.

        Returns:
            int: Index of the matching cluster in :attr:`figure_clusters`.

        """
        if len(self.figure_clusters) == 0:
            return -1
        for i, cluster in enumerate(self.figure_clusters):
            if image_id in cluster.image_ids:
                return i
        return -1
