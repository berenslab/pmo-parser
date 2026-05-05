"""
Figure data classes used in the extraction pipeline.

Defines :class:`FigureType`, the base :class:`Figure` (a region with optional
captions), and :class:`OutputFigure` enriched with a rendered image, caption
scores and clustering information.
"""

import enum
import re
from collections.abc import Sequence
from typing import TypedDict

from PIL import Image

from pmo_parser.bounding_boxes import (
    BBox,
    BBoxDict,
    TextBox,
    TextBoxDict,
    join_in_reading_order,
)
from pmo_parser.page import PageText


class FigureType(enum.Enum):
    """
    Coarse classification of an extracted figure.

    Attributes:
        FIGURE: Regular pictorial figure.
        TABLE: Tabular figure.
        OTHER: Anything else (logos, decorative elements, ...).

    """

    FIGURE = 0
    TABLE = 1
    OTHER = 3


class FigureDict(TypedDict):
    """Dictionary representation of a :class:`Figure` for JSON serialization."""

    caption: None | list[TextBoxDict]
    type: str
    figure_bbox: BBoxDict
    page: int
    name: str | None


class OutputFigureDict(FigureDict):
    """Dictionary representation of an :class:`OutputFigure` for JSON serialization."""

    caption_scores: None | Sequence[float]
    base_figure_bbox: BBoxDict
    used_screenshot: bool
    cluster_bboxes: None | Sequence[BBoxDict]
    dpi: None | int


class Figure:
    """
    A figure region on a page, optionally with associated captions.

    Attributes:
        page (int): Zero-based page number on which the figure appears.
        figure_bbox (BBox): Bounding box of the figure on the page.
        captions (PageText | None): Caption text boxes belonging to the
            figure. ``None`` when no captions are attached.
        name (str | None): Optional human-readable name (e.g. ``"Figure 1"``).
        type (FigureType): Coarse classification of the figure.

    """

    def __init__(
        self,
        page: int,
        figure_bbox: BBox,
        captions: PageText | None = None,
        name: str | None = None,
        figure_type: FigureType = FigureType.FIGURE,
    ):
        """
        Initialize a figure.

        Args:
            page (int): Zero-based page number on which the figure appears.
            figure_bbox (BBox): Bounding box of the figure on the page.
            captions (PageText | None, optional): Caption text boxes
                belonging to the figure. Defaults to ``None``.
            name (str | None, optional): Human-readable name. Defaults to
                ``None``.
            figure_type (FigureType, optional): Coarse classification of the
                figure. Defaults to :attr:`FigureType.FIGURE`.

        """
        self.name = name
        self.type: FigureType = figure_type
        self.captions: PageText | None = captions
        self.figure_bbox: BBox = figure_bbox
        self.page: int = page

    def has_caption(self) -> bool:
        """
        Check whether the figure has any caption text boxes attached.

        Returns:
            bool: ``True`` if at least one caption is attached, ``False``
                otherwise.

        """
        return self.captions is not None

    def join_texts(self):
        """Join caption text boxes in natural reading order in place."""
        if self.captions is not None:
            self.captions = join_in_reading_order(self.captions)

    def get_caption_texts(self) -> list[str]:
        """
        Return the joined text of each caption block.

        Returns:
            list[str]: One string per caption block. Empty list when no
                captions are attached.

        """
        if self.captions is None:
            return []
        self.join_texts()
        return [cap.text for cap in self.captions]

    @property
    def caption_bbox(self) -> BBox | None:
        """
        Get a single bounding box enclosing all captions.

        If no captions are attached, ``None`` is returned.

        Returns:
            BBox: Smallest box enclosing every caption.

        """
        if self.captions is not None:
            return BBox.union_boxes([cap for cap in self.captions])
        return None

    def contains_caption(self, caption: TextBox) -> bool:
        """
        Check whether ``caption`` is one of this figure's caption boxes.

        Args:
            caption (TextBox): Caption to look for.

        Returns:
            bool: ``True`` if a caption with identical coordinates is
                attached, ``False`` otherwise.

        """
        if self.captions is None:
            return False

        for cap in self.captions:
            if cap.is_equal(caption):
                return True
        return False

    def to_dict(
        self,
    ) -> FigureDict:
        """
        Serialize the figure to a JSON-compatible dictionary.

        Returns:
            FigureDict: Dictionary with keys ``"caption"``, ``"type"``,
                ``"figure_bbox"``, ``"page"`` and ``"name"``.

        """
        self.join_texts()

        return FigureDict(
            caption=None
            if self.captions is None
            else [cap.to_dict() for cap in self.captions],
            type=self.type.name,
            figure_bbox=self.figure_bbox.to_dict(),
            page=self.page,
            name=self.name,
        )

    def copy(self) -> "Figure":
        """
        Return an independent copy of this figure.

        Captions and the figure bounding box are deep-copied.

        Returns:
            Figure: New figure with the same data and independent captions.

        """
        return Figure(
            page=self.page,
            figure_type=self.type,
            figure_bbox=self.figure_bbox.copy(),
            captions=None
            if self.captions is None
            else [cap.copy() for cap in self.captions],
            name=self.name,
        )


class OutputFigure(Figure):
    """
    A :class:`Figure` enriched with rendering, scores and clustering data.

    Attributes:
        page (int): Zero-based page number on which the figure appears.
        figure_bbox (BBox): Refined bounding box of the figure on the page.
        captions (PageText | None): Caption text boxes belonging to the
            figure. ``None`` when no captions are attached.
        name (str | None): Optional human-readable name.
        type (FigureType): Coarse classification of the figure.
        base_figure_bbox (BBox): Original bounding box detected before any
            refinement step.
        used_screenshot (bool): Whether the rendered image was produced from
            a page screenshot rather than an embedded image.
        image (PIL.Image.Image | None): Rendered image of the figure region.
            ``None`` until the figure is rendered.
        figure_id (int | None): Numeric figure id (e.g. ``1`` for "Figure 1"),
            assigned during post-processing.
        caption_scores (list[float] | None): Score per caption produced by the
            caption-detection algorithm.
        cluster_bboxes (list[BBox] | None): Bounding boxes of the individual
            sub-figures when the figure represents a compound figure cluster.

    """

    def __init__(
        self,
        page: int,
        figure_bbox: BBox,
        captions: PageText | None = None,
        caption_scores: Sequence[float] | None = None,
        name: str | None = None,
        figure_type: FigureType = FigureType.FIGURE,
        base_figure_bbox: BBox | None = None,
        image: Image.Image | None = None,
        cluster_bboxes: Sequence[BBox] | None = None,
    ):
        """
        Initialize an output figure.

        Args:
            page (int): Zero-based page number on which the figure appears.
            figure_bbox (BBox): Refined bounding box of the figure.
            captions (PageText | None, optional): Caption text boxes.
                Defaults to ``None``.
            caption_scores (list[float] | None, optional): Score per caption.
                Defaults to ``None``.
            name (str | None, optional): Human-readable name. Defaults to
                ``None``.
            figure_type (FigureType, optional): Coarse classification of the
                figure. Defaults to :attr:`FigureType.FIGURE`.
            base_figure_bbox (BBox | None, optional): Original bounding box
                before any refinement step. Defaults to ``figure_bbox``.
            image (PIL.Image.Image | None, optional): Rendered image of the
                figure. Defaults to ``None``.
            cluster_bboxes (list[BBox] | None, optional): Sub-figure bounding
                boxes when the figure is a compound figure. Defaults to
                ``None``.

        """
        super().__init__(
            page,
            figure_bbox,
            captions=captions,
            name=name,
            figure_type=figure_type,
        )

        self.base_figure_bbox: BBox = (
            figure_bbox if base_figure_bbox is None else base_figure_bbox
        )
        self.used_screenshot: bool = False
        self.image: Image.Image | None = image
        self.figure_id: None | int = None
        self.caption_scores = caption_scores
        self.cluster_bboxes: Sequence[BBox] | None = cluster_bboxes

        self._dpi: None | int = None
        if self.image is not None:
            self._dpi = int(round(72 * self.image.height / self.figure_bbox.height))

    @staticmethod
    def from_base_figure(
        figure: Figure,
        actual_figure_bbox: BBox | None = None,
        used_screenshot: bool = False,
        image: Image.Image | None = None,
    ) -> "OutputFigure":
        """
        Build an :class:`OutputFigure` from a base :class:`Figure`.

        Args:
            figure (Figure): Source figure providing the page, type, captions
                and original bounding box.
            actual_figure_bbox (BBox | None, optional): Refined bounding box.
                Defaults to ``figure.figure_bbox``.
            used_screenshot (bool, optional): Whether the rendered image was
                produced from a page screenshot. Defaults to ``False``.
            image (PIL.Image.Image | None, optional): Rendered image of the
                figure. Defaults to ``None``.

        Returns:
            OutputFigure: New output figure carrying the data of ``figure``.

        """
        output = OutputFigure(
            page=figure.page,
            figure_type=figure.type,
            figure_bbox=figure.figure_bbox
            if actual_figure_bbox is None
            else actual_figure_bbox,
            captions=figure.captions,
            name=figure.name,
            base_figure_bbox=figure.figure_bbox,
        )

        output.used_screenshot = used_screenshot
        output.image = image

        return output

    @property
    def estimated_figure_ids(self) -> list[int]:
        """
        Get figure ids parsed from the caption texts.

        Looks for patterns like "Figure 1" or "Fig. 2" in each caption.

        Returns:
            list[int]: Unique numeric ids found across all captions.

        """
        potential_figure_ids = []

        for cap in self.get_caption_texts():
            match = re.match(r"fig\.?\s*(\d+)|figure\.?\s*(\d+)", cap, re.IGNORECASE)

            if match is not None:
                potential_figure_ids.append(int(match.group(1) or match.group(2)))

        return list(set(potential_figure_ids))

    def serialize(
        self, join: bool = True
    ) -> tuple[
        OutputFigureDict,
        Image.Image | None,
    ]:
        """
        Serialize the figure for persisted output.

        Args:
            join (bool, optional): When ``True``, captions are first joined
                in reading order. Defaults to ``True``.

        Returns:
            tuple[OutputFigureDict, PIL.Image.Image | None]: ``(dict, image)`` pair;
                ``image`` is ``None`` when the figure has not been rendered.

        """
        if join:
            self.join_texts()

        return OutputFigureDict(
            caption=None
            if self.captions is None
            else [cap.to_dict() for cap in self.captions],
            caption_scores=self.caption_scores,
            type=self.type.name,
            figure_bbox=self.figure_bbox.to_dict(),
            page=self.page,
            name=self.name,
            base_figure_bbox=self.base_figure_bbox.to_dict(),
            used_screenshot=self.used_screenshot,
            cluster_bboxes=None
            if self.cluster_bboxes is None
            else [bbox.to_dict() for bbox in self.cluster_bboxes],
            dpi=self._dpi,
        ), self.image

    def __repr__(self) -> str:
        """
        Return a debug representation including page, captions, name and type.

        Returns:
            str: Multi-field representation suitable for logs.

        """
        return (
            f"OutputFigure(page={self.page}, captions={self.captions}, "
            f"name={self.name}, figure_type={self.type}, "
            f"image={self.image is not None})"
        )
