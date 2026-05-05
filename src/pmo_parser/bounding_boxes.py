"""
Bounding-box primitives used throughout the parser.

Defines :class:`BBox`, the image-bearing :class:`ImageBBox`, the text-bearing
:class:`TextBox` (with reading-order awareness via :class:`ReadingOrder`), and
helpers for sorting and joining text boxes in reading order.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any, TypedDict

import numpy as np
import pymupdf
from PIL import Image


class BBoxDict(TypedDict):
    """Dictionary representation of a :class:`BBox` for JSON serialization."""

    x0: float | int
    y0: float | int
    x1: float | int
    y1: float | int


class TextBoxDict(BBoxDict):
    """Dictionary representation of a :class:`TextBox` for JSON serialization."""

    text: str
    reading_order: int


class BBox:
    """
    Axis-aligned bounding box with float or int coordinates.

    The coordinates are expected to satisfy ``x0 <= x1`` and ``y0 <= y1`` but
    this is not currently enforced.

    Attributes:
        x0 (float | int): Left coordinate of the box.
        y0 (float | int): Top coordinate of the box.
        x1 (float | int): Right coordinate of the box.
        y1 (float | int): Bottom coordinate of the box.

    """

    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self, *, x0: float | int, y0: float | int, x1: float | int, y1: float | int
    ):
        """
        Initialize a bounding box from its corner coordinates.

        Args:
            x0 (float | int): Left coordinate of the box.
            y0 (float | int): Top coordinate of the box.
            x1 (float | int): Right coordinate of the box.
            y1 (float | int): Bottom coordinate of the box.

        """
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    @property
    def width(self) -> float | int:
        """
        Get the width of the box.

        Returns:
            float | int: Absolute difference ``|x1 - x0|``.

        """
        return abs(self.x1 - self.x0)

    @property
    def height(self) -> float | int:
        """
        Get the height of the box.

        Returns:
            float | int: Absolute difference ``|y1 - y0|``.

        """
        return abs(self.y1 - self.y0)

    @width.setter
    def width(self, value: float | int):
        """
        Set ``x1`` so that the box has the requested width.

        Args:
            value (float | int): New width; ``x1`` is set to ``x0 + value``.

        """
        self.x1 = self.x0 + value

    @height.setter
    def height(self, value: float | int):
        """
        Set ``y1`` so that the box has the requested height.

        Args:
            value (float | int): New height; ``y1`` is set to ``y0 + value``.

        """
        self.y1 = self.y0 + value

    @property
    def area(self) -> float | int:
        """
        Get the area of the box.

        Returns:
            float | int: Product of :attr:`width` and :attr:`height`.

        """
        return self.width * self.height

    @property
    def center(self) -> tuple[float | int, float | int]:
        """
        Get the center point of the box.

        Returns:
            tuple[float | int, float | int]: ``(x_center, y_center)``.

        """
        return (self.x0 + self.width / 2, self.y0 + self.height / 2)

    def to_dict(self) -> BBoxDict:
        """
        Serialize the box to a JSON-compatible dictionary.

        Returns:
            BBoxDict: Dictionary with keys ``"x0"``, ``"y0"``,
                ``"x1"`` and ``"y1"``.

        """
        return BBoxDict(x0=self.x0, y0=self.y0, x1=self.x1, y1=self.y1)

    @staticmethod
    def from_dict(dictionary: dict[str, float | int | Any]) -> BBox:
        """
        Construct a :class:`BBox` from a dictionary produced by :meth:`to_dict`.

        Args:
            dictionary (dict[str, float | int | Any]): Dictionary with keys ``"x0"``,
                ``"y0"``, ``"x1"`` and ``"y1"``.

        Returns:
            BBox: Bounding box with the coordinates from ``dictionary``.

        """
        for key in ["x0", "y0", "x1", "y1"]:
            if key not in dictionary:
                raise ValueError(f"Missing key {key} in dictionary")
            if not isinstance(dictionary[key], (int, float)):
                raise TypeError(
                    f"Expected numeric value for key {key}, got {type(dictionary[key])}"
                )

        return BBox(
            x0=dictionary["x0"],
            y0=dictionary["y0"],
            x1=dictionary["x1"],
            y1=dictionary["y1"],
        )

    def copy(self) -> BBox:
        """
        Return an independent copy of the box.

        Returns:
            BBox: New box with the same coordinates.

        """
        return BBox(x0=self.x0, y0=self.y0, x1=self.x1, y1=self.y1)

    @staticmethod
    def from_rect(rectangle: pymupdf.Rect) -> BBox:
        """
        Build a :class:`BBox` from a :class:`pymupdf.Rect`.

        Args:
            rectangle (pymupdf.Rect): MuPDF rectangle to copy coordinates from.

        Returns:
            BBox: Bounding box with the coordinates of ``rectangle``.

        """
        return BBox(x0=rectangle.x0, y0=rectangle.y0, x1=rectangle.x1, y1=rectangle.y1)

    def union(self, other: BBox | list[BBox]) -> BBox:
        """
        Return the smallest box enclosing ``self`` and ``other``.

        Args:
            other (BBox | list[BBox]): A single box or a list of boxes that
                should be unioned with ``self``.

        Returns:
            BBox: Smallest axis-aligned box enclosing all inputs.

        """
        if isinstance(other, BBox):
            return BBox.union_boxes([self, other])
        else:
            return BBox.union_boxes([self] + other)

    def is_equal(self, other: BBox) -> bool:
        """
        Check whether two boxes have identical coordinates.

        Args:
            other (BBox): Box to compare against.

        Returns:
            bool: ``True`` when all four coordinates of ``self`` and ``other``
                match exactly, ``False`` otherwise.

        """
        return (
            self.x0 == other.x0
            and self.y0 == other.y0
            and self.x1 == other.x1
            and self.y1 == other.y1
        )

    @staticmethod
    def union_boxes(boxes: Sequence[BBox]) -> BBox:
        """
        Return the smallest box enclosing every box in ``boxes``.

        Args:
            boxes (list[BBox]): Non-empty list of boxes to union.

        Returns:
            BBox: Smallest axis-aligned box enclosing all inputs.

        """
        x0 = min([box.x0 for box in boxes])
        y0 = min([box.y0 for box in boxes])
        x1 = max([box.x1 for box in boxes])
        y1 = max([box.y1 for box in boxes])
        return BBox(x0=x0, y0=y0, x1=x1, y1=y1)

    def __repr__(self) -> str:
        """
        Return a debug representation including all four coordinates.

        Returns:
            str: ``"<ClassName>(x0=..., y0=..., x1=..., y1=...)"``.

        """
        return (
            f"{type(self).__name__}("
            f"x0={self.x0}, y0={self.y0}, x1={self.x1}, y1={self.y1})"
        )

    def __str__(self) -> str:
        """
        Return a human-readable representation of the box.

        Returns:
            str: Same value as :meth:`__repr__`.

        """
        return self.__repr__()

    def __eq__(self, other) -> bool:
        """
        Raise to flag that ``==`` is deprecated; use :meth:`is_equal` instead.

        :class:`BBox` does not support ``==``; this method always raises.

        Args:
            other: Right-hand side of the comparison.

        Returns:
            bool: Never returns; always raises.

        Raises:
            DeprecationWarning: Always.

        """
        raise DeprecationWarning("Use is_equal instead")
        return (
            self.x0 == other.x0
            and self.y0 == other.y0
            and self.x1 == other.x1
            and self.y1 == other.y1
        )

    def __ne__(self, other) -> bool:
        """
        Raise to flag that ``!=`` is deprecated; mirror of :meth:`__eq__`.

        Args:
            other: Right-hand side of the comparison.

        Returns:
            bool: Never returns; always raises through :meth:`__eq__`.

        Raises:
            DeprecationWarning: Always.

        """
        return not self.__eq__(other)

    def overlap_ratio(self, other: BBox) -> float:
        """
        Compute the fraction of ``self`` that is covered by ``other``.

        If the boxes do not intersect, ``0.0`` is returned.

        Args:
            other (BBox): Other box to intersect with.

        Returns:
            float: ``intersection_area / self.area``.

        """
        # overlap ratio based on self
        intersection = self.intersect(other)
        if intersection is None:
            return 0.0

        return intersection.area / self.area

    def distance_vector(self, other: BBox) -> tuple[float | int, float | int]:
        """
        Compute the minimal vector pointing from ``self`` to ``other``.

        ``(0, 0)`` indicates that the two boxes intersect. Adapted from
        https://stackoverflow.com/questions/4978323/.

        Args:
            other (BBox): Other box to measure to.

        Returns:
            tuple[float | int, float | int]: ``(dx, dy)`` from ``self`` to
                ``other``.

        """
        left = other.x1 < self.x0
        right = self.x1 < other.x0
        bottom = other.y1 < self.y0
        top = self.y1 < other.y0
        if top and left:
            return (self.x0 - other.x1, self.y1 - other.y0)
        if left and bottom:
            return (self.x0 - other.x1, self.y0 - other.y1)
        if bottom and right:
            return (self.x1 - other.x0, self.y0 - other.y1)
        if right and top:
            return (self.x1 - other.x0, self.y1 - other.y0)
        if left:
            return (self.x0 - other.x1, 0)
        if right:
            return (self.x1 - other.x0, 0)
        if bottom:
            return (0, self.y0 - other.y1)
        if top:
            return (0, self.y1 - other.y0)

        # rectangles intersect
        return (0.0, 0.0)

    def distance(self, other: BBox) -> float | int:
        """
        Return the Manhattan distance between two boxes.

        The distance is zero when the boxes intersect.

        Args:
            other (BBox): Other box to measure to.

        Returns:
            float | int: ``|dx| + |dy|`` where ``(dx, dy)`` is the
                :meth:`distance_vector` to ``other``.

        """
        x_dist, y_dist = self.distance_vector(other)
        return abs(x_dist) + abs(y_dist)

    def shift(self, x_shift: float | int = 0, y_shift: float | int = 0):
        """
        Translate the box in place.

        Args:
            x_shift (float | int, optional): Amount added to ``x0`` and
                ``x1``. Defaults to ``0``.
            y_shift (float | int, optional): Amount added to ``y0`` and
                ``y1``. Defaults to ``0``.

        """
        self.x0 += x_shift
        self.y0 += y_shift
        self.x1 += x_shift
        self.y1 += y_shift

    def intersect(self, other: BBox) -> BBox | None:
        """
        Compute the intersection of ``self`` and ``other``.

        If the boxes are disjoint, ``None`` is returned.

        Args:
            other (BBox): Other box to intersect with.

        Returns:
            BBox: Box covering the area shared by ``self`` and ``other``.

        """
        intersection_x0 = max(self.x0, other.x0)
        intersection_y0 = max(self.y0, other.y0)
        intersection_x1 = min(self.x1, other.x1)
        intersection_y1 = min(self.y1, other.y1)
        if intersection_x0 < intersection_x1 and intersection_y0 < intersection_y1:
            return BBox(
                x0=intersection_x0,
                y0=intersection_y0,
                x1=intersection_x1,
                y1=intersection_y1,
            )
        else:
            return None

    def get_intermediate_box(self, other: BBox) -> BBox | None:
        """
        Get the gap separating two non-overlapping boxes.

        For boxes that lie side-by-side or stacked, the gap is the rectangle
        between them. If the boxes intersect or are positioned diagonally
        relative to each other, no meaningful gap exists and ``None`` is
        returned.

        Args:
            other (BBox): Other box to compute the gap to.

        Returns:
            BBox: Rectangle between ``self`` and ``other``.

        """
        x_overlap = min(self.x1, other.x1) - max(self.x0, other.x0)
        y_overlap = min(self.y1, other.y1) - max(self.y0, other.y0)
        x_gap = max(0, -x_overlap)
        y_gap = max(0, -y_overlap)

        if x_gap > 0 and y_overlap > 0:
            # Boxes are next to each other
            return BBox(
                x0=min(self.x1, other.x1),
                y0=max(self.y0, other.y0),
                x1=max(self.x0, other.x0),
                y1=min(self.y1, other.y1),
            )

        if y_gap > 0 and x_overlap > 0:
            # Boxes are above each other
            return BBox(
                x0=max(self.x0, other.x0),
                y0=min(self.y1, other.y1),
                x1=min(self.x1, other.x1),
                y1=max(self.y0, other.y0),
            )

        # Boxes are diagonal to each other
        # or boxes intersect
        return None


class ImageBBox(BBox):
    """
    A :class:`BBox` carrying an associated :class:`PIL.Image.Image`.

    Attributes:
        x0 (float | int): Left coordinate of the box.
        y0 (float | int): Top coordinate of the box.
        x1 (float | int): Right coordinate of the box.
        y1 (float | int): Bottom coordinate of the box.
        image (PIL.Image.Image | None): Pixel data associated with the region.
            ``None`` when only the geometry is known.

    """

    def __init__(
        self,
        *,
        x0: float | int,
        y0: float | int,
        x1: float | int,
        y1: float | int,
        image: str | Image.Image | None,
    ):
        """
        Initialize from coordinates and an image.

        Args:
            x0 (float | int): Left coordinate of the box.
            y0 (float | int): Top coordinate of the box.
            x1 (float | int): Right coordinate of the box.
            y1 (float | int): Bottom coordinate of the box.
            image (str | PIL.Image.Image | None): Path to an image file, an
                already loaded PIL image, or ``None`` if no image is
                available yet.

        """
        super().__init__(x0=x0, y0=y0, x1=x1, y1=y1)
        if isinstance(image, str):
            self.image = Image.open(image)
        else:
            self.image = image

        self._virtual_dpi: None | int = None
        self._virtual_size: None | tuple[int, int] = None

    @property
    def dpi(self) -> int:
        """
        Get the effective DPI implied by the image height vs box height.

        Falls back to the configured virtual DPI when no image is attached.

        Returns:
            int: DPI of the image (or the configured virtual DPI).

        Raises:
            ValueError: If neither an image nor a virtual DPI is set.

        """
        if self.image is None:
            if self._virtual_dpi is not None:
                return self._virtual_dpi
            raise ValueError("Image is None, cannot determine DPI")
        return int(round(72 * self.image.height / self.height))

    def copy(self) -> ImageBBox:
        """
        Return an independent copy, also copying the underlying image.

        Returns:
            ImageBBox: New box with the same coordinates and a copy of the
                image.

        """
        bbox_copy = super().copy()
        image_copy = None if self.image is None else self.image.copy()
        return ImageBBox(
            x0=bbox_copy.x0,
            y0=bbox_copy.y0,
            x1=bbox_copy.x1,
            y1=bbox_copy.y1,
            image=image_copy,
        )

    def get_bbox(self) -> BBox:
        """
        Return a plain :class:`BBox` with the same coordinates.

        Returns:
            BBox: Geometry-only copy of this box.

        """
        return BBox(x0=self.x0, y0=self.y0, x1=self.x1, y1=self.y1)

    @staticmethod
    def from_dict(dictionary: dict[str, float | int | str | Image.Image]) -> ImageBBox:
        """
        Build an :class:`ImageBBox` from a serialized dictionary.

        Args:
            dictionary (dict[str, float | int | str | PIL.Image.Image]):
                Dictionary with keys ``"x0"``, ``"y0"``, ``"x1"``, ``"y1"``
                and ``"image"``.

        Returns:
            ImageBBox: Box constructed from ``dictionary``.

        """
        x0 = dictionary.get("x0")
        y0 = dictionary.get("y0")
        x1 = dictionary.get("x1")
        y1 = dictionary.get("y1")
        image = dictionary.get("image")

        if x0 is None or y0 is None or x1 is None or y1 is None:
            raise ValueError("Missing coordinate keys in dictionary")

        if not isinstance(x0, (int, float)):
            raise TypeError(f"Expected numeric value for x0, got {type(x0)}")
        if not isinstance(y0, (int, float)):
            raise TypeError(f"Expected numeric value for y0, got {type(y0)}")
        if not isinstance(x1, (int, float)):
            raise TypeError(f"Expected numeric value for x1, got {type(x1)}")
        if not isinstance(y1, (int, float)):
            raise TypeError(f"Expected numeric value for y1, got {type(y1)}")

        if image is not None and not isinstance(image, (str, Image.Image)):
            raise TypeError(
                f"Expected image to be str or PIL.Image.Image, got {type(image)}"
            )

        return ImageBBox(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            image=image,
        )

    @property
    def image_size(self) -> tuple[int, int]:
        """
        Get the pixel size of the attached image.

        Falls back to the configured virtual size when no image is attached.

        Returns:
            tuple[int, int]: ``(width, height)`` in pixels.

        Raises:
            ValueError: If neither an image nor a virtual size is set.

        """
        if self.image is None:
            if self._virtual_size is not None:
                return self._virtual_size
            raise ValueError("Image is None, cannot determine size")
        return self.image.size

    def calc_virtual_size(self):
        """
        Compute the virtual pixel size from the configured virtual DPI.

        Raises:
            ValueError: If ``_virtual_dpi`` has not been set.

        """
        if self._virtual_dpi is not None:
            self._virtual_size = (
                int((self.x1 - self.x0) * self._virtual_dpi / 72),
                int((self.y1 - self.y0) * self._virtual_dpi / 72),
            )
        else:
            raise ValueError("Virtual DPI is not set, cannot calculate virtual size")

    @staticmethod
    def from_bbox(bbox: BBox, image: str | Image.Image | None) -> ImageBBox:
        """
        Build an :class:`ImageBBox` from a :class:`BBox` and an image.

        Args:
            bbox (BBox): Source box providing the coordinates.
            image (str | PIL.Image.Image | None): Path to an image file or an already
                loaded PIL image.

        Returns:
            ImageBBox: Image-bearing box at the coordinates of ``bbox``.

        """
        return ImageBBox(
            x0=bbox.x0,
            y0=bbox.y0,
            x1=bbox.x1,
            y1=bbox.y1,
            image=image,
        )


class ReadingOrder(Enum):
    """
    Possible reading orders for a sequence of text boxes.

    Attributes:
        LTR_TD: Left to right, then top to bottom.
        RTL_TD: Right to left, then top to bottom.
        LTR_BU: Left to right, then bottom to top.
        RTL_BU: Right to left, then bottom to top.
        TD_LTR: Top to bottom, then left to right.
        TD_RTL: Top to bottom, then right to left.
        BU_LTR: Bottom to top, then left to right.
        BU_RTL: Bottom to top, then right to left.

    """

    LTR_TD = 0
    RTL_TD = 1
    LTR_BU = 2
    RTL_BU = 3
    TD_LTR = 4
    TD_RTL = 5
    BU_LTR = 6
    BU_RTL = 7


class TextBox(BBox):
    """
    A :class:`BBox` carrying its text content and a reading order.

    Attributes:
        x0 (float | int): Left coordinate of the box.
        y0 (float | int): Top coordinate of the box.
        x1 (float | int): Right coordinate of the box.
        y1 (float | int): Bottom coordinate of the box.
        text (str): Text contained in the region.
        reading_order (ReadingOrder): Reading order used when sorting or
            joining this text box with others.

    """

    def __init__(
        self,
        *,
        x0: float | int,
        y0: float | int,
        x1: float | int,
        y1: float | int,
        text: str,
        reading_order: ReadingOrder = ReadingOrder.LTR_TD,
    ):
        """
        Initialize from coordinates, text and a reading order.

        Args:
            x0 (float | int): Left coordinate of the box.
            y0 (float | int): Top coordinate of the box.
            x1 (float | int): Right coordinate of the box.
            y1 (float | int): Bottom coordinate of the box.
            text (str): Text contained in the region.
            reading_order (ReadingOrder, optional): Reading order used when
                sorting or joining boxes. Defaults to
                :attr:`ReadingOrder.LTR_TD`.

        """
        super().__init__(x0=x0, y0=y0, x1=x1, y1=y1)
        self.text = text
        self.reading_order = reading_order

    def union(self, other: BBox | list[BBox]) -> BBox:
        """
        Not implemented for :class:`TextBox`.

        Use :meth:`union_text_box` instead to combine text boxes.

        Args:
            other (BBox | list[BBox]): Other box or list of boxes (unused).

        Returns:
            BBox: Never returns; always raises.

        Raises:
            NotImplementedError: Always.

        """
        raise NotImplementedError("TextBox does not support union yet")

    def copy(self) -> TextBox:
        """
        Return an independent copy of this text box.

        Returns:
            TextBox: New text box with identical coordinates, text and
                reading order.

        """
        bbox_copy = super().copy()
        return TextBox.from_bbox(bbox_copy, self.text, self.reading_order)

    @staticmethod
    def from_bbox(
        bbox: BBox, text: str, reading_order: ReadingOrder = ReadingOrder.LTR_TD
    ) -> TextBox:
        """
        Build a :class:`TextBox` from a :class:`BBox`, text and reading order.

        Args:
            bbox (BBox): Source box providing the coordinates.
            text (str): Text contained in the region.
            reading_order (ReadingOrder, optional): Reading order used when
                sorting or joining boxes. Defaults to
                :attr:`ReadingOrder.LTR_TD`.

        Returns:
            TextBox: New text box at the coordinates of ``bbox``.

        """
        return TextBox(
            x0=bbox.x0,
            y0=bbox.y0,
            x1=bbox.x1,
            y1=bbox.y1,
            text=text,
            reading_order=reading_order,
        )

    @staticmethod
    def sort_text_in_reading_order(
        text_boxes: Sequence[TextBox], epsilon: float | None = None
    ) -> Sequence[TextBox]:
        """
        Sort the text boxes in their common reading order.

        Boxes whose y-centers differ by less than ``epsilon`` are treated as
        the same row and sorted by ``x0`` within that row.

        Args:
            text_boxes (list[TextBox]): Non-empty list of boxes that share a
                single :class:`ReadingOrder`.
            epsilon (float | None, optional): Tolerance used to decide whether
                two y-centers belong to the same row. Defaults to one quarter
                of the smallest box height.

        Returns:
            list[TextBox]: ``text_boxes`` sorted in reading order.

        Raises:
            AssertionError: If the boxes do not share a single reading order.
            NotImplementedError: For any reading order other than
                :attr:`ReadingOrder.LTR_TD`.

        """
        assert (
            text_boxes[0].reading_order in ReadingOrder
        ), "Reading order not recognized"
        assert all(
            [text_boxes[0].reading_order == box.reading_order for box in text_boxes]
        ), "Reading order must be the same"
        if epsilon is None:
            epsilon = min([box.height for box in text_boxes]) / 4
        reading_order = text_boxes[0].reading_order
        if reading_order == ReadingOrder.LTR_TD:
            # sort by y_center and x_0
            # Allow a small epsilon for the y_center
            text_box_dict: dict[float | int, list[TextBox]] = {}
            for box in text_boxes:
                if box.center[1] not in text_box_dict:
                    # Check if there is a similar y value
                    key_deltas = [abs(y - box.center[1]) for y in text_box_dict]
                    if len(text_box_dict) > 0 and min(key_deltas) < epsilon:
                        text_box_dict[
                            list(text_box_dict.keys())[
                                key_deltas.index(min(key_deltas))
                            ]
                        ].append(box)
                        continue
                    text_box_dict[box.center[1]] = []
                text_box_dict[box.center[1]].append(box)

            text_boxes = [
                box
                for y in text_box_dict
                for box in sorted(text_box_dict[y], key=lambda x: x.x0)
            ]
            return text_boxes
        else:
            raise NotImplementedError("Reading order not implemented yet")

    @staticmethod
    def get_text_from_boxes(
        boxes: Sequence[TextBox],
        epsilon_x: float | None = None,
        epsilon_y: float | None = None,
        sort: bool = True,
        add_new_lines: bool = True,
    ) -> str:
        """
        Concatenate the text of ``boxes`` with separators between them.

        Boxes that are far apart horizontally are separated with a space; if
        ``add_new_lines`` is ``True``, boxes whose y-centers differ by more
        than ``epsilon_y`` are separated with a newline instead.

        Args:
            boxes (list[TextBox]): Non-empty list of boxes whose text should
                be joined.
            epsilon_x (float | None, optional): Horizontal-gap threshold above
                which a space is inserted. Defaults to one third of the
                smallest box height.
            epsilon_y (float | None, optional): Vertical-center tolerance used
                to decide whether two boxes are on the same line. Defaults to
                one quarter of the smallest box height.
            sort (bool, optional): When ``True``, boxes are first sorted in
                reading order. Defaults to ``True``.
            add_new_lines (bool, optional): When ``True``, vertical gaps
                produce newlines. Defaults to ``True``.

        Returns:
            str: Concatenated text with spaces and (optionally) newlines
                inserted between boxes.

        """
        # Estimate epsilon if not given
        min_height = min([box.height for box in boxes])
        if epsilon_x is None:
            epsilon_x = min_height / 3
        if epsilon_y is None:
            epsilon_y = min_height / 4

        if sort:
            sorted_boxes = TextBox.sort_text_in_reading_order(boxes, epsilon=epsilon_y)
        else:
            sorted_boxes = boxes

        joined_text = ""
        previous_y = -1
        for index, box in enumerate(sorted_boxes):
            if previous_y == -1:
                joined_text += box.text
                previous_y = box.center[1]
                continue
            if add_new_lines and abs(box.center[1] - previous_y) > epsilon_y:
                joined_text += "\n"
            elif box.distance(sorted_boxes[index - 1]) > epsilon_x:
                joined_text += " "
            joined_text += box.text
            previous_y = box.center[1]
        return joined_text

    @staticmethod
    def union_text_boxes(boxes: list[TextBox]) -> TextBox:
        """
        Combine multiple text boxes into a single :class:`TextBox`.

        Args:
            boxes (list[TextBox]): Non-empty list of boxes to merge.

        Returns:
            TextBox: Box whose geometry encloses all inputs and whose text is
                the reading-order join of the input texts.

        """
        # Sort boxes by reading order
        sorted_boxes = TextBox.sort_text_in_reading_order(boxes)
        return TextBox.from_bbox(
            BBox.union_boxes([box.get_bbox() for box in sorted_boxes]),
            text=TextBox.get_text_from_boxes(sorted_boxes, sort=False),
            reading_order=sorted_boxes[0].reading_order,
        )

    def union_text_box(self, other: TextBox | list[TextBox]) -> TextBox:
        """
        Combine ``self`` with one or more text boxes.

        Args:
            other (TextBox | list[TextBox]): A single text box or a list of
                text boxes that should be merged with ``self``.

        Returns:
            TextBox: Box whose geometry encloses all inputs and whose text is
                the reading-order join of the input texts.

        """
        if isinstance(other, TextBox):
            return TextBox.union_text_boxes([self, other])
        else:
            return TextBox.union_text_boxes([self] + other)

    def get_bbox(self) -> BBox:
        """
        Return a plain :class:`BBox` with the same coordinates.

        Returns:
            BBox: Geometry-only copy of this text box.

        """
        return BBox(x0=self.x0, y0=self.y0, x1=self.x1, y1=self.y1)

    def to_dict(self) -> TextBoxDict:
        """
        Serialize the text box to a JSON-compatible dictionary.

        Returns:
            TextBoxDict: Dictionary with keys ``"x0"``,
                ``"y0"``, ``"x1"``, ``"y1"``, ``"text"`` and
                ``"reading_order"``.

        """
        super_dict = super().to_dict()
        return TextBoxDict(
            x0=super_dict["x0"],
            y0=super_dict["y0"],
            x1=super_dict["x1"],
            y1=super_dict["y1"],
            text=self.text,
            reading_order=self.reading_order.value,
        )

    @staticmethod
    def from_dict(dictionary: dict[str, float | int | str]) -> TextBox:
        """
        Build a :class:`TextBox` from a dictionary produced by :meth:`to_dict`.

        Args:
            dictionary (dict[str, float | int | str]): Dictionary with keys
                ``"x0"``, ``"y0"``, ``"x1"``, ``"y1"``, ``"text"`` and
                ``"reading_order"``.

        Returns:
            TextBox: New text box reconstructed from ``dictionary``.

        """
        x0 = dictionary.get("x0")
        y0 = dictionary.get("y0")
        x1 = dictionary.get("x1")
        y1 = dictionary.get("y1")
        text = dictionary.get("text")

        if x0 is None or y0 is None or x1 is None or y1 is None:
            raise ValueError("Missing coordinate keys in dictionary")
        if text is None:
            raise ValueError("Missing text key in dictionary")

        if not isinstance(x0, (int, float)):
            raise TypeError(f"Expected numeric value for x0, got {type(x0)}")
        if not isinstance(y0, (int, float)):
            raise TypeError(f"Expected numeric value for y0, got {type(y0)}")
        if not isinstance(x1, (int, float)):
            raise TypeError(f"Expected numeric value for x1, got {type(x1)}")
        if not isinstance(y1, (int, float)):
            raise TypeError(f"Expected numeric value for y1, got {type(y1)}")
        if not isinstance(text, str):
            raise TypeError(f"Expected string value for text, got {type(text)}")

        return TextBox(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            text=text,
            reading_order=ReadingOrder(dictionary["reading_order"]),
        )

    def __repr__(self) -> str:
        """
        Return a debug representation truncating long texts.

        Returns:
            str: ``"TextBox(x0=..., y0=..., x1=..., y1=..., text=...)"`` with
                ``text`` truncated to ten characters.

        """
        text_preview = self.text[:10] + "..." if len(self.text) > 10 else self.text
        return (
            f"TextBox(x0={self.x0:.2f}, y0={self.y0:.2f}, "
            f"x1={self.x1:.2f}, y1={self.y1:.2f}, text={text_preview})"
        )


def join_in_reading_order(
    text_boxes: Sequence[TextBox], cut_off_distance: float = np.inf
) -> Sequence[TextBox]:
    """
    Sort and group text boxes in natural reading order.

    Uses the reading order of the text boxes to sort them and groups boxes
    that are closer than ``cut_off_distance``; boxes farther apart are kept
    in separate groups in the output.

    Args:
        text_boxes (Sequence[TextBox]): List of text boxes to sort and group.
        cut_off_distance (float, optional): Distance above which two boxes
            are placed into separate output groups. Defaults to ``np.inf``.

    Returns:
        Sequence[TextBox]: One :class:`TextBox` per resulting group, each
            enclosing the boxes of that group.

    Raises:
        ValueError: If any input box has an unrecognised reading order.
        NotImplementedError: For any reading order other than
            :attr:`ReadingOrder.LTR_TD`.

    """
    if any(box.reading_order not in ReadingOrder for box in text_boxes):
        raise ValueError("Reading order not recognized")

    if any([box.reading_order != ReadingOrder.LTR_TD for box in text_boxes]):
        raise NotImplementedError("Reading order other than LTR_TD not implemented yet")

    # Create dictionary with the text boxes sorted by their y0 and x0 value
    sorted_boxes: dict[float, dict[float, list[TextBox]]] = {}
    for box in text_boxes:
        if box.y0 not in sorted_boxes:
            sorted_boxes[box.y0] = {}
        if box.x0 not in sorted_boxes[box.y0]:
            sorted_boxes[box.y0][box.x0] = []
        sorted_boxes[box.y0][box.x0].append(box)

    row_boxes = [
        BBox.union_boxes(
            [b.get_bbox() for box_list in boxes.values() for b in box_list]
        )
        for boxes in sorted_boxes.values()
    ]
    # Calculate the distances to the previous boxes
    y_groups: list[dict[float, list[TextBox]]] = []

    for index, y0 in enumerate(sorted_boxes):
        if index == 0:
            y_groups.append(sorted_boxes[y0])
            continue

        # Calculate the distance to the previous group
        distance = row_boxes[index].distance(row_boxes[index - 1])
        if distance > cut_off_distance:
            y_groups.append(sorted_boxes[y0])
        else:
            # join the dictionaries
            for x0 in sorted_boxes[y0]:
                if x0 not in y_groups[-1]:
                    y_groups[-1][x0] = sorted_boxes[y0][x0]
                else:
                    y_groups[-1][x0].extend(sorted_boxes[y0][x0])

    # Now split the groups by x0 value
    text_groups: list[list[list[TextBox]]] = []
    for group in y_groups:
        column_boxes = [
            BBox.union_boxes([b.get_bbox() for b in boxes]) for boxes in group.values()
        ]

        for index, x0 in enumerate(group):
            if index == 0:
                text_groups.append([group[x0]])
                continue

            distance = column_boxes[index].distance(column_boxes[index - 1])
            if distance > cut_off_distance:
                text_groups[-1].append(group[x0])
            else:
                text_groups[-1][-1].extend(group[x0])

    # join text boxes within the groups
    resulting_text_boxes = [
        TextBox.union_text_boxes(subgroup)
        for group in text_groups
        for subgroup in group
    ]

    return resulting_text_boxes
