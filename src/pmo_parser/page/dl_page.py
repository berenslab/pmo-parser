"""
Deep-learning-based :class:`Page` implementation built on layoutparser.

Detects figure regions on each PDF page using a layoutparser model, then
refines the detected boxes by unioning them with overlapping text, image and
path bounding boxes obtained from MuPDF.
"""

import io
import math
import warnings
from collections.abc import Sequence
from typing import Any, Protocol, TypedDict, TypeGuard, cast

import pymupdf
from PIL import Image

try:
    from layoutparser.elements import Layout  # pyright: ignore
    from layoutparser.models import AutoLayoutModel  # pyright: ignore
    from layoutparser.models.base_layoutmodel import BaseLayoutModel  # pyright: ignore
except ImportError:
    # layoutparser is not installed
    raise ImportError(
        "Missing dependency layoutparser. Reinstall the package with pmo-parser[dl]."
    )

from pmo_parser.bounding_boxes import BBox, ImageBBox, TextBox
from pmo_parser.page import Page, PageBlocks, sort_bboxes_in_reading_order
from pmo_parser.renderer import render_page


class MuPDFImageBlock(TypedDict, total=False):
    """Subset of MuPDF image-block keys used by this parser."""

    type: int
    bbox: Sequence[float]
    image: bytes


class MuPDFPageDict(TypedDict):
    """Subset of page-level MuPDF text-dict keys used by this parser."""

    width: float
    height: float
    blocks: list[MuPDFImageBlock | dict[str, Any]]


class _FigureLayoutItem(Protocol):
    """Layout item protocol exposing a string type label."""

    type: str


class _ScoredLayoutItem(Protocol):
    """Layout item protocol exposing a numeric score."""

    score: float | int


def _is_figure_layout_item(item: Any) -> TypeGuard[_FigureLayoutItem]:
    """Return True when a layout item has type 'Figure'."""
    item_type = item.type
    return isinstance(item_type, str) and item_type == "Figure"


def _layout_item_score(item: Any) -> float:
    """Read layout item score and raise when unavailable or non-numeric."""
    if _has_layout_score(item):
        return float(item.score)

    # Preserve fail-fast behavior: missing .score should raise AttributeError.
    score = item.score
    raise TypeError(f"Layout item score must be numeric, got {type(score).__name__}")


def _has_layout_score(item: Any) -> TypeGuard[_ScoredLayoutItem]:
    """Return True when a layout item exposes an int/float score."""
    try:
        return isinstance(item.score, (int, float))
    except AttributeError:
        return False


class DLPage(Page):
    """
    :class:`Page` backend that detects figures via a layout-parser model.

    Inherits all attributes from :class:`Page`. The figure clusters and
    remaining paths are not populated by this backend (returned empty).
    """

    def __init__(
        self,
        document: pymupdf.Document,
        page_num: int,
        dpi: int = 300,
        model: BaseLayoutModel | None = None,
        always_create_screenshots: bool = False,
    ):
        """
        Initialize the page.

        Args:
            document (pymupdf.Document): Already opened PDF document.
            page_num (int): Zero-based page index to parse.
            dpi (int, optional): DPI used when rendering the page for figure
                detection. Defaults to ``300``.
            model (BaseLayoutModel | None, optional): Pre-loaded layoutparser
                model. When ``None``, a default ``efficientdet/PubLayNet``
                model is loaded on first use. Defaults to ``None``.
            always_create_screenshots (bool, optional): When ``True``, page
                screenshots are rendered eagerly. Defaults to ``False``.

        """
        super().__init__(
            document,
            page_num,
            dpi=dpi,
            model=model,
            always_create_screenshots=always_create_screenshots,
        )

    def parse_blocks(
        self,
        doc: pymupdf.Document,
        dpi=300,
        model: BaseLayoutModel | None = None,  # pyright: ignore[reportRedeclaration]
        **kwargs,
    ) -> PageBlocks:
        """
        Run figure detection on the page.

        Renders the page, runs the layoutparser model, then refines the
        detected figure regions by unioning them with overlapping text
        assignments, embedded images and vector paths obtained from MuPDF.

        Args:
            doc (pymupdf.Document): Already opened PDF document.
            dpi (int, optional): DPI used when rendering the page. Defaults
                to ``300``.
            model (BaseLayoutModel | None, optional): Pre-loaded layoutparser
                model. When ``None``, a default ``efficientdet/PubLayNet``
                model is loaded. Defaults to ``None``.
            **kwargs: Unrecognized keyword arguments produce a warning.

        Returns:
            PageBlocks: Parsed blocks from the page as a single object with attributes
                ``(page_texts, page_figures, [], [])``. Clusters and
                remaining paths are not produced by this backend.

        Raises:
            ValueError: If two words on the page share the same ``(block,
                line, word)`` index.

        """
        if len(kwargs) > 0:
            warnings.warn(
                f"Unrecognized keyword arguments: {', '.join(kwargs.keys())}",
                UserWarning,
            )

        page_words = doc[self.page_num].get_text("words")  # pyright: ignore[reportAttributeAccessIssue]
        page_blocks = cast(
            MuPDFPageDict,
            doc[self.page_num].get_text("dict"),  # pyright: ignore[reportAttributeAccessIssue]
        )
        page_paths = doc[self.page_num].get_drawings()
        page_height = float(page_blocks["height"])
        page_width = float(page_blocks["width"])

        if model is None:
            # Cast to correct type
            model: BaseLayoutModel = AutoLayoutModel(
                config_path="lp://efficientdet/PubLayNet"
            )

        page_dict = {}
        for w in page_words:
            if w[5] not in page_dict:
                page_dict[w[5]] = {}
            if w[6] not in page_dict[w[5]]:
                page_dict[w[5]][w[6]] = {}
            if w[7] in page_dict[w[5]][w[6]]:
                raise ValueError("Word index already exists")
            page_dict[w[5]][w[6]][w[7]] = w
        page_assignments = [
            [
                TextBox(
                    x0=page_dict[b][line_idx][w][0],
                    y0=page_dict[b][line_idx][w][1],
                    x1=page_dict[b][line_idx][w][2],
                    y1=page_dict[b][line_idx][w][3],
                    text=page_dict[b][line_idx][w][4],
                )
                for line_idx in sorted(page_dict[b].keys())
                for w in sorted(page_dict[b][line_idx].keys())
            ]
            for b in sorted(page_dict.keys())
        ]

        page_drawing = render_page(doc, self.page_num, dpi=dpi)
        png_ratio = float(page_drawing.size[1]) / page_height

        layout = model.detect(page_drawing.copy())
        assert layout is not None, "Layout detection failed"
        layout_items = list(cast(Sequence[Any], layout))
        figure_blocks = Layout([b for b in layout_items if _is_figure_layout_item(b)])

        lp_figure_boxes = []

        for block in figure_blocks:
            x0 = int(block.block.x_1 / png_ratio)
            y0 = int(block.block.y_1 / png_ratio)
            x1 = int(block.block.x_2 / png_ratio)
            y1 = int(block.block.y_2 / png_ratio)
            image = page_drawing.copy().crop(
                (
                    int(block.block.x_1),
                    int(block.block.y_1),
                    int(block.block.x_2),
                    int(block.block.y_2),
                )
            )
            c_bbox = ImageBBox(
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                image=image,
            )
            found_better_estimate = False
            for j in range(len(lp_figure_boxes)):
                if (
                    lp_figure_boxes[j] is not None
                    and c_bbox.overlap_ratio(lp_figure_boxes[j]) > 0.7
                ):
                    if _layout_item_score(figure_blocks[j]) < _layout_item_score(block):
                        lp_figure_boxes[j] = None
                    else:
                        found_better_estimate = True
                        break

            if not found_better_estimate:
                lp_figure_boxes.append(c_bbox)
            else:
                lp_figure_boxes.append(None)
        lp_figure_boxes = [im for im in lp_figure_boxes if im is not None]

        page_images: list[ImageBBox] = []
        for block in page_blocks["blocks"]:
            if "type" not in block:
                raise KeyError("Missing required MuPDF block key: 'type'")

            if block["type"] != 1:
                continue

            if "bbox" not in block:
                raise KeyError(
                    "Missing required MuPDF block key for image block: 'bbox'"
                )

            bbox = block["bbox"]
            if not (isinstance(bbox, Sequence) and len(bbox) == 4):
                raise TypeError("MuPDF image block 'bbox' must be a 4-item sequence")

            x0, y0, x1, y1 = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue

            if "image" not in block:
                raise KeyError(
                    "Missing required MuPDF block key for image block: 'image'"
                )

            raw_image = block["image"]
            if not isinstance(raw_image, (bytes, bytearray, memoryview)):
                raise TypeError("MuPDF image block 'image' must be bytes-like")

            page_images.append(
                ImageBBox(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    image=Image.open(io.BytesIO(bytes(raw_image))),
                )
            )

        img_types = [None for _ in page_images]

        for type_name, rect in doc[self.page_num].get_bboxlog():
            if type_name.lower() not in ["fill-imgmask", "fill-image"]:
                continue

            bbox = BBox(x0=rect[0], y0=rect[1], x1=rect[2], y1=rect[3])
            for i, im in enumerate(page_images):
                if (
                    abs(bbox.center[0] - im.center[0]) < 10
                    and abs(bbox.center[1] - im.center[1]) < 10
                    and abs(bbox.area - im.area) < 100
                ):
                    img_types[i] = type_name.lower()

        joined_masks = []
        joined_mask_types = []
        joined_mask_indices = []
        for i, im in enumerate(page_images):
            if img_types[i] is None or img_types[i] == "fill-image":
                joined_masks.append(im)
                joined_mask_types.append("image")
                joined_mask_indices.append([i])
                continue
            is_joined_mask = False
            for j, im2 in enumerate(joined_masks):
                if img_types[i] is None or img_types[i] == "fill-image":
                    continue
                if im2.overlap_ratio(im) > 0.05 or im2.distance(im) < 4:
                    joined_masks[j] = im.union(im2)
                    is_joined_mask = True
                    joined_mask_indices[j].append(i)
                    joined_mask_types[j] = "image-mask"
                    break
            if not is_joined_mask:
                joined_masks.append(im)
                joined_mask_types.append("image-mask")
                joined_mask_indices.append([i])

        page_images = joined_masks

        # Filter images in the header and footer
        page_images = [
            im
            for im in page_images
            if im.y0 > 0.05 * page_height and im.y1 < 0.95 * page_height
        ]

        path_bboxes = [BBox.from_rect(path["rect"]) for path in page_paths]
        path_bboxes = [
            b for b in path_bboxes if b.width < page_width and b.height < page_height
        ]

        # Refine boxes by intersecting them with the objects on the page
        page_figures = []

        for i, block in enumerate(lp_figure_boxes):
            new_block = block.copy()
            matching_objects = []
            max_image_dpi = -1
            for j, assignment in enumerate(page_assignments):
                assignment_box = TextBox.union_boxes(assignment)
                if assignment_box.overlap_ratio(new_block) > 0.05 and all(
                    [not a.text.lower().startswith("fig") for a in assignment]
                ):
                    new_block = new_block.union(assignment_box)
                    matching_objects.append(assignment_box)

            for j, im in enumerate(page_images):
                if new_block.overlap_ratio(im) > 0.05:
                    new_block = new_block.union(im)
                    matching_objects.append(im)

                    max_image_dpi = max(max_image_dpi, im.dpi)

            for j, drawing in enumerate(path_bboxes):
                if new_block.overlap_ratio(drawing) > 0.05:
                    new_block = new_block.union(drawing)
                    matching_objects.append(drawing)

            if max_image_dpi < 0:
                hq_page_rendering = page_drawing.copy()
            else:
                max_image_dpi = round(max_image_dpi)

                if (
                    abs(max_image_dpi - dpi) < 2
                ):  # Small negilgable difference -> not worth rerendering
                    hq_page_rendering = page_drawing.copy()
                else:
                    # Rerender
                    hq_page_rendering = render_page(
                        doc, self.page_num, dpi=max_image_dpi
                    )

            if len(matching_objects) == 0:
                continue

            hq_png_ratio = float(hq_page_rendering.size[1]) / page_height
            detected_box = BBox.union_boxes(matching_objects)  # TODO clip to page
            figure_block = ImageBBox.from_bbox(
                detected_box,
                hq_page_rendering.crop(
                    (
                        int(hq_png_ratio * detected_box.x0),
                        int(hq_png_ratio * detected_box.y0),
                        math.ceil(hq_png_ratio * detected_box.x1),
                        math.ceil(hq_png_ratio * detected_box.y1),
                    )
                ),
            )
            page_figures.append(figure_block)
        page_figures = sort_bboxes_in_reading_order(page_figures)

        return PageBlocks(
            page_texts=page_assignments,
            page_figures=page_figures,
            figure_clusters=[],
            remaining_paths=[],
        )
