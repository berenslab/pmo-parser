"""
SVG-based utilities for extracting figure regions from a PDF page.

The MuPDF backend renders each page to an SVG, which is then traversed by
:func:`walk_and_clip` to collect bounding boxes for paths and images while
respecting clip paths.
"""

from io import StringIO
from typing import Any, TypeGuard

import pymupdf
import svgelements

from pmo_parser.bounding_boxes import BBox


def get_bounding_box_svgelements(element: svgelements.SVGElement) -> BBox | None:
    """
    Compute the bounding box of a single SVG element.

    For :class:`svgelements.SVGImage`, the box is computed from ``x``, ``y``,
    ``width``, ``height`` and the element transform; for groups no bounding
    box exists and ``None`` is returned. Other elements fall back to the
    library's :meth:`svgelements.SVGElement.bbox` method.

    Args:
        element (svgelements.SVGElement): SVG element to inspect.

    Returns:
        BBox: Axis-aligned bounding box of ``element``.

    """
    if isinstance(element, svgelements.Group):
        return None  # Groups do not have a bbox themselves

    if isinstance(element, svgelements.SVGImage):
        # Manually calculate bbox for SVGImage
        x = element.x if element.x is not None else 0
        y = element.y if element.y is not None else 0
        width = element.width if element.width is not None else 0
        height = element.height if element.height is not None else 0

        # get transform
        transform = (
            element.transform if element.transform is not None else svgelements.Matrix()
        )
        # Apply transform to corners
        x0 = transform.a * x + transform.c * y + transform.e
        y0 = transform.b * x + transform.d * y + transform.f
        x1 = transform.a * (x + width) + transform.c * (y + height) + transform.e
        y1 = transform.b * (x + width) + transform.d * (y + height) + transform.f

        return BBox(x0=min(x0, x1), y0=min(y0, y1), x1=max(x0, x1), y1=max(y0, y1))

    bbox = element.bbox()  # pyright: ignore[reportAttributeAccessIssue]
    if bbox is None:
        return None
    return BBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3])


def intersect_box_with_clips(box: BBox | None, clip_stack: list[BBox]) -> BBox | None:
    """
    Clip the given box with every clipping box in the clip stack.

    If ``box`` is ``None`` or any intersection in the stack is empty, ``None``
    is returned.

    Args:
        box (BBox | None): The bounding box of the current element.
        clip_stack (list[BBox]): Clipping boxes to apply, applied left to
            right.

    Returns:
        BBox: The clipped bounding box.

    """
    if box is None:
        return None
    clip_box = box
    for clip in clip_stack:
        clip_box = clip_box.intersect(clip)
        if clip_box is None:
            return None
    return clip_box


def walk_and_clip(
    node: svgelements.SVGElement,
    clip_box: BBox | None = None,
    keep_top_level_groups=True,
    group_name=None,
    font_cache=None,
) -> list[dict[str, Any]]:
    """
    Walk the SVG tree, apply clipping paths and collect bounding boxes.

    Only bounding boxes of non-container elements are collected. Top-level
    groups are treated as figure boundaries when ``keep_top_level_groups`` is
    set.

    Args:
        node (svgelements.SVGElement): Root SVG element to start walking from.
        clip_box (BBox | None, optional): Active clip rectangle accumulated
            from ancestor clip paths. Defaults to ``None``.
        keep_top_level_groups (bool, optional): When ``True``, treat
            top-level groups as figure boundaries. Defaults to ``True``.
        group_name (str | None, optional): Name assigned to the enclosing
            group for downstream grouping. Defaults to ``None``.
        font_cache (dict | None, optional): Cache mapping ``font_*`` ids to
            their bounding boxes. Built lazily on the first call. Defaults
            to ``None``.

    Returns:
        list[dict[str, Any]]: One dictionary per collected element with keys
            ``"type"``, ``"clip_box"``, ``"dpi"`` and ``"group"``.

    Raises:
        NotImplementedError: When an element carries more than one clip path.

    """

    def _is_container(n) -> TypeGuard[svgelements.SVG | svgelements.Group]:
        return (
            isinstance(n, (svgelements.SVG, svgelements.Group))
            or hasattr(n, "__iter__")
        ) and not isinstance(
            n, (svgelements.Use, svgelements.Path, svgelements.Text, svgelements.Title)
        )

    def detect_empty_groups(n):
        if not isinstance(n, svgelements.Group):
            return False

        if len(n) == 0:
            return True

        container_children = []
        for child in n:
            if _is_container(child):
                container_children.append(child)
            else:
                return False

        for child in container_children:
            if not detect_empty_groups(child):
                return False
        return True

    if isinstance(node, svgelements.Use) and "data-text" not in node.values:  # pyright: ignore[reportOperatorIssue]
        return []

    if font_cache is None:
        font_cache = {}

        for font_name in node.objects:  # pyright: ignore[reportAttributeAccessIssue]
            if not font_name.startswith("font_"):
                continue
            font_bbox = get_bounding_box_svgelements(node.objects[font_name])  # pyright: ignore[reportAttributeAccessIssue]
            if font_bbox is not None:
                font_cache[font_name] = font_bbox
    node_type = (
        node.__class__.__name__
        if not isinstance(node, (svgelements.Use, svgelements.Text, svgelements.Title))
        else "Text"
    )
    if (
        isinstance(node, svgelements.Use)
        and "{http://www.w3.org/1999/xlink}href" in node.values  # pyright: ignore[reportOperatorIssue]
    ):
        href = node.values["{http://www.w3.org/1999/xlink}href"]  # pyright: ignore[reportOptionalSubscript]
        font_name = href.removeprefix("#")

        if font_name in font_cache:
            bbox = font_cache[font_name]
            transform = node.transform
            if transform is not None and bbox is not None:
                # Apply transform to corners
                x0 = transform.a * bbox.x0 + transform.c * bbox.y0 + transform.e
                y0 = transform.b * bbox.x0 + transform.d * bbox.y0 + transform.f
                x1 = transform.a * bbox.x1 + transform.c * bbox.y1 + transform.e
                y1 = transform.b * bbox.x1 + transform.d * bbox.y1 + transform.f

                bbox = BBox(
                    x0=min(x0, x1), y0=min(y0, y1), x1=max(x0, x1), y1=max(y0, y1)
                )
            else:
                bbox = get_bounding_box_svgelements(node)
        else:
            bbox = get_bounding_box_svgelements(node)
    else:
        bbox = get_bounding_box_svgelements(node)

    node_clip = getattr(node, "clip_path", None)
    if node_clip is not None and (
        clip_box is None or (clip_box.height > 0 and clip_box.width > 0)
    ):
        if len(node_clip) > 1:
            raise NotImplementedError("Multiple clip paths not supported yet")

        node_clip_box = node_clip[0].bbox()
        if clip_box is None:
            clip_box = BBox(
                x0=node_clip_box[0],
                y0=node_clip_box[1],
                x1=node_clip_box[2],
                y1=node_clip_box[3],
            )
        else:
            clip_box = clip_box.intersect(
                BBox(
                    x0=node_clip_box[0],
                    y0=node_clip_box[1],
                    x1=node_clip_box[2],
                    y1=node_clip_box[3],
                )
            )

            if clip_box is None:
                clip_box = BBox(x0=0, y0=0, x1=0, y1=0)

    if clip_box is not None and (clip_box.height <= 0.001 or clip_box.width <= 0.001):
        # Fully clipped out
        return []

    is_container = _is_container(node)

    results = []

    if is_container:
        # Special case in case the top level is a single group
        num_non_emprty_children = sum([1 for c in node if not detect_empty_groups(c)])
        if keep_top_level_groups and num_non_emprty_children == 1:
            child = next(iter(node))
            if isinstance(child, (svgelements.Group)):
                return walk_and_clip(
                    child, clip_box, keep_top_level_groups=True, font_cache=font_cache
                )

        for c_ind, child in enumerate(node):
            if keep_top_level_groups:
                group_name = f"group_{c_ind}"

                if isinstance(
                    child, (svgelements.Use, svgelements.Text, svgelements.Title)
                ):
                    # Ignore top level text
                    continue

            child_result = walk_and_clip(
                child,
                clip_box,
                keep_top_level_groups=False,
                group_name=group_name,
                font_cache=font_cache,
            )
            results.extend(child_result)

        if all([fig["type"] == "Text" for fig in results]):
            if keep_top_level_groups:
                # Ignore this group if it has no children with a type
                results = []
            else:
                # Treat as one box
                relevant_boxes = [
                    fig["clip_box"] for fig in results if fig["clip_box"] is not None
                ]
                if len(relevant_boxes) > 0:
                    combined_box = BBox.union_boxes(relevant_boxes)
                    results = [
                        {
                            "type": "Text",
                            "clip_box": combined_box,
                            "dpi": None,
                            "group": group_name,
                        }
                    ]
                else:
                    results = []
    else:
        if clip_box is not None:
            output_clip_box = (
                intersect_box_with_clips(bbox, [clip_box]) if bbox is not None else None
            )
        else:
            output_clip_box = bbox.copy() if bbox is not None else None
        if output_clip_box is not None:
            assert bbox is not None
            results.append(
                {
                    "type": node_type,
                    "clip_box": output_clip_box,
                    "dpi": None
                    if node_type != "Image"
                    else round(72 * node.height / bbox.height),  # pyright: ignore[reportAttributeAccessIssue,reportOperatorIssue]
                    "group": group_name,
                }
            )

    return results


def get_figures_from_page(
    doc: pymupdf.Document, page_num: int
) -> dict[str, list[dict[str, Any]]]:
    """
    Extract figures from a PDF page using its SVG representation.

    Each entry returned by :func:`walk_and_clip` is classified into ``"path"``
    or ``"image"`` and trivially small entries (less than half a point in
    either dimension) are filtered out.

    Args:
        doc (pymupdf.Document): The PDF document.
        page_num (int): Zero-based page index to extract figures from.

    Returns:
        dict[str, list[dict[str, Any]]]: Dictionary with keys ``"path"`` and
            ``"image"``, each containing a list of figure descriptions in
            the format produced by :func:`walk_and_clip`.

    """
    pdf_page = doc[page_num]
    svg_text = pdf_page.get_svg_image(text_as_path=True)
    parsed_svg_page = svgelements.SVG.parse(StringIO(svg_text))

    detected_figures = {
        "path": [],
        "image": [],
    }

    walked_figs = walk_and_clip(parsed_svg_page)

    for fig in walked_figs:
        # Classify text boxes as paths
        type_name = "path" if fig["type"] != "Image" else "image"

        fig_width = fig["clip_box"].x1 - fig["clip_box"].x0
        fig_height = fig["clip_box"].y1 - fig["clip_box"].y0
        if fig_width < 0.5 or fig_height < 0.5:  # about 2 pixels in
            continue
        detected_figures[type_name].append(fig)
    return detected_figures
