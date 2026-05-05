"""
MuPDF-based :class:`Page` implementation.

Uses :func:`pmo_parser.svg_utils.get_figures_from_page` to derive figure
candidates from the SVG representation of each page, then merges nearby
candidates and rejects boxes that fall in the page margins.
"""

import math
import warnings
from typing import Any, TypeGuard

import numpy as np
import pymupdf

from pmo_parser.bounding_boxes import BBox, ImageBBox, TextBox
from pmo_parser.page import ImageCluster, Page, PageBlocks
from pmo_parser.renderer import render_page
from pmo_parser.svg_utils import get_figures_from_page


def is_int_cluster(cluster: Any) -> TypeGuard[list[list[int]]]:
    """
    Type guard for list of list of ints.

    Args:
        cluster (Any): Object to check.

    Returns:
        bool: ``True`` if ``cluster`` is a list of list of ints, ``False``
            otherwise.

    """
    return (
        isinstance(cluster, list)
        and all(isinstance(sublist, list) for sublist in cluster)
        and all(isinstance(item, int) for sublist in cluster for item in sublist)
    )


class MuPDFPage(Page):
    """
    :class:`Page` backend that detects figures from the page's SVG.

    Inherits all attributes from :class:`Page`. Figure clusters and
    remaining paths are populated from the SVG walker output.
    """

    def __init__(
        self,
        document: pymupdf.Document,
        page_num: int,
        dpi: int = 300,
        always_create_screenshots: bool = False,
        no_render_mode: bool = False,
    ):
        """
        Initialize the page.

        Args:
            document (pymupdf.Document): Already opened PDF document.
            page_num (int): Zero-based page index to parse.
            dpi (int, optional): DPI used for any rendering performed during
                parsing. Defaults to ``300``.
            always_create_screenshots (bool, optional): When ``True``, page
                screenshots are rendered eagerly. Defaults to ``False``.
            no_render_mode (bool, optional): When ``True``, all screenshot
                rendering is skipped (useful for unit tests and headless
                pipelines). Defaults to ``False``.

        """
        super().__init__(
            document,
            page_num,
            dpi=dpi,
            always_create_screenshots=always_create_screenshots,
            no_render_mode=no_render_mode,
        )

    def _parse_figures_from_svg(
        self,
        doc,
        x_threshold_path=10,
        y_threshold_path=10,
        x_threshold_image=1,
        y_threshold_image=1,
        min_area=100,
        dpi=300,
        merge_images=True,
        merge_paths=True,
        no_render_mode: bool = False,
    ) -> dict[str, list[ImageBBox]]:
        """
        Parse figures from the page SVG and merge nearby candidates.

        Args:
            doc (pymupdf.Document): Already opened PDF document.
            x_threshold_path (float, optional): Maximum horizontal gap (in
                points) below which two path figures are merged. Defaults to
                ``10``.
            y_threshold_path (float, optional): Maximum vertical gap (in
                points) below which two path figures are merged. Defaults to
                ``10``.
            x_threshold_image (float, optional): Maximum horizontal gap below
                which two image figures are merged. Defaults to ``1``.
            y_threshold_image (float, optional): Maximum vertical gap below
                which two image figures are merged. Defaults to ``1``.
            min_area (float, optional): Minimum area (in square points) below
                which an image figure is discarded. Defaults to ``100``.
            dpi (int, optional): DPI used when rendering the page. Defaults
                to ``300``.
            merge_images (bool, optional): Whether to merge nearby image
                figures. Defaults to ``True``.
            merge_paths (bool, optional): Whether to merge nearby path
                figures. Defaults to ``True``.
            no_render_mode (bool, optional): When ``True``, no screenshots
                are rendered. Defaults to ``False``.

        Returns:
            dict[str, list[ImageBBox]]: Dictionary with keys ``"path"`` and
                ``"image"`` mapping to the merged figure boxes of each type.

        """
        detected_figures = get_figures_from_page(doc, self.page_num)

        group_names = {
            fig["group"]
            for figure_type, figure_list in detected_figures.items()
            for fig in figure_list
            if fig.get("group") is not None
        }

        # First merge by group
        in_group_distance_threshold = 50
        group_figures = {}
        remaining_figures = {
            "image": [
                fig for fig in detected_figures["image"] if fig.get("group") is None
            ],
            "path": [
                fig for fig in detected_figures["path"] if fig.get("group") is None
            ],
        }
        for group in group_names:
            relevant_figures = [
                fig
                for figure_list in detected_figures.values()
                for fig in figure_list
                if fig.get("group") == group
            ]
            if len(relevant_figures) <= 1:
                group_figures[group] = relevant_figures
                continue

            dpi = max([fig.get("dpi", 0) or 0 for fig in relevant_figures] + [dpi])
            new_clip_box = BBox.union_boxes(
                [fig["clip_box"] for fig in relevant_figures]
            )
            new_figure = {
                "type": "Image",
                "clip_box": new_clip_box,
                "dpi": dpi
                if any([fig["type"] == "Image" for fig in relevant_figures])
                else None,
            }
            group_figures[group] = [new_figure]

        # Merge figures
        for figure_type in remaining_figures:  # noqa: PLC0206
            if figure_type == "image" and not merge_images:
                continue
            if figure_type == "path" and not merge_paths:
                continue

            x_threshold = (
                x_threshold_image if figure_type == "image" else x_threshold_path
            )
            y_threshold = (
                y_threshold_image if figure_type == "image" else y_threshold_path
            )

            if len(remaining_figures[figure_type]) == 0:
                continue
            merged_figures_changed = True
            calculate_dpi = remaining_figures[figure_type][0]["type"] == "Image"

            while merged_figures_changed:
                merged_figures_changed = False
                figure_list = remaining_figures[figure_type]
                for figure_index, fig in enumerate(figure_list):
                    figure_group = fig.get("group", None)
                    for other_index, other_fig in enumerate(figure_list):
                        if figure_index >= other_index or other_fig is None:
                            continue
                        other_figure_group = other_fig.get("group", None)
                        has_equal_groups = (
                            figure_group is not None
                            and other_figure_group is not None
                            and figure_group == other_figure_group
                        )
                        bbox_x_dist, bbox_y_dist = fig["clip_box"].distance_vector(
                            other_fig["clip_box"]
                        )

                        if (
                            bbox_x_dist < x_threshold
                            and bbox_y_dist < y_threshold
                            and not has_equal_groups
                        ) or (
                            bbox_x_dist < in_group_distance_threshold
                            and bbox_y_dist < in_group_distance_threshold
                            and has_equal_groups
                        ):
                            # Merge
                            new_clip_box = fig["clip_box"].union(other_fig["clip_box"])

                            new_figure = {
                                "type": fig["type"],
                                "clip_box": new_clip_box,
                                "dpi": None
                                if not calculate_dpi
                                else max(fig["dpi"], other_fig["dpi"]),
                            }
                            figure_list[figure_index] = new_figure
                            figure_list[other_index] = None  # pyright: ignore[reportArgumentType, reportCallIssue]
                            merged_figures_changed = True

                    if merged_figures_changed:
                        break
                if merged_figures_changed:
                    remaining_figures[figure_type] = [
                        fig for fig in figure_list if fig is not None
                    ]

        for group, figs in group_figures.items():
            if len(figs) == 0:
                continue
            figure_type = "path" if figs[0]["type"].lower() != "image" else "image"
            remaining_figures[figure_type].extend(figs)

        # Convert to output format
        output_figures = {}
        for figure_type, figure_list in remaining_figures.items():
            output_figures[figure_type] = []
            for fig in figure_list:
                if fig["clip_box"].area < min_area and figure_type == "image":
                    continue

                if (
                    fig["clip_box"].x0 < 0.05 * self.page_width
                    or fig["clip_box"].x1 > 0.97 * self.page_width
                ):
                    # Ignore figures in the margin
                    continue

                if (
                    fig["clip_box"].y0 < 0.02 * self.page_height
                    or fig["clip_box"].y1 > 0.95 * self.page_height
                ):
                    # Ignore figures in the margin
                    continue

                if fig["type"] == "Text":
                    # Ignore text boxes
                    continue
                current_dpi = max(dpi, fig.get("dpi", dpi) or dpi)

                rendered_figure = None

                image_box = ImageBBox.from_bbox(fig["clip_box"], rendered_figure)
                image_box._virtual_dpi = int(round(current_dpi))
                image_box.calc_virtual_size()
                output_figures[figure_type].append(image_box)

            images_have_changed = True
            while images_have_changed and figure_type == "image":
                images_have_changed = False
                for i in range(len(output_figures[figure_type])):
                    if output_figures[figure_type][i] is None:
                        continue
                    for j in range(i + 1, len(output_figures[figure_type])):
                        if output_figures[figure_type][j] is None:
                            continue

                        overlap_ratio = output_figures[figure_type][i].overlap_ratio(
                            output_figures[figure_type][j]
                        )

                        if overlap_ratio > 0.1:
                            new_box = output_figures[figure_type][i].union(
                                output_figures[figure_type][j]
                            )
                            # Create new screenshot
                            current_dpi = max(
                                dpi,
                                output_figures[figure_type][i].dpi,
                                output_figures[figure_type][j].dpi,
                            )
                            new_image_box = ImageBBox.from_bbox(new_box, None)
                            new_image_box._virtual_dpi = current_dpi
                            new_image_box.calc_virtual_size()
                            output_figures[figure_type][i] = new_image_box
                            output_figures[figure_type][j] = None
                            images_have_changed = True
                if images_have_changed:
                    output_figures[figure_type] = [
                        fig for fig in output_figures[figure_type] if fig is not None
                    ]

            # Sort by y0, then x0
            output_figures[figure_type] = sorted(
                output_figures[figure_type], key=lambda x: (x.y0, x.x0)
            )

        if not no_render_mode:
            # Render all images
            for figure_type in output_figures:  # noqa: PLC0206
                for i in range(len(output_figures[figure_type])):
                    if output_figures[figure_type][i].image is not None:
                        continue
                    current_dpi = output_figures[figure_type][i].dpi

                    rendered_figure = render_page(
                        doc,
                        self.page_num,
                        dpi=int(current_dpi),
                        bbox=output_figures[figure_type][i],
                    )
                    output_figures[figure_type][i].image = rendered_figure
        return output_figures

    def parse_blocks(
        self,
        doc: pymupdf.Document,
        dpi=300,
        always_create_screenshots: bool = False,
        no_render_mode: bool = False,
        **kwargs,
    ) -> PageBlocks:
        """
        Run SVG-based figure detection on the page.

        Reads the words via MuPDF, then calls :meth:`_parse_figures_from_svg`
        and post-processes the resulting figure candidates into
        :class:`ImageBBox` objects, image clusters and remaining vector
        paths.

        Args:
            doc (pymupdf.Document): Already opened PDF document.
            dpi (int, optional): DPI used when rendering the page. Defaults
                to ``300``.
            always_create_screenshots (bool, optional): When ``True``, page
                screenshots are rendered eagerly. Defaults to ``False``.
            no_render_mode (bool, optional): When ``True``, no screenshots
                are rendered. Defaults to ``False``.
            **kwargs: Unrecognized keyword arguments produce a warning.

        Returns:
            PageBlocks: Parsed blocks from the page as a single object with attributes
                ``(page_texts, page_figures, figure_clusters,
                remaining_paths)`` matching the same-named attributes on
                :class:`Page`.

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

        if not no_render_mode:
            page_drawing = None
        else:
            page_drawing = render_page(doc, self.page_num, dpi=dpi)

        detected_svg_figures = self._parse_figures_from_svg(
            doc, dpi=dpi, no_render_mode=no_render_mode
        )
        page_images = [
            im
            for im in detected_svg_figures.get("image", [])
            if im.width > 20 and im.height > 20
        ]

        # Filter images in the header and footer
        page_images = [
            im
            for im in page_images
            if im.y0 > 0.05 * self.page_height
            and im.y1 < 0.95 * self.page_height
            or (
                im.width > 0.3 * self.page_width
                and im.height > 0.2 * self.page_height
                and im.x0 > 0.05 * self.page_width  # not in the margin
                and im.x1 < 0.95 * self.page_width
            )
        ]

        joined_rectangles = detected_svg_figures.get("path", [])

        # Replace images with screenshots if there is text in the image
        page_figures = []
        added_paths = []

        for im in page_images:
            has_overlap = False

            for assignment in page_assignments:
                assignment_box = TextBox.union_boxes(assignment)
                if assignment_box.overlap_ratio(im) > 0.05:
                    has_overlap = True
                    break

            image_box = im.get_bbox()
            has_changed = True

            img_dpi = im.dpi

            checked_overlap = False
            while has_changed:
                has_changed = False
                for rect_ind, p in enumerate(joined_rectangles):
                    if rect_ind in added_paths and checked_overlap:
                        continue
                    if p.overlap_ratio(image_box) > 0.05:
                        has_overlap = True

                        if rect_ind not in added_paths and image_box.distance(p) < 5:
                            image_box = image_box.union(p)
                            added_paths.append(rect_ind)
                            has_changed = True
                            has_overlap = True

                im.x0 = image_box.x0
                im.y0 = image_box.y0
                im.x1 = image_box.x1
                im.y1 = image_box.y1

                checked_overlap = True

            if not no_render_mode and (has_overlap or always_create_screenshots):
                # Use screenshot instead
                if abs(img_dpi - dpi) < 2 and page_drawing is not None:
                    hq_page_rendering = page_drawing.copy()
                else:
                    # Rerender
                    hq_page_rendering = render_page(
                        doc, self.page_num, dpi=int(img_dpi)
                    )

                hq_png_ratio = float(hq_page_rendering.size[1]) / self.page_height
                block = ImageBBox.from_bbox(
                    im,
                    hq_page_rendering.crop(
                        (
                            int(hq_png_ratio * im.x0),
                            int(hq_png_ratio * im.y0),
                            math.ceil(hq_png_ratio * im.x1),
                            math.ceil(hq_png_ratio * im.y1),
                        )
                    ),
                )
                page_figures.append(block)
            else:
                # Use original image
                page_figures.append(im)

        # Join images that are 90% overlapped
        has_changed = True
        while has_changed:
            has_changed = False

            area_sorted_figures = sorted(
                range(len(page_figures)), key=lambda k: page_figures[k].area
            )
            for seq_ind, i in enumerate(area_sorted_figures):
                if page_figures[i] is None:
                    continue
                for j in area_sorted_figures[seq_ind + 1 :]:
                    if page_figures[j] is None:
                        continue

                    overlap_ratio = page_figures[i].overlap_ratio(page_figures[j])

                    if overlap_ratio > 0.9:
                        new_box = page_figures[i].union(page_figures[j])
                        # Create new screenshot
                        current_dpi = max(dpi, page_figures[i].dpi, page_figures[j].dpi)

                        if not no_render_mode:
                            new_image_box = ImageBBox.from_bbox(new_box, None)
                            new_image_box._virtual_dpi = current_dpi
                            new_image_box.calc_virtual_size()
                        else:
                            rendered_figure = render_page(
                                doc,
                                self.page_num,
                                dpi=int(current_dpi),
                                bbox=new_box,
                            )
                            new_image_box = ImageBBox.from_bbox(
                                new_box, rendered_figure
                            )
                        page_figures[i] = new_image_box
                        page_figures[j] = None
                        has_changed = True
            if has_changed:
                page_figures = [fig for fig in page_figures if fig is not None]

        # Find potential image clusters = Multi panel images
        clusters = []
        if len(page_figures) > 0:
            high_const = 100000
            image_distances = np.array(
                [
                    [
                        im1.distance(im2) if i != j else high_const
                        for j, im2 in enumerate(page_figures)
                    ]
                    for i, im1 in enumerate(page_figures)
                ]
            )

            image_distances_tri = (
                image_distances + np.tri(len(page_figures), k=-1) * high_const
            )

            clusters = np.argwhere(
                image_distances_tri
                < np.minimum(1.5 * image_distances.min(axis=0), 37).reshape(
                    [len(page_figures), 1]
                )
            ).tolist()

            if not is_int_cluster(clusters):
                raise ValueError("Expected clusters to be list of list of ints")

            found_clusters = True
            while found_clusters:
                found_clusters = False
                new_clusters = []

                # join two clusters if they have one index in common
                for i in range(len(clusters)):
                    c_cluster = clusters[i]

                    if len(c_cluster) == 0:
                        continue

                    for j in range(i + 1, len(clusters)):
                        other_cluster = clusters[j]
                        if len(set(c_cluster) & set(other_cluster)) > 0:
                            c_cluster = list(set(c_cluster) | set(other_cluster))
                            found_clusters = True
                            clusters[j] = []

                    new_clusters.append(c_cluster)
                clusters = [c for c in new_clusters if len(c) > 1]

            output_clusters = []

            for c in clusters:
                # ImageCluster
                max_dpi = max([page_figures[i].dpi for i in c])
                if no_render_mode:
                    output_clusters.append(
                        ImageCluster(
                            image_ids=c,
                            screenshot=None,
                        )
                    )
                else:
                    if abs(max_dpi - dpi) < 2 and page_drawing is not None:
                        hq_page_rendering = page_drawing.copy()
                    else:
                        # Rerender
                        hq_page_rendering = render_page(doc, self.page_num, dpi=max_dpi)
                    hq_png_ratio = float(hq_page_rendering.size[1]) / self.page_height
                    bbox_union = BBox.union_boxes([page_figures[i] for i in c])
                    screenshot = hq_page_rendering.crop(
                        (
                            int(hq_png_ratio * bbox_union.x0),
                            int(hq_png_ratio * bbox_union.y0),
                            math.ceil(hq_png_ratio * bbox_union.x1),
                            math.ceil(hq_png_ratio * bbox_union.y1),
                        )
                    )
                    output_clusters.append(
                        ImageCluster(
                            image_ids=c,
                            screenshot=screenshot,
                        )
                    )
            clusters = output_clusters

        remaining_blocks = [
            joined_rectangles[i].get_bbox()
            for i in range(len(joined_rectangles))
            if i not in added_paths
        ]

        return PageBlocks(
            page_texts=page_assignments,
            page_figures=page_figures,
            figure_clusters=clusters,
            remaining_paths=remaining_blocks,
        )
