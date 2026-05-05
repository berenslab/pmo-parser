"""Core caption-detection algorithm: turn a PDF into :class:`OutputFigure` objects."""

import io
import multiprocessing
import re
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, TypeGuard

import numpy as np
from tqdm.auto import tqdm

from pmo_parser.bounding_boxes import BBox, TextBox
from pmo_parser.const import FIGURE_ID_REGEX_PATTERN
from pmo_parser.figure import OutputFigure
from pmo_parser.layout.layout_registry import LAYOUT_REGISTRY
from pmo_parser.page import Page, PageText, convert_to_string
from pmo_parser.page.mupdf_page import MuPDFPage
from pmo_parser.renderer import render_page
from pmo_parser.util import DocumentWrapper


def is_int_list(lst) -> TypeGuard[list[int]]:
    """Check if a list is a list of integers."""
    return isinstance(lst, list) and all(isinstance(i, int) for i in lst)


def is_list_of_int_lists(lst) -> TypeGuard[list[list[int]]]:
    """Check if a list is a list of lists of integers."""
    return isinstance(lst, list) and all(is_int_list(i) for i in lst)


def is_list_of_float_lists(lst) -> TypeGuard[list[list[float]]]:
    """Check if a list is a list of lists of floats."""
    return isinstance(lst, list) and all(
        isinstance(i, list) and all(isinstance(j, float) for j in i) for i in lst
    )


@dataclass
class CaptionResult:
    """
    Return value of :func:`find_captions`.

    Attributes:
        figure_caption_indices: For each figure, the indices into
            ``page.page_texts`` of its assigned caption blocks.
        figure_caption_scores: For each figure, the caption-likelihood score
            of each assigned caption block.
        cluster_caption_indices: For each figure cluster, the shared caption
            text-block indices (empty list when no compound image was found).
        cluster_caption_scores: Per-cluster caption-likelihood scores
            matching ``cluster_caption_indices``.
        score_matrix: Full ``(num_figures × num_texts)`` caption-likelihood
            score matrix as a nested list (figure-major order).

    """

    figure_caption_indices: Sequence[Sequence[int]] = field(default_factory=list)
    figure_caption_scores: Sequence[Sequence[float]] = field(default_factory=list)
    cluster_caption_indices: Sequence[Sequence[int]] = field(default_factory=list)
    cluster_caption_scores: Sequence[Sequence[float]] = field(default_factory=list)
    score_matrix: Sequence[Sequence[float]] = field(default_factory=list)


def find_captions(
    page: Page,
) -> CaptionResult:
    """
    Score and assign caption text blocks to figures on a page.

    For every text block on the page a "caption likelihood" score is computed
    from cues such as a leading "Figure"/"Fig" word, distance to each figure,
    number of intervening text blocks, and whether the block sits below the
    figure. The score matrix is then thresholded to assign zero or more
    captions to each figure.

    Args:
        page (Page): Already parsed page providing figures and text blocks.

    Returns:
        CaptionResult: Tuple of
            ``(assigned_indices, scores, score_matrices, excluded_images)``;
            see the inline structure for details. The score matrices triple
            contains the per-figure / per-text caption likelihood, the
            distance matrix and the intersection-area matrix used by later
            stages.

    """
    if len(page.page_figures) == 0 or len(page.page_texts) == 0:
        return CaptionResult()

    caption_likelihood_score = []

    image_distances = np.zeros((len(page.page_figures), len(page.page_texts)))
    max_image_intersection_areas = np.zeros(
        (len(page.page_figures), len(page.page_texts))
    )
    num_captions_between = np.zeros((len(page.page_figures), len(page.page_texts)))
    below_image = np.zeros((len(page.page_figures), len(page.page_texts)))

    text_boxes = [TextBox.union_boxes(t) for t in page.page_texts]

    for i, text in enumerate(page.get_string_texts()):
        caption_likelihood_score.append(0)
        if text.lower().startswith("figure") or text.lower().startswith("fig"):
            caption_likelihood_score[-1] += 2
        if "figure" in text.lower() or "fig" in text.lower():
            caption_likelihood_score[-1] += 1

        bbox = text_boxes[i]

        for im_ind, im in enumerate(page.page_figures):
            image_distances[im_ind, i] = im.distance(bbox)

            inter_box = bbox.get_intermediate_box(im)
            num_boxes_between = 0
            if inter_box is None:
                if bbox.intersect(im) is not None:
                    # Overlapping boxes
                    num_boxes_between = 0
                else:
                    # Diagonal set to higher number
                    num_boxes_between = 3
            else:
                # Count other boxes that are in between
                for j, other_box in enumerate(text_boxes):
                    if j == i:
                        continue
                    if other_box.intersect(inter_box) is not None:
                        image_intersection = other_box.intersect(im)
                        if (
                            image_intersection is None
                            or image_intersection.area / other_box.area < 0.1
                        ):
                            num_boxes_between += 1

            num_captions_between[im_ind, i] = (
                num_boxes_between if num_boxes_between > 1 else 2 * num_boxes_between
            )

        min_dist = image_distances[:, i].min()
        mean_dist = image_distances[:, i].mean()
        if min_dist == 0:
            caption_likelihood_score[-1] += 1.5
        elif min_dist > 300:
            caption_likelihood_score[-1] -= 0.5
        else:
            caption_likelihood_score[-1] += (
                min(0.025 * mean_dist / min_dist, 1.5)
                if len(page.page_figures) > 1
                else 10 / min_dist
            )

        # Extend the image bounding box to the top of the page and check
        # for intersection.
        top_intersections = [
            bbox.intersect(BBox(x0=im.x0, y0=0, x1=im.x1, y1=im.y1))
            for im in page.page_figures
        ]
        # Extend the image bounding box to the bottom of the page and check
        # for intersection.
        bottom_intersections = [
            bbox.intersect(BBox(x0=im.x0, y0=im.y0, x1=im.x1, y1=page.page_height))
            for im in page.page_figures
        ]
        # Extend the image bounding box to the left of the page and check
        # for intersection.
        left_intersections = [
            bbox.intersect(BBox(x0=0, y0=im.y0, x1=im.x1, y1=im.y1))
            for im in page.page_figures
        ]
        # Extend the image bounding box to the right of the page and check
        # for intersection.
        right_intersections = [
            bbox.intersect(BBox(x0=im.x0, y0=im.y0, x1=page.page_width, y1=im.y1))
            for im in page.page_figures
        ]

        max_image_intersection_areas[:, i] = [
            max(
                top.width / im.width if top is not None else 0,
                bottom.width / im.width if bottom is not None else 0,
                left.height / im.height if left is not None else 0,
                right.height / im.height if right is not None else 0,
            )
            for top, bottom, left, right, im in zip(
                top_intersections,
                bottom_intersections,
                left_intersections,
                right_intersections,
                page.page_figures,
            )
        ]

        if caption_likelihood_score[-1] > 2:
            below_image[:, i] = [
                1 if bottom is not None else 0 for bottom in bottom_intersections
            ]

    caption_likelihood_score_per_image = np.array(
        [
            [
                caption_likelihood_score[i]
                - num_captions_between[j][i]
                + below_image[j][i]
                - 0.2 * (1 - max_image_intersection_areas[j][i])
                for i in range(len(caption_likelihood_score))
            ]
            for j in range(len(page.page_figures))
        ]
    )

    caption_likelihood_score_per_image_list = (
        caption_likelihood_score_per_image.tolist()
    )
    if not is_list_of_float_lists(caption_likelihood_score_per_image_list):
        raise TypeError(
            f"""Expected caption_likelihood_score_per_image to be a list of lists of \
floats, got {caption_likelihood_score_per_image_list}"""
        )

    assigned_indices = [
        np.where(caption_likelihood_score_per_image[j] > 3)[0].tolist()
        for j in range(len(page.page_figures))
    ]

    num_caption_textboxes = (caption_likelihood_score_per_image.max(axis=0) > 3).sum()

    # Exclude images where the caption likelihood score is very low
    excluded_images = np.where(caption_likelihood_score_per_image.max(axis=1) < 2)[
        0
    ].tolist()

    if not is_int_list(excluded_images):
        raise TypeError(
            f"Expected excluded_images to be a list of integers, got {excluded_images}"
        )

    if not is_list_of_int_lists(assigned_indices):
        raise TypeError(
            f"Expected assigned_indices to be list[list[int]], got {assigned_indices}"
        )

    if num_caption_textboxes == (len(page.page_figures) - len(excluded_images)):
        sub_list_lengths = [len(sub_list) for sub_list in assigned_indices]
        if sum(sub_list_lengths) == (
            len(page.page_figures) - len(excluded_images)
        ) and all([length <= 1 for length in sub_list_lengths]):
            # Found exactly one match for each caption
            caption_distances = [
                image_distances[i, assigned_indices[i][0]]
                for i in range(len(page.page_figures))
                if i not in excluded_images
            ]
            if (np.array(caption_distances) < 50).all():
                # All captions are very close to their assigned image
                return CaptionResult(
                    figure_caption_indices=assigned_indices,
                    figure_caption_scores=[
                        [
                            float(caption_likelihood_score_per_image[im, c])
                            for c in assigned_indices[im]
                        ]
                        for im in range(len(page.page_figures))
                    ],
                    cluster_caption_indices=[list() for _ in page.figure_clusters],
                    cluster_caption_scores=[list() for _ in page.figure_clusters],
                    score_matrix=caption_likelihood_score_per_image_list,
                )
        # Found a definite caption for each image assign closest
        captions = sorted({i for sub_list in assigned_indices for i in sub_list})

        output_captions = []
        distances_in_reading_order = []

        seq_index = 0
        for i in range(len(page.page_figures)):
            if i in excluded_images:
                output_captions.append([])
                distances_in_reading_order.append(0)
                continue
            current_caption = captions[seq_index]
            distances_in_reading_order.append(image_distances[i, current_caption])
            output_captions.append([current_caption])
            seq_index += 1

        if (np.array(distances_in_reading_order) < 50).all():
            return CaptionResult(
                figure_caption_indices=output_captions,
                figure_caption_scores=[
                    [
                        float(caption_likelihood_score_per_image[im, c])
                        for c in output_captions[im]
                    ]
                    for im in range(len(page.page_figures))
                ],
                cluster_caption_indices=[list() for _ in page.figure_clusters],
                cluster_caption_scores=[list() for _ in page.figure_clusters],
                score_matrix=caption_likelihood_score_per_image_list,
            )
        elif (np.array(distances_in_reading_order) < 200).all():
            # Return captions ordered by distance
            # Go through images in reading order and assign closest caption
            output_captions: list[list[int]] = []
            for i in range(len(page.page_figures)):
                if i in excluded_images:
                    output_captions.append([])
                    continue
                assigned_captions = [c for o in output_captions for c in o]
                current_distances = image_distances[i]
                current_captions = np.argsort(current_distances).tolist()

                if not is_int_list(current_captions):
                    raise TypeError(
                        "Expected current_captions to be list[int], "
                        f"got {current_captions}"
                    )

                current_captions = [
                    c
                    for c in current_captions
                    if c in captions and c not in assigned_captions
                ]
                if len(current_captions) == 0:
                    output_captions.append([])
                    continue
                output_captions.append([current_captions[0]])
            return CaptionResult(
                figure_caption_indices=output_captions,
                figure_caption_scores=[
                    [
                        float(caption_likelihood_score_per_image[im, c])
                        for c in output_captions[im]
                    ]
                    for im in range(len(page.page_figures))
                ],
                cluster_caption_indices=[list() for _ in page.figure_clusters],
                cluster_caption_scores=[list() for _ in page.figure_clusters],
                score_matrix=caption_likelihood_score_per_image_list,
            )

    captions = []

    distances_between_images = [
        [im.distance(im2) for im2 in page.page_figures] for im in page.page_figures
    ]
    # Score below 2 = instant reject
    for figure_index in range(len(page.page_figures)):
        if figure_index in excluded_images:
            captions.append([])
            continue
        captions.append(
            sorted(
                [
                    i
                    for i in range(len(page.page_texts))
                    if caption_likelihood_score_per_image[figure_index][i] >= 2
                ],
                key=lambda x: -caption_likelihood_score_per_image[figure_index][x],
            )
        )

    inverted_captions = [
        [im for im in range(len(page.page_figures)) if i in captions[im]]
        for i in range(len(page.page_texts))
    ]

    # Decouple double assigned captions and apply distance threshold.
    # If the distance to the other image with the same caption is small,
    # assume compound image.
    new_captions = captions.copy()
    for im in range(len(page.page_figures)):
        current_captions = []
        for cap in new_captions[im]:
            if image_distances[im, cap] > 50:
                # Check if im is close to other image that points to cap
                found_close_image = False
                for im2 in inverted_captions[cap]:
                    if im == im2:
                        continue

                    dist_vec = page.page_figures[im].distance_vector(
                        page.page_figures[im2]
                    )
                    if (
                        distances_between_images[im][im2] < 10
                        or (dist_vec[0] < 2 and distances_between_images[im][im2] < 15)
                        or (dist_vec[1] < 2 and distances_between_images[im][im2] < 15)
                    ):
                        found_close_image = True
                        break

                if not found_close_image:
                    continue  # Remove from captions
            current_captions.append(cap)
        new_captions[im] = current_captions

    # Search for image clusters. An image cluster is where two or more
    # images are close to each other but have separate captions.
    captions = new_captions
    inverted_captions = [
        [im for im in range(len(page.page_figures)) if i in captions[im]]
        for i in range(len(page.page_texts))
    ]
    new_captions = []

    for im in range(len(page.page_figures)):
        if im in excluded_images:
            new_captions.append([])
            continue

        # Check if two captions point to this image
        if len(captions[im]) <= 1:
            new_captions.append(captions[im])
            continue

        # Captions are sorted by likelihood
        # Check if caption is also part of another image
        points_to_multiple_images = [
            len(inverted_captions[c]) > 1 for c in captions[im]
        ]
        if not any(points_to_multiple_images):
            new_captions.append(captions[im])
            continue

        # This shows that the other images only have one caption
        is_uniquely_assigned = [
            any([len(captions[im1]) == 1 for im1 in inverted_captions[c] if im != im1])
            for c in captions[im]
        ]

        # Remove captions that have been uniquely assigned
        caps = [c for c, u in zip(captions[im], is_uniquely_assigned) if not u]

        # Check if the number of captions is still more than one
        if len(caps) <= 1:
            new_captions.append(caps)
            continue

        new_captions.append(caps)

    captions = new_captions

    # Check if we can use image clusters to assign captions
    output_cluster_captions = []
    cluster_caption_likelihood = []
    for cluster in page.figure_clusters:
        cluster_captions = [captions[i] for i in cluster.image_ids]

        # Check if all images in the cluster have the same caption
        if all([len(c) <= 1 for c in cluster_captions]):
            assigned_captions = list({c[0] for c in cluster_captions if len(c) > 0})
            if len(assigned_captions) == 1:
                # Found compound image
                for i in cluster.image_ids:
                    captions[i] = [assigned_captions[0]]
                output_cluster_captions.append(assigned_captions)
                cluster_caption_likelihood.append(
                    [
                        max(
                            [
                                float(
                                    caption_likelihood_score_per_image[
                                        i, assigned_captions[0]
                                    ]
                                )
                                for i in cluster.image_ids
                            ]
                        )
                    ]
                )
                continue
        output_cluster_captions.append([])
        cluster_caption_likelihood.append([])
    return CaptionResult(
        figure_caption_indices=captions,
        figure_caption_scores=[
            [float(caption_likelihood_score_per_image[im, c]) for c in captions[im]]
            for im in range(len(page.page_figures))
        ],
        cluster_caption_indices=output_cluster_captions,
        cluster_caption_scores=cluster_caption_likelihood,
        score_matrix=caption_likelihood_score_per_image_list,
    )


def merge_surrounding_boxes(
    page: Page, captions: Sequence[Sequence[int]]
) -> list[list[PageText]]:
    """
    Merge text blocks adjacent to a caption into the caption itself.

    Walks every text block on the page and attaches it to the closest
    caption when the block sits within roughly two letter widths or heights
    of the caption box. This recovers caption continuations that MuPDF
    splits into separate blocks.

    Args:
        page (Page): Already parsed page providing text blocks.
        captions (list[list[int]]): For each figure, the indices of its
            assigned caption text blocks (output of :func:`find_captions`).

    Returns:
        list[list[PageText]]: For each figure, a list of caption
            blocks; each block is itself a list of :class:`TextBox` objects
            forming a single caption region after merging.

    """
    if len(captions) == 0:
        return []
    caption_set = list({c for caption in captions for c in caption})

    if len(caption_set) == 0:
        return [list() for _ in range(len(captions))]

    caption_boxes = [page.page_texts[c] for c in caption_set]
    set_indices = {caption_set[i]: i for i in range(len(caption_set))}

    page_bboxes = [TextBox.union_boxes(c) for c in page.page_texts]

    assigned_indices = []

    for index, caption_num in enumerate(caption_set):
        current_bbox = TextBox.union_boxes(caption_boxes[index])
        mean_letter_height = sum([w.height for w in caption_boxes[index]]) / len(
            caption_boxes[index]
        )
        mean_letter_width = sum(
            [w.width / len(w.text) for w in caption_boxes[index]]
        ) / len(caption_boxes[index])

        added_captions = []
        for other_index, bbox in enumerate(page_bboxes):
            if other_index in caption_set or other_index in assigned_indices:
                continue

            x_dist, y_dist = current_bbox.distance_vector(bbox)
            if (
                abs(x_dist) < 1.7 * mean_letter_width
                and abs(y_dist) < 0.5 * mean_letter_height
            ):
                added_captions.append(other_index)
                current_bbox = current_bbox.union(bbox)
                assigned_indices.append(other_index)

        # Join new caption blocks in correct order
        caption_boxes[index] = [
            b
            for i in sorted(added_captions + [caption_num])
            for b in page.page_texts[i]
        ]

    return [
        [caption_boxes[set_indices[i]] for i in captions[im]]
        for im in range(len(captions))
    ]


def finalize_figures(
    page: Page,
    pdf_path: str | io.BytesIO,
    captions_per_page: CaptionResult,
    merged_captions: list[list[PageText]],
    no_render_mode: bool = False,
) -> list[OutputFigure]:
    """
    Build the final list of :class:`OutputFigure` objects for a page.

    Combines the figure regions and clusters from ``page`` with the caption
    assignments and merged captions produced by the earlier pipeline stages,
    then renders each figure unless ``no_render_mode`` is set.

    Args:
        page (Page): Already parsed page.
        pdf_path (str | io.BytesIO): PDF source used for rendering.
        captions_per_page (tuple): Output of :func:`find_captions`,
            ``(assigned_indices, scores, score_matrices, excluded_images)``.
        merged_captions (list[list[PageText]]): Output of
            :func:`merge_surrounding_boxes`.
        no_render_mode (bool, optional): When ``True``, no rendering is
            performed and figures are returned without an attached image.
            Defaults to ``False``.

    Returns:
        list[OutputFigure]: One output figure per detected figure (or
            cluster) on the page.

    """
    output_figures = []
    added_cluster = [False for i in range(len(page.figure_clusters))]

    page_bboxes = [TextBox.union_boxes(t) for t in page.page_texts]
    for i, caption_indices in enumerate(captions_per_page.figure_caption_indices):
        scores = captions_per_page.figure_caption_scores[i]
        if len(caption_indices) == 0:
            continue

        cluster_id = page.get_cluster_index(i)

        if (
            cluster_id > -1
            and len(captions_per_page.cluster_caption_indices[cluster_id]) > 0
        ):
            if added_cluster[cluster_id]:
                continue
            added_cluster[cluster_id] = True
            current_cluster = page.figure_clusters[cluster_id]
            cluster_image = current_cluster.screenshot
            cluster_box = BBox.union_boxes(
                [page.page_figures[im_id] for im_id in current_cluster.image_ids]
            )
            cluster_dpi = max(
                [page.page_figures[im_id].dpi for im_id in current_cluster.image_ids]
            )

            current_caption = merged_captions[i]

            # Add boxes between caption and image
            if len(current_caption) == 1:
                # Check for other boxes to add to image
                current_caption = current_caption[0]
                current_caption_box = TextBox.union_boxes(current_caption)

                additional_captions = []
                intermediate_box = current_caption_box.get_intermediate_box(cluster_box)
                if intermediate_box is not None:
                    for j, other_box in enumerate(page_bboxes):
                        if j == current_caption:
                            continue

                        box_intersection = intermediate_box.intersect(other_box)
                        if (
                            box_intersection is not None
                            and box_intersection.area / other_box.area > 0.5
                        ):
                            additional_captions.append(j)

                    additional_drawing_indices = []
                    additional_drawings = []

                    for j, drawing in enumerate(page.remaining_paths):
                        box_intersection = intermediate_box.intersect(drawing)
                        if (
                            box_intersection is not None
                            and box_intersection.area / drawing.area > 0.5
                        ):
                            additional_drawing_indices.append(j)
                            additional_drawings.append(drawing)

                    if len(additional_captions) > 0:
                        cluster_box = cluster_box.union(
                            TextBox.union_boxes(
                                [page_bboxes[c] for c in additional_captions]
                            )
                        )

                    if len(additional_drawings) > 0:
                        page.remaining_paths = [
                            path
                            for k, path in enumerate(page.remaining_paths)
                            if k not in additional_drawing_indices
                        ]
                        cluster_box = cluster_box.union(
                            BBox.union_boxes(additional_drawings)
                        )

                    if (
                        len(additional_captions) > 0 or len(additional_drawings) > 0
                    ) and not no_render_mode:
                        # Render new image
                        cluster_image = render_page(
                            pdf_path,
                            page.page_num,
                            dpi=int(cluster_dpi),
                            bbox=cluster_box,
                        )

            if no_render_mode:
                # TODO Render figure here?
                pass

            out_fig = OutputFigure(
                page=page.page_num,
                figure_bbox=cluster_box,
                captions=[
                    TextBox.from_bbox(
                        TextBox.union_boxes(c),
                        text=convert_to_string(
                            c, page.mean_letter_width, page.mean_letter_height
                        ),
                    )
                    for c in merged_captions[i]
                ],
                caption_scores=captions_per_page.cluster_caption_scores[cluster_id],
                name=None,
                image=cluster_image,
                cluster_bboxes=[
                    page.page_figures[im_id].get_bbox()
                    for im_id in current_cluster.image_ids
                ],
            )

            if no_render_mode:
                out_fig._dpi = cluster_dpi
        else:
            current_caption = merged_captions[i]

            # Add boxes between caption and image
            if len(current_caption) == 1:
                # Check for other boxes to add to image
                current_caption = current_caption[0]
                current_caption_box = TextBox.union_boxes(current_caption)

                additional_captions = []
                intermediate_box = current_caption_box.get_intermediate_box(
                    page.page_figures[i]
                )
                if intermediate_box is not None:
                    for j, other_box in enumerate(page_bboxes):
                        if j == current_caption:
                            continue

                        box_intersection = intermediate_box.intersect(other_box)
                        if (
                            box_intersection is not None
                            and box_intersection.area / other_box.area > 0.5
                        ):
                            additional_captions.append(j)

                    additional_drawing_indices = []
                    additional_drawings = []

                    for j, drawing in enumerate(page.remaining_paths):
                        box_intersection = intermediate_box.intersect(drawing)
                        if (
                            box_intersection is not None
                            and box_intersection.area / drawing.area > 0.5
                        ):
                            additional_drawing_indices.append(j)
                            additional_drawings.append(drawing)

                    image_box = page.page_figures[i].get_bbox()
                    dpi = page.page_figures[i].dpi
                    if len(additional_captions) > 0:
                        image_box = image_box.union(
                            TextBox.union_boxes(
                                [page_bboxes[c] for c in additional_captions]
                            )
                        )

                    if len(additional_drawings) > 0:
                        page.remaining_paths = [
                            path
                            for k, path in enumerate(page.remaining_paths)
                            if k not in additional_drawing_indices
                        ]
                        image_box = image_box.union(
                            BBox.union_boxes(additional_drawings)
                        )

                    if (
                        len(additional_captions) > 0 or len(additional_drawings) > 0
                    ) and not no_render_mode:
                        # Render new image
                        page.page_figures[i].x0 = image_box.x0
                        page.page_figures[i].y0 = image_box.y0
                        page.page_figures[i].x1 = image_box.x1
                        page.page_figures[i].y1 = image_box.y1

                        page.page_figures[i].image = render_page(
                            pdf_path,
                            page.page_num,
                            dpi=int(dpi),
                            bbox=image_box,
                        )
            figure_image = page.page_figures[i].image

            if no_render_mode:
                # Render figure here?
                pass
            out_fig = OutputFigure(
                page=page.page_num,
                figure_bbox=page.page_figures[i].get_bbox(),
                captions=[
                    TextBox.from_bbox(
                        TextBox.union_boxes(c),
                        text=convert_to_string(
                            c, page.mean_letter_width, page.mean_letter_height
                        ),
                    )
                    for c in merged_captions[i]
                ],
                caption_scores=scores,
                name=None,
                image=figure_image,
            )

            if no_render_mode:
                out_fig._dpi = page.page_figures[i].dpi

        # estimate figure ids
        if out_fig.captions is not None and len(out_fig.captions) > 0:
            max_cap = out_fig.captions[0].text
            if max_cap.lower().startswith("figure") or max_cap.lower().startswith(
                "fig"
            ):
                match = re.match(FIGURE_ID_REGEX_PATTERN, max_cap, re.IGNORECASE)
                if match is not None:
                    out_fig.figure_id = int(match.group(1))
        output_figures.append(out_fig)

    return output_figures


def caption_page_mupdf(
    pdf_path: str | io.BytesIO,
    page_num: int,
    always_create_screenshots: bool = False,
    loading_function: Callable[[str | io.BytesIO], str | io.BytesIO] | None = None,
    no_render_mode: bool = False,
) -> list[OutputFigure]:
    """
    Run the full MuPDF pipeline on a single PDF page.

    Opens the document, builds a :class:`MuPDFPage`, runs caption detection
    and merging, then materialises :class:`OutputFigure` objects.

    Args:
        pdf_path (str | io.BytesIO): PDF source.
        page_num (int): Zero-based page index to process.
        always_create_screenshots (bool, optional): When ``True``, page
            screenshots are rendered eagerly. Defaults to ``False``.
        loading_function (Callable | None, optional): Hook applied to
            ``pdf_path`` before opening (see :class:`DocumentWrapper`).
            Defaults to ``None``.
        no_render_mode (bool, optional): When ``True``, no figure rendering
            is performed. Defaults to ``False``.

    Returns:
        list[OutputFigure]: Detected figures for the given page.

    """
    with DocumentWrapper(pdf_path, loading_function=loading_function) as pdf_document:
        page = MuPDFPage(
            pdf_document,
            page_num,
            always_create_screenshots=always_create_screenshots,
            no_render_mode=no_render_mode,
        )

        page_captions = find_captions(page)
        merged_captions = merge_surrounding_boxes(
            page, page_captions.figure_caption_indices
        )

        output_figures = finalize_figures(
            page,
            pdf_path,
            page_captions,
            merged_captions,
            no_render_mode=no_render_mode,
        )
        return output_figures


def page_worker(args):
    """
    Multiprocessing worker for parallel page processing.

    Wraps :func:`caption_page_mupdf` in a try/except so that a stack trace is
    printed before the exception is re-raised in the worker process.

    Args:
        args (tuple): ``(pdf_path, page_num, always_create_screenshots,
            no_render_mode)`` packed into a single tuple to be compatible
            with :meth:`multiprocessing.Pool.map`.

    Returns:
        list[OutputFigure]: Detected figures for the given page.

    Raises:
        Exception: Re-raises any exception raised by
            :func:`caption_page_mupdf` after logging a traceback.

    """
    pdf_path, page_num, always_create_screenshots, no_render_mode = args
    try:
        result = caption_page_mupdf(
            pdf_path,
            page_num,
            always_create_screenshots=always_create_screenshots,
            no_render_mode=no_render_mode,
        )
    except Exception as e:
        print(f"Error processing page {page_num}: {e}")
        print(traceback.format_exc())
        raise e

    return result


def caption_pdf_parallel(
    pdf_path: str | io.BytesIO,
    always_create_screenshots: bool = False,
    num_processes: int = 1,
    use_tqdm: bool = False,
    loading_function: Callable[[str | io.BytesIO], str | io.BytesIO] | None = None,
    no_render_mode: bool = False,
) -> list[OutputFigure]:
    """
    Run the MuPDF pipeline on every page of a PDF, optionally in parallel.

    For ``num_processes <= 1`` (or a single-page PDF) the loop runs in the
    calling process; otherwise pages are processed in a
    :class:`multiprocessing.Pool`.

    Args:
        pdf_path (str | io.BytesIO): PDF source.
        always_create_screenshots (bool, optional): When ``True``, page
            screenshots are rendered eagerly. Defaults to ``False``.
        num_processes (int, optional): Maximum number of worker processes.
            Defaults to ``1``.
        use_tqdm (bool, optional): When ``True``, a progress bar is shown.
            Defaults to ``False``.
        loading_function (Callable | None, optional): Hook applied to
            ``pdf_path`` before opening. Defaults to ``None``.
        no_render_mode (bool, optional): When ``True``, no figure rendering
            is performed. Defaults to ``False``.

    Returns:
        list[OutputFigure]: All detected figures from every page of the PDF.

    """
    with DocumentWrapper(pdf_path, loading_function=loading_function) as pdf_document:
        num_pages = pdf_document.page_count

    if num_pages == 0:
        return []

    if num_pages == 1:
        return caption_page_mupdf(
            pdf_path,
            0,
            always_create_screenshots=always_create_screenshots,
            no_render_mode=no_render_mode,
        )

    if num_processes < 2:
        return [
            fig
            for page_num in tqdm(range(num_pages), disable=not use_tqdm)
            for fig in caption_page_mupdf(
                pdf_path,
                page_num,
                always_create_screenshots=always_create_screenshots,
                no_render_mode=no_render_mode,
            )
        ]

    if not use_tqdm:
        with multiprocessing.Pool(processes=min(num_processes, num_pages)) as pool:
            args = [
                (pdf_path, page_num, always_create_screenshots, no_render_mode)
                for page_num in range(num_pages)
            ]
            results = pool.map(page_worker, args)

        return [fig for page_figs in results for fig in page_figs]

    with multiprocessing.Pool(processes=min(num_processes, num_pages)) as pool:
        args = [
            (pdf_path, page_num, always_create_screenshots, no_render_mode)
            for page_num in range(num_pages)
        ]
        results = list(
            tqdm(
                pool.imap(page_worker, args),
                total=num_pages,
                desc="Processing pages",
            )
        )
    return [fig for page_figs in results for fig in page_figs]


def caption_pdf(
    pdf_path: str | io.BytesIO,
    use_dl: bool = False,
    always_create_screenshots: bool = False,
    num_processes: int = 1,
) -> list[OutputFigure]:
    """
    Extract figures and captions from every page of a PDF.

    Selects the layout backend (MuPDF or layoutparser-DL), parses the
    document, and runs caption detection, caption merging and figure
    finalization on every page.

    Args:
        pdf_path (str | io.BytesIO): PDF source.
        use_dl (bool, optional): When ``True``, use the deep-learning
            layout backend. Requires the optional ``layoutparser`` extra.
            Defaults to ``False``.
        always_create_screenshots (bool, optional): When ``True``, page
            screenshots are rendered eagerly (only used by the MuPDF
            backend). Defaults to ``False``.
        num_processes (int, optional): Maximum number of worker processes
            used during page parsing. Defaults to ``1``.

    Returns:
        list[OutputFigure]: All detected figures from every page of the PDF.

    """
    if use_dl:
        pdf_layout = LAYOUT_REGISTRY.get("PDFLayoutDL")(
            pdf_path, num_processes=num_processes
        )
    else:
        pdf_layout = LAYOUT_REGISTRY.get("MuPDFLayout")(
            pdf_path,
            always_create_screenshots=always_create_screenshots,
            num_processes=num_processes,
        )

    captions_per_page = [find_captions(page) for page in pdf_layout.pages]
    merged_captions = [
        merge_surrounding_boxes(page, captions.figure_caption_indices)
        for page, captions in zip(pdf_layout.pages, captions_per_page)
    ]

    output_figures = [
        fig
        for page, page_captions, page_merged_captions in zip(
            pdf_layout.pages, captions_per_page, merged_captions
        )
        for fig in finalize_figures(
            page,
            pdf_path,
            page_captions,
            page_merged_captions,
        )
    ]
    return output_figures
