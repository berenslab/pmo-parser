"""
Unit tests for pipeline functions using synthetic Page objects.

All tests use FakePage from conftest.py — no real PDF is required.
"""

from __future__ import annotations

from pmo_parser.algorithm import (
    CaptionResult,
    finalize_figures,
    find_captions,
    merge_surrounding_boxes,
)
from pmo_parser.bounding_boxes import BBox, ImageBBox, ReadingOrder, TextBox
from pmo_parser.page.base_page import ImageCluster
from tests.conftest import FakePage, make_image_bbox

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tbox(x0, y0, x1, y1, text) -> TextBox:
    return TextBox(
        x0=x0, y0=y0, x1=x1, y1=y1, text=text, reading_order=ReadingOrder.LTR_TD
    )


def empty_page():
    """Page with no figures and no texts."""
    return FakePage(page_figures=[], page_texts=[])


# ===========================================================================
# find_captions
# ===========================================================================


def test_find_captions_no_figures():
    """Returns empty assignment when there are no figures on the page."""
    page = FakePage(
        page_figures=[],
        page_texts=[[tbox(0, 200, 200, 220, "Figure 1. Caption")]],
    )
    caption_result = find_captions(page)
    captions = caption_result.figure_caption_indices
    scores = caption_result.figure_caption_scores
    assert captions == []
    assert scores == []


def test_find_captions_no_texts():
    """Returns empty assignment when there are no text blocks on the page."""
    page = FakePage(
        page_figures=[make_image_bbox(100, 100, 300, 300)],
        page_texts=[],
    )
    caption_result = find_captions(page)
    captions = caption_result.figure_caption_indices
    scores = caption_result.figure_caption_scores
    assert captions == []
    assert scores == []


def test_find_captions_single_figure_single_caption():
    """A text starting with 'Figure 1' is assigned to the only figure on the page."""
    page = FakePage(
        page_figures=[make_image_bbox(100, 50, 300, 250)],
        page_texts=[[tbox(100, 260, 300, 280, "Figure 1. A test figure.")]],
    )
    caption_result = find_captions(page)
    captions = caption_result.figure_caption_indices
    assert len(captions) == 1
    assert 0 in captions[0]  # text index 0 assigned to image 0


def test_find_captions_caption_keyword_score():
    """Text starting with 'fig' scores higher than generic text."""
    page = FakePage(
        page_figures=[make_image_bbox(100, 50, 300, 250)],
        page_texts=[
            [tbox(100, 260, 300, 280, "Figure 1. Caption text.")],
            [tbox(100, 300, 300, 320, "Some other paragraph text.")],
        ],
    )
    matrix = find_captions(page).score_matrix
    # matrix row 0 = image 0; col 0 = "Figure 1..." caption, col 1 = generic text
    assert matrix[0][0] > matrix[0][1]


def test_find_captions_close_caption_preferred():
    """Of two 'Figure' captions, the one physically closer to the image scores higher."""
    page = FakePage(
        page_figures=[make_image_bbox(100, 50, 300, 250)],
        page_texts=[
            [tbox(100, 260, 300, 280, "Figure 1. Close caption.")],  # gap ≈ 10
            [tbox(100, 600, 300, 620, "Figure 2. Far caption.")],  # gap ≈ 350
        ],
    )
    matrix = find_captions(page).score_matrix
    assert matrix[0][0] > matrix[0][1]


def test_find_captions_far_caption_excluded():
    """A caption more than 300 pts from every figure is not assigned."""
    page = FakePage(
        page_figures=[make_image_bbox(100, 50, 300, 250)],
        page_texts=[[tbox(100, 700, 300, 720, "Figure 1. Very far caption.")]],
        page_height=842,
    )
    captions = find_captions(page).figure_caption_indices
    # The caption may still appear in the score matrix with a low score,
    # but the assigned list should be empty for this image.
    assert len(captions[0]) == 0


def test_find_captions_obstructed_caption_penalised():
    """A caption with boxes in between the figure and caption scores lower."""
    # Caption on left of image with several intervening text boxes
    page = FakePage(
        page_figures=[make_image_bbox(300, 50, 500, 250)],
        page_texts=[
            [tbox(100, 100, 200, 120, "Obstructed text block A")],
            [tbox(100, 130, 200, 150, "Obstructed text block B")],
            [tbox(100, 160, 200, 180, "Obstructed text block C")],
            [tbox(100, 260, 200, 280, "Figure 1. Caption.")],  # also left of image
        ],
    )
    matrix = find_captions(page).score_matrix
    # The "Figure 1. Caption." (index 3) should score lower than if it were unobstructed.
    # We check that it scores lower than the direct distance would suggest by
    # comparing it to an unobstructed baseline created separately.
    unobstructed_page = FakePage(
        page_figures=[make_image_bbox(300, 50, 500, 250)],
        page_texts=[[tbox(100, 260, 200, 280, "Figure 1. Caption.")]],
    )
    matrix_unobstructed = find_captions(unobstructed_page).score_matrix
    assert matrix[0][3] <= matrix_unobstructed[0][0]


def test_find_captions_caption_below_scores_higher():
    """A caption in the column directly below the image receives a bonus score."""
    # Caption directly below vs caption to the side at equal distance
    image = make_image_bbox(100, 50, 300, 250)

    page_below = FakePage(
        page_figures=[image],
        page_texts=[[tbox(100, 260, 300, 280, "Figure 1. Below caption.")]],
    )
    page_side = FakePage(
        page_figures=[image],
        page_texts=[[tbox(320, 150, 520, 170, "Figure 1. Side caption.")]],
    )
    matrix_below = find_captions(page_below).score_matrix
    matrix_side = find_captions(page_side).score_matrix
    assert matrix_below[0][0] >= matrix_side[0][0]


def test_find_captions_two_figures_two_captions():
    """Each figure gets exactly one distinct caption assigned."""
    page = FakePage(
        page_figures=[
            make_image_bbox(50, 50, 200, 200),
            make_image_bbox(350, 50, 500, 200),
        ],
        page_texts=[
            [tbox(50, 210, 200, 230, "Figure 1. Left figure.")],
            [tbox(350, 210, 500, 230, "Figure 2. Right figure.")],
        ],
    )
    captions = find_captions(page).figure_caption_indices
    assert len(captions) == 2
    all_assigned = [c for caps in captions for c in caps]
    # Each caption index appears exactly once
    assert len(all_assigned) == len(set(all_assigned))


def test_find_captions_compound_image():
    """Two images sharing one caption are both assigned the same caption index."""
    cluster = ImageCluster(image_ids=[0, 1], screenshot=None)
    page = FakePage(
        page_figures=[
            make_image_bbox(50, 50, 200, 200),
            make_image_bbox(210, 50, 360, 200),
        ],
        page_texts=[[tbox(50, 210, 360, 230, "Figure 1. Compound figure.")]],
        figure_clusters=[cluster],
    )
    caption_result = find_captions(page)
    captions = caption_result.figure_caption_indices
    cluster_captions = caption_result.cluster_caption_indices  # noqa: F841
    # Both images should have caption index 0 (the single caption)
    assert captions[0] == captions[1] == [0]


def test_find_captions_excluded_image():
    """An image with max per-image score < 2 produces an empty caption assignment."""
    page = FakePage(
        page_figures=[make_image_bbox(100, 50, 300, 250)],
        page_texts=[[tbox(10, 400, 200, 420, "Completely unrelated paragraph.")]],
        page_height=842,
    )
    captions = find_captions(page).figure_caption_indices
    assert captions[0] == []


def test_find_captions_raw_score_matrix_shape():
    """The raw score matrix has shape (num_figures, num_texts)."""
    page = FakePage(
        page_figures=[
            make_image_bbox(100, 50, 300, 250),
            make_image_bbox(350, 50, 500, 250),
        ],
        page_texts=[
            [tbox(100, 260, 300, 280, "Figure 1.")],
            [tbox(350, 260, 500, 280, "Figure 2.")],
            [tbox(10, 400, 200, 420, "Other text.")],
        ],
    )
    matrix = find_captions(page).score_matrix
    assert len(matrix) == 2  # 2 figures → 2 rows
    assert len(matrix[0]) == 3  # 3 text blocks → 3 cols
    assert len(matrix[1]) == 3


# ===========================================================================
# merge_surrounding_boxes
# ===========================================================================


def test_merge_no_captions_input():
    """An empty captions list returns an empty list."""
    page = FakePage(page_figures=[], page_texts=[])
    assert merge_surrounding_boxes(page, []) == []


def test_merge_all_empty_assignments():
    """When no figure has any caption assigned, returns a list of empty lists."""
    page = FakePage(
        page_figures=[make_image_bbox(100, 50, 300, 250)],
        page_texts=[[tbox(100, 260, 300, 280, "Figure 1.")]],
    )
    result = merge_surrounding_boxes(page, [[]])
    assert result == [[]]


def test_merge_single_caption_no_neighbours():
    """When there are no adjacent free text boxes, the caption is returned unchanged."""
    page = FakePage(
        page_figures=[make_image_bbox(100, 50, 300, 250)],
        page_texts=[[tbox(100, 260, 300, 280, "Figure 1. Caption.")]],
    )
    result = merge_surrounding_boxes(page, [[0]])
    assert len(result) == 1
    assert len(result[0]) == 1
    assert len(result[0][0]) == 1  # one list of TextBoxes, containing the single box


def test_merge_adjacent_box_absorbed():
    """A text box within 1.7 * letter_width horizontally is absorbed into the caption."""
    # caption at text index 0; continuation at index 1, placed just to the right
    cap = tbox(100, 260, 250, 280, "Figure 1.")  # width 150 → mean_lw ~21 pts per char
    continuation = tbox(252, 260, 350, 280, "A")  # x-gap = 2 < 1.7 * ~8 = ~13.6

    page = FakePage(
        page_figures=[make_image_bbox(100, 50, 300, 250)],
        page_texts=[[cap], [continuation]],
        mean_letter_width=8.0,
        mean_letter_height=12.0,
    )
    result = merge_surrounding_boxes(page, [[0]])
    # The result for image 0, caption-group 0 should contain both boxes
    all_boxes = result[0][0]
    assert len(all_boxes) == 2


def test_merge_distant_box_not_absorbed():
    """A text box beyond the distance threshold is not included in the caption."""
    cap = tbox(100, 260, 250, 280, "Figure 1.")
    far = tbox(500, 260, 600, 280, "Far text.")  # gap >> 1.7 * letter_width

    page = FakePage(
        page_figures=[make_image_bbox(100, 50, 300, 250)],
        page_texts=[[cap], [far]],
        mean_letter_width=8.0,
        mean_letter_height=12.0,
    )
    result = merge_surrounding_boxes(page, [[0]])
    all_boxes = result[0][0]
    assert len(all_boxes) == 1


def test_merge_already_assigned_box_skipped():
    """A text box already assigned as a caption to another figure is not absorbed."""
    cap0 = tbox(100, 260, 250, 280, "Figure 1.")
    cap1 = tbox(
        252, 260, 350, 280, "Figure 2."
    )  # close to cap0 — but already a caption

    page = FakePage(
        page_figures=[
            make_image_bbox(100, 50, 300, 250),
            make_image_bbox(350, 50, 500, 250),
        ],
        page_texts=[[cap0], [cap1]],
        mean_letter_width=8.0,
        mean_letter_height=12.0,
    )
    result = merge_surrounding_boxes(page, [[0], [1]])
    # cap1 belongs to image 1 (caption_set includes both 0 and 1), so it should
    # not be merged into the boxes for image 0's caption
    boxes_for_image0 = result[0][0]
    texts = {b.text for b in boxes_for_image0}
    assert "Figure 2." not in texts


# ===========================================================================
# finalize_figures
# ===========================================================================


def _make_captions_per_page(captions, scores, clusters=None, cluster_scores=None):
    """Build the 4-tuple that find_captions returns, for use in finalize_figures."""
    n_clusters = len(clusters) if clusters else 0
    return CaptionResult(
        figure_caption_indices=captions,
        figure_caption_scores=scores,
        cluster_caption_indices=clusters
        if clusters
        else [[] for _ in range(n_clusters)],
        cluster_caption_scores=cluster_scores
        if cluster_scores
        else [[] for _ in range(n_clusters)],
        score_matrix=[],  # not used by finalize_figures, so we can leave it empty
    )


def test_finalize_empty_captions_skipped():
    """Figures with no captions assigned produce no OutputFigure."""
    page = FakePage(
        page_figures=[make_image_bbox(100, 50, 300, 250)],
        page_texts=[[tbox(100, 260, 300, 280, "Figure 1.")]],
    )
    captions_per_page = _make_captions_per_page([[]], [[]])
    merged = merge_surrounding_boxes(page, [[]])
    result = finalize_figures(
        page, "unused.pdf", captions_per_page, merged, no_render_mode=True
    )
    assert result == []


def test_finalize_single_figure_output_fields():
    """OutputFigure has the expected page, figure_bbox, and captions fields."""
    page = FakePage(
        page_figures=[make_image_bbox(100, 50, 300, 250)],
        page_texts=[[tbox(100, 260, 300, 280, "Figure 1. A caption.")]],
        page_num=2,
    )
    captions_per_page = _make_captions_per_page([[0]], [[3.5]])
    merged = merge_surrounding_boxes(page, [[0]])
    result = finalize_figures(
        page, "unused.pdf", captions_per_page, merged, no_render_mode=True
    )
    assert len(result) == 1
    fig = result[0]
    assert fig.page == 2
    assert fig.figure_bbox is not None
    assert fig.captions is not None and len(fig.captions) > 0


def test_finalize_figure_id_extracted():
    """Caption 'Fig. 2 ...' causes figure_id to be set to 2."""
    page = FakePage(
        page_figures=[make_image_bbox(100, 50, 300, 250)],
        page_texts=[[tbox(100, 260, 300, 280, "Fig. 2 A caption.")]],
    )
    captions_per_page = _make_captions_per_page([[0]], [[3.5]])
    merged = merge_surrounding_boxes(page, [[0]])
    result = finalize_figures(
        page, "unused.pdf", captions_per_page, merged, no_render_mode=True
    )
    assert result[0].figure_id == 2


def test_finalize_figure_id_not_set():
    """A caption not starting with 'fig' leaves figure_id as None."""
    page = FakePage(
        page_figures=[make_image_bbox(100, 50, 300, 250)],
        page_texts=[[tbox(100, 260, 300, 280, "Table 1. A table caption.")]],
    )
    captions_per_page = _make_captions_per_page([[0]], [[3.5]])
    merged = merge_surrounding_boxes(page, [[0]])
    result = finalize_figures(
        page, "unused.pdf", captions_per_page, merged, no_render_mode=True
    )
    assert result[0].figure_id is None


def test_finalize_cluster_figure_uses_cluster_bbox():
    """A compound image uses the union of cluster image bboxes as figure_bbox."""
    img0 = make_image_bbox(50, 50, 200, 200)
    img1 = make_image_bbox(210, 50, 360, 200)
    cluster = ImageCluster(image_ids=[0, 1], screenshot=None)

    page = FakePage(
        page_figures=[img0, img1],
        page_texts=[[tbox(50, 210, 360, 230, "Figure 1. Compound figure.")]],
        figure_clusters=[cluster],
    )
    captions_per_page = _make_captions_per_page(
        [[0], [0]],
        [[3.5], [3.5]],
        clusters=[[0]],
        cluster_scores=[[3.5]],
    )
    merged = merge_surrounding_boxes(page, [[0], [0]])
    result = finalize_figures(
        page, "unused.pdf", captions_per_page, merged, no_render_mode=True
    )
    assert len(result) == 1
    union = BBox.union_boxes([img0.get_bbox(), img1.get_bbox()])
    assert result[0].figure_bbox.is_equal(union)


def test_finalize_no_render_mode_no_rerender():
    """
    With no_render_mode=True, no render call is attempted and _dpi is preserved.

    In real pipeline usage, pages are parsed with no_render_mode=True so that
    ImageBBox.image is None. We replicate that by building an ImageBBox with no
    image and a virtual_dpi set, then verifying that finalize_figures never tries
    to call render_page (which would fail with a bad path).
    """
    # Build an un-rendered ImageBBox (image=None, virtual dpi set)
    im = ImageBBox(x0=100, y0=50, x1=300, y1=250, image=None)
    im._virtual_dpi = 72

    page = FakePage(
        page_figures=[im],
        page_texts=[[tbox(100, 260, 300, 280, "Figure 1. Caption.")]],
    )
    captions_per_page = _make_captions_per_page([[0]], [[3.5]])
    merged = merge_surrounding_boxes(page, [[0]])
    # Passing a non-existent path — if render were attempted it would raise
    result = finalize_figures(
        page, "/nonexistent/path.pdf", captions_per_page, merged, no_render_mode=True
    )
    assert len(result) == 1
    assert result[0].image is None
    # _dpi should be forwarded from the figure
    assert result[0]._dpi == 72
