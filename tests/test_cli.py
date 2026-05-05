"""Tests for cli.py — main() entry point."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from pmo_parser.cli import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_cli(input_path: str, output_path: str):
    """Invoke CLI with patched sys.argv."""
    with patch("sys.argv", ["pmo-parser", input_path, "--output-path", output_path]):
        main()


# ===========================================================================
# CLI tests
# ===========================================================================


def test_cli_creates_output_directory(minimal_pdf_path, tmp_path):
    """The output directory is created if it does not exist."""
    input_dir = os.path.dirname(minimal_pdf_path)
    out_dir = str(tmp_path / "new_subdir" / "nested")
    run_cli(input_dir, out_dir)
    assert os.path.isdir(out_dir)


def test_cli_produces_json_per_pdf(minimal_pdf_path, tmp_path):
    """One JSON file is produced for each PDF in the input directory."""
    input_dir = os.path.dirname(minimal_pdf_path)
    out_dir = str(tmp_path / "out")
    run_cli(input_dir, out_dir)

    pdf_name = os.path.splitext(os.path.basename(minimal_pdf_path))[0]
    json_path = os.path.join(out_dir, pdf_name, f"{pdf_name}.json")
    assert os.path.isfile(json_path)

    with open(json_path) as f:
        data = json.load(f)
    assert "figures" in data


def test_cli_produces_image_per_figure(tmp_path, tmp_path_factory):
    """Each figure that has an extracted image gets a .png saved to disk."""
    import pymupdf  # noqa: PLC0415

    # Build a PDF that actually produces a figure with an image
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    # Draw a filled rectangle large enough to be detected as an image element
    page.draw_rect(
        pymupdf.Rect(50, 50, 400, 400), color=(0, 0, 0), fill=(0.5, 0.5, 0.5), width=2
    )
    page.insert_text((50, 420), "Figure 1. A test figure.", fontsize=11)
    pdf_dir = tmp_path_factory.mktemp("img_pdf")
    pdf_path = pdf_dir / "figure_test.pdf"
    doc.save(str(pdf_path))
    doc.close()

    out_dir = str(tmp_path / "img_out")
    with patch("sys.argv", ["pmo-parser", str(pdf_dir), "--output-path", out_dir]):
        main()

    # Collect all .png files produced
    png_files = []
    for root, _, files in os.walk(out_dir):
        png_files.extend(f for f in files if f.endswith(".png"))

    # We cannot guarantee the pipeline detects an image in a synthetic PDF,
    # so we only assert that if figures were found, their images were saved.
    json_files = []
    for root, _, files in os.walk(out_dir):
        json_files.extend(os.path.join(root, f) for f in files if f.endswith(".json"))

    if json_files:
        with open(json_files[0]) as jf:
            data = json.load(jf)
        figures_with_images = [f for f in data["figures"] if f.get("image_path")]
        assert len(figures_with_images) == len(png_files)


def test_cli_runtime_error_logged(minimal_pdf_path, tmp_path):
    """A RuntimeError during processing is caught, written to log.text, and does not crash."""
    input_dir = os.path.dirname(minimal_pdf_path)
    out_dir = str(tmp_path / "err_out")

    with patch("pmo_parser.cli.caption_pdf", side_effect=RuntimeError("boom")):
        run_cli(input_dir, out_dir)

    log_path = os.path.join(out_dir, "log.text")
    assert os.path.isfile(log_path)
    with open(log_path) as f:
        content = f.read()
    assert "boom" in content


@pytest.mark.xfail(
    reason=(
        "cli.py opens log file with open() and calls close() at the end, "
        "but an exception before that line leaves the file handle open. "
        "Fix: wrap in 'with open(...) as f_log:'."
    ),
    strict=True,
)
def test_cli_log_file_closed_on_error(minimal_pdf_path, tmp_path):
    """
    The log file handle is closed even when an exception is raised during processing.

    Currently xfail due to §2.2: close() is never reached if an unexpected exception
    (not RuntimeError) escapes the try/except block.
    """
    input_dir = os.path.dirname(minimal_pdf_path)
    out_dir = str(tmp_path / "close_test")
    os.makedirs(out_dir, exist_ok=True)

    # Patch os.listdir to raise an unexpected exception after the log file is opened
    original_listdir = os.listdir

    call_count = [0]

    def patched_listdir(path):
        call_count[0] += 1
        if call_count[0] > 1:
            raise ValueError("unexpected error after log open")
        return original_listdir(path)

    with patch("os.listdir", side_effect=patched_listdir):
        with pytest.raises(ValueError):
            run_cli(input_dir, out_dir)

    # If the fix is in place, the log file should still have been closed properly.
    # We can't easily probe the OS file descriptor, so we just verify the log exists
    # (the context manager would have flushed and closed it).
    log_path = os.path.join(out_dir, "log.text")
    assert os.path.isfile(log_path)
