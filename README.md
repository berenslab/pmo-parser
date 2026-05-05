# pmo-parser

A Python library and CLI tool for extracting figures and their captions from scientific publications in PDF format. Originally developed for processing PubMed ophthalmology papers.

## Features

- Detects figures and their associated captions in multi-page PDFs
- Handles compound figures (multiple image panels sharing one caption)
- Exports cropped figure images (PNG) and structured metadata (JSON)
- Supports parallel processing across pages for faster throughput
- Optional deep-learning-based layout detection via [LayoutParser](https://github.com/Layout-Parser/layout-parser)

> **Note:** LayoutParser is installed from a fork that fixed import errors (see [here](https://github.com/verena-hallitschke/layout-parser))

## Installation

Requires Python 3.12+. Install with [uv](https://docs.astral.sh/uv/) or pip:

```bash
pip install .
```

For deep-learning layout detection (optional):

```bash
pip install ".[dl]"
```

## Usage

### CLI

```bash
pmo-parser <input_path> [--output-path <output_path>]
```

- `input_path` — directory containing one or more `.pdf` files
- `--output-path` — destination directory (defaults to `<input_path>/results`)

For each PDF, the tool creates a subdirectory under `output_path` containing:
- One `.png` per detected figure
- A `.json` file with figure bounding boxes, caption text, page numbers, and confidence scores

A `log.text` file is written to the output directory listing any processing errors.

### Python API

```python
from pmo_parser import caption_pdf

figures = caption_pdf("path/to/paper.pdf")

for fig in figures:
    serialized, image = fig.serialize()
    print(serialized["caption"])   # list of caption dicts with text and bbox
    if image is not None:
        image.save(f"figure_{fig.page}_{fig.name}.png")
```

`caption_pdf` accepts either a file path string or a `BytesIO` object and returns a list of `OutputFigure` objects.

#### `caption_pdf` parameters

| Parameter | Default | Description |
|---|---|---|
| `pdf_path` | — | Path or `BytesIO` of the PDF |
| `use_dl` | `False` | Use LayoutParser DL model for layout detection |
| `always_create_screenshots` | `False` | Render page screenshots even when not needed |
| `num_processes` | `1` | Number of parallel worker processes |

### Output format

Each figure in the JSON output has the following structure:

```json
{
  "page": 2,
  "name": null,
  "type": "FIGURE",
  "figure_bbox": {"x0": 50.0, "y0": 100.0, "x1": 300.0, "y1": 400.0},
  "caption": [
    {
      "text": "Figure 1. Example caption text.",
      "x0": 50.0, "y0": 405.0, "x1": 300.0, "y1": 420.0
    }
  ],
  "caption_scores": [4.5],
  "dpi": 150,
  "image_path": "results/paper/page_2_figure_None.png"
}
```

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@article{hallitschke2026pubmedophtha,
  title   = {PubMed-Ophtha: An open resource for training ophthalmology vision-language models on scientific literature},
  author  = {Hallitschke, Verena Jasmin and Eickhoff, Carsten and Berens, Philipp},
  journal = {arXiv preprint arXiv:2605.02720},
  year    = {2026}
}
