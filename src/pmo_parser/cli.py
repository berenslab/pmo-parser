"""
Command-line entry point of the parser.

Iterates over every ``*.pdf`` file in a folder, runs
:func:`pmo_parser.caption_pdf` on each and writes one JSON file plus one
image per detected figure to the output folder.
"""

import argparse
import json
import os

from pmo_parser import caption_pdf


def main():
    """
    Run the CLI.

    Parses ``input_path`` and ``--output-path`` from ``argv``, processes every
    PDF in the input folder and writes the results (json metadata plus png images)
    into ``output_path``. Errors raised while processing a single PDF are
    logged to ``log.text`` and do not stop the loop.
    """
    parser = argparse.ArgumentParser(
        description="Extract figures and captions from PDFs"
    )
    parser.add_argument("input_path", help="Path to folder containing PDFs")
    parser.add_argument(
        "--output-path", help="Path to output folder (default: {input_path}/results)"
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    output_path = args.output_path or os.path.join(input_path, "results")
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, "log.text")

    f_log = open(log_file, "w")

    # Read each files in the input path
    for pdf in os.listdir(input_path):
        if pdf.endswith(".pdf") and (not pdf.startswith("._")):
            try:
                output_file_path = os.path.join(output_path, pdf[:-4])
                if not os.path.isdir(output_file_path):
                    os.makedirs(output_file_path)
                figure_list = caption_pdf(
                    os.path.join(input_path, pdf),
                )

                # Save files
                serializable_caption_data = {"figures": []}

                for figure in figure_list:
                    serialized_figure, image = figure.serialize()

                    output_dictionary = {
                        **serialized_figure,
                        "image_path": None,
                    }
                    if image is not None:
                        image_path = os.path.join(
                            output_file_path,
                            f"page_{figure.page}_figure_{figure.name}.png",
                        )
                        image.save(image_path)
                        output_dictionary["image_path"] = image_path
                    serializable_caption_data["figures"].append(output_dictionary)

                json_file = os.path.join(output_file_path, f"{pdf[:-4]}.json")
                with open(json_file, "w") as outfile:
                    json.dump(
                        serializable_caption_data, outfile, ensure_ascii=False, indent=4
                    )

            except RuntimeError as e:
                print(f"Error processing {pdf}:\n{str(e)}\n\n")
                f_log.write(f"Error processing {pdf}:\n{str(e)}\n\n")

    f_log.close()


if __name__ == "__main__":
    main()
