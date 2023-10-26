from tika import parser
from pdf2image import convert_from_path
import numpy as np
import easyocr
from logging import getLogger


logger = getLogger(__name__)


def extract_text_from_pdf(pdf_path: str, max_y_difference: int, gpu: bool) -> str:
    """Extract text from pdf using the tika module.

    Args:
        pdf_path (Path):
            Path to pdf
        max_y_difference (int):
            Maximum difference between y coordinates of bounding boxes for them
            to be considered as being on the same line.
        gpu (bool):
            Whether to use gpu for OCR.

    Returns:
        str:
            Text from pdf
    """
    # Tika works well for PDFs that are not scanned/flattened.

    text = parser.from_file(str(pdf_path))["content"]
    if text is not None:
        logger.info("Not using OCR.")
        return text.strip()
    else:  # Tika didn't work, use OCR instead.
        logger.info("Using OCR.")
        images = convert_from_path(pdf_path)
        reader = easyocr.Reader(["da"], gpu=gpu)
        text = ""
        for image in images:
            # Extract bounding boxes
            result = reader.readtext(np.array(image)) # I have tried more arguments, but it doesn't seem to help.
            # result = reader.readtext(np.array(image), min_size=1, decoder="beamer", beamWidth=10, mag_ratio=5)

            # Sort w.r.t y coordinate.
            # bbox[0][0][1] is the y coordinate of the top left corner of the bounding box.
            result_y = sorted(result, key=lambda bbox: bbox[0][0][1])

            # Group bounding boxes that are on the same line.
            # E.g. the variable lines will be a list of lists, where each list contains
            # the bounding boxes for a given line of text in the pdf.
            # The variable `max_y_difference` is used to determine if two bounding boxes
            # are on the same line. E.g. if the difference between the y coordinate of
            # the top left corner of two bounding boxes is less than `max_y_difference`,
            # then the two bounding boxes are said to be on the same line.
            lines = []
            current_line = [result_y[0]]
            i = 1
            n = len(result_y)
            last_bbox_appended = False
            while i < n:
                # Get y coordinates for current and previous bounding box
                y = _get_top_left_y(result_y, i)
                y_prev = _get_top_left_y(result_y, i - 1)

                # While the current and previous bounding box are on the same line,
                # add the current bounding box to the current line.
                while i < n and _on_same_line(y, y_prev, max_y_difference):
                    current_line.append(result_y[i])

                    if i == n - 1:
                        last_bbox_appended = True
                        break
                    else:
                        i += 1
                        y = _get_top_left_y(result_y, i)
                        y_prev = _get_top_left_y(result_y, i - 1)

                # Add current line to lines
                lines.append(current_line)

                if i < n:
                    # Next line
                    current_line = [result_y[i]]

                # Edge case: last line
                if i == n - 1 and not last_bbox_appended:
                    lines.append(current_line)

                i += 1

            # Now sort the lines w.r.t x coordinate.
            # The lines should then be sorted w.r.t how the text is read.
            for i, line in enumerate(lines):
                lines[i] = sorted(line, key=lambda x: x[0][0][0])

            # Each bounding box on a line is joined together with a space,
            # and the lines of text are joined together with \n.
            page_text = "\n".join(
                [" ".join([bbox[1] for bbox in line]) for line in lines]
            )

            # Add page text to text
            text += f"\n\n{page_text}\n\n"

        return text.strip()


def _on_same_line(y: int, y_prev: int, max_y_difference: int) -> bool:
    """Helper function to determine if two bounding boxes are on the same line.
    
    Used in `extract_text_from_pdf()` in the OCR part.

    Args:
        y (int):
            y coordinate of top left corner of current bounding box.
        y_prev (int):
            y coordinate of top left corner of previous bounding box.
        max_y_difference (int):
            Maximum difference between y coordinates of two bounding boxes on the same line.
    
    Returns:
        bool:
            True if the two bounding boxes are on the same line. False otherwise.
    """
    return abs(y - y_prev) < max_y_difference


def _get_top_left_y(result, idx):
    """Helper function to get the y coordinate of the top left corner of a bounding box.

    Used in `extract_text_from_pdf()` in the OCR part.

    Args:
        result (list):
            Result from using `easyocr.Reader.readtext()`
        idx (int):
            Index of bounding box

    Returns:
        int:
            y coordinate of the top left corner of the bounding box.
    """
    y = result[idx][0][0][1]
    return y
