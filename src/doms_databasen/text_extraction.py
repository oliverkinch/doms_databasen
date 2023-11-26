from logging import getLogger
from typing import List, Tuple

import cv2
import easyocr
import fitz
import numpy as np
import pypdfium2 as pdfium
import pytesseract
import skimage
from omegaconf import DictConfig
from pdf2image import convert_from_path
from pypdf import PdfReader
from skimage import measure
from skimage.filters import rank
from skimage.measure._regionprops import RegionProperties
from tika import parser

from src.doms_databasen.constants import (
    BOX_LENGTH_LOWER_BOUND,
    DPI,
    TOLERANCE_FLOOD_FILL,
    USUAL_BOX_HEIGHT,
)

logger = getLogger(__name__)


def extract_text_easyocr(
    config: DictConfig, pdf_path: str, reader: easyocr.Reader
) -> str:
    """Extracts text from a PDF using easyocr.

    Some text is anonymized with boxes, and some text is anonymized with underlines.
    This function tries to find these anonymization, read the anonymized text,
    and then remove the anonymized text from the image before
    reading the rest of the text with easyocr.

    Args:
        config (DictConfig):
            Config.
        pdf_path (str):
            Path to PDF.
        reader (easyocr.Reader):
            Easyocr reader.
    """
    images = map(np.array, convert_from_path(pdf_path, dpi=DPI))
    pdf_text = ""

    # I have not seen a single PDF that uses both methods.
    # I have not either seen a PDF where there not
    # anonymization on the first page, if there are
    # any anonymization at all.
    # Therefore try both methods on the first page,
    # and then use the method that seems to be used, for the
    # rest of the pages.
    underlines_anonymization = True
    box_anonymization = True

    for i, image in enumerate(images):
        # Log info about which anonymization methods are used in the PDF.
        if i == 1:
            if not box_anonymization:
                logger.info(config.message_pdf_has_anonymized_boxes)
            if not underlines_anonymization:
                logger.info(config.message_pdf_has_underline_anonymizations)

        if i == 0:
            # Remove logo top right corner
            image = _remove_logo(
                image=image, indices=(config.logo_row_idx, config.logo_col_idx)
            )

        if box_anonymization:
            anonymized_boxes = _find_anonymized_boxes(
                image=image.copy(),
                box_area_min=config.box_area_min,
                box_height_min=config.box_height_min,
                box_accept_ratio=config.box_accept_ratio,
                slight_shift=config.slight_shift,
            )
            anonymized_boxes_with_text = [
                _get_text_from_anonymized_box(
                    image=image.copy(), anonymized_box=box, reader=reader
                )
                for box in anonymized_boxes
            ]
        else:
            anonymized_boxes_with_text = []

        if underlines_anonymization:
            anonymized_boxes_from_underlines, underlines = _line_anonymization_to_boxes(
                image=image.copy(),
                bounds=(
                    config.underline_height_lower_bound,
                    config.underline_height_upper_bound,
                ),
            )
            anonymized_boxes_from_underlines_with_text = [
                _get_text_from_anonymized_box(image, box, reader, invert=True)
                for box in anonymized_boxes_from_underlines
            ]
        else:
            anonymized_boxes_from_underlines_with_text = []
            underlines = []

        # PDFs seem to be either anonymized with boxes or underlines.
        # After first page of pdf, just use on of the methods,
        # if it is clear that the other method is not used.
        # If no indication of anonymization on first page, then try use Tika.
        # If Tika doesn't work, then use easrocr.
        if i == 0:
            box_anonymization = bool(anonymized_boxes_with_text)
            underlines_anonymization = bool(anonymized_boxes_from_underlines_with_text)
            if not box_anonymization and not underlines_anonymization:
                # No indication of anonymization on first page.
                # Then try use Tika
                logger.info(config.message_try_use_tika)
                pdf_text = _read_text_with_tika(pdf_path=pdf_path)
                if pdf_text:
                    return pdf_text
                else:
                    logger.info(config.message_tika_failed)
                    # I have not seen a PDF where tika is not able
                    # to extract some text.
                    # However,
                    # pdf is supposedly then scanned, e.g. worst case pdf - scanned and no indications of anonymizations
                    # Then use easrocr
                    anonymized_boxes_with_text = []
                    anonymized_boxes_from_underlines_with_text = []
                    underlines = []

        all_anonymized_boxes_with_text = (
            anonymized_boxes_with_text + anonymized_boxes_from_underlines_with_text
        )

        # Remove anonymized boxes from image
        image_processed = _process_image(
            image.copy(), all_anonymized_boxes_with_text, underlines
        )
        save_cv2_image_tmp(image_processed)

        # Read core text of image
        result = reader.readtext(image_processed)

        # Make boxes on same format as anonymized boxes
        boxes = [_change_box_format(box) for box in result]

        # Merge all boxes
        all_boxes = boxes + all_anonymized_boxes_with_text
        page_text = _get_text_from_boxes(
            boxes=all_boxes, max_y_difference=config.max_y_difference
        )
        if i == 0:
            pdf_text += f"{page_text}\n\n"
        else:
            pdf_text += f"\n\n{page_text}\n\n"

    return pdf_text.strip()


def _get_blobs(binary: np.ndarray) -> list:
    """Get blobs from binary image.

    Find all blobs in a binary image, and return the
    blobs sorted by area of its bounding box.

    Args:
        binary (np.ndarray):
            Binary image

    Returns:
        blobs (list):
            List of blobs sorted by area of its bounding box.
    """

    labels = measure.label(binary, connectivity=1)
    blobs = measure.regionprops(labels)
    blobs = sorted(blobs, key=lambda blob: blob.area_bbox, reverse=True)
    return blobs


def _line_anonymization_to_boxes(
    image: np.ndarray, bounds: Tuple[int] = (1, 7)
) -> tuple:
    """Finds all underlines and makes anonymized boxes above them.

    Args:
        image (np.ndarray):
            Image to find anonymized boxes in.
        bounds (Tuple[int]):
            Bounds for height of underline.

    Returns:
        anonymized_boxes (List[dict]):
            List of anonymized boxes with coordinates
            (boxes above found underlines).
        underlines (List[tuple]):
            List of underlines with coordinates
            (will later be used to remove the underlines from the image).
    """
    lb, ub = bounds

    # Use this one when refining boxes
    image_inverted = cv2.bitwise_not(image)

    # Grayscale and invert, such that underlines are white.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)

    # Morphological opening.
    # Removed everything that doesn't look like an underline.
    eroded = cv2.erode(inverted, np.ones((1, 50)), iterations=1)
    dilated = cv2.dilate(eroded, np.ones((1, 50)), iterations=1)

    # Binarize and locate blobs
    binary = binarize(image=dilated, threshold=200)
    blobs = _get_blobs(binary)

    anonymized_boxes = []
    underlines = []
    for blob in blobs:
        row_min, col_min, row_max, col_max = blob.bbox

        # For a blob to be an underline, it should be
        # a "perfect" rectangle.

        height = row_max - row_min
        if blob.area == blob.area_bbox and lb < height < ub:
            box_row_min = row_min - 50  # Give box a height of 50 pixels
            box_row_max = row_min - 1  # Just above underline
            box_col_min = col_min + 5  # Avoid ,) etc. Box will be refined later.
            box_col_max = col_max - 5

            anonymized_box = {
                "coordinates": (box_row_min, box_col_min, box_row_max, box_col_max)
            }

            anonymized_box_refined = _refine_anonymized_box(
                anonymized_box, image_inverted
            )
            if anonymized_box_refined:
                anonymized_boxes.append(anonymized_box_refined)
                underlines.append(blob.bbox)

    return anonymized_boxes, underlines


def _remove_logo(image: np.ndarray, indices: Tuple[int]) -> np.ndarray:
    """Removes logo from image.

    For most PDFs, the logo is in the top right corner of the first page

    Args:
        image (np.ndarray):
            Image to remove logo from.

    Returns:
        np.ndarray:
            Image with logo removed.
    """
    r, c = indices
    image[:r, c:, :] = 255
    return image


def _on_same_line(y: int, y_prev: int, max_y_difference: int) -> bool:
    """Determine if two bounding boxes are on the same line.

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


def _process_image(
    image: np.ndarray, anonymized_boxes: List[dict], underlines: List[tuple]
) -> np.ndarray:
    """Prepare image for easyocr to read the main text (all non-anonymized text).

    Removes all anonymized boxes and underlines from the image,
    and then performs some image processing to make the text easier to read.

    Args:
        image (np.ndarray):
            Image to be processed.
        anonymized_boxes (List[dict]):
            List of anonymized boxes with coordinates.
        underlines (List[tuple]):
            List of underlines with coordinates.

    Returns:
        np.ndarray:
            Processed image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # For the anonymized boxes there already are black boxes,
    # but we will remove the text inside them, by making the text black.
    # The boxes made above underlines is included in the anonymized boxes.
    # For these there are no boxes above them, but only text,
    # but that text is simply removed by making a black box.
    for box in anonymized_boxes:
        row_min, col_min, row_max, col_max = box["coordinates"]
        gray[row_min : row_max + 1, col_min : col_max + 1] = 0
        save_cv2_image_tmp(gray)

    # Make underlines black
    for underline in underlines:
        row_min, col_min, row_max, col_max = underline
        gray[row_min : row_max + 1, col_min : col_max + 1] = 0

    # Invert such that it is white text on black background.
    inverted = cv2.bitwise_not(gray)
    save_cv2_image_tmp(inverted)

    # Image has been inverted, such that it is white text on black background.
    # However this also means that the boxes currently are white.
    # We want to remove them entirely. We do this using flood fill.
    filled = inverted.copy()

    # In flood fill a tolerance of 254 is used.
    # This means that when initiating a flood fill operation at a seed point
    # with a value of 255, all pixels greater than 0 within the object of the seed point
    # will be altered to 0.
    for anonymized_box in anonymized_boxes:
        row_min, col_min, row_max, col_max = anonymized_box["coordinates"]
        seed_point = (row_min, col_min)

        if filled[seed_point] == 0:
            # Box is already removed, supposedly because
            # it overlaps with a previous box.
            continue

        filled = skimage.segmentation.flood_fill(
            image=filled,
            seed_point=seed_point,
            new_value=0,
            connectivity=1,
            tolerance=TOLERANCE_FLOOD_FILL,
        )

    for underline in underlines:
        row_min, col_min, row_max, col_max = underline
        seed_point = (row_min, col_min)

        if filled[seed_point] == 0:
            # Underline is already removed, supposedly because
            # it overlaps with a previous box.
            # I have not seen this happen for underlines,
            # but I suppose it could happen, similar to the boxes.
            continue

        filled = skimage.segmentation.flood_fill(
            image=filled,
            seed_point=seed_point,
            new_value=0,
            connectivity=1,
            tolerance=TOLERANCE_FLOOD_FILL,
        )

    # Increase size of letters slightly
    dilated = cv2.dilate(filled, np.ones((2, 2)))

    return dilated


def _get_text_from_boxes(boxes: List[dict], max_y_difference: int) -> str:
    """Get text from boxes.

    Sorts all boxes w.r.t how a person would read the text,
    and then joins the text together.

    Args:
        boxes (List[dict]):
            List of boxes with coordinates and text
        max_y_difference (int):
            Maximum difference between y coordinates of two bounding boxes on the same line.

    Returns:
        page_text (str):
            Text from current page.
    """
    # Sort w.r.t y coordinate.
    boxes_y_sorted = sorted(boxes, key=lambda box: _middle_y_cordinate(box))

    # Group bounding boxes that are on the same line.
    # E.g. the variable `lines`` will be a list of lists, where each list contains
    # the bounding boxes for a given line of text in the pdf.
    # The variable `max_y_difference` is used to determine if two bounding boxes
    # are on the same line. E.g. if the difference between the y coordinates of
    # two bounding boxes is less than `max_y_difference`,
    # then the two bounding boxes are said to be on the same line.
    current_line = [boxes_y_sorted[0]]
    lines = [current_line]
    ys = []
    for i in range(1, len(boxes_y_sorted)):
        box = boxes_y_sorted[i]
        box_prev = boxes_y_sorted[i - 1]
        y = _middle_y_cordinate(box)
        y_prev = _middle_y_cordinate(box_prev)
        ys.append(y_prev)
        if _on_same_line(y, y_prev, max_y_difference):
            # Box is on current line.
            lines[-1].append(box)
        else:
            # Box is on a new line.
            new_line = [box]
            lines.append(new_line)

    # Now sort each line w.r.t x coordinate.
    # The lines should as a result be sorted w.r.t how a text is read.
    for i, line in enumerate(lines):
        lines[i] = sorted(line, key=lambda box: _left_x_cordinate(box))

    # Each bounding box on a line is joined together with a space,
    # and the lines of text are joined together with \n.
    page_text = "\n".join(
        [" ".join([box["text"] for box in line]) for line in lines]
    ).strip()
    return page_text


def _left_x_cordinate(anonymized_box: dict) -> int:
    """Returns the left x coordinate of a box.

    Used in `_get_text_from_boxes` to sort every line of boxes
    from left to right.

    Args:
        anonymized_box (dict):
            Anonymized box with coordinates.

    Returns:
        int:
            Left x coordinate of box.
    """
    _, col_min, _, _ = anonymized_box["coordinates"]
    return col_min


def _middle_y_cordinate(anonymized_box: dict) -> int:
    """Returns the middle y coordinate of a box.

    Used in `_get_text_from_boxes` to determine if two boxes are on the same line.

    Args:
        anonymized_box (dict):
            Anonymized box with coordinates.

    Returns:
        int:
            Middle y coordinate of box.
    """
    row_min, _, row_max, _ = anonymized_box["coordinates"]
    return (row_min + row_max) // 2


def save_cv2_image_tmp(image):
    """Saves image to tmp.png.

    Used for debugging.
    """
    if image.max() < 2:
        image = image * 255
    cv2.imwrite("tmp.png", image)


def _change_box_format(easyocr_box: tuple) -> dict:
    """Change box format from easyocr style to anonymized box style.

    Easyocr uses (x, y) format and represents a box by
    its corners. We want to represent a box by its min/max row/col.

    Args:
        easyocr_box (tuple):
            Easyocr box.

    Returns:
        anonymized_box (dict):
            Anonymized box.
    """

    tl, tr, _, bl = easyocr_box[0]  # easyocr uses (x, y) format
    row_min, col_min, row_max, col_max = tl[1], tl[0], bl[1], tr[0]
    text = easyocr_box[1]
    anonymized_box = {"coordinates": (row_min, col_min, row_max, col_max), "text": text}
    return anonymized_box


def _get_text_from_anonymized_box(
    image: np.ndarray,
    anonymized_box: dict,
    reader: easyocr.Reader,
    threshold_box: float = 0.3,
    invert: bool = False,
) -> dict:
    """Read text from anonymized box.

    Args:
        image (np.ndarray):
            Image of the current page.
        anonymized_box (dict):
            Anonymized box with coordinates.
        reader (easyocr.Reader):
            Easyocr reader.
        threshold_box (float):
            Threshold used to filter out easyocr outputs with low confidence.
        invert (bool):
            Whether to invert the image or not.
            Easyocr seems to work best with white text on black background.

    Returns:
        anonymized_box (dict):
            Anonymized box with anonymized text.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Easyocr seems to work best with white text on black background.
    if invert:
        gray = cv2.bitwise_not(gray)

    row_min, col_min, row_max, col_max = anonymized_box["coordinates"]

    crop = gray[row_min : row_max + 1, col_min : col_max + 1]

    # Make a box for each word in the box
    # I get better results with easyocr using this approach.
    anonymized_boxes = _split_box(crop=crop, anonymized_box=anonymized_box)

    texts = []
    # `anonymized_boxes` are sorted left to right
    # E.g. the first box will contain the first word of `anonymized_box`.
    for anonymized_box_ in anonymized_boxes:
        row_min, col_min, row_max, col_max = anonymized_box_["coordinates"]

        crop = gray[row_min : row_max + 1, col_min : col_max + 1]
        crop_boundary = _add_boundary(crop)

        # If length of box is short, then there are probably < 2 letters in the box.
        # In this case, scale the image up.
        box_length = col_max - col_min
        scale = 1 if box_length > BOX_LENGTH_LOWER_BOUND else 2

        scaled = cv2.resize(crop_boundary, (0, 0), fx=scale, fy=scale)

        # Increase size of letters
        dilated = cv2.dilate(scaled, np.ones((2, 2)))

        dilated_boundary = _add_boundary(dilated)

        sharpened = (
            np.array(
                skimage.filters.unsharp_mask(dilated_boundary, radius=20, amount=1.9),
                dtype=np.uint8,
            )
            * 255  # output of unsharp_mask is in range [0, 1], but we want [0, 255]
        )

        # Read text from image with easyocr
        result = reader.readtext(sharpened)

        if len(result) == 0:
            text = ""
        else:
            text = " ".join([box[1] for box in result if box[2] > threshold_box])

        texts.append(text)

    text_all = " ".join(texts).strip()

    anonymized_box["text"] = f"<anonym>{text_all}</anonym>" if text_all else ""
    return anonymized_box


def _find_anonymized_boxes(
    image: np.ndarray,
    box_area_min: int = 2000,
    box_height_min: int = 35,
    box_accept_ratio: float = 0.4,
    slight_shift: int = 5,
    threshold_binarize: int = 1,
) -> List[dict]:
    """Finds anonymized boxes in image.

    Args:
        image (np.ndarray):
            Image to find anonymized boxes in.
        box_area_min (int):
            Minimum area of a blob to be considered an anonymized box.
        box_height_min (int):
            Minimum height of a blob to be considered an anonymized box.
        box_accept_ratio (float):
            Minimum ratio between filled area and bounding box area
            for a blob to be considered an anonymized box.
        slight_shift (int):
            Shifts the bottom of the anonymized box up by `slight_shift` pixels.
            This is done as the text is usually in the top of the box.
        threshold_binarize (int):
            Threshold used to binarize the image.

    Returns:
        List[dict]:
            List of anonymized boxes.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Mean filter to make text outside boxes
    # brigther than color of boxes.
    footprint = np.ones((1, 15))
    averaged = rank.mean(gray, footprint=footprint)

    binary = binarize(
        image=averaged, threshold=threshold_binarize, val_min=0, val_max=255
    )
    inverted = cv2.bitwise_not(binary)

    blobs = _get_blobs(inverted)

    # First split multiple boxes into separate boxes
    for blob in blobs:
        if blob.area_bbox < box_area_min:
            break
        row_min, col_min, row_max, col_max = blob.bbox

        box_height = row_max - row_min
        if box_height > 2 * USUAL_BOX_HEIGHT:
            # Split into multiple boxes (horizontal split)

            # Count how many boxes that supposedly are
            # stacked on top of each other.
            n_boxes_in_stack = box_height // USUAL_BOX_HEIGHT
            box_height_ = box_height // n_boxes_in_stack
            for j in range(n_boxes_in_stack):
                row_min_ = row_min + box_height_ * j
                row_max_ = row_min + box_height_ * (j + 1)
                sub_image = inverted[row_min_ : row_max_ + 1, col_min : col_max + 1]

                # Opening - remove "bridges" between boxes
                eroded = cv2.erode(sub_image, np.ones((10, 1)), iterations=1)
                dilated = cv2.dilate(eroded, np.ones((10, 1)), iterations=1)

                # Make a line to separate boxes (except for last box).
                if j != n_boxes_in_stack - 1:
                    dilated[-8, :] = 0

                # Overwrite original sub image with modified sub image
                inverted[row_min_ : row_max_ + 1, col_min : col_max + 1] = dilated

    # Detects blobs again. Now there should be no overlapping boxes
    # (`inverted` is an inverted binary image).
    blobs = _get_blobs(binary=inverted)

    anonymized_boxes = []
    heights = []
    for blob in blobs:
        if blob.area_bbox < box_area_min:
            # Blob is too small to be considered an anonymized box.
            break
        row_min, col_min, row_max, col_max = blob.bbox

        box_height = row_max - row_min
        heights.append(box_height)

        if (
            blob.area_filled / blob.area_bbox > box_accept_ratio
            and box_height > box_height_min
        ):
            assert 40 < box_height < 80, "Box height is not in expected range?"

            # `row_max - slight_shift` as text is usually in the top of the box.
            anonymized_box = {
                "coordinates": (row_min, col_min, row_max - slight_shift, col_max)
            }
            anonymized_box_refined = _refine_anonymized_box(anonymized_box, image)
            anonymized_boxes.append(anonymized_box_refined)
        else:
            # Blob is not a bounding box.
            pass

    return anonymized_boxes


def _has_neighboring_white_pixels(
    a: np.ndarray, b: np.ndarray, upper_bound: int = 2
) -> bool:
    """Checks if two arrays have neighboring white pixels.

    The two arrays are said to have neighboring white pixels,
    if `a` has a white pixel at index i and `b` has a white pixel at index j,
    and abs(i - j) < 2.

    The arrays must be of same size.

    The function is used in the `_refine_`.

    Args:
        a (np.ndarray):
            Array of 0s and 1s.
        b (np.ndarray):
            Array of 0s and 1s.
        upper_bound (int):
            The Manhattan distance between two white pixels of the arrays
            must be less than `upper_bound` for the arrays to be seen as having
            neighboring white pixels.

    Returns:
        bool:
            True if `a` and `b` have neighboring white pixels. False otherwise.
    """
    # Check if a and b have neighboring white pixels.
    assert len(a) == len(b), "Arrays must be of same size."
    a_indices = np.where(a != 0)[0]
    b_indices = np.where(b != 0)[0]
    distances = np.abs(a_indices - b_indices[:, None])  # Manhattan distance

    if len(distances) == 0:
        # Edge case if at least one of the arrays is all zeros.
        return False
    else:
        # Arrays are seen as having white pixel neighbors if the Manhattan distance is
        # between two white pixels of the arrays is less than 2.
        return distances.min() < upper_bound


def _not_only_white(a: np.ndarray) -> bool:
    """Checks if binary array is not only white.

    The function is used in the `_refine_`.

    Args:
        a (np.ndarray):
            Binary array.

    Returns:
        bool:
            True if binary array is not only white. False otherwise.

    """
    return not np.all(np.bool_(a))


def binarize(
    image: np.ndarray, threshold: int, val_min: int = 0, val_max: int = 1
) -> np.ndarray:
    """Binarize image.

    Args:
        image (np.ndarray):
            Image to be binarized.
        threshold (int):
            Threshold used to binarize the image.
        val_min (int):
            Value to be assigned to pixels below threshold.
        val_max (int):
            Value to be assigned to pixels above threshold.

    Returns:
        np.ndarray:
            Binarized image.
    """

    t = threshold
    binary = image.copy()
    binary[binary < t] = val_min
    binary[binary >= t] = val_max
    return binary


def _remove_boundary_noise(binary_crop: np.ndarray) -> np.ndarray:
    """Removes noise on the boundary of a an anonymized box.

    All white pixels in a perfect bounding box should be a pixel of a relevant character.
    Some images have white pixel defect at the boundary of the bounding box, and
    this function removes those white pixels.

    Args:
        binary_crop (np.ndarray):
            Cropped binary image showing the anonymized box.

    Returns:
        np.ndarray:
            Cropped binary image (anonymized box) with noise removed.
    """

    blobs = _get_blobs(binary_crop)
    blob = blobs[0]

    for blob in blobs:
        row_min, _, row_max, _ = blob.bbox
        height = row_max - row_min
        # make blob zeros

        # All letters have a height > ~ 22 pixels.
        # A blob that touches the boundary and doesn't cross the
        # middle row of the image is supposedly not a letter, but noise.
        if (
            height < 15
            and _touches_boundary(binary_crop, blob)
            and not _has_center_pixels(binary_crop, blob)
        ):
            # Remove blob
            coords = blob.coords
            binary_crop[coords[:, 0], coords[:, 1]] = 0
    return binary_crop


def _touches_boundary(binary_crop: np.ndarray, blob: RegionProperties) -> bool:
    """Check if blob touches the boundary of the image.

    Used in _remove_boundary_noise to determine if a blob is noise or not.

    Args:
        binary_crop (np.ndarray):
            Anonymized box.
            (used to get the non-zero boundaries of the image).
        blob (skimage.measure._regionprops._RegionProperties):
            A blob in the image.

    Returns:
        bool:
            True if blob touches the boundary of the image. False otherwise.
    """
    for boundary in [0, *binary_crop.shape]:
        if boundary in blob.bbox:
            return True
    return False


def _has_center_pixels(binary_crop: np.ndarray, blob: RegionProperties) -> bool:
    """Check if blob has pixels in the center (vertically) of the image.

    Used in _remove_boundary_noise to determine if a blob is noise or not.

    Args:
        binary_crop (np.ndarray):
            Anonymized box.
        blob (skimage.measure._regionprops._RegionProperties):
            A blob in the image.

    Returns:
        bool:
            True if blob has pixels in the center (vertically) of the image. False otherwise.
    """
    image_midpoint = binary_crop.shape[0] // 2
    row_min, _, row_max, _ = blob.bbox
    return row_min < image_midpoint < row_max


def _refine_anonymized_box(
    anonymized_box: dict, image: np.ndarray, threshold: int = 30
) -> dict:
    """Refines bounding box.

    Two scenarios:
        1. The box is too big, i.e. there is too much black space around the text.
        2. The box is too small, i.e. some letters are not fully included in the box.

    Args:
        anonymized_box (dict):
            Anonymized box with coordinates.
        image (np.ndarray):
            Image of the current page.
        threshold (int):
            Threshold used to binarize the image.

    Returns:
        anonymized_box (dict):
            Anonymized box with refined coordinates.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    binary = binarize(image=gray, threshold=threshold)

    row_min, col_min, row_max, col_max = anonymized_box["coordinates"]

    # +1 as slice is exclusive and boxes coordinates are inclusive.
    crop = gray[row_min : row_max + 1, col_min : col_max + 1]

    # If empty/black box, box should be ignored
    if crop.sum() == 0:
        return {}

    crop_binary = binarize(image=crop, threshold=threshold)
    crop_binary_ = _remove_boundary_noise(binary_crop=crop_binary.copy())
    binary[row_min : row_max + 1, col_min : col_max + 1] = crop_binary_

    # Refine box
    anonymized_box = _refine(binary=binary, anonymized_box=anonymized_box)
    return anonymized_box


def _refine(binary: np.ndarray, anonymized_box: List[int]) -> List[int]:
    """Refines bounding box.

    Args:
        binary (np.ndarray):
            Binary image of the current page.
        anonymized_box (tuple):
            Tuple with coordinates given by min/max row/col of box.

    Returns:
        anonymized_box (tuple):
            Tuple with refined coordinates given by min/max row/col of box.
    """

    row_min, col_min, row_max, col_max = anonymized_box["coordinates"]

    # Rows from top
    row = binary[row_min, col_min : col_max + 1]
    if not row.sum() == 0:
        anonymized_box = _refine_(
            top_bottom_left_right="top",
            expanding=True,
            binary=binary,
            anonymized_box=anonymized_box,
        )
    else:
        anonymized_box = _refine_(
            top_bottom_left_right="top",
            expanding=False,
            binary=binary,
            anonymized_box=anonymized_box,
        )

    row_min, col_min, row_max, col_max = anonymized_box["coordinates"]

    # Rows from bottom
    row = binary[row_max, col_min : col_max + 1]
    if not row.sum() == 0:
        anonymized_box = _refine_(
            top_bottom_left_right="bottom",
            expanding=True,
            binary=binary,
            anonymized_box=anonymized_box,
        )
    else:
        anonymized_box = _refine_(
            top_bottom_left_right="bottom",
            expanding=False,
            binary=binary,
            anonymized_box=anonymized_box,
        )

    row_min, col_min, row_max, col_max = anonymized_box["coordinates"]

    # Columns from left
    col = binary[row_min : row_max + 1, col_min]
    if not col.sum() == 0:
        anonymized_box = _refine_(
            top_bottom_left_right="left",
            expanding=True,
            binary=binary,
            anonymized_box=anonymized_box,
        )
    else:
        anonymized_box = _refine_(
            top_bottom_left_right="left",
            expanding=False,
            binary=binary,
            anonymized_box=anonymized_box,
        )

    row_min, col_min, row_max, col_max = anonymized_box["coordinates"]

    # Columns from right
    col = binary[row_min : row_max + 1, col_max]
    if not col.sum() == 0:
        anonymized_box = _refine_(
            top_bottom_left_right="right",
            expanding=True,
            binary=binary,
            anonymized_box=anonymized_box,
        )
    else:
        anonymized_box = _refine_(
            top_bottom_left_right="right",
            expanding=False,
            binary=binary,
            anonymized_box=anonymized_box,
        )

    return anonymized_box


def _refine_(
    top_bottom_left_right: str,
    expanding: bool,
    binary: np.ndarray,
    anonymized_box: dict,
) -> dict:
    """Refines bounding box in one direction.

    Args:
        top_bottom_left_right (str):
            String indicating which direction to refine.
        expanding (bool):
            Boolean indicating if the box should be expanded or shrunk.
        binary (np.ndarray):
            Binary image of the current page.
        anonymized_box (tuple):
            Tuple with coordinates given by min/max row/col of box.

    Returns:
        anonymized_box (tuple):
            Tuple with refined coordinates given by min/max row/col of box.
    """
    if expanding:
        row_col_next, row_col, anonymized_box = _next_row_col(
            top_bottom_left_right=top_bottom_left_right,
            expanding=expanding,
            binary=binary,
            anonymized_box=anonymized_box,
        )
        while _not_only_white(row_col_next) and _has_neighboring_white_pixels(
            row_col, row_col_next
        ):
            row_col_next, row_col, anonymized_box = _next_row_col(
                top_bottom_left_right=top_bottom_left_right,
                expanding=expanding,
                binary=binary,
                anonymized_box=anonymized_box,
            )
        row_min, col_min, row_max, col_max = anonymized_box["coordinates"]

        # Undo last change, because while loop has gone one step too far.
        if top_bottom_left_right == "top":
            row_min += 1
        elif top_bottom_left_right == "bottom":
            row_max -= 1
        elif top_bottom_left_right == "left":
            col_min += 1
        elif top_bottom_left_right == "right":
            col_max -= 1
        anonymized_box["coordinates"] = [row_min, col_min, row_max, col_max]

    else:
        row_col_next, _, anonymized_box = _next_row_col(
            top_bottom_left_right=top_bottom_left_right,
            expanding=False,
            binary=binary,
            anonymized_box=anonymized_box,
        )
        while row_col_next.sum() == 0:
            row_col_next, _, anonymized_box = _next_row_col(
                top_bottom_left_right=top_bottom_left_right,
                expanding=False,
                binary=binary,
                anonymized_box=anonymized_box,
            )
    return anonymized_box


def _next_row_col(
    top_bottom_left_right: str,
    expanding: bool,
    binary: np.ndarray,
    anonymized_box: dict,
) -> tuple:
    """Helper function for _refine_.

    This function is used to get the next row/column when refining a bounding box.

    Args:
        top_bottom_left_right (str):
            String indicating which direction is being refined.
        expanding (bool):
            Boolean indicating if the box is being expanded or shrunk.
        binary (np.ndarray):
            Binary image of the current page.
        anonymized_box (dict):
            Anonymized box with coordinates.

    Returns:
        row_col_next (np.ndarray):
            Next row/column.
        row_col (np.ndarray):
            Current row/column.
        anonymized_box (dict):
            Anonymized box with refined coordinates.
    """
    row_min, col_min, row_max, col_max = anonymized_box["coordinates"]
    if top_bottom_left_right == "top":
        row_col = binary[row_min, col_min : col_max + 1]

        change = -1 if expanding else 1
        row_min += change
        row_col_next = binary[row_min, col_min : col_max + 1]
    elif top_bottom_left_right == "bottom":
        row_col = binary[row_max, col_min : col_max + 1]

        change = 1 if expanding else -1
        row_max += change
        row_col_next = binary[row_max, col_min : col_max + 1]
    elif top_bottom_left_right == "left":
        row_col = binary[row_min : row_max + 1, col_min]

        change = -1 if expanding else 1
        col_min += change
        row_col_next = binary[row_min : row_max + 1, col_min]
    elif top_bottom_left_right == "right":
        row_col = binary[row_min : row_max + 1, col_max]

        change = 1 if expanding else -1
        col_max += change
        row_col_next = binary[row_min : row_max + 1, col_max]

    anonymized_box["coordinates"] = [row_min, col_min, row_max, col_max]
    return row_col_next, row_col, anonymized_box


def _split_box(crop: np.ndarray, anonymized_box: dict) -> List[dict]:
    """Split box into multiple boxes - one for each word.

    Args:
        crop (np.ndarray):
            Image of the box.
        anonymized_box (dict):
            Anonymized box with coordinates.

    Returns:
        List[dict]:
            List of anonymized boxes - one for each word of the input box.
    """
    split_indices = _get_split_indices(crop=crop)
    if not split_indices:
        return [anonymized_box]
    else:
        anonymized_boxes = []
        row_min, col_min, row_max, col_max = anonymized_box["coordinates"]
        first_box = {
            "coordinates": (row_min, col_min, row_max, col_min + split_indices[0])
        }

        anonymized_boxes.append(first_box)

        # Get box in between first and last box
        if len(split_indices) > 1:

            for split_index_1, split_index_2 in zip(
                split_indices[:-1], split_indices[1:]
            ):
                anonymized_box_ = {
                    "coordinates": (
                        row_min,
                        col_min + split_index_1 + 1,
                        row_max,
                        col_min + split_index_2,
                    )
                }
                anonymized_boxes.append(anonymized_box_)

        # Get last box
        last_box = {
            "coordinates": (row_min, col_min + split_indices[-1] + 1, row_max, col_max)
        }
        anonymized_boxes.append(last_box)
    return anonymized_boxes


def _get_split_indices(
    crop: np.ndarray, threshold_binarize: int = 100, threshold_gap: int = 15
) -> List[int]:
    """Split box into multiple boxes - one for each word.

    Used in the function `_split_box`.

    Arg:
        crop (np.ndarray):
            Image of the box.
        threshold_binarize (int):
            Threshold for binarization.
        threshold_gap (int):
            Minimum size of gap between words,
            before splitting.

    Returns:
        List[int]:
            List of indices where the box should be split.
    """
    inverted = cv2.bitwise_not(crop)

    binary = binarize(inverted, threshold=threshold_binarize)

    # One bool value for each column.
    # True if all pixels in column are white.
    booled = binary.all(axis=0)

    split_indices = []

    gap_length = 0
    for i, bool_value in enumerate(booled):
        if bool_value:
            gap_length += 1
        else:
            if gap_length > threshold_gap:
                split_idx = i - 1 - gap_length // 2
                if split_idx > 0:
                    split_indices.append(split_idx)
            gap_length = 0
    return split_indices


def _add_boundary(image: np.ndarray) -> np.ndarray:
    """Add boundary to image.

    EasyOCR seems to give the best results when the text is surrounded by black pixels.

    Args:
        image (np.ndarray):
            Image to add boundary to.

    Returns:
        np.ndarray:
            Image with boundary.
    """
    padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = image
    return padded


def _read_text_with_tika(pdf_path: str) -> str:
    """Read text from pdf with tika.

    Args:
        pdf_path (str):
            Path to pdf.

    Returns:
        str:
            Text from pdf.
    """
    request_options = {"timeout": 300}
    text = ""
    result = parser.from_file(pdf_path, requestOptions=request_options)
    if result["status"] == 200:
        text = result["content"]
    return text.strip()


# This class is not used, but is kept for future reference.
class PDFTextExtractor:
    @staticmethod
    def tika(pdf_path: str):
        request_options = {"timeout": 300}
        text = ""
        result = parser.from_file(pdf_path, requestOptions=request_options)
        if result["status"] == 200:
            text = result["content"]
        return text.strip()

    @staticmethod
    def pypdf(pdf_path: str):
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        return text.strip()

    @staticmethod
    def pypdfium2(pdf_path: str):
        pdf = pdfium.PdfDocument(pdf_path)

        text = ""
        for page in pdf:
            text_page = page.get_textpage()
            text += text_page.get_text_range() + "\n\n"
        return text.strip()

    @staticmethod
    def pymupdf(pdf_path: str):
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text() + "\n\n"
        return text.strip()

    @staticmethod
    def easyocr(
        image: np.ndarray,
        gpu: bool,
        reader: easyocr.Reader = None,
        languages: List[str] = ["da"],
    ):
        if reader is None:
            reader = easyocr.Reader([languages], gpu=gpu)
        result = reader.readtext(image)
        # Result should then be sorted w.r.t how the text is read.

    @staticmethod
    def tesseract(pdf_path: str, first_page_only: bool = False):
        images = convert_from_path(pdf_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image, lang="dan") + "\n\n"
            if first_page_only:
                return text.strip()
        return text.strip()
