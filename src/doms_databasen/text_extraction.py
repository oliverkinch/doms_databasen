from logging import getLogger

import easyocr
import numpy as np
from pdf2image import convert_from_path
from tika import parser

from typing import List

from pypdf import PdfReader
import pypdfium2 as pdfium
import fitz
import pytesseract

import cv2
import re
from skimage import measure
from skimage.filters import rank
import skimage

from PIL import Image


def save_numpy_image(image, path):
    image = Image.fromarray(image)
    image.save(path)


logger = getLogger(__name__)


def _remove_logo(image):
    # Remove logo top right corner
    c = 1500
    r = 500
    image[:r, c:, :] = 255
    return image


def extract_text_easyocr(
    pdf_path: str, dpi: int = 300, gpu: bool = False, max_y_difference: int = 15
) -> str:
    # Maybe just give images as input, instead of pdf_path?
    # images = map(np.array, convert_from_path(pdf_path, dpi=dpi))
    images = map(np.array, convert_from_path(pdf_path, dpi=dpi))
    reader = easyocr.Reader(["da"], gpu=gpu)  # Load only once when script starts?
    pdf_text_extractor = PDFTextExtractor()
    text = ""

    # Try both methods on first page.
    # I have not seen a single pdf that uses both methods or first has anonymization
    # on e.g. second page.
    underlines_anon = True
    box_anon = True

    for i, image in enumerate(images):

        # Remove logo top right corner
        if i == 0:
            image = _remove_logo(image)

        if box_anon:
            anonymized_boxes = find_anonymized_boxes(image=image.copy())
            anonymized_boxes_with_text = [
                get_text_from_anonymized_box(image, box, reader)
                for box in anonymized_boxes
            ]
        else:
            anonymized_boxes_with_text = []

        if underlines_anon:
            anonymized_boxes_from_underlines, underlines = line_anonymization_to_boxes(
                image=image.copy()
            )
            anonymized_boxes_from_underlines_with_text = [
                get_text_from_anonymized_box(image, box, reader, invert=True)
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
            box_anon = bool(anonymized_boxes_with_text)
            underlines_anon = bool(anonymized_boxes_from_underlines_with_text)
            if not box_anon and not underlines_anon:
                # No indication of anonymization on first page.
                # Then try use Tika
                text = pdf_text_extractor.tika(str(pdf_path))
                if text:
                    return text
                else:
                    # pdf is supposedly then scanned, e.g. worst case pdf - scanned and no indications of anonymizations
                    # Then use easrocr
                    anonymized_boxes_with_text = []
                    anonymized_boxes_from_underlines_with_text = []
                    underlines = []

        all_anonymized_boxes_with_text = (
            anonymized_boxes_with_text + anonymized_boxes_from_underlines_with_text
        )

        # Remove anonymized boxes from image
        image_processed = process_image(
            image.copy(), all_anonymized_boxes_with_text, underlines
        )
        save_cv2_image_tmp(image_processed)

        # Read core text of image
        result = reader.readtext(image_processed)

        # Make boxes on same format as anonymized boxes
        boxes = [_change_box_format(box) for box in result]

        # Merge all boxes
        all_boxes = boxes + all_anonymized_boxes_with_text
        page_text = get_text_from_boxes(all_boxes, max_y_difference)
        if i == 0:
            text += f"{page_text}\n\n"
        else:
            text += f"\n\n{page_text}\n\n"

    return text.strip()


def get_blobs(binary: np.ndarray) -> list:
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


def line_anonymization_to_boxes(image):
    """
    Alle understegninger findes og der laves boks rundt om ordet over understregningen.
    """
    save_cv2_image_tmp(image)

    image_inverted = cv2.bitwise_not(image)
    save_cv2_image_tmp(image_inverted)

    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    save_cv2_image_tmp(inverted)

    # Opening
    eroded = cv2.erode(inverted, np.ones((1, 50)), iterations=1)
    dilated = cv2.dilate(eroded, np.ones((1, 50)), iterations=1)
    save_cv2_image_tmp(dilated)

    binary = dilated.copy()
    thresh = 200
    binary[binary < thresh] = 0
    binary[binary >= thresh] = 1
    save_cv2_image_tmp(binary)

    labels = measure.label(binary, connectivity=1)
    blobs = measure.regionprops(labels)
    blobs = sorted(blobs, key=lambda blob: blob.area_bbox, reverse=True)

    anonymized_boxes = []
    underlines = []
    for i, blob in enumerate(blobs):
        row_min, col_min, row_max, col_max = blob.bbox
        # Remove blob from image

        # Blobs must be perfectly rectangular
        height = row_max - row_min

        if blob.area == blob.area_bbox and 1 < height < 7:
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

            (
                box_row_min_,
                box_col_min_,
                box_row_max_,
                box_col_max_,
            ) = anonymized_box_refined["coordinates"]
            # If box above underline is not empty, then add box to anonymized_boxes
            # Also save coordinates for underline, such that it can be removed later.
            if (
                img[
                    box_row_min_ : box_row_max_ + 1, box_col_min_ : box_col_max_ + 1, :
                ].sum()
                > 0
            ):
                anonymized_boxes.append(anonymized_box_refined)
                underlines.append(blob.bbox)

            save_cv2_image_tmp(img)

    return anonymized_boxes, underlines


def _on_same_line(y: int, y_prev: int, max_y_difference: int) -> bool:
    """Helper function to determine if two bounding boxes are on the same line.

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


def process_image(image, anonymized_boxes, underlines):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_cv2_image_tmp(gray)
    # Make boxes black
    for box in anonymized_boxes:
        row_min, col_min, row_max, col_max = box["coordinates"]
        gray[row_min : row_max + 1, col_min : col_max + 1] = 0
        save_cv2_image_tmp(gray)

    for underline in underlines:
        row_min, col_min, row_max, col_max = underline
        gray[row_min : row_max + 1, col_min : col_max + 1] = 0
    save_cv2_image_tmp(gray)

    inverted = cv2.bitwise_not(gray)
    save_cv2_image_tmp(inverted)

    dilated = cv2.dilate(inverted, np.ones((2, 2)))
    save_cv2_image_tmp(dilated)

    # Maybe just return at this point?
    # Maybe slighty better result not returning at this point.

    # Image has been inverted, such that it is white text on black background.
    # However this also means that the boxes currently are white.
    # Make them black again + p small padding because of dilation.
    filled = dilated.copy()

    # Perform the flood fill
    # cv2.floodFill(image=filled, mask=None, seedPoint=(col_min, row_min), newVal=0, loDiff=(loDiff, loDiff, loDiff), upDiff=(upDiff, upDiff, upDiff))
    tolerance = 5  # Remove everything that is not black
    # Square footprint
    # footprint = np.ones((3, 3))
    for box in anonymized_boxes:

        row_min, col_min, row_max, col_max = box["coordinates"]
        save_cv2_image_tmp(filled)
        seed_point = (row_min, col_min)
        print(f"Seed point value: {filled[seed_point]}")

        # filled_ = filled.copy()
        # filled_[seed_point] = 0
        # save_cv2_image_tmp(filled_)

        filled = skimage.segmentation.flood_fill(
            image=filled,
            seed_point=seed_point,
            new_value=0,
            connectivity=1,
            tolerance=tolerance,
        )
        save_cv2_image_tmp(filled)

    for underline in underlines:
        row_min,
        row_min, col_min, row_max, col_max = underline
        # Draw seed point
        filled = skimage.segmentation.flood_fill(
            image=filled,
            seed_point=(row_min, col_min),
            new_value=0,
            connectivity=1,
            tolerance=tolerance,
        )

    save_cv2_image_tmp(filled)

    # Opening
    eroded = cv2.erode(filled, np.ones((2, 2)), iterations=1)
    dilated2 = cv2.dilate(eroded, np.ones((2, 2)), iterations=1)
    save_cv2_image_tmp(dilated2)

    # Not found yet to have much effect.
    # sharpened = np.array(skimage.filters.unsharp_mask(dilated, radius=50, amount=3.8), dtype=np.uint8) * 255
    # save_cv2_image_tmp(sharpened)

    return dilated2


def get_text_from_boxes(boxes, max_y_difference):
    # Sort w.r.t y coordinate.
    boxes_y_sorted = sorted(boxes, key=lambda box: _middle_y_cordinate(box))

    # Group bounding boxes that are on the same line.
    # E.g. the variable lines will be a list of lists, where each list contains
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

    # Now sort the lines w.r.t x coordinate.
    # The lines should as a result be sorted w.r.t how the text is read.
    for i, line in enumerate(lines):
        lines[i] = sorted(line, key=lambda box: _left_x_cordinate(box))

        # Each bounding box on a line is joined together with a space,
        # and the lines of text are joined together with \n.
        page_text = "\n".join(
            [" ".join([box["text"] for box in line]) for line in lines]
        )
    return page_text.strip()


def _left_x_cordinate(box):
    _, col_min, _, _ = box["coordinates"]
    return col_min


def _middle_y_cordinate(box):
    row_min, _, row_max, _ = box["coordinates"]
    return (row_min + row_max) / 2


def save_cv2_image_tmp(image):
    if image.max() < 2:
        image = image * 255
    cv2.imwrite("tmp.png", image)


def _change_box_format(easyocr_box):
    tl, tr, _, bl = easyocr_box[0]  # easyocr uses (x, y) format
    row_min, col_min, row_max, col_max = tl[1], tl[0], bl[1], tr[0]
    text = easyocr_box[1]
    box = {"coordinates": (row_min, col_min, row_max, col_max), "text": text}
    return box


def get_text_from_anonymized_box(
    image, anonymized_box, reader, theshold=0.3, invert=False
):

    row_min, col_min, row_max, col_max = anonymized_box["coordinates"]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_cv2_image_tmp(image)

    if invert:
        gray = cv2.bitwise_not(gray)
    save_cv2_image_tmp(gray)

    crop = gray[row_min : row_max + 1, col_min : col_max + 1]
    save_cv2_image_tmp(crop)

    split_indices = _split_box(crop)
    anonymized_boxes = []
    if split_indices:
        # Get first box
        first_box = {"coordinates": (row_min, col_min, row_max, col_min + split_indices[0])}
        
        anonymized_boxes.append(first_box)
        
        # Get box in between first and last box
        if len(split_indices) > 1:
            
            for i, (split_index_1, split_index_2) in enumerate(zip(split_indices[:-1], split_indices[1:])):
                anonymized_box_ = {"coordinates": (row_min, col_min + split_index_1 + 1, row_max, col_min + split_index_2)}
                anonymized_boxes.append(anonymized_box_)

        # Get last box
        last_box = {"coordinates": (row_min, col_min + split_indices[-1] + 1, row_max, col_max)}
        anonymized_boxes.append(last_box)
    else:
        anonymized_boxes.append(anonymized_box)

    texts = []
    # anonymized_boxes_ are sorted left to right
    for anonymized_box_ in anonymized_boxes:
        row_min, col_min, row_max, col_max = anonymized_box_["coordinates"]

        crop = gray[row_min : row_max + 1, col_min : col_max + 1]
        save_cv2_image_tmp(crop)

        crop_boundary = _add_boundary(crop)

        # If length of box is short, then there are probably < 2 letters in the box.
        # In this case, scale the image up.
        box_length = col_max - col_min
        scale = 1 if box_length > 50 else 2

        scaled = cv2.resize(crop_boundary, (0, 0), fx=scale, fy=scale)
        save_cv2_image_tmp(scaled)

        # Increase size of letters
        dilated = cv2.dilate(scaled, np.ones((2, 2)))
        save_cv2_image_tmp(dilated)

        dilated1_boundary = _add_boundary(dilated)
        save_cv2_image_tmp(dilated1_boundary)

        sharpened = (
            np.array(
                skimage.filters.unsharp_mask(dilated1_boundary, radius=20, amount=1.9),
                dtype=np.uint8,
            )
            * 255
        )
        save_cv2_image_tmp(sharpened)

        inverted = cv2.bitwise_not(sharpened)
        save_cv2_image_tmp(inverted)
        save_cv2_image_tmp(sharpened)

        result = reader.readtext(sharpened)

        result_inverted = reader.readtext(inverted)

        print("Result:")
        print(result)
        print("Result inverted:")
        print(result_inverted)
        if len(result) == 0:
            text = ""
        else:

            text = " ".join([box[1] for box in result if box[2] > theshold])
        # map typical errors as `:` -> `.`
        texts.append(text)

    text_all = " ".join(texts).strip()
    # if len(result) == 0:
    #     text = ""
    # else:
    #     # Hopefully only one box in most case.
    #     # There can be multiple boxes,
    #     # if fx there are text on two lines,
    #     # in the anonymized box.
    #     # Or simply if easyocr splits a line into multiple boxes.
    #     # Only accept boxes with confidence > threshold
    #     text = " ".join([box[1] for box in result if box[2] > theshold])

    anonymized_box["text"] = f"<anonym>{text_all}</anonym>" if text else ""
    print("Text:")
    print(text_all)
    # print(result)
    return anonymized_box


def find_anonymized_boxes(
    image, box_height_split=120, box_area_min=2000, slight_shift=5
):
    WHITE = 255
    BLACK = 0
    img = image.copy()
    print(f"Image size: {img.shape}")
    save_cv2_image_tmp(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    save_cv2_image_tmp(gray)

    # Mean filter to make text outside boxes
    # brigther than color of boxes.
    footprint = np.ones((1, 15))
    averaged = rank.mean(gray, footprint=footprint)
    save_cv2_image_tmp(averaged)

    binary = averaged.copy()
    thresh = 1
    binary[binary < thresh] = BLACK
    binary[binary >= thresh] = WHITE
    save_cv2_image_tmp(binary)

    inverted = cv2.bitwise_not(binary)
    save_cv2_image_tmp(inverted)

    # opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, np.ones((10, 10)))
    # save_cv2_image_tmp(opened)

    labels = measure.label(inverted, connectivity=1)
    blobs = measure.regionprops(labels)
    blobs = sorted(blobs, key=lambda blob: blob.area_bbox, reverse=True)

    # First split multiple boxes into separate boxes
    for i, blob in enumerate(blobs):
        if blob.area_bbox < box_area_min:
            break
        row_min, col_min, row_max, col_max = blob.bbox
        box_height = row_max - row_min
        if box_height > box_height_split:
            # Split into multiple boxes (horizontal split)
            n_boxes = box_height // (box_height_split // 2)
            box_height_ = box_height // n_boxes
            for j in range(n_boxes):
                row_min_ = row_min + box_height_ * j
                row_max_ = row_min + box_height_ * (j + 1)
                sub_image = inverted[row_min_:row_max_, col_min:col_max]
                save_cv2_image_tmp(sub_image)

                # sub_image[-5, :] = 0
                # save_cv2_image_tmp(sub_image)

                # Opening
                eroded = cv2.erode(sub_image, np.ones((10, 1)), iterations=1)
                save_cv2_image_tmp(eroded)
                dilated = cv2.dilate(eroded, np.ones((10, 1)), iterations=1)
                save_cv2_image_tmp(dilated)

                dilated[-10:, :] = 0
                save_cv2_image_tmp(dilated)

                inverted[row_min_:row_max_, col_min:col_max] = dilated
                save_cv2_image_tmp(inverted)

                # inverted[row, :] = 0

                # sub_image = inverted[row_min:row_max_, col_min:col_max]

                # eroded = cv2.erode(sub_image, np.ones((10, 1)), iterations=1)
                # save_cv2_image_tmp(eroded)

    # Detects blobs again. Now there should be no overlapping boxes.
    # Please.
    labels = measure.label(inverted, connectivity=1)
    blobs = measure.regionprops(labels)
    blobs = sorted(blobs, key=lambda blob: blob.area_bbox, reverse=True)

    anonymized_boxes = []
    heights = []
    for i, blob in enumerate(blobs):
        if blob.area_bbox < box_area_min:
            break
        row_min, col_min, row_max, col_max = blob.bbox

        box_height = row_max - row_min
        heights.append(box_height)

        # if box height is above 120, split box in two
        if blob.area_filled / blob.area_bbox > 0.4 and box_height > 10:
            if box_height > box_height_split:

                for j in range(n_boxes):
                    # Text is in general in the top of the box.
                    # slight_shift_ = slight_shift if not j == 0 else 0

                    # tl_ = (col_min, row_min + box_height_ * j - slight_shift_)
                    # tr_ = (col_max, row_min + box_height_ * j - slight_shift_)

                    # slight_shift_ = slight_shift if not j == n_boxes - 1 else 0
                    # br_ = (col_max, row_min + box_height_ * (j + 1) - slight_shift_)
                    # bl_ = (col_min, row_min + box_height_ * (j + 1) - slight_shift_)

                    # anonymized_box = {"coordinates": (tl_, tr_, br_, bl_)}
                    # Slight shift to row_max, as the text in general
                    # is in the top of the box.
                    anonymized_box = {
                        "coordinates": (
                            row_min,
                            col_min,
                            row_max - slight_shift,
                            col_max,
                        )
                    }
                    anonymized_box_ = _refine_anonymized_box(anonymized_box, image)
                    anonymized_boxes.append(anonymized_box_)

            else:

                anonymized_box = {
                    "coordinates": (row_min, col_min, row_max - slight_shift, col_max)
                }
                anonymized_box_ = _refine_anonymized_box(anonymized_box, image)

                anonymized_boxes.append(anonymized_box_)

        else:
            # Blob is not a bounding box.
            pass

    return anonymized_boxes


def _has_neighboring_white_pixels(a: np.ndarray, b: np.ndarray):
    """

    The arrays must be of same size.

    """
    # Check if a and b have neighboring white pixels.
    assert len(a) == len(b), "Arrays must be of same size."
    a_indices = np.where(a != 0)[0]
    b_indices = np.where(b != 0)[0]
    distances = np.abs(a_indices - b_indices[:, None])  # Manhattan distance

    if len(distances) == 0:
        # Edge case if at least one of the arrays are all zeros.
        return False
    else:
        # Arrays are seen as having white pixel neighbors if the Manhattan distance is
        # between two white pixels of the arrays is less than 2.
        return distances.min() < 2


def _not_only_white(a: np.ndarray):
    return not np.all(np.bool_(a))


def binarize(image, threshold):
    t = threshold
    binary = image.copy()
    binary[binary < t] = 0
    binary[binary >= t] = 1
    return binary


def _remove_boundary_noise(binary: np.ndarray):
    """Removes noise on the boundary of a binary image.

    All white pixels in a perfect bounding box should be a pixel of a relevant character.
    Some images have white pixel defect at the boundary of the bounding box, and
    this function removes those white pixels.

    Args:
        binary (np.ndarray):
            Binary image of the current page.

    Returns:
        np.ndarray:
            Binary image of the current page with the boundary noise removed.
    """

    blobs = get_blobs(binary)
    blob = blobs[0]

    for blob in blobs:
        binary_edit = binary.copy()
        row_min, col_min, row_max, col_max = blob.bbox
        height = row_max - row_min
        # make blob zeros
        binary_edit[row_min:row_max, col_min:col_max] = 0
        save_cv2_image_tmp(binary_edit)

        # All letters have a height > ~ 22 pixels.
        # A blob that touches the boundary and doesn't cross the
        # middle row of the image is supposedly not a letter, but noise.
        if (
            height < 15
            and _touches_boundary(binary, blob)
            and not _has_center_pixels(binary, blob)
        ):
            # Remove blob
            coords = blob.coords
            binary[coords[:, 0], coords[:, 1]] = 0
    save_cv2_image_tmp(binary)
    return binary


def _touches_boundary(binary, blob):
    """Check if blob touches the boundary of the image.

    Args:
        binary (np.ndarray):
            Binary image of the current page
            (used to get the non-zero boundaries of the image).
        blob (skimage.measure._regionprops._RegionProperties):
            A blob in the image.

    Returns:
        bool:
            True if blob touches the boundary of the image. False otherwise.
    """
    for boundary in [0, *binary.shape]:
        if boundary in blob.bbox:
            return True
    return False


def _has_center_pixels(binary, blob):
    image_midpoint = binary.shape[0] // 2
    row_min, _, row_max, _ = blob.bbox
    return row_min < image_midpoint < row_max


def _refine_anonymized_box(anonymized_box, image, threshold: int = 30):

    row_min, col_min, row_max, col_max = anonymized_box["coordinates"]

    box_coordinates_unpacked = [row_min, row_max, col_min, col_max]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_cv2_image_tmp(gray)
    binary = binarize(gray, threshold)
    save_cv2_image_tmp(binary)

    # Eksempel med venstre side af box

    # +1 fordi at slice er exclusive og boxes coordinates er inclusive.
    crop = gray[row_min : row_max + 1, col_min : col_max + 1]
    save_cv2_image_tmp(crop)

    # If empty/black box, text is empty.
    if crop.sum() == 0:
        anonymized_box["text"] = ""
        return anonymized_box

    crop_binary = binarize(crop, threshold)
    save_cv2_image_tmp(crop_binary)

    crop_binary_ = _remove_boundary_noise(crop_binary.copy())
    save_cv2_image_tmp(crop_binary_)
    binary[row_min : row_max + 1, col_min : col_max + 1] = crop_binary_
    save_cv2_image_tmp(binary)

    # Refine box
    binary, box_coordinates_unpacked = _refine(
        binary=binary, box_coordinates_unpacked=box_coordinates_unpacked
    )
    row_min, row_max, col_min, col_max = box_coordinates_unpacked
    save_cv2_image_tmp(binary[row_min : row_max + 1, col_min : col_max + 1])

    anonymized_box["coordinates"] = (row_min, col_min, row_max, col_max)
    return anonymized_box


def _refine(binary: np.ndarray, box_coordinates_unpacked: List[int]):
    """Refines bounding box.

    Two scenarios:
        1. The box is too big, i.e. there is too much black space around the text.
        2. The box is too small, i.e. some letters are not fully included in the box.

    Args:
        binary (np.ndarray):
            Binary image of the current page.
        box_coordinates_unpacked (tuple):
            Tuple with coordinates given by min/max row/col of box.

    Returns:
        binary (np.ndarray):
            Binary image of the current page with changes made w.r.t the bounding box
            (this is output could be optional, as it only used to visualize the changes made).
        box_coordinates_unpacked (tuple):
            Tuple with refined coordinates given by min/max row/col of box.
    """

    row_min, row_max, col_min, col_max = box_coordinates_unpacked
    save_cv2_image_tmp(binary[row_min : row_max + 1, col_min : col_max + 1])

    # Rows from top
    row = binary[row_min, col_min : col_max + 1]
    if not row.sum() == 0:
        binary, box_coordinates_unpacked = _refine_(
            top_bottom_left_right="top",
            expanding=True,
            binary=binary,
            box_coordinates_unpacked=box_coordinates_unpacked,
        )
    else:
        binary, box_coordinates_unpacked = _refine_(
            top_bottom_left_right="top",
            expanding=False,
            binary=binary,
            box_coordinates_unpacked=box_coordinates_unpacked,
        )

    row_min, row_max, col_min, col_max = box_coordinates_unpacked
    save_cv2_image_tmp(binary[row_min : row_max + 1, col_min : col_max + 1])

    # Rows from bottom
    row = binary[row_max, col_min : col_max + 1]
    if not row.sum() == 0:
        binary, box_coordinates_unpacked = _refine_(
            top_bottom_left_right="bottom",
            expanding=True,
            binary=binary,
            box_coordinates_unpacked=box_coordinates_unpacked,
        )
    else:
        binary, box_coordinates_unpacked = _refine_(
            top_bottom_left_right="bottom",
            expanding=False,
            binary=binary,
            box_coordinates_unpacked=box_coordinates_unpacked,
        )

    row_min, row_max, col_min, col_max = box_coordinates_unpacked
    save_cv2_image_tmp(binary[row_min : row_max + 1, col_min : col_max + 1])

    # Columns from left
    col = binary[row_min : row_max + 1, col_min]
    if not col.sum() == 0:
        binary, box_coordinates_unpacked = _refine_(
            top_bottom_left_right="left",
            expanding=True,
            binary=binary,
            box_coordinates_unpacked=box_coordinates_unpacked,
        )
    else:
        binary, box_coordinates_unpacked = _refine_(
            top_bottom_left_right="left",
            expanding=False,
            binary=binary,
            box_coordinates_unpacked=box_coordinates_unpacked,
        )

    row_min, row_max, col_min, col_max = box_coordinates_unpacked
    save_cv2_image_tmp(binary[row_min : row_max + 1, col_min : col_max + 1])

    # Columns from right
    col = binary[row_min : row_max + 1, col_max]
    if not col.sum() == 0:
        binary, box_coordinates_unpacked = _refine_(
            top_bottom_left_right="right",
            expanding=True,
            binary=binary,
            box_coordinates_unpacked=box_coordinates_unpacked,
        )
    else:
        binary, box_coordinates_unpacked = _refine_(
            top_bottom_left_right="right",
            expanding=False,
            binary=binary,
            box_coordinates_unpacked=box_coordinates_unpacked,
        )

    row_min, row_max, col_min, col_max = box_coordinates_unpacked
    save_cv2_image_tmp(binary[row_min : row_max + 1, col_min : col_max + 1])

    return binary, box_coordinates_unpacked


def _refine_(
    top_bottom_left_right: str, expanding: bool, binary, box_coordinates_unpacked
):
    """Refines bounding box in one direction.

    Args:
        top_bottom_left_right (str):
            String indicating which direction to refine.
        expanding (bool):
            Boolean indicating if the box should be expanded or shrunk.
        binary (np.ndarray):
            Binary image of the current page.
        box_coordinates_unpacked (tuple):
            Tuple with coordinates given by min/max row/col of box.

    Returns:
        binary (np.ndarray):
            Binary image of the current page with changes made w.r.t the bounding box
            (this is output could be optional, as it only used to visualize the changes made).
        box_coordinates_unpacked (tuple):
            Tuple with refined coordinates given by min/max row/col of box.
    """
    if expanding:
        row_col_next, row_col, box_coordinates_unpacked = _next_row_col(
            top_bottom_left_right=top_bottom_left_right,
            expanding=expanding,
            binary=binary,
            box_coordinates_unpacked=box_coordinates_unpacked,
        )
        while _not_only_white(row_col_next) and _has_neighboring_white_pixels(
            row_col, row_col_next
        ):
            row_col_next, row_col, box_coordinates_unpacked = _next_row_col(
                top_bottom_left_right=top_bottom_left_right,
                expanding=expanding,
                binary=binary,
                box_coordinates_unpacked=box_coordinates_unpacked,
            )
        row_min, row_max, col_min, col_max = box_coordinates_unpacked
        # Make the first row/col not accepted black.
        if top_bottom_left_right == "top":
            binary[row_min, :] = 0
        elif top_bottom_left_right == "bottom":
            binary[row_max, :] = 0
        elif top_bottom_left_right == "left":
            binary[:, col_min] = 0
        elif top_bottom_left_right == "right":
            binary[:, col_max] = 0

    else:
        row_col_next, _, box_coordinates_unpacked = _next_row_col(
            top_bottom_left_right=top_bottom_left_right,
            expanding=False,
            binary=binary,
            box_coordinates_unpacked=box_coordinates_unpacked,
        )
        while row_col_next.sum() == 0:
            row_col_next, _, box_coordinates_unpacked = _next_row_col(
                top_bottom_left_right=top_bottom_left_right,
                expanding=False,
                binary=binary,
                box_coordinates_unpacked=box_coordinates_unpacked,
            )
    return binary, box_coordinates_unpacked


def _next_row_col(
    top_bottom_left_right,
    expanding: bool,
    binary,
    box_coordinates_unpacked,
):
    row_min, row_max, col_min, col_max = box_coordinates_unpacked
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

    box_coordinates_unpacked = [row_min, row_max, col_min, col_max]
    return row_col_next, row_col, box_coordinates_unpacked


def _split_box(crop_refined, threshold=100, gap_threshold=15):
    """
    Split box into two boxes if there is a gap larger than threshold (15 pixels - hardcoded)
    Could change to have multiple splits?
    E.g. just for every gap larger than threshold, return a split index.
    The function would then return a list of split indices, instead of just one.
    """
    inverted = cv2.bitwise_not(crop_refined)
    save_cv2_image_tmp(inverted)

    binary = binarize(inverted, threshold=threshold)
    save_cv2_image_tmp(binary)
    booled = binary.all(axis=0)

    split_indices = []

    consecutive_sum = 0
    for i, b in enumerate(booled):
        if b:
            consecutive_sum += 1
        else:
            if consecutive_sum > gap_threshold:
                split_idx = i - consecutive_sum // 2
                split_indices.append(split_idx)
            consecutive_sum = 0
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
