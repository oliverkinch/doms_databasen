"""Code to read text from PDFs obtained from domsdatabasen.dk"""

from logging import getLogger
from typing import List

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
    BOX_HEIGHT_LOWER_BOUND,
    BOX_LENGTH_SCALE_THRESHOLD,
    DPI,
    TOLERANCE_FLOOD_FILL,
)

logger = getLogger(__name__)


class PDFTextReader:
    """Class for reading text from PDFs obtained from domsdatabasen.dk

    Args:
        config (DictConfig):
            Config file

    Attributes:
        config (DictConfig):
            Config file
        reader (easyocr.Reader):
            Easyocr reader
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.reader = easyocr.Reader(["da"], gpu=config.gpu)

    def extract_text(self, pdf_path: str) -> str:
        """Extracts text from a PDF using easyocr or tika.

        Tika is only used if there are no indications of anonymization.

        Some text is anonymized with boxes, and some text is anonymized with underlines.
        This function tries to find these anonymization, read the anonymized text,
        and then remove the anonymized text from the image before
        reading the rest of the text with easyocr.

        Args:
            pdf_path (str):
                Path to PDF.

        Returns:
            pdf_text (str):
                Text from PDF.
        """

        if self.config.image_idx:
            # Used to test on a single page
            images = map(
                np.array,
                convert_from_path(
                    pdf_path,
                    dpi=DPI,
                    first_page=self.config.image_idx,
                    last_page=self.config.image_idx,
                ),
            )
        else:
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
            i = self.config.image_idx or i
            # Log info about which anonymization methods are used in the PDF.
            if i == 1:
                if not box_anonymization:
                    logger.info(self.config.message_pdf_has_no_anonymized_boxes)
                if not underlines_anonymization:
                    logger.info(self.config.message_pdf_has_no_underline_anonymizations)

            if i == 0:
                # Remove logo top right corner
                image = self._remove_logo(image=image)

            if box_anonymization:
                anonymized_boxes = self._find_anonymized_boxes(image=image.copy())
                anonymized_boxes_with_text = [
                    self._get_text_from_anonymized_box(
                        image=image.copy(),
                        anonymized_box=box,
                        invert=self.config.invert_find_anonymized_boxes,
                    )
                    for box in anonymized_boxes
                ]
            else:
                anonymized_boxes_with_text = []

            if underlines_anonymization:
                (
                    anonymized_boxes_from_underlines,
                    underlines,
                ) = self._line_anonymization_to_boxes(
                    image=image.copy(),
                )
                anonymized_boxes_from_underlines_with_text = [
                    self._get_text_from_anonymized_box(
                        image,
                        box,
                        invert=self.config.invert_find_underline_anonymizations,
                    )
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
                underlines_anonymization = bool(
                    anonymized_boxes_from_underlines_with_text
                )
                if not box_anonymization and not underlines_anonymization:
                    # No indication of anonymization on first page.
                    # Then try use Tika
                    logger.info(self.config.message_try_use_tika)
                    pdf_text = self._read_text_with_tika(pdf_path=pdf_path)
                    if pdf_text:
                        return pdf_text
                    else:
                        logger.info(self.config.message_tika_failed)
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
            image_processed = self._process_image(
                image.copy(), all_anonymized_boxes_with_text, underlines
            )

            # Read core text of image
            result = self.reader.readtext(image_processed)

            # Make boxes on same format as anonymized boxes
            boxes = [self._change_box_format(box) for box in result]

            # Merge all boxes
            all_boxes = boxes + all_anonymized_boxes_with_text
            page_text = self._get_text_from_boxes(boxes=all_boxes)
            pdf_text += f"{page_text}\n\n"

        return pdf_text.strip()

    @staticmethod
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

    def _line_anonymization_to_boxes(self, image: np.ndarray) -> tuple:
        """Finds all underlines and makes anonymized boxes above them.

        Args:
            image (np.ndarray):
                Image to find anonymized boxes in.

        Returns:
            anonymized_boxes (List[dict]):
                List of anonymized boxes with coordinates
                (boxes above found underlines).
            underlines (List[tuple]):
                List of underlines with coordinates
                (will later be used to remove the underlines from the image).
        """
        # Bounds for height of underline.
        lb, ub = (
            self.config.underline_height_lower_bound,
            self.config.underline_height_upper_bound,
        )

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
        binary = self._binarize(image=dilated, threshold=200)
        blobs = self._get_blobs(binary)

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

                anonymized_box_refined = self._refine_anonymized_box(
                    anonymized_box, image_inverted
                )

                if anonymized_box_refined:
                    # Some anonymizations have two underlines, which
                    # results in two boxes overlapping.
                    box_is_duplicate = any(
                        self._too_much_overlap(box_1=anonymized_box_refined, box_2=box)
                        for box in anonymized_boxes
                    )
                    # Only store boxes that are not duplicates
                    if not box_is_duplicate:
                        anonymized_boxes.append(anonymized_box_refined)

                    # Store all underlines, so that they can be removed from the image later.
                    underlines.append(blob.bbox)

        return anonymized_boxes, underlines

    def _too_much_overlap(self, box_1: dict, box_2: dict) -> bool:
        """Used to determine if two boxes overlap too much.

        Case 1586 page 4 has an anonymization with two underlines,
        which results in two boxes overlapping. This function is used
        to determine if the boxes overlap too much.

        Args:
            box_1 (dict):
                Anonymized box with coordinates.
            box_2 (dict):
                Anonymized box with coordinates.

        Returns:
            bool:
                True if boxes overlap too much. False otherwise.
        """
        return (
            self._intersection_over_union(box_1=box_1, box_2=box_2)
            > self.config.iou_overlap_threshold
        )

    def _intersection_over_union(self, box_1: dict, box_2: dict) -> float:
        """Calculates intersection over union (IoU) between two boxes.

        Args:
            box_1 (dict):
                Anonymized box with coordinates.
            box_2 (dict):
                Anonymized box with coordinates.

        Returns:
            float:
                Intersection over union (IoU) between two boxes.
        """
        return self._intersection(box_1=box_1, box_2=box_2) / self._union(
            box_1=box_1, box_2=box_2
        )

    @staticmethod
    def _intersection(box_1: dict, box_2: dict) -> float:
        """Calculates intersection between two boxes.

        Args:
            box_1 (dict):
                Anonymized box with coordinates.
            box_2 (dict):
                Anonymized box with coordinates.

        Returns:
            float:
                Intersection between two boxes.
        """
        row_min1, col_min1, row_max1, col_max1 = box_1["coordinates"]
        row_min2, col_min2, row_max2, col_max2 = box_2["coordinates"]
        y_side_length = min(row_max1, row_max2) - max(row_min1, row_min2)
        x_side_length = min(col_max1, col_max2) - max(col_min1, col_min2)
        return (
            y_side_length * x_side_length
            if y_side_length > 0 and x_side_length > 0
            else 0
        )

    def _union(self, box_1: dict, box_2: dict) -> float:
        """Calculates the area of the union between two boxes.

        Args:
            box_1 (dict):
                Anonymized box with coordinates.
            box_2 (dict):
                Anonymized box with coordinates.

        Returns:
            float:
                area of the union between the two boxes.
        """
        area_1 = self._area(box=box_1)
        area_2 = self._area(box=box_2)
        return area_1 + area_2 - self._intersection(box_1=box_1, box_2=box_2)

    @staticmethod
    def _area(box: dict) -> float:
        """Calculates the area of a box.

        Args:
            box (dict):
                Anonymized box with coordinates.

        Returns:
            float:
                Area of the box.
        """
        row_min, col_min, row_max, col_max = box["coordinates"]
        return (row_max - row_min) * (col_max - col_min)

    def _remove_logo(self, image: np.ndarray) -> np.ndarray:
        """Removes logo from image.

        For many PDFs, there is a logo in the top right corner of the first page.

        Args:
            image (np.ndarray):
                Image to remove logo from.

        Returns:
            np.ndarray:
                Image with logo removed.
        """
        
        r, c = self.config.logo_row_idx, self.config.logo_col_idx
        logo = image[:r, c:, :]
        logo_binary = self._process_logo(logo=logo)

        blob_largest = self._get_blobs(binary=logo_binary)[0]
        # If largest blob is too large, then we are probably dealing with a logo.
        if blob_largest.area_bbox > self.config.logo_circumference_threshold:
            row_min, col_min, row_max, col_max = blob_largest.bbox
            logo[row_min: row_max, col_min: col_max, :] = 255

        image[:r, c:, :] = logo
        return image
    
    def _process_logo(self, logo: np.ndarray) -> np.ndarray:
        """Processes logo for blob detection.
        
        Args:
            logo (np.ndarray):
                Sub image which might contain a logo.

        Returns:
            np.ndarray:
                Processed logo.
        """
        logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
        logo_binary = self._binarize(
            image=logo_gray, threshold=230, val_min=0, val_max=255
        )
        inverted = cv2.bitwise_not(logo_binary)
        return inverted

    def _on_same_line(self, y: int, y_prev: int) -> bool:
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
        return abs(y - y_prev) < self.config.max_y_difference

    def _process_image(
        self, image: np.ndarray, anonymized_boxes: List[dict], underlines: List[tuple]
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

        # Make underlines black
        for underline in underlines:
            row_min, col_min, row_max, col_max = underline
            gray[row_min : row_max + 1, col_min : col_max + 1] = 0

        # Invert such that it is white text on black background.
        inverted = cv2.bitwise_not(gray)

        # Image has been inverted, such that it is white text on black background.
        # However this also means that the boxes currently are white.
        # We want to remove them entirely. We do this using flood fill.
        filled = inverted.copy()

        filled[filled < 5] = 0
        opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((30, 1)))
        save_cv2_image_tmp(opened)

        # filled[filled < 5] = 0
        # save_cv2_image_tmp
        # binary = self._binarize(image=inverted.copy(), threshold=5)
        # save_cv2_image_tmp(binary)

        # In flood fill a tolerance of 254 is used.
        # This means that when initiating a flood fill operation at a seed point
        # with a value of 255, all pixels greater than 0 within the object of the seed point
        # will be altered to 0.

        for anonymized_box in anonymized_boxes:
            row_min, col_min, row_max, col_max = anonymized_box["coordinates"]
            center = (row_min + row_max) // 2, (col_min + col_max) // 2
            seed_point = center

            if opened[seed_point] != 255:
                # Box is already removed, supposedly because
                # it overlaps with a previous box.
                continue

            mask = skimage.segmentation.flood(
                image=opened,
                seed_point=seed_point,
                tolerance=TOLERANCE_FLOOD_FILL,
            )
            filled[mask] = 0

            # filled = skimage.segmentation.flood_fill(
            #     image=filled,
            #     seed_point=seed_point,
            #     new_value=0,
            #     connectivity=1,
            #     tolerance=TOLERANCE_FLOOD_FILL,
            # )

        pad = self.config.underline_remove_pad
        for underline in underlines:
            row_min, col_min, row_max, col_max = underline
            seed_point = (row_min, col_min)
            # Remove underline
            filled[
                row_min - pad : row_max + 1 + pad, col_min - pad : col_max + 1 + pad
            ] = 0

        # Increase size of letters slightly
        dilated = cv2.dilate(filled, np.ones((2, 2)))
        return dilated

    def _get_text_from_boxes(self, boxes: List[dict]) -> str:
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
        boxes_y_sorted = sorted(boxes, key=lambda box: self._middle_y_cordinate(box))

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
            y = self._middle_y_cordinate(box)
            y_prev = self._middle_y_cordinate(box_prev)
            ys.append(y_prev)
            if self._on_same_line(y, y_prev):
                # Box is on current line.
                lines[-1].append(box)
            else:
                # Box is on a new line.
                new_line = [box]
                lines.append(new_line)

        # Now sort each line w.r.t x coordinate.
        # The lines should as a result be sorted w.r.t how a text is read.
        for i, line in enumerate(lines):
            lines[i] = sorted(line, key=lambda box: self._left_x_cordinate(box))

        # Ignore unwanted lines
        # Currently only footnotes are ignored.
        # Might want to add more conditions later.
        # For example, ignore page numbers.
        lines_ = [line for line in lines if not self._ignore_line(line)]

        # Each bounding box on a line is joined together with a space,
        # and the lines of text are joined together with \n.
        page_text = "\n".join(
            [" ".join([box["text"] for box in line]) for line in lines_]
        ).strip()
        return page_text

    def _ignore_line(self, line: List[dict]) -> bool:
        """Checks if line should be ignored.

        We want to ignore lines that are footnotes.
        Might want to add more conditions later.
        For example, ignore page numbers.

        Args:
            line (List[dict]):
                List of boxes on the line.

        Returns:
            bool:
                True if line should be ignored. False otherwise.
        """
        first_box = line[0]
        return self._is_footnote(first_box)

    def _is_footnote(self, first_box: dict):
        """Checks if line is a footnote.

        If the first box in the line is far to the right and far down,
        then it is probably a footnote.

        Args:
            first_box (dict):
                First box in line.

        Returns:
            bool:
                True if line is a footnote. False otherwise.
        """
        row_min, col_min, _, _ = first_box["coordinates"]
        return (
            col_min > self.config.line_start_ignore_col
            and row_min > self.config.line_start_ignore_row
        )

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
        tl, tr, _, bl = easyocr_box[0]
        row_min, col_min, row_max, col_max = tl[1], tl[0], bl[1], tr[0]
        text = easyocr_box[1]
        anonymized_box = {
            "coordinates": (row_min, col_min, row_max, col_max),
            "text": text,
        }
        return anonymized_box

    def _get_text_from_anonymized_box(
        self,
        image: np.ndarray,
        anonymized_box: dict,
        invert: bool = False,
    ) -> dict:
        """Read text from anonymized box.

        Args:
            image (np.ndarray):
                Image of the current page.
            anonymized_box (dict):
                Anonymized box with coordinates.
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
        anonymized_boxes = self._split_box(crop=crop, anonymized_box=anonymized_box)

        texts = []
        # `anonymized_boxes` are sorted left to right
        # E.g. the first box will contain the first word of `anonymized_box`.
        for anonymized_box_ in anonymized_boxes:
            row_min, col_min, row_max, col_max = anonymized_box_["coordinates"]

            crop = gray[row_min : row_max + 1, col_min : col_max + 1]
            crop_boundary = self._add_boundary(crop)

            # If length of box is short, then there are probably only a few letters in the box.
            # In this case, scale the image up.
            box_length = col_max - col_min
            scale = (
                1
                if box_length > BOX_LENGTH_SCALE_THRESHOLD
                else BOX_LENGTH_SCALE_THRESHOLD / box_length + 1
            )

            scaled = cv2.resize(crop_boundary, (0, 0), fx=scale, fy=scale)

            # Increase size of letters
            dilated = cv2.dilate(scaled, np.ones((2, 2)))

            dilated_boundary = self._add_boundary(dilated)

            sharpened = (
                np.array(
                    skimage.filters.unsharp_mask(
                        dilated_boundary, radius=20, amount=1.9
                    ),
                    dtype=np.uint8,
                )
                * 255  # output of unsharp_mask is in range [0, 1], but we want [0, 255]
            )

            # Read text from image with easyocr
            result = self.reader.readtext(sharpened)

            if len(result) == 0:
                text = ""
            else:
                text = " ".join(
                    [
                        box[1]
                        for box in result
                        if box[2] > self.config.threshold_box_confidence
                    ]
                )

            texts.append(text)

        text_all = " ".join(texts).strip()

        anonymized_box["text"] = f"<anonym>{text_all}</anonym>" if text_all else ""
        return anonymized_box

    def _find_anonymized_boxes(self, image: np.ndarray) -> List[dict]:
        """Finds anonymized boxes in image.

        Args:
            image (np.ndarray):
                Image to find anonymized boxes in.

        Returns:
            List[dict]:
                List of anonymized boxes.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Mean filter to make text outside boxes
        # brigther than color of boxes.
        footprint = np.ones((1, 15))
        averaged = rank.mean(gray, footprint=footprint)

        binary = self._binarize(
            image=averaged,
            threshold=self.config.threshold_binarize_anonymized_boxes,
            val_min=0,
            val_max=255,
        )
        inverted = cv2.bitwise_not(binary)

        # Some boxes are overlapping (horizontally).
        # Split them into separate boxes.
        inverted_boxes_split = self._split_boxes_in_image(inverted=inverted.copy())

        # (`inverted` is an inverted binary image).
        blobs = self._get_blobs(binary=inverted_boxes_split)

        anonymized_boxes = []
        heights = []
        for blob in blobs:
            if blob.area_bbox < self.config.box_area_min:
                # Blob is too small to be considered an anonymized box.
                break
            row_min, col_min, row_max, col_max = blob.bbox

            box_height = row_max - row_min
            heights.append(box_height)

            if (
                blob.area_filled / blob.area_bbox > self.config.box_accept_ratio
                and box_height > self.config.box_height_min
            ):
                assert 40 < box_height < 80, "Box height is not in expected range?"

                # `row_max - slight_shift_to_bottom` as text is usually in the top of the box.
                anonymized_box = {
                    "coordinates": (
                        row_min,
                        col_min,
                        row_max - self.config.slight_shift_to_bottom,
                        col_max,
                    )
                }
                anonymized_box_refined = self._refine_anonymized_box(
                    anonymized_box, image
                )
                anonymized_boxes.append(anonymized_box_refined)
            else:
                # Blob is not a bounding box.
                pass

        return anonymized_boxes

    def _split_boxes_in_image(self, inverted: np.ndarray) -> np.ndarray:
        """Splits overlapping boxes in image

        Some boxes are overlapping horizontally.
        This function splits them into separate boxes.

        Args:
            inverted (np.ndarray):
                Inverted binary image used to find the blobs/boxes.


        Returns:
            np.ndarray:
                Inverted binary image with overlapping boxes split into separate boxes.
        """
        blobs = self._get_blobs(inverted)

        # First split multiple boxes into separate boxes
        for blob in blobs:
            if blob.area_bbox < self.config.box_area_min:
                break
            row_min, col_min, row_max, col_max = blob.bbox

            box_height = row_max - row_min
            if box_height > 2 * BOX_HEIGHT_LOWER_BOUND:

                # Blob to uint8 image
                blob_image = np.array(blob.image * 255, dtype=np.uint8)

                # Get indices of rows to split
                row_indices_to_split = self._get_row_indices_to_split(
                    blob_image=blob_image
                )

                # Split
                for row_idx in row_indices_to_split:
                    blob_image[row_idx, :] = 0

                # Overwrite original sub image with modified sub image
                inverted[row_min:row_max, col_min:col_max] = blob_image

        return inverted

    def _get_row_indices_to_split(self, blob_image: np.ndarray) -> List[int]:
        """Split blob of overlapping boxes into separate boxes.

        Split blob where horizontal edges are found.

        Args:
            blob_image (np.ndarray):
                uint8 image of the found blobs.

        Returns:
            List[int]:
                List of row indices to split.
        """
        closed = cv2.morphologyEx(blob_image, cv2.MORPH_CLOSE, np.ones((40, 1)))

        edges_h = self._get_horizontal_edges(closed=closed)

        row_indices_to_split = [0]
        edge_row_indices = np.where(edges_h > 0)[0]
        unique, counts = np.unique(edge_row_indices, return_counts=True)
        for row_idx, count in zip(unique, counts):
            if (
                count >= self.config.indices_to_split_count_min
                and row_idx
                > row_indices_to_split[-1] + self.config.indices_to_split_row_diff
            ):
                row_indices_to_split.append(row_idx)
        return row_indices_to_split[1:]

    def _get_horizontal_edges(self, closed: np.ndarray) -> np.ndarray:
        """Get horizontal edges from image.

        Args:
            closed (np.ndarray):
                Image to get horizontal edges from.

        Returns:
            np.ndarray:
                All horizontal edges.
        """
        edges_h = skimage.filters.sobel_h(closed)
        edges_h = np.abs(edges_h)
        edges_h = np.array(edges_h * 255, dtype=np.uint8)
        return edges_h

    def _has_neighboring_white_pixels(self, a: np.ndarray, b: np.ndarray) -> bool:
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
            return distances.min() <= self.config.neighbor_distance_max

    @staticmethod
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

    @staticmethod
    def _binarize(
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

    def _remove_boundary_noise(self, binary_crop: np.ndarray) -> np.ndarray:
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

        blobs = self._get_blobs(binary_crop)
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
                and self._touches_boundary(binary_crop, blob)
                and not self._has_center_pixels(binary_crop, blob)
            ):
                # Remove blob
                coords = blob.coords
                binary_crop[coords[:, 0], coords[:, 1]] = 0
        return binary_crop

    @staticmethod
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

    @staticmethod
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

    def _refine_anonymized_box(self, anonymized_box: dict, image: np.ndarray) -> dict:
        """Refines bounding box.

        Two scenarios:
            1. The box is too big, i.e. there is too much black space around the text.
            2. The box is too small, i.e. some letters are not fully included in the box.

        Args:
            anonymized_box (dict):
                Anonymized box with coordinates.
            image (np.ndarray):
                Image of the current page.

        Returns:
            anonymized_box (dict):
                Anonymized box with refined coordinates.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        binary = self._binarize(
            image=gray, threshold=self.config.threshold_binarize_refine_anonymized_box
        )

        row_min, col_min, row_max, col_max = anonymized_box["coordinates"]

        # +1 as slice is exclusive and box coordinates are inclusive.
        crop = gray[row_min : row_max + 1, col_min : col_max + 1]

        # If empty/black box, box should be ignored
        if crop.sum() == 0:
            return {}

        crop_binary = self._binarize(
            image=crop, threshold=self.config.threshold_binarize_refine_anonymized_box
        )
        crop_binary_ = self._remove_boundary_noise(binary_crop=crop_binary.copy())
        binary[row_min : row_max + 1, col_min : col_max + 1] = crop_binary_

        # Refine box
        anonymized_box = self._refine(binary=binary, anonymized_box=anonymized_box)
        return anonymized_box

    def _refine(self, binary: np.ndarray, anonymized_box: List[int]) -> List[int]:
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
            anonymized_box = self._refine_(
                top_bottom_left_right="top",
                expanding=True,
                binary=binary,
                anonymized_box=anonymized_box,
            )
        else:
            anonymized_box = self._refine_(
                top_bottom_left_right="top",
                expanding=False,
                binary=binary,
                anonymized_box=anonymized_box,
            )

        row_min, col_min, row_max, col_max = anonymized_box["coordinates"]

        # Rows from bottom
        row = binary[row_max, col_min : col_max + 1]
        if not row.sum() == 0:
            anonymized_box = self._refine_(
                top_bottom_left_right="bottom",
                expanding=True,
                binary=binary,
                anonymized_box=anonymized_box,
            )
        else:
            anonymized_box = self._refine_(
                top_bottom_left_right="bottom",
                expanding=False,
                binary=binary,
                anonymized_box=anonymized_box,
            )

        row_min, col_min, row_max, col_max = anonymized_box["coordinates"]

        # Columns from left
        col = binary[row_min : row_max + 1, col_min]
        if not col.sum() == 0:
            anonymized_box = self._refine_(
                top_bottom_left_right="left",
                expanding=True,
                binary=binary,
                anonymized_box=anonymized_box,
            )
        else:
            anonymized_box = self._refine_(
                top_bottom_left_right="left",
                expanding=False,
                binary=binary,
                anonymized_box=anonymized_box,
            )

        row_min, col_min, row_max, col_max = anonymized_box["coordinates"]

        # Columns from right
        col = binary[row_min : row_max + 1, col_max]
        if not col.sum() == 0:
            anonymized_box = self._refine_(
                top_bottom_left_right="right",
                expanding=True,
                binary=binary,
                anonymized_box=anonymized_box,
            )
        else:
            anonymized_box = self._refine_(
                top_bottom_left_right="right",
                expanding=False,
                binary=binary,
                anonymized_box=anonymized_box,
            )

        return anonymized_box

    def _refine_(
        self,
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
            row_col_next, row_col, anonymized_box = self._next_row_col(
                top_bottom_left_right=top_bottom_left_right,
                expanding=expanding,
                binary=binary,
                anonymized_box=anonymized_box,
            )
            expand_steps_counter = 0
            while (
                self._not_only_white(row_col_next)
                and self._has_neighboring_white_pixels(row_col, row_col_next)
                and expand_steps_counter <= self.config.max_expand_steps
            ):
                row_col_next, row_col, anonymized_box = self._next_row_col(
                    top_bottom_left_right=top_bottom_left_right,
                    expanding=expanding,
                    binary=binary,
                    anonymized_box=anonymized_box,
                )
                expand_steps_counter += 1
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
            row_col_next, _, anonymized_box = self._next_row_col(
                top_bottom_left_right=top_bottom_left_right,
                expanding=False,
                binary=binary,
                anonymized_box=anonymized_box,
            )
            while row_col_next.sum() == 0:
                row_col_next, _, anonymized_box = self._next_row_col(
                    top_bottom_left_right=top_bottom_left_right,
                    expanding=False,
                    binary=binary,
                    anonymized_box=anonymized_box,
                )
        return anonymized_box

    @staticmethod
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

    def _split_box(self, crop: np.ndarray, anonymized_box: dict) -> List[dict]:
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
        split_indices = self._get_split_indices(crop=crop)
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
                "coordinates": (
                    row_min,
                    col_min + split_indices[-1] + 1,
                    row_max,
                    col_max,
                )
            }
            anonymized_boxes.append(last_box)
        return anonymized_boxes

    def _get_split_indices(self, crop: np.ndarray) -> List[int]:
        """Split box into multiple boxes - one for each word.

        Used in the function `_split_box`.

        Arg:
            crop (np.ndarray):
                Image of the box.

        Returns:
            List[int]:
                List of indices where the box should be split.
        """
        inverted = cv2.bitwise_not(crop)

        binary = self._binarize(
            inverted, threshold=self.config.threshold_binarize_split_indices
        )

        # One bool value for each column.
        # True if all pixels in column are white.
        booled = binary.all(axis=0)

        split_indices = []

        gap_length = 0
        for i, bool_value in enumerate(booled):
            if bool_value:
                gap_length += 1
            else:
                if gap_length > self.config.threshold_max_gap:
                    split_idx = i - 1 - gap_length // 2
                    if split_idx > 0:
                        split_indices.append(split_idx)
                gap_length = 0
        return split_indices

    @staticmethod
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

    @staticmethod
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


def save_cv2_image_tmp(image):
    """Saves image to tmp.png.

    Used for debugging.
    """
    image = image.copy()
    if image.max() < 2:
        image = image * 255
    cv2.imwrite("tmp.png", image)


def draw_box(image, box):
    """Draws box on image.

    Used for debugging.
    """
    image = image.copy()
    if isinstance(box, dict):
        row_min, col_min, row_max, col_max = box["coordinates"]
    else:
        # blob
        row_min, col_min, row_max, col_max = box.bbox
    if len(image.shape) == 2:
        image[row_min : row_max + 1, col_min : col_max + 1] = 0
    else:
        image[row_min : row_max + 1, col_min : col_max + 1, :] = 0
    save_cv2_image_tmp(image)
