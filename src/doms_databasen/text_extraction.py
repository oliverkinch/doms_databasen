"""Code to read text from PDFs obtained from domsdatabasen.dk"""

import tempfile
from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import easyocr
import fitz
import numpy as np
import pypdfium2 as pdfium
import pytesseract
import skimage
from img2table.document import Image as TableImage
from img2table.tables.objects.extraction import ExtractedTable, TableCell
from omegaconf import DictConfig
from pdf2image import convert_from_path
from pypdf import PdfReader
from skimage import measure
from skimage.filters import rank
from skimage.measure._regionprops import RegionProperties
from tabulate import tabulate
from tika import parser

from src.doms_databasen.constants import (
    BOX_HEIGHT_LOWER_BOUND,
    DPI,
    LENGTH_TEN_LETTERS,
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

    def extract_text(self, pdf_path: Path) -> str:
        """Extracts text from a PDF using easyocr or tika.

        Tika is only used if there are no indications of anonymization.

        Some text is anonymized with boxes, and some text is anonymized with underlines.
        This function tries to find these anonymization, read the anonymized text,
        and then remove the anonymized text from the image before
        reading the rest of the text with easyocr.

        Args:
            pdf_path (Path):
                Path to PDF.

        Returns:
            pdf_text (str):
                Text from PDF.
        """
        images = self._get_images(pdf_path=pdf_path)
        pdf_text = ""

        # I have not seen a single PDF that uses both methods.
        # I have not either seen a PDF where there not
        # anonymization on the first page, if there are
        # any anonymization at all.
        # Therefore try both methods on the first page,
        # and then use the method that seems to be used, for the
        # rest of the pages.
        underline_anonymization = True
        box_anonymization = True
        text_tika = None

        for i, image in enumerate(images):
            if i == 0:
                image = self._remove_logo(image=image)

            # Log info about which anonymization methods are used in the PDF.
            if i == 1:
                self._log_anonymization_methods(
                    box_anonymization, underline_anonymization
                )

            # Anonymized boxes
            anonymized_boxes = []
            if box_anonymization:
                anonymized_boxes = self._extract_anonymized_boxes(image)
                box_anonymization = bool(anonymized_boxes)
                underline_anonymization = not bool(anonymized_boxes)

            anonymized_boxes_underlines = []
            underlines = []
            if underline_anonymization:
                (
                    anonymized_boxes_underlines,
                    underlines,
                ) = self._extract_underline_anonymization_boxes(image)
                text_tika = self._read_text_with_tika(pdf_path=str(pdf_path))

            # If no indication of anonymization on first page, then try use Tika.
            # If Tika doesn't work, then use easrocr.
            if i == 0:
                (
                    box_anonymization,
                    underline_anonymization,
                ) = self._anonymization_methods_used_in_pdf(
                    anonymized_boxes=anonymized_boxes,
                    anonymized_boxes_underlines=anonymized_boxes_underlines,
                )
                if not box_anonymization and not underline_anonymization:
                    logger.info(self.config.message_try_use_tika)
                    text_tika = self._read_text_with_tika(pdf_path=str(pdf_path))
                    if text_tika:
                        return None, text_tika
                    else:
                        logger.info(self.config.message_tika_failed)
                        # I have not seen a PDF where tika is not able
                        # to extract some text (even scanned PDFs)
                        # However, if it is the case that
                        # no anonymization is found on the first page,
                        # and that Tika fails, then use easyocr.

            all_anonymized_boxes = anonymized_boxes + anonymized_boxes_underlines

            table_boxes = self._find_tables(image=image.copy())

            # Remove anonymized boxes and tables from image
            image_processed = self._process_image(
                image=image.copy(),
                anonymized_boxes=all_anonymized_boxes,
                underlines=underlines,
                table_boxes=table_boxes,
            )

            main_text_boxes = self._get_main_text_boxes(image=image_processed)

            # Merge all boxes
            all_boxes = main_text_boxes + all_anonymized_boxes + table_boxes
            page_text = self._get_text_from_boxes(boxes=all_boxes)
            pdf_text += f"{page_text}\n\n"

        return pdf_text.strip(), text_tika

    def _get_main_text_boxes(self, image: np.ndarray) -> List[dict]:
        """Read main text of page.

        Args:
            image (np.ndarray):
                Image to read text from.

        Returns:
            main_text_boxes (List[dict]):
                List of boxes with coordinates and text.
        """
        result = self.reader.readtext(image)

        main_text_boxes = [self._change_box_format(box) for box in result]
        return main_text_boxes

    def _anonymization_methods_used_in_pdf(
        self, anonymized_boxes: List[dict], anonymized_boxes_underlines: List
    ) -> Tuple[bool, bool]:
        """Determine which anonymization methods are used in the PDF.

        Args:
            anonymized_boxes (List[dict]):
                List of anonymized boxes with coordinates.
            anonymized_boxes_underlines (List[dict]):
                List of anonymized boxes with coordinates.

        Returns:
            box_anonymization (bool):
                True if anonymized boxes are used in PDF. False otherwise.
            underline_anonymization (bool):
                True if underlines are used in PDF. False otherwise.
        """
        box_anonymization = bool(anonymized_boxes)
        underline_anonymization = bool(anonymized_boxes_underlines)
        return box_anonymization, underline_anonymization

    def _extract_anonymized_boxes(self, image: np.ndarray) -> List[dict]:
        """Extract anonymized boxes from image.

        Find and read text from anonymized boxes in image.

        Args:
            image (np.ndarray):
                Image to find anonymized boxes in.

        Returns:
            anonymized_boxes_with_text (List[dict]):
                List of anonymized boxes with coordinates and text.
        """
        anonymized_boxes = self._find_anonymized_boxes(image=image.copy())

        anonymized_boxes_with_text = [
            self._read_text_from_anonymized_box(
                image=image.copy(),
                anonymized_box=anonymized_box,
                invert=self.config.invert_find_anonymized_boxes,
            )
            for anonymized_box in anonymized_boxes
        ]

        return anonymized_boxes_with_text

    def _extract_underline_anonymization_boxes(self, image: np.ndarray) -> Tuple:
        """Extract boxes from underline anonymization.

        Find underlines, make boxes above them, and read text from the boxes.

        Args:
            image (np.ndarray):
                Image to find underline anonymization in.

        Returns:
            anonymized_boxes_underlines_ (List[dict]):
                List of boxes with coordinates and text.
            underlines (List[tuple]):
                List of underlines with coordinates.
        """
        anonymized_boxes_underlines, underlines = self._line_anonymization_to_boxes(
            image=image.copy(),
        )

        anonymized_boxes_underlines_ = [
            self._read_text_from_anonymized_box(
                image.copy(),
                box,
                invert=self.config.invert_find_underline_anonymizations,
            )
            for box in anonymized_boxes_underlines
        ]
        return anonymized_boxes_underlines_, underlines

    def _log_anonymization_methods(
        self, box_anonymization, underlines_anonymization
    ) -> None:
        """Log info about which anonymization methods are used in the PDF.

        Args:
            box_anonymization (bool):
                True if anonymized boxes are used in PDF. False otherwise.
            underlines_anonymization (bool):
                True if underlines are used in PDF. False otherwise.
        """
        if not box_anonymization:
            logger.info(self.config.message_pdf_has_no_anonymized_boxes)
        if not underlines_anonymization:
            logger.info(self.config.message_pdf_has_no_underline_anonymizations)

    def _get_images(self, pdf_path):
        """Get images from PDF.

        Args:
            pdf_path (Path):
                Path to PDF.

        Returns:
            images (List[np.ndarray]):
                List of images.
        """
        if self.config.image_idx:
            # Used for debugging a single page
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

        # Grayscale
        images = map(lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), images)
        return images

    def _find_tables(self, image: np.ndarray) -> List[dict]:
        """Extract tables from the image.

        Args:
            image (np.ndarray):
                Image to find tables in.

        Returns:
            table_boxes (List[dict]):
                List of tables with coordinates and text.
        """
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            inverted = cv2.bitwise_not(image)
            cv2.imwrite(tmp.name, inverted)
            table_image = TableImage(src=tmp.name, detect_rotation=False)
            tables = table_image.extract_tables()

        for table in tables:
            self._read_table(table=table, image=image)

        table_boxes = [self._table_to_box_format(table) for table in tables]

        return table_boxes

    def _get_coordinates(
        self, table_or_cell: List[Union[ExtractedTable, TableCell]]
    ) -> tuple:
        """Get coordinates of table or cell.

        Args:
            table_or_cell (Union[ExtractedTable, TableCell]):
                Table or cell to get coordinates from.

        Returns:
            (tuple):
                Coordinates of table or cell.
        """
        row_min, col_min, row_max, col_max = (
            table_or_cell.bbox.y1,
            table_or_cell.bbox.x1,
            table_or_cell.bbox.y2,
            table_or_cell.bbox.x2,
        )
        return row_min, col_min, row_max, col_max

    def _table_to_box_format(self, table: ExtractedTable) -> dict:
        """Convert table to box format.

        Args:
            table (ExtractedTable):
                Table to convert.

        Returns:
            table_box (dict):
                Table in box format.
        """
        row_min, col_min, row_max, col_max = self._get_coordinates(table_or_cell=table)

        table_string = tabulate(
            table.df, showindex="never", tablefmt=self.config.table_format
        )

        table_box = {
            "coordinates": (row_min, col_min, row_max, col_max),
            "text": table_string,
        }
        return table_box

    def _read_table(self, table: ExtractedTable, image: np.ndarray) -> None:
        """Read text in table.

        Args:
            table (ExtractedTable):
                Table to read text from.
            image (np.ndarray):
                Image that the table is extracted from.
        """
        for row in table.content.values():
            for cell in row:
                self._read_text_from_cell(cell, image)

    def _read_text_from_cell(self, cell: TableCell, image: np.ndarray) -> None:
        """Read text from cell with easyocr.

        Args:
            cell (TableCell):
                Cell to read text from.
            image (np.ndarray):
                Image that the cell is extracted from.
        """
        inverted = cv2.bitwise_not(image)
        cell_box = self._cell_to_box(cell)
        row_min, col_min, row_max, col_max = cell_box["coordinates"]

        crop = inverted[row_min:row_max, col_min:col_max]
        binary = self._binarize(
            image=crop, threshold=self.config.threshold_binarize_empty_box
        )
        if binary.sum() == 0:
            cell.value = ""
            return

        split_indices = self._multiple_lines(binary=binary)
        if not split_indices:
            cell_boxes = [cell_box]
        else:
            cell_boxes = self._split_cell_box(cell_box, split_indices)

        all_text = ""
        for cell_box_ in cell_boxes:
            row_min, col_min, row_max, col_max = cell_box_["coordinates"]
            crop = inverted[row_min:row_max, col_min:col_max]
            crop_refined, _ = self._refine_crop(
                crop=crop, padding=self.config.cell_box_crop_padding
            )

            text = self._read_text(crop_refined)
            all_text = self._add_text(text, all_text)

        # Remove last newline character
        all_text = all_text[:-1] if all_text[-1:] == "\n" else all_text
        cell.value = all_text

    def _read_text(self, crop_refined: np.ndarray) -> str:
        """Read text from subimage of cell.

        Args:
            crop_refined (np.ndarray):
                Subimage of cell.

        Returns:
            (str):
                Text from subimage of cell.
        """

        result = self.reader.readtext(crop_refined)
        if not result:
            text = ""
        else:
            text = result[0][1]
            for box in result[1:]:
                box_text = box[1]
                sep = "" if text[-1] == "-" else " "
                text += f"{sep}{box_text}"
        return f"{text}\n"

    def _add_text(self, text: str, all_text: str) -> str:
        """Add text from subimage of cell to all text from cell.

        Args:
            text (str):
                Text from subimage of cell.
            all_text (str):
                All text from cell.

        Returns:
            all_text (str):
                All text from cell.
        """
        if not all_text:
            all_text = text
        else:
            # text[-1] is `\n`
            sep = "" if all_text[-2] == "-" else " "
            all_text += f"{sep}{text}"

        return all_text

    def _split_cell_box(
        self, cell_box: TableCell, split_indices: List[int]
    ) -> List[dict]:
        """Split cell box into multiple cell boxes.

        Split each cell box into multiple cell boxes, one for each line.

        Args:
            cell_box (TableCell):
                Cell box to split.
            split_indices (List[int]):
                Indices to split cell box at.

        Returns:
            cell_boxes (List[dict]):
                List of cell boxes.
        """
        row_min, col_min, row_max, col_max = cell_box["coordinates"]

        cell_boxes = []

        # First box.
        first_box = {
            "coordinates": (row_min, col_min, row_min + split_indices[0], col_max)
        }
        cell_boxes.append(first_box)

        # Boxes in between first and last.
        if len(split_indices) > 1:
            for split_index_1, split_index_2 in zip(
                split_indices[:-1], split_indices[1:]
            ):
                cell_box_ = {
                    "coordinates": (
                        row_min + split_index_1 + 1,
                        col_min,
                        row_min + split_index_2,
                        col_max,
                    )
                }
                cell_boxes.append(cell_box_)

        # Last box.
        last_box = {
            "coordinates": (row_min + split_indices[-1] + 1, col_min, row_max, col_max)
        }
        cell_boxes.append(last_box)

        return cell_boxes

    def _cell_to_box(self, cell: TableCell) -> dict:
        """Convert cell to box format.

        Args:
            cell (TableCell):
                Cell to convert.

        Returns:
            cell_box (dict):
                Cell in box format.
        """
        p = self.config.remove_cell_border
        row_min, col_min, row_max, col_max = self._get_coordinates(table_or_cell=cell)
        cell_box = {"coordinates": (row_min + p, col_min + p, row_max - p, col_max - p)}
        return cell_box

    def _multiple_lines(self, binary: np.ndarray) -> List[int]:
        """Used to detect multiple lines in a cell.

        Args:
            binary (np.ndarray):
                Binary image of cell.

        Returns:
            split_indices (List[int]):
                Row indices to split cell at.
        """

        rows, _ = np.where(binary > 0)
        diffs = np.diff(rows)

        # Locate where the there are large gaps without text.
        jump_indices = np.where(diffs > self.config.cell_multiple_lines_gap_threshold)[
            0
        ]
        split_indices = []
        for jump_idx in jump_indices:
            top = rows[jump_idx]
            bottom = rows[jump_idx + 1]
            split_index = (top + bottom) // 2
            split_indices.append(split_index)
        return split_indices

    @staticmethod
    def _get_blobs(binary: np.ndarray, sort_function=None) -> list:
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
        if sort_function is None:
            sort_function = lambda blob: blob.area_bbox

        labels = measure.label(binary, connectivity=1)
        blobs = measure.regionprops(labels)
        blobs = sorted(blobs, key=sort_function, reverse=True)
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

        # Grayscale and invert, such that underlines are white.
        inverted = cv2.bitwise_not(image)

        # Morphological opening.
        # Remove everything that doesn't look like an underline.
        eroded = cv2.erode(inverted, np.ones((1, 50)), iterations=1)
        dilated = cv2.dilate(eroded, np.ones((1, 50)), iterations=1)

        # Binarize and locate blobs
        binary = self._binarize(image=dilated, threshold=200, val_min=0, val_max=255)

        blobs = self._get_blobs(binary)

        anonymized_boxes = []
        underlines = []
        for blob in blobs:
            row_min, col_min, row_max, col_max = blob.bbox

            # For a blob to be an underline, it should be
            # a "perfect" rectangle.

            height = row_max - row_min
            if blob.area == blob.area_bbox and lb < height < ub:
                box_row_min = row_min - self.config.underline_box_height
                box_row_max = row_min - 1  # Just above underline
                box_col_min = col_min
                box_col_max = col_max

                anonymized_box = {
                    "coordinates": [box_row_min, box_col_min, box_row_max, box_col_max],
                    "origin": "underline",
                }

                crop = inverted[box_row_min:box_row_max, box_col_min:box_col_max]
                if crop.sum() == 0:
                    # Box is empty
                    continue
                underlines.append(blob.bbox)

                box_is_duplicate = any(
                    self._too_much_overlap(box_1=anonymized_box, box_2=box)
                    for box in anonymized_boxes
                )
                if not box_is_duplicate:
                    anonymized_boxes.append(anonymized_box)

        return anonymized_boxes, underlines

    def _make_split_between_overlapping_box_and_line(
        self, binary: np.ndarray
    ) -> np.ndarray:
        edges = self._get_vertical_edges(binary=binary)
        sort_function = lambda blob: blob.bbox[2] - blob.bbox[0]
        edge_blobs = self._get_blobs(binary=edges, sort_function=sort_function)
        heights = []

        for blob in edge_blobs:
            row_min, col_min, row_max, col_max = blob.bbox
            height = row_max - row_min
            if height < 30:
                break
            heights.append(height)

            row_min, col_min, row_max, col_max = blob.bbox
            p = 10
            binary[row_min - p : row_max + p, col_min:col_max] = 0
        return binary

    def _too_much_overlap(self, box_1: dict, box_2: dict) -> bool:
        """Used to determine if two boxes overlap too much.

        For example case 1586 page 4 has an anonymization with two underlines,
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

        For many PDFs, there is a logo in the top of the first page.

        Args:
            image (np.ndarray):
                Image to remove logo from.

        Returns:
            np.ndarray:
                Image with logo removed.
        """
        r = self.config.page_from_top_to_this_row
        page_top = image[:r, :]
        page_top_binary = self._process_top_page(page_top=page_top)

        blobs = self._get_blobs(binary=page_top_binary)
        if blobs:
            blob_largest = blobs[0]
            # If largest blob is too large, then we are probably dealing with a logo.
            if blob_largest.area_bbox > self.config.logo_bbox_area_threshold:
                # Remove logo
                row_min, col_min, row_max, col_max = blob_largest.bbox
                page_top[row_min:row_max, col_min:col_max] = 255
                image[:r, :] = page_top

        return image

    def _process_top_page(self, page_top: np.ndarray) -> np.ndarray:
        """Processes logo for blob detection.

        Args:
            page_top (np.ndarray):
                Top part of page.

        Returns:
            np.ndarray:
                Processed top part.
        """
        logo_binary = self._binarize(
            image=page_top,
            threshold=self.config.threshold_binarize_top_page,
            val_min=0,
            val_max=255,
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
        self,
        image: np.ndarray,
        anonymized_boxes: List[dict],
        underlines: List[tuple],
        table_boxes,
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
        # For the anonymized boxes there already are black boxes,
        # but we will remove the text inside them, by making the text black.
        # The boxes made above underlines is included in the anonymized boxes.
        # For these there are no boxes above them, but only text,
        # but that text is simply removed by making a black box.
        image = self._remove_text_in_anonymized_boxes(
            image=image, anonymized_boxes=anonymized_boxes
        )

        image = self._draw_bbox_for_underlines(image=image, underlines=underlines)

        # Invert such that it is white text on black background.
        inverted = cv2.bitwise_not(image)

        # Image has been inverted, such that it is white text on black background.
        # However this also means that the boxes and underlines currently are white.
        # We want to remove them entirely. We do this using flood fill.
        filled = inverted.copy()

        filled[filled < 5] = 0
        opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, np.ones((30, 1)))

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

        pad = self.config.underline_remove_pad
        for underline in underlines:
            row_min, col_min, row_max, col_max = underline
            seed_point = (row_min, col_min)
            # Remove underline
            filled[row_min - pad : row_max + pad, col_min - pad : col_max + pad] = 0

        # Remove tables
        filled = self._remove_tables(image=filled, table_boxes=table_boxes)

        # Increase size of letters slightly
        dilated = cv2.dilate(filled, np.ones((2, 2)))
        image_processed = dilated

        return image_processed

    def _draw_bbox_for_underlines(
        self, image: np.ndarray, underlines: List[tuple]
    ) -> np.ndarray:
        """Draws bounding boxes for underlines.

        Args:
            image (np.ndarray):
                Image to draw bounding boxes on.
            underlines (List[tuple]):
                List of underlines with coordinates.

        Returns:
            np.ndarray:
                Image with bounding boxes drawn on the underlines.
        """
        for underline in underlines:
            row_min, col_min, row_max, col_max = underline
            image[row_min:row_max, col_min:col_max] = 0
        return image

    def _remove_text_in_anonymized_boxes(
        self, image: np.ndarray, anonymized_boxes: List[dict]
    ) -> np.ndarray:
        """Removes text in anonymized boxes.

        Args:
            image (np.ndarray):
                Image where boxes are found in.
            anonymized_boxes (List[dict]):
                List of anonymized boxes with coordinates.
        """
        for box in anonymized_boxes:
            row_min, col_min, row_max, col_max = box["coordinates"]
            image[row_min:row_max, col_min:col_max] = 0
        return image

    def _remove_tables(self, image: np.ndarray, table_boxes: List[dict]) -> np.ndarray:
        """Removes tables from image.

        Args:
            image (np.ndarray):
                Image to remove tables from.
            table_boxes (List[dict]):
                List of tables with coordinates.

        Returns:
            np.ndarray:
                Image with tables removed.
        """
        for table_box in table_boxes:
            row_min, col_min, row_max, col_max = table_box["coordinates"]

            p = self.config.remove_table_border
            image[row_min - p : row_max + p, col_min - p : col_max + p] = 0
        return image

    def _to_box_format(self, cell: TableCell):
        """Convert cell to box format.

        Args:
            cell (TableCell):
                Cell to convert.

        Returns:
            cell_box (dict):
                Cell in box format.
        """
        row_min, col_min, row_max, col_max = self._get_coordinates(table_or_cell=cell)
        s = self.config.cell_box_shrink
        # Better way to remove white border?
        # Flood fill if border is white?
        # Might be possible to use `_remove_boundary_noise`
        # Keep code as it is for now, as long as
        # no problems are encountered.
        cell_box = {"coordinates": (row_min + s, col_min + s, row_max - s, col_max - s)}
        return cell_box

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
        if not boxes:
            # Empty page supposedly
            return ""

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

    def _read_text_from_anonymized_box(
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
        # Easyocr seems to work best with white text on black background.
        if invert:
            image = cv2.bitwise_not(image)

        image_binary = self._binarize(
            image=image,
            threshold=self.config.threshold_binarize_empty_box,
            val_min=0,
            val_max=255,
        )

        row_min, col_min, row_max, col_max = anonymized_box["coordinates"]

        crop = image[row_min:row_max, col_min:col_max]

        # Make a box for each word in the box
        # I get better results with easyocr using this approach.
        anonymized_boxes = self._split_box(crop=crop, anonymized_box=anonymized_box)

        texts = []
        # `anonymized_boxes` are sorted left to right
        # E.g. the first box will contain the first word of `anonymized_box`.
        for anonymized_box_ in anonymized_boxes:
            row_min, col_min, row_max, col_max = anonymized_box_["coordinates"]
            crop_binary = image_binary[row_min:row_max, col_min:col_max]

            if crop_binary.sum() == 0:
                # `_split_box` might output boxes that are empty.
                # Could change `_split_box` such that it doesn't output empty boxes,
                # such that this if statement is not needed.
                texts.append("")
                continue

            crop = image[row_min:row_max, col_min:col_max]
            crop_processed = self._process_crop_before_read(crop)

            # Read text from image with easyocr
            result = self.reader.readtext(crop_processed)

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

    def _process_crop_before_read(self, crop: np.ndarray) -> np.ndarray:
        """Processes crop before reading text with easyocr.

        I get better results with easyocr using this approach.

        Args:
            crop (np.ndarray):
                Crop (representing the anonymized box) to be processed.

        Returns:
            crop_to_read (np.ndarray):
                Crop to read text from.
        """
        crop_refined, box_length = self._refine_crop(crop)

        scale = self._get_scale(box_length)
        crop_scaled = self._scale_image(image=crop_refined, scale=scale)
        crop_to_read = crop_scaled
        return crop_to_read

    def _get_scale(self, box_length: int) -> float:
        """Get scale to scale box/crop with.

        Args:
            box_length (int):
                Length of box.

        Returns:
            float:
                Scale to scale box/crop with.
        """
        scale = LENGTH_TEN_LETTERS / box_length
        scale = min(scale, self.config.max_scale)
        scale = max(scale, self.config.min_scale)
        return scale

    def _scale_image(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Scale image.

        Args:
            image (np.ndarray):
                Image to scale.
            scale (float):
                Scale to scale image with.

        Returns:
            np.ndarray:
                Scaled image.
        """
        scaled = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        return scaled

    def _refine_crop(
        self, crop: np.ndarray, padding: int = 3
    ) -> Tuple[np.ndarray, float]:
        """Refine crop.

        Args:
            crop (np.ndarray):
                Crop of image representing a box.

        Returns:
            np.ndarray:
                Refined crop.
            float:
                Length of box
        """
        binary = self._binarize(
            image=crop,
            threshold=100,
            val_min=0,
            val_max=255,
        )
        rows, cols = np.where(binary > 0)
        col_first, col_last = cols.min(), cols.max()
        row_first, row_last = rows.min(), rows.max()
        crop_fitted = crop[row_first : row_last + 1, col_first : col_last + 1]
        crop_boundary = self._add_boundary(crop_fitted, padding=padding)
        box_length = col_last - col_first
        return crop_boundary, box_length

    def _find_anonymized_boxes(self, image: np.ndarray) -> List[dict]:
        """Finds anonymized boxes in image.

        Args:
            image (np.ndarray):
                Image to find anonymized boxes in.

        Returns:
            List[dict]:
                List of anonymized boxes.
        """

        # Mean filter to make text outside boxes
        # brigther than color of boxes.
        footprint = np.ones((1, 15))
        averaged = rank.mean(image, footprint=footprint)

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

        binary_splitted = self._make_split_between_overlapping_box_and_line(
            binary=inverted_boxes_split
        )

        sort_function = lambda blob: blob.area
        blobs = self._get_blobs(binary=binary_splitted, sort_function=sort_function)

        anonymized_boxes = []
        for blob in blobs:
            if blob.area < self.config.box_area_min:
                # Blob is too small to be considered an anonymized box.
                break

            if not self._conditions_for_box(blob):
                continue

            assert (
                40 < blob.bbox[2] - blob.bbox[0] < 80
            ), "Box height is not in expected range?"
            anonymized_box = {
                "coordinates": [*blob.bbox],
                "origin": "box",
            }
            anonymized_boxes.append(anonymized_box)

        return anonymized_boxes

    def _conditions_for_box(self, blob: RegionProperties):
        box_height = blob.bbox[2] - blob.bbox[0]

        return (
            blob.area_filled / blob.area_bbox > self.config.box_accept_ratio
            and box_height > self.config.box_height_min
        )

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

        edge_lengths = self._get_edge_lengths(edges_h=edges_h)

        rows_to_split = self._rows_to_split(edge_lengths=edge_lengths)
        return rows_to_split

    def _rows_to_split(self, edge_lengths: dict) -> List[int]:
        """Get rows to split blob of overlapping boxes into separate boxes.

        Args:
            edge_lengths (dict):
                Dictionary with indices and lengths of horizontal edges.

        Returns:
            List[int]:
                List of row indices to split.
        """
        rows_to_split = [0]
        for idx, length in edge_lengths.items():
            if self._split_conditions(
                length=length, idx=idx, predesessor_idx=rows_to_split[-1]
            ):
                rows_to_split.append(idx)
        return rows_to_split[1:]

    def _split_conditions(self, length: int, idx: int, predesessor_idx: int) -> bool:
        """Checks if conditions for splitting are met.

        Args:
            length (int):
                Length of edge.
            idx (int):
                Index of edge.
            predesessor_idx (int):
                Index of previous edge.

        Returns:
            bool:
                True if conditions for splitting are met. False otherwise.
        """
        return (
            length > self.config.indices_to_split_edge_min_length
            and idx - predesessor_idx > self.config.indices_to_split_row_diff
        )

    def _get_edge_lengths(self, edges_h: np.ndarray):
        """Get lengths of horizontal edges.

        Args:
            edges_h (np.ndarray):
                Horizontal edges.

        Returns:
            dict:
                Dictionary with indices and lengths of horizontal edges.
        """
        edge_row_indices = np.where(edges_h > 0)[0]
        indices, lengths = np.unique(edge_row_indices, return_counts=True)
        edge_lengths = dict(zip(indices, lengths))

        edges_grouped = self._group_edges(indices)
        edge_lengths_merged = self._merge_adjacent_edges(
            edges_grouped=edges_grouped, edge_lengths=edge_lengths
        )

        return edge_lengths_merged

    @staticmethod
    def _group_edges(indices: np.ndarray):
        """Group indices of horizontal edges.

        Adjacent indices are grouped together.

        Args:
            indices (np.ndarray):
                Indices of horizontal edges.

        Returns:
            List[List[int]]:
                List of grouped indices.
        """
        edges_grouped = [[indices[0]]]

        adjacent_indices = np.diff(indices) == 1
        for i in range(1, len(indices)):
            if adjacent_indices[i - 1]:
                edges_grouped[-1].append(indices[i])
            else:
                edges_grouped.append([indices[i]])
        return edges_grouped

    def _merge_adjacent_edges(self, edges_grouped: List[List[int]], edge_lengths: dict):
        """Merge adjacent edges.

        Adjacent edges are merged together. The edge in each group with
        the largest length is used as the index for the merged edge.

        Args:
            edges_grouped (List[List[int]]):
                List of grouped indices.
            edge_lengths (dict):
                Dictionary with indices and lengths of horizontal edges.

        Returns:
            dict:
                Dictionary with indices and lengths of horizontal edges.
        """
        edge_lengths_merged = dict()
        for group in edges_grouped:
            idx = self._largest_edge_in_group(group=group, edge_lengths=edge_lengths)
            total_length = sum(edge_lengths[i] for i in group)
            edge_lengths_merged[idx] = total_length
        return edge_lengths_merged

    @staticmethod
    def _largest_edge_in_group(group: List[int], edge_lengths: dict):
        """Get index of largest edge in group.

        Args:
            group (List[int]):
                List of indices.
            edge_lengths (dict):
                Dictionary with indices and lengths of horizontal edges.

        Returns:
            int:
                Index of largest edge in group.
        """
        idx = group[0]
        for i in group[1:]:
            if edge_lengths[i] > edge_lengths[idx]:
                idx = i
        return idx

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

    def _get_vertical_edges(self, binary: np.ndarray) -> np.ndarray:
        """Get vertical edges from image.

        Args:
            binary (np.ndarray):
                Image to get vertical edges from.

        Returns:
            np.ndarray:
                All vertical edges.
        """
        edges_v = skimage.filters.sobel_v(binary)
        edges_v = np.abs(edges_v)
        edges_v = np.array(edges_v * 255, dtype=np.uint8)
        return edges_v

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
        """Removes noise on the boundary of an anonymized box.

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
                "coordinates": [row_min, col_min, row_max, col_min + split_indices[0]],
                "origin": "box",
            }

            anonymized_boxes.append(first_box)

            # Get box in between first and last box
            if len(split_indices) > 1:

                for split_index_1, split_index_2 in zip(
                    split_indices[:-1], split_indices[1:]
                ):
                    anonymized_box_ = {
                        "coordinates": [
                            row_min,
                            col_min + split_index_1 + 1,
                            row_max,
                            col_min + split_index_2,
                        ],
                        "origin": "box",
                    }
                    anonymized_boxes.append(anonymized_box_)

            # Get last box
            last_box = {
                "coordinates": [
                    row_min,
                    col_min + split_indices[-1] + 1,
                    row_max,
                    col_max,
                ],
                "origin": "box",
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
    def _add_boundary(image: np.ndarray, padding: int = 1) -> np.ndarray:
        """Add boundary to image.

        EasyOCR seems to give the best results when the text is surrounded by black pixels.

        Args:
            image (np.ndarray):
                Image to add boundary to.

        Returns:
            np.ndarray:
                Image with boundary.
        """
        p = padding
        padded = np.zeros(
            (image.shape[0] + p * 2, image.shape[1] + p * 2), dtype=np.uint8
        )
        padded[p:-p, p:-p] = image
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


def draw_box(image, box, pixel_value=0):
    """Draws box on image.

    Used for debugging.
    """
    image = image.copy()
    if isinstance(box, dict):
        row_min, col_min, row_max, col_max = box["coordinates"]
    else:
        # blob
        row_min, col_min, row_max, col_max = box.bbox

    image[row_min:row_max, col_min:col_max] = pixel_value

    save_cv2_image_tmp(image)
