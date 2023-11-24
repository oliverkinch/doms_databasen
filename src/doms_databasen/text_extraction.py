from logging import getLogger

import easyocr
import numpy as np
from pdf2image import convert_from_path
from tika import parser

from typing import List, Tuple

from pypdf import PdfReader
import pypdfium2 as pdfium
import fitz
import pytesseract

import cv2
import re
from skimage import measure
from skimage.filters import rank
import skimage


logger = getLogger(__name__)


def extract_text_easyocr(
    pdf_path: str,
    dpi: int = 300,
    gpu: bool = False,
    max_y_difference: int = 15,
    truncate_image: Tuple[int] = (0, 0),
    anon: bool = False,
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
        # image = [1]
        # image = np.array(image)
        # Remove top and bottom of image
        # truncate_image_top, truncate_image_bottom = truncate_image
        # if truncate_image_top or truncate_image_bottom:
        #     truncate_image_top = truncate_image_top if not i == 0 else truncate_image_top * 2
        #     truncate_image_bottom_ = truncate_image_bottom
        #     image = image[truncate_image_top:-truncate_image_bottom_, :]

        # Remove logo top right corner
        if i == 0:
            save_cv2_image_tmp(image)
            c = 1500
            r = 500
            image[:r, c:, :] = 255
            save_cv2_image_tmp(image)

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
        boxes = [change_box_format(box) for box in result]

        # Merge all boxes
        all_boxes = boxes + all_anonymized_boxes_with_text
        page_text = get_text_from_boxes(all_boxes, max_y_difference)
        if i == 0:
            text += f"{page_text}\n\n"
        else:
            text += f"\n\n{page_text}\n\n"

    return text.strip()


def get_blobs(binary):
    labels = measure.label(binary, connectivity=1)
    blobs = measure.regionprops(labels)
    blobs = sorted(blobs, key=lambda blob: blob.area_bbox, reverse=True)
    return blobs


def line_anonymization_to_boxes(image):
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
        # Draw blob

        min_row, min_col, max_row, max_col = blob.bbox
        # Remove blob from image

        # img_2 = cv2.rectangle(image, tl, br, (0, 0, 255), thickness=1)
        # save_cv2_image_tmp(img_2)
        # Blobs must be perfectly rectangular
        height = max_row - min_row
        print(height)
        if blob.area == blob.area_bbox and 1 < height < 7:
            white_pad = 1
            img[
                min_row - white_pad : max_row + white_pad,
                min_col - white_pad : max_col + white_pad,
                :,
            ] = 255

            box_row_min = min_row - 50
            box_row_max = min_row - 1  # Just above underline
            box_col_min = min_col + 5  # Avoid ,) etc. Box will be refined later.
            box_col_max = max_col - 5
            tl, tr, br, bl = (
                (box_col_min, box_row_min),
                (box_col_max, box_row_min),
                (box_col_max, box_row_max),
                (box_col_min, box_row_max),
            )

            img[box_row_min:box_row_max, box_col_min:box_col_max, :] = 0
            anonymized_box = {"coordinates": (tl, tr, br, bl)}

            anonymized_box_ = _refine_anonymized_box(anonymized_box, image_inverted)
            # for anonymized_box_ in anonymized_boxes:
            tl_, tr_, _, bl_ = anonymized_box_["coordinates"]
            box_row_min_, box_row_max_, box_col_min_, box_col_max_ = (
                tl_[1],
                bl_[1],
                tl_[0],
                tr_[0],
            )
            anonymized_boxes.append(anonymized_box_)

            if img[box_row_min_:box_row_max_, box_col_min_:box_col_max_, :].sum() > 0:
                img[box_row_min_:box_row_max_, box_col_min_:box_col_max_, :] = 0
                underlines.append(blob.bbox)

            save_cv2_image_tmp(img)

    return anonymized_boxes, underlines


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

    # file_data = []
    # _buffer = StringIO()
    # data = parser.from_file(str(pdf_path), xmlContent=True)
    # xhtml_data = BeautifulSoup(data['content'])
    # for page, content in enumerate(xhtml_data.find_all('div', attrs={'class': 'page'})):
    #     print('Parsing page {} of pdf file...'.format(page+1))
    #     _buffer.write(str(content))
    #     parsed_content = parser.from_buffer(_buffer.getvalue())
    #     _buffer.truncate()
    #     file_data.append({'id': 'page_'+str(page+1), 'content': parsed_content['content']})

    headers = {
        "X-Tika-OCRLanguage": "dan",
        "X-Tika-OCRTimeout": "300",
        "X-Tika-PDFocrStrategy": "auto",
    }
    request_options = {"timeout": 300}

    pdf_text_extractor = PDFTextExtractor()

    texts = {}
    # texts["tika"] = pdf_text_extractor.tika(str(pdf_path))
    # texts["pypdf"] = pdf_text_extractor.pypdf(str(pdf_path))
    # texts["pypdfium2"] = pdf_text_extractor.pypdfium2(str(pdf_path))
    # texts["pymupdf"] = pdf_text_extractor.pymupdf(str(pdf_path))
    texts["easyocr"] = pdf_text_extractor.easyocr(
        str(pdf_path), gpu, max_y_difference, first_page_only=False
    )
    # texts["tesseract"] = pdf_text_extractor.tesseract(str(pdf_path), first_page_only=True)
    # texts["tika_xml"] = pdf_text_extractor.tika_xml(str(pdf_path))

    # write texts to file
    for name, text in texts.items():
        with open(f"tmp_{name}.txt", "w") as f:
            f.write(text)

    text = parser.from_file(str(pdf_path), requestOptions=request_options)["content"]
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
            result = reader.readtext(
                np.array(image)
            )  # I have tried more arguments, but it doesn't seem to help.
            # result = reader.readtext(np.array(image), min_size=1, decoder="beamer", beamWidth=10, mag_ratio=5)

            # Sort w.r.t y coordinate.
            # bbox[0][0][1] is the y coordinate of the top left corner of the bounding box.
            result_y = sorted(result, key=lambda box: box[0][0][1])

            # Group bounding boxes that are on the same line.
            # E.g. the variable lines will be a list of lists, where each list contains
            # the bounding boxes for a given line of text in the pdf.
            # The variable `max_y_difference` is used to determine if two bounding boxes
            # are on the same line. E.g. if the difference between the y coordinate of
            # the top left corner of two bounding boxes is less than `max_y_difference`,
            # then the two bounding boxes are said to be on the same line.
            current_line = [result_y[0]]
            lines = [current_line]

            for i in range(1, len(result_y)):
                y = _get_top_left_y(result_y, i)
                y_prev = _get_top_left_y(result_y, i - 1)
                if _on_same_line(y, y_prev, max_y_difference):
                    # Append to current line
                    lines[-1].append(result_y[i])
                else:
                    # Append new line
                    new_line = [result_y[i]]
                    lines.append(new_line)

            # Now sort the lines w.r.t x coordinate.
            # The lines should then be sorted w.r.t how the text is read.
            for i, line in enumerate(lines):
                lines[i] = sorted(line, key=lambda x: x[0][0][0])

            # Each bounding box on a line is joined together with a space,
            # and the lines of text are joined together with \n.
            page_text = "\n".join(
                [" ".join([box[1] for box in line]) for line in lines]
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


def keep_only_letters_and_digits(s):
    """Helper function to keep only letters in a string.

    Args:
        s (str):
            String to keep only letters in.

    Returns:
        str:
            String with only letters.
    """
    return re.sub("[^a-zA-Z0-9]", "", s)


class PDFTextExtractor:
    @staticmethod
    def tika(pdf_path: str):
        # headers = {
        #         "X-Tika-OCRLanguage": "dan",
        #         # "X-Tika-OCRTimeout": "300",
        #         "X-Tika-PDFocrStrategy": "auto"
        #     }

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

    @staticmethod
    def tika_xml(pdf_path: str):
        headers = {
            "X-Tika-OCRLanguage": "dan",
            # "X-Tika-OCRTimeout": "300",
            "X-Tika-PDFocrStrategy": "auto",
        }
        request_options = {"timeout": 300}
        result = parser.from_file(
            pdf_path, xmlContent=True, requestOptions=request_options, headers=headers
        )
        return result["content"]


def process_image(image, anonymized_boxes, underlines):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_cv2_image_tmp(gray)
    # Make boxes black
    for box in anonymized_boxes:
        tl, tr, _, bl = box["coordinates"]
        row_min, row_max, col_min, col_max = tl[1], bl[1], tl[0], tr[0]
        gray[row_min:row_max, col_min:col_max] = 0
        save_cv2_image_tmp(gray)

    for underline in underlines:
        min_row, min_col, max_row, max_col = underline
        gray[min_row:max_row, min_col:max_col] = 0
    save_cv2_image_tmp(gray)

    inverted = cv2.bitwise_not(gray)
    save_cv2_image_tmp(inverted)

    dilated = cv2.dilate(inverted, np.ones((2, 2)))
    save_cv2_image_tmp(dilated)

    # Image has been inverted, such that it is white text on black background.
    # However this also means that the boxes currently are white.
    # Make them black again + p small padding because of dilation.
    drow, dcol = 1, 1
    filled = dilated.copy()
    loDiff = 5  # Lower difference threshold
    upDiff = 5  # Upper difference threshold

    # Perform the flood fill
    # cv2.floodFill(image=filled, mask=None, seedPoint=(col_min, row_min), newVal=0, loDiff=(loDiff, loDiff, loDiff), upDiff=(upDiff, upDiff, upDiff))
    tolerance = 5  # Remove everything that is not black
    # Square footprint
    # footprint = np.ones((3, 3))
    for box in anonymized_boxes:

        tl, tr, _, bl = box["coordinates"]
        row_min, row_max, col_min, col_max = tl[1], bl[1], tl[0], tr[0]
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
        min_row, min_col, max_row, max_col = underline
        # Draw seed point
        filled = skimage.segmentation.flood_fill(
            image=filled,
            seed_point=(min_row, min_col),
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
    boxes_y_sorted = sorted(boxes, key=lambda box: middle_y_cordinate(box))

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
        y = middle_y_cordinate(box)
        y_prev = middle_y_cordinate(box_prev)
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
        lines[i] = sorted(line, key=lambda box: left_x_cordinate(box))

        # Each bounding box on a line is joined together with a space,
        # and the lines of text are joined together with \n.
        page_text = "\n".join(
            [" ".join([box["text"] for box in line]) for line in lines]
        )
    return page_text.strip()


# def save_pil_image_tmp(image):
#     # image = Image.fromarray(image)
#     image.save("tmp.png")


def left_x_cordinate(box):
    tl, _, _, _ = box["coordinates"]
    return tl[0]


def middle_y_cordinate(box):
    tl, _, _, bl = box["coordinates"]
    return (tl[1] + bl[1]) / 2


def save_cv2_image_tmp(image):
    if image.max() < 2:
        image = image * 255
    cv2.imwrite("tmp.png", image)


def change_box_format(easyocr_box):
    tl, tr, br, bl = easyocr_box[0]
    text = easyocr_box[1]
    box = {"coordinates": (tl, tr, br, bl), "text": text}
    return box


def get_text_from_anonymized_box(
    image, anonymized_box, reader, theshold=0.3, invert=False, anonym=False
):

    tl, tr, br, bl = anonymized_box["coordinates"]

    row_min = tl[1]
    row_max = bl[1]
    col_max = tr[0]
    col_min = tl[0]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if invert:
        gray = cv2.bitwise_not(gray)
    save_cv2_image_tmp(gray)

    crop = gray[row_min:row_max, col_min:col_max]
    save_cv2_image_tmp(crop)

    split_idx = _split_into_two_boxes(crop)
    if split_idx:
        left_side, right_side = crop[:, :split_idx], crop[:, split_idx:]
        save_cv2_image_tmp(left_side)
        save_cv2_image_tmp(right_side)
        left_max_col = np.where(left_side != 0)[1].max() + 1
        right_min_col = np.where(right_side != 0)[1].min() - 1

        col_min_1 = col_min
        col_max_1 = col_min_1 + left_max_col + 1
        col_min_2 = col_min + split_idx + right_min_col
        col_max_2 = col_max + 1
        row_max += 1

        left_side_ = gray[row_min:row_max, col_min_1:col_max_1]
        save_cv2_image_tmp(left_side_)
        right_side_ = gray[row_min:row_max, col_min_2:col_max_2]
        save_cv2_image_tmp(right_side_)

        tl_1, tr_1, br_1, bl_1 = (
            (col_min_1, row_min),
            (col_max_1, row_min),
            (col_max_1, row_max),
            (col_min_1, row_max),
        )

        tl_2, tr_2, br_2, bl_2 = (
            (col_min_2, row_min),
            (col_max_2, row_min),
            (col_max_2, row_max),
            (col_min_2, row_max),
        )

        anonymized_boxes_ = [
            {"coordinates": (tl_1, tr_1, br_1, bl_1)},
            {"coordinates": (tl_2, tr_2, br_2, bl_2)},
        ]
    else:
        anonymized_boxes_ = [anonymized_box]

    texts = []
    # anonymized_boxes_ are sorted left to right
    for anonymized_box_ in anonymized_boxes_:
        tl, tr, _, bl = anonymized_box_["coordinates"]

        row_min = tl[1]
        row_max = bl[1]
        col_max = tr[0]
        col_min = tl[0]

        crop = gray[row_min:row_max, col_min:col_max]

        crop_padded = crop.copy()
        crop_padded[0, :] = 0
        crop_padded[-1, :] = 0
        crop_padded[:, 0] = 0
        crop_padded[:, -1] = 0

        # Care to dilate horizontally, such that letters are not split.
        # eroded = cv2.erode(dilated, np.ones((3, 1)), iterations=1)
        # dilated = cv2.dilate(eroded, np.ones((3, 1)), iterations=1)
        # save_cv2_image_tmp(dilated)

        # Make crop_refined with 0-padding. Just one col/row 0-padding.
        # n_cols = col_max - col_min
        # n_rows = row_max - row_min

        # crop_padded = np.zeros((n_rows + 2, n_cols + 2))
        # crop_padded[1:-1, 1:-1] = crop
        save_cv2_image_tmp(crop_padded)

        # eroded = cv2.erode(crop_padded, np.ones((1, 3)), iterations=1)
        # dilated = cv2.dilate(eroded, np.ones((1, 3)), iterations=1)
        # save_cv2_image_tmp(dilated)

        # eroded = cv2.erode(dilated, np.ones((3, 1)), iterations=1)
        # dilated = cv2.dilate(eroded, np.ones((3, 1)), iterations=1)
        # save_cv2_image_tmp(dilated)

        # Opening

        # If length of box is short, then there are probably < 2 letters in the box.
        # In this case, scale the image up.
        box_length = col_max - col_min
        scale = 1 if box_length > 50 else 2

        scaled = cv2.resize(crop_padded, (0, 0), fx=scale, fy=scale)
        save_cv2_image_tmp(scaled)

        dilated1 = cv2.dilate(scaled, np.ones((2, 2)))
        save_cv2_image_tmp(dilated1)

        sharpened = (
            np.array(
                skimage.filters.unsharp_mask(dilated1, radius=20, amount=1.9),
                dtype=np.uint8,
            )
            * 255
        )
        save_cv2_image_tmp(sharpened)

        inverted = cv2.bitwise_not(sharpened)
        save_cv2_image_tmp(inverted)
        save_cv2_image_tmp(sharpened)

        # zero of shape sharpened + 2
        # pad = 2
        # padded = np.zeros((sharpened.shape[0] + pad, sharpened.shape[1] + pad), dtype=np.uint8)
        # padded[pad // 2:-pad//2, pad//2:-pad//2] = sharpened
        # save_cv2_image_tmp(padded)

        # inverted = cv2.bitwise_not(padded)
        # save_cv2_image_tmp(inverted)
        # save_cv2_image_tmp(padded)

        # Descale image
        # crop_small = cv2.resize(sharp, (0, 0), fx=1/scale, fy=1/scale)
        # save_cv2_image_tmp(crop_small)

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
        min_row, min_col, max_row, max_col = blob.bbox
        box_height = max_row - min_row
        if box_height > box_height_split:
            # Split into multiple boxes (horizontal split)
            n_boxes = box_height // (box_height_split // 2)
            box_height_ = box_height // n_boxes
            for j in range(n_boxes):
                min_row_ = min_row + box_height_ * j
                max_row_ = min_row + box_height_ * (j + 1)
                sub_image = inverted[min_row_:max_row_, min_col:max_col]
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

                inverted[min_row_:max_row_, min_col:max_col] = dilated
                save_cv2_image_tmp(inverted)

                # inverted[row, :] = 0

                # sub_image = inverted[min_row:max_row_, min_col:max_col]

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
        min_row, min_col, max_row, max_col = blob.bbox
        box_height = max_row - min_row
        heights.append(box_height)
        pad_above = 0
        pad_left = 0
        pad_right = 0
        tl, tr, br, bl = (
            (min_col - pad_left, min_row - pad_above),
            (max_col + pad_right, min_row - pad_above),
            (max_col + pad_right, max_row),
            (min_col - pad_left, max_row),
        )

        # if box height is above 120, split box in two

        if blob.area_filled / blob.area_bbox > 0.4 and box_height > 10:
            if box_height > box_height_split:
                # # Split into multiple boxes (horizontal split)
                # n_boxes = box_height // (box_height_split // 2)
                # box_height_ = box_height // n_boxes
                # for j in range(1, n_boxes):
                #     row = min_row + box_height_ * j - slight_shift
                #     inverted[row, :] = 0
                #     save_cv2_image_tmp(inverted)

                # labels_ = measure.label(inverted, connectivity=1)
                # blobs_ = measure.regionprops(labels)
                # blobs_ = sorted(blobs, key=lambda blob: blob.area_bbox, reverse=True)

                for j in range(n_boxes):
                    # Text is in general in the top of the box.
                    slight_shift_ = slight_shift if not j == 0 else 0

                    tl_ = (min_col, min_row + box_height_ * j - slight_shift_)
                    tr_ = (max_col, min_row + box_height_ * j - slight_shift_)

                    slight_shift_ = slight_shift if not j == n_boxes - 1 else 0
                    br_ = (max_col, min_row + box_height_ * (j + 1) - slight_shift_)
                    bl_ = (min_col, min_row + box_height_ * (j + 1) - slight_shift_)

                    anonymized_box = {"coordinates": (tl_, tr_, br_, bl_)}
                    anonymized_box_ = _refine_anonymized_box(anonymized_box, image)
                    # for anonymized_box_refined in anonymized_boxes_refined:
                    anonymized_boxes.append(anonymized_box_)
                    tl_, tr_, br_, bl_ = anonymized_box_["coordinates"]
                    img_2 = cv2.rectangle(img, tl_, br_, (0, 255, 255), thickness=2)
                    save_cv2_image_tmp(img_2)
            else:
                img_2 = cv2.rectangle(img, tl, br, (0, 255, 255), thickness=2)
                save_cv2_image_tmp(img_2)
                anonymized_box = {"coordinates": (tl, tr, br, bl)}
                anonymized_box_ = _refine_anonymized_box(anonymized_box, image)
                # for anonymized_box_refined in anonymized_boxes_refined:
                anonymized_boxes.append(anonymized_box_)
                tl_, tr_, br_, bl_ = anonymized_box_["coordinates"]
                img_2 = cv2.rectangle(img, tl_, br_, (0, 0, 255), thickness=2)
                save_cv2_image_tmp(img_2)
        else:
            img_2 = cv2.rectangle(img, tl, br, (0, 255, 255), thickness=2)
            save_cv2_image_tmp(img_2)
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


def _unpack_coordinates(box):
    """Unpack coordinates of box.

    Args:
        box (dict):
            Anonymized box with coordinates given as corners of box
            , e.g. tl, tr, br, bl.

    Returns:
        tuple:
            Tuple with coordinates given by min/max row/col of box.
    """
    tl, tr, _, bl = box["coordinates"]
    row_min, row_max, col_min, col_max = tl[1], bl[1], tl[0], tr[0]
    return row_min, row_max, col_min, col_max


def _refine_anonymized_box(anonymized_box, image, threshold: int = 30):
    tl, tr, _, bl = anonymized_box["coordinates"]

    row_min = tl[1]
    row_max = bl[1]
    col_max = tr[0]
    col_min = tl[0]
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

    tl_, tr_, br_, bl_ = (
        (col_min, row_min),
        (col_max, row_min),
        (col_max, row_max),
        (col_min, row_max),
    )

    anonymized_box["coordinates"] = (tl_, tr_, br_, bl_)
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


def _split_into_two_boxes(crop_refined, threshold=100):
    inverted = cv2.bitwise_not(crop_refined)
    save_cv2_image_tmp(inverted)

    # Opening
    N, M = crop_refined.shape
    # eroded = cv2.erode(inverted, np.ones((N // 2, 10)), iterations=1)
    # save_cv2_image_tmp(eroded)
    binary = binarize(inverted, threshold=threshold)
    save_cv2_image_tmp(binary)
    booled = binary.all(axis=0)

    largest_consecutive_sum = 0
    idx = None
    current_sum = 0
    for i, b in enumerate(booled):
        if b:
            current_sum += 1
            if current_sum > largest_consecutive_sum:
                largest_consecutive_sum = current_sum
                idx = i
        else:
            current_sum = 0
    if largest_consecutive_sum > 15:
        split_idx = idx - largest_consecutive_sum // 2
        return split_idx
    else:
        return False
