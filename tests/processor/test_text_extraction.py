"""Test code used for text extraction with easyocr."""

import numpy as np
import pytest
from easyocr import Reader
from PIL import Image

from src.doms_databasen.text_extraction import (
    _find_anonymized_boxes,
    _get_blobs,
    _get_split_indices,
    _get_text_from_anonymized_box,
    _get_text_from_boxes,
    _line_anonymization_to_boxes,
    _process_image,
    _refine_anonymized_box,
    _remove_boundary_noise,
    extract_text_easyocr,
)


@pytest.fixture(scope="module")
def reader(config):
    return Reader(["da"], gpu=config.gpu)


@pytest.mark.parametrize(
    "pdf_path, expected_text",
    [
        (
            "tests/data/processor/no_anonymization.pdf",
            "No anonymizations, so tika will read the text and \nwill do it correctly.;:..",
        ),
    ],
)
def test_extract_text_easyocr(pdf_path, expected_text, config, reader):
    text = extract_text_easyocr(pdf_path=pdf_path, config=config, reader=reader)
    assert text == expected_text


@pytest.mark.parametrize(
    "image_path, n_blobs",
    [("tests/data/processor/blobs.png", 4)],
)
def test_get_blobs(image_path, n_blobs):
    binary_image = np.array(Image.open(image_path))
    blobs = _get_blobs(binary_image)
    assert len(blobs) == n_blobs


@pytest.mark.parametrize(
    "image_path, n_matches_expected",
    [("tests/data/processor/underlines.png", 11)],
)
def test_line_anonymization_to_boxes(image_path, n_matches_expected):
    image = np.array(Image.open(image_path))
    anonymized_boxes, underlines = _line_anonymization_to_boxes(image)
    assert len(anonymized_boxes) == n_matches_expected
    assert len(underlines) == n_matches_expected


@pytest.mark.parametrize(
    "image_path, anonymized_boxes, underlines",
    [
        (
            "tests/data/processor/underlines.png",
            [{"coordinates": (0, 0, 15, 15)}],
            [(20, 20, 22, 30)],
        )
    ],
)
def test_process_image(image_path, anonymized_boxes, underlines):
    image = np.array(Image.open(image_path))
    processed_image = _process_image(image, anonymized_boxes, underlines)
    assert isinstance(processed_image, np.ndarray)


@pytest.mark.parametrize(
    "boxes, text_expected",
    [
        (
            [
                {"coordinates": (0, 0, 30, 75), "text": "Hej"},
                {"coordinates": (80, 0, 110, 100), "text": "smukke"},
            ],
            "Hej\nsmukke",
        )
    ],
)
def test_get_text_from_boxes(config, boxes, text_expected):
    text = _get_text_from_boxes(boxes, config.max_y_difference)
    assert text == text_expected


@pytest.mark.parametrize(
    "image_path, anonymized_box, invert, text_expected",
    [
        (
            "tests/data/processor/underlines.png",
            {"coordinates": (2863, 296, 2898, 490)},
            True,
            "<anonym>Tiltalte 2</anonym>",
        ),
        (
            "tests/data/processor/underlines.png",
            {"coordinates": (1186, 296, 1221, 490)},
            True,
            "<anonym>CPR nr. 1</anonym>",
        ),
    ],
)
def test_get_text_from_anonymized_box(
    reader, image_path, anonymized_box, invert, text_expected
):
    image = np.array(Image.open(image_path))
    anonymized_box = _get_text_from_anonymized_box(
        reader=reader, image=image, anonymized_box=anonymized_box, invert=invert
    )
    assert anonymized_box["text"] == text_expected


@pytest.mark.parametrize(
    "image_path, n_matches_expected",
    [
        ("tests/data/processor/page_with_boxes.png", 4),
        ("tests/data/processor/page_with_stacked_boxes.png", 9),
    ],
)
def test_find_anonymized_boxes(image_path, n_matches_expected):
    image = np.array(Image.open(image_path))
    anonymized_boxes = _find_anonymized_boxes(image=image)
    assert len(anonymized_boxes) == n_matches_expected


@pytest.mark.parametrize(
    "image_path",
    [
        ("tests/data/processor/boundary_noise.png"),
    ],
)
def test_remove_boundary_noise(image_path):
    binary_image = np.array(Image.open(image_path))
    N, M = binary_image.shape
    binary_image = _remove_boundary_noise(binary_image)
    assert binary_image[:, 0].sum() == 0
    assert binary_image[:, M - 1].sum() == 0
    assert binary_image[0, :].sum() == 0
    assert binary_image[N - 1, :].sum() == 0


@pytest.mark.parametrize(
    "image_path, anonymized_box, anonymized_box_expected",
    [
        (
            "tests/data/processor/page_with_boxes.png",
            {"coordinates": [2757, 572, 2818, 809]},
            {"coordinates": [2758, 623, 2789, 757]},
        ),
    ],
)
def test_refine_anonymized_box(anonymized_box, image_path, anonymized_box_expected):
    image = np.array(Image.open(image_path))
    anonymized_box = _refine_anonymized_box(anonymized_box=anonymized_box, image=image)
    assert anonymized_box["coordinates"] == anonymized_box_expected["coordinates"]


@pytest.mark.parametrize(
    "image_path, n_splits_expected",
    [
        (
            "tests/data/processor/box_with_multiple_words.png",
            5,
        ),
    ],
)
def test_get_split_indices(image_path, n_splits_expected):
    image = np.array(Image.open(image_path))
    split = _get_split_indices(crop=image)
    assert len(split) == n_splits_expected


if __name__ == "__main__":
    import pytest
    from hydra import compose, initialize

    # Initialise Hydra
    initialize(config_path="../../config", version_base=None)

    config = compose(
        config_name="config",
        overrides=["testing=True"],
    )

    r = Reader(["da"], gpu=config.gpu)
    pdf_path = "tests/data/processor/Sbizhub_C2219080509040.pdf"
    expected_text = (
        "No anonymizations, so tika will read the text and \nwill do it correctly.;:.."
    )
    test_extract_text_easyocr(
        pdf_path=pdf_path, config=config, reader=r, expected_text=expected_text
    )
