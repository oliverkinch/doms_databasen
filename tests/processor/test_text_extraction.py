"""Test code used for text extraction with easyocr."""

import pytest
from PIL import Image
import numpy as np
from easyocr import Reader

from src.doms_databasen.text_extraction import (
    get_blobs,
    line_anonymization_to_boxes,
    save_cv2_image_tmp,
    process_image,
    get_text_from_boxes,
    get_text_from_anonymized_box,
    find_anonymized_boxes,
    _remove_boundary_noise,
    _refine_anonymized_box,
)


@pytest.fixture(scope="module")
def reader(config):
    return Reader(["da"], gpu=config.gpu)


@pytest.mark.parametrize(
    "image_path, n_blobs",
    [("tests/data/processor/blobs.png", 4)],
)
def test_get_blobs(image_path, n_blobs):
    binary_image = np.array(Image.open(image_path))
    blobs = get_blobs(binary_image)
    assert len(blobs) == n_blobs


@pytest.mark.parametrize(
    "image_path, n_matches_expected",
    [("tests/data/processor/underlines.png", 11)],
)
def test_line_anonymization_to_boxes(image_path, n_matches_expected):
    image = np.array(Image.open(image_path))
    anonymized_boxes, underlines = line_anonymization_to_boxes(image)
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
    processed_image = process_image(image, anonymized_boxes, underlines)
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
    text = get_text_from_boxes(boxes, config.max_y_difference)
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
    anonymized_box = get_text_from_anonymized_box(
        reader=reader, image=image, anonymized_box=anonymized_box, invert=invert
    )
    assert anonymized_box["text"] == text_expected


@pytest.mark.parametrize(
    "image_path, n_matches_expected",
    [
        ("tests/data/processor/page_with_boxes.png", 4),
    ],
)
def test_find_anonymized_boxes(image_path, n_matches_expected):
    image = np.array(Image.open(image_path))
    anonymized_boxes = find_anonymized_boxes(image=image)
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
            {"coordinates": (2757, 572, 2818, 809)},
            {"coordinates": (2758, 623, 2789, 757)},
        ),
    ],
)
def test_refine_anonymized_box(anonymized_box, image_path, anonymized_box_expected):
    image = np.array(Image.open(image_path))
    anonymized_box = _refine_anonymized_box(anonymized_box=anonymized_box, image=image)
    assert anonymized_box["coordinates"] == anonymized_box_expected["coordinates"]


if __name__ == "__main__":
    from hydra import compose, initialize
    import pytest

    # Initialise Hydra
    initialize(config_path="../../config", version_base=None)

    config = compose(
        config_name="config",
        overrides=["testing=True"],
    )

    image_path = "tests/data/processor/underlines.png"
    box = {"coordinates": (2863, 296, 2898, 490)}
    expected_text = "<anonym>Tiltalte 2</anonym>"

    r = Reader(["da"], gpu=config.gpu)
    test_get_text_from_anonymized_box(r, image_path, box, expected_text)
