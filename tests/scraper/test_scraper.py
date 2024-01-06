from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def test_case_path(config):
    return Path(config.paths.test_dir) / config.scrape.test_case_name


def test_case_contains_pdf(config, test_case_path):
    assert (test_case_path / config.file_names.pdf_document).exists()


def test_case_contains_tabular_data(config, test_case_path):
    assert (test_case_path / config.file_names.tabular_data).exists()
