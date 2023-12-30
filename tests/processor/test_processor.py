"""Code for testing the Processor class."""


import pytest

from src.doms_databasen.processor import Processor


@pytest.fixture(scope="module")
def processor(config):
    return Processor(config=config)


@pytest.fixture(scope="module")
def processed_data(config, processor):
    return processor.process(case_id=config.process.test_case_id)


@pytest.mark.parametrize(
    "key",
    ["tabular_data", "case_id", "text", "text_anon_tagged", "pdf"],
)
def test_tabular_data(processed_data, key):
    assert processed_data[key]
