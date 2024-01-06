"""Session-wide fixtures for tests."""

import random
import shutil

import pytest
from hydra import compose, initialize

from src.doms_databasen.scraper import DomsDatabasenScraper

# Initialise Hydra
initialize(config_path="../../config", version_base=None)


@pytest.fixture(scope="session")
def config():
    return compose(
        config_name="config",
        overrides=["testing=True"],
    )


@pytest.fixture(scope="session")
def scraper(config):
    return DomsDatabasenScraper(cfg=config)


def pytest_sessionstart(session):
    """Scrape a single random case before running tests."""
    cfg = compose(
        config_name="config",
        overrides=["testing=True"],
    )
    scraper = DomsDatabasenScraper(cfg=cfg)
    case_id = str(cfg.scrape.test_case_id)
    scraper.scrape(case_id)

    session.__CACHE = scraper.test_dir


def pytest_sessionfinish(session, exitstatus):
    """Delete scraped data after running tests."""
    shutil.rmtree(session.__CACHE)
