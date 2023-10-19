"""Script for scraping the DomsDatabasen website.

Examples usages:
    Scrape single case:
    >>> python src/scripts/scrape.py --case_id=123

    Overwrite existing single case:
    >>> python src/scripts/scrape.py --case_id=123 --force

    Scrape all cases:
    >>> python src/scripts/scrape.py

    Scrape all cases and overwrite existing data:
    >>> python src/scripts/scrape.py --force
"""

import logging

import click
from hydra import compose, initialize
from omegaconf import DictConfig

from src.doms_databasen.scraper import DomsDatabasenScraper

logger = logging.getLogger(__name__)


def read_config() -> DictConfig:
    """Reads the config file.

    Returns:
        DictConfig:
            Config file
    """
    initialize(config_path="../../config")
    cfg = compose(config_name="config")
    return cfg


@click.command()
@click.option("--force", is_flag=True, default=False, help="Force scraping")
@click.option("--case_id", type=str, default="", help="Specify a case ID to scrape")
@click.option("--scrape_all", is_flag=True, default=False, help="Scrape all cases")
def main(force: bool, case_id: str, scrape_all: bool):
    cfg = read_config()
    if (not case_id and not scrape_all) or (case_id and scrape_all):
        logger.info(cfg.messages.give_correct_inputs)
        exit()

    scraper = DomsDatabasenScraper(cfg=cfg)
    if scrape_all:
        logger.info("Scraping all cases")
        scraper.scrape_all(force=force)
    else:
        logger.info(f"Scraping case {case_id}")
        scraper.scrape_case(case_id, force=force)

    logger.info("Done!")


if __name__ == "__main__":
    main()
