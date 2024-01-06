"""Script for scraping the DomsDatabasen website.

Examples usages:
    Scrape single case:
    >>> python src/scripts/scrape.py 'scrape.case_id=12'

    Scrape single case and overwrite existing data:
    >>> python src/scripts/scrape.py 'scrape.case_id=123' 'scrape.force=True'

    Scrape all cases:
    >>> python src/scripts/scrape.py 'scrape.all=True' 'scrape.force=False'

    Scrape all cases and overwrite existing data:
    >>> python src/scripts/scrape.py 'scrape.force=True' 'scrape.all=True'
"""

import logging

import hydra
from omegaconf import DictConfig

from src.doms_databasen.scraper import DomsDatabasenScraper

# Importing as a module, doesn't work when running as a script?
# from doms_databasen.scraper import DomsDatabasenScraper

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    scraper = DomsDatabasenScraper(config=config)
    if config.scrape.all:
        scraper.scrape_all()
    elif config.scrape.case_id:
        scraper.scrape(config.scrape.case_id)
    else:
        logger.info(config.scrape.messages.give_correct_inputs)

    logger.info(config.scrape.messages.done)


if __name__ == "__main__":
    main()
