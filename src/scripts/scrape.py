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

from src.doms_databasen.scraper import DomsDatabasenScraper
from src.doms_databasen.constants import GIVE_CORRECT_INPUTS

logger = logging.getLogger(__name__)


@click.command()
@click.option("--force", is_flag=True, default=False, help="Force scraping")
@click.option("--case_id", type=str, default="", help="Specify a case ID to scrape")
@click.option("--scrape_all", is_flag=True, default=False, help="Scrape all cases")
def main(force, case_id, scrape_all):
    if not case_id and not scrape_all:
        logger.info(GIVE_CORRECT_INPUTS)
        exit()
    if case_id and scrape_all:
        logger.info(GIVE_CORRECT_INPUTS)
        exit()

    scraper = DomsDatabasenScraper()
    if scrape_all:
        logger.info("Scraping all cases")
        scraper.scrape_all(force=force)
    elif case_id:
        logger.info(f"Scraping case {case_id}")
        scraper.scrape_case(case_id, force=force)
    else:
        logger.info("Hvad har jeg glemt at tage højde for?")

    logger.info("Done!")


if __name__ == "__main__":
    main()
