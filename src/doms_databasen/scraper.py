import logging
import os
import shutil
import time
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from .xpaths import XPATHS, XPATHS_TABULAR_DATA
from .exceptions import PDFDownloadException
from .utils import save_dict_to_json

logger = logging.getLogger(__name__)


class DomsDatabasenScraper:
    """Scraper for domsdatabasen.dk

    Args:
        cfg (DictConfig):
            Config file

    Attributes:
        cfg (DictConfig):
            Config file
        test_dir (Path):
            Path to test directory
        download_dir (Path):
            Path to download directory
        data_raw_dir (Path):
            Path to raw data directory
        force (bool):
            If True, existing data will be overwritten.
        cookies_clicked (bool):
            True if cookies have been clicked. False otherwise.
        driver (webdriver.Chrome):
            Chrome webdriver
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.test_dir = Path(self.cfg.paths.test_dir)
        self.download_dir = (
            Path(self.cfg.paths.download_dir) if not self.cfg.testing else self.test_dir
        )
        self.data_raw_dir = Path(self.cfg.paths.data_raw_dir)

        self.force = self.cfg.scrape.force
        self.cookies_clicked = False

        self.intialize_downloader_folder()
        self.driver = self.start_driver()

    def scrape(self, case_id: str) -> bool:
        """Scrapes a single case from domsdatabasen.dk

        Args:
            case_id (str):
                Case ID

        Returns:
            bool:
                The return value is used in `self.scrape_all`, to determine
                if the scraping should continue to the next case or stop.
        """
        case_id = str(case_id)
        case_dir = (
            self.data_raw_dir / case_id
            if not self.cfg.testing
            else self.test_dir / self.cfg.test_case_name
        )

        if self._already_scraped(case_dir) and not self.force:
            logger.info(
                f"Case {case_id} is already scraped. Use 'scrape.force' to overwrite"
            )
            return True

        logger.info(f"Scraping case {case_id}")

        case_url = f"{self.cfg.domsdatabasen.url}/{case_id}"
        self.driver.get(case_url)
        # Wait for page to load
        time.sleep(1)
        if not self.cookies_clicked:
            self._accept_cookies()
            self.cookies_clicked = True
            time.sleep(1)

        if not self._case_id_exists():
            # This will be triggered if no case has the given ID.
            # As cases are listed in ascending order starting from 1,
            # this means that no more cases exist.
            logger.info(f"Case {case_id} does not exist")
            return False

        elif not self._case_is_accessible():
            # Some cases might be unavailable for some reason.
            # A description is usually given on the page for case.
            # Thus if this is the case, just go to the next case.
            logger.info(f"Case {case_id} is not accessible")
            return True

        # Scrape data for the case.
        case_dir.mkdir(parents=True, exist_ok=True)

        self._download_pdf(case_dir)
        tabular_data = self._get_tabular_data()
        save_dict_to_json(tabular_data, case_dir / self.cfg.file_names.tabular_data)

        return True

    def scrape_all(self) -> None:
        """Scrapes all cases from domsdatabasen.dk

        Args:
            force (bool, optional):
                If True, overwrites existing data. Defaults to False.
        """
        logger.info("Scraping all cases")
        case_id = 1
        while True:
            continue_ = self.scrape(str(case_id))
            if not continue_:
                break
            case_id += 1

    def start_driver(self) -> webdriver.Chrome:
        """Starts a Chrome webdriver.

        Returns:
            webdriver.Chrome:
                Chrome webdriver
        """
        options = Options()

        options.add_experimental_option(
            "prefs",
            {
                "download.default_directory": os.path.abspath(self.download_dir),
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "plugins.always_open_pdf_externally": True,
            },
        )
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--headless")

        driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()), options=options
        )
        return driver

    def intialize_downloader_folder(self) -> None:
        """Initializes the download folder.

        Deletes the download folder if it exists and creates a new one.
        """
        if self.download_dir.exists():
            shutil.rmtree(self.download_dir)
        self.download_dir.mkdir()

    def _already_scraped(self, case_dir) -> bool:
        """Checks if a case has already been scraped.

        If a case has already been scraped, the case directory will contain
        two files: the PDF document and the tabular data.

        Args:
            case_dir (Path):
                Path to case directory

        Returns:
            bool:
                True if case has already been scraped. False otherwise.
        """
        return (
            case_dir.exists()
            and len(os.listdir(case_dir)) == self.cfg.n_files_raw_case_dir
        )

    def _wait_download(self, files_before: set, timeout: int = 10) -> str:
        """Waits for a file to be downloaded to the download directory.

        Args:
            files_before (set):
                Set of file names in download folder before download.
            timeout (int, optional):
                Number of seconds to wait before timing out. Defaults to 10.

        Returns:
            file_name (str):
                Name of downloaded file (empty string if timeout)
        """
        time.sleep(1)
        endtime = time.time() + timeout
        while True:
            files_now = set(os.listdir(self.download_dir))
            new_files = files_now - files_before
            if len(new_files) == 1:
                file_name = new_files.pop()
                return file_name
            if time.time() > endtime:
                file_name = ""
                return file_name

    def _download_pdf(self, case_dir: Path) -> None:
        """Downloads the PDF document of the case.

        Args:
            case_dir (Path):
                Path to case directory
        """
        files_before_download = set(os.listdir(self.download_dir))

        download_element = WebDriverWait(self.driver, self.cfg.sleep).until(
            EC.presence_of_element_located((By.XPATH, XPATHS["download_pdf"]))
        )

        download_element.click()
        file_name = self._wait_download(files_before_download)
        if file_name:
            from_ = (
                self.download_dir / file_name
                if not self.cfg.testing
                else self.test_dir / file_name
            )
            to_ = case_dir / self.cfg.file_names.pdf_document
            shutil.move(from_, to_)
        else:
            raise PDFDownloadException()

    def _get_tabular_data(self) -> dict:
        """Gets the tabular data from the case.

        Returns:
            tabular_data (dict):
                Tabular data
        """
        self.driver.find_element(By.XPATH, XPATHS["Øvrige sagsoplysninger"]).click()
        # Wait for section to expand
        time.sleep(1)
        tabular_data = {}
        for key, xpath in XPATHS_TABULAR_DATA.items():
            element = self.driver.find_element(By.XPATH, xpath)
            tabular_data[key] = element.text.strip()

        return tabular_data

    def _accept_cookies(self) -> None:
        """Accepts cookies on the page."""
        element = WebDriverWait(self.driver, self.cfg.sleep).until(
            EC.presence_of_element_located((By.XPATH, XPATHS["Accept cookies"]))
        )
        element.click()

    def _case_id_exists(self) -> bool:
        """Checks if the case exists.

        If a case does not exist, the page will contain the text "Fejlkode 404". This
        is used to check if the case exists.

        Returns:
            bool:
                True if case exists. False otherwise.
        """
        return not self.element_exists(XPATHS["Fejlkode 404"])

    def _case_is_accessible(self) -> bool:
        """Checks if the case is accessible.

        Some cases are not accessible for some reason. If this is the
        case, the page will contain the text "Sagen er ikke tilgængelig".

        Returns:
            bool:
                True if case is accessible. False otherwise.
        """
        return not self.element_exists(XPATHS["Sagen er ikke tilgængelig"])

    def element_exists(self, xpath) -> bool:
        """Checks if an element exists on the page.

        Args:
            xpath (str):
                Xpath to element

        Returns:
            bool:
                True if element exists. False otherwise.
        """
        try:
            _ = self.driver.find_element(By.XPATH, xpath)
            return True
        except NoSuchElementException:
            return False
        except Exception as e:
            logger.error(e)
            raise e
