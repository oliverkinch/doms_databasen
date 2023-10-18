import logging
import os
import shutil
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager


from .constants import (
    DATA_RAW_DIR,
    DOWNLOAD_DIR,
    URL_MAIN,
    WAIT_TIME,
    XPATHS,
    XPATHS_TABULAR_DATA,
    TABULAR_DATA_FILE_NAME,
    PDF_DOCUMENT_FILE_NAME,
)
from .utils import save_dict_to_json

logger = logging.getLogger(__name__)


class DomsDatabasenScraper:
    def __init__(self):
        self.intialize_downloader_folder()
        self.driver = self.start_driver()

    def scrape_case(self, case_id: str, force=False) -> bool:
        """Scrapes a single case from domsdatabasen.dk

        Args:
            case_id (str):
                Case ID
            force (bool, optional):
                If True, overwrites existing data. Defaults to False.

        Returns:
            bool:
                True if case has successfully been scraped or already
                has been scraped. False if case does not exist.
        """
        case_url = f"{URL_MAIN}/{case_id}"
        self.driver.get(case_url)
        # Wait for page to load
        time.sleep(1)
        self._accept_cookies()
        time.sleep(1)
        if not self._case_id_exists():
            logger.info(f"Case {case_id} does not exist")
            return False

        case_dir = DATA_RAW_DIR / case_id
        if case_dir.exists():
            if force:
                shutil.rmtree(case_dir)
            else:
                logger.info(f"Case {case_id} already scraped. Use --force to overwrite")
                return True

        case_dir.mkdir()

        self._download_pdf(case_dir)
        tabular_data = self._get_tabular_data()
        save_dict_to_json(tabular_data, case_dir / TABULAR_DATA_FILE_NAME)

        return True

    def scrape_all(self, force=False) -> None:
        """Scrapes all cases from domsdatabasen.dk

        Args:
            force (bool, optional):
                If True, overwrites existing data. Defaults to False.
        """
        case_id = 1
        while True:
            success = self.scrape_case(str(case_id), force=force)
            if not success:
                break
            case_id += 1

    @staticmethod
    def start_driver() -> webdriver.Chrome:
        """Starts a Chrome webdriver with

        Returns:
            webdriver.Chrome:
                Chrome webdriver
        """
        options = Options()

        options.add_experimental_option(
            "prefs",
            {
                "download.default_directory": os.path.abspath(DOWNLOAD_DIR),
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "plugins.always_open_pdf_externally": True,
            },
        )

        driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()), options=options
        )
        return driver

    @staticmethod
    def intialize_downloader_folder():
        """Initializes the download folder.

        Deletes the download folder if it exists and creates a new one.
        """
        if DOWNLOAD_DIR.exists():
            shutil.rmtree(DOWNLOAD_DIR)
        DOWNLOAD_DIR.mkdir()

    @staticmethod
    def _wait_download(files_before: set, timeout: int = 10) -> str:
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
            files_now = set(os.listdir(DOWNLOAD_DIR))
            new_files = files_now - files_before
            if len(new_files) == 1:
                file_name = new_files.pop()
                return file_name
            if time.time() > endtime:
                file_name = ""
                return file_name

    def _download_pdf(self, case_dir: Path):
        """Downloads the PDF document of the case.

        Args:
            case_dir (Path):
                Path to case directory
        """
        files_before_download = set(os.listdir(DOWNLOAD_DIR))

        download_element = WebDriverWait(self.driver, WAIT_TIME).until(
            EC.presence_of_element_located((By.XPATH, XPATHS["download_pdf"]))
        )

        download_element.click()
        file_name = self._wait_download(files_before_download)
        if file_name:
            shutil.move(DOWNLOAD_DIR / file_name, case_dir / PDF_DOCUMENT_FILE_NAME)

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
            if "Dørlukning" in key:
                print("aloha")
            element = self.driver.find_element(By.XPATH, xpath)
            tabular_data[key] = element.text.strip()

        return tabular_data

    def _case_id_exists(self) -> bool:
        """Checks if the case exists.

        If a case does not exist, the page will contain the text "Fejlkode 404". This
        is used to check if the case exists.

        Returns:
            bool:
                True if case exists. False otherwise.
        """
        try:
            _ = self.driver.find_element(By.XPATH, XPATHS["Fejlkode 404"])
            return False
        except NoSuchElementException:
            return True
        except Exception as e:
            logger.error(e)
            raise e

    def _accept_cookies(self) -> None:
        """Accepts cookies on the page."""
        element = WebDriverWait(self.driver, WAIT_TIME).until(
            EC.presence_of_element_located((By.XPATH, XPATHS["Accept cookies"]))
        )
        element.click()
