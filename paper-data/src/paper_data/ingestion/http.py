"""
Connector for downloading files from Google Sheets.
"""

from __future__ import annotations
import re
import requests
from pathlib import Path
import logging
import polars as pl

from .base import DataConnector

logger = logging.getLogger(__name__)


class GoogleSheetConnector(DataConnector):
    """
    Connector that downloads a specific sheet from a Google Sheet URL.
    It handles caching the file as a CSV and then loads it into a Polars DataFrame.
    """

    def __init__(
        self,
        url: str,
        cache_path: Path,
        ignore_thousands_separator: bool = False,
        **read_csv_kwargs,
    ):
        """
        Args:
            url: The full URL to the Google Sheet.
            cache_path: The local path to save/load the cached CSV file.
            ignore_thousands_separator: If True, will attempt to remove commas from numeric-like columns.
            **read_csv_kwargs: Extra keyword arguments for :func:`polars.read_csv`.
        """
        self.url = url
        self.cache_path = cache_path
        self.ignore_thousands_separator = ignore_thousands_separator
        self._read_csv_kwargs = read_csv_kwargs
        self._download_url = self._prepare_download_url()

    def _prepare_download_url(self) -> str:
        """Parses the Google Sheet URL to create a direct CSV download link."""
        file_id_match = re.search(r"/d/([a-zA-Z0-9_-]+)", self.url)
        if not file_id_match:
            raise ValueError("Could not parse Google Sheet file ID from URL.")
        file_id = file_id_match.group(1)

        gid_match = re.search(r"gid=(\d+)", self.url)
        gid = gid_match.group(1) if gid_match else "0"

        return f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=csv&gid={gid}"

    def _download_if_needed(self):
        """Downloads the file from Google Sheets if it's not already cached."""
        if self.cache_path.is_file():
            logger.info(
                f"Cache hit for Google Sheet. Using existing file: '{self.cache_path}'"
            )
            return

        logger.info(
            f"Cache miss. Downloading from Google Sheets to '{self.cache_path}'."
        )
        try:
            resp = requests.get(self._download_url, stream=True, timeout=120)
            resp.raise_for_status()
            with open(self.cache_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Successfully downloaded and cached Google Sheet.")
        except requests.RequestException as e:
            if self.cache_path.exists():
                self.cache_path.unlink(missing_ok=True)
            raise OSError(
                f"Failed to download from Google Sheets URL '{self.url}': {e}"
            ) from e

    def get_data(self) -> pl.DataFrame:
        """
        Ensures the Google Sheet is downloaded and returns it as a Polars DataFrame.

        Returns:
            A Polars DataFrame containing the data from the sheet.
        """
        self._download_if_needed()

        # Make a copy of the kwargs to avoid modifying the original dict.
        # Pop our custom 'date_column' argument so it's not passed to Polars.
        kwargs_for_polars = self._read_csv_kwargs.copy()
        date_column_info = kwargs_for_polars.pop("date_column", {})

        if not self.ignore_thousands_separator:
            # Pass the cleaned kwargs to Polars
            return pl.read_csv(self.cache_path, **kwargs_for_polars)

        logger.info(
            "`ignore_thousands_separator` is True. Reading all columns as strings for cleaning."
        )

        # By setting infer_schema_length=0, Polars reads all columns as Utf8 (string)
        # by default, which prevents parsing errors on numbers with commas.
        df = pl.read_csv(
            self.cache_path,
            infer_schema_length=0,
            **kwargs_for_polars,  # Pass the cleaned kwargs here too
        )

        # Identify columns that are not the date column and attempt to clean them.
        date_col_name = list(date_column_info.keys())
        cols_to_clean = [col for col in df.schema.names() if col not in date_col_name]

        # Create expressions to remove commas and then cast to a numeric type.
        cleaning_expressions = [
            pl.col(c).str.replace_all(",", "").cast(pl.Float64, strict=False).alias(c)
            for c in cols_to_clean
        ]

        df = df.with_columns(cleaning_expressions)
        logger.info(f"Cleaned thousands separators from columns: {cols_to_clean}")

        return df
