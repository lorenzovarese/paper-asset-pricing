"""
Connector for downloading files from arbitrary HTTP/HTTPS URLs.
"""

from __future__ import annotations
import tempfile
from pathlib import Path
import requests  # type: ignore[import-untyped]
import pandas as pd
from tqdm import tqdm  # type: ignore[import-untyped]
from urllib.parse import urlparse, parse_qs

from .base import BaseConnector


class HTTPConnector(BaseConnector):
    """
    Connector that downloads a file from any HTTP/HTTPS URL and loads it
    as a pandas DataFrame (CSV or Parquet) using LocalConnector.
    """

    def __init__(self, url: str, timeout: int = 30) -> None:
        self.url = url
        self.timeout = timeout

    def _download(self) -> Path:
        # Stream download to a temporary file with a progress bar
        print(f"Downloading {self.url}...")
        resp = requests.get(self.url, stream=True, timeout=self.timeout)
        resp.raise_for_status()

        # Try to read content-length, default to 0 if unavailable
        headers = getattr(resp, "headers", None)
        if headers is None:
            total = 0
        else:
            total = int(headers.get("content-length", 0) or 0)

        tmp = tempfile.NamedTemporaryFile(delete=False)

        # tqdm progress bar in bytes
        with tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {Path(self.url).name}",
            leave=True,
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                tmp.write(chunk)
                pbar.update(len(chunk))

        tmp.flush()
        return Path(tmp.name)

    def get_data(self) -> pd.DataFrame:
        """
        Download the remote file and return it as a DataFrame.
        Supports CSV and Parquet based on file extension.
        """
        print(f"Fetching data from {self.url}...")
        local_path = self._download()
        try:
            # Determine file format from URL
            parsed = urlparse(self.url)
            qs = parse_qs(parsed.query)
            suffix = Path(parsed.path).suffix.lower()

            # 1) If query says format=csv, or URL ends with .csv → CSV
            if qs.get("format", [""])[0] == "csv" or suffix == ".csv":
                return pd.read_csv(local_path)
            # 2) Parquet extensions
            if suffix in {".parquet", ".pq"}:
                return pd.read_parquet(local_path)
            # 3) Fallback → CSV
            return pd.read_csv(local_path)
        finally:
            # Clean up temporary file
            local_path.unlink(missing_ok=True)
