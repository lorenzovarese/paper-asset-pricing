"""
Connector that downloads a file shared via a Google Drive link.
After download, the file is parsed just like LocalConnector.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
import requests
import polars as pl


from .base import DataConnector


_DRIVE_FILE_ID_RE = re.compile(r"(?:file/d/|id=)([a-zA-Z0-9_-]{10,})")


def _extract_file_id(url: str) -> str:
    m = _DRIVE_FILE_ID_RE.search(url)
    if not m:
        raise ValueError("Could not parse Google Drive file id from URL.")
    return m.group(1)


class GoogleDriveConnector(DataConnector):
    """
    Connector that downloads a file shared via a Google Drive link.
    """

    def __init__(self, drive_url: str) -> None:
        self.file_id = _extract_file_id(drive_url)
        self.drive_url = (
            f"https://drive.google.com/uc?export=download&id={self.file_id}"
        )

    def _download(self) -> Path:
        with requests.Session() as s:
            resp = s.get(self.drive_url, stream=True, timeout=30)
            resp.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(delete=False)
            for chunk in resp.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp.flush()
            return Path(tmp.name)

    def get_data(self) -> pl.DataFrame:
        local_path = self._download()
        try:
            from .local import LocalConnector

            return LocalConnector(local_path).get_data()
        finally:
            local_path.unlink(missing_ok=True)
