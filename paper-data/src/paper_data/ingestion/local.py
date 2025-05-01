"""
Connector for local CSV or Parquet files.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from .base import BaseConnector


class LocalConnector(BaseConnector):
    """
    Connector for local CSV or Parquet files.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def get_data(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(self.path)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(self.path)
        raise ValueError(f"Unsupported file type: {suffix}")
