"""
Connector for the Wharton Research Data Services (WRDS) platform.

Requires `wrds` package and a valid account.
By default uses the WRDS CLI credentials in ~/.pgpass or ~/.wrds.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
import os
import pandas as pd
import wrds  # type: ignore[import-untyped]

from .base import BaseConnector


class WRDSConnector(BaseConnector):
    """
    Connector for the Wharton Research Data Services (WRDS) platform.
    Requires `wrds` package and valid credentials.
    """

    def __init__(
        self,
        query: str,
        user: str | None = None,
        password: str | None = None,
        max_rows: int | None = None,
    ) -> None:
        self.query = query
        self.user = user or os.getenv("WRDS_USER")
        self.password = password or os.getenv("WRDS_PASSWORD")
        self.max_rows = max_rows

    @contextmanager
    def _conn(self):
        db = wrds.Connection(wrds_username=self.user, wrds_password=self.password)
        try:
            yield db
        finally:
            db.close()

    def get_data(self) -> pd.DataFrame:
        with self._conn() as db:
            if self.max_rows:
                q = f"SELECT * FROM ({self.query}) LIMIT {self.max_rows}"
            else:
                q = self.query
            return db.raw_sql(q)
