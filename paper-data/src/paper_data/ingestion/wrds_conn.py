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
import polars as pl
import wrds  # type: ignore[import-untyped]

from . import DataConnector


class WRDSConnector(DataConnector):
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

    def get_data(self) -> pl.DataFrame:
        with self._conn() as db:
            if self.max_rows:
                q = f"SELECT * FROM ({self.query}) LIMIT {self.max_rows}"
            else:
                q = self.query

            result = db.raw_sql(q)
            # If pandas.DataFrame, convert to Polars
            if isinstance(result, pd.DataFrame):
                return pl.from_pandas(result)
            # If it’s already a Polars DataFrame
            if isinstance(result, pl.DataFrame):
                return result
            # If it’s an iterator of DataFrames (chunks), collect and concat
            if hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
                dfs = []
                for chunk in result:  # type: ignore
                    if isinstance(chunk, pd.DataFrame):
                        dfs.append(pl.from_pandas(chunk))
                    elif isinstance(chunk, pl.DataFrame):
                        dfs.append(chunk)
                    else:
                        raise TypeError(f"Chunk of type {type(chunk)} not supported")
                return pl.concat(dfs)

            raise TypeError(
                f"Cannot convert WRDS raw_sql result {type(result)} to pl.DataFrame"
            )
