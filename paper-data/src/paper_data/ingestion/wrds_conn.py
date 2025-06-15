"""
Connector for the Wharton Research Data Services (WRDS) platform.
Requires `wrds` package and a valid account.
"""

from __future__ import annotations
from contextlib import contextmanager
import os
from pathlib import Path
import pandas as pd
import polars as pl
import wrds  # type: ignore[import-untyped]
import logging

from .base import DataConnector

logger = logging.getLogger(__name__)


class WRDSConnector(DataConnector):
    """
    Connector for the Wharton Research Data Services (WRDS) platform.
    Handles caching query results to a local CSV file.
    """

    def __init__(
        self,
        query: str,
        cache_path: Path,
        user: str | None = None,
        password: str | None = None,
        **read_csv_kwargs,
    ):
        """
        Args:
            query: The SQL query to execute on WRDS.
            cache_path: The local path to save/load the cached CSV file.
            user: WRDS username. Defaults to WRDS_USER env var.
            password: WRDS password. Defaults to WRDS_PASSWORD env var.
            **read_csv_kwargs: Extra keyword arguments for :func:`polars.read_csv`.
        """
        self.query = query
        self.cache_path = cache_path
        self.user = user or os.getenv("WRDS_USER")
        self.password = password or os.getenv("WRDS_PASSWORD")
        self._read_csv_kwargs = read_csv_kwargs

    @contextmanager
    def _conn(self):
        """Context manager for the WRDS connection."""
        db = wrds.Connection(wrds_username=self.user, wrds_password=self.password)
        try:
            yield db
        finally:
            db.close()

    def _query_and_cache(self):
        """Executes the WRDS query and saves the result to the cache path."""
        logger.info(
            "Cache miss for WRDS query. Connecting to WRDS. This may take a while."
        )
        with self._conn() as db:
            try:
                result_df = db.raw_sql(self.query)
                # The wrds library can return pandas DataFrames
                if isinstance(result_df, pd.DataFrame):
                    pl_df = pl.from_pandas(result_df)
                elif isinstance(result_df, pl.DataFrame):
                    pl_df = result_df
                else:
                    raise TypeError(
                        f"Unsupported data type from WRDS: {type(result_df)}"
                    )

                pl_df.write_csv(self.cache_path)
                logger.info(
                    f"Successfully queried WRDS and cached to '{self.cache_path}'"
                )
            except Exception as e:
                # Clean up failed download attempt
                if self.cache_path.exists():
                    self.cache_path.unlink(missing_ok=True)
                raise ConnectionError(f"Failed to execute WRDS query: {e}") from e

    def get_data(self) -> pl.DataFrame:
        """
        Retrieves data from WRDS, using a local cache.
        If the data is cached, it's loaded from the local CSV.
        Otherwise, it queries WRDS, caches the result, and then loads it.

        Returns:
            A Polars DataFrame of the WRDS query result.
        """
        if not self.cache_path.is_file():
            self._query_and_cache()
        else:
            logger.info(f"Cache hit for WRDS query. Loading from '{self.cache_path}'")

        return pl.read_csv(self.cache_path, **self._read_csv_kwargs)
