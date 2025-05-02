"""
Abstract connector that defines the interface
every concrete connector must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import polars as pl


class BaseConnector(ABC):
    """
    Abstract connector that defines the interface every concrete connector must implement.

    Sub-classes must implement `get_data`, returning a pandas DataFrame.
    """

    @abstractmethod
    def get_data(self) -> pl.DataFrame:  # pragma: no cover
        """Fetch data and return it as a pandas DataFrame."""
        raise NotImplementedError

    # Optional common helpers
    def __call__(self) -> pl.DataFrame:
        return self.get_data()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
