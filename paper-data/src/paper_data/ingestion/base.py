"""Base interface for data-ingestion connectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
import polars as pl


class DataConnector(ABC):
    """Abstract object that returns a :class:`polars.DataFrame`."""

    @abstractmethod
    def get_data(self) -> pl.DataFrame:  # pragma: no cover
        """Return the ingested data."""
        raise NotImplementedError

    def __call__(self) -> pl.DataFrame:
        """Alias for :meth:`get_data`."""
        return self.get_data()

    def __repr__(self) -> str:  # pragma: no cover
        """Return debugging representation."""
        return f"<{self.__class__.__name__}>"

    def to_lazy(self) -> pl.LazyFrame:
        """Return a lazy version of the data."""
        return self.get_data().lazy()


# Backwards-compatibility alias. # TODO: remove when we update other connectors supporting this.
BaseConnector = DataConnector
