"""CSV loader connector."""

from __future__ import annotations

from pathlib import Path
import polars as pl

from .base import DataConnector


class CSVLoader(DataConnector):
    """Load a CSV file into a :class:`polars.DataFrame`."""

    def __init__(
        self,
        path: str | Path,
        date_col: str,
        id_col: str,
        **read_csv_kwargs,
    ) -> None:
        """
        Args:
            path: Path to the CSV file.
            date_col: Name of the date column (converted to ``pl.Date``).
            id_col: Name of the identifier column (e.g. ``permco``).
            **read_csv_kwargs: Extra keyword arguments for :func:`polars.read_csv`.
        """
        self._path = Path(path).expanduser()
        self._date_col = date_col
        self._id_col = id_col
        self._read_csv_kwargs = read_csv_kwargs

        if not self._path.is_file():
            raise FileNotFoundError(self._path)

    def get_data(self, *, date_format: str | None = "%Y-%m-%d") -> pl.DataFrame:
        """
        Parameters
        ----------
        date_format : str | None, default '%Y-%m-%d'
            If ``None`` the date column is left unchanged; otherwise it is
            parsed with :func:`polars.Expr.str.strptime`.
        """
        df = pl.read_csv(self._path, **self._read_csv_kwargs)

        missing = {self._date_col, self._id_col} - set(df.columns)
        if missing:
            raise ValueError(f"missing required column(s): {', '.join(missing)}")

        if date_format is not None:
            df = df.with_columns(
                pl.col(self._date_col).str.strptime(
                    pl.Date,
                    format=date_format,
                    strict=False,
                )
            )
        return df
