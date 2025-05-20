"""CSV loader connector."""

from __future__ import annotations

from pathlib import Path
import polars as pl
import logging


from .base import DataConnector

logger = logging.getLogger(__name__)


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
            error_msg = (
                f"Missing required columns: {', '.join(missing)} from file '{self._path.name}'.\n"
                f"  - Expected columns based on config: ['{self._date_col}', '{self._id_col}']\n"
                f"  - Columns found in file: {df.columns}\n"
                "  - Please check your 'data-config.yaml' for typos and ensure the column names (and their casing) match the CSV file."
            )
            raise ValueError(error_msg)

        if date_format is not None:
            # Check the current dtype of the date column
            current_dtype = df[self._date_col].dtype

            # If the date column is an integer type, cast it to Utf8 (string) first
            if current_dtype in [
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
            ]:
                df = df.with_columns(
                    pl.col(self._date_col).cast(pl.Utf8).alias(self._date_col)
                )
                logger.info(
                    f"Info: Date column '{self._date_col}' was numeric, cast to string for parsing."
                )

            # Now attempt to parse the date string
            df = df.with_columns(
                pl.col(self._date_col)
                .str.strptime(
                    pl.Date,
                    format=date_format,
                    strict=False,
                )
                .alias(self._date_col)
            )

            # Check for nulls after parsing and raise a more informative error
            if df[self._date_col].is_null().any():
                null_count = df[self._date_col].is_null().sum()
                total_rows = df.shape[0]
                raise ValueError(
                    f"Failed to parse {null_count}/{total_rows} date values in column '{self._date_col}' "
                    f"using format '{date_format}' from file '{self._path.name}'. "
                    "This often happens if the date format in the CSV does not match the 'date_column' format in your config. "
                    "Please check the date format in your CSV or configuration."
                )
        return df
