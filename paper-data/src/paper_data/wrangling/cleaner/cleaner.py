# TODO: implementation need to be finished

import polars as pl
from typing import Sequence, Any, Literal
from dataclasses import dataclass


@dataclass
class RawDataset:
    """
    Wraps a raw DataFrame plus its intended objective ("firm" or "macro").
    """

    df: pl.DataFrame
    objective: Literal["firm", "macro"]


class BaseCleaner:
    """
    Generic cleaner for both firm and macro datasets.
    Branches on RawDataset.objective to prevent invalid operations.
    """

    def __init__(self, raw: RawDataset):
        self.raw = raw
        self.df = raw.df

    def _require(self, *allowed: Literal["firm", "macro"]):
        if self.raw.objective not in allowed:
            raise ValueError(
                f"Operation only valid for {allowed}, but dataset.objective={self.raw.objective}"
            )

    def normalize_columns(self) -> "BaseCleaner":
        """
        Lowercase and strip whitespace from column names.
        """
        self.df = self.df.rename({col: col.strip().lower() for col in self.df.columns})
        return self

    def rename_date_column(
        self,
        candidates: Sequence[str] = ("date", "yyyymm", "time"),
        target: str = "date",
    ) -> "BaseCleaner":
        """
        Rename the first matching date-like column to a standard target.
        Only valid for firm datasets.
        """
        self._require("firm", "macro")
        # Find the first candidate in current columns (case-insensitive)
        cols = self.df.columns
        for cand in candidates:
            for c in cols:
                if c.lower() == cand.lower():
                    self.df = self.df.rename({c: target})
                    return self
        return self

    def parse_date(
        self,
        date_col: str = "date",
        date_format: str | None = None,
        monthly_option: Literal["start", "end"] | None = None,
    ) -> "BaseCleaner":
        """
        Parse `date_col` to datetime, optionally truncating to month start/end.
        """
        self._require("firm", "macro")
        # 1) String→Datetime conversion :contentReference[oaicite:1]{index=1}
        if date_format:
            expr = pl.col(date_col).str.to_datetime(date_format)
        else:
            expr = pl.col(date_col).str.to_datetime()
        self.df = self.df.with_columns(expr.alias(date_col))

        # 2) Truncate to month start/end if requested :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}
        if monthly_option == "start":
            self.df = self.df.with_columns(
                pl.col(date_col).dt.month_start().alias(date_col)
            )
        elif monthly_option == "end":
            self.df = self.df.with_columns(
                # month_end() exists as Series method
                pl.col(date_col).dt.month_end().alias(date_col)
            )
        return self

    def clean_numeric_column(self, col: str) -> "BaseCleaner":
        """
        Coerce a column to numeric dtype.
        """
        self._require("firm", "macro")
        self.df[col] = pl.to_numeric(self.df[col], errors="coerce")
        return self

    def impute_constant(self, cols: Sequence[str], value: Any) -> "BaseCleaner":
        """
        Fill missing values in specified columns with a constant.
        """
        self._require("firm", "macro")
        for c in cols:
            self.df = self.df.with_columns(pl.col(c).fill_null(value).alias(c))
        return self


class FirmCleaner(BaseCleaner):
    """
    Adds firm‐specific, cross‐sectional imputations.
    """

    def __init__(
        self, raw: RawDataset, date_col: str = "date", id_col: str = "company_id"
    ):
        super().__init__(raw)
        if raw.objective != "firm":
            raise ValueError("FirmCleaner requires RawDataset.objective == 'firm'")
        self.date_col = date_col
        self.id_col = id_col

    def _ensure_datetime(self):
        if not pl.datatypes.is_datetime(self.df[self.date_col]):
            # Coerce to datetime if not already :contentReference[oaicite:6]{index=6}
            self.df = self.df.with_columns(
                pl.col(self.date_col).str.to_datetime().alias(self.date_col)
            )

    def impute_cross_section_median(self, cols: Sequence[str]) -> "FirmCleaner":
        """
        Fill missing by monthly cross-sectional median.
        """
        self._ensure_datetime()
        # 1) Add a month-period column via truncate to month start :contentReference[oaicite:7]{index=7}
        self.df = self.df.with_columns(
            pl.col(self.date_col).dt.truncate("1mo").alias("__month")
        )
        # 2) Compute per-month median with a window over `__month` :contentReference[oaicite:8]{index=8}
        for c in cols:
            self.df = self.df.with_columns(
                pl.col(c).fill_null(pl.col(c).median().over("__month")).alias(c)
            )
        # 3) Drop helper column
        self.df = self.df.drop("__month")
        return self

    def impute_cross_section_mean(self, cols: Sequence[str]) -> "FirmCleaner":
        """
        Fill missing by monthly cross-sectional mean.
        """
        self._ensure_datetime()
        self.df = self.df.with_columns(
            pl.col(self.date_col).dt.truncate("1mo").alias("__month")
        )
        for c in cols:
            self.df = self.df.with_columns(
                pl.col(c).fill_null(pl.col(c).mean().over("__month")).alias(c)
            )
        self.df = self.df.drop("__month")
        return self

    def impute_cross_section_mode(self, cols: Sequence[str]) -> "FirmCleaner":
        """
        Fill missing by monthly cross-sectional mode.
        """
        self._ensure_datetime()
        self.df = self.df.with_columns(
            pl.col(self.date_col).dt.truncate("1mo").alias("__month")
        )
        for c in cols:
            # Polars doesn’t have a built-in mode transform; use `value_counts` then join back
            mode_df = (
                self.df.group_by(["__month", c])
                .count()
                .sort("__month", descending=True)
                .unique(subset="__month")
                .select(["__month", c])
            )
            self.df = (
                self.df.join(mode_df, on="__month", how="left")
                .with_columns(pl.col(c).fill_null(pl.col(f"{c}_right")).alias(c))
                .drop([f"{c}_right", "__month"])
            )
        return self


class MacroCleaner(BaseCleaner):
    """
    Macro-specific cleaner. Currently no extra methods.
    """

    def __init__(self, raw: RawDataset, date_col: str = "date"):
        super().__init__(raw)
        if raw.objective != "macro":
            raise ValueError("MacroCleaner requires RawDataset.objective == 'macro'")
        self.date_col = date_col


class CleanerFactory:
    """
    Returns the correct cleaner for the given RawDataset.
    """

    @staticmethod
    def get_cleaner(raw: RawDataset) -> BaseCleaner:
        if raw.objective == "firm":
            return FirmCleaner(raw)
        elif raw.objective == "macro":
            return MacroCleaner(raw)
        else:
            raise ValueError(f"Unknown objective {raw.objective}")
