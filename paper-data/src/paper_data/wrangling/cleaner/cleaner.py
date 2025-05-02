# TODO: implementation need to be finished

import pandas as pd
from typing import Sequence, Any, Literal
from dataclasses import dataclass


@dataclass
class RawDataset:
    """
    Wraps a raw DataFrame plus its intended objective ("firm" or "macro").
    """

    df: pd.DataFrame
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
        self.df.columns = self.df.columns.astype(str).str.lower().str.strip()
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
        # Lookup lowercase names to original
        lcols = {c.lower(): c for c in self.df.columns}
        for cand in candidates:
            if cand in lcols:
                self.df.rename(columns={lcols[cand]: target}, inplace=True)
                break
        return self

    def parse_date(
        self,
        date_col: str = "date",
        date_format: str | None = None,
        monthly_option: Literal["start", "end"] | None = None,
    ) -> "BaseCleaner":
        """
        Parse a date column, optionally adjusting to start or end of month.
        """
        self._require("firm", "macro")
        if date_format:
            self.df[date_col] = pd.to_datetime(
                self.df[date_col], format=date_format, errors="coerce"
            )
        else:
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors="coerce")
        if monthly_option:
            period = self.df[date_col].dt.to_period("M")
            self.df[date_col] = period.dt.to_timestamp(how=monthly_option)
        return self

    def clean_numeric_column(self, col: str) -> "BaseCleaner":
        """
        Coerce a column to numeric dtype.
        """
        self._require("firm", "macro")
        self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        return self

    def impute_constant(self, cols: Sequence[str], value: Any) -> "BaseCleaner":
        """
        Fill missing values in specified columns with a constant.
        """
        self._require("firm", "macro")
        for c in cols:
            # avoid chained assignment warnings
            self.df[c] = self.df[c].fillna(value)
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
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(
                self.df[self.date_col], errors="coerce"
            )

    def impute_cross_section_median(self, cols: Sequence[str]) -> "FirmCleaner":
        """
        Fill missing by monthly cross-sectional median.
        """
        self._ensure_datetime()
        month = self.df[self.date_col].dt.to_period("M")
        med = self.df.groupby(month)[cols].transform("median")
        for c in cols:
            self.df[c] = self.df[c].fillna(med[c])
        return self

    def impute_cross_section_mean(self, cols: Sequence[str]) -> "FirmCleaner":
        """
        Fill missing by monthly cross-sectional mean.
        """
        self._ensure_datetime()
        month = self.df[self.date_col].dt.to_period("M")
        mn = self.df.groupby(month)[cols].transform("mean")
        for c in cols:
            self.df[c] = self.df[c].fillna(mn[c])
        return self

    def impute_cross_section_mode(self, cols: Sequence[str]) -> "FirmCleaner":
        """
        Fill missing by monthly cross-sectional mode.
        """
        self._ensure_datetime()
        month = self.df[self.date_col].dt.to_period("M")
        modes = self.df.groupby(month)[cols].transform(
            lambda s: s.mode(dropna=True).iloc[0]
            if not s.mode(dropna=True).empty
            else pd.NA
        )
        for c in cols:
            self.df[c] = self.df[c].fillna(modes[c])
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
