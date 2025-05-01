# TODO: implementation need to be finished

import pandas as pd
from typing import Sequence, Any, Literal
from dataclasses import dataclass, field


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
        self.df = raw.df  # alias for convenience

    def _require(self, *allowed: Literal["firm", "macro"]):
        if self.raw.objective not in allowed:
            raise ValueError(
                f"Operation only valid for {allowed}, "
                f"but dataset.objective={self.raw.objective}"
            )

    def normalize_columns(self) -> "BaseCleaner":
        self.df.columns = self.df.columns.astype(str).str.lower().str.strip()
        return self

    def rename_date_column(
        self,
        candidates: Sequence[str] = ("date", "yyyymm", "time"),
        target: str = "date",
    ) -> "BaseCleaner":
        self._require("firm", "macro")
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
        self._require("firm", "macro")
        if date_format:
            self.df[date_col] = pd.to_datetime(
                self.df[date_col], format=date_format, errors="coerce"
            )
        else:
            self.df[date_col] = pd.to_datetime(self.df[date_col], errors="coerce")
        if monthly_option:
            period = self.df[date_col].dt.to_period("M")
            self.df[date_col] = (
                period.dt.to_timestamp("start")
                if monthly_option == "start"
                else period.dt.to_timestamp("end")
            )
        return self

    def clean_numeric_column(self, col: str) -> "BaseCleaner":
        self._require("firm", "macro")
        self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        return self

    def impute_constant(self, cols: Sequence[str], value: Any) -> "BaseCleaner":
        self._require("firm", "macro")
        for c in cols:
            self.df[c].fillna(value, inplace=True)
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
        self._ensure_datetime()
        month = self.df[self.date_col].dt.to_period("M")
        med = self.df.groupby(month)[cols].transform("median")
        for c in cols:
            self.df[c].fillna(med[c], inplace=True)
        return self

    def impute_cross_section_mean(self, cols: Sequence[str]) -> "FirmCleaner":
        self._ensure_datetime()
        month = self.df[self.date_col].dt.to_period("M")
        mn = self.df.groupby(month)[cols].transform("mean")
        for c in cols:
            self.df[c].fillna(mn[c], inplace=True)
        return self

    def impute_cross_section_mode(self, cols: Sequence[str]) -> "FirmCleaner":
        self._ensure_datetime()
        month = self.df[self.date_col].dt.to_period("M")
        modes = self.df.groupby(month)[cols].transform(
            lambda s: s.mode(dropna=True).iloc[0]
            if not s.mode(dropna=True).empty
            else pd.NA
        )
        for c in cols:
            self.df[c].fillna(modes[c], inplace=True)
        return self


class MacroCleaner(BaseCleaner):
    """
    Macro‐specific imputation can be added here if needed.
    Currently inherits only generic methods.
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
