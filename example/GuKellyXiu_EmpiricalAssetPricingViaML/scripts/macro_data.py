"""
Original macroeconomic dataset from Welch & Goyal (2024).

The data is available at:
https://docs.google.com/spreadsheets/d/1OIZg6htTK60wtnCVXvxAujvG1aKEOVYv/edit?gid=1660564386#gid=1660564386
The link was last checked on 2024-05-10 from Amit Goyal's website: https://sites.google.com/view/agoyal145
"""

from __future__ import annotations

from typing import Dict
import math
import re

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

# ────────────────────────────────────────────────────────────────────────────────
# 0. Helpers for significant–figure logic
# ────────────────────────────────────────────────────────────────────────────────


def _count_sig_digits_from_string(num_str: str) -> int:
    """Return the number of *significant* digits in *num_str*."""
    num_str = num_str.strip()
    if num_str.lower() in {"nan", ""}:
        return 0

    # Handle scientific notation
    if "e" in num_str or "E" in num_str:
        mantissa, _ = re.split(r"[eE]", num_str, maxsplit=1)
    else:
        mantissa = num_str

    mantissa = mantissa.lstrip("+-").replace(",", "")

    if "." in mantissa:
        int_part, frac_part = mantissa.split(".")
        int_part = int_part.lstrip("0")
        digits = int_part + frac_part
        return len(digits)
    else:
        return len(mantissa.lstrip("0"))


def _infer_sig_digits_per_column(
    df_raw: pd.DataFrame, numeric_cols: list[str]
) -> Dict[str, int]:
    """Return the *maximum* number of significant digits found in each column."""
    sig_map: Dict[str, int] = {col: 0 for col in numeric_cols}
    for col in numeric_cols:
        for val in df_raw[col].dropna().astype(str):
            sig_map[col] = max(sig_map[col], _count_sig_digits_from_string(val))
    return sig_map


def _round_to_sig(x: float | np.floating, sig: int) -> float | np.floating:
    """Round *x* to *sig* significant digits."""
    if pd.isna(x) or x == 0:
        return x
    ndigits = sig - int(math.floor(math.log10(abs(x)))) - 1
    return round(x, ndigits)


def _format_sig(x: float | np.floating, sig: int) -> str:
    """Return a string with *sig* significant digits (plain notation when possible)."""
    if pd.isna(x):  # type: ignore[call-arg]
        return ""
    return f"{x:.{sig}g}"


# ────────────────────────────────────────────────────────────────────────────────
# 1. Load CSV → month‑end index DataFrame
# ────────────────────────────────────────────────────────────────────────────────


def load_monthly_data(
    path: str,
    date_col: str = "date",
    date_format: str = "%Y%m",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    dtype: str | np.dtype = "float64",
) -> pd.DataFrame:
    """Read Welch–Goyal monthly data and attach significant‑digit metadata."""

    df = pd.read_csv(path)
    df.replace({",": ""}, regex=True, inplace=True)

    numeric_cols = df.columns.drop(date_col).tolist()
    df.attrs["sig_digits_per_col"] = _infer_sig_digits_per_column(df, numeric_cols)

    df[numeric_cols] = (
        df[numeric_cols].apply(pd.to_numeric, errors="coerce").astype(dtype, copy=False)
    )

    date_idx = pd.to_datetime(df[date_col].astype(str), format=date_format) + MonthEnd(
        0
    )
    df.drop(columns=date_col, inplace=True)
    df.index = pd.Index(date_idx, name=date_col)

    if start is not None or end is not None:
        start_ts = pd.to_datetime(start) + MonthEnd(0) if start else None
        end_ts = pd.to_datetime(end) + MonthEnd(0) if end else None
        df = df.loc[start_ts:end_ts]

    return df.sort_index()


# ────────────────────────────────────────────────────────────────────────────────
# 2. Construct predictors (high‑precision → significant‑figure rounding)
# ────────────────────────────────────────────────────────────────────────────────


def construct_welch_goyal_predictors(
    raw: pd.DataFrame, sig_cap: int = 6
) -> pd.DataFrame:
    hp = raw.astype(np.longdouble, copy=False)
    sig_map: Dict[str, int] = raw.attrs.get("sig_digits_per_col", {})

    p = pd.DataFrame(index=hp.index, dtype="longdouble")
    p["dp"] = np.log(hp["D12"] / hp["Index"])
    p["ep_macro"] = np.log(hp["E12"] / hp["Index"]) # Add _macro suffix to avoid confusion with ep in firm characteristics
    p["bm_macro"] = hp["b/m"]
    p["ntis"] = hp["ntis"]
    p["tbl"] = hp["tbl"]
    p["tms"] = hp["lty"] - hp["tbl"]
    p["dfy"] = hp["BAA"] - hp["AAA"]
    p["svar"] = hp["svar"]

    target_sig: Dict[str, int] = {
        "dp": min(sig_cap, sig_map.get("D12", sig_cap), sig_map.get("Index", sig_cap)),
        "ep": min(sig_cap, sig_map.get("E12", sig_cap), sig_map.get("Index", sig_cap)),
        "bm": min(sig_cap, sig_map.get("b/m", sig_cap)),
        "ntis": min(sig_cap, sig_map.get("ntis", sig_cap)),
        "tbl": min(sig_cap, sig_map.get("tbl", sig_cap)),
        "tms": min(sig_cap, sig_map.get("lty", sig_cap), sig_map.get("tbl", sig_cap)),
        "dfy": min(sig_cap, sig_map.get("BAA", sig_cap), sig_map.get("AAA", sig_cap)),
        "svar": min(sig_cap, sig_map.get("svar", sig_cap)),
    }

    for col, sig in target_sig.items():
        p[col] = p[col].apply(_round_to_sig, sig=sig)

    p = p.astype("float64")
    p.attrs["target_sig"] = target_sig
    return p


# ────────────────────────────────────────────────────────────────────────────────
# 3. Driver script
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    macro_path = "data/welch_goyal/Welch_Goyal_2024_macro_monthly.csv"

    raw = load_monthly_data(
        path=str(macro_path),
        date_col="yyyymm",
        start="1956-12",
        end="2021-12",
        dtype="float64",
    )

    wg_pred = construct_welch_goyal_predictors(raw, sig_cap=6)

    # Prepare a copy with string representation preserving significant digits
    csv_ready = wg_pred.copy()
    for col, sig in wg_pred.attrs["target_sig"].items():
        csv_ready[col] = csv_ready[col].apply(lambda x, s=sig: _format_sig(x, s))

    csv_ready.to_csv(
        "data/welch_goyal/macro_1956_2021.csv",
        date_format="%Y%m%d",
        index_label="date",
    )
