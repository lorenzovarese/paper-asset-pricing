"""
Connector for datasets hosted on the Hugging Face Hub.
Requires `datasets>=2.0`.
"""

from __future__ import annotations

import polars as pl
from datasets import Dataset, DatasetDict, load_dataset  # type: ignore[import-untyped]

from .base import DataConnector


class HuggingFaceConnector(DataConnector):
    """
    Connector for datasets hosted on the Hugging Face Hub.
    Requires `datasets>=2.0`.
    """

    def __init__(
        self,
        repo_id: str,
        split: str | None = None,
        **load_kwargs,  # forwarded to `load_dataset`
    ) -> None:
        self.repo_id = repo_id
        self.split = split
        self.load_kwargs = load_kwargs

    def get_data(self) -> pl.DataFrame:
        raw = load_dataset(self.repo_id, split=self.split, **self.load_kwargs)

        # Helper to coerce any Polars output into a DataFrame
        def ensure_df(x: pl.DataFrame | pl.Series) -> pl.DataFrame:
            if isinstance(x, pl.Series):
                return x.to_frame()
            return x

        # 1) Single Dataset → Arrow → Polars
        if isinstance(raw, Dataset):
            table = raw.data.table
            df = pl.from_arrow(table)
            return ensure_df(df)

        # 2) DatasetDict → pick split → Arrow → Polars
        if isinstance(raw, DatasetDict):
            if self.split is None:
                raise ValueError("When loading a DatasetDict you must specify `split`.")
            ds = raw[self.split]
            table = ds.data.table
            df = pl.from_arrow(table)
            return ensure_df(df)

        # 3) Objects with a to_pandas() method
        try:
            import pandas as pd

            to_pd = getattr(raw, "to_pandas", None)
            if callable(to_pd):
                pdf = to_pd()
                if isinstance(pdf, pl.DataFrame):
                    return pdf
                if isinstance(pdf, pd.DataFrame):
                    return pl.from_pandas(pdf)
        except ImportError:
            pass

        # 4) Pandas DataFrame fallback
        try:
            import pandas as pd

            if isinstance(raw, pd.DataFrame):
                return pl.from_pandas(raw)
        except ImportError:
            pass

        # 5) List of dicts → direct Polars
        if isinstance(raw, list):
            return pl.DataFrame(raw)

        raise TypeError(f"Cannot convert {type(raw)} to polars.DataFrame")
