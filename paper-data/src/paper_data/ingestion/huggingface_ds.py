"""
Connector for datasets hosted on the Hugging Face Hub.
Requires `datasets>=2.0`.
"""

from __future__ import annotations

import pandas as pd
from datasets import load_dataset  # type: ignore[import-untyped]

from .base import BaseConnector


class HuggingFaceConnector(BaseConnector):
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

    def get_data(self) -> pd.DataFrame:
        ds = load_dataset(self.repo_id, split=self.split, **self.load_kwargs)
        return ds.to_pandas()  # type: ignore[return-value]
