"""Typed configuration objects used to validate YAML aggregation specs."""

from __future__ import annotations
from pathlib import Path
from typing import Literal, Sequence

import pydantic


class SourceConfig(pydantic.BaseModel):
    """Location and merge-key information for a single data source."""

    name: str
    connector: Literal["local"]  # extendable (wrds, s3 …)
    path: Path
    join_on: Sequence[str]


class OneHotConfig(pydantic.BaseModel):
    """One-hot-encoding specification."""

    type: Literal["one_hot"]
    column: str
    prefix: str = ""
    drop_original: bool = True


class FillNaConfig(pydantic.BaseModel):
    """Missing-value imputation specification."""

    type: Literal["fillna"]
    method: Literal["mean", "median", "ffill", "bfill"] = "mean"


class LagConfig(pydantic.BaseModel):
    """Simple lag of macro variables."""

    type: Literal["lag"]
    columns: Sequence[str]
    periods: int = 1


TransformationConfig = OneHotConfig | FillNaConfig | LagConfig


class AggregationConfig(pydantic.BaseModel):
    """Top-level YAML structure."""

    sources: Sequence[SourceConfig]
    transformations: Sequence[TransformationConfig] = []
