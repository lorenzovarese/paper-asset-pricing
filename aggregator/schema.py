"""Typed configuration objects used to validate YAML aggregation specs."""

from __future__ import annotations
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field


class SourceConfig(BaseModel):
    """Location and merge-key information for a single data source."""

    name: str
    connector: Literal["local"]  # extendable (wrds, s3 …)
    path: Path
    join_on: Sequence[str]


class OutputConfig(BaseModel):
    """Configuration for the output file."""

    format: Optional[Literal["csv", "parquet"]] = None
    parquet_engine: Optional[Literal["auto", "pyarrow", "fastparquet"]] = "auto"
    parquet_compression: Optional[str] = "snappy"
    # csv_index: bool = False # Example if you wanted to configure CSV index writing


class OneHotConfig(BaseModel):
    """One-hot-encoding specification."""

    type: Literal["one_hot"]
    column: str
    prefix: str = ""
    drop_original: bool = True


class FillNaConfig(BaseModel):
    """Missing-value imputation specification."""

    type: Literal["fillna"]
    method: Literal["mean", "median", "ffill", "bfill"] = "mean"


class LagConfig(BaseModel):
    """Simple lag of macro variables."""

    type: Literal["lag"]
    columns: Sequence[str]
    periods: int = 1


TransformationConfig = OneHotConfig | FillNaConfig | LagConfig


class AggregationConfig(BaseModel):
    """Root model for the YAML specification."""

    sources: List[SourceConfig]
    transformations: List[Union[OneHotConfig, FillNaConfig, LagConfig]] = Field(
        default_factory=list
    )
    output: Optional[OutputConfig] = None
