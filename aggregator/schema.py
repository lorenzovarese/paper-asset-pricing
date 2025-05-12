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
    level: Literal["firm", "macro"]  # New field to distinguish source type


class OutputConfig(BaseModel):
    """Configuration for the output file."""

    format: Optional[Literal["csv", "parquet"]] = None
    parquet_engine: Optional[Literal["auto", "pyarrow", "fastparquet"]] = "auto"
    parquet_compression: Optional[str] = "snappy"


class OneHotConfig(BaseModel):
    """One-hot-encoding specification."""

    type: Literal["one_hot"]
    column: str
    prefix: str
    drop_original: bool


class FillNaGroupedConfig(BaseModel):
    """Configuration for grouped fillna operations."""

    type: Literal["fillna_grouped"]
    method: Literal["mean", "median"]
    group_by_column: str
    columns: Optional[List[str]] = None


class LagConfig(BaseModel):
    """Lagging specification."""

    type: Literal["lag"]
    columns: List[str]
    periods: int


TransformationConfig = Union[OneHotConfig, FillNaGroupedConfig, LagConfig]


class AggregationConfig(BaseModel):
    """Root model for the YAML specification."""

    sources: List[SourceConfig]
    transformations: List[TransformationConfig] = Field(default_factory=list)
    output: Optional[OutputConfig] = None
