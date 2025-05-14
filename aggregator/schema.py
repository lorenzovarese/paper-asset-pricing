"""Typed configuration objects used to validate YAML aggregation specs."""

from __future__ import annotations
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Union

from pydantic import (
    BaseModel,
    Field,
    root_validator,
    validator,
)


class DateHandlingConfig(BaseModel):
    """Configuration for how date columns are processed and aligned."""

    frequency: Literal["daily", "monthly", "yearly"] = Field(
        description="The intended frequency of the data in the source."
    )

    @validator("frequency")
    def monthly_is_supported(cls, v):
        if v not in ["monthly"]:
            raise ValueError(
                f"Frequency '{v}' is not yet supported. Currently only 'monthly' is implemented."
            )
        return v


class SourceConfig(BaseModel):
    """Location and merge-key information for a single data source."""

    name: str
    connector: Literal["local"]
    path: Path
    join_on: Sequence[str]
    level: Literal["firm", "macro"]  # This will now determine the automatic suffix
    is_primary_firm_base: Optional[bool] = Field(default=False)
    date_handling: Optional[DateHandlingConfig] = Field(default=None)
    # column_suffix: Optional[str] = Field(default=None) # REMOVED


class OutputConfig(BaseModel):
    format: Optional[Literal["csv", "parquet"]] = None
    parquet_engine: Optional[Literal["auto", "pyarrow", "fastparquet"]] = "auto"
    parquet_compression: Optional[str] = "snappy"


class OneHotConfig(BaseModel):
    type: Literal["one_hot"]
    column: str  # This will refer to the auto-suffixed column name
    prefix: str
    drop_original: bool


class LagConfig(BaseModel):
    type: Literal["lag"]
    columns: List[
        str
    ]  # These will refer to auto-suffixed column names or inferred names
    periods: int


class CleanNumericConfig(BaseModel):
    type: Literal["clean_numeric"]
    columns: List[
        str
    ]  # These will refer to auto-suffixed column names or inferred names
    action: Literal["to_nan"] = Field(default="to_nan")


class GroupedFillMissingConfig(BaseModel):
    type: Literal["grouped_fill_missing"]
    method: Literal["mean", "median"]
    group_by_column: str  # Typically 'date' or 'permno', not auto-suffixed
    columns: Optional[List[str]] = (
        None  # These will refer to auto-suffixed or inferred names
    )
    missing_threshold_warning: float = Field(default=0.5, ge=0.0, le=1.0)
    missing_threshold_error: float = Field(default=0.8, ge=0.0, le=1.0)

    @validator("missing_threshold_error")
    def check_error_threshold_greater_than_warning(cls, v, values):
        if (
            "missing_threshold_warning" in values
            and v < values["missing_threshold_warning"]
        ):
            raise ValueError(
                "missing_threshold_error must be >= missing_threshold_warning"
            )
        return v


class ExpandCartesianConfig(BaseModel):
    type: Literal["expand_cartesian"]
    macro_columns: List[str]  # User provides base names
    firm_columns: List[str]  # User provides base names
    infer_suffix: bool = Field(
        default=True,  # CHANGED DEFAULT: Suffixes are now auto-applied, so inference is the natural mode.
        description="If True (default), attempts to find auto-suffixed versions (_macro, _firm) of macro/firm columns. If False, expects exact pre-suffixed names provided by user (less common with auto-suffixing).",
    )


class DropColumnsConfig(BaseModel):
    type: Literal["drop_columns"]
    macro_columns: Optional[List[str]] = Field(
        default=None,
        description="Base names of macro-level columns to drop (e.g., 'dp' will target 'dp_macro').",
    )
    firm_columns: Optional[List[str]] = Field(
        default=None,
        description="Base names of firm-level columns to drop (e.g., 'beta' will target 'beta_firm').",
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="Exact names of columns to drop (e.g., 'permno', 'specific_suffixed_col').",
    )

    @root_validator(skip_on_failure=True)
    def check_at_least_one_column_list_provided(cls, values):
        macro_cols = values.get("macro_columns")
        firm_cols = values.get("firm_columns")
        exact_cols = values.get("columns")
        if not macro_cols and not firm_cols and not exact_cols:
            raise ValueError(
                "For 'drop_columns' transformation, at least one of 'macro_columns', 'firm_columns', or 'columns' must be specified."
            )
        return values


TransformationConfig = Union[
    OneHotConfig,
    LagConfig,
    CleanNumericConfig,
    GroupedFillMissingConfig,
    ExpandCartesianConfig,
    DropColumnsConfig,
]


class AggregationConfig(BaseModel):
    sources: List[SourceConfig]
    transformations: List[TransformationConfig] = Field(default_factory=list)
    output: Optional[OutputConfig] = None

    @validator("sources")
    def validate_sources_config(cls, sources: List[SourceConfig]) -> List[SourceConfig]:
        primary_firm_sources_marked = [
            s for s in sources if s.level == "firm" and s.is_primary_firm_base
        ]
        if len(primary_firm_sources_marked) > 1:
            primary_names = [s.name for s in primary_firm_sources_marked]
            raise ValueError(
                f"Config error: Only one firm-level source can be 'is_primary_firm_base: True'. Found: {primary_names}"
            )

        for src in sources:
            join_keys_lower = {key.lower() for key in src.join_on}
            if src.date_handling and "date" not in join_keys_lower:
                raise ValueError(
                    f"Source '{src.name}': If 'date_handling' is specified, 'date' must be a join_on key."
                )
            if src.date_handling and src.date_handling.frequency == "monthly":
                if src.level == "macro" and join_keys_lower != {"date"}:
                    print(
                        f"Warning: Source '{src.name}' (macro, monthly) has join_on: {src.join_on}. Expected just ['date']."
                    )

        source_names = [s.name for s in sources]
        if len(source_names) != len(set(source_names)):
            raise ValueError(
                f"Config error: Source names must be unique. Duplicates: {[name for name in set(source_names) if source_names.count(name) > 1]}"
            )

        # The check for duplicate column names *after auto-suffixing* will be done in DataAggregator.load()
        return sources
