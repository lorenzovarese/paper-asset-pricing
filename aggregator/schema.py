"""Typed configuration objects used to validate YAML aggregation specs."""

from __future__ import annotations
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field, validator


class DateHandlingConfig(BaseModel):
    """Configuration for how date columns are processed and aligned."""

    frequency: Literal["daily", "monthly", "yearly"] = Field(
        description="The intended frequency of the data in the source."
    )
    # For monthly, how to align the date (e.g., to month end or month start)
    # For now, we'll implicitly align monthly to month_end if frequency is 'monthly'
    # More options like 'month_start' or specific day can be added later.
    # monthly_alignment: Optional[Literal["month_end", "month_start"]] = "month_end"
    # Let's make it simpler for now: if frequency is monthly, it implies alignment.

    @validator("frequency")
    def monthly_is_supported(cls, v):
        if v not in ["monthly"]:  # Extend this list as you add support
            raise ValueError(
                f"Frequency '{v}' is not yet supported. Currently only 'monthly' is implemented."
            )
        return v


class SourceConfig(BaseModel):
    """Location and merge-key information for a single data source."""

    name: str
    connector: Literal["local"]  # extendable (wrds, s3 …)
    path: Path
    join_on: Sequence[str]
    level: Literal["firm", "macro"]
    is_primary_firm_base: Optional[bool] = Field(
        default=False,
        description="If True, this firm-level source is used as the base for merging other firm-level data. "
        "Only one firm-level source can be marked as primary. "
        "If multiple firm sources exist and none are marked primary, an error will be raised during merge.",
    )
    date_handling: Optional[DateHandlingConfig] = Field(
        default=None,
        description="Specifies how to handle and align date columns for this source, e.g., for monthly data.",
    )


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


class ImputationConfig(BaseModel):
    """Configuration for data imputation after merging and before transformations."""

    method: Literal["mean", "median"]
    group_by_column: str = Field(
        default="date", description="Column to group by for imputation (e.g., 'date')."
    )
    target_columns: Optional[List[str]] = Field(
        default=None,
        description="Specific columns to impute. If None, applies to eligible numeric columns not from macro sources.",
    )
    missing_threshold_warning: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Warning threshold for missing data ratio in a group-column (0.0 to 1.0).",
    )
    missing_threshold_error: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Error threshold for missing data ratio in a group-column (0.0 to 1.0).",
    )

    @validator("missing_threshold_error")
    def check_error_threshold_greater_than_warning(cls, v, values):
        if (
            "missing_threshold_warning" in values
            and v < values["missing_threshold_warning"]
        ):
            raise ValueError(
                "missing_threshold_error must be greater than or equal to missing_threshold_warning"
            )
        return v


class AggregationConfig(BaseModel):
    """Root model for the YAML specification."""

    sources: List[SourceConfig]
    imputation: Optional[ImputationConfig] = Field(
        default=None,
        description="Configuration for data imputation after merging sources.",
    )
    transformations: List[TransformationConfig] = Field(default_factory=list)
    output: Optional[OutputConfig] = None

    @validator("sources")
    def check_primary_firm_base_declaration(
        cls, sources: List[SourceConfig]
    ) -> List[SourceConfig]:
        """
        Validates that at most one firm-level source is marked as 'is_primary_firm_base: True'.
        The logic for requiring a primary base if multiple firm sources exist is handled at merge time.
        """
        primary_firm_sources_marked = [
            s for s in sources if s.level == "firm" and s.is_primary_firm_base
        ]
        if len(primary_firm_sources_marked) > 1:
            primary_names = [s.name for s in primary_firm_sources_marked]
            raise ValueError(
                f"Configuration error: Only one firm-level source can be marked as 'is_primary_firm_base: True'. "
                f"Found: {primary_names}"
            )

        # Validate that if date_handling is used, 'date' is a join key
        for src in sources:
            if src.date_handling and "date" not in [key.lower() for key in src.join_on]:
                raise ValueError(
                    f"Source '{src.name}': If 'date_handling' is specified, 'date' must be one of the 'join_on' keys."
                )
            if src.date_handling and src.date_handling.frequency == "monthly":
                # For now, we only support 'date' as the sole join key for monthly macro,
                # or 'permno', 'date' for monthly firm.
                # This is a simplification; more complex scenarios might need more thought.
                join_keys_lower = {key.lower() for key in src.join_on}
                if src.level == "macro" and join_keys_lower != {"date"}:
                    print(
                        f"Warning: Source '{src.name}' is 'macro' with 'monthly' date_handling. "
                        f"Expected 'join_on: [\"date\"]'. Found: {src.join_on}. This might lead to unexpected merge behavior if other keys are present."
                    )
                # No specific check for firm level yet, as ['permno', 'date'] is common.

        return sources
