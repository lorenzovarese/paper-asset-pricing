# src/paper_data/schema/macro.py
import pandera.polars as pa
from pandera.engines.polars_engine import Date
from pandera.errors import SchemaError
from polars.exceptions import ColumnNotFoundError

_macro_schema = pa.DataFrameSchema(
    columns={"date": pa.Column(Date, required=True)},
    strict=False,
    coerce=True,
)


def macro_schema(df):
    """
    Validate `df` against the macro schema.
    Raises SchemaError if 'date' is missing or cannot be coerced.
    """
    try:
        return _macro_schema(df)
    except ColumnNotFoundError as e:
        # Normalize Polars' missing‐column error into a SchemaError
        raise SchemaError(
            data=df, schema=_macro_schema, message="required column 'date' not found"
        ) from e
