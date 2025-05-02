import pandera.polars as pa
from pandera import Column
from pandera.engines.polars_engine import DateTime

firm_schema = pa.DataFrameSchema(
    columns={
        # required: parse/validate as Timestamp
        "date": pa.Column(DateTime, required=True),
        # required: integer identifier
        "company_id": pa.Column(int, required=True),
        # required: float return (nullable ok)
        "ret": pa.Column(float, required=True),
        # catch-all for any other columns → float features
        r"^(?!date$|company_id$).*": pa.Column(
            float,
            required=False,
            regex=True,
        ),
    },
    strict=False,  # allow columns not explicitly listed (they’ll match the regex)
    coerce=True,  # cast columns to the specified dtypes
)
