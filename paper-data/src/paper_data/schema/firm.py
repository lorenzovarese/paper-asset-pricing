import pandera.polars as pa
from pandera.engines.polars_engine import Date

firm_schema = pa.DataFrameSchema(
    columns={
        "date": pa.Column(Date, required=True),
        "company_id": pa.Column(int, required=True),
        "ret": pa.Column(float, required=True),
    },
    strict=False,  # allow any other columns
    coerce=True,  # cast date/ids/ret
)
