import pandera.polars as pa
from pandera import Column
from pandera.engines.polars_engine import DateTime

macro_schema = pa.DataFrameSchema(
    columns={
        # required date
        "date": Column(DateTime, required=True),
        # catch-all: any macro feature columns → float
        r"^(?!date$).*": Column(
            float,
            required=False,
            regex=True,
        ),
    },
    strict=False,
    coerce=True,
)
