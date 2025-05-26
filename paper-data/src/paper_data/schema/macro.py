import pandera.pandas as pa
from pandera import Column

macro_schema = pa.DataFrameSchema(
    columns={
        # required date
        "date": Column(pa.DateTime, required=True),
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
