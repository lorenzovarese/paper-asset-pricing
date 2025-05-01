import pandera.pandas as pa
from pandera import Column

firm_schema = pa.DataFrameSchema(
    columns={
        # required: parse/validate as Timestamp
        "date": Column(pa.DateTime, required=True),
        # required: integer identifier
        "company_id": Column(int, required=True),
        # required: float return (nullable ok)
        "ret": Column(float, required=True),
        # catch-all for any other columns → float features
        r"^(?!date$|company_id$).*": Column(
            float,
            required=False,
            regex=True,
        ),
    },
    strict=False,  # allow columns not explicitly listed (they’ll match the regex)
    coerce=True,  # cast columns to the specified dtypes
)
