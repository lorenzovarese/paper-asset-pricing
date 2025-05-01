import pytest
import pandas as pd
from pandera.errors import SchemaError
from paper_data.schema.firm import firm_schema  # type: ignore[import-untyped]


def test_valid_firm_schema():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "company_id": [1, 2],
            "ret": [0.1, -0.1],
            "extra": [3.5, 4.0],
        }
    )
    validated = firm_schema(df)
    pd.testing.assert_frame_equal(validated, df)


def test_invalid_date_column():
    df = pd.DataFrame({"date": ["not a date"], "company_id": [1], "ret": [0.0]})
    with pytest.raises(SchemaError):
        firm_schema(df)
