import pytest
import polars as pl
from pandera.errors import SchemaError
from paper_data.schema.firm import firm_schema  # type: ignore[import-untyped]


def test_valid_firm_schema():
    df = pl.DataFrame(
        {
            "date": (pl.Series(["2020-01-01", "2020-01-02"]).str.to_date("%Y-%m-%d")),
            "company_id": [1, 2],
            "ret": [0.1, -0.1],
            "extra": [3.5, 4.0],
        }
    )
    validated = firm_schema(df)
    pl.testing.assert_frame_equal(validated, df)


def test_invalid_date_column():
    df = pl.DataFrame({"date": ["not a date"], "company_id": [1], "ret": [0.0]})
    with pytest.raises(SchemaError):
        firm_schema(df)
