import pytest
import polars as pl
from pandera.errors import SchemaError
from paper_data.schema.macro import macro_schema  # type: ignore[import-untyped]


def test_valid_macro_schema():
    df = pl.DataFrame(
        {
            "date": (pl.Series(["2021-05-01", "2021-06-01"]).str.to_date("%Y-%m-%d")),
            "feature1": [1.2, 2.3],
            "feature2": [3.4, 4.5],
        }
    )
    validated = macro_schema(df)
    pl.testing.assert_frame_equal(validated, df)


def test_missing_date_column():
    df = pl.DataFrame({"feature": [1.0, 2.0]})
    with pytest.raises(SchemaError):
        macro_schema(df)
