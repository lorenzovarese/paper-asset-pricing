import polars as pl
from datetime import date
from paper_data.wrangling.cleaner import impute_monthly, scale_to_range  # type: ignore


def test_impute_monthly_fallback():
    df = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 1)], "x": [1, None]})
    res = impute_monthly(df, "date", ["x"], [], fallback_to_zero=True)
    assert res["x"].null_count() == 0


def test_scale_to_range():
    df = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 1)], "x": [0, 10]})
    res = scale_to_range(df, ["x"], "date", 0, 1)
    assert res["x"].min() == 0
    assert res["x"].max() == 1
