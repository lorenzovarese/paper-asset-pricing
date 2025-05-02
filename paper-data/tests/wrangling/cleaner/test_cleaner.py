import polars as pl
import pytest

from paper_data.wrangling.cleaner.cleaner import (  # type: ignore[import-untyped]
    RawDataset,
    BaseCleaner,
    FirmCleaner,
    MacroCleaner,
    CleanerFactory,
)


def test_normalize_columns_and_rename_and_parse_date_and_clean_numeric():
    # Create DataFrame with messy columns, date in yyyymm format, numeric as strings
    data = {
        " yyYYmm ": ["202001", "202002"],
        "Company_ID": ["1", "2"],
        "Value": ["10.5", "20.0"],
    }
    df = pl.DataFrame(data)
    raw = RawDataset(df.clone(), objective="firm")
    cleaner = BaseCleaner(raw)
    # Normalize columns
    cleaner.normalize_columns()
    assert "yyyymm" in cleaner.df.columns
    assert "company_id" in cleaner.df.columns
    assert "value" in cleaner.df.columns

    # Rename date column
    cleaner.rename_date_column(candidates=("yyyymm",), target="date")
    assert "date" in cleaner.df.columns

    # Parse date with monthly start
    cleaner.parse_date(date_col="date", date_format="%Y%m", monthly_option="start")
    from polars.datatypes import Date

    assert cleaner.df["date"].dtype == Date()
    # The parsed dates should correspond to first day of month
    assert cleaner.df["date"].dt.day().eq(1).all()

    # Clean numeric column
    cleaner.clean_numeric_column("value")
    assert cleaner.df["value"].dtype == pl.Float64


def test_impute_constant():
    df = pl.DataFrame({"a": [1, None, 3], "b": [None, 2, None]})
    raw = RawDataset(df.clone(), objective="macro")
    cleaner = BaseCleaner(raw)
    cleaner.impute_constant(["a", "b"], value=0)
    assert cleaner.df["a"].to_list() == [1, 0, 3]
    assert cleaner.df["b"].to_list() == [0, 2, 0]


def test_firm_cleaner_cross_section_imputations():
    # Create sample firm data for two months and two companies
    dates = pl.Series(
        ["2021-01-15", "2021-01-20", "2021-02-15", "2021-02-20"]
    ).str.to_date("%Y-%m-%d")
    perm = [1, 2, 1, 2]
    feature = [10.0, None, None, 40.0]
    df = pl.DataFrame({"date": dates, "company_id": perm, "feature": feature})
    raw = RawDataset(df.clone(), objective="firm")
    cleaner = FirmCleaner(raw)

    # Test median imputation: January median=(10), February median=(40)
    cleaner.impute_cross_section_median(["feature"])
    jan_vals = cleaner.df.filter(cleaner.df["date"].dt.strftime("%Y-%m") == "2021-01")[
        "feature"
    ]
    feb_vals = cleaner.df.filter(cleaner.df["date"].dt.strftime("%Y-%m") == "2021-02")[
        "feature"
    ]
    assert jan_vals.to_list() == [10.0, 10.0]
    assert feb_vals.to_list() == [40.0, 40.0]

    # Reset raw df and test mean imputation: same as median here
    raw = RawDataset(df.clone(), objective="firm")
    cleaner = FirmCleaner(raw)
    cleaner.impute_cross_section_mean(["feature"])
    jan_vals = cleaner.df.filter(cleaner.df["date"].dt.strftime("%Y-%m") == "2021-01")[
        "feature"
    ]
    feb_vals = cleaner.df.filter(cleaner.df["date"].dt.strftime("%Y-%m") == "2021-02")[
        "feature"
    ]
    assert jan_vals.to_list() == [10.0, 10.0]
    assert feb_vals.to_list() == [40.0, 40.0]

    # Test mode imputation when mode is defined
    df_mode = pl.DataFrame({"date": dates, "company_id": perm, "flag": [1, 1, None, 2]})
    raw_mode = RawDataset(df_mode.clone(), objective="firm")
    cleaner_mode = FirmCleaner(raw_mode)
    cleaner_mode.impute_cross_section_mode(["flag"])
    jan_flags = cleaner_mode.df.filter(
        cleaner_mode.df["date"].dt.strftime("%Y-%m") == "2021-01"
    )["flag"]
    feb_flags = cleaner_mode.df.filter(
        cleaner_mode.df["date"].dt.strftime("%Y-%m") == "2021-02"
    )["flag"]
    # January mode is 1
    assert jan_flags.to_list() == [1, 1]
    # February mode is 2
    assert feb_flags.to_list() == [2, 2]


def test_cleaner_objective_validation():
    # BaseCleaner _require should raise for invalid objective
    df = pl.DataFrame({"a": [1]})

    # FirmCleaner init with wrong objective
    with pytest.raises(ValueError):
        FirmCleaner(RawDataset(df, objective="macro"))
    # MacroCleaner init with wrong objective
    with pytest.raises(ValueError):
        MacroCleaner(RawDataset(df, objective="firm"))


def test_cleaner_factory():
    df = pl.DataFrame({"a": [1]})
    raw_f = RawDataset(df, objective="firm")
    raw_m = RawDataset(df, objective="macro")
    assert isinstance(CleanerFactory.get_cleaner(raw_f), FirmCleaner)
    assert isinstance(CleanerFactory.get_cleaner(raw_m), MacroCleaner)
    with pytest.raises(ValueError):
        CleanerFactory.get_cleaner(RawDataset(df, objective="unknown"))  # type: ignore[call-arg]
