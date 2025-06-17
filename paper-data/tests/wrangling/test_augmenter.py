import pytest
import polars as pl
from polars.testing import assert_frame_equal
from datetime import date

from paper_data.wrangling.augmenter import (  # type: ignore
    merge_datasets,
    lag_columns,
    create_macro_firm_interactions,
    create_macro_firm_interactions_lazy,
    create_dummies,
)

# --- Fixtures for sample data ---


@pytest.fixture
def sample_panel_df():
    """A sample panel DataFrame for testing."""
    return pl.DataFrame(
        {
            "date": [
                date(2020, 1, 31),
                date(2020, 2, 29),
                date(2020, 3, 31),
                date(2020, 1, 31),
                date(2020, 2, 29),
                date(2020, 3, 31),
            ],
            "id": [1, 1, 1, 2, 2, 2],
            "value": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            "other": [1, 2, 3, 4, 5, 6],
        }
    ).sort("id", "date")


@pytest.fixture
def sample_interaction_df():
    """A sample DataFrame for testing interactions."""
    return pl.DataFrame(
        {
            "firm_char1": [1, 2, 3],
            "firm_char2": [4, 5, 6],
            "macro_var1": [10, 10, 10],
            "macro_var2": [0.5, 0.5, 0.5],
        }
    )


# --- Tests for merge_datasets ---


def test_merge_datasets():
    df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df2 = pl.DataFrame({"a": [1], "c": [5]})
    merged = merge_datasets(df1, df2, on_cols=["a"], how="left")
    assert "c" in merged.columns
    assert merged.shape == (2, 3)
    assert merged.filter(pl.col("a") == 2)["c"].is_null().all()


# --- Tests for lag_columns ---


def test_lag_columns_panel_data(sample_panel_df):
    """Test lagging on panel data with an id_col."""
    result = lag_columns(
        sample_panel_df, "date", "id", ["value"], 1, False, False, False
    )
    assert "value_lag_1" in result.columns
    expected = pl.Series("value_lag_1", [None, 10.0, 20.0, None, 100.0, 200.0])
    assert_frame_equal(result.select("value_lag_1"), expected.to_frame())


def test_lag_columns_time_series(sample_panel_df):
    """Test lagging on time-series data (id_col=None)."""
    ts_df = sample_panel_df.filter(pl.col("id") == 1).drop("id")
    result = lag_columns(ts_df, "date", None, ["value"], 1, False, False, False)
    expected = pl.Series("value_lag_1", [None, 10.0, 20.0])
    assert_frame_equal(result.select("value_lag_1"), expected.to_frame())


def test_lag_columns_drop_and_restore(sample_panel_df):
    """Test drop_original_cols_after_lag and restore_names flags."""
    result = lag_columns(sample_panel_df, "date", "id", ["value"], 1, True, True, False)
    assert "value" in result.columns
    assert "value_lag_1" not in result.columns
    # The original 'value' column is now the lagged one
    expected = pl.Series("value", [None, 10.0, 20.0, None, 100.0, 200.0])
    assert_frame_equal(result.select("value"), expected.to_frame())


def test_lag_columns_drop_nans(sample_panel_df):
    """Test drop_generated_nans flag."""
    result = lag_columns(
        sample_panel_df, "date", "id", ["value"], 1, False, False, True
    )
    # The two rows with generated NaNs should be dropped
    # The original df has 4 columns, plus 1 new lagged column = 5 columns
    assert result.shape == (4, 5)
    assert not result["value_lag_1"].is_null().any()


def test_lag_columns_no_cols_to_lag(sample_panel_df):
    """Test that it returns the original DataFrame if cols_to_lag is empty."""
    result = lag_columns(sample_panel_df, "date", "id", [], 1, False, False, False)
    assert_frame_equal(result, sample_panel_df)


def test_lag_columns_raises_error_on_missing_cols(sample_panel_df):
    """Test that it raises ValueError for non-existent columns."""
    with pytest.raises(ValueError, match="Date column 'wrong_date' not found"):
        lag_columns(
            sample_panel_df, "wrong_date", "id", ["value"], 1, False, False, False
        )

    with pytest.raises(ValueError, match="Columns specified for lagging not found"):
        lag_columns(
            sample_panel_df, "date", "id", ["wrong_value"], 1, False, False, False
        )


# --- Tests for create_macro_firm_interactions (Eager) ---


def test_create_macro_firm_interactions_eager(sample_interaction_df):
    result = create_macro_firm_interactions(
        sample_interaction_df,
        macro_columns=["macro_var1", "macro_var2"],
        firm_columns=["firm_char1"],
        drop_macro_columns=False,
    )
    assert "firm_char1_x_macro_var1" in result.columns
    assert "firm_char1_x_macro_var2" in result.columns
    assert result["firm_char1_x_macro_var1"].to_list() == [10, 20, 30]
    assert result["firm_char1_x_macro_var2"].to_list() == [0.5, 1.0, 1.5]


def test_create_macro_firm_interactions_drop_macro(sample_interaction_df):
    result = create_macro_firm_interactions(
        sample_interaction_df,
        macro_columns=["macro_var1", "macro_var2"],
        firm_columns=["firm_char1"],
        drop_macro_columns=True,
    )
    assert "macro_var1" not in result.columns
    assert "macro_var2" not in result.columns
    assert "firm_char1_x_macro_var1" in result.columns


def test_create_macro_firm_interactions_raises_error_on_missing_cols(
    sample_interaction_df,
):
    with pytest.raises(
        ValueError, match="Macro columns specified for interaction not found"
    ):
        create_macro_firm_interactions(
            sample_interaction_df, ["bad_macro"], ["firm_char1"], False
        )

    with pytest.raises(
        ValueError, match="Firm columns specified for interaction not found"
    ):
        create_macro_firm_interactions(
            sample_interaction_df, ["macro_var1"], ["bad_firm"], False
        )


# --- Tests for create_macro_firm_interactions_lazy ---


def test_create_macro_firm_interactions_lazy(sample_interaction_df):
    ldf = sample_interaction_df.lazy()
    result_ldf = create_macro_firm_interactions_lazy(
        ldf,
        macro_columns=["macro_var1"],
        firm_columns=["firm_char1"],
        drop_macro_columns=False,
    )
    result_df = result_ldf.collect()
    assert "firm_char1_x_macro_var1" in result_df.columns
    assert result_df["firm_char1_x_macro_var1"].to_list() == [10, 20, 30]


# --- Tests for create_dummies ---


def test_create_dummies():
    df = pl.DataFrame({"category": ["A", "B", "A", "C"]})
    result = create_dummies(df, "category", drop_original_col=False)
    assert "category_A" in result.columns
    assert "category_B" in result.columns
    assert "category_C" in result.columns
    assert "category" in result.columns
    assert result["category_A"].to_list() == [1, 0, 1, 0]


def test_create_dummies_drop_original():
    df = pl.DataFrame({"category": ["A", "B"]})
    result = create_dummies(df, "category", drop_original_col=True)
    assert "category" not in result.columns
    assert "category_A" in result.columns


def test_create_dummies_raises_error_on_missing_col():
    df = pl.DataFrame({"category": ["A", "B"]})
    with pytest.raises(ValueError, match="Column 'wrong_col' not found"):
        create_dummies(df, "wrong_col", False)
