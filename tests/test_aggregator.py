import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from pathlib import Path
import yaml

# Ensure imports from your project work (conftest.py helps with sys.path)
from aggregator.schema import AggregationConfig
from aggregator.aggregate import DataAggregator, aggregate_from_yaml
# core.settings.DATA_DIR should be set by conftest.py to point to tests/data/


# --- Helper Functions ---
def create_yaml_config_file(
    config_dict: dict, tmp_path: Path, filename="test_config.yaml"
) -> Path:
    """Creates a YAML config file in the temporary path."""
    config_file = tmp_path / filename
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)
    return config_file


def get_raw_test_df(filename: str) -> pd.DataFrame:
    """Loads a raw CSV from the tests/data directory for comparison."""
    # This path is relative to this test file's location
    raw_df_path = Path(__file__).parent / "data" / filename
    df = pd.read_csv(raw_df_path)
    # Mimic some initial processing for fair comparison if needed (e.g., date parsing)
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        except ValueError:
            df["date"] = pd.to_datetime(df["date"])  # Fallback
    if "PERMNO" in df.columns:  # From ret.csv
        df = df.rename(columns={"PERMNO": "permno"})
    df.columns = df.columns.str.lower()
    return df


# --- Base Configuration for Tests ---
@pytest.fixture
def base_config_dict_paths_as_filenames():
    """
    Provides a base configuration dictionary.
    Paths are just filenames, relying on DATA_DIR being set to 'tests/data/'.
    """
    return {
        "sources": [
            {
                "name": "firm_chars",
                "connector": "local",
                "path": "firm.csv",  # Will be resolved as tests/data/firm.csv
                "join_on": ["permno", "date"],
                "level": "firm",
            },
            {
                "name": "crsp_returns",
                "connector": "local",
                "path": "ret.csv",  # Will be resolved as tests/data/ret.csv
                "join_on": ["permno", "date"],
                "level": "firm",
            },
            {
                "name": "macro_data",
                "connector": "local",
                "path": "macro.csv",  # Will be resolved as tests/data/macro.csv
                "join_on": ["date"],
                "level": "macro",
            },
        ],
        "transformations": [],
        "output": {"format": "csv"},  # Default output for tests if not specified
    }


# --- Test Cases ---


def test_load_data_frames_correctly(base_config_dict_paths_as_filenames, tmp_path):
    """Test that individual dataframes are loaded and preprocessed."""
    cfg = AggregationConfig.model_validate(base_config_dict_paths_as_filenames)
    aggregator = DataAggregator(cfg)
    aggregator.load()

    assert "firm_chars" in aggregator._frames
    assert "crsp_returns" in aggregator._frames
    assert "macro_data" in aggregator._frames

    df_firm = aggregator._frames["firm_chars"]
    assert "permno" in df_firm.columns
    assert "date" in df_firm.columns
    assert pd.api.types.is_datetime64_any_dtype(df_firm["date"])
    assert all(col == col.lower() for col in df_firm.columns)

    df_ret = aggregator._frames["crsp_returns"]
    assert "permno" in df_ret.columns  # Was PERMNO
    assert "ret" in df_ret.columns  # Was RET
    assert pd.api.types.is_datetime64_any_dtype(df_ret["date"])

    df_macro = aggregator._frames["macro_data"]
    assert "dp" in df_macro.columns
    assert pd.api.types.is_datetime64_any_dtype(df_macro["date"])


def test_merge_firm_chars_primary(base_config_dict_paths_as_filenames, tmp_path):
    """Test merge with firm_chars as the primary firm base."""
    config_dict = base_config_dict_paths_as_filenames.copy()
    config_dict["sources"][0]["is_primary_firm_base"] = True  # firm_chars is primary

    cfg_file = create_yaml_config_file(config_dict, tmp_path)
    df_merged, _ = aggregate_from_yaml(cfg_file)

    raw_firm_df = get_raw_test_df("firm.csv")
    assert len(df_merged) == len(raw_firm_df)  # All rows from firm_chars should be kept

    # PERMNO 12345 is only in ret.csv, should NOT be in the output
    assert 12345 not in df_merged["permno"].unique()

    # Check data for a specific permno/date
    row_10001_jan = df_merged[
        (df_merged["permno"] == 10001)
        & (df_merged["date"] == pd.Timestamp("2002-01-31"))
    ]
    assert not row_10001_jan.empty
    assert row_10001_jan["char1"].iloc[0] == 1
    assert row_10001_jan["ret"].iloc[0] == 1
    assert row_10001_jan["dp"].iloc[0] == 1  # Macro data

    # Check a row from firm_chars that has a match in ret.csv
    row_10004_mar = df_merged[
        (df_merged["permno"] == 10004)
        & (df_merged["date"] == pd.Timestamp("2002-03-31"))
    ]
    assert not row_10004_mar.empty
    assert row_10004_mar["char1"].iloc[0] == 2  # from firm.csv
    assert row_10004_mar["ret"].iloc[0] == -5  # from ret.csv


def test_merge_crsp_returns_primary(base_config_dict_paths_as_filenames, tmp_path):
    """Test merge with crsp_returns as the primary firm base."""
    config_dict = base_config_dict_paths_as_filenames.copy()
    config_dict["sources"][1]["is_primary_firm_base"] = True  # crsp_returns is primary

    cfg_file = create_yaml_config_file(config_dict, tmp_path)
    df_merged, _ = aggregate_from_yaml(cfg_file)

    raw_ret_df = get_raw_test_df("ret.csv")
    assert len(df_merged) == len(
        raw_ret_df
    )  # All rows from crsp_returns should be kept

    # PERMNO 12345 IS in ret.csv, SHOULD be in the output
    row_12345_apr = df_merged[
        (df_merged["permno"] == 12345)
        & (df_merged["date"] == pd.Timestamp("2002-04-30"))
    ]
    assert not row_12345_apr.empty
    assert row_12345_apr["ret"].iloc[0] == 100000000
    assert pd.isna(
        row_12345_apr["char1"].iloc[0]
    )  # char1 from firm_chars should be NaN
    assert row_12345_apr["dp"].iloc[0] == 3  # Macro data should be present


def test_merge_error_multiple_firm_no_primary(
    base_config_dict_paths_as_filenames, tmp_path
):
    """Test error if multiple firm sources and no primary is specified."""
    config_dict = base_config_dict_paths_as_filenames.copy()
    # Ensure no firm source has is_primary_firm_base = True (default is False)
    config_dict["sources"][0].pop("is_primary_firm_base", None)
    config_dict["sources"][1].pop("is_primary_firm_base", None)

    cfg_file = create_yaml_config_file(config_dict, tmp_path)
    with pytest.raises(
        ValueError,
        match="Multiple firm-level sources are provided. Please specify exactly one as the primary base",
    ):
        aggregate_from_yaml(cfg_file)


def test_merge_error_too_many_primaries(base_config_dict_paths_as_filenames, tmp_path):
    """Test Pydantic validation error if multiple firm sources are marked as primary."""
    config_dict = base_config_dict_paths_as_filenames.copy()
    config_dict["sources"][0]["is_primary_firm_base"] = True
    config_dict["sources"][1]["is_primary_firm_base"] = (
        True  # Both firm sources marked primary
    )

    with pytest.raises(
        ValueError,
        match="Only one firm-level source can be marked as 'is_primary_firm_base: True'",
    ):
        AggregationConfig.model_validate(
            config_dict
        )  # Error from Pydantic model validation


def test_merge_single_firm_source_is_implicitly_primary(
    base_config_dict_paths_as_filenames, tmp_path
):
    """Test merging with a single firm source (implicitly primary)."""
    config_dict = {
        "sources": [
            base_config_dict_paths_as_filenames["sources"][0],  # Only firm_chars
            base_config_dict_paths_as_filenames["sources"][2],  # And macro_data
        ],
        "transformations": [],
        "output": {"format": "csv"},
    }
    cfg_file = create_yaml_config_file(config_dict, tmp_path)
    df_merged, _ = aggregate_from_yaml(cfg_file)

    raw_firm_df = get_raw_test_df("firm.csv")
    assert len(df_merged) == len(raw_firm_df)
    assert "char1" in df_merged.columns
    assert "dp" in df_merged.columns
    assert "ret" not in df_merged.columns  # crsp_returns was not included


def test_merge_only_macro_sources(base_config_dict_paths_as_filenames, tmp_path):
    """Test merging only macro sources."""
    macro2_data = pd.DataFrame(
        {
            "date": [20020131, 20020228, 20020531],  # 20020531 is an extra date
            "macro_var_x": [100, 200, 300],
        }
    )
    # Create macro2.csv in the location DATA_DIR points to (tests/data)
    # This requires conftest.py to have set DATA_DIR correctly.
    import core.settings  # To access the patched DATA_DIR

    macro2_file_path = Path(core.settings.DATA_DIR) / "macro2_temp.csv"
    macro2_data.to_csv(macro2_file_path, index=False)

    config_dict = {
        "sources": [
            base_config_dict_paths_as_filenames["sources"][
                2
            ],  # Original macro_data (macro.csv)
            {
                "name": "macro2",
                "connector": "local",
                "path": "macro2_temp.csv",  # Path relative to DATA_DIR
                "join_on": ["date"],
                "level": "macro",
            },
        ],
        "transformations": [],
        "output": {"format": "csv"},
    }
    cfg_file = create_yaml_config_file(config_dict, tmp_path)
    df_merged, _ = aggregate_from_yaml(cfg_file)

    raw_macro_df = get_raw_test_df("macro.csv")
    # Base is macro.csv (4 rows). macro2_temp.csv is left-joined.
    # The date 20020531 from macro2_temp won't create a new row.
    assert len(df_merged) == len(raw_macro_df)
    assert "dp" in df_merged.columns
    assert "macro_var_x" in df_merged.columns

    row_jan = df_merged[df_merged["date"] == pd.Timestamp("2002-01-31")]
    assert row_jan["dp"].iloc[0] == 1 and row_jan["macro_var_x"].iloc[0] == 100

    row_mar = df_merged[
        df_merged["date"] == pd.Timestamp("2002-03-31")
    ]  # macro_var_x not present for this date
    assert row_mar["dp"].iloc[0] == 2 and pd.isna(row_mar["macro_var_x"].iloc[0])

    macro2_file_path.unlink()  # Clean up the temporary file


def test_one_hot_encoding(base_config_dict_paths_as_filenames, tmp_path):
    config_dict = base_config_dict_paths_as_filenames.copy()
    config_dict["sources"][0]["is_primary_firm_base"] = True
    config_dict["transformations"] = [
        {
            "type": "one_hot",
            "column": "sic2",
            "prefix": "industry",
            "drop_original": True,
        }
    ]
    cfg_file = create_yaml_config_file(config_dict, tmp_path)
    df_merged, _ = aggregate_from_yaml(cfg_file)

    assert "sic2" not in df_merged.columns
    assert "industry_1" in df_merged.columns
    assert "industry_8" in df_merged.columns
    # Check a specific row from firm.csv: 20020131,10001,...,sic2=1
    row = df_merged[
        (df_merged["permno"] == 10001)
        & (df_merged["date"] == pd.Timestamp("2002-01-31"))
    ]
    assert row["industry_1"].iloc[0] == 1 and row["industry_64"].iloc[0] == 0


def test_grouped_fill_missing_median_specific_column(
    base_config_dict_paths_as_filenames, tmp_path
):
    config_dict = base_config_dict_paths_as_filenames.copy()
    # To introduce NaNs in 'ret' for a firm present in the primary base (firm_chars),
    # we need a (permno, date) in firm_chars that is NOT in ret.csv.
    # firm.csv has (10003, 20020131), (10003, 20020228), (10003, 20020331)
    # ret.csv has all these for 10003.
    # Let's modify firm.csv for the test: add a row for 10003, 20020430 (not in ret.csv for 10003)

    import core.settings  # To access DATA_DIR for creating temp file

    firm_custom_df = get_raw_test_df("firm.csv")
    new_row_data = {
        "date": pd.Timestamp("2002-04-30"),
        "permno": 10003,
        "char1": 50,
        "sic2": 1,
    }  # other chars can be NaN or defaults
    # Ensure all columns from original firm.csv are present
    for col in firm_custom_df.columns:
        if col not in new_row_data:
            new_row_data[col] = np.nan  # Or some default if appropriate

    new_row_df = pd.DataFrame([new_row_data])
    firm_custom_df = pd.concat([firm_custom_df, new_row_df], ignore_index=True)

    firm_custom_file_path = Path(core.settings.DATA_DIR) / "firm_custom_temp.csv"
    firm_custom_df.to_csv(
        firm_custom_file_path, index=False, date_format="%Y%m%d"
    )  # Save with YYYYMMDD

    config_dict["sources"][0]["path"] = "firm_custom_temp.csv"  # Use custom firm data
    config_dict["sources"][0]["is_primary_firm_base"] = True
    config_dict["transformations"] = [
        {
            "type": "grouped_fill_missing",
            "method": "median",
            "group_by_column": "date",
            "columns": ["ret"],
        }
    ]
    cfg_file = create_yaml_config_file(config_dict, tmp_path)
    df_merged, _ = aggregate_from_yaml(cfg_file)

    # For date 2002-04-30, returns from ret.csv (for permnos in firm_custom_temp.csv):
    # 10001: 1, 10002: 2, 10004: 3, 10005: 4. (12345 is dropped as not in firm_custom_temp)
    # For 10003, ret is NaN. Median of [1,2,3,4] is 2.5.
    row_10003_apr = df_merged[
        (df_merged["permno"] == 10003)
        & (df_merged["date"] == pd.Timestamp("2002-04-30"))
    ]
    assert not row_10003_apr.empty
    assert row_10003_apr["ret"].iloc[0] == 2.5
    assert (
        row_10003_apr["char1"].iloc[0] == 50
    )  # Ensure other columns are not affected if not specified

    firm_custom_file_path.unlink()  # Clean up


def test_grouped_fill_missing_all_numeric(
    base_config_dict_paths_as_filenames, tmp_path
):
    config_dict = base_config_dict_paths_as_filenames.copy()
    # Use crsp_returns as primary to get NaNs in char columns
    config_dict["sources"][1]["is_primary_firm_base"] = True
    config_dict["sources"][0]["is_primary_firm_base"] = False  # firm_chars is secondary

    config_dict["transformations"] = [
        {
            "type": "grouped_fill_missing",
            "method": "mean",
            "group_by_column": "date",  # No "columns" specified
        }
    ]
    cfg_file = create_yaml_config_file(config_dict, tmp_path)
    df_merged, _ = aggregate_from_yaml(cfg_file)

    # Check permno 12345, date 2002-04-30. char1 should have been NaN, then filled.
    row_12345_apr = df_merged[
        (df_merged["permno"] == 12345)
        & (df_merged["date"] == pd.Timestamp("2002-04-30"))
    ]
    assert not row_12345_apr.empty
    assert not pd.isna(row_12345_apr["char1"].iloc[0])  # Should be filled

    # Expected mean for char1 on 2002-04-30:
    # firm.csv data for 2002-04-30:
    # 10001: char1=3
    # 10002: char1=3
    # 10004: char1=3
    # 10005: char1=3
    # For 12345, char1 is NaN. Mean of [3,3,3,3] is 3.
    assert row_12345_apr["char1"].iloc[0] == 3.0
    assert (
        row_12345_apr["ret"].iloc[0] == 100000000
    )  # Ret should be untouched as it wasn't NaN


def test_lag_transformation(base_config_dict_paths_as_filenames, tmp_path):
    config_dict = base_config_dict_paths_as_filenames.copy()
    config_dict["sources"][0]["is_primary_firm_base"] = True
    config_dict["transformations"] = [
        {"type": "lag", "columns": ["char1", "ret"], "periods": 1}
    ]
    cfg_file = create_yaml_config_file(config_dict, tmp_path)
    df_merged, _ = aggregate_from_yaml(cfg_file)

    assert "char1_lag1" in df_merged.columns
    assert "ret_lag1" in df_merged.columns

    df_10001 = df_merged[df_merged["permno"] == 10001].sort_values("date")
    # Expected:
    # date       char1 ret | char1_lag1 ret_lag1
    # 20020131   1     1   | NaN        NaN
    # 20020228   2     2   | 1.0        1.0
    assert pd.isna(df_10001.iloc[0]["char1_lag1"])
    assert pd.isna(df_10001.iloc[0]["ret_lag1"])
    assert df_10001.iloc[1]["char1_lag1"] == 1.0
    assert df_10001.iloc[1]["ret_lag1"] == 1.0


def test_full_pipeline_from_yaml(base_config_dict_paths_as_filenames, tmp_path):
    """Test a more complete pipeline defined in YAML."""
    config_dict = base_config_dict_paths_as_filenames.copy()
    config_dict["sources"][0]["is_primary_firm_base"] = True  # firm_chars is primary
    config_dict["transformations"] = [
        {
            "type": "one_hot",
            "column": "sic2",
            "prefix": "sic2_ohe",
            "drop_original": True,
        },
        {
            "type": "grouped_fill_missing",
            "method": "median",
            "group_by_column": "date",
            "columns": ["ret"],
        },
        {"type": "lag", "columns": ["char1", "dp"], "periods": 1},  # dp from macro
    ]
    config_dict["output"]["format"] = "parquet"  # Test output parsing

    cfg_file = create_yaml_config_file(config_dict, tmp_path)
    df_final, cfg_obj = aggregate_from_yaml(cfg_file)

    assert isinstance(df_final, pd.DataFrame)
    assert isinstance(cfg_obj, AggregationConfig)
    assert not df_final.empty
    assert cfg_obj.output.format == "parquet"

    assert "sic2" not in df_final.columns
    assert "sic2_ohe_1" in df_final.columns
    assert "ret" in df_final.columns  # NaNs should be filled (if any were created)
    assert "char1_lag1" in df_final.columns
    assert "dp_lag1" in df_final.columns

    # Check dp_lag1 for a specific date
    # dp for 20020131 is 1. dp for 20020228 is 2.
    # So, for any firm on 20020228, dp_lag1 should be 1.
    df_feb = df_final[df_final["date"] == pd.Timestamp("2002-02-28")]
    assert not df_feb.empty
    assert (df_feb["dp_lag1"] == 1).all()

    df_jan = df_final[df_final["date"] == pd.Timestamp("2002-01-31")]
    assert not df_jan.empty
    assert df_jan["dp_lag1"].isna().all()  # First period, lag is NaN
