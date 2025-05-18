# tests/test_aggregator.py
import pytest
import pandas as pd
from pathlib import Path
import yaml
import warnings

from paperassetpricing.etl.schema import AggregationConfig  # type: ignore
from paperassetpricing.etl.aggregator import DataAggregator, aggregate_from_yaml  # type: ignore
import paperassetpricing.settings as settings  # type: ignore


# --- Helper Functions ---
def create_yaml_config_file(
    config_dict: dict, tmp_path: Path, filename="test_config.yaml"
) -> Path:
    config_file = tmp_path / filename
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)
    return config_file


def get_raw_test_df(
    filename: str, parse_dates_as_int: bool = True, date_col_name: str = "date"
) -> pd.DataFrame:
    raw_df_path = Path(settings.DATA_DIR) / filename
    df = pd.read_csv(raw_df_path, low_memory=False)

    actual_date_col = date_col_name.lower()
    original_date_col_casing_list = [
        col for col in df.columns if col.lower() == actual_date_col
    ]

    if original_date_col_casing_list:
        original_date_col_casing = original_date_col_casing_list[0]
        if parse_dates_as_int:
            try:
                df[original_date_col_casing] = pd.to_datetime(
                    df[original_date_col_casing].astype(str), format="%Y%m%d"
                )
            except ValueError:
                df[original_date_col_casing] = pd.to_datetime(
                    df[original_date_col_casing]
                )
        else:
            df[original_date_col_casing] = pd.to_datetime(df[original_date_col_casing])

        # Standardize to 'date' only if the original column was successfully found and parsed
        # And only if the target standardized name is 'date'
        if actual_date_col == "date":  # Ensure we are renaming to 'date'
            df = df.rename(columns={original_date_col_casing: "date"})

    if "PERMNO" in df.columns:
        df = df.rename(columns={"PERMNO": "permno"})
    df.columns = df.columns.str.lower()
    return df


@pytest.fixture
def base_config_dict_no_suffix():
    return {
        "sources": [
            {
                "name": "firm_chars",
                "connector": "local",
                "path": "firm.csv",
                "join_on": ["permno", "date"],
                "level": "firm",
            },
            {
                "name": "crsp_returns",
                "connector": "local",
                "path": "ret.csv",
                "join_on": ["permno", "date"],
                "level": "firm",
            },
            {
                "name": "macro_data",
                "connector": "local",
                "path": "macro.csv",
                "join_on": ["date"],
                "level": "macro",
            },
        ],
        "transformations": [],
        "output": {"format": "csv"},
    }


# --- Test Cases ---


def test_load_data_frames_auto_suffixing(base_config_dict_no_suffix):
    cfg_model = AggregationConfig.model_validate(base_config_dict_no_suffix)
    aggregator = DataAggregator(cfg_model).load()
    assert "firm_chars" in aggregator._frames
    df_firm = aggregator._frames["firm_chars"]
    assert "permno" in df_firm.columns and "date" in df_firm.columns
    assert "char1_firm" in df_firm.columns
    assert "sic2_firm" in df_firm.columns
    assert pd.api.types.is_datetime64_any_dtype(df_firm["date"])
    assert (df_firm["date"] == df_firm["date"].dt.normalize()).all()
    df_ret = aggregator._frames["crsp_returns"]
    assert "ret_firm" in df_ret.columns
    df_macro = aggregator._frames["macro_data"]
    assert "dp_macro" in df_macro.columns
    assert "ep_macro" in df_macro.columns


def test_error_duplicate_original_col_same_level(tmp_path):
    firm1_data = pd.DataFrame({"permno": [1], "date": [20230131], "feature_a": [10]})
    firm1_file = Path(settings.DATA_DIR) / "firm1_dup_temp.csv"
    firm1_data.to_csv(firm1_file, index=False)
    firm2_data = pd.DataFrame({"permno": [2], "date": [20230131], "feature_a": [20]})
    firm2_file = Path(settings.DATA_DIR) / "firm2_dup_temp.csv"
    firm2_data.to_csv(firm2_file, index=False)
    config_dict = {
        "sources": [
            {
                "name": "f1",
                "connector": "local",
                "path": "firm1_dup_temp.csv",
                "join_on": ["permno", "date"],
                "level": "firm",
            },
            {
                "name": "f2",
                "connector": "local",
                "path": "firm2_dup_temp.csv",
                "join_on": ["permno", "date"],
                "level": "firm",
            },
        ]
    }
    cfg_model = AggregationConfig.model_validate(config_dict)
    aggregator = DataAggregator(cfg_model)
    with pytest.raises(
        ValueError,
        match=r"Duplicate original column name 'feature_a' found in multiple sources of level 'firm'",
    ):
        aggregator.load()
    firm1_file.unlink()
    firm2_file.unlink()


def test_no_error_duplicate_original_col_different_level(tmp_path):
    firm_data = pd.DataFrame({"permno": [1], "date": [20230131], "ep": [0.5]})
    (Path(settings.DATA_DIR) / "firm_ep_temp.csv").write_text(
        firm_data.to_csv(index=False)
    )
    macro_data = pd.DataFrame({"date": [20230131], "ep": [0.05]})
    (Path(settings.DATA_DIR) / "macro_ep_temp.csv").write_text(
        macro_data.to_csv(index=False)
    )
    config_dict = {
        "sources": [
            {
                "name": "firm_source_with_ep",
                "connector": "local",
                "path": "firm_ep_temp.csv",
                "join_on": ["permno", "date"],
                "level": "firm",
            },
            {
                "name": "macro_source_with_ep",
                "connector": "local",
                "path": "macro_ep_temp.csv",
                "join_on": ["date"],
                "level": "macro",
            },
        ]
    }
    cfg_model = AggregationConfig.model_validate(config_dict)
    aggregator = DataAggregator(cfg_model)
    aggregator.load()
    assert "ep_firm" in aggregator._frames["firm_source_with_ep"].columns
    assert "ep_macro" in aggregator._frames["macro_source_with_ep"].columns
    (Path(settings.DATA_DIR) / "firm_ep_temp.csv").unlink()
    (Path(settings.DATA_DIR) / "macro_ep_temp.csv").unlink()


def test_date_handling_monthly_alignment(tmp_path):
    source1_data = pd.DataFrame({"date": [20230315], "value1": [10]})
    source1_file = Path(settings.DATA_DIR) / "source1_temp.csv"
    source1_data.to_csv(source1_file, index=False)
    source2_data = pd.DataFrame({"date": [20230328], "value2": [20]})
    source2_file = Path(settings.DATA_DIR) / "source2_temp.csv"
    source2_data.to_csv(source2_file, index=False)
    config_dict = {
        "sources": [
            {
                "name": "src1",
                "connector": "local",
                "path": "source1_temp.csv",
                "join_on": ["date"],
                "level": "macro",
                "date_handling": {"frequency": "monthly"},
            },
            {
                "name": "src2",
                "connector": "local",
                "path": "source2_temp.csv",
                "join_on": ["date"],
                "level": "macro",
                "date_handling": {"frequency": "monthly"},
            },
        ],
        "transformations": [],
    }
    cfg_file = create_yaml_config_file(config_dict, tmp_path)
    cfg_model = AggregationConfig.model_validate(config_dict)
    aggregator = DataAggregator(cfg_model).load()
    expected_month_end = pd.Timestamp("2023-03-31").normalize()
    # 'date' is a join key, so it's NOT suffixed.
    assert aggregator._frames["src1"]["date"].iloc[0] == expected_month_end
    assert aggregator._frames["src2"]["date"].iloc[0] == expected_month_end
    df_merged, _ = aggregate_from_yaml(cfg_file)
    assert len(df_merged) == 1
    assert df_merged["date"].iloc[0] == expected_month_end
    assert df_merged["value1_macro"].iloc[0] == 10  # value1 gets _macro suffix
    assert df_merged["value2_macro"].iloc[0] == 20  # value2 gets _macro suffix
    source1_file.unlink()
    source2_file.unlink()


def test_clean_numeric_transformation(tmp_path):
    mixed_data = pd.DataFrame({"id": [1], "date": [20230131], "mixed_val": ["B"]})
    mixed_file = Path(settings.DATA_DIR) / "mixed_data_temp.csv"
    mixed_data.to_csv(mixed_file, index=False)
    config_dict = {
        "sources": [
            {
                "name": "mixed_src",
                "connector": "local",
                "path": "mixed_data_temp.csv",
                "join_on": ["id", "date"],
                "level": "firm",
                "is_primary_firm_base": True,
            }
        ],
        "transformations": [
            {"type": "clean_numeric", "columns": ["mixed_val"], "action": "to_nan"}
        ],
    }
    cfg_file = create_yaml_config_file(config_dict, tmp_path)
    df_transformed, _ = aggregate_from_yaml(cfg_file)
    assert pd.isna(df_transformed["mixed_val_firm"].iloc[0])
    assert pd.api.types.is_float_dtype(df_transformed["mixed_val_firm"])
    mixed_file.unlink()


def test_grouped_fill_missing_specific_col_and_thresholds(tmp_path, capsys):
    firm_data = pd.DataFrame(
        {
            "permno": [101, 101, 102, 102, 103, 103, 104, 104, 105, 105],
            "date": [
                20230131,
                20230228,
                20230131,
                20230228,
                20230131,
                20230228,
                20230131,
                20230228,
                20230131,
                20230228,
            ],
            "char_val": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        }
    )
    firm_file = Path(settings.DATA_DIR) / "firm_gfm_temp.csv"
    firm_data.to_csv(firm_file, index=False)
    ret_data = pd.DataFrame(
        {
            "permno": [101, 101, 102, 103, 104, 105],
            "date": [20230131, 20230228, 20230131, 20230228, 20230228, 20230228],
            "ret": [0.1, 0.11, 0.2, 0.3, 0.4, 0.5],
        }
    )
    ret_file = Path(settings.DATA_DIR) / "ret_gfm_temp.csv"
    ret_data.to_csv(ret_file, index=False)
    config_base = {
        "sources": [
            {
                "name": "firm_data",
                "connector": "local",
                "path": "firm_gfm_temp.csv",
                "join_on": ["permno", "date"],
                "level": "firm",
                "is_primary_firm_base": True,
                "date_handling": {"frequency": "monthly"},
            },
            {
                "name": "return_data",
                "connector": "local",
                "path": "ret_gfm_temp.csv",
                "join_on": ["permno", "date"],
                "level": "firm",
                "date_handling": {"frequency": "monthly"},
            },
        ],
        "transformations": [],
    }
    config_fill = config_base.copy()
    config_fill["transformations"] = [
        {
            "type": "grouped_fill_missing",
            "method": "median",
            "group_by_column": "date",
            "columns": ["ret"],
            "missing_threshold_warning": 0.7,
            "missing_threshold_error": 0.9,
        }
    ]
    cfg_file = create_yaml_config_file(config_fill, tmp_path, "cfg_fill.yaml")
    df, _ = aggregate_from_yaml(cfg_file)
    jan_data = df[df["date"] == pd.Timestamp("2023-01-31")]
    assert not jan_data[jan_data["permno"] == 101].empty
    assert pytest.approx(jan_data[jan_data["permno"] == 101]["ret_firm"].iloc[0]) == 0.1
    assert not jan_data[jan_data["permno"] == 102].empty
    assert pytest.approx(jan_data[jan_data["permno"] == 102]["ret_firm"].iloc[0]) == 0.2
    assert not jan_data[jan_data["permno"] == 103].empty
    assert (
        pytest.approx(jan_data[jan_data["permno"] == 103]["ret_firm"].iloc[0]) == 0.15
    )

    config_warn = config_base.copy()
    config_warn["transformations"] = [
        {
            "type": "grouped_fill_missing",
            "method": "median",
            "group_by_column": "date",
            "columns": ["ret"],
            "missing_threshold_warning": 0.55,
            "missing_threshold_error": 0.9,
        }
    ]
    cfg_file_w = create_yaml_config_file(config_warn, tmp_path, "cfg_warn.yaml")
    with warnings.catch_warnings(record=True) as w_list:
        warnings.simplefilter("always")
        aggregate_from_yaml(cfg_file_w)
        captured_stdout = capsys.readouterr().out
        assert (
            "Warning (GroupedFillMissing): Column 'ret_firm' in group 'date=2023-01-31' has 60.00% missing"
            in captured_stdout
        )
        assert any(
            "Warning (GroupedFillMissing): Column 'ret_firm' in group 'date=2023-01-31' has 60.00% missing"
            in str(warn.message)
            for warn in w_list
        )

    config_err = config_base.copy()
    config_err["transformations"] = [
        {
            "type": "grouped_fill_missing",
            "method": "median",
            "group_by_column": "date",
            "columns": ["ret"],
            "missing_threshold_warning": 0.1,
            "missing_threshold_error": 0.55,
        }
    ]
    cfg_file_e = create_yaml_config_file(config_err, tmp_path, "cfg_err.yaml")
    with pytest.raises(
        ValueError,
        match=r"Error \(GroupedFillMissing\): Column 'ret_firm' in group 'date=2023-01-31' has 60.00% missing",
    ):
        aggregate_from_yaml(cfg_file_e)
    firm_file.unlink()
    ret_file.unlink()


def test_expand_cartesian_with_infer_suffix(base_config_dict_no_suffix, tmp_path):
    config_dict = base_config_dict_no_suffix.copy()
    config_dict["sources"][0]["is_primary_firm_base"] = True
    for src in config_dict["sources"]:
        src["date_handling"] = {"frequency": "monthly"}
    config_dict["transformations"] = [
        {
            "type": "expand_cartesian",
            "infer_suffix": True,
            "macro_columns": ["dp", "ep"],
            "firm_columns": ["char1", "ret"],
        }
    ]
    cfg_file = create_yaml_config_file(config_dict, tmp_path)
    df_transformed, _ = aggregate_from_yaml(cfg_file)
    assert "dp_macro_x_char1_firm" in df_transformed.columns
    assert "dp_macro_x_ret_firm" in df_transformed.columns
    assert "ep_macro_x_char1_firm" in df_transformed.columns
    assert "ep_macro_x_ret_firm" in df_transformed.columns
    row = df_transformed[
        (df_transformed["permno"] == 10001)
        & (df_transformed["date"] == pd.Timestamp("2002-01-31"))
    ]
    assert not row.empty
    # Print values for debugging the specific row if an assert fails
    # print(f"DEBUG: For P10001 Jan2002: dp_macro={row['dp_macro'].iloc[0]}, ep_macro={row['ep_macro'].iloc[0]}, char1_firm={row['char1_firm'].iloc[0]}, ret_firm={row['ret_firm'].iloc[0]}")
    assert pytest.approx(row["dp_macro_x_char1_firm"].iloc[0]) == 1.0 * 1.0
    assert (
        pytest.approx(row["ep_macro_x_ret_firm"].iloc[0]) == 2.0 * 1.0
    )  # ep_macro=2.0, ret_firm=1


# ... (Other tests like full_pipeline, merge tests, etc. would need similar review for column names) ...


def test_full_pipeline_auto_suffix(base_config_dict_no_suffix, tmp_path):
    config_dict = base_config_dict_no_suffix.copy()
    config_dict["sources"][0]["is_primary_firm_base"] = True
    for src in config_dict["sources"]:
        src["date_handling"] = {"frequency": "monthly"}
    config_dict["transformations"] = [
        {"type": "clean_numeric", "columns": ["ret"]},
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
        {"type": "lag", "columns": ["char1", "dp"], "periods": 1},
        {
            "type": "expand_cartesian",
            "infer_suffix": True,
            "macro_columns": ["dp"],
            "firm_columns": ["char1"],
        },
    ]
    config_dict["output"]["format"] = "parquet"
    cfg_file = create_yaml_config_file(config_dict, tmp_path)
    df_final, cfg_obj = aggregate_from_yaml(cfg_file)

    assert "sic2_firm" not in df_final.columns
    assert "sic2_ohe_1" in df_final.columns
    assert "ret_firm" in df_final.columns
    assert "char1_firm_lag1" in df_final.columns
    assert "dp_macro_lag1" in df_final.columns
    assert "dp_macro_x_char1_firm" in df_final.columns
    df_feb = df_final[df_final["date"] == pd.Timestamp("2002-02-28")]
    assert not df_feb.empty
    assert (df_feb["dp_macro_lag1"] == 1.0).all()


# --- Original merge tests adapted for auto suffixing and monthly alignment ---
def test_merge_firm_chars_primary_auto_suffix(base_config_dict_no_suffix, tmp_path):
    config_dict = base_config_dict_no_suffix.copy()
    config_dict["sources"][0]["is_primary_firm_base"] = True
    for src in config_dict["sources"]:
        src["date_handling"] = {"frequency": "monthly"}
    cfg_file = create_yaml_config_file(config_dict, tmp_path)
    df_merged, _ = aggregate_from_yaml(cfg_file)
    raw_firm_df = get_raw_test_df("firm.csv")
    # Convert to datetime first
    raw_firm_df["date"] = pd.to_datetime(raw_firm_df["date"])
    # Then convert to period and back to timestamp
    raw_firm_df["date"] = (
        raw_firm_df["date"].dt.to_period("M").dt.to_timestamp(how="end")
    )
    expected_len = len(raw_firm_df.drop_duplicates(subset=["permno", "date"]))
    assert len(df_merged) == expected_len
    assert 12345 not in df_merged["permno"].unique()
    row_10001_jan = df_merged[
        (df_merged["permno"] == 10001)
        & (df_merged["date"] == pd.Timestamp("2002-01-31"))
    ]
    assert not row_10001_jan.empty
    assert row_10001_jan["char1_firm"].iloc[0] == 1
    assert row_10001_jan["ret_firm"].iloc[0] == 1
    assert row_10001_jan["dp_macro"].iloc[0] == 1
    row_10004_mar = df_merged[
        (df_merged["permno"] == 10004)
        & (df_merged["date"] == pd.Timestamp("2002-03-31"))
    ]
    assert not row_10004_mar.empty
    assert row_10004_mar["char1_firm"].iloc[0] == 2
    assert row_10004_mar["ret_firm"].iloc[0] == -5
