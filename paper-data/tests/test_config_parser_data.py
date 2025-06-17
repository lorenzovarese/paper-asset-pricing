import pytest
from pydantic import ValidationError
import yaml

from paper_data.config_parser import load_config, DataConfig, MergeConfig, LagConfig  # type: ignore

# --- Fixtures ---


@pytest.fixture
def valid_config_dict():
    """A dictionary representing a minimal, valid configuration."""
    return {
        "ingestion": [
            {
                "name": "firm_data",
                "format": "csv",
                "path": "firm.csv",
                "date_column": {"date": "%Y%m%d"},
                "id_column": "permno",
            },
            {
                "name": "macro_data",
                "format": "csv",
                "path": "macro.csv",
                "date_column": {"date": "%Y%m%d"},
            },
        ],
        "wrangling_pipeline": [
            {
                "operation": "merge",
                "left_dataset": "firm_data",
                "right_dataset": "macro_data",
                "on": ["date"],
                "how": "left",
                "output_name": "merged_data",
            },
            {
                "operation": "lag",
                "dataset": "merged_data",
                "periods": 1,
                "columns_to_lag": {
                    "method": "all_except",
                    "columns": ["date", "permno"],
                },
                "output_name": "final_data",
            },
        ],
        "export": [
            {
                "dataset_name": "final_data",
                "output_filename_base": "final_panel",
                "format": "parquet",
                "partition_by": "year",
            }
        ],
    }


@pytest.fixture
def config_file(tmp_path, valid_config_dict):
    """Creates a temporary valid config file and returns its path."""
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(valid_config_dict, f)
    return p


# --- Tests for load_config function ---


def test_load_config_success(config_file):
    """Tests that a valid config file is loaded without errors."""
    config = load_config(config_file)
    assert isinstance(config, DataConfig)
    assert len(config.ingestion) == 2
    assert len(config.wrangling_pipeline) == 2
    assert isinstance(config.wrangling_pipeline[0], MergeConfig)
    assert isinstance(config.wrangling_pipeline[1], LagConfig)
    # assert config.export[0].partition_by.value == "year"


def test_load_config_file_not_found():
    """Tests that a FileNotFoundError is raised for a non-existent file."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config("non_existent_file.yaml")


def test_load_config_empty_file(tmp_path):
    """Tests that a ValueError is raised for an empty config file."""
    p = tmp_path / "empty.yaml"
    p.touch()
    with pytest.raises(ValueError, match="is empty or invalid"):
        load_config(p)


def test_load_config_invalid_yaml(tmp_path):
    """Tests that a YAMLError is raised for a malformed file."""
    p = tmp_path / "invalid.yaml"
    p.write_text("key: [unclosed")
    with pytest.raises(yaml.YAMLError, match="Error parsing YAML file"):
        load_config(p)


# --- Tests for Pydantic Model Validation ---


def test_config_validation_fails_on_missing_ingestion(valid_config_dict):
    """Tests that validation fails if the 'ingestion' key is missing."""
    del valid_config_dict["ingestion"]
    with pytest.raises(ValidationError):
        DataConfig.model_validate(valid_config_dict)


def test_config_validation_fails_on_unknown_dataset(valid_config_dict):
    """Tests that validation fails if a wrangling step uses an undefined dataset."""
    valid_config_dict["wrangling_pipeline"][0]["left_dataset"] = "unknown_dataset"
    # MODIFIED: Call model_validate directly, not load_config
    with pytest.raises(
        ValueError, match="left_dataset 'unknown_dataset' is not defined"
    ):
        DataConfig.model_validate(valid_config_dict)


def test_config_validation_fails_on_exporting_unknown_dataset(valid_config_dict):
    """Tests that validation fails if an export step uses an undefined dataset."""
    valid_config_dict["export"][0]["dataset_name"] = "unknown_final_data"
    # MODIFIED: Call model_validate directly, not load_config
    with pytest.raises(ValueError, match="dataset 'unknown_final_data' is not defined"):
        DataConfig.model_validate(valid_config_dict)


def test_lag_config_validation_fails_on_restore_names(valid_config_dict):
    """Tests that LagConfig validation fails if restore_names=True but drop_original=False."""
    lag_op = valid_config_dict["wrangling_pipeline"][1]
    lag_op["restore_names"] = True
    lag_op["drop_original_cols_after_lag"] = False
    # MODIFIED: Call model_validate directly, not load_config
    with pytest.raises(
        ValueError,
        match="If 'restore_names' is true, 'drop_original_cols_after_lag' must also be true",
    ):
        DataConfig.model_validate(valid_config_dict)


def test_ingestion_date_column_validation():
    """Tests that the date_column must be a single-entry dictionary."""
    with pytest.raises(
        ValidationError, match="must contain exactly one key-value pair"
    ):
        DataConfig.model_validate(
            {
                "ingestion": [
                    {
                        "name": "test",
                        "format": "csv",
                        "path": "p",
                        "date_column": {"d1": "f1", "d2": "f2"},  # Invalid
                    }
                ],
                "export": [],
            }
        )
