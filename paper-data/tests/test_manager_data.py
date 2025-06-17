from pydantic import ValidationError
import pytest
from unittest.mock import patch
import polars as pl
from polars.testing import assert_frame_equal

from paper_data.manager import DataManager  # type: ignore
from paper_data.config_parser import DataConfig  # type: ignore

# --- Fixtures ---


@pytest.fixture
def mock_project_root(tmp_path):
    """Creates a mock project structure."""
    proj_dir = tmp_path / "TestProject"
    (proj_dir / "data" / "raw").mkdir(parents=True)
    (proj_dir / "data" / "processed").mkdir(parents=True)
    return proj_dir


@pytest.fixture
def sample_config_dict():
    """A sample config for testing the manager's logic."""
    return {
        "ingestion": [
            {
                "name": "firm_data",
                "format": "csv",
                "path": "firm.csv",
                "date_column": {"date": "%Y-%m-%d"},
                "id_column": "permno",
            }
        ],
        "wrangling_pipeline": [
            {
                "operation": "monthly_imputation",
                "dataset": "firm_data",
                "numeric_columns": ["volume"],
                "output_name": "imputed_data",
            }
        ],
        "export": [
            {
                "dataset_name": "imputed_data",
                "output_filename_base": "final_panel",
                "format": "parquet",
            }
        ],
    }


@pytest.fixture
def sample_firm_df():
    """A sample DataFrame to be returned by the mocked CSVLoader."""
    return pl.DataFrame(
        {
            "date": ["2020-01-31", "2020-01-31", "2020-02-29"],
            "permno": [1, 2, 1],
            "volume": [100.0, None, 150.0],
        }
    ).with_columns(pl.col("date").str.to_date("%Y-%m-%d"))


# --- Tests for DataManager ---


@patch("paper_data.manager.CSVLoader")
def test_manager_ingestion(
    mock_csv_loader, sample_config_dict, sample_firm_df, mock_project_root
):
    """Tests that the manager correctly calls the ingestion connector."""
    mock_csv_loader.return_value.get_data.return_value = sample_firm_df

    config = DataConfig.model_validate(sample_config_dict)
    manager = DataManager(config)

    manager._project_root = mock_project_root
    manager._ingest_data()

    mock_csv_loader.assert_called_once()
    assert "firm_data" in manager.datasets
    assert_frame_equal(manager.datasets["firm_data"], sample_firm_df)
    assert manager._ingestion_metadata["firm_data"]["date_column"] == "date"


@patch("paper_data.manager.impute_monthly")
def test_manager_wrangling(mock_impute, sample_config_dict, sample_firm_df):
    """Tests that the manager correctly calls a wrangling function."""
    imputed_df = sample_firm_df.with_columns(pl.col("volume").fill_null(100.0))
    mock_impute.return_value = imputed_df

    config = DataConfig.model_validate(sample_config_dict)
    manager = DataManager(config)

    manager.datasets["firm_data"] = sample_firm_df
    manager._ingestion_metadata["firm_data"] = {
        "date_column": "date",
        "id_column": "permno",
    }

    manager._wrangle_data()

    mock_impute.assert_called_once()
    assert "imputed_data" in manager.datasets
    assert_frame_equal(manager.datasets["imputed_data"], imputed_df)


@patch("polars.DataFrame.write_parquet")
def test_manager_export(
    mock_write_parquet, sample_config_dict, sample_firm_df, mock_project_root
):
    """Tests that the manager correctly calls the export function."""
    config = DataConfig.model_validate(sample_config_dict)
    manager = DataManager(config)

    manager.datasets["imputed_data"] = sample_firm_df
    manager._project_root = mock_project_root

    manager._export_data()

    mock_write_parquet.assert_called_once()
    expected_path = mock_project_root / "data" / "processed" / "final_panel.parquet"
    assert mock_write_parquet.call_args[0][0] == expected_path


@patch("paper_data.manager.DataManager._ingest_data")
@patch("paper_data.manager.DataManager._wrangle_data")
@patch("paper_data.manager.DataManager._export_data")
def test_manager_run_orchestrates_calls(
    mock_export, mock_wrangle, mock_ingest, sample_config_dict, mock_project_root
):
    """Tests that the main `run` method calls all pipeline steps in order."""
    config = DataConfig.model_validate(sample_config_dict)
    manager = DataManager(config)

    manager.run(project_root=mock_project_root)

    mock_ingest.assert_called_once()
    mock_wrangle.assert_called_once()
    mock_export.assert_called_once()


def test_manager_fails_on_unsupported_operation(sample_config_dict):
    """Tests that the manager raises an error for an unknown wrangling operation."""
    sample_config_dict["wrangling_pipeline"].append(
        {
            "operation": "fly_to_the_moon",
            "dataset": "firm_data",
            "output_name": "moon_data",
        }
    )

    # This now correctly tests that Pydantic's discriminator catches the error
    with pytest.raises(
        ValidationError,
        match="Input tag 'fly_to_the_moon' found using 'operation' does not match any of the expected tags",
    ):
        DataConfig.model_validate(sample_config_dict)
