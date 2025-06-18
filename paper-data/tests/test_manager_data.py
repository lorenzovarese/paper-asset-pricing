import pytest
from unittest.mock import patch
import polars as pl
from polars.testing import assert_frame_equal
from pydantic import ValidationError

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
def sample_firm_df():
    """A sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "date": ["2020-01-31", "2020-01-31", "2020-02-29"],
            "permno": [1, 2, 1],
            "volume": [100.0, None, 150.0],
            "category": ["A", "B", "A"],
        }
    ).with_columns(pl.col("date").str.to_date("%Y-%m-%d"))


# --- Tests for DataManager ---


@patch("paper_data.manager.CSVLoader")
def test_manager_ingestion(mock_csv_loader, sample_firm_df, mock_project_root):
    """Tests that the manager correctly calls the ingestion connector."""
    mock_csv_loader.return_value.get_data.return_value = sample_firm_df

    config_dict = {
        "ingestion": [
            {
                "name": "firm_data",
                "format": "csv",
                "path": "firm.csv",
                "date_column": {"date": "%Y-%m-%d"},
                "id_column": "permno",
            }
        ],
        "export": [
            {
                "dataset_name": "firm_data",
                "output_filename_base": "final",
                "format": "parquet",
            }
        ],
    }
    config = DataConfig.model_validate(config_dict)
    manager = DataManager(config)

    manager._project_root = mock_project_root
    manager._ingest_data()

    mock_csv_loader.assert_called_once()
    assert "firm_data" in manager.datasets
    assert_frame_equal(manager.datasets["firm_data"], sample_firm_df)
    assert manager._ingestion_metadata["firm_data"]["date_column"] == "date"


@pytest.mark.parametrize(
    "operation_config, mock_function_path",
    [
        # Test Case 1: ScaleConfig
        (
            {
                "operation": "scale_to_range",
                "dataset": "firm_data",
                "cols_to_scale": ["volume"],
                "range": {"min": -1, "max": 1},
                "output_name": "scaled_data",
            },
            "paper_data.manager.scale_to_range",
        ),
        # Test Case 2: MergeConfig
        (
            {
                "operation": "merge",
                "left_dataset": "firm_data",
                "right_dataset": "firm_data",
                "on": ["permno"],
                "how": "left",
                "output_name": "merged_data",
            },
            "paper_data.manager.merge_datasets",
        ),
        # Test Case 3: LagConfig
        (
            {
                "operation": "lag",
                "dataset": "firm_data",
                "periods": 1,
                "columns_to_lag": {
                    "method": "all_except",
                    "columns": ["date", "permno"],
                },
                "output_name": "lagged_data",
            },
            "paper_data.manager.lag_columns",
        ),
        # Test Case 4: DummyConfig
        (
            {
                "operation": "dummy_generation",
                "dataset": "firm_data",
                "column_to_dummy": "category",
                "output_name": "dummied_data",
            },
            "paper_data.manager.create_dummies",
        ),
        # Test Case 5: InteractionConfig (eager)
        (
            {
                "operation": "create_macro_interactions",
                "dataset": "firm_data",
                "macro_columns": ["volume"],
                "firm_columns": ["permno"],
                "output_name": "interacted_data",
            },
            "paper_data.manager.create_macro_firm_interactions",
        ),
        # Test Case 6: InteractionConfig (lazy)
        (
            {
                "operation": "create_macro_interactions",
                "dataset": "firm_data",
                "macro_columns": ["volume"],
                "firm_columns": ["permno"],
                "use_lazy_engine": True,
                "output_name": "lazy_interacted_data",
            },
            "paper_data.manager.create_macro_firm_interactions_lazy",
        ),
    ],
)
def test_wrangling_operations(operation_config, mock_function_path, sample_firm_df):
    """Tests that the manager correctly calls each type of wrangling function."""
    full_config_dict = {
        "ingestion": [
            {
                "name": "firm_data",
                "format": "csv",
                "path": "f.csv",
                "date_column": {"date": "d"},
                "id_column": "permno",
            }
        ],
        "wrangling_pipeline": [operation_config],
        "export": [
            {
                "dataset_name": operation_config["output_name"],
                "output_filename_base": "out",
                "format": "parquet",
            }
        ],
    }
    config = DataConfig.model_validate(full_config_dict)
    manager = DataManager(config)

    manager.datasets["firm_data"] = sample_firm_df
    manager._ingestion_metadata["firm_data"] = {
        "date_column": "date",
        "id_column": "permno",
    }

    with patch(mock_function_path) as mock_func:
        if operation_config.get("use_lazy_engine"):
            mock_func.return_value = pl.LazyFrame()
        else:
            mock_func.return_value = pl.DataFrame()

        manager._wrangle_data()

    mock_func.assert_called_once()

    pos_args, _ = mock_func.call_args
    op_type = operation_config["operation"]

    # Assert based on the specific function that was called
    if op_type == "scale_to_range":
        assert pos_args[1] == ["volume"]  # cols_to_scale
        assert pos_args[2] == "date"  # date_col
        assert pos_args[3] == -1.0  # min
        assert pos_args[4] == 1.0  # max
    elif op_type == "merge":
        # The first two args are DataFrames, we can check their type
        assert isinstance(pos_args[0], pl.DataFrame)
        assert isinstance(pos_args[1], pl.DataFrame)
        assert pos_args[2] == ["permno"]  # on_cols
        assert pos_args[3] == "left"  # how
    elif op_type == "lag":
        assert pos_args[1] == "date"
        assert pos_args[2] == "permno"
        assert pos_args[3] == ["volume", "category"]  # cols_to_lag
        assert pos_args[4] == 1  # periods
    elif op_type == "dummy_generation":
        assert pos_args[1] == "category"
        assert pos_args[2] is False  # drop_original_col
    elif op_type == "create_macro_interactions":
        if operation_config.get("use_lazy_engine"):
            assert isinstance(pos_args[0], pl.LazyFrame)
        else:
            assert isinstance(pos_args[0], pl.DataFrame)
        assert pos_args[1] == ["volume"]  # macro_columns
        assert pos_args[2] == ["permno"]  # firm_columns
        assert pos_args[3] is False  # drop_macro_columns

    # Check that the output dataset was created
    output_name = operation_config["output_name"]
    if operation_config.get("use_lazy_engine"):
        assert output_name in manager.lazy_datasets
    else:
        assert output_name in manager.datasets


@patch("paper_data.manager.run_custom_script")
def test_manager_delegates_to_run_script_operation(
    mock_run_custom_script, mock_project_root, sample_firm_df
):
    """
    Tests that the DataManager correctly calls (delegates to) the
    run_custom_script function when it encounters the operation in the pipeline.
    """
    # Arrange
    mock_run_custom_script.return_value = pl.DataFrame({"new_col": [1]})

    config_dict = {
        "ingestion": [
            {
                "name": "firm_data",
                "format": "csv",
                "path": "f.csv",
                "date_column": {"date": "d"},
            }
        ],
        "wrangling_pipeline": [
            {
                "operation": "run_script",
                "dataset": "firm_data",
                "script": "my_script.py",
                "function_name": "my_func",
                "output_name": "scripted_data",
            }
        ],
        "export": [
            {
                "dataset_name": "scripted_data",
                "output_filename_base": "out",
                "format": "parquet",
            }
        ],
    }
    config = DataConfig.model_validate(config_dict)
    manager = DataManager(config)
    manager._project_root = mock_project_root
    manager.datasets["firm_data"] = sample_firm_df

    # Act
    manager._wrangle_data()

    # Assert
    mock_run_custom_script.assert_called_once_with(
        df=sample_firm_df,
        project_root=mock_project_root,
        script="my_script.py",
        function_name="my_func",
    )

    # Verify the manager stored the result from the mocked function
    assert "scripted_data" in manager.datasets
    assert_frame_equal(
        manager.datasets["scripted_data"], pl.DataFrame({"new_col": [1]})
    )


@patch("polars.DataFrame.write_parquet")
def test_manager_export(mock_write_parquet, mock_project_root):
    """Tests that the manager correctly calls the export function."""
    config_dict = {
        "ingestion": [
            {
                "name": "imputed_data",
                "format": "csv",
                "path": "f.csv",
                "date_column": {"d": "f"},
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
    config = DataConfig.model_validate(config_dict)
    manager = DataManager(config)

    manager.datasets["imputed_data"] = pl.DataFrame()
    manager._project_root = mock_project_root

    manager._export_data()

    mock_write_parquet.assert_called_once()
    expected_path = mock_project_root / "data" / "processed" / "final_panel.parquet"
    assert mock_write_parquet.call_args[0][0] == expected_path


@patch("paper_data.manager.DataManager._ingest_data")
@patch("paper_data.manager.DataManager._wrangle_data")
@patch("paper_data.manager.DataManager._export_data")
def test_manager_run_orchestrates_calls(
    mock_export, mock_wrangle, mock_ingest, mock_project_root
):
    """Tests that the main `run` method calls all pipeline steps in order."""
    config_dict = {"ingestion": [], "export": []}
    config = DataConfig.model_validate(config_dict)
    manager = DataManager(config)

    manager.run(project_root=mock_project_root)

    mock_ingest.assert_called_once()
    mock_wrangle.assert_called_once()
    mock_export.assert_called_once()


def test_manager_fails_on_unsupported_operation():
    """Tests that Pydantic validation catches an unknown wrangling operation."""
    invalid_config_dict = {
        "ingestion": [
            {"name": "d", "format": "csv", "path": "p", "date_column": {"d": "f"}}
        ],
        "wrangling_pipeline": [{"operation": "fly_to_the_moon", "output_name": "m"}],
        "export": [],
    }
    with pytest.raises(
        ValidationError,
        match="Input tag 'fly_to_the_moon' found using 'operation' does not match any of the expected tags",
    ):
        DataConfig.model_validate(invalid_config_dict)
