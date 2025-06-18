import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import polars as pl
from datetime import datetime
from polars.testing import assert_frame_equal

from paper_model.manager import ModelManager  # type: ignore
from paper_model.config_parser import ModelsConfig, FeatureSelectionConfig  # type: ignore

# --- Fixtures ---


@pytest.fixture
def mock_project_root(tmp_path):
    """Creates a mock project structure with dummy data files."""
    proj_dir = tmp_path / "TestProject"
    data_dir = proj_dir / "data" / "processed"
    data_dir.mkdir(parents=True)

    dummy_df = pl.DataFrame(
        {
            "date": pl.date_range(
                datetime(2020, 1, 1), datetime(2020, 12, 31), "1mo", eager=True
            ),
            "id": range(12),
            "feature": range(12),
            "ret": np.random.rand(12),
        }
    )
    dummy_df.write_parquet(data_dir / "test_data_2020.parquet")
    dummy_df.with_columns(pl.col("date").dt.offset_by("1y")).write_parquet(
        data_dir / "test_data_2021.parquet"
    )

    return proj_dir


@pytest.fixture
def sample_models_config_dict():
    """A sample config dictionary for testing the manager."""
    return {
        "input_data": {
            "dataset_name": "test_data",
            "splitted": "year",
            "date_column": "date",
            "id_column": "id",
        },
        "evaluation": {
            "implementation": "rolling window",
            "train_month": 12,
            "validation_month": 0,
            "testing_month": 12,
            "step_month": 12,
            "metrics": ["r2_oos"],
        },
        "models": [
            {
                "name": "ols_model",
                "type": "ols",
                "target_column": "ret",
                "features": ["feature"],
                "save_prediction_results": True,
            }
        ],
    }


# --- Test class for _get_data_for_window ---


class TestGetDataForWindow:
    def test_raises_error_if_project_root_is_not_set(self, sample_models_config_dict):
        """Tests that a ValueError is raised if _project_root is None."""
        config = ModelsConfig.model_validate(sample_models_config_dict)
        manager = ModelManager(config)
        with pytest.raises(ValueError, match="Project root must be set"):
            manager._get_data_for_window(datetime(2020, 1, 1), datetime(2020, 1, 1))

    def test_loads_non_splitted_data_and_caches_it(
        self, sample_models_config_dict, mock_project_root
    ):
        """Tests loading a single, non-partitioned parquet file and caching it."""
        # Modify config for non-splitted data
        sample_models_config_dict["input_data"]["splitted"] = "none"
        config = ModelsConfig.model_validate(sample_models_config_dict)
        manager = ModelManager(config)
        manager._project_root = mock_project_root

        # Create the dummy non-splitted file
        data_path = mock_project_root / "data" / "processed" / "test_data.parquet"
        dummy_df = pl.DataFrame({"col1": [1.0], "col2": [2.0]}).with_columns(
            pl.all().cast(pl.Float64)
        )
        dummy_df.write_parquet(data_path)

        # First call: should read from disk and cache
        with patch("paper_model.manager.pl.read_parquet") as mock_read:
            mock_read.return_value = dummy_df
            df1 = manager._get_data_for_window(
                datetime(2020, 1, 1), datetime(2020, 1, 1)
            )
            mock_read.assert_called_once_with(data_path)
            assert hasattr(manager, "_cached_full_data")
            # Check that Float64 columns were cast to Float32
            assert df1["col1"].dtype == pl.Float32

        # Second call: should use the cache and not read from disk
        with patch("paper_model.manager.pl.read_parquet") as mock_read_again:
            df2 = manager._get_data_for_window(
                datetime(2021, 1, 1), datetime(2021, 1, 1)
            )
            mock_read_again.assert_not_called()
            assert_frame_equal(df1, df2)

    def test_non_splitted_data_file_not_found(
        self, sample_models_config_dict, mock_project_root
    ):
        """Tests that FileNotFoundError is raised if the non-splitted file is missing."""
        sample_models_config_dict["input_data"]["splitted"] = "none"
        config = ModelsConfig.model_validate(sample_models_config_dict)
        manager = ModelManager(config)
        manager._project_root = mock_project_root

        with pytest.raises(FileNotFoundError):
            manager._get_data_for_window(datetime(2020, 1, 1), datetime(2020, 1, 1))

    def test_returns_empty_df_if_no_year_files_found(
        self, sample_models_config_dict, mock_project_root, caplog
    ):
        """Tests that an empty DataFrame with the correct schema is returned if no files match the window."""
        config = ModelsConfig.model_validate(sample_models_config_dict)
        manager = ModelManager(config)
        manager._project_root = mock_project_root
        # Manually set schema for the test
        manager._all_data_columns = ["date", "id", "feature", "ret"]

        # Request a window where no files exist
        result_df = manager._get_data_for_window(
            datetime(2025, 1, 1), datetime(2025, 12, 31)
        )

        assert "No data files found for window" in caplog.text
        assert result_df.is_empty()
        assert result_df.schema.names() == ["date", "id", "feature", "ret"]
        assert result_df["date"].dtype == pl.Date
        assert result_df["id"].dtype == pl.Int64


# --- Tests for ModelManager ---


def test_manager_initialization(sample_models_config_dict):
    """Tests that the manager initializes correctly from a config."""
    config = ModelsConfig.model_validate(sample_models_config_dict)
    manager = ModelManager(config)
    assert manager.config == config
    assert manager.models == {}


@patch("paper_model.manager.pl.read_parquet_schema")
@patch("paper_model.manager.pl.scan_parquet")
def test_get_all_data_columns_and_months(
    mock_scan, mock_schema, sample_models_config_dict, mock_project_root
):
    """Tests the logic for scanning data files to get metadata."""
    # Setup mocks
    mock_schema.return_value = {
        "date": pl.Date,
        "id": pl.Int64,
        "feature": pl.Float64,
        "ret": pl.Float64,
    }
    mock_scan.return_value.select.return_value.unique.return_value.collect.return_value = pl.DataFrame(
        {
            "date": pl.date_range(
                datetime(2020, 1, 1), datetime(2020, 12, 31), "1mo", eager=True
            )
        }
    )

    config = ModelsConfig.model_validate(sample_models_config_dict)
    manager = ModelManager(config)
    manager._project_root = mock_project_root

    columns, months = manager._get_all_data_columns_and_months()

    assert columns == ["date", "id", "feature", "ret"]
    assert len(months) == 12
    assert mock_scan.call_count == 2  # For 2020 and 2021 files


def test_resolve_feature_columns(sample_models_config_dict):
    """Tests the feature resolution logic."""
    config = ModelsConfig.model_validate(sample_models_config_dict)
    manager = ModelManager(config)
    all_cols = ["date", "id", "ret", "feature1", "feature2"]

    # Test explicit list
    resolved = manager._resolve_feature_columns(["feature1"], all_cols)
    assert resolved == ["feature1"]

    feature_config_obj = FeatureSelectionConfig(
        method="all_except", columns=["date", "id", "ret"]
    )
    # Pass the object to the function
    resolved = manager._resolve_feature_columns(feature_config_obj, all_cols)
    assert resolved == ["feature1", "feature2"]

    # Test error on missing feature
    with pytest.raises(ValueError, match="Specified feature columns not found"):
        manager._resolve_feature_columns(["non_existent_feature"], all_cols)


@patch("paper_model.manager.SklearnModel")
def test_initialize_models(mock_sklearn_model_class, sample_models_config_dict):
    """Tests that models are correctly instantiated from the config."""
    config = ModelsConfig.model_validate(sample_models_config_dict)

    # We patch the class in the registry.
    with patch.dict(ModelManager.MODEL_REGISTRY, {"ols": mock_sklearn_model_class}):
        manager = ModelManager(config)
        manager._all_data_columns = ["date", "id", "feature", "ret"]

        manager._initialize_models()

        mock_sklearn_model_class.assert_called_once()

        # call_args is a tuple: (args, kwargs)
        # args is a tuple of positional args: (name, config_dict)
        pos_args, _ = mock_sklearn_model_class.call_args
        name_arg = pos_args[0]
        config_arg = pos_args[1]

        assert name_arg == "ols_model"
        assert config_arg["feature_columns"] == ["feature"]
        assert "ols_model" in manager.models


@patch("paper_model.manager.ModelManager._get_data_for_window")
def test_run_rolling_window_evaluation(mock_get_data, sample_models_config_dict):
    """Tests the main rolling window evaluation loop."""
    # Setup mocks
    mock_model_instance = MagicMock()
    mock_model_instance.config = {
        "target_column": "ret",
        "save_prediction_results": True,
        "feature_columns": ["feature"],
    }
    # The predict method will be called for each test month. Let's make it return a single value.
    mock_model_instance.predict.return_value = pl.Series("pred", [0.5])

    mock_sklearn_model_class = MagicMock(return_value=mock_model_instance)

    # Create mock data that covers both the first training window (2020) and the first test window (2021)
    train_window_data = pl.DataFrame(
        {
            "date": pl.date_range(
                datetime(2020, 1, 31), datetime(2020, 12, 31), "1mo", eager=True
            ),
            "id": range(12),
            "feature": [float(i) for i in range(12)],
            "ret": [float(i) * 0.1 for i in range(12)],
        }
    )
    test_window_data = pl.DataFrame(
        {
            "date": pl.date_range(
                datetime(2021, 1, 31), datetime(2021, 12, 31), "1mo", eager=True
            ),
            "id": range(12),
            "feature": [float(i) for i in range(12, 24)],
            "ret": [float(i) * 0.1 for i in range(12, 24)],
        }
    )
    # The mock should return the combined data for the first window
    mock_get_data.return_value = pl.concat([train_window_data, test_window_data])

    config = ModelsConfig.model_validate(sample_models_config_dict)

    with patch.dict(ModelManager.MODEL_REGISTRY, {"ols": mock_sklearn_model_class}):
        manager = ModelManager(config)
        manager._all_data_columns = ["date", "id", "feature", "ret"]
        manager._initialize_models()

        # The unique months should match the full range the manager will iterate over
        unique_months = pl.date_range(
            datetime(2020, 1, 1), datetime(2021, 12, 31), "1mo", eager=True
        ).to_list()

        manager._run_rolling_window_evaluation(unique_months)

    # Assertions
    # It's called once for the first (and only) window in this test
    mock_get_data.assert_called_once()
    mock_model_instance.train.assert_called_once()
    # It should be called 12 times, once for each month in the test period (2021)
    assert mock_model_instance.predict.call_count == 12
    assert len(manager.all_evaluation_results["ols_model"]) == 12
