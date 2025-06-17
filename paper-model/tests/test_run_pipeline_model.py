import pytest
from unittest.mock import patch, MagicMock


# Import the main function from your script
from paper_model.run_pipeline import main  # type: ignore

# --- Fixtures ---


@pytest.fixture
def mock_project(tmp_path):
    """Creates a minimal mock project structure."""
    proj_dir = tmp_path / "MyTestProject"
    configs_dir = proj_dir / "configs"
    configs_dir.mkdir(parents=True)
    # Create a dummy config file for the script to find
    (configs_dir / "models-config.yaml").touch()
    return proj_dir


# --- Tests for run_pipeline.py ---


@patch("paper_model.run_pipeline.load_config")
@patch("paper_model.run_pipeline.ModelManager")
@patch("paper_model.run_pipeline.argparse.ArgumentParser.parse_args")
def test_main_happy_path(
    mock_parse_args, mock_model_manager, mock_load_config, mock_project
):
    """
    Tests the main function's successful execution path.
    """
    # --- Arrange ---
    # Mock the command-line arguments that argparse would parse
    mock_parse_args.return_value = MagicMock(
        project_root=str(mock_project), config=None
    )

    # Mock the objects that would be created
    mock_config_object = MagicMock()
    mock_load_config.return_value = mock_config_object

    mock_manager_instance = MagicMock()
    mock_model_manager.return_value = mock_manager_instance

    # --- Act ---
    # Run the main function. We don't need to check for SystemExit because it shouldn't be called.
    main()

    # --- Assert ---
    # 1. Assert that the config was loaded with the correct default path
    expected_config_path = mock_project / "configs" / "models-config.yaml"
    mock_load_config.assert_called_once_with(config_path=expected_config_path)

    # 2. Assert that the ModelManager was initialized with the loaded config
    mock_model_manager.assert_called_once_with(config=mock_config_object)

    # 3. Assert that the manager's run method was called with the correct project root
    mock_manager_instance.run.assert_called_once_with(project_root=mock_project)


@patch("paper_model.run_pipeline.sys.exit")
@patch("paper_model.run_pipeline.argparse.ArgumentParser.parse_args")
def test_main_invalid_project_path(mock_parse_args, mock_sys_exit, tmp_path):
    """
    Tests that the script exits if the project path is not a valid directory.
    """
    # --- Arrange ---
    invalid_path = tmp_path / "not_a_real_directory"
    mock_parse_args.return_value = MagicMock(
        project_root=str(invalid_path), config=None
    )

    # --- Act ---
    main()

    # --- Assert ---
    # Assert that sys.exit(1) was called
    mock_sys_exit.assert_called_once_with(1)


@patch("paper_model.run_pipeline.sys.exit")
@patch("paper_model.run_pipeline.load_config")
@patch("paper_model.run_pipeline.argparse.ArgumentParser.parse_args")
def test_main_handles_file_not_found(
    mock_parse_args, mock_load_config, mock_sys_exit, mock_project
):
    """
    Tests that the script exits gracefully if a required file is not found.
    """
    # --- Arrange ---
    mock_parse_args.return_value = MagicMock(
        project_root=str(mock_project), config=None
    )
    # Simulate a FileNotFoundError during config loading
    mock_load_config.side_effect = FileNotFoundError("File not found!")

    # --- Act ---
    main()

    # --- Assert ---
    mock_sys_exit.assert_called_once_with(1)
