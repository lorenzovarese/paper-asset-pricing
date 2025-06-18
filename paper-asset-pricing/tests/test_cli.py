import pytest
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch, call
import subprocess

import paper_asset_pricing.cli as cli  # type: ignore

runner = CliRunner()


# --- Fixtures ---


@pytest.fixture(autouse=True)
def isolate_cwd(tmp_path, monkeypatch):
    """
    For every test, switch cwd into a fresh tmp_path to prevent side-effects.
    """
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture
def project(tmp_path):
    """
    Provides a pre-initialized project directory for tests that need it.
    This avoids re-running the `init` command in every test.
    """
    project_name = "my-test-proj"
    # Patch git calls during fixture setup to avoid side-effects
    with patch("paper_asset_pricing.cli.shutil.which", return_value=None):
        runner.invoke(cli.app, ["init", project_name], catch_exceptions=False)

    proj_path = tmp_path / project_name

    # Write minimal valid content to component configs to avoid "file not found" errors
    (proj_path / "configs" / cli.DATA_COMPONENT_CONFIG_FILENAME).write_text("key: val")
    (proj_path / "configs" / cli.MODELS_COMPONENT_CONFIG_FILENAME).write_text(
        "key: val"
    )
    (proj_path / "configs" / cli.PORTFOLIO_COMPONENT_CONFIG_FILENAME).write_text(
        "key: val"
    )
    return proj_path


# --- Tests for `paper init` ---


def test_init_creates_full_project(tmp_path):
    # Patch git calls to isolate this test from git logic
    with patch("paper_asset_pricing.cli.shutil.which", return_value=None):
        result = runner.invoke(cli.app, ["init", "myproj"])

    assert result.exit_code == 0, result.stdout + result.stderr

    proj = tmp_path / "myproj"
    expected_dirs = [
        "configs",
        "data/raw",
        "data/processed",
        "models/evaluations",
        "models/predictions",
        "models/saved",
        "portfolios/results",
        "portfolios/additional_datasets",
    ]
    for d in expected_dirs:
        assert (proj / d).is_dir(), f"Missing directory: {d}"

    files = [
        ("configs", cli.DEFAULT_PROJECT_CONFIG_NAME),
        (".", ".gitignore"),
        (".", "README.md"),
        (".", cli.LOG_FILE_NAME),
    ]
    for parent, fname in files:
        assert (proj / parent / fname).exists(), f"Missing file: {parent}/{fname}"

    for comp in (
        cli.DATA_COMPONENT_CONFIG_FILENAME,
        cli.MODELS_COMPONENT_CONFIG_FILENAME,
        cli.PORTFOLIO_COMPONENT_CONFIG_FILENAME,
    ):
        path = proj / "configs" / comp
        assert path.exists(), f"Config file {comp} should exist"
        assert path.read_text().startswith(
            "# ---------------------------------------------------------------------------"
        ), f"Config file {comp} should be generated from the new template"


def test_init_fails_if_name_is_existing_file(tmp_path):
    (tmp_path / "foo").write_text("hello")
    result = runner.invoke(cli.app, ["init", "foo"])
    assert result.exit_code == 1
    assert "Error: A file named 'foo' already exists." in result.stderr


def test_init_fails_on_nonempty_dir_without_force(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "keep.txt").write_text("something")
    result = runner.invoke(cli.app, ["init", "proj"])
    assert result.exit_code == 1
    assert "already exists and is not empty" in result.stderr


def test_init_overwrites_nonempty_dir_with_force(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "keep.txt").write_text("something")
    with patch("paper_asset_pricing.cli.shutil.which", return_value=None):
        result = runner.invoke(cli.app, ["init", "proj", "--force"])
    assert result.exit_code == 0
    assert "Overwriting due to --force" in result.stdout
    assert not (d / "keep.txt").exists()


def test_init_force_on_empty_dir(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    with patch("paper_asset_pricing.cli.shutil.which", return_value=None):
        result = runner.invoke(cli.app, ["init", "proj", "--force"])
    assert result.exit_code == 0
    assert "exists but is empty. Proceeding" in result.stdout


def test_init_handles_generic_exception(monkeypatch):
    """Covers the main try-except block in `init`."""
    # Mock Path.mkdir to raise an unexpected error
    monkeypatch.setattr(
        "pathlib.Path.mkdir", MagicMock(side_effect=OSError("Disk full"))
    )
    result = runner.invoke(cli.app, ["init", "any-proj"])
    assert result.exit_code == 1
    assert "An error occurred during project initialization: Disk full" in result.stderr


# --- Tests for Git Initialization ---


@patch("paper_asset_pricing.cli.subprocess.run")
@patch("paper_asset_pricing.cli.shutil.which", return_value="/usr/bin/git")
def test_init_initializes_git_repository_when_git_is_available(
    mock_shutil_which, mock_subprocess_run, tmp_path
):
    """Tests that `git init`, `add`, and `commit` are called when git is found."""
    project_name = "git-proj"
    project_path = tmp_path / project_name

    result = runner.invoke(cli.app, ["init", project_name])

    assert result.exit_code == 0, result.stderr
    assert "✓ Initialized git repository." in result.stdout
    assert "✓ Created initial commit." in result.stdout

    mock_shutil_which.assert_called_once_with("git")

    expected_calls = [
        call(
            ["git", "init"],
            cwd=project_path,
            check=True,
            capture_output=True,
            text=True,
        ),
        call(
            ["git", "add", "."],
            cwd=project_path,
            check=True,
            capture_output=True,
            text=True,
        ),
        call(
            ["git", "commit", "-m", "Initial commit: P.A.P.E.R project setup"],
            cwd=project_path,
            check=True,
            capture_output=True,
            text=True,
        ),
    ]
    mock_subprocess_run.assert_has_calls(expected_calls)
    assert mock_subprocess_run.call_count == 3


@patch("paper_asset_pricing.cli.subprocess.run")
@patch("paper_asset_pricing.cli.shutil.which", return_value=None)
def test_init_skips_git_initialization_when_git_is_not_available(
    mock_shutil_which, mock_subprocess_run
):
    """Tests that git commands are skipped and a warning is shown if git is not found."""
    result = runner.invoke(cli.app, ["init", "no-git-proj"])

    assert result.exit_code == 0, result.stderr
    assert "Warning: `git` command not found." in result.stdout
    assert "Initialized git repository" not in result.stdout

    mock_shutil_which.assert_called_once_with("git")
    mock_subprocess_run.assert_not_called()


@patch("paper_asset_pricing.cli.subprocess.run")
@patch("paper_asset_pricing.cli.shutil.which", return_value="/usr/bin/git")
def test_init_handles_git_command_failure_gracefully(
    mock_shutil_which, mock_subprocess_run
):
    """Tests that a failure in a git command shows a warning but doesn't crash."""
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        returncode=128, cmd="git init", stderr="fatal: could not create work tree"
    )

    result = runner.invoke(cli.app, ["init", "git-fail-proj"])

    assert result.exit_code == 0, result.stderr
    assert "Warning: Failed to initialize git repository" in result.stderr
    assert "fatal: could not create work tree" in result.stderr
    assert "✓ Initialized git repository." not in result.stdout

    mock_shutil_which.assert_called_once_with("git")
    mock_subprocess_run.assert_called_once()


# --- Tests for `paper execute` ---


@pytest.mark.parametrize(
    ("subcmd", "attr"),
    [
        ("data", "PAPER_DATA_AVAILABLE"),
        ("models", "PAPER_MODEL_AVAILABLE"),
        ("portfolio", "PAPER_PORTFOLIO_AVAILABLE"),
    ],
)
def test_execute_phase_errors_if_component_missing(subcmd, attr):
    original_value = getattr(cli, attr)
    setattr(cli, attr, False)
    result = runner.invoke(cli.app, ["execute", subcmd])
    assert result.exit_code == 1
    expected_error = (
        f"Error: The 'paper-{subcmd}' component is not installed or importable"
    )
    assert expected_error in result.stderr
    setattr(cli, attr, original_value)


def test_execute_autodetect_fails_if_no_project_found(tmp_path):
    # No project exists in tmp_path, so auto-detect should fail.
    result = runner.invoke(cli.app, ["execute", "data"])
    assert result.exit_code == 1
    assert "Could not auto-detect project root" in result.stderr


def test_execute_fails_if_project_path_is_a_file(project):
    # Create a file and try to use it as a project path
    file_path = project.parent / "not_a_dir.txt"
    file_path.touch()
    result = runner.invoke(cli.app, ["execute", "data", "-p", str(file_path)])
    assert result.exit_code == 1
    assert f"'{file_path}' is not a directory" in result.stderr


def test_execute_fails_if_main_config_is_missing(project):
    # Delete the main project config file
    (project / "configs" / cli.DEFAULT_PROJECT_CONFIG_NAME).unlink()
    result = runner.invoke(cli.app, ["execute", "data", "-p", str(project)])
    assert result.exit_code == 1
    assert "Main project config 'paper-project.yaml' not found" in result.stderr


def test_execute_fails_if_main_config_is_invalid_yaml(project):
    # Write invalid YAML to the config file
    (project / "configs" / cli.DEFAULT_PROJECT_CONFIG_NAME).write_text("key: [invalid")
    result = runner.invoke(cli.app, ["execute", "data", "-p", str(project)])
    assert result.exit_code == 1
    assert "Error loading or parsing main project config" in result.stderr


def test_execute_fails_if_component_config_is_missing(project):
    # Delete the data-specific config file
    (project / "configs" / cli.DATA_COMPONENT_CONFIG_FILENAME).unlink()
    result = runner.invoke(cli.app, ["execute", "data", "-p", str(project)])
    assert result.exit_code == 1
    assert "Data component config file 'data-config.yaml' not found" in result.stderr


@patch("paper_asset_pricing.cli.DataManager")
@patch("paper_asset_pricing.cli.load_data_config")
def test_execute_happy_path_with_autodetect(
    mock_load_data_config, mock_data_manager_class, project, monkeypatch
):
    """Tests a successful run, including auto-detection of the project."""
    # Mock the config loader to return a dummy config object
    mock_config_object = MagicMock()
    mock_load_data_config.return_value = mock_config_object

    # Mock the manager instance to confirm it's called correctly
    mock_manager_instance = MagicMock()
    mock_data_manager_class.return_value = mock_manager_instance

    # Change the current working directory into the project root for auto-detection to work.
    monkeypatch.chdir(project)

    # The project is now the CWD, so auto-detection should succeed.
    result = runner.invoke(cli.app, ["execute", "data"])

    assert result.exit_code == 0, result.stderr
    assert "Auto-detected project root" in result.stdout
    assert "Data phase completed successfully" in result.stdout

    # Verify the config loader was called with the correct path
    component_config_path = project / "configs" / cli.DATA_COMPONENT_CONFIG_FILENAME
    mock_load_data_config.assert_called_once_with(config_path=component_config_path)

    # Verify the manager was instantiated with the loaded config object
    mock_data_manager_class.assert_called_once_with(config=mock_config_object)

    # Verify the manager's run method was called with the project root
    mock_manager_instance.run.assert_called_once_with(project_root=project)


@pytest.mark.parametrize(
    ("exc_to_raise", "expected_msg"),
    [
        (FileNotFoundError, "A required file was not found."),
        (ValueError, "Configuration issue."),
        (NotImplementedError, "A feature is not yet implemented."),
        (KeyError, "An unexpected error occurred"),  # Test the generic Exception case
    ],
)
@patch("paper_asset_pricing.cli.ModelManager")
@patch("paper_asset_pricing.cli.load_models_config")
def test_execute_phase_handles_manager_errors(
    mock_load_config, mock_model_manager_class, exc_to_raise, expected_msg, project
):
    """
    Tests that the runner correctly catches and reports exceptions from the manager's .run() method.
    """
    # Mock the manager's run method to raise a specific exception
    mock_model_manager_class.return_value.run.side_effect = exc_to_raise("test error")

    result = runner.invoke(cli.app, ["execute", "models", "-p", str(project)])

    assert result.exit_code == 1
    assert expected_msg in result.stderr
    assert "Check logs for details" in result.stderr
