import pytest
import requests
from typer.testing import CliRunner
from unittest.mock import MagicMock, patch, call
import subprocess
import zipfile

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


# --- Tests for `paper publish zenodo` ---


@pytest.fixture
def mock_zenodo_project(project):
    """Creates a project with some files to be archived."""
    # Add some files that should be included
    (project / "results.txt").write_text("some results")
    (project / "configs" / "extra.conf").write_text("config")

    # Add some files that should be excluded
    (project / "data" / "raw" / "data.csv").write_text("1,2,3")
    (project / "data" / "processed" / "out.parquet").write_text("...")
    (project / "logs.log").write_text("some logs")  # Excluded by EXCLUDED_FILES
    (project / ".git" / "config").mkdir(parents=True)  # Excluded by EXCLUDED_DIRS

    return project


def test_create_archive_excludes_correctly(mock_zenodo_project):
    """Unit test for the _create_archive helper function."""
    archive_path = mock_zenodo_project.parent / "test_archive.zip"

    file_count = cli._create_archive(mock_zenodo_project, archive_path)

    # Check that the correct number of files were added
    # Expected files: README.md, configs/paper-project.yaml, configs/data-config.yaml,
    # configs/models-config.yaml, configs/portfolio-config.yaml, results.txt, configs/extra.conf
    assert file_count == 7

    with zipfile.ZipFile(archive_path, "r") as zf:
        archived_files = zf.namelist()

        # Assert included files
        assert "README.md" in archived_files
        assert "results.txt" in archived_files
        assert "configs/extra.conf" in archived_files

        # Assert excluded files
        assert "data/raw/data.csv" not in archived_files
        assert "data/processed/out.parquet" not in archived_files
        assert "logs.log" not in archived_files
        assert ".git/config" not in archived_files


@patch("paper_asset_pricing.cli.requests.post")
@patch("paper_asset_pricing.cli.requests.put")
@patch("paper_asset_pricing.cli.os.getenv", return_value="DUMMY_TOKEN")
@patch("paper_asset_pricing.cli.typer.prompt")
def test_publish_zenodo_happy_path(
    mock_prompt, mock_getenv, mock_put, mock_post, mock_zenodo_project
):
    """Tests a successful run of the 'publish zenodo' command."""
    # --- Arrange Mocks ---
    # Mock the author prompt
    mock_prompt.side_effect = ["Test Author", "Test University"]

    # Mock the API responses
    mock_post.return_value.raise_for_status.return_value = None
    mock_post.return_value.json.return_value = {
        "id": 12345,
        "links": {
            "bucket": "https://zenodo.example.com/api/files/bucket-id",
            "html": "https://zenodo.example.com/record/12345",
        },
    }
    mock_put.return_value.raise_for_status.return_value = None

    # --- Act ---
    result = runner.invoke(
        cli.app, ["publish", "zenodo", "-p", str(mock_zenodo_project)]
    )

    # --- Assert ---
    assert result.exit_code == 0, result.stderr
    assert ">>> Creating Zenodo Draft Publication <<<" in result.stdout
    assert "✓ Created new draft deposition on Zenodo." in result.stdout
    assert "✓ Uploaded project archive." in result.stdout
    assert "✓ Set basic metadata" in result.stdout
    assert "🎉 Draft created successfully!" in result.stdout
    assert "https://zenodo.example.com/record/12345" in result.stdout

    # Check that the API calls were made correctly
    assert mock_post.call_count == 1  # Only one POST to create the deposition
    assert mock_put.call_count == 2  # One PUT to upload file, one PUT to set metadata

    # Check the metadata payload
    metadata_call = mock_put.call_args_list[1]  # The second PUT call is for metadata
    sent_metadata = metadata_call.kwargs["json"]["metadata"]
    assert sent_metadata["title"] == "my-test-proj"
    assert sent_metadata["creators"][0]["name"] == "Test Author"
    assert sent_metadata["creators"][0]["affiliation"] == "Test University"


@patch("paper_asset_pricing.cli.os.getenv", return_value=None)  # No env var
@patch("paper_asset_pricing.cli.typer.prompt", return_value="TOKEN_FROM_PROMPT")
def test_publish_zenodo_prompts_for_token(
    mock_prompt, mock_getenv, mock_zenodo_project
):
    """Tests that the user is prompted for a token if the env var is not set."""
    # We only need to test the prompt, so we can mock the rest to fail early
    with patch(
        "paper_asset_pricing.cli.requests.post", side_effect=Exception("API call made")
    ):
        # We expect the command to fail after the prompt, which is fine for this test
        result = runner.invoke(
            cli.app, ["publish", "zenodo", "-p", str(mock_zenodo_project)]
        )

    # Assert that the tutorial text was printed to standard output
    assert "--- How to get a Zenodo API Token ---" in result.stdout
    assert "https://zenodo.org/account/settings/applications/" in result.stdout

    # Assert that typer.prompt was called to get the token
    # The prompt for the token is the first time typer.prompt is called.
    mock_prompt.assert_any_call("\nPlease enter your Zenodo API token", hide_input=True)


@patch("paper_asset_pricing.cli.requests.post")
@patch("paper_asset_pricing.cli.os.getenv", return_value="DUMMY_TOKEN")
@patch("paper_asset_pricing.cli.typer.prompt")
def test_publish_zenodo_handles_api_error(
    mock_prompt, mock_getenv, mock_post, mock_zenodo_project
):
    """Tests graceful failure when the Zenodo API returns an error."""
    # Arrange
    mock_prompt.side_effect = ["Test Author", "Test University"]

    # Simulate an HTTP error
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {"status": 400, "message": "Validation error."}
    mock_post.return_value.raise_for_status.side_effect = requests.HTTPError(
        response=mock_response
    )

    # Act
    result = runner.invoke(
        cli.app, ["publish", "zenodo", "-p", str(mock_zenodo_project)]
    )

    # Assert
    assert result.exit_code == 1
    assert "❌ A Zenodo API error occurred: 400" in result.stderr
    assert "Validation error" in result.stderr
