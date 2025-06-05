import pytest
from typer.testing import CliRunner
from pathlib import Path
import shutil
import yaml
import subprocess  # For mocking
import os

from paper_tools.cli import app, paper_tools_version, TEMPLATE_DIR  # type: ignore
from paper_tools.cli import (
    CONFIGS_DIR_NAME,
    DATA_DIR_NAME,
    MODELS_DIR_NAME,
    PORTFOLIOS_DIR_NAME,
    LOG_FILE_NAME,
    DEFAULT_PROJECT_CONFIG_NAME,
    DATA_COMPONENT_CONFIG_FILENAME,
    MODELS_COMPONENT_CONFIG_FILENAME,
    PORTFOLIO_COMPONENT_CONFIG_FILENAME,
)

runner = CliRunner()

# --- Fixtures ---


@pytest.fixture(scope="function")
def temp_project_dir(tmp_path_factory):
    """Creates a temporary directory for a test project and cd's into it."""
    project_parent = tmp_path_factory.mktemp("projects")
    original_cwd = Path.cwd()
    # Create a subdirectory within project_parent to run tests from,
    # so that the project itself is created as a subdir of CWD for the test.
    test_run_dir = project_parent / "test_run_space"
    test_run_dir.mkdir()
    os.chdir(test_run_dir)
    yield test_run_dir  # This is where the command will be run from
    os.chdir(original_cwd)  # Clean up by changing back


@pytest.fixture
def mock_subprocess_run(monkeypatch):
    """Mocks subprocess.run to control its behavior."""
    mock_results = []

    def mock_run(*args, **kwargs):
        mock_run.calls.append({"args": args, "kwargs": kwargs})
        if mock_results:
            res = mock_results.pop(0)
            if isinstance(res, Exception):  # If it's any exception, raise it
                raise res
            return res  # Otherwise, it's a CompletedProcess object
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="Mock success", stderr=""
        )

    mock_run.calls = []
    monkeypatch.setattr(subprocess, "run", mock_run)
    return mock_run, mock_results


@pytest.fixture
def mock_shutil_which(monkeypatch):
    """Mocks shutil.which to control CLI existence checks."""
    available_clis = set()

    def mock_which(cmd):
        mock_which.calls.append(cmd)
        return cmd if cmd in available_clis else None

    mock_which.calls = []
    monkeypatch.setattr(shutil, "which", mock_which)
    return available_clis  # Allow tests to set which CLIs are "available"


# --- Tests for `init` command ---


def test_init_successful_creation(temp_project_dir):
    project_name = "MyTestProject"
    result = runner.invoke(app, ["init", project_name])

    assert result.exit_code == 0, f"CLI exited with error: {result.stdout}"
    assert (
        f"P.A.P.E.R project '{project_name}' initialized successfully!" in result.stdout
    )

    project_path = temp_project_dir / project_name
    assert project_path.is_dir()

    # Check main directories
    expected_main_dirs = [
        CONFIGS_DIR_NAME,
        DATA_DIR_NAME,
        MODELS_DIR_NAME,
        PORTFOLIOS_DIR_NAME,
    ]
    for dir_name in expected_main_dirs:
        assert (project_path / dir_name).is_dir()

    # Check subdirectories
    assert (project_path / DATA_DIR_NAME / "raw").is_dir()
    assert (project_path / DATA_DIR_NAME / "processed").is_dir()
    assert (project_path / MODELS_DIR_NAME / "saved").is_dir()
    assert (project_path / PORTFOLIOS_DIR_NAME / "results").is_dir()

    # Check main files
    assert (project_path / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME).is_file()
    assert (project_path / ".gitignore").is_file()
    assert (project_path / "README.md").is_file()
    assert (project_path / LOG_FILE_NAME).is_file()

    # Check placeholder component configs
    for conf_file in [
        DATA_COMPONENT_CONFIG_FILENAME,
        MODELS_COMPONENT_CONFIG_FILENAME,
        PORTFOLIO_COMPONENT_CONFIG_FILENAME,
    ]:
        assert (project_path / CONFIGS_DIR_NAME / conf_file).is_file()
        # Check content of placeholder data config for example
        if conf_file == DATA_COMPONENT_CONFIG_FILENAME:
            content = (project_path / CONFIGS_DIR_NAME / conf_file).read_text()
            assert f'path: "{DATA_DIR_NAME}/raw/your_data.csv"' in content

    # Check .gitkeep files (example: data/raw should be empty and have .gitkeep)
    data_raw_path = project_path / DATA_DIR_NAME / "raw"
    assert (data_raw_path / ".gitkeep").is_file()
    assert len(list(f for f in data_raw_path.iterdir() if f.name != ".gitkeep")) == 0

    configs_path = project_path / CONFIGS_DIR_NAME
    # configs dir should NOT have .gitkeep because it contains config files
    assert not (configs_path / ".gitkeep").exists()
    assert any(configs_path.iterdir())  # Should have files

    # Check content of paper-project.yaml
    with open(project_path / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME, "r") as f:
        main_config = yaml.safe_load(f)
    assert main_config["project_name"] == project_name
    assert main_config["paper_tools_version"] == paper_tools_version
    assert "components" in main_config


def test_init_force_overwrite_existing_dir(temp_project_dir):
    project_name = "OverwriteProject"
    project_path = temp_project_dir / project_name
    project_path.mkdir()
    (project_path / "some_old_file.txt").write_text("old content")

    result = runner.invoke(app, ["init", project_name, "--force"])

    assert result.exit_code == 0
    assert "Overwriting due to --force" in result.stdout
    assert not (project_path / "some_old_file.txt").exists()  # Old file should be gone
    assert (
        project_path / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME
    ).is_file()  # New structure exists


def test_init_existing_dir_no_force_fails(temp_project_dir):
    project_name = "ExistingProject"
    project_path = temp_project_dir / project_name
    project_path.mkdir()
    (project_path / "a_file.txt").write_text("hello")  # Make it non-empty

    result = runner.invoke(app, ["init", project_name])

    assert result.exit_code != 0
    assert "already exists and is not empty" in result.stderr


def test_init_existing_empty_dir_no_force_succeeds(temp_project_dir):
    project_name = "EmptyExistingProject"
    project_path = temp_project_dir / project_name
    project_path.mkdir()  # Exists but empty

    result = runner.invoke(app, ["init", project_name])
    assert result.exit_code == 0
    assert (project_path / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME).is_file()


def test_init_target_is_file_fails(temp_project_dir):
    project_name = "FileAsProject"
    file_path = temp_project_dir / project_name
    file_path.write_text("I am a file")

    result = runner.invoke(app, ["init", project_name])

    assert result.exit_code != 0
    assert f"A file named '{project_name}' already exists" in result.stderr


def test_init_gitkeep_logic_after_populating_configs(temp_project_dir):
    project_name = "GitkeepTest"
    result = runner.invoke(app, ["init", project_name])
    assert result.exit_code == 0

    project_path = temp_project_dir / project_name
    configs_dir = project_path / CONFIGS_DIR_NAME

    # configs_dir should contain paper-project.yaml and placeholder configs, so no .gitkeep
    assert (configs_dir / DEFAULT_PROJECT_CONFIG_NAME).exists()
    assert not (configs_dir / ".gitkeep").exists()

    # data/raw should be empty except for .gitkeep
    data_raw_dir = project_path / DATA_DIR_NAME / "raw"
    assert (data_raw_dir / ".gitkeep").exists()
    items_in_data_raw = [item.name for item in data_raw_dir.iterdir()]
    assert items_in_data_raw == [".gitkeep"]


def test_init_existing_empty_dir_with_force(temp_project_dir):  # Covers line 218
    project_name = "EmptyForceProject"
    project_path = temp_project_dir / project_name
    project_path.mkdir()  # Exists but empty

    result = runner.invoke(app, ["init", project_name, "--force"])  # Add --force
    assert result.exit_code == 0
    assert (
        f"Info: Project directory '{project_name}' exists but is empty. Proceeding."
        in result.stdout
    )
    assert (project_path / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME).is_file()


def test_init_force_rmtree_fails(temp_project_dir, monkeypatch):  # Covers lines 202-208
    project_name = "RmtreeFailProject"
    project_path = temp_project_dir / project_name
    project_path.mkdir()
    (project_path / "a_file.txt").write_text("content")

    def mock_rmtree_error(path):
        raise OSError("Simulated rmtree failure")

    monkeypatch.setattr(shutil, "rmtree", mock_rmtree_error)

    result = runner.invoke(app, ["init", project_name, "--force"])
    assert result.exit_code != 0
    assert "Error: Could not remove existing directory" in result.stderr
    assert "Simulated rmtree failure" in result.stderr


def test_init_general_exception_handling(
    temp_project_dir, monkeypatch
):  # Covers 384-393
    project_name = "InitGeneralException"

    # Mock the 'open' builtin, which is used to write placeholder configs
    original_open = open

    def mock_open_error(file, mode="r", *args, **kwargs):
        if "w" in mode and Path(file).name == PORTFOLIO_COMPONENT_CONFIG_FILENAME:
            # Let other files be written, fail on the last placeholder
            raise IOError("Simulated I/O error during placeholder creation")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", mock_open_error)  # Mock builtins.open

    result = runner.invoke(app, ["init", project_name])

    assert result.exit_code != 0
    assert "An error occurred during project initialization" in result.stderr
    assert "Simulated I/O error during placeholder creation" in result.stderr


# --- Tests for helper functions (can be tested more directly if needed, or via CLI) ---


def test_load_project_config_malformed_yaml(temp_project_dir):  # Covers 90-96
    project_name = "MalformedConfig"
    project_path = temp_project_dir / project_name
    configs_dir = project_path / CONFIGS_DIR_NAME
    configs_dir.mkdir(parents=True)
    malformed_config_path = configs_dir / DEFAULT_PROJECT_CONFIG_NAME
    malformed_config_path.write_text(
        "project_name: Test\n  bad_indent: here"
    )  # Invalid YAML

    # To test _load_project_config directly, we'd call it.
    # To test via CLI, we need an execute command to trigger it.
    # Let's test via execute command.
    original_cwd = Path.cwd()
    os.chdir(project_path)  # _get_project_root uses cwd()
    result = runner.invoke(app, ["execute", "data"])  # This will try to load config
    os.chdir(original_cwd)

    assert result.exit_code != 0
    assert (
        f"Error loading project configuration '{malformed_config_path}'"
        in result.stderr
    )


def test_render_template_not_found(temp_project_dir, monkeypatch):  # Covers 130-136
    project_name = "MissingTemplateProject"

    # Temporarily make a template appear missing by renaming TEMPLATE_DIR for this test
    # This is a bit intrusive; a better way might be to mock FileSystemLoader or env.get_template
    original_template_dir = TEMPLATE_DIR

    # Create a dummy TEMPLATE_DIR that doesn't have the expected template
    dummy_template_dir = temp_project_dir / "dummy_templates"
    dummy_template_dir.mkdir()

    monkeypatch.setattr("paper_tools.cli.TEMPLATE_DIR", dummy_template_dir)

    result = runner.invoke(app, ["init", project_name])

    monkeypatch.setattr(
        "paper_tools.cli.TEMPLATE_DIR", original_template_dir
    )  # Restore

    assert result.exit_code != 0
    assert "Error: Template 'paper_project.yaml.template' not found" in result.stderr


# --- Tests for `execute` commands ---


def _setup_basic_project(project_path: Path, project_name: str = "ExecProject"):
    """Helper to create a minimal project structure for execute tests."""
    # Run init to get the basic structure and main config
    runner.invoke(app, ["init", project_name])
    # The project is created in current CWD, which is project_path for this helper
    # So, the actual project is at project_path / project_name
    return project_path / project_name


def test_execute_data_successful(
    temp_project_dir, mock_subprocess_run, mock_shutil_which
):
    project_name = "DataExecTest"
    actual_project_path = _setup_basic_project(temp_project_dir, project_name)

    # Make paper-data CLI "available"
    mock_shutil_which.add("paper-data")

    # Create the component config file that paper-data would use
    data_config_content = {
        "sources": [{"name": "test_source", "path": "data/raw/test.csv"}]
    }
    data_config_path = (
        actual_project_path / CONFIGS_DIR_NAME / DATA_COMPONENT_CONFIG_FILENAME
    )
    with open(data_config_path, "w") as f:
        yaml.dump(data_config_content, f)

    # Change CWD to the actual project path for the execute command
    original_cwd = Path.cwd()
    os.chdir(actual_project_path)

    mock_run, mock_results = mock_subprocess_run
    mock_results.append(
        subprocess.CompletedProcess(
            args=[], returncode=0, stdout="paper-data ran", stderr=""
        )
    )

    result = runner.invoke(app, ["execute", "data"])

    os.chdir(original_cwd)  # Restore CWD

    assert result.exit_code == 0, f"CLI error: {result.stdout}"
    assert "paper-data ran" in result.stdout
    assert "'paper-data process' executed successfully" in result.stdout

    assert len(mock_run.calls) == 1
    call_args = mock_run.calls[0]["args"][0]  # The cmd list
    assert call_args[0] == "paper-data"
    assert call_args[1] == "process"  # Default command
    assert "--config" in call_args
    assert (
        str(data_config_path.resolve()) in call_args
    )  # Ensure absolute path is passed
    assert "--project-root" in call_args
    assert str(actual_project_path.resolve()) in call_args


def test_execute_data_cli_command_override(
    temp_project_dir, mock_subprocess_run, mock_shutil_which
):
    project_name = "DataCmdOverride"
    actual_project_path = _setup_basic_project(temp_project_dir, project_name)
    mock_shutil_which.add("custom-data-tool")

    # Modify paper-project.yaml to use a custom CLI tool and command
    main_config_path = (
        actual_project_path / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME
    )
    with open(main_config_path, "r") as f:
        main_config = yaml.safe_load(f)
    main_config["components"]["data"]["cli_tool"] = "custom-data-tool"
    main_config["components"]["data"]["cli_command"] = "run-pipeline"
    with open(main_config_path, "w") as f:
        yaml.dump(main_config, f)

    data_config_path = (
        actual_project_path / CONFIGS_DIR_NAME / DATA_COMPONENT_CONFIG_FILENAME
    )
    (data_config_path).touch()  # Just needs to exist

    original_cwd = Path.cwd()
    os.chdir(actual_project_path)
    mock_run, _ = mock_subprocess_run
    result = runner.invoke(app, ["execute", "data"])
    os.chdir(original_cwd)

    assert result.exit_code == 0
    assert len(mock_run.calls) == 1
    call_args = mock_run.calls[0]["args"][0]
    assert call_args[0] == "custom-data-tool"
    assert call_args[1] == "run-pipeline"


def test_execute_data_component_cli_not_found(
    temp_project_dir, mock_subprocess_run, mock_shutil_which
):
    project_name = "NoDataCli"
    actual_project_path = _setup_basic_project(temp_project_dir, project_name)
    # Do NOT add "paper-data" to mock_shutil_which

    data_config_path = (
        actual_project_path / CONFIGS_DIR_NAME / DATA_COMPONENT_CONFIG_FILENAME
    )
    (data_config_path).touch()

    original_cwd = Path.cwd()
    os.chdir(actual_project_path)
    result = runner.invoke(app, ["execute", "data"])
    os.chdir(original_cwd)

    assert result.exit_code != 0
    assert "Component CLI 'paper-data' not found in PATH" in result.stderr


def test_execute_data_component_config_missing_warning(
    temp_project_dir, mock_subprocess_run, mock_shutil_which
):
    project_name = "NoDataCompCfg"
    # _setup_basic_project runs 'init', which creates the placeholder config files
    actual_project_path = _setup_basic_project(temp_project_dir, project_name)
    mock_shutil_which.add("paper-data")

    # Explicitly delete the placeholder config file for this test case ---
    component_config_file_to_delete = (
        actual_project_path / CONFIGS_DIR_NAME / DATA_COMPONENT_CONFIG_FILENAME
    )
    if component_config_file_to_delete.exists():
        component_config_file_to_delete.unlink()
    else:
        # This case should ideally not happen if _setup_basic_project works as expected
        # but good for robustness or if init logic changes.
        print(
            f"WARNING: Test expected {component_config_file_to_delete} to exist for deletion, but it didn't."
        )

    original_cwd = Path.cwd()
    os.chdir(actual_project_path)
    mock_run, _ = mock_subprocess_run  # Mock will return success by default
    result = runner.invoke(app, ["execute", "data"])
    os.chdir(original_cwd)

    assert (
        result.exit_code == 0
    )  # Should still try to run because the warning doesn't cause an exit

    expected_warning = f"Warning: Component configuration file '{actual_project_path / CONFIGS_DIR_NAME / DATA_COMPONENT_CONFIG_FILENAME}' not found"
    assert (
        expected_warning in result.stderr
    )  # The warning message from typer.secho(..., err=True)

    # The CLI should still attempt to run the command, and the mock will make it "succeed"
    assert f"Executing: paper-data process" in result.stdout
    assert (
        "'paper-data process' executed successfully" in result.stdout
    )  # From the mock

    assert len(mock_run.calls) == 1  # Ensure subprocess.run was called


def test_execute_data_subprocess_failure(
    temp_project_dir, mock_subprocess_run, mock_shutil_which
):
    project_name = "DataFail"
    actual_project_path = _setup_basic_project(temp_project_dir, project_name)
    mock_shutil_which.add("paper-data")
    (actual_project_path / CONFIGS_DIR_NAME / DATA_COMPONENT_CONFIG_FILENAME).touch()

    original_cwd = Path.cwd()
    os.chdir(actual_project_path)

    mock_run, mock_results = mock_subprocess_run

    cpe = subprocess.CalledProcessError(
        returncode=1,
        cmd="paper-data ...",  # You can make this more specific if needed
    )
    cpe.stdout = "Component stdout during failure"  # Simulate stdout content
    cpe.stderr = "Component detailed error message"  # Simulate stderr content
    mock_results.append(cpe)

    result = runner.invoke(app, ["execute", "data"])
    os.chdir(original_cwd)

    assert result.exit_code != 0
    assert "Error executing 'paper-data process'" in result.stderr

    # Now check for the stdout and stderr content you set on the exception
    assert "Stdout:" in result.stderr
    assert "Component stdout during failure" in result.stderr
    assert "Stderr:" in result.stderr
    assert "Component detailed error message" in result.stderr


def test_execute_missing_project_config(temp_project_dir):
    # Don't run init, just create an empty dir and cd into it
    project_name = "NoMainConfig"
    project_path = temp_project_dir / project_name
    project_path.mkdir()

    original_cwd = Path.cwd()
    os.chdir(project_path)
    result = runner.invoke(app, ["execute", "data"])
    os.chdir(original_cwd)

    assert result.exit_code != 0
    assert "Project configuration file not found" in result.stderr


def test_execute_missing_component_section_in_project_config(temp_project_dir):
    project_name = "NoCompSection"
    actual_project_path = _setup_basic_project(temp_project_dir, project_name)

    # Corrupt the main config
    main_config_path = (
        actual_project_path / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME
    )
    with open(main_config_path, "r") as f:
        config = yaml.safe_load(f)
    del config["components"]["data"]  # Remove data component section
    with open(main_config_path, "w") as f:
        yaml.dump(config, f)

    original_cwd = Path.cwd()
    os.chdir(actual_project_path)
    result = runner.invoke(app, ["execute", "data"])
    os.chdir(original_cwd)

    assert result.exit_code != 0
    assert "'components.data' configuration missing" in result.stderr


def test_execute_missing_config_file_path_in_project_config(temp_project_dir):
    project_name = "NoCompFilePath"
    actual_project_path = _setup_basic_project(temp_project_dir, project_name)

    main_config_path = (
        actual_project_path / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME
    )
    with open(main_config_path, "r") as f:
        config = yaml.safe_load(f)
    del config["components"]["data"]["config_file"]  # Remove the path
    with open(main_config_path, "w") as f:
        yaml.dump(config, f)

    original_cwd = Path.cwd()
    os.chdir(actual_project_path)
    result = runner.invoke(app, ["execute", "data"])
    os.chdir(original_cwd)

    assert result.exit_code != 0
    assert "'components.data.config_file' path missing" in result.stderr


# Similar tests can be written for `execute models` and `execute portfolio`
# by adapting the component names and config files.


def test_execute_portfolio_successful(
    temp_project_dir, mock_subprocess_run, mock_shutil_which
):
    project_name = "PortfolioExecTest"
    actual_project_path = _setup_basic_project(temp_project_dir, project_name)
    mock_shutil_which.add("paper-portfolio")

    portfolio_config_path = (
        actual_project_path / CONFIGS_DIR_NAME / PORTFOLIO_COMPONENT_CONFIG_FILENAME
    )
    portfolio_config_path.touch()  # Create empty config

    original_cwd = Path.cwd()
    os.chdir(actual_project_path)
    mock_run, _ = mock_subprocess_run
    result = runner.invoke(app, ["execute", "portfolio"])
    os.chdir(original_cwd)

    assert result.exit_code == 0
    assert "'paper-portfolio process' executed successfully" in result.stdout
    assert len(mock_run.calls) == 1
    call_args = mock_run.calls[0]["args"][0]
    assert call_args[0] == "paper-portfolio"
    assert str(portfolio_config_path.resolve()) in call_args


# Test execute_models error paths (analogous to execute_data)
def test_execute_models_missing_project_config(temp_project_dir):  # Covers 478
    project_name = "NoMainConfigModels"
    project_path = temp_project_dir / project_name
    project_path.mkdir()
    original_cwd = Path.cwd()
    os.chdir(project_path)
    result = runner.invoke(app, ["execute", "models"])
    os.chdir(original_cwd)
    assert result.exit_code != 0
    assert "Project configuration file not found" in result.stderr


def test_execute_models_missing_component_section(temp_project_dir):  # Covers 481-488
    project_name = "NoModelsSection"
    actual_project_path = _setup_basic_project(temp_project_dir, project_name)
    main_config_path = (
        actual_project_path / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME
    )
    with open(main_config_path, "r") as f:
        config = yaml.safe_load(f)
    del config["components"]["models"]
    with open(main_config_path, "w") as f:
        yaml.dump(config, f)
    original_cwd = Path.cwd()
    os.chdir(actual_project_path)
    result = runner.invoke(app, ["execute", "models"])
    os.chdir(original_cwd)
    assert result.exit_code != 0
    assert "'components.models' configuration missing" in result.stderr


def test_execute_models_missing_config_file_path(temp_project_dir):  # Covers 494-500
    project_name = "NoModelsFilePath"
    actual_project_path = _setup_basic_project(temp_project_dir, project_name)
    main_config_path = (
        actual_project_path / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME
    )
    with open(main_config_path, "r") as f:
        config = yaml.safe_load(f)
    del config["components"]["models"]["config_file"]
    with open(main_config_path, "w") as f:
        yaml.dump(config, f)
    original_cwd = Path.cwd()
    os.chdir(actual_project_path)
    result = runner.invoke(app, ["execute", "models"])
    os.chdir(original_cwd)
    assert result.exit_code != 0
    assert "'components.models.config_file' path missing" in result.stderr


# Test execute_portfolio error paths
def test_execute_portfolio_missing_project_config(temp_project_dir):  # Covers 528
    project_name = "NoMainConfigPortfolio"
    project_path = temp_project_dir / project_name
    project_path.mkdir()
    original_cwd = Path.cwd()
    os.chdir(project_path)
    result = runner.invoke(app, ["execute", "portfolio"])
    os.chdir(original_cwd)
    assert result.exit_code != 0
    assert "Project configuration file not found" in result.stderr


def test_execute_portfolio_missing_component_section(
    temp_project_dir,
):  # Covers 531-538
    project_name = "NoPortfolioSection"
    actual_project_path = _setup_basic_project(temp_project_dir, project_name)
    main_config_path = (
        actual_project_path / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME
    )
    with open(main_config_path, "r") as f:
        config = yaml.safe_load(f)
    del config["components"]["portfolio"]
    with open(main_config_path, "w") as f:
        yaml.dump(config, f)
    original_cwd = Path.cwd()
    os.chdir(actual_project_path)
    result = runner.invoke(app, ["execute", "portfolio"])
    os.chdir(original_cwd)
    assert result.exit_code != 0
    assert "'components.portfolio' configuration missing" in result.stderr


def test_execute_portfolio_missing_config_file_path(temp_project_dir):  # Covers 544-550
    project_name = "NoPortfolioFilePath"
    actual_project_path = _setup_basic_project(temp_project_dir, project_name)
    main_config_path = (
        actual_project_path / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME
    )
    with open(main_config_path, "r") as f:
        config = yaml.safe_load(f)
    del config["components"]["portfolio"]["config_file"]
    with open(main_config_path, "w") as f:
        yaml.dump(config, f)
    original_cwd = Path.cwd()
    os.chdir(actual_project_path)
    result = runner.invoke(app, ["execute", "portfolio"])
    os.chdir(original_cwd)
    assert result.exit_code != 0
    assert "'components.portfolio.config_file' path missing" in result.stderr


# Tests for _run_component_cli specific error paths
def test_run_component_cli_subprocess_stderr_output(
    temp_project_dir, mock_subprocess_run, mock_shutil_which
):  # Covers 624-625
    project_name = "SubprocessStderr"
    actual_project_path = _setup_basic_project(temp_project_dir, project_name)
    mock_shutil_which.add("paper-data")
    (actual_project_path / CONFIGS_DIR_NAME / DATA_COMPONENT_CONFIG_FILENAME).touch()

    original_cwd = Path.cwd()
    os.chdir(actual_project_path)
    mock_run, mock_results = mock_subprocess_run
    # Simulate subprocess having output on stderr but still succeeding
    mock_results.append(
        subprocess.CompletedProcess(
            args=[], returncode=0, stdout="OK", stderr="Subprocess warning"
        )
    )
    result = runner.invoke(app, ["execute", "data"])
    os.chdir(original_cwd)

    assert result.exit_code == 0
    assert "OK" in result.stdout
    assert "Component output (stderr):" in result.stderr
    assert "Subprocess warning" in result.stderr


def test_run_component_cli_called_process_error_with_stdout_stderr(
    temp_project_dir, mock_subprocess_run, mock_shutil_which
):  # Covers lines for handling e.stdout and e.stderr in CalledProcessError
    project_name = "CalledProcessErrorFull"
    actual_project_path = _setup_basic_project(temp_project_dir, project_name)
    mock_shutil_which.add("paper-data")
    (actual_project_path / CONFIGS_DIR_NAME / DATA_COMPONENT_CONFIG_FILENAME).touch()

    original_cwd = Path.cwd()
    os.chdir(actual_project_path)
    mock_run, mock_results = mock_subprocess_run

    # Create the CalledProcessError instance
    cpe = subprocess.CalledProcessError(
        returncode=1,
        cmd="paper-data ...",  # The command that "failed"
    )
    # Manually set the stdout and stderr attributes on the instance
    cpe.stdout = "Component stdout output during error"
    cpe.stderr = "Component stderr detail of error"
    mock_results.append(cpe)

    result = runner.invoke(app, ["execute", "data"])
    os.chdir(original_cwd)

    assert result.exit_code != 0
    assert "Error executing 'paper-data process'" in result.stderr  # Main error message

    # Assert that the "Stdout:" prefix and the content of cpe.stdout are present
    assert "Stdout:" in result.stderr
    assert "Component stdout output during error" in result.stderr

    # Assert that the "Stderr:" prefix and the content of cpe.stderr are present
    assert "Stderr:" in result.stderr
    assert "Component stderr detail of error" in result.stderr


def test_run_component_cli_general_exception(
    temp_project_dir, mock_subprocess_run, mock_shutil_which
):  # Covers 644-650
    project_name = "RunCliGeneralException"
    actual_project_path = _setup_basic_project(temp_project_dir, project_name)
    mock_shutil_which.add("paper-data")
    (actual_project_path / CONFIGS_DIR_NAME / DATA_COMPONENT_CONFIG_FILENAME).touch()

    original_cwd = Path.cwd()
    os.chdir(actual_project_path)
    mock_run, mock_results = mock_subprocess_run
    # This RuntimeError will now be raised by the mock_run
    mock_results.append(RuntimeError("Simulated unexpected subprocess.run error"))
    result = runner.invoke(app, ["execute", "data"])
    os.chdir(original_cwd)

    assert result.exit_code != 0
    assert (
        "An unexpected error occurred while running 'paper-data process': Simulated unexpected subprocess.run error"
        in result.stderr  # The {e} part will be the string of the RuntimeError
    )


# Test the main app help and execute help
def test_app_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert (
        "P.A.P.E.R Tools: Initialize and execute P.A.P.E.R research project phases."
        in result.stdout
    )
    assert "init" in result.stdout
    assert "execute" in result.stdout


def test_execute_help():
    result = runner.invoke(app, ["execute", "--help"])
    assert result.exit_code == 0
    assert "Execute P.A.P.E.R project phases." in result.stdout
    assert "data" in result.stdout
    assert "models" in result.stdout
    assert "portfolio" in result.stdout
