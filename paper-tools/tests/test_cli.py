import pytest
from typer.testing import CliRunner

import paper_tools.cli as cli  # type: ignore

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolate_cwd(tmp_path, monkeypatch):
    """
    For every test, switch cwd into a fresh tmp_path.
    """
    monkeypatch.chdir(tmp_path)
    yield


def test_init_creates_full_project(tmp_path):
    # run `paper init myproj`
    result = runner.invoke(cli.app, ["init", "myproj"])
    assert result.exit_code == 0, result.stdout + result.stderr

    proj = tmp_path / "myproj"
    # check top-level directories
    expected_dirs = [
        "configs",
        "data/raw",
        "data/processed",
        "models/saved",
        "portfolios/results",
        "portfolios/indexes",
    ]
    for d in expected_dirs:
        assert (proj / d).is_dir(), f"Missing directory: {d}"

    # check files created
    files = [
        ("configs", cli.DEFAULT_PROJECT_CONFIG_NAME),
        (".", ".gitignore"),
        (".", "README.md"),
        (".", cli.LOG_FILE_NAME),
    ]
    for parent, fname in files:
        assert (proj / parent / fname).exists(), f"Missing file: {parent}/{fname}"

    # component placeholders
    for comp in (
        cli.DATA_COMPONENT_CONFIG_FILENAME,
        cli.MODELS_COMPONENT_CONFIG_FILENAME,
        cli.PORTFOLIO_COMPONENT_CONFIG_FILENAME,
    ):
        path = proj / "configs" / comp
        assert path.exists() and path.read_text().startswith("# Placeholder"), comp


def test_init_fails_if_name_is_existing_file(tmp_path):
    # create a file named 'foo'
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
    result = runner.invoke(cli.app, ["init", "proj", "--force"])
    assert result.exit_code == 0
    # warning about overwrite
    assert "Overwriting due to --force" in result.stdout
    # old file should be gone
    assert not (d / "keep.txt").exists()


@pytest.mark.parametrize(
    ("subcmd", "flag", "pkg_name", "attr"),
    [
        ("data", "data", "paper-data", "PAPER_DATA_AVAILABLE"),
        ("models", "models", "paper-model", "PAPER_MODEL_AVAILABLE"),
        ("portfolio", "portfolio", "paper-portfolio", "PAPER_PORTFOLIO_AVAILABLE"),
    ],
)
def test_execute_phase_errors_if_component_missing(subcmd, flag, pkg_name, attr):
    # ensure component is reported missing
    setattr(cli, attr, False)
    result = runner.invoke(cli.app, ["execute", subcmd])
    assert result.exit_code == 1
    assert f"Error: The '{pkg_name}' component is not installed" in result.stderr


def test_execute_data_autodetect_fails(tmp_path):
    # pretend data component is present, but no project-config to auto-detect
    cli.PAPER_DATA_AVAILABLE = True
    # ensure DataManager is not actually invoked
    cli.DataManager = lambda *args, **kwargs: None

    result = runner.invoke(cli.app, ["execute", "data"])
    assert result.exit_code == 1
    assert "Could not auto-detect project root" in result.stderr


def test_execute_models_autodetect_fails(tmp_path):
    cli.PAPER_MODEL_AVAILABLE = True
    cli.load_models_config = lambda *args, **kwargs: None
    result = runner.invoke(cli.app, ["execute", "models"])
    assert result.exit_code == 1
    assert "Could not auto-detect project root" in result.stderr


def test_execute_portfolio_autodetect_fails(tmp_path):
    cli.PAPER_PORTFOLIO_AVAILABLE = True
    cli.load_portfolio_config = lambda *args, **kwargs: None
    result = runner.invoke(cli.app, ["execute", "portfolio"])
    assert result.exit_code == 1
    assert "Could not auto-detect project root" in result.stderr
