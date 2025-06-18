"""
Module: cli.py

This module implements the command-line interface for the P.A.P.E.R Tools package. It provides
commands to initialize a new research project with a standardized directory structure and
to execute various project phases (data processing, modeling, portfolio analysis).
"""

import json
import os
import requests
import typer
from pathlib import Path
import yaml
import shutil
import datetime
import sys
import subprocess
from jinja2 import Environment, FileSystemLoader
import logging
from typing import Callable, Any, Type

# --- Initial logger for paper-asset-pricing, before project-specific logging is set up ---
# This logger will initially print to stdout, but will be reconfigured later.
# It's important that this is NOT basicConfig, as basicConfig configures the root logger
# and we want more control.
logger = logging.getLogger(__name__)
# No handlers are added here initially. Typer.secho will handle early console output.


# --- Try to import from sub-packages ---
# This allows paper-asset-pricing to function (e.g., `init`) even if optional components aren't installed.
try:
    from paper_data.manager import DataManager  # type: ignore
    from paper_data.config_parser import load_config as load_data_config  # type: ignore

    PAPER_DATA_AVAILABLE = True
    logger.debug("paper_data components imported successfully.")
except ImportError:
    PAPER_DATA_AVAILABLE = False
    DataManager = None  # type: ignore
    load_data_config = None  # type: ignore
    logger.debug("Failed to import paper_data components. PAPER_DATA_AVAILABLE=False")

try:
    from paper_model.manager import ModelManager  # type: ignore
    from paper_model.config_parser import load_config as load_models_config  # type: ignore

    PAPER_MODEL_AVAILABLE = True
    logger.debug("paper_model components imported successfully.")
except ImportError:
    PAPER_MODEL_AVAILABLE = False
    ModelManager = None  # type: ignore
    load_models_config = None  # type: ignore
    logger.debug("Failed to import paper_model components. PAPER_MODEL_AVAILABLE=False")

try:
    from paper_portfolio.manager import PortfolioManager  # type: ignore
    from paper_portfolio.config_parser import load_config as load_portfolio_config  # type: ignore

    PAPER_PORTFOLIO_AVAILABLE = True
    logger.debug("paper_portfolio components imported successfully.")
except ImportError:
    PAPER_PORTFOLIO_AVAILABLE = False
    PortfolioManager = None  # type: ignore
    load_portfolio_config = None  # type: ignore
    logger.debug(
        "Failed to import paper_portfolio components. PAPER_PORTFOLIO_AVAILABLE=False"
    )


# Try to get version from __init__ for the config file, fallback if not found
try:
    from . import __version__ as paper_asset_pricing_version
except ImportError:
    paper_asset_pricing_version = "unknown"

app = typer.Typer(
    name="paper",
    help="P.A.P.E.R Tools: Initialize and execute P.A.P.E.R research project phases.",
    add_completion=False,
    no_args_is_help=True,
)

# --- Constants for Project Structure ---
CONFIGS_DIR_NAME = "configs"
DATA_DIR_NAME = "data"
MODELS_DIR_NAME = "models"
PORTFOLIOS_DIR_NAME = "portfolios"
LOG_FILE_NAME = "logs.log"

DEFAULT_PROJECT_CONFIG_NAME = "paper-project.yaml"
DATA_COMPONENT_CONFIG_FILENAME = "data-config.yaml"
MODELS_COMPONENT_CONFIG_FILENAME = "models-config.yaml"
PORTFOLIO_COMPONENT_CONFIG_FILENAME = "portfolio-config.yaml"

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _render_template(template_name: str, context: dict, output_path: Path):
    """
    Render a Jinja2 template with the provided context and write the result to a file.
    """
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    try:
        template = env.get_template(template_name)
    except Exception as e:
        typer.secho(
            f"Error: Template '{template_name}' not found in '{TEMPLATE_DIR}'. Error: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    rendered_content = template.render(context)
    with open(output_path, "w") as f:
        f.write(rendered_content)


def _configure_project_logging(project_root: Path, log_file_name: str, level: str):
    """
    Configures the root logger to write to the project's log file.
    Removes any existing handlers to prevent duplicate output.
    """
    log_file_path = project_root / log_file_name
    log_file_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure log directory exists

    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Clear existing handlers to prevent duplicate output (e.g., from previous basicConfig or runs)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add a FileHandler
    file_handler = logging.FileHandler(log_file_path, mode="a")  # Append mode
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # No StreamHandler is added here, as per the requirement to only print
    # the final success message and errors to console via typer.secho.


@app.command()
def init(
    project_name: str = typer.Argument(
        ...,
        help="The name for the new P.A.P.E.R project directory.",
        metavar="PROJECT_NAME",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing project directory if it exists.",
    ),
):
    """
    Initialize a new P.A.P.E.R project.
    """
    project_path = Path(project_name).resolve()

    if project_path.exists():
        if project_path.is_file():
            typer.secho(
                f"Error: A file named '{project_path.name}' already exists.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        if any(project_path.iterdir()):
            if force:
                typer.secho(
                    f"Warning: Project directory '{project_path.name}' exists and is not empty. "
                    "Overwriting due to --force.",
                    fg=typer.colors.YELLOW,
                )
                try:
                    shutil.rmtree(project_path)
                except Exception as e:
                    typer.secho(
                        f"Error: Could not remove existing directory '{project_path.name}': {e}",
                        fg=typer.colors.RED,
                        err=True,
                    )
                    raise typer.Exit(code=1)
            else:
                typer.secho(
                    f"Error: Project directory '{project_path.name}' already exists and is not empty. "
                    "Use --force or choose a different name.",
                    fg=typer.colors.RED,
                    err=True,
                )
                raise typer.Exit(code=1)
        elif force:  # Directory exists but is empty, and force is used
            typer.secho(
                f"Info: Project directory '{project_path.name}' exists but is empty. Proceeding with overwrite due to --force.",
                fg=typer.colors.BLUE,
            )

    typer.secho(
        f"Initializing P.A.P.E.R project '{project_path.name}' at: {project_path.parent}",
        bold=True,
    )

    try:
        project_path.mkdir(parents=True, exist_ok=True)

        dir_structure_map = {
            CONFIGS_DIR_NAME: [],
            DATA_DIR_NAME: ["raw", "processed", "scripts"],
            MODELS_DIR_NAME: ["evaluations", "predictions", "saved"],
            PORTFOLIOS_DIR_NAME: ["results", "additional_datasets"],
        }
        all_dirs_to_create_paths: list[Path] = []
        for main_dir_name, sub_dir_names in dir_structure_map.items():
            base_path = project_path / main_dir_name
            all_dirs_to_create_paths.append(base_path)
            for sub_dir_name in sub_dir_names:
                all_dirs_to_create_paths.append(base_path / sub_dir_name)
        for dir_p in all_dirs_to_create_paths:
            dir_p.mkdir(parents=True, exist_ok=True)
        typer.secho("✓ Created project directories.", fg=typer.colors.GREEN)

        template_context = {
            "project_name": project_path.name,
            "paper_asset_pricing_version": paper_asset_pricing_version,
            "creation_date": datetime.date.today().isoformat(),
            "CONFIGS_DIR_NAME": CONFIGS_DIR_NAME,
            "DATA_DIR_NAME": DATA_DIR_NAME,
            "MODELS_DIR_NAME": MODELS_DIR_NAME,
            "PORTFOLIOS_DIR_NAME": PORTFOLIOS_DIR_NAME,
            "LOG_FILE_NAME": LOG_FILE_NAME,
            "DEFAULT_PROJECT_CONFIG_NAME": DEFAULT_PROJECT_CONFIG_NAME,
            "DATA_COMPONENT_CONFIG_FILENAME": DATA_COMPONENT_CONFIG_FILENAME,
            "MODELS_COMPONENT_CONFIG_FILENAME": MODELS_COMPONENT_CONFIG_FILENAME,
            "PORTFOLIO_COMPONENT_CONFIG_FILENAME": PORTFOLIO_COMPONENT_CONFIG_FILENAME,
        }

        main_config_output_path = (
            project_path / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME
        )
        _render_template(
            "paper-project.yaml.template", template_context, main_config_output_path
        )
        typer.secho(
            f"✓ Created main project config: {main_config_output_path.relative_to(Path.cwd())}",
            fg=typer.colors.GREEN,
        )

        gitignore_output_path = project_path / ".gitignore"
        _render_template("gitignore.template", template_context, gitignore_output_path)
        typer.secho("✓ Created .gitignore file.", fg=typer.colors.GREEN)

        readme_output_path = project_path / "README.md"
        _render_template(
            "project_readme.md.template", template_context, readme_output_path
        )
        typer.secho("✓ Created project README.md.", fg=typer.colors.GREEN)

        log_file_path = project_path / LOG_FILE_NAME
        with open(log_file_path, "w") as f:
            f.write(
                f"# P.A.P.E.R Project Log for '{project_path.name}' - Initialized: {datetime.datetime.now().isoformat()}\n"
            )
        typer.secho(
            f"✓ Created log file: {log_file_path.relative_to(Path.cwd())}",
            fg=typer.colors.GREEN,
        )

        component_templates = {
            DATA_COMPONENT_CONFIG_FILENAME: "data-config.yaml.template",
            MODELS_COMPONENT_CONFIG_FILENAME: "models-config.yaml.template",
            PORTFOLIO_COMPONENT_CONFIG_FILENAME: "portfolio-config.yaml.template",
        }

        typer.secho(
            "\nCreating component configuration files from templates:", bold=True
        )
        for conf_filename, template_name in component_templates.items():
            output_path = project_path / CONFIGS_DIR_NAME / conf_filename
            _render_template(template_name, template_context, output_path)
            typer.secho(
                f"✓ Created: {output_path.relative_to(Path.cwd())}",
                fg=typer.colors.GREEN,
            )

        for dir_p in all_dirs_to_create_paths:
            is_empty = not any(
                item for item in dir_p.iterdir() if item.name != ".gitkeep"
            )
            if is_empty:
                (dir_p / ".gitkeep").touch()
        typer.secho(
            "✓ Ensured .gitkeep in empty project subdirectories.", fg=typer.colors.GREEN
        )

        # --- Initialize Git Repository ---
        if not shutil.which("git"):
            typer.secho(
                "\nWarning: `git` command not found. Skipping git repository initialization.",
                fg=typer.colors.YELLOW,
            )
        else:
            try:
                # Use capture_output=True to hide git's default messages
                # Use text=True for cleaner error messages if they occur
                subprocess.run(
                    ["git", "init"],
                    cwd=project_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                typer.secho("✓ Initialized git repository.", fg=typer.colors.GREEN)

                subprocess.run(
                    ["git", "add", "."],
                    cwd=project_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    [
                        "git",
                        "commit",
                        "-m",
                        "Initial commit: P.A.P.E.R project setup",
                    ],
                    cwd=project_path,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                typer.secho("✓ Created initial commit.", fg=typer.colors.GREEN)

            except subprocess.CalledProcessError as e:
                typer.secho(
                    f"\nWarning: Failed to initialize git repository or create initial commit. Error: {e.stderr}",
                    fg=typer.colors.YELLOW,
                    err=True,
                )
            except Exception as e:
                typer.secho(
                    f"\nWarning: An unexpected error occurred during git initialization: {e}",
                    fg=typer.colors.YELLOW,
                    err=True,
                )

        typer.secho(
            f"\n🎉 P.A.P.E.R project '{project_path.name}' initialized successfully!",
            bold=True,
            fg=typer.colors.BRIGHT_GREEN,
        )
        typer.secho(
            f'\nNavigate to your project:\n  cd "{project_path.relative_to(Path.cwd())}"',
            fg=typer.colors.CYAN,
        )
        typer.secho("\nNext steps:", fg=typer.colors.CYAN)
        typer.secho(
            f"  1. Populate your component-specific YAML configuration files in '{CONFIGS_DIR_NAME}/'.",
            fg=typer.colors.CYAN,
        )
        typer.secho(
            f"     (e.g., '{DATA_COMPONENT_CONFIG_FILENAME}', '{MODELS_COMPONENT_CONFIG_FILENAME}', '{PORTFOLIO_COMPONENT_CONFIG_FILENAME}')",
            fg=typer.colors.CYAN,
        )
        typer.secho(
            f"  2. (Optional) If using local files, place them in the `{DATA_DIR_NAME}/raw/` directory.",
            fg=typer.colors.CYAN,
        )
        typer.secho(
            "  3. Run the first phase of your research, e.g., `paper execute data`.",
            fg=typer.colors.CYAN,
        )

    except Exception as e:
        typer.secho(
            f"\n❌ An error occurred during project initialization: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        import traceback

        traceback.print_exc(file=sys.stderr)
        raise typer.Exit(code=1)


# --- Helper function to get project root and load main config ---
def _get_project_root_and_load_main_config(
    project_path_str: str | None,
) -> tuple[Path, dict]:
    """
    Determines the project root directory and loads the main paper-project.yaml.
    Tries to auto-detect project root if project_path_str is None by looking for
    'configs/paper-project.yaml' in current or parent directories.
    """
    project_root: Path
    if project_path_str:
        project_root = Path(project_path_str).resolve()
        if not project_root.is_dir():
            typer.secho(
                f"Error: Provided project path '{project_root}' is not a directory.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
    else:
        # Auto-detection logic
        current_dir = Path.cwd()
        project_root_found = None
        # Check current dir then up to 5 parent levels (arbitrary limit to prevent excessive searching)
        for p_dir in [current_dir] + list(current_dir.parents)[:5]:
            if (p_dir / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME).exists():
                project_root_found = p_dir
                break
        if not project_root_found:
            typer.secho(
                f"Error: Could not auto-detect project root (looking for '{CONFIGS_DIR_NAME}/{DEFAULT_PROJECT_CONFIG_NAME}'). "
                "Please run from within a project or use --project-path.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)
        project_root = project_root_found
        typer.secho(f"Auto-detected project root: {project_root}", fg=typer.colors.BLUE)

    main_config_path = project_root / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME
    if not main_config_path.exists():
        typer.secho(
            f"Error: Main project config '{DEFAULT_PROJECT_CONFIG_NAME}' not found in '{project_root / CONFIGS_DIR_NAME}'. "
            "Is this a valid P.A.P.E.R project directory? Did you run `paper init`?",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        with open(main_config_path) as f:
            project_config = yaml.safe_load(f)
        if project_config is None:  # Handle empty YAML file
            project_config = {}
            logger.warning(f"Main project config '{main_config_path}' is empty.")
    except Exception as e:
        typer.secho(
            f"Error loading or parsing main project config '{main_config_path}': {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    return project_root, project_config


# --- `execute` Command Group  ---
execute_app = typer.Typer(
    name="execute",
    help="Execute P.A.P.E.R project phases.",
    no_args_is_help=True,
)
app.add_typer(execute_app)


def _execute_phase_runner(
    phase_name: str,
    project_path_str: str | None,
    is_available: bool,
    manager_class: Type[Any] | None,
    config_loader_fn: Callable | None,
    default_config_filename: str,
    install_hint: str,
):
    """
    A generic runner for executing a project phase (data, models, portfolio).

    This function encapsulates the common logic for:
    - Checking if the required component is installed.
    - Setting up project-specific logging.
    - Finding and loading the component's configuration file.
    - Instantiating the component's Manager class.
    - Running the manager's main execution method.
    - Handling common exceptions and reporting errors.

    Args:
        phase_name: The name of the phase (e.g., "data", "models").
        project_path_str: The project path provided via CLI option.
        is_available: A boolean flag indicating if the component is installed.
        manager_class: The Manager class for the component (e.g., DataManager).
        config_loader_fn: An optional function to load the component's config.
                          If None, the manager is assumed to take the config path directly.
        default_config_filename: The default name for the component's config file.
        install_hint: A help string for how to install the missing component.
    """
    if not is_available or manager_class is None:
        typer.secho(
            f"Error: The 'paper-{phase_name}' component is not installed or importable. "
            f"Please install it, e.g., {install_hint}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    typer.secho(
        f">>> Executing {phase_name.capitalize()} Phase <<<",
        fg=typer.colors.CYAN,
        bold=True,
    )
    try:
        project_root, project_config = _get_project_root_and_load_main_config(
            project_path_str
        )
    except typer.Exit:
        raise

    log_file_name = project_config.get("logging", {}).get("log_file", LOG_FILE_NAME)
    log_level = project_config.get("logging", {}).get("level", "INFO")
    _configure_project_logging(project_root, log_file_name, log_level)

    log_file_path = project_root / log_file_name
    typer.secho(f"Logging details to: {log_file_path.resolve()}", fg=typer.colors.BLUE)

    logger.info(
        f"Starting {phase_name.capitalize()} Phase for project: {project_root.name}"
    )
    logger.info(f"Project root: {project_root}")

    component_config_filename = (
        project_config.get("components", {})
        .get(phase_name, {})
        .get("config_file", default_config_filename)
    )
    component_config_path = project_root / CONFIGS_DIR_NAME / component_config_filename

    if not component_config_path.exists():
        msg = f"{phase_name.capitalize()} component config file '{component_config_path.name}' not found in '{project_root / CONFIGS_DIR_NAME}'."
        logger.error(msg)
        typer.secho(
            f"Error: {msg} Check logs for details: '{log_file_path}'",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    logger.info(f"Using {phase_name} configuration: {component_config_path}")

    try:
        # Instantiate manager based on whether a config loader is provided
        if config_loader_fn:
            component_config = config_loader_fn(config_path=component_config_path)
            manager = manager_class(config=component_config)
        else:
            # This branch is now deprecated but kept for potential future use
            # where a manager might not have a separate config loader.
            manager = manager_class(config_path=component_config_path)

        # Run the main logic
        manager.run(project_root=project_root)

        typer.secho(
            f"{phase_name.capitalize()} phase completed successfully. Additional information in '{log_file_path}'",
            fg=typer.colors.GREEN,
        )
        logger.info(f"{phase_name.capitalize()} phase completed successfully.")

    except (FileNotFoundError, ValueError, NotImplementedError) as e:
        error_type_map = {
            FileNotFoundError: "A required file was not found.",
            ValueError: "Configuration issue.",
            NotImplementedError: "A feature is not yet implemented.",
        }
        error_message = error_type_map.get(type(e), "An unexpected error occurred.")
        logger.error(f"{error_message} during {phase_name} phase: {e}", exc_info=True)
        typer.secho(
            f"Error: {error_message} Check logs for details: '{log_file_path}'",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception(
            f"An unexpected error occurred while running the {phase_name} phase: {e}"
        )
        typer.secho(
            f"An unexpected error occurred during the {phase_name} phase. Check logs for details: '{log_file_path}'",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


@execute_app.command("data")
def execute_data_phase(
    project_path: str = typer.Option(
        None,
        "--project-path",
        "-p",
        help="Path to the P.A.P.E.R project root directory. If not provided, tries to auto-detect.",
        show_default=False,
    ),
):
    """
    Executes the data processing phase using the 'paper-data' component.
    """
    _execute_phase_runner(
        phase_name="data",
        project_path_str=project_path,
        is_available=PAPER_DATA_AVAILABLE,
        manager_class=DataManager,
        config_loader_fn=load_data_config,
        default_config_filename=DATA_COMPONENT_CONFIG_FILENAME,
        install_hint="`pip install paper-asset-pricing[data]` or `pip install paper-data`",
    )


@execute_app.command("models")
def execute_models_phase(
    project_path: str = typer.Option(
        None,
        "--project-path",
        "-p",
        help="Path to the P.A.P.E.R project root directory. If not provided, tries to auto-detect.",
        show_default=False,
    ),
):
    """
    Executes the modeling phase using the 'paper-model' component.
    """
    _execute_phase_runner(
        phase_name="models",
        project_path_str=project_path,
        is_available=PAPER_MODEL_AVAILABLE,
        manager_class=ModelManager,
        config_loader_fn=load_models_config,
        default_config_filename=MODELS_COMPONENT_CONFIG_FILENAME,
        install_hint="`pip install paper-asset-pricing[models]` or `pip install paper-model`",
    )


@execute_app.command("portfolio")
def execute_portfolio_phase(
    project_path: str = typer.Option(
        None,
        "--project-path",
        "-p",
        help="Path to the P.A.P.E.R project root directory. If not provided, tries to auto-detect.",
        show_default=False,
    ),
):
    """
    Executes the portfolio analysis phase using the 'paper-portfolio' component.
    """
    _execute_phase_runner(
        phase_name="portfolio",
        project_path_str=project_path,
        is_available=PAPER_PORTFOLIO_AVAILABLE,
        manager_class=PortfolioManager,
        config_loader_fn=load_portfolio_config,
        default_config_filename=PORTFOLIO_COMPONENT_CONFIG_FILENAME,
        install_hint="`pip install paper-asset-pricing[portfolio]` or `pip install paper-portfolio`",
    )


# --- `publish` Command Group ---
publish_app = typer.Typer(
    name="publish",
    help="Publish a P.A.P.E.R project to a research repository.",
    no_args_is_help=True,
)
app.add_typer(publish_app)

# --- Constants for Publishing ---
EXCLUDED_EXTENSIONS = {
    ".csv",
    ".parquet",
    ".zip",
    ".tar",
    ".gz",
    ".bzip",
    ".7z",
    ".json",
    ".xlsx",
    ".xls",
}
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".venv", ".git"}
EXCLUDED_FILES = {
    ".DS_Store",
    "Thumbs.db",
    ".coverage",
    ".gitignore",
    ".gitkeep",
    LOG_FILE_NAME,
}


def _create_archive(project_root: Path, archive_path: Path) -> int:
    """
    Creates a zip archive of the project, excluding specified data formats,
    directories, and specific filenames.
    Returns the number of files added to the archive.
    """
    import zipfile

    count = 0
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in project_root.rglob("*"):
            if (
                file_path.is_file()
                and file_path.name not in EXCLUDED_FILES
                and file_path.suffix.lower() not in EXCLUDED_EXTENSIONS
                and not any(d in file_path.parts for d in EXCLUDED_DIRS)
            ):
                archive_name = file_path.relative_to(project_root)
                zf.write(file_path, archive_name)
                count += 1
    return count


@publish_app.command("zenodo")
def publish_to_zenodo(
    project_path: str = typer.Option(
        None,
        "--project-path",
        "-p",
        help="Path to the P.A.P.E.R project root. Tries to auto-detect if not provided.",
        show_default=False,
    ),
    sandbox: bool = typer.Option(
        False,
        "--sandbox",
        help="Use the Zenodo Sandbox for testing instead of the production site.",
    ),
):
    """
    Creates a draft publication on Zenodo and uploads the project archive.

    This command handles the technical steps of bundling your project files
    (excluding large data files) and uploading them to Zenodo. It will create
    a new entry in 'draft' mode.

    You must then go to the provided Zenodo URL to review, complete the
    metadata (like authors), and click the final 'Publish' button.
    """
    typer.secho(
        ">>> Creating Zenodo Draft Publication <<<",
        fg=typer.colors.CYAN,
        bold=True,
    )

    # --- 1. Determine API endpoint and get token ---
    if sandbox:
        typer.secho("Using Zenodo Sandbox environment.", fg=typer.colors.YELLOW)
        base_url = "https://sandbox.zenodo.org/api"
        token_url = "https://sandbox.zenodo.org/account/settings/applications/"
    else:
        base_url = "https://zenodo.org/api"
        token_url = "https://zenodo.org/account/settings/applications/"

    token = os.getenv("ZENODO_TOKEN")
    if not token:
        # ... (The token tutorial and prompt logic remains the same) ...
        typer.secho(
            "\n--- How to get a Zenodo API Token ---", fg=typer.colors.YELLOW, bold=True
        )
        typer.secho("To create a draft on Zenodo, a personal access token is required.")
        typer.secho(f"\n1. Go to: {token_url}", fg=typer.colors.CYAN)
        typer.secho(
            "2. Click 'New token', name it, and select the 'deposit:write' scope."
        )
        typer.secho("3. Copy the token and paste it below.")
        typer.secho(
            '\nTo skip this prompt next time, run: export ZENODO_TOKEN="<your_token>"',
            fg=typer.colors.MAGENTA,
        )
        typer.secho(
            "----------------------------------------",
            fg=typer.colors.YELLOW,
            bold=True,
        )
        token = typer.prompt("\nPlease enter your Zenodo API token", hide_input=True)

    headers = {"Authorization": f"Bearer {token}"}

    # --- 2. Load project config and gather basic metadata ---
    try:
        project_root, project_config = _get_project_root_and_load_main_config(
            project_path
        )
    except typer.Exit:
        raise

    # We only need basic metadata. The user will complete the rest online.
    metadata = {
        "title": project_config.get("project_name", project_root.name),
        "description": project_config.get(
            "description", "A P.A.P.E.R. research project."
        ),
        "upload_type": "software",
        "creators": [],  # Start with an empty list
    }

    # --- Prompt for a single, primary author ---
    typer.secho("\nPlease provide information for the primary author.", bold=True)
    name = typer.prompt("Author Name")
    affiliation = typer.prompt(
        f"Affiliation for {name}", default="", show_default=False
    )
    metadata["creators"].append({"name": name, "affiliation": affiliation})

    typer.secho(
        "Additional authors can be added on the Zenodo website.",
        dim=True,
    )

    # --- 3. Create filtered archive ---
    archive_path = project_root.parent / f"{project_root.name}_zenodo_archive.zip"
    typer.secho(f"\nCreating project archive at: {archive_path}", fg=typer.colors.BLUE)

    try:
        file_count = _create_archive(project_root, archive_path)
        if file_count == 0:
            typer.secho(
                "No files found to archive. Aborting.", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(code=1)
        typer.secho(
            f"✓ Successfully added {file_count} files to the archive.",
            fg=typer.colors.GREEN,
        )

        # --- 4. Zenodo API Interaction ---
        typer.secho("\nUploading to Zenodo as a draft...", fg=typer.colors.BLUE)

        # Step 1: Create a new deposition
        r = requests.post(f"{base_url}/deposit/depositions", json={}, headers=headers)
        r.raise_for_status()
        deposition_data = r.json()
        deposition_id = deposition_data["id"]
        bucket_url = deposition_data["links"]["bucket"]
        edit_url = deposition_data["links"][
            "html"
        ]  # Get the URL for the draft's edit page
        typer.secho("✓ Created new draft deposition on Zenodo.", fg=typer.colors.GREEN)

        # Step 2: Upload the archive
        with open(archive_path, "rb") as fp:
            r = requests.put(
                f"{bucket_url}/{archive_path.name}", data=fp, headers=headers
            )
            r.raise_for_status()
        typer.secho("✓ Uploaded project archive.", fg=typer.colors.GREEN)

        # Step 3: Set the basic metadata
        metadata_payload = {"metadata": metadata}
        r = requests.put(
            f"{base_url}/deposit/depositions/{deposition_id}",
            json=metadata_payload,
            headers=headers,
        )
        r.raise_for_status()
        typer.secho(
            "✓ Set basic metadata (title and description).", fg=typer.colors.GREEN
        )

        # --- 5. Final Instructions ---
        typer.secho(
            "\n🎉 Draft created successfully!", bold=True, fg=typer.colors.BRIGHT_GREEN
        )
        typer.secho("\nNext Steps:", bold=True)
        typer.secho("1. Go to the following URL to edit your draft:")
        typer.secho(f"   {edit_url}", fg=typer.colors.CYAN)
        typer.secho("2. Complete the metadata (especially the 'Authors' section).")
        typer.secho(
            "3. Click the 'Publish' button on the Zenodo website to finalize your submission."
        )

    except requests.HTTPError as e:
        typer.secho(
            f"\n❌ A Zenodo API error occurred: {e.response.status_code}",
            fg=typer.colors.RED,
            err=True,
        )
        try:
            typer.secho(f"Error details: {e.response.json()}", err=True)
        except json.JSONDecodeError:
            typer.secho(f"Error details: {e.response.text}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(
            f"\n❌ An unexpected error occurred: {e}", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=1)
    finally:
        # Clean up the temporary archive file
        if archive_path.exists():
            archive_path.unlink()
            typer.secho(
                f"\nCleaned up temporary archive: {archive_path}", fg=typer.colors.BLUE
            )


@app.callback()
def main_callback(ctx: typer.Context):
    """
    Main callback function for the Typer application.

    This function is invoked before any subcommand and can be used to set up
    global context, validate environment, or display a generic help message.
    In this case, it serves as a placeholder to show the application header.

    Args:
        ctx (typer.Context): The Typer context object.
    """
    pass


if __name__ == "__main__":
    app()
