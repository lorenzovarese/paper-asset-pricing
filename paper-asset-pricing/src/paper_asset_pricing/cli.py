"""
Module: cli.py

This module implements the command-line interface for the P.A.P.E.R Tools package. It provides
commands to initialize a new research project with a standardized directory structure and
to execute various project phases (data processing, modeling, portfolio analysis).
"""

import typer
from pathlib import Path
import yaml
import shutil
import datetime
import sys
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
            DATA_DIR_NAME: ["raw", "processed"],
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
            "paper_project.yaml.template", template_context, main_config_output_path
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

        for conf_filename in [
            DATA_COMPONENT_CONFIG_FILENAME,
            MODELS_COMPONENT_CONFIG_FILENAME,
            PORTFOLIO_COMPONENT_CONFIG_FILENAME,
        ]:
            placeholder_conf_path = project_path / CONFIGS_DIR_NAME / conf_filename
            with open(placeholder_conf_path, "w") as f:
                f.write(
                    f"# Placeholder for {conf_filename}\n# Please refer to the respective component's documentation for structure.\n"
                )
            typer.secho(
                f"✓ Created placeholder component config: {placeholder_conf_path.relative_to(Path.cwd())}",
                fg=typer.colors.BLUE,
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
            f"  2. Place raw data in '{DATA_DIR_NAME}/raw/'.", fg=typer.colors.CYAN
        )
        typer.secho(
            "  3. Run phases using `paper execute <phase>`.",
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
        with open(main_config_path, "r") as f:
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
