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

# --- Configure logging for paper-tools ---
# This helps see logs from both paper-tools and imported libraries like paper-data
# if they also use logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # Send logs to stdout
)
logger = logging.getLogger(__name__)


# --- Try to import from sub-packages ---
# This allows paper-tools to function (e.g., `init`) even if optional components aren't installed.
try:
    from paper_data.main import run_data_pipeline_from_config

    PAPER_DATA_AVAILABLE = True
    logger.debug("paper_data.main.run_data_pipeline_from_config imported successfully.")
except ImportError:
    PAPER_DATA_AVAILABLE = False
    run_data_pipeline_from_config = None  # type: ignore # Make linters happy
    logger.debug(
        "Failed to import paper_data.main.run_data_pipeline_from_config. PAPER_DATA_AVAILABLE=False"
    )


# Try to get version from __init__ for the config file, fallback if not found
try:
    from . import __version__ as paper_tools_version
except ImportError:
    paper_tools_version = "unknown"

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
            MODELS_DIR_NAME: ["saved"],
            PORTFOLIOS_DIR_NAME: ["results"],
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
            "paper_tools_version": paper_tools_version,
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
    if not PAPER_DATA_AVAILABLE or run_data_pipeline_from_config is None:
        typer.secho(
            "Error: The 'paper-data' component is not installed or importable. "
            "Please install it, e.g., `pip install paper-tools[data]` or `pip install paper-data`.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    typer.secho(">>> Executing Data Phase <<<", fg=typer.colors.CYAN, bold=True)

    try:
        project_root, project_config = _get_project_root_and_load_main_config(
            project_path
        )
    except typer.Exit:
        raise  # Propagate exit if helper failed

    # Determine the data component's config file path
    # For now, we assume it's the default name. A more advanced setup might get this
    # from project_config.
    data_config_filename = (
        project_config.get("components", {})
        .get("data", {})
        .get(
            "config_file",
            DATA_COMPONENT_CONFIG_FILENAME,  # Fallback to default constant
        )
    )
    component_config_path = project_root / CONFIGS_DIR_NAME / data_config_filename

    if not component_config_path.exists():
        typer.secho(
            f"Error: Data component config file '{component_config_path.name}' "
            f"not found in '{project_root / CONFIGS_DIR_NAME}'. "
            f"Ensure '{DATA_COMPONENT_CONFIG_FILENAME}' exists after running `paper init`.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    logger.info(f"Using project root: {project_root}")
    logger.info(f"Using data configuration: {component_config_path}")

    try:
        result = run_data_pipeline_from_config(
            config_path=component_config_path, project_root_path=project_root
        )

        if result and result.get("status") == "success":
            typer.secho(f"Data phase completed successfully.", fg=typer.colors.GREEN)
            if "output_path" in result and result["output_path"]:
                typer.secho(
                    f"Output generated at: {result['output_path']}",
                    fg=typer.colors.GREEN,
                )
            else:
                typer.secho(
                    "No output path reported by data phase.", fg=typer.colors.YELLOW
                )
        else:
            error_message = "Unknown error in paper-data."
            if result and "message" in result:
                error_message = result["message"]
            typer.secho(
                f"Data phase failed: {error_message}", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(code=1)

    except Exception as e:
        typer.secho(
            f"An unexpected error occurred while running the data phase: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        import traceback

        traceback.print_exc(file=sys.stderr)
        raise typer.Exit(code=1)


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
    typer.secho(
        "Models phase execution is not yet implemented. (Would import from 'paper-model')",
        fg=typer.colors.YELLOW,
    )
    typer.Exit(code=0)


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
    typer.secho(
        "Portfolio phase execution is not yet implemented. (Would import from 'paper-portfolio')",
        fg=typer.colors.YELLOW,
    )
    typer.Exit(code=0)


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
