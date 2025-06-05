"""
Module: cli.py

This module implements the command-line interface for the P.A.P.E.R Tools package. It provides
commands to initialize a new research project with a standardized directory structure and
to execute various project phases (data processing, modeling, portfolio analysis) by delegating
to component-specific CLI tools. The module relies on Typer for command-line parsing, Jinja2 for
template rendering, and subprocess for invoking external component commands.
"""

import typer
from pathlib import Path
import yaml
import shutil
import datetime
import subprocess
import sys
from jinja2 import Environment, FileSystemLoader  # For template rendering

# Try to get version from __init__ for the config file, fallback if not found
try:
    from . import __version__ as paper_tools_version
except ImportError:
    paper_tools_version = "unknown"

app = typer.Typer(
    name="paper-tools",
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


def _get_project_root() -> Path:
    """
    Retrieve the current working directory path, which is assumed to be the project root.

    Returns:
        Path: The absolute path to the current working directory.
    """
    return Path.cwd()


def _load_project_config(project_root: Path) -> dict | None:
    """
    Load the main project configuration file (paper-project.yaml) from the 'configs' directory.

    Args:
        project_root (Path): The root directory of the P.A.P.E.R project.

    Returns:
        dict | None:
            A dictionary representing the loaded YAML configuration if successful;
            None if the configuration file does not exist or an error occurs during loading.

    Side Effects:
        Prints error messages to stderr if the configuration file is missing or cannot be parsed.

    """
    config_path = project_root / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME
    if not config_path.exists():
        typer.secho(
            f"Error: Project configuration file not found at '{config_path}'.",
            fg=typer.colors.RED,
            err=True,
        )
        typer.secho(
            f"Please ensure you are in a P.A.P.E.R project directory and '{DEFAULT_PROJECT_CONFIG_NAME}' "
            f"exists in '{CONFIGS_DIR_NAME}/'.",
            err=True,
        )
        return None
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        typer.secho(
            f"Error loading project configuration '{config_path}': {e}",
            fg=typer.colors.RED,
            err=True,
        )
        return None


def _check_component_cli_exists(cli_name: str) -> bool:
    """
    Check whether a given CLI tool is available in the system PATH.

    Args:
        cli_name (str): The name of the CLI executable to verify.

    Returns:
        bool: True if the executable is found in the PATH; False otherwise.
    """
    return shutil.which(cli_name) is not None


def _render_template(template_name: str, context: dict, output_path: Path):
    """
    Render a Jinja2 template with the provided context and write the result to a file.

    Args:
        template_name (str): The filename of the template within the TEMPLATE_DIR.
        context (dict): A dictionary of variables to pass into the Jinja2 renderer.
        output_path (Path): The path where the rendered file should be written.

    Raises:
        typer.Exit: If the specified template cannot be found or another rendering error occurs.

    Side Effects:
        Writes the rendered content to 'output_path'.
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

    This command creates a directory structure with standard subdirectories
    (configs, data/raw, data/processed, models/saved, portfolios/results),
    populates it with placeholder configuration files and templates, and
    initializes a log file. If the target directory already exists and is
    non-empty, the --force flag must be used to overwrite its contents.

    Args:
        project_name (str): The desired name of the project directory.
        force (bool): If True and the target directory exists, it will be
            removed (including all contents) and recreated.

    Raises:
        typer.Exit: If a file with the given name exists, if the directory is
            not empty and force is not specified, or if errors occur during
            directory creation or template rendering.

    Side Effects:
        - Creates directories and .gitkeep files in each empty subdirectory.
        - Renders and writes 'paper-project.yaml', '.gitignore', and 'README.md'
          from templates.
        - Creates placeholder component YAML files in the 'configs' directory.
        - Prints status messages to stdout/stderr.
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
        elif force:
            typer.secho(
                f"Info: Project directory '{project_path.name}' exists but is empty. Proceeding.",
                fg=typer.colors.BLUE,
            )

    typer.secho(
        f"Initializing P.A.P.E.R project '{project_path.name}' at: {project_path.parent}",
        bold=True,
    )

    try:
        # 1. Create project root directory
        project_path.mkdir(parents=True, exist_ok=True)

        # 2. Define directory structure (paths only, creation happens next)
        # These are relative to project_path
        dir_structure_map = {
            CONFIGS_DIR_NAME: [],
            DATA_DIR_NAME: ["raw", "processed"],
            MODELS_DIR_NAME: ["saved"],
            PORTFOLIOS_DIR_NAME: ["results"],
        }

        # Create all directories first
        all_dirs_to_create_paths: list[Path] = []
        for main_dir_name, sub_dir_names in dir_structure_map.items():
            base_path = project_path / main_dir_name
            all_dirs_to_create_paths.append(base_path)
            for sub_dir_name in sub_dir_names:
                all_dirs_to_create_paths.append(base_path / sub_dir_name)

        for dir_p in all_dirs_to_create_paths:
            dir_p.mkdir(parents=True, exist_ok=True)  # parents=True is good for subdirs
        typer.secho("✓ Created project directories.", fg=typer.colors.GREEN)

        # 3. Prepare context for template rendering
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

        # 4. Render and write the main project configuration file
        main_config_output_path = (
            project_path / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME
        )
        _render_template(
            "paper_project.yaml.template",
            template_context,
            main_config_output_path,
        )
        typer.secho(
            f"✓ Created main project config: {main_config_output_path.relative_to(Path.cwd())}",
            fg=typer.colors.GREEN,
        )

        # 5. Render and write .gitignore from template
        gitignore_output_path = project_path / ".gitignore"
        _render_template("gitignore.template", template_context, gitignore_output_path)
        typer.secho("✓ Created .gitignore file.", fg=typer.colors.GREEN)

        # 6. Render and write README.md from template
        readme_output_path = project_path / "README.md"
        _render_template(
            "project_readme.md.template",
            template_context,
            readme_output_path,
        )
        typer.secho("✓ Created project README.md.", fg=typer.colors.GREEN)

        # 7. Create a root log file with an initialization header
        log_file_path = project_path / LOG_FILE_NAME
        with open(log_file_path, "w") as f:
            f.write(
                f"# P.A.P.E.R Project Log for '{project_path.name}' - Initialized: "
                f"{datetime.datetime.now().isoformat()}\n"
            )
        typer.secho(
            f"✓ Created log file: {log_file_path.relative_to(Path.cwd())}",
            fg=typer.colors.GREEN,
        )

        # 8. Create placeholder component configuration files in 'configs'
        for conf_filename in [
            DATA_COMPONENT_CONFIG_FILENAME,
            MODELS_COMPONENT_CONFIG_FILENAME,
            PORTFOLIO_COMPONENT_CONFIG_FILENAME,
        ]:
            placeholder_conf_path = project_path / CONFIGS_DIR_NAME / conf_filename
            with open(placeholder_conf_path, "w") as f:
                f.write(f"# Placeholder for {conf_filename}\n")
                f.write(
                    "# Please refer to the respective component's documentation for structure.\n"
                )
                if conf_filename == DATA_COMPONENT_CONFIG_FILENAME:
                    f.write(
                        f"""\
# Example structure for {DATA_COMPONENT_CONFIG_FILENAME}:
# sources:
#   - name: my_data
#     connector: local
#     path: "{DATA_DIR_NAME}/raw/your_data.csv"
# transformations:
#   # - type: ...
# output:
#   format: parquet
#   # ...
"""
                    )
            typer.secho(
                f"✓ Created placeholder component config: "
                f"{placeholder_conf_path.relative_to(Path.cwd())}",
                fg=typer.colors.BLUE,
            )

        # 9. Add .gitkeep files to empty directories *AFTER* all other files are created
        # Iterate through the same list of directories that were created.
        # all_dirs_to_create_paths was defined in step 2.
        for dir_p in all_dirs_to_create_paths:
            # Check if the directory is empty (no files or subdirectories other than potentially .gitkeep itself)
            is_empty = True
            for item in dir_p.iterdir():
                if item.name != ".gitkeep":
                    is_empty = False
                    break
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
            f"     (e.g., '{DATA_COMPONENT_CONFIG_FILENAME}', "
            f"'{MODELS_COMPONENT_CONFIG_FILENAME}', '{PORTFOLIO_COMPONENT_CONFIG_FILENAME}')",
            fg=typer.colors.CYAN,
        )
        typer.secho(
            f"  2. Place raw data in '{DATA_DIR_NAME}/raw/'.", fg=typer.colors.CYAN
        )
        typer.secho(
            "  3. Run phases using `paper-tools execute <phase>`.", fg=typer.colors.CYAN
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


# --- `execute` Command Group (Portfolio command simplified) ---
execute_app = typer.Typer(
    name="execute",
    help="Execute P.A.P.E.R project phases.",
    no_args_is_help=True,
)
app.add_typer(execute_app)


@execute_app.command("data")
def execute_data_phase(ctx: typer.Context):
    """
    Execute the data processing phase using the configured 'paper-data' component.

    This command:
      1. Locates and loads the main project configuration (paper-project.yaml).
      2. Retrieves the 'components.data' section to find the CLI tool name, command, and
         relative path to the data component's configuration file.
      3. Resolves the absolute path of the component config file.
      4. Invokes the external component CLI using _run_component_cli.

    Args:
        ctx (typer.Context): The Typer context (unused but required by Typer signature).

    Raises:
        typer.Exit: If the project configuration is missing, if the 'components.data'
            section is absent or malformed, or if required fields are not specified.
    """
    project_root = _get_project_root()
    project_config = _load_project_config(project_root)
    if not project_config:
        raise typer.Exit(code=1)

    data_cfg = project_config.get("components", {}).get("data")
    if not data_cfg:
        typer.secho(
            "Error: 'components.data' configuration missing in 'paper-project.yaml'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    config_file_rel_path = data_cfg.get("config_file")
    cli_tool_name = data_cfg.get("cli_tool", "paper-data")
    cli_main_command = data_cfg.get(
        "cli_command", "process"
    )  # Allow overriding 'process'

    if not config_file_rel_path:
        typer.secho(
            "Error: 'components.data.config_file' path missing in 'paper-project.yaml'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    component_config_abs_path = (project_root / config_file_rel_path).resolve()
    _run_component_cli(
        cli_tool_name, cli_main_command, component_config_abs_path, project_root
    )


@execute_app.command("models")
def execute_models_phase(ctx: typer.Context):
    """
    Execute the modeling phase using the configured 'paper-model' component.

    This command:
      1. Loads the main project configuration.
      2. Retrieves the 'components.models' section for CLI details.
      3. Resolves the model component's configuration file path.
      4. Invokes the external modeling CLI via _run_component_cli.

    Args:
        ctx (typer.Context): The Typer context (unused but required by Typer signature).

    Raises:
        typer.Exit: If the project configuration or required 'components.models'
            entry or config_file path is missing.
    """
    project_root = _get_project_root()
    project_config = _load_project_config(project_root)
    if not project_config:
        raise typer.Exit(code=1)

    models_cfg = project_config.get("components", {}).get("models")
    if not models_cfg:
        typer.secho(
            "Error: 'components.models' configuration missing in 'paper-project.yaml'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    config_file_rel_path = models_cfg.get("config_file")
    cli_tool_name = models_cfg.get("cli_tool", "paper-model")
    cli_main_command = models_cfg.get("cli_command", "process")

    if not config_file_rel_path:
        typer.secho(
            "Error: 'components.models.config_file' path missing in 'paper-project.yaml'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    component_config_abs_path = (project_root / config_file_rel_path).resolve()
    _run_component_cli(
        cli_tool_name, cli_main_command, component_config_abs_path, project_root
    )


@execute_app.command("portfolio")
def execute_portfolio_phase(ctx: typer.Context):
    """
    Execute the portfolio analysis phase using the configured 'paper-portfolio' component.

    This command:
      1. Loads the main project configuration.
      2. Retrieves the 'components.portfolio' section for CLI details.
      3. Resolves the portfolio component's configuration file path.
      4. Invokes the external portfolio CLI via _run_component_cli.

    Args:
        ctx (typer.Context): The Typer context (unused but required by Typer signature).

    Raises:
        typer.Exit: If the project configuration or required 'components.portfolio'
            entry or config_file path is missing.
    """
    project_root = _get_project_root()
    project_config = _load_project_config(project_root)
    if not project_config:
        raise typer.Exit(code=1)

    portfolio_cfg = project_config.get("components", {}).get("portfolio")
    if not portfolio_cfg:
        typer.secho(
            "Error: 'components.portfolio' configuration missing in 'paper-project.yaml'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    config_file_rel_path = portfolio_cfg.get("config_file")
    cli_tool_name = portfolio_cfg.get("cli_tool", "paper-portfolio")
    cli_main_command = portfolio_cfg.get("cli_command", "process")

    if not config_file_rel_path:
        typer.secho(
            "Error: 'components.portfolio.config_file' path missing in 'paper-project.yaml'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    component_config_abs_path = (project_root / config_file_rel_path).resolve()
    _run_component_cli(
        cli_tool_name, cli_main_command, component_config_abs_path, project_root
    )


def _run_component_cli(
    component_cli_name: str,
    component_main_command: str,
    component_config_file: Path,
    project_root: Path,
):
    """
    Invoke an external component CLI tool with a specified main command and configuration.

    Args:
        component_cli_name (str): The name of the component CLI executable.
        component_main_command (str): The subcommand or action to pass to the CLI.
        component_config_file (Path): The path to the component's YAML configuration file.
        project_root (Path): The root directory of the P.A.P.E.R project.

    Raises:
        typer.Exit: If the CLI executable is not found or returns a non-zero exit status.

    Side Effects:
        - Prints a warning if the configuration file does not exist.
        - Constructs and prints the command to be executed.
        - Executes the command via subprocess.
        - Prints stdout, stderr, and success/failure messages accordingly.
    """
    if not _check_component_cli_exists(component_cli_name):
        typer.secho(
            f"Error: Component CLI '{component_cli_name}' not found in PATH.",
            fg=typer.colors.RED,
            err=True,
        )
        typer.secho(
            f"Please ensure '{component_cli_name}' (from package like {component_cli_name.replace('-', '_')}) "
            f"is installed and accessible.",
            err=True,
        )
        raise typer.Exit(code=1)

    if not component_config_file.exists():
        typer.secho(
            f"Warning: Component configuration file '{component_config_file}' not found.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        typer.secho(
            f"  The component '{component_cli_name}' might fail or use default behavior if its config is optional.",
            fg=typer.colors.YELLOW,
            err=True,
        )

    cmd = [
        component_cli_name,
        component_main_command,
        "--config",
        str(component_config_file),
        "--project-root",
        str(project_root),
    ]

    typer.secho(f"Executing: {' '.join(cmd)}", fg=typer.colors.BLUE)
    try:
        result = subprocess.run(
            cmd, cwd=project_root, check=True, text=True, capture_output=True
        )
        if result.stdout:
            typer.echo(result.stdout)
        if result.stderr:
            typer.secho("Component output (stderr):", fg=typer.colors.YELLOW, err=True)
            typer.echo(result.stderr, err=True)
        typer.secho(
            f"'{component_cli_name} {component_main_command}' executed successfully.",
            fg=typer.colors.GREEN,
        )
    except subprocess.CalledProcessError as e:
        typer.secho(
            f"Error executing '{component_cli_name} {component_main_command}': "
            f"Command returned non-zero exit status {e.returncode}.",
            fg=typer.colors.RED,
            err=True,
        )
        if e.stdout:
            typer.secho("Stdout:", fg=typer.colors.YELLOW, err=True)
            typer.echo(e.stdout, err=True)
        if e.stderr:
            typer.secho("Stderr:", fg=typer.colors.YELLOW, err=True)
            typer.echo(e.stderr, err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(
            f"An unexpected error occurred while running '{component_cli_name} {component_main_command}': {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


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
