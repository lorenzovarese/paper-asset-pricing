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

# Path to the templates directory within the package
# This assumes cli.py is in src/paper_tools/ and templates/ is a sibling
TEMPLATE_DIR = Path(__file__).parent / "templates"


# --- Helper Functions (existing _get_project_root, _load_project_config, etc. remain the same) ---
def _get_project_root() -> Path:
    return Path.cwd()


def _load_project_config(project_root: Path) -> dict | None:
    config_path = project_root / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME
    if not config_path.exists():
        typer.secho(
            f"Error: Project configuration file not found at '{config_path}'.",
            fg=typer.colors.RED,
            err=True,
        )
        typer.secho(
            f"Please ensure you are in a P.A.P.E.R project directory and '{DEFAULT_PROJECT_CONFIG_NAME}' exists in '{CONFIGS_DIR_NAME}/'.",
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
    return shutil.which(cli_name) is not None


def _run_component_cli(
    component_cli_name: str,
    component_config_file: Path,
    project_root: Path,
):
    if not _check_component_cli_exists(component_cli_name):
        typer.secho(
            f"Error: Component CLI '{component_cli_name}' not found in PATH.",
            fg=typer.colors.RED,
            err=True,
        )
        typer.secho(
            f"Please ensure '{component_cli_name}' (from package like {component_cli_name.replace('-', '_')}) is installed and accessible.",
            err=True,
        )
        raise typer.Exit(code=1)

    if not component_config_file.exists():
        # This is now a warning, as the user is expected to create these.
        typer.secho(
            f"Warning: Component configuration file '{component_config_file}' not found.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        typer.secho(
            f"  The component '{component_cli_name}' might fail or use default behavior.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        # We still try to run the component; it's up to the component to handle missing config.
        # Or, we could make this an error:
        # raise typer.Exit(code=1)

    cmd = [
        component_cli_name,
        "process",  # Main command assumption
        "--config",
        str(component_config_file),
        "--project-root",
        str(project_root),
    ]
    # if additional_args: # Removed
    #     cmd.extend(additional_args)

    typer.secho(f"Executing: {' '.join(cmd)}", fg=typer.colors.BLUE)
    try:
        result = subprocess.run(
            cmd, cwd=project_root, check=True, text=True, capture_output=True
        )
        if result.stdout:
            typer.echo(result.stdout)
        if result.stderr:
            typer.secho(
                result.stderr, fg=typer.colors.YELLOW, err=True
            )  # check=True means stderr is not for errors
        typer.secho(
            f"'{component_cli_name}' executed successfully.", fg=typer.colors.GREEN
        )
    except subprocess.CalledProcessError as e:
        typer.secho(
            f"Error executing '{component_cli_name}': Command returned non-zero exit status {e.returncode}.",
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
            f"An unexpected error occurred while running '{component_cli_name}': {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)


def _render_template(template_name: str, context: dict, output_path: Path):
    """Renders a Jinja2 template and writes it to the output path."""
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    try:
        template = env.get_template(template_name)
    except (
        Exception
    ) as e:  # Catch specific jinja2.exceptions.TemplateNotFound if preferred
        typer.secho(
            f"Error: Template '{template_name}' not found in '{TEMPLATE_DIR}'. Error: {e}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    rendered_content = template.render(context)
    with open(output_path, "w") as f:
        f.write(rendered_content)


# --- `init` Command (Modified) ---
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
    Initializes a new P.A.P.E.R project with a standard directory
    structure and configuration files from templates.
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
        if any(project_path.iterdir()):  # Check if directory is not empty
            if force:
                typer.secho(
                    f"Warning: Project directory '{project_path.name}' exists and is not empty. Overwriting due to --force.",
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
                    f"Error: Project directory '{project_path.name}' already exists and is not empty. Use --force or choose a different name.",
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
        # 1. Create project root
        project_path.mkdir(parents=True, exist_ok=True)

        # 2. Create standard directories
        dirs_to_create = {
            CONFIGS_DIR_NAME: [],
            DATA_DIR_NAME: ["raw", "processed"],
            MODELS_DIR_NAME: ["saved"],
            PORTFOLIOS_DIR_NAME: ["results"],
        }
        for main_dir, sub_dirs in dirs_to_create.items():
            base_path = project_path / main_dir
            base_path.mkdir(exist_ok=True)
            (base_path / ".gitkeep").touch()
            for sub_dir in sub_dirs:
                sub_path = base_path / sub_dir
                sub_path.mkdir(exist_ok=True)
                (sub_path / ".gitkeep").touch()
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

        # 4. Create main project configuration file from template
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

        # 5. Create .gitignore from template
        gitignore_output_path = project_path / ".gitignore"
        _render_template("gitignore.template", template_context, gitignore_output_path)
        typer.secho("✓ Created .gitignore file.", fg=typer.colors.GREEN)

        # 6. Create README.md from template
        readme_output_path = project_path / "README.md"
        _render_template(
            "project_readme.md.template", template_context, readme_output_path
        )
        typer.secho("✓ Created project README.md.", fg=typer.colors.GREEN)

        # 7. Create root log file (simple, no template needed)
        log_file_path = project_path / LOG_FILE_NAME
        with open(log_file_path, "w") as f:
            f.write(
                f"# P.A.P.E.R Project Log for '{project_path.name}' - Initialized: {datetime.datetime.now().isoformat()}\n"
            )
        typer.secho(
            f"✓ Created log file: {log_file_path.relative_to(Path.cwd())}",
            fg=typer.colors.GREEN,
        )

        # 8. Create empty placeholder component config files (optional, but helpful)
        #    User is still responsible for filling them.
        for conf_filename in [
            DATA_COMPONENT_CONFIG_FILENAME,
            MODELS_COMPONENT_CONFIG_FILENAME,
            PORTFOLIO_COMPONENT_CONFIG_FILENAME,
        ]:
            placeholder_conf_path = project_path / CONFIGS_DIR_NAME / conf_filename
            with open(placeholder_conf_path, "w") as f:
                f.write(f"# Placeholder for {conf_filename}\n")
                f.write(
                    f"# Please refer to the respective component's documentation for structure.\n"
                )
                if conf_filename == DATA_COMPONENT_CONFIG_FILENAME:
                    f.write(f"""\
# Example structure for {DATA_COMPONENT_CONFIG_FILENAME}:
# sources:
#   - name: my_data
#     connector: local
#     path: "{DATA_DIR_NAME}/raw/your_data.csv"
#     # ...
# transformations:
#   # - type: ...
# output:
#   format: parquet
#   # ...
""")
            typer.secho(
                f"✓ Created placeholder component config: {placeholder_conf_path.relative_to(Path.cwd())}",
                fg=typer.colors.BLUE,
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
    name="execute", help="Execute P.A.P.E.R project phases.", no_args_is_help=True
)
app.add_typer(execute_app)


@execute_app.command("data")
def execute_data_phase(ctx: typer.Context):
    """Executes the data processing phase using the configured 'paper-data' component."""
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
    # Modify _run_component_cli to accept cli_main_command
    _run_component_cli_v2(
        cli_tool_name, cli_main_command, component_config_abs_path, project_root
    )


@execute_app.command("models")
def execute_models_phase(ctx: typer.Context):
    """Executes the modeling phase using the configured 'paper-model' component."""
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
    _run_component_cli_v2(
        cli_tool_name, cli_main_command, component_config_abs_path, project_root
    )


@execute_app.command("portfolio")
def execute_portfolio_phase(ctx: typer.Context):  # Removed override_config
    """Executes the portfolio phase using the configured 'paper-portfolio' component."""
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
    _run_component_cli_v2(
        cli_tool_name, cli_main_command, component_config_abs_path, project_root
    )  # No additional_args


# Updated _run_component_cli to accept main_command
def _run_component_cli_v2(
    component_cli_name: str,
    component_main_command: str,  # New argument
    component_config_file: Path,
    project_root: Path,
):
    if not _check_component_cli_exists(component_cli_name):
        typer.secho(
            f"Error: Component CLI '{component_cli_name}' not found in PATH.",
            fg=typer.colors.RED,
            err=True,
        )
        typer.secho(
            f"Please ensure '{component_cli_name}' (from package like {component_cli_name.replace('-', '_')}) is installed and accessible.",
            err=True,
        )
        raise typer.Exit(code=1)

    # Warning for missing config, but attempt to run anyway
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
        component_main_command,  # Use the specified main command
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
        # For check=True, stderr is usually for non-error informational messages from the tool
        if result.stderr:
            typer.secho("Component output (stderr):", fg=typer.colors.YELLOW, err=True)
            typer.echo(result.stderr, err=True)
        typer.secho(
            f"'{component_cli_name} {component_main_command}' executed successfully.",
            fg=typer.colors.GREEN,
        )
    except subprocess.CalledProcessError as e:
        typer.secho(
            f"Error executing '{component_cli_name} {component_main_command}': Command returned non-zero exit status {e.returncode}.",
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
    P.A.P.E.R Tools: Initialize and Execute research project phases.
    Run `paper-tools init --help` or `paper-tools execute --help` for more information.
    """
    pass


if __name__ == "__main__":
    app()
