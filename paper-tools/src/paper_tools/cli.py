import typer
from pathlib import Path
import yaml
import shutil
import datetime
import subprocess
import sys
import importlib.util

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
LOG_FILE_NAME = "logs.log"  # Root level log file

DEFAULT_PROJECT_CONFIG_NAME = "paper-project.yaml"  # Main project config

# Component-specific config file names (to be placed inside CONFIGS_DIR_NAME)
DATA_COMPONENT_CONFIG_FILENAME = "data-config.yaml"
MODELS_COMPONENT_CONFIG_FILENAME = "models-config.yaml"
PORTFOLIO_COMPONENT_CONFIG_FILENAME = "portfolio-config.yaml"


# --- Helper Functions ---


def _get_project_root() -> Path:
    """Determines the project root, assuming the main config is there."""
    # A common pattern is to look for a marker file, like paper-project.yaml
    # For simplicity now, we assume CWD is project root when 'execute' is called.
    # More robust: search upwards from CWD for paper-project.yaml
    return Path.cwd()


def _load_project_config(project_root: Path) -> dict | None:
    """Loads the main project configuration file."""
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
    """Checks if a component's CLI tool is available in PATH."""
    return shutil.which(cli_name) is not None


def _check_component_installed(module_name: str) -> bool:
    """Checks if a Python module (component) is installed."""
    spec = importlib.util.find_spec(module_name)
    return spec is not None


def _run_component_cli(
    component_cli_name: str,
    component_config_file: Path,
    project_root: Path,
    additional_args: list[str] | None = None,
):
    """Executes a component's CLI tool."""
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
        typer.secho(
            f"Error: Component configuration file '{component_config_file}' not found.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    cmd = [
        component_cli_name,
        # Assuming a common pattern like 'process' or 'run' for the main command
        # This might need to be configurable per component in paper-project.yaml
        "process",  # Or "run", "execute" - this is a key assumption about component CLIs
        "--config",
        str(component_config_file),
        "--project-root",
        str(project_root),  # Pass project root for context
    ]
    if additional_args:
        cmd.extend(additional_args)

    typer.secho(f"Executing: {' '.join(cmd)}", fg=typer.colors.BLUE)
    try:
        # Run from project root so relative paths in configs work as expected
        result = subprocess.run(cmd, cwd=project_root, check=True, text=True)
        if result.stdout:
            typer.echo(result.stdout)
        if result.stderr:  # Should be empty on success with check=True
            typer.secho(result.stderr, fg=typer.colors.YELLOW, err=True)
        typer.secho(
            f"'{component_cli_name}' executed successfully.", fg=typer.colors.GREEN
        )
    except FileNotFoundError:  # Should be caught by _check_component_cli_exists
        typer.secho(
            f"Error: CLI command '{component_cli_name}' not found. Is it installed?",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
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


# --- `init` Command ---
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
    structure and a main configuration file.
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
            (base_path / ".gitkeep").touch()  # Keep main phase dirs
            for sub_dir in sub_dirs:
                sub_path = base_path / sub_dir
                sub_path.mkdir(exist_ok=True)
                (sub_path / ".gitkeep").touch()  # Keep sub dirs like raw, processed

        # 3. Create main project configuration file
        main_config_path = project_path / CONFIGS_DIR_NAME / DEFAULT_PROJECT_CONFIG_NAME
        project_cfg_content = {
            "project_name": project_path.name,
            "version": "0.1.0",
            "paper_tools_version": paper_tools_version,
            "creation_date": datetime.date.today().isoformat(),
            "description": f"P.A.P.E.R project: {project_path.name}",
            "components": {
                "data": {
                    "config_file": str(
                        Path(CONFIGS_DIR_NAME) / DATA_COMPONENT_CONFIG_FILENAME
                    ),
                    "raw_dir": str(Path(DATA_DIR_NAME) / "raw"),
                    "processed_dir": str(Path(DATA_DIR_NAME) / "processed"),
                    "cli_tool": "paper-data",  # Assumed CLI name
                },
                "models": {
                    "config_file": str(
                        Path(CONFIGS_DIR_NAME) / MODELS_COMPONENT_CONFIG_FILENAME
                    ),
                    "saved_dir": str(Path(MODELS_DIR_NAME) / "saved"),
                    "cli_tool": "paper-model",  # Assumed CLI name
                },
                "portfolio": {
                    "config_file": str(
                        Path(CONFIGS_DIR_NAME) / PORTFOLIO_COMPONENT_CONFIG_FILENAME
                    ),
                    "results_dir": str(Path(PORTFOLIOS_DIR_NAME) / "results"),
                    "cli_tool": "paper-portfolio",  # Assumed CLI name
                },
            },
            "logging": {
                "log_file": LOG_FILE_NAME,  # Relative to project root
                "level": "INFO",
            },
        }
        with open(main_config_path, "w") as f:
            yaml.dump(project_cfg_content, f, sort_keys=False, indent=2)
        typer.secho(
            f"✓ Created main project config: {main_config_path.relative_to(Path.cwd())}",
            fg=typer.colors.GREEN,
        )

        # 4. Create root log file
        log_file_path = project_path / LOG_FILE_NAME
        with open(log_file_path, "w") as f:
            f.write(
                f"# P.A.P.E.R Project Log for '{project_path.name}' - Initialized: {datetime.datetime.now().isoformat()}\n"
            )
        typer.secho(
            f"✓ Created log file: {log_file_path.relative_to(Path.cwd())}",
            fg=typer.colors.GREEN,
        )

        # 5. Create .gitignore
        gitignore_content = f"""
# Python
__pycache__/
*.py[cod]
.DS_Store

# Virtual environment
.venv/
venv/
ENV/
env/

# Data (typically not versioned, but .gitkeep files are for structure)
/{DATA_DIR_NAME}/raw/*
!/{DATA_DIR_NAME}/raw/.gitkeep
/{DATA_DIR_NAME}/processed/*
!/{DATA_DIR_NAME}/processed/.gitkeep

# Models (saved models typically not versioned)
/{MODELS_DIR_NAME}/saved/*
!/{MODELS_DIR_NAME}/saved/.gitkeep

# Portfolios (results typically not versioned)
/{PORTFOLIOS_DIR_NAME}/results/*
!/{PORTFOLIOS_DIR_NAME}/results/.gitkeep

# Logs
/{LOG_FILE_NAME}
"""
        (project_path / ".gitignore").write_text(gitignore_content.strip())
        typer.secho("✓ Created .gitignore file.", fg=typer.colors.GREEN)

        # 6. Create README.md
        readme_content = f"""
# {project_path.name}

A P.A.P.E.R (Platform for Asset Pricing Experimentation and Research) project.

Initialized on: {datetime.date.today().isoformat()}
P.A.P.E.R Tools Version: {paper_tools_version}

## Project Structure

- `{CONFIGS_DIR_NAME}/{DEFAULT_PROJECT_CONFIG_NAME}`: Main project configuration.
- `{CONFIGS_DIR_NAME}/`: Directory for component-specific YAML configurations:
    - `{DATA_COMPONENT_CONFIG_FILENAME}`: For `paper-data` processing.
    - `{MODELS_COMPONENT_CONFIG_FILENAME}`: For `paper-model` tasks.
    - `{PORTFOLIO_COMPONENT_CONFIG_FILENAME}`: For `paper-portfolio` strategies.
- `{DATA_DIR_NAME}/`: Data storage.
    - `raw/`: Place for raw input data.
    - `processed/`: Output for processed data from `paper-data`.
- `{MODELS_DIR_NAME}/`: Model-related files.
    - `saved/`: Output for trained models from `paper-model`.
- `{PORTFOLIOS_DIR_NAME}/`: Portfolio-related files.
    - `results/`: Output for portfolio backtests/results from `paper-portfolio`.
- `{LOG_FILE_NAME}`: Project-level log file.
- `.gitignore`: Specifies files for Git to ignore.
- `README.md`: This file.

## Getting Started

1.  **Navigate to the project directory:**
    ```bash
    cd "{project_path.name}"
    ```

2.  **Set up your Python environment** and install P.AP.E.R components:
    ```bash
    # Example:
    # uv venv
    # source .venv/bin/activate
    pip install paper-data paper-model paper-portfolio # Or use paper-tools[all]
    ```

3.  **Create Component Configurations:**
    - In the `{CONFIGS_DIR_NAME}/` directory, create and populate:
        - `{DATA_COMPONENT_CONFIG_FILENAME}` (for `paper-data`)
        - `{MODELS_COMPONENT_CONFIG_FILENAME}` (for `paper-model`)
        - `{PORTFOLIO_COMPONENT_CONFIG_FILENAME}` (for `paper-portfolio`)
    - Refer to the documentation of each P.AP.E.R component for its specific YAML structure.

4.  **Place Raw Data:**
    - Put your raw data files into the `{DATA_DIR_NAME}/raw/` directory.

5.  **Execute Project Phases:**
    Use `paper-tools execute` from the project root:
    ```bash
    paper-tools execute data      # Runs the data processing phase
    paper-tools execute models    # Runs the modeling phase
    paper-tools execute portfolio # Runs the portfolio phase
    ```
    You can also run them sequentially:
    ```bash
    paper-tools execute data && paper-tools execute models && paper-tools execute portfolio
    ```

Refer to `{CONFIGS_DIR_NAME}/{DEFAULT_PROJECT_CONFIG_NAME}` to see how `paper-tools` locates component configurations.
"""
        (project_path / "README.md").write_text(readme_content.strip())
        typer.secho("✓ Created project README.md.", fg=typer.colors.GREEN)

        typer.secho(
            f"\n🎉 P.AP.E.R project '{project_path.name}' initialized successfully!",
            bold=True,
            fg=typer.colors.BRIGHT_GREEN,
        )
        typer.secho(
            f'\nNavigate to your project:\n  cd "{project_path.relative_to(Path.cwd())}"',
            fg=typer.colors.CYAN,
        )
        typer.secho("\nNext steps:", fg=typer.colors.CYAN)
        typer.secho(
            f"  1. Create your component-specific YAML configuration files in '{CONFIGS_DIR_NAME}/'.",
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


# --- `execute` Command Group ---
execute_app = typer.Typer(
    name="execute", help="Execute P.AP.E.R project phases.", no_args_is_help=True
)
app.add_typer(execute_app)


@execute_app.command("data")
def execute_data_phase(
    ctx: typer.Context,  # Access parent context if needed
):
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
    cli_tool_name = data_cfg.get("cli_tool", "paper-data")  # Default to paper-data

    if not config_file_rel_path:
        typer.secho(
            "Error: 'components.data.config_file' path missing in 'paper-project.yaml'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    component_config_abs_path = (project_root / config_file_rel_path).resolve()
    _run_component_cli(cli_tool_name, component_config_abs_path, project_root)


@execute_app.command("models")
def execute_models_phase(
    ctx: typer.Context,
):
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

    if not config_file_rel_path:
        typer.secho(
            "Error: 'components.models.config_file' path missing in 'paper-project.yaml'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    component_config_abs_path = (project_root / config_file_rel_path).resolve()
    _run_component_cli(cli_tool_name, component_config_abs_path, project_root)


@execute_app.command("portfolio")
def execute_portfolio_phase(
    ctx: typer.Context,
    override_config: Path = typer.Option(
        None,
        "--override-config",
        help="Path to an additional/override YAML configuration file for the portfolio component.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,  # Resolve to absolute path
    ),
):
    """
    Executes the portfolio phase using the configured 'paper-portfolio' component.
    Allows an optional override configuration file.
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

    if not config_file_rel_path:
        typer.secho(
            "Error: 'components.portfolio.config_file' path missing in 'paper-project.yaml'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    component_config_abs_path = (project_root / config_file_rel_path).resolve()

    additional_args = []
    if override_config:
        # How paper-portfolio CLI handles an override needs to be defined by paper-portfolio.
        # Example: it might take another --config or a specific --override flag.
        # For now, let's assume it can take multiple --config flags or a specific one.
        additional_args.extend(["--additional-config", str(override_config)])
        typer.secho(
            f"Using override portfolio config: {override_config}",
            fg=typer.colors.YELLOW,
        )

    _run_component_cli(
        cli_tool_name,
        component_config_abs_path,
        project_root,
        additional_args=additional_args,
    )


@app.callback()
def main_callback(ctx: typer.Context):
    """
    P.AP.E.R Tools: Initialize and Execute research project phases.
    Run `paper-tools init --help` or `paper-tools execute --help` for more information.
    """
    # You can add global options or context setup here if needed
    pass


if __name__ == "__main__":
    app()
