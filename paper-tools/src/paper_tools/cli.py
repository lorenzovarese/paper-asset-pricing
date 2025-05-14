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
                # Optionally, we could render a more complex template here if needed
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


# --- `execute` Command Group  ---
execute_app = typer.Typer(
    name="execute",
    help="Execute P.A.P.E.R project phases.",
    no_args_is_help=True,
)
app.add_typer(execute_app)


@execute_app.command("data")
def execute_data_phase():
    typer.secho(
        "Data phase execution is not yet implemented.",
        fg=typer.colors.YELLOW,
    )
    typer.Exit(code=0)


@execute_app.command("models")
def execute_models_phase():
    typer.secho(
        "Models phase execution is not yet implemented.",
        fg=typer.colors.YELLOW,
    )
    typer.Exit(code=0)


@execute_app.command("portfolio")
def execute_portfolio_phase():
    typer.secho(
        "Portfolio phase execution is not yet implemented.",
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
