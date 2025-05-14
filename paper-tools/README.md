# paper-tools

The **paper-tools** package provides a unified CLI for creating and running P.A.P.E.R research projects.

## Installation

```bash
pip install paper-tools
# or with extras:
pip install paper-tools[data]
pip install paper-tools[models]
pip install paper-tools[portfolio]
pip install paper-tools[all]
````

## Commands

* **`paper init <PROJECT_NAME>`**
  Scaffold a new project directory with standard subfolders:

  ```
  <PROJECT_NAME>/
  ├── configs/
  │   └── paper-project.yaml
  ├── data/
  │   ├── raw/
  │   └── processed/
  ├── models/
  │   └── saved/
  ├── portfolios/
  │   └── results/
  ├── .gitignore
  ├── README.md
  └── logs.log
  ```

* **`paper execute <phase>`**
  Delegate to sub-component CLIs based on `configs/paper-project.yaml`:

  * `data` → `paper-data process`
  * `models` → `paper-model process`
  * `portfolio` → `paper-portfolio process`

## Configuration

Edit `configs/paper-project.yaml`, for example:

```yaml
project_name: MyResearch
version: 0.1.0
components:
  data:
    cli_tool: paper-data
    cli_command: process
    config_file: configs/data-config.yaml
  models:
    cli_tool: paper-model
    cli_command: process
    config_file: configs/models-config.yaml
  portfolio:
    cli_tool: paper-portfolio
    cli_command: process
    config_file: configs/portfolio-config.yaml
```

## Development

```bash
# From within paper-tools directory
pip install -e .[dev]
pytest
```