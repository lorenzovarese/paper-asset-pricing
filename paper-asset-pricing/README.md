# paper-asset-pricing: The Orchestrator for P.A.P.E.R Research 🚀

[![codecov](https://codecov.io/github/lorenzovarese/paper-asset-pricing/graph/badge.svg?token=ZUDEPEPJFK)](https://codecov.io/github/lorenzovarese/paper-asset-pricing)
[![PyPI version](https://badge.fury.io/py/paper-asset-pricing.svg)](https://badge.fury.io/py/paper-asset-pricing)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`paper-asset-pricing` is the central command-line interface (CLI) and orchestration engine for the P.A.P.E.R (Platform for Asset Pricing Experimentation and Research) monorepo. It provides a unified entry point for initializing new research projects and executing the entire end-to-end asset pricing workflow.

Think of `paper-asset-pricing` as the conductor of your research symphony, ensuring each component (`paper-data`, `paper-model`, `paper-portfolio`) performs its role seamlessly and in the correct sequence.

---

## ✨ Features

*   **Project Initialization (`paper init`):** 🏗️
    *   Quickly scaffolds a standardized project directory structure with a single command.
    *   Generates rich, well-commented configuration templates (`data-config.yaml`, etc.) and a project-specific `README.md` to guide you.
    *   Creates a `.gitignore` file tailored for P.A.P.E.R projects and initializes a local Git repository.
    *   Ensures a consistent and reproducible starting point for all your research.
*   **Pipeline Orchestration (`paper execute`):** ➡️
    *   **`paper execute data`**: Triggers the `paper-data` pipeline to ingest, wrangle, and process raw data based on your `data-config.yaml`.
    *   **`paper execute models`**: Orchestrates model training and evaluation using `paper-model`, driven by your `models-config.yaml`.
    *   **`paper execute portfolio`**: Manages portfolio construction and performance analysis using `paper-portfolio`, based on your `portfolio-config.yaml`.
*   **Research Publication (`paper publish`):** 📦
    *   **`paper publish zenodo`**: Bundles your project's code and configurations (excluding data) and uploads it as a draft to [Zenodo](https://zenodo.org/). This makes it easy to get a permanent DOI for your research, enhancing reproducibility and citability.
*   **Modular & Extensible:** Designed to integrate seamlessly with other P.A.P.E.R components, prompting you to install them if they are not found.
*   **Centralized Configuration:** Manages project-wide settings via `paper-project.yaml`.
*   **Intelligent Logging:** Directs detailed operational logs from all components to a single `logs.log` file within your project, keeping your console clean for critical messages and final summaries.
*   **Auto-detection of Project Root:** Smartly identifies your project's root directory, allowing you to run commands from any subdirectory.
---

## 📦 Installation

`paper-asset-pricing` is the primary entry point for the P.A.P.E.R ecosystem. You can install it with specific components as optional dependencies.

**Recommended (with all components):**

To get `paper-asset-pricing` and all its components (`paper-data`, `paper-model`, `paper-portfolio`), use the `all` extra.

```bash
pip install "paper-asset-pricing[all]"
```

**Install Specific Components:**

If you only need `paper-asset-pricing` and a particular component (e.g., `paper-data`), you can install it like this:

```bash
pip install "paper-asset-pricing[data]"
```

**Core `paper-asset-pricing` only:**

If you only want the `paper init` command without any component dependencies:

```bash
pip install paper-asset-pricing
```

**From Source (for development within the monorepo):**

Navigate to the root of your P.A.P.E.R monorepo and install `paper-asset-pricing` in editable mode.

```bash
pip install -e ./paper-asset-pricing
```

---

## 🚀 Usage Workflow

Let's walk through initializing a new project and running the full research pipeline.

### 1. Initialize a New P.A.P.E.R Project

From your desired parent directory, run the `init` command:

```bash
paper init ExampleProject
```

**Expected Console Output:**

```
Initializing P.A.P.E.R project 'ExampleProject' at: /path/to/your/parent/directory
✓ Created project directories.
✓ Created main project config: ExampleProject/configs/paper-project.yaml
✓ Created .gitignore file.
✓ Created project README.md.
✓ Created log file: ExampleProject/logs.log

Creating component configuration files from templates:
✓ Created: ExampleProject/configs/data-config.yaml
✓ Created: ExampleProject/configs/models-config.yaml
✓ Created: ExampleProject/configs/portfolio-config.yaml

✓ Ensured .gitkeep in empty project subdirectories.
✓ Initialized git repository.
✓ Created initial commit.

🎉 P.A.P.E.R project 'ExampleProject' initialized successfully!

Navigate to your project:
  cd "ExampleProject"

Next steps:
  1. Review and edit the configuration templates in the 'configs/' directory.
  2. (Optional) If using local files, place them in the `data/raw/` directory.
  3. Run the first phase of your research, e.g., `paper execute data`.
```

This command creates a standard directory structure inside `ExampleProject/`.

### 2. Prepare Data and Configurations

Before executing the pipeline, you need to:

1.  **Place Raw Data**: Add your raw data files (e.g., `firm_data.csv`) to the `ExampleProject/data/raw/` directory.
2.  **Configure `data-config.yaml`**: Edit `ExampleProject/configs/data-config.yaml` to define your data ingestion and wrangling steps.
3.  **Configure `models-config.yaml`**: Edit `ExampleProject/configs/models-config.yaml` to define the models you want to train and evaluate.
4.  **Configure `portfolio-config.yaml`**: Edit `ExampleProject/configs/portfolio-config.yaml` to define the portfolio strategies you want to backtest.

### 3. Execute the Full Pipeline

The `paper execute` commands orchestrate the entire research workflow. It's best to run them from within your project directory.

```bash
# Navigate to the project directory
cd ExampleProject

# 1. Run the data phase
paper execute data

# 2. Run the models phase
paper execute models

# 3. Run the portfolio phase
paper execute portfolio
```

Each command will provide a clean success message on the console, with all detailed output captured in the `logs.log` file. For example, after running the data phase:

**Console Output:**

```
>>> Executing Data Phase <<<
Data phase completed successfully. Additional information in 'logs.log'
```

### 4. Publish Your Research for Reproducibility

Once your research is complete, you can easily create a citable, permanent archive of your code and configurations on Zenodo.

```bash
# This command will guide you through the process
paper publish zenodo
```

This will create a draft on Zenodo. You can then visit the provided URL to add final details and officially publish your work to get a DOI.

**`logs.log` Content (Snippet):**

```log
INFO - Starting Data Phase for project: ExampleProject
INFO - Project root: /path/to/monorepo/ExampleProject
INFO - Using data configuration: /path/to/monorepo/ExampleProject/configs/data-config.yaml
INFO - Running data pipeline for project: /path/to/monorepo/ExampleProject
... (detailed logs from paper-data) ...
INFO - Data pipeline completed successfully.
```

---

## ⚙️ Configuration (`paper-project.yaml`)

The `paper-project.yaml` file, located in your project's `configs/` directory, serves as the central configuration hub.


```yaml
# ---------------------------------------------------------------------------
# P.A.P.E.R Main Project Configuration File
#
# This file acts as the central hub for your research project.
# It defines project metadata and points to the configuration files
# for each phase of the research pipeline (data, models, portfolio).
# ---------------------------------------------------------------------------

# --- Project Metadata ---
# Basic information about your research project.
project_name: "ExampleProject"
version: "0.1.0"
paper_asset_pricing_version: "0.1.2" # Version of P.A.P.E.R used to create this project.
creation_date: "2025-06-19"
description: "A research project on..." # TODO: Add a brief, one-line description of your project.

# --- Component Configuration ---
# This section maps each research phase to its specific YAML configuration file.
# The file paths are relative to this 'configs' directory.
components:
  # Configuration for the data ingestion and wrangling phase.
  data:
    config_file: "data-config.yaml"
  # Configuration for the model training and evaluation phase.
  models:
    config_file: "models-config.yaml"
  # Configuration for the portfolio construction and analysis phase.
  portfolio:
    config_file: "portfolio-config.yaml"

# --- Logging Configuration ---
# Controls how logs are recorded for the entire project.
logging:
  # The name of the log file, located in the project's root directory.
  log_file: "logs.log"
  # The minimum level of log messages to record.
  # Options (from most to least verbose): DEBUG, INFO, WARNING, ERROR, CRITICAL
  level: "INFO"
```

*   **`components`**: Defines the configuration files for each P.A.P.E.R component. The paths are relative to the `configs/` directory.
*   **`logging`**: Configures the project's logging behavior.
    *   `log_file`: The name of the log file in the project's root directory.
    *   `level`: The minimum logging level to capture (e.g., `INFO`, `DEBUG`).

---

## 🤝 Contributing

We welcome contributions to `paper-asset-pricing`! If you have suggestions for new commands, improvements to existing features, or bug fixes, please feel free to open an issue or submit a pull request.

---

## 📄 License

`paper-asset-pricing` is distributed under the MIT License. See the `LICENSE` file for more information.

---
