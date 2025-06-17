# paper-asset-pricing: The Orchestrator for P.A.P.E.R Research 🚀

`paper-asset-pricing` is the central command-line interface (CLI) and orchestration engine for the P.A.P.E.R (Platform for Asset Pricing Experimentation and Research) monorepo. It provides a unified entry point for initializing new research projects and executing the entire end-to-end asset pricing workflow.

Think of `paper-asset-pricing` as the conductor of your research symphony, ensuring each component (`paper-data`, `paper-model`, `paper-portfolio`) performs its role seamlessly and in the correct sequence.

---

## ✨ Features

*   **Project Initialization (`paper init`):** 🏗️
    *   Quickly scaffolds a standardized project directory structure with a single command.
    *   Generates essential configuration files (`paper-project.yaml`, `data-config.yaml`, etc.) and a project-specific `README.md` from templates.
    *   Creates a `.gitignore` file tailored for P.A.P.E.R projects.
    *   Ensures a consistent and reproducible starting point for all your research.
*   **Pipeline Orchestration (`paper execute`):** ➡️
    *   **`paper execute data`**: Triggers the `paper-data` pipeline to ingest, wrangle, and process raw data based on your `data-config.yaml`.
    *   **`paper execute models`**: Orchestrates model training and evaluation using `paper-model`, driven by your `models-config.yaml`.
    *   **`paper execute portfolio`**: Manages portfolio construction and performance analysis using `paper-portfolio`, based on your `portfolio-config.yaml`.
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
# Using pip
pip install "paper-asset-pricing[all]"

# Using uv
uv pip install "paper-asset-pricing[all]"
```

**Install Specific Components:**

If you only need `paper-asset-pricing` and a particular component (e.g., `paper-data`), you can install it like this:

```bash
# Using pip
pip install "paper-asset-pricing[data]"

# Using uv
uv pip install "paper-asset-pricing[data]"
```

**Core `paper-asset-pricing` only:**

If you only want the `paper init` command without any component dependencies:

```bash
# Using pip
pip install paper-asset-pricing

# Using uv
uv pip install paper-asset-pricing
```

**From Source (for development within the monorepo):**

Navigate to the root of your P.A.P.E.R monorepo and install `paper-asset-pricing` in editable mode.

```bash
# Using pip
pip install -e ./paper-asset-pricing

# Using uv
uv pip install -e ./paper-asset-pricing
```

---

## 🚀 Usage Workflow

Let's walk through initializing a new project and running the full research pipeline.

### 1. Initialize a New P.A.P.E.R Project

From your desired parent directory, run the `init` command:

```bash
paper init ToolsExampleProject
```

**Expected Console Output:**

```
Initializing P.A.P.E.R project 'ToolsExampleProject' at: /path/to/your/parent/directory
✓ Created project directories.
✓ Created main project config: ToolsExampleProject/configs/paper-project.yaml
✓ Created .gitignore file.
✓ Created project README.md.
✓ Created log file: ToolsExampleProject/logs.log
✓ Created placeholder component config: ToolsExampleProject/configs/data-config.yaml
✓ Created placeholder component config: ToolsExampleProject/configs/models-config.yaml
✓ Created placeholder component config: ToolsExampleProject/configs/portfolio-config.yaml
✓ Ensured .gitkeep in empty project subdirectories.

🎉 P.A.P.E.R project 'ToolsExampleProject' initialized successfully!

Navigate to your project:
  cd "ToolsExampleProject"

Next steps:
  1. Populate your component-specific YAML configuration files in 'configs/'.
  2. Place raw data in 'data/raw/'.
  3. Run phases using `paper execute <phase>`.
```

This command creates a standard directory structure inside `ToolsExampleProject/`.

### 2. Prepare Data and Configurations

Before executing the pipeline, you need to:

1.  **Place Raw Data**: Add your raw data files (e.g., `firm_data.csv`) to the `ToolsExampleProject/data/raw/` directory.
2.  **Configure `data-config.yaml`**: Edit `ToolsExampleProject/configs/data-config.yaml` to define your data ingestion and wrangling steps.
3.  **Configure `models-config.yaml`**: Edit `ToolsExampleProject/configs/models-config.yaml` to define the models you want to train and evaluate.
4.  **Configure `portfolio-config.yaml`**: Edit `ToolsExampleProject/configs/portfolio-config.yaml` to define the portfolio strategies you want to backtest.

### 3. Execute the Full Pipeline

The `paper execute` commands orchestrate the entire research workflow. It's best to run them from within your project directory.

```bash
# Navigate to the project directory
cd ToolsExampleProject

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

**`logs.log` Content (Snippet):**

```log
INFO - Starting Data Phase for project: ToolsExampleProject
INFO - Project root: /path/to/monorepo/ToolsExampleProject
INFO - Using data configuration: /path/to/monorepo/ToolsExampleProject/configs/data-config.yaml
INFO - Running data pipeline for project: /path/to/monorepo/ToolsExampleProject
... (detailed logs from paper-data) ...
INFO - Data pipeline completed successfully.
```

---

## ⚙️ Configuration (`paper-project.yaml`)

The `paper-project.yaml` file, located in your project's `configs/` directory, serves as the central configuration hub.

```yaml
# configs/paper-project.yaml
project_name: "ToolsExampleProject"
version: "0.1.0"
paper_asset_pricing_version: "0.1.0"
creation_date: "2025-05-01"
description: "P.A.P.E.R project: ToolsExampleProject"

components:
  data:
    config_file: "data-config.yaml"
  models:
    config_file: "models-config.yaml"
  portfolio:
    config_file: "portfolio-config.yaml"

logging:
  log_file: "logs.log"
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