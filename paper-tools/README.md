# paper-tools: The Orchestrator for P.A.P.E.R Research 🚀

`paper-tools` is the central command-line interface (CLI) and orchestration engine for the P.A.P.E.R (Platform for Asset Pricing Experimentation and Research) monorepo. It provides a unified entry point for initializing new research projects and executing various phases of your asset pricing workflow, including data processing, model training, and portfolio construction.

Think of `paper-tools` as the conductor of your research symphony, ensuring each component (`paper-data`, `paper-models`, `paper-portfolio`) plays its part seamlessly.

---

## ✨ Features

*   **Project Initialization (`paper init`):** 🏗️
    *   Quickly set up a standardized project directory structure.
    *   Generates essential configuration files (`paper-project.yaml`, `data-config.yaml` placeholders, etc.).
    *   Creates a `.gitignore` and `README.md` for your new project.
    *   Ensures a consistent and reproducible starting point for all your research.
*   **Phase Execution (`paper execute`):** ➡️
    *   **Data Phase (`paper execute data`):** Triggers the `paper-data` pipeline to ingest, wrangle, and process your raw data based on your `data-config.yaml`.
    *   **Models Phase (`paper execute models`):** (Planned) Will orchestrate model training and evaluation using `paper-models`.
    *   **Portfolio Phase (`paper execute portfolio`):** (Planned) Will manage portfolio construction and performance analysis using `paper-portfolio`.
*   **Modular & Extensible:** Designed to integrate seamlessly with other `PAPER` components, allowing you to install and use only the parts you need.
*   **Centralized Configuration:** Manages project-wide settings and component-specific configuration file paths via `paper-project.yaml`.
*   **Intelligent Logging:** Directs detailed operational logs to a dedicated `logs.log` file within your project, keeping your console clean for critical messages.
*   **Auto-detection of Project Root:** Smartly identifies your project's root directory, making it easier to run commands from any subdirectory.

---

## 📦 Installation

`paper-tools` is the primary entry point for the P.A.P.E.R ecosystem. You can install it with specific components as optional dependencies.

**Recommended (with all components):**

To get `paper-tools` and all its current and future components (`paper-data`, `paper-models`, `paper-portfolio`), use the `all` extra:

```bash
pip install paper-tools[all]
# Or if using uv:
uv add paper-tools[all]
```

**Install Specific Components:**

If you only need `paper-tools` and a particular component (e.g., `paper-data`), you can install it like this:

```bash
pip install paper-tools[data]
# Or for models:
pip install paper-tools[models]
# Or for portfolio:
pip install paper-tools[portfolio]
```

**Core `paper-tools` only:**

If you only want the basic `paper-tools` CLI (e.g., just for `init`), without any component dependencies:

```bash
pip install paper-tools
```

**From Source (for development within the monorepo):**

Navigate to the root of your `PAPER` monorepo and install `paper-tools` in editable mode:

```bash
cd /path/to/your/PAPER_monorepo
uv pip install -e ./paper-tools
```

---

## 🚀 Usage Example

Let's walk through initializing a new project and running the data processing phase.

### 1. Initialize a New P.A.P.E.R Project

From your desired parent directory (e.g., the root of your monorepo or any other location), run the `init` command:

```bash
paper init ThesisExample
```

**Expected Console Output:**

```Initializing P.A.P.E.R project 'ThesisExample' at: /path/to/your/parent/directory
✓ Created project directories.
✓ Created main project config: ThesisExample/configs/paper-project.yaml
✓ Created .gitignore file.
✓ Created project README.md.
✓ Created log file: ThesisExample/logs.log
✓ Created placeholder component config: ThesisExample/configs/data-config.yaml
✓ Created placeholder component config: ThesisExample/configs/models-config.yaml
✓ Created placeholder component config: ThesisExample/configs/portfolio-config.yaml
✓ Ensured .gitkeep in empty project subdirectories.

🎉 P.A.P.E.R project 'ThesisExample' initialized successfully!

Navigate to your project:
  cd "ThesisExample"

Next steps:
  1. Populate your component-specific YAML configuration files in 'configs/'.
     (e.g., 'data-config.yaml', 'models-config.yaml', 'portfolio-config.yaml')
  2. Place raw data in 'data/raw/'.
  3. Run phases using `paper execute <phase>`.
```

This command creates a directory structure like this:

```
ThesisExample/
├── configs/
│   ├── data-config.yaml
│   ├── models-config.yaml
│   ├── paper-project.yaml
│   └── portfolio-config.yaml
├── data/
│   ├── processed/
│   └── raw/
├── logs.log
├── models/
│   └── saved/
├── portfolios/
│   └── results/
└── README.md
```

### 2. Prepare Data and Configuration

For this example, let's use the synthetic data from `paper-data`.

*   **Generate Synthetic Data:**
    ```bash
    # Assuming you are in the monorepo root
    cd paper-data/examples/synthetic_data
    python firm_synthetic.py
    python macro_synthetic.py
    ```
*   **Copy Data to Project:**
    ```bash
    # Assuming you are back in the monorepo root
    cp paper-data/examples/synthetic_data/firm_synthetic.csv ThesisExample/data/raw/
    cp paper-data/examples/synthetic_data/macro_synthetic.csv ThesisExample/data/raw/
    ```
*   **Populate `data-config.yaml`:**
    Edit `ThesisExample/configs/data-config.yaml` with the content provided in the `paper-data` README. This file tells `paper-data` how to ingest and process your data.

### 3. Execute the Data Phase

Now, from your project root (`ThesisExample/`) or the monorepo root, you can execute the data phase:

```bash
# From the ThesisExample project root:
cd ThesisExample
paper execute data

# Or from the monorepo root, specifying the project path:
paper execute data --project-path ThesisExample
```

**Expected Console Output:**

You will see minimal output on the console, primarily indicating the start and successful completion of the data phase, along with the path to the detailed logs.

```
>>> Executing Data Phase <<<
Auto-detected project root: /path/to/your/PAPER_monorepo/ThesisExample
Data phase completed successfully. Additional information in '/path/to/your/PAPER_monorepo/ThesisExample/logs.log'
```

**`ThesisExample/logs.log` Content (Snippet):**

The `logs.log` file will contain detailed information about each step of the data pipeline, including data ingestion, wrangling operations, and export, as generated by `paper-data`.

```log
2025-06-12 12:30:01,123 - paper_tools.cli - INFO - Starting Data Phase for project: ThesisExample
2025-06-12 12:30:01,124 - paper_tools.cli - INFO - Project root: /path/to/your/PAPER_monorepo/ThesisExample
2025-06-12 12:30:01,125 - paper_tools.cli - INFO - Using data configuration: /path/to/your/PAPER_monorepo/ThesisExample/configs/data-config.yaml
2025-06-12 12:30:01,126 - paper_data.manager - INFO - Running data pipeline for project: /path/to/your/PAPER_monorepo/ThesisExample
... (detailed logs from paper-data) ...
2025-06-12 12:30:01,163 - paper_data.manager - INFO - Data pipeline completed successfully.
2025-06-12 12:30:01,164 - paper_tools.cli - INFO - Data phase completed successfully.
```

After successful execution, you will find the processed Parquet files in your project's `data/processed` directory.

---

## ⚙️ Configuration (`paper-project.yaml`)

The `paper-project.yaml` file, located in your project's `configs/` directory, serves as the central configuration hub for your P.A.P.E.R project.

```yaml
# configs/paper-project.yaml
project_name: "ThesisExample"
version: "0.1.0"
paper_tools_version: "0.1.0"
creation_date: "2025-06-11"
description: "P.A.P.E.R project: ThesisExample"

components:
  data:
    config_file: "data-config.yaml" # Path to data component's config relative to configs/
  models:
    config_file: "models-config.yaml" # Path to models component's config relative to configs/
  portfolio:
    config_file: "portfolio-config.yaml" # Path to portfolio component's config relative to configs/

logging:
  log_file: "logs.log" # Name of the main log file in the project root
  level: "INFO" # Global logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
```

*   **`project_name`**: A human-readable name for your project.
*   **`version`**: Your project's version.
*   **`paper_tools_version`**: The version of `paper-tools` used to initialize the project.
*   **`creation_date`**: The date the project was initialized.
*   **`description`**: A brief description of your project.
*   **`components`**: Defines the configuration files for each `PAPER` component.
    *   Each sub-key (`data`, `models`, `portfolio`) points to the YAML file that configures that specific component. These paths are relative to the `configs/` directory.
*   **`logging`**: Configures the project's logging behavior.
    *   `log_file`: The name of the log file (e.g., `logs.log`) located in the project's root directory.
    *   `level`: The minimum logging level to capture (e.g., `INFO`, `DEBUG`, `WARNING`, `ERROR`).

---

## 🤝 Contributing

We welcome contributions to `paper-tools`! If you have suggestions for new commands, improvements to existing features, or bug fixes, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes and write tests.
4.  Commit your changes (`git commit -am 'Add new feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Create a new Pull Request.

---

## 📄 License

`paper-tools` is distributed under the MIT License. See the `LICENSE` file for more information.

---