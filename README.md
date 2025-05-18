# P.A.P.E.R: Platform for Asset Pricing Experimentation and Research 📈🔬

Welcome to **P.A.P.E.R** (Platform for Asset Pricing Experimentation and Research)! This monorepo is a comprehensive suite of tools designed to streamline the entire workflow of quantitative asset pricing research, from raw data ingestion to portfolio construction and performance analysis.

Our goal is to provide a modular, reproducible, and efficient framework for academics and practitioners to conduct rigorous asset pricing studies.

---

## 🌳 Monorepo Structure

This repository is organized as a monorepo, housing several interconnected Python packages. This structure allows for cohesive development, shared tooling, and easy management of inter-package dependencies, while still enabling independent deployment and installation of individual components.

```
.
├── LICENSE
├── paper-data/             # 📊 Data Ingestion & Preprocessing
├── paper-model/            # 🧠 Model Implementation & Evaluation
├── paper-portfolio/        # 💰 Portfolio Construction & Analysis
├── paper-tools/            # 🚀 CLI & Orchestration
├── pyproject.toml          # uv workspace configuration
└── README.md               # You are here!
```

### Core Components:

*   **`paper-tools`**: The central command-line interface (CLI) and orchestrator for the entire P.A.P.E.R platform. It handles project initialization, manages configurations, and executes various research phases (data, models, portfolio).
    *   **Key Features:** Project scaffolding, phase execution (`paper execute data/models/portfolio`), centralized logging.
    *   **Learn More:** See [`paper-tools/README.md`](./paper-tools/README.md)
*   **`paper-data`**: Dedicated to data ingestion, cleaning, and preprocessing. It provides a flexible, configuration-driven pipeline to transform raw financial and economic data into clean, analysis-ready datasets.
    *   **Key Features:** Connectors for local files, HTTP, Google Drive, Hugging Face, WRDS; monthly imputation, scaling, merging, lagging, interaction terms.
    *   **Learn More:** See [`paper-data/README.md`](./paper-data/README.md)
*   **`paper-model`**: (Under Development) This module will focus on the implementation of various asset pricing models. Its primary goal will be to generate model evaluations and checkpoints for subsequent portfolio construction.
    *   **Key Features (Planned):** Factor model estimation, machine learning models for return prediction, model evaluation metrics.
*   **`paper-portfolio`**: (Under Development) A lightweight module designed to utilize the processed data from `paper-data` and model outputs from `paper-model` to construct long-short portfolios and visualize their cross-sectional performance.
    *   **Key Features (Planned):** Portfolio sorting, performance attribution, characteristic-managed portfolios.

---

## 🚀 Getting Started

To get started with P.A.P.E.R, we recommend using `uv` for dependency management within the monorepo.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/paper-asset-pricing.git
cd paper-asset-pricing
```

### 2. Set up the `uv` Environment

`uv` is used to manage the monorepo's dependencies efficiently. Ensure you have `uv` installed (e.g., `pip install uv`).

```bash
# Install all dependencies for the entire monorepo, including optional ones
uv pip install -e ".[all]"

# This command will:
# - Install all core dependencies defined in the root pyproject.toml.
# - Install all packages in editable mode (`-e .`) within the workspace.
# - Install all optional dependencies (e.g., paper-data's dependencies) via `[all]`.
```

### 3. Initialize Your First Project

Use the `paper-tools` CLI to create a new research project with a standardized directory structure. Let's call our first project `ThesisExample`:

```bash
paper init ThesisExample
```

This command will create a new directory `ThesisExample/` with all the necessary subdirectories (`configs/`, `data/`, `models/`, `portfolios/`) and placeholder configuration files.

### 4. Prepare Your Data

Place your raw data files (e.g., CSVs, Parquet) into the `ThesisExample/data/raw/` directory.

Next, edit the `ThesisExample/configs/data-config.yaml` file to define how `paper-data` should ingest, wrangle, and export your data. Refer to the [`paper-data/README.md`](./paper-data/README.md) for detailed configuration options and examples.

### 5. Run the Data Processing Pipeline

Once your `data-config.yaml` is set up and raw data is in place, you can execute the data processing phase:

```bash
# Navigate into your project directory
cd ThesisExample

# Execute the data phase
paper execute data
```

**What to Expect:**
*   The console output will be minimal, indicating the start and successful completion of the data phase.
*   All detailed logs, including progress and intermediate steps, will be written to `ThesisExample/logs.log`.
*   Your processed data will be saved in `ThesisExample/data/processed/` as specified in your `data-config.yaml`.

### 6. Continue Your Research!

With your data processed, you're ready to move on to the modeling and portfolio construction phases (as `paper-model` and `paper-portfolio` become available).

---

## 🛠️ Development & Contribution

We welcome contributions from the community! If you're interested in contributing to P.A.P.E.R, please refer to the individual `README.md` files within each sub-package for specific development guidelines and contribution processes.

### Running Tests

To run tests for all packages in the monorepo:

```bash
uv run pytest
```

---

## 📄 License

This project is licensed under the MIT License - see the top-level `LICENSE` file for details.

---