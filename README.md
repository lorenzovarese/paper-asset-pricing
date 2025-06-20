# P.A.P.E.R: Platform for Asset-Pricing Experimentation and Research 📈🔬

[![Python CI](https://github.com/lorenzovarese/paper-asset-pricing/actions/workflows/ci.yaml/badge.svg)](https://github.com/lorenzovarese/paper-asset-pricing/actions/workflows/ci.yaml)
[![codecov](https://codecov.io/github/lorenzovarese/paper-asset-pricing/graph/badge.svg?token=ZUDEPEPJFK)](https://codecov.io/github/lorenzovarese/paper-asset-pricing)
[![PyPI version](https://badge.fury.io/py/paper-asset-pricing.svg)](https://badge.fury.io/py/paper-asset-pricing)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
├── paper-asset-pricing/    # 🚀 CLI & Orchestration
├── pyproject.toml          # uv workspace configuration
└── README.md               # You are here!
```

### Core Components:

*   **`paper-asset-pricing`**: The central command-line interface (CLI) and orchestrator for the entire P.A.P.E.R platform. It handles project initialization (`paper init`), manages configurations, executes the research workflow (`paper execute ...`), and facilitates publication (`paper publish ...`).
    *   **Key Features:** Project scaffolding with a standardized structure, execution of data, modeling, and portfolio pipelines, centralized logging, and a built-in tool to create a draft publication on Zenodo.
    *   **Learn More:** See [`paper-asset-pricing/README.md`](./paper-asset-pricing/README.md)
*   **`paper-data`**: Dedicated to data ingestion, cleaning, and preprocessing. It provides a flexible, configuration-driven pipeline to transform raw financial and economic data into clean, analysis-ready datasets.
    *   **Key Features:** Connectors for local files, Google Sheets, and WRDS. A powerful wrangling toolkit including imputation, scaling, merging, and lagging. **Supports custom Python scripts for complex, user-defined transformations.**
    *   **Learn More:** See [`paper-data/README.md`](./paper-data/README.md)
*   **`paper-model`**: Implements a robust framework for training and evaluating asset pricing models. It uses a rolling-window backtesting methodology to generate out-of-sample predictions and performance metrics.
    *   **Key Features:** Support for various scikit-learn (OLS, ElasticNet, PCR, PLS, RandomForest, etc.) and PyTorch (Neural Network) models. Pydantic-validated configuration, rolling-window evaluation, hyperparameter tuning, and generation of prediction files.
    *   **Learn More:** See [`paper-model/README.md`](./paper-model/README.md)
*   **`paper-portfolio`**: Constructs and analyzes long-short portfolios based on model predictions. It calculates various performance metrics and generates reports and visualizations to assess strategy viability.
    *   **Key Features:** Config-driven portfolio construction (long/short quantiles), equal and value weighting schemes, performance evaluation (Sharpe Ratio, Expected Shortfall), and generation of reports and plots.
    *   **Learn More:** See [`paper-portfolio/README.md`](./paper-portfolio/README.md)

---

## 🚀 Getting Started

The P.A.P.E.R workflow is designed to be sequential and intuitive. You start by setting up a project, then process data, train models, and finally, build and analyze portfolios.

### 1. Initial Setup

First, clone the repository and set up the development environment using `uv` or `venv`.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/paper-asset-pricing.git
cd paper-asset-pricing

# 2. Create and activate a virtual environment (e.g., with venv)
# uv sync (creates .venv in root)
python -m venv .venv
source .venv/bin/activate

# 3. Install all packages in editable mode
# The [all] extra installs optional dependencies for all components.
# uv -> The packages are already visible in the workspace
pip install -e .[all]
```

### 2. Initialize Your Research Project

Use the `paper` CLI from `paper-asset-pricing` to scaffold a new project. This creates a standardized directory structure for your work.

```bash
paper init ThesisExample
```

This command creates a `ThesisExample/` directory with `configs/`, `data/`, `models/`, and `portfolios/` subdirectories, along with placeholder configuration files.

```bash
# Navigate into your new project directory
cd ThesisExample
```

### 3. Phase I: Data Processing

This phase transforms your raw source data into a clean, analysis-ready dataset.

1.  **Add Raw Data**: Place your raw data files (e.g., `firm_data.csv`, `macro_data.csv`) inside the `data/raw/` directory.
2.  **Configure Pipeline**: Edit `configs/data-config.yaml`. Define how to load your raw files, what cleaning and transformation steps to perform (e.g., impute missing values, lag features), and how to save the final dataset.
3.  **Execute**: Run the data pipeline.

    ```bash
    paper execute data
    ```

**Output**: The console will show a simple success message. All detailed steps are logged in `logs.log`. The final, processed dataset(s) will be saved in the `data/processed/` directory (e.g., as `processed_panel_data.parquet`).

### 4. Phase II: Model Training & Prediction

This phase uses your processed data to train predictive models and generate out-of-sample return predictions.

1.  **Configure Models**: Edit `configs/models-config.yaml`. Specify which processed dataset to use, define your target variable (e.g., `ret_excess`), list your features, and configure one or more models (e.g., OLS, ElasticNet) and the rolling-window evaluation parameters.
2.  **Execute**: Run the modeling pipeline.

    ```bash
    paper execute models
    ```

**Output**: This will run the full backtesting loop. The `models/` directory will be populated with:
*   `models/predictions/`: Parquet files containing firm-month level return predictions for each model.
*   `models/evaluations/`: Reports and metrics (e.g., R², MSE) for each model's predictive performance.
*   `models/saved/`: (Optional) Saved model checkpoints for each training window.

### 5. Phase III: Portfolio Construction & Analysis

The final phase uses the model predictions to construct long-short portfolios and evaluate their performance.

1.  **Configure Strategies**: Edit `configs/portfolio-config.yaml`. Point to the prediction files you just generated, and define your portfolio sorting strategies (e.g., "Top-Decile vs. Bottom-Decile"), weighting schemes (equal or value-weighted), and desired performance metrics.
2.  **Execute**: Run the portfolio analysis pipeline.

    ```bash
    paper execute portfolio
    ```

**Output**: The `portfolios/results/` directory will be populated with performance reports, charts (e.g., cumulative returns), and detailed monthly return data for each strategy.

### 6. Phase IV: Publishing Your Research

P.A.P.E.R. helps you make your research reproducible and citable by providing a simple command to archive your project on [Zenodo](https://zenodo.org/), a free and open research data repository.

1.  **Get a Zenodo API Token**: Follow the instructions provided by the command to get a personal access token from Zenodo.
2.  **Create a Draft Publication**: Run the publish command. This will bundle your project's code and configuration (excluding large data files), create a new draft entry on Zenodo, and upload the archive.

    ```bash
    # Use the --sandbox flag to test with Zenodo's testing server first
    paper publish zenodo --sandbox
    ```

**Output**: The command will provide a direct link to your new draft on the Zenodo website. You can then go to this link to review the metadata, add co-authors, and click the final "Publish" button to receive a permanent Digital Object Identifier (DOI) for your work.

---

## 🛠️ Development & Contribution

We welcome contributions from the community! If you're interested in contributing to P.A.P.E.R, please refer to the individual `README.md` files within each sub-package for specific development guidelines and contribution processes.

### Running Tests

To run tests for all packages in the monorepo, first ensure you have `pytest` installed, then run:

```bash
pytest
```

---

✅ Code Quality & Test Coverage

We use continuous integration to ensure the quality and reliability of the codebase. The following graph shows the test coverage for each package in the monorepo. You can click on any part of the graph to explore the detailed coverage report on Codecov.

![Codecov Grid](https://codecov.io/github/lorenzovarese/paper-asset-pricing/graphs/tree.svg?token=ZUDEPEPJFK)

---

## 📄 License

This project is licensed under the MIT License - see the top-level `LICENSE` file for details.

---
