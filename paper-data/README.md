# paper-data: Data Ingestion & Preprocessing for Asset Pricing Research 📊

[![codecov](https://codecov.io/github/lorenzovarese/paper-asset-pricing/graph/badge.svg?token=ZUDEPEPJFK)](https://codecov.io/github/lorenzovarese/paper-asset-pricing)
[![PyPI version](https://badge.fury.io/py/paper-data.svg)](https://badge.fury.io/py/paper-data)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`paper-data` is a core component of the P.A.P.E.R (Platform for Asset Pricing Experimentation and Research) monorepo. It provides a robust, flexible, and configuration-driven pipeline for ingesting raw financial and economic data, performing essential wrangling operations, and exporting clean, processed datasets ready for modeling and portfolio construction.

Built with [Polars](https://pola.rs/) for high performance and memory efficiency, `paper-data` streamlines the often complex and time-consuming process of data preparation in quantitative finance.

---

## ✨ Features

*   **Modular Data Connectors:** Seamlessly ingest data from various sources:
    *   📁 **Local Files**: Load data from local CSV files (`CSVLoader`).
    *   📝 **Google Sheets**: Download and cache public Google Sheets (`GoogleSheetConnector`).
    *   🔒 **WRDS**: Execute SQL queries on Wharton Research Data Services and cache results locally (`WRDSConnector`).
*   **Comprehensive Wrangling Operations:** Apply common data transformations declaratively via a YAML configuration:
    *   **Monthly Imputation:** Fill missing numeric values with cross-sectional medians and categorical values with modes.
    *   **Min-Max Scaling:** Normalize features to a specified range (e.g., `[-1, 1]`) on a monthly cross-sectional basis.
    *   **Dummy Variable Generation:** Create one-hot encoded (dummy) columns from a categorical feature (e.g., industry codes).
    *   **Dataset Merging:** Combine different datasets (e.g., firm-level with macro-level data) using various join types.
    *   **Lagging/Leading:** Create lagged or lead versions of columns for time-series analysis, with support for panel data grouping.
    *   **Interaction Terms:** Generate interaction features between different sets of columns (e.g., firm characteristics and macro indicators).
*   **Configuration-Driven Pipeline:** Define your entire data pipeline (ingestion, wrangling, export) in a human-readable YAML file, promoting reproducibility and ease of experimentation.
*   **Performance-Optimized:** Leverages the speed and efficiency of the Polars DataFrame library for all data manipulation tasks, including support for lazy (out-of-core) execution for memory-intensive operations.
*   **Flexible Export:** Export processed data to the efficient Parquet format, with optional partitioning by year for easy downstream consumption by the modeling pipeline.
*   **Integrated Logging:** Detailed logs are written to a file, providing transparency and debugging capabilities without cluttering the console.

---

## 🚀 Installation

`paper-data` is designed to be part of the larger `PAPER` monorepo. You can install it as an optional dependency of `paper-asset-pricing` or as a standalone package.

**Recommended (as part of `paper-asset-pricing`):**

This method ensures `paper-data` is available to the main `paper` CLI orchestrator.

```bash
# Using pip
pip install "paper-asset-pricing[data]"

# Using uv
uv add "paper-asset-pricing[data]"
```

**Standalone Installation:**

If you only need `paper-data` and its core functionalities for a different project.

```bash
# Using pip
pip install paper-data

# Using uv
uv add paper-data
```

**From Source (for development within the monorepo):**

Navigate to the root of your `PAPER` monorepo and install `paper-data` in editable mode.

```bash
# Using pip
pip install -e ./paper-data

# Using uv
uv pip install -e ./paper-data
```

---

## 📖 Usage Example: Synthetic Data Pipeline

This example demonstrates how to use `paper-data` to process synthetic firm-level and macro-economic data.

### 1. Project Setup & Data Generation

First, ensure you have initialized a project using `paper init ThesisExample`. For this example, we'll assume your project directory `ThesisExample/` is at the root of the monorepo.

Navigate to the `paper-data/examples/synthetic_data` directory and generate the raw CSV files:

```bash
# Assuming you are in the monorepo root
cd paper-data/examples/synthetic_data

# Generate synthetic firm and macro data
python firm_synthetic.py
python macro_synthetic.py
```

This will create `firm_synthetic.csv` and `macro_synthetic.csv`.

### 2. Data Configuration (`data-config.yaml`)

Create a `data-config.yaml` file in your project's `configs` directory (e.g., `ThesisExample/configs/data-config.yaml`). This file defines the entire data processing pipeline.

```yaml
# ThesisExample/configs/data-config.yaml
ingestion:
  - name: "firm_data_raw"
    path: "firm_synthetic.csv" # Path relative to ThesisExample/data/raw
    format: "csv"
    date_column: { "date": "%Y%m%d" }
    firm_id_column: "permco"
    to_lowercase_cols: true

  - name: "macro_data_raw"
    path: "macro_synthetic.csv" # Path relative to ThesisExample/data/raw
    format: "csv"
    date_column: { "date": "%Y%m%d" }
    to_lowercase_cols: true

wrangling_pipeline:
  - operation: "monthly_imputation"
    dataset: "firm_data_raw"
    numeric_columns: [ "volume", "marketcap" ]
    output_name: "firm_data_imputed"

  - operation: "merge"
    left_dataset: "firm_data_imputed"
    right_dataset: "macro_data_raw"
    on: [ "date" ]
    how: "left"
    output_name: "merged_data"

  - operation: "lag"
    dataset: "merged_data"
    periods: 1
    columns_to_lag:
      - method: "all_except"
        columns: [ "date", "permco", "return", "volume", "marketcap" ]
    drop_original_cols_after_lag: false
    restore_names: false
    drop_generated_nans: true
    output_name: "panel_with_lags"

  - operation: "create_macro_interactions"
    dataset: "panel_with_lags"
    macro_columns: [ "gdp_growth_lag_1", "cpi_lag_1", "unemployment_lag_1" ]
    firm_columns: [ "marketcap" ]
    drop_macro_columns: false
    output_name: "final_panel_data"

export:
  - dataset_name: "final_panel_data"
    output_filename_base: "processed_panel_data"
    format: "parquet"
    partition_by: "year" # 'year' or 'none' are supported
```

**Important:** Copy the generated CSV files into your project's raw data directory.

```bash
# From the monorepo root
cp paper-data/examples/synthetic_data/*.csv ThesisExample/data/raw/
```

### 3. Running the Data Pipeline

The intended way to run the pipeline is with the `paper-asset-pricing` CLI from within your project directory.

```bash
# Navigate to your project directory from the monorepo root
cd ThesisExample

# Execute the data phase
paper execute data
```

### 4. Expected Output

**Console Output:**

The console output is minimal, confirming the process and directing you to the logs.

```
>>> Executing Data Phase <<<
Data phase completed successfully. Additional information in 'ThesisExample/logs.log'
```

**`ThesisExample/logs.log` Content (Snippet):**

The log file provides a detailed, step-by-step account of the pipeline's execution.

```log
INFO - Starting Data Phase for project: ThesisExample
INFO - Using data configuration: /path/to/monorepo/ThesisExample/configs/data-config.yaml
INFO - Running data pipeline for project: /path/to/monorepo/ThesisExample
INFO - --- Ingesting Data ---
INFO - Dataset 'firm_data_raw' ingested. Shape: (125, 5)
INFO - Dataset 'macro_data_raw' ingested. Shape: (25, 4)
INFO - --- Wrangling Data ---
INFO - --- Wrangling Step 1: monthly_imputation ---
INFO -   Input Dataset: 'firm_data_raw'
INFO -   Numeric Columns: ['volume', 'marketcap']
INFO -   Output Dataset: 'firm_data_imputed'
INFO - --- Wrangling Step 2: merge ---
INFO -   Left Dataset: 'firm_data_imputed' (Shape: (125, 5))
INFO -   Right Dataset: 'macro_data_raw' (Shape: (25, 4))
INFO -   -> Merge complete. New dataset 'merged_data' shape: (125, 8)
INFO - --- Wrangling Step 3: lag ---
INFO -   Input Dataset: 'merged_data'
INFO -   Periods: 1
INFO -   Columns to Lag: ['gdp_growth', 'cpi', 'unemployment']
INFO -   -> Lag operation complete. New dataset 'panel_with_lags' shape: (120, 11)
INFO - --- Wrangling Step 4: create_macro_interactions ---
INFO -   Input Dataset: 'panel_with_lags'
INFO -   Macro Columns: ['gdp_growth_lag_1', 'cpi_lag_1', 'unemployment_lag_1']
INFO -   Firm Columns: ['marketcap']
INFO -   -> Eager macro-firm interaction creation complete. New dataset 'final_panel_data' shape: (120, 14)
INFO - --- Exporting Data ---
INFO - Found eager dataset 'final_panel_data' for export.
INFO - Exporting 'final_panel_data' by year to separate files:
INFO -   Exported data for year 2024 to '.../ThesisExample/data/processed/processed_panel_data_2024.parquet'.
INFO -   Exported data for year 2025 to '.../ThesisExample/data/processed/processed_panel_data_2025.parquet'.
INFO - Data pipeline completed successfully.
```

### 5. Processed Data Output

After successful execution, you will find the processed Parquet files in your project's `data/processed` directory:

```
ThesisExample/data/processed/
├── processed_panel_data_2024.parquet
└── processed_panel_data_2025.parquet
```

---

## ⚙️ Configuration Reference

The `data-config.yaml` file is the heart of `paper-data`. Here's a breakdown of its main sections:

### `ingestion`

A list of datasets to ingest. Each item defines a source:

*   `name` (string, required): A unique identifier for the dataset within the pipeline.
*   `format` (string, required): The ingestion format. Supports `"csv"`, `"google_sheet"`, `"wrds"`, `"google_drive"`.
*   **For `csv`**:
    *   `path` (string, required): Relative path to the raw data file (from `project_root/data/raw/`).
*   **For `google_sheet` / `google_drive`**:
    *   `url` (string, required): The full URL to the shareable resource.
*   **For `wrds`**:
    *   `query` (string, required): The SQL query to execute.
*   `date_column` (object, required): Specifies the date column and its format. E.g., `{ "date": "%Y%m%d" }`.
*   `firm_id_column` (string, optional): The column name for the firm identifier (e.g., "permco").
*   `to_lowercase_cols` (boolean, optional, default: `false`): Whether to convert all column names to lowercase.

### `wrangling_pipeline`

A sequential list of operations to apply to your datasets.

*   **`operation: "monthly_imputation"`**
    *   `dataset` (string, required): The name of the dataset to apply imputation to.
    *   `numeric_columns` (list, optional): Columns to impute with monthly cross-sectional median.
    *   `categorical_columns` (list, optional): Columns to impute with monthly cross-sectional mode.
    *   `output_name` (string, required): The name for the resulting dataset.
*   **`operation: "scale_to_range"`**
    *   `dataset` (string, required): The name of the dataset to scale.
    *   `range` (object, required): Defines the target `min` and `max`. E.g., `{ min: -1, max: 1 }`.
    *   `cols_to_scale` (list, required): Numeric columns to apply min-max scaling to.
    *   `output_name` (string, required): The name for the resulting dataset.
*   **`operation: "merge"`**
    *   `left_dataset` & `right_dataset` (string, required): Names of the datasets to merge.
    *   `on` (list, required): Columns to merge on.
    *   `how` (string, required): Join type (`"left"`, `"inner"`, etc.).
    *   `output_name` (string, required): Name for the merged dataset.
*   **`operation: "lag"`**
    *   `dataset` (string, required): The dataset to use.
    *   `periods` (integer, required): Number of periods to shift.
    *   `columns_to_lag` (list, required): Defines which columns to lag. The only supported `method` is `"all_except"`.
    *   `drop_original_cols_after_lag` (boolean, optional): If `true`, original columns are dropped.
    *   `restore_names` (boolean, optional): If `true` and `drop_original_cols_after_lag` is `true`, renames lagged columns to their original names.
    *   `drop_generated_nans` (boolean, optional): If `true`, drops rows with NaNs introduced by lagging.
    *   `output_name` (string, required): The name for the resulting dataset.
*   **`operation: "create_macro_interactions"`**
    *   `dataset` (string, required): The dataset to use.
    *   `macro_columns` & `firm_columns` (list, required): Lists of columns to interact.
    *   `use_lazy_engine` (boolean, optional): If `true`, uses Polars' lazy API to reduce memory usage.
    *   `output_name` (string, required): The name for the resulting dataset.

### `export`

A list of processed datasets to export.

*   `dataset_name` (string, required): The name of the dataset to export.
*   `output_filename_base` (string, required): The base name for the output file(s).
*   `format` (string, required): Currently supports `"parquet"`.
*   `partition_by` (string, optional): How to partition the output. Supports `"year"` or `"none"`.

---

## 🤝 Contributing

We welcome contributions to `paper-data`! If you have suggestions for new data connectors, wrangling operations, or performance improvements, please feel free to open an issue or submit a pull request.

---

## 📄 License

`paper-data` is distributed under the MIT License. See the `LICENSE` file for more information.

---
