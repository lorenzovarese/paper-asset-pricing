# paper-data: Data Ingestion & Preprocessing for Asset Pricing Research 📊

`paper-data` is a core component of the P.A.P.E.R (Platform for Asset Pricing Experimentation and Research) monorepo. It provides a robust, flexible, and configuration-driven pipeline for ingesting raw financial and economic data, performing essential wrangling operations, and exporting clean, processed datasets ready for modeling and portfolio construction.

Built with [Polars](https://pola.rs/) for high performance and memory efficiency, `paper-data` streamlines the often complex and time-consuming process of data preparation in quantitative finance.

---

## ✨ Features

*   **Modular Data Connectors:** Seamlessly ingest data from various sources:
    *   📁 Local CSV files (`CSVLoader`)
    *   🌐 HTTP/HTTPS URLs (`HTTPConnector`)
    *   ☁️ Google Drive shared links (`GoogleDriveConnector`)
    *   🤗 Hugging Face Datasets Hub (`HuggingFaceConnector`)
    *   🔒 WRDS (Wharton Research Data Services) (`WRDSConnector`)
*   **Comprehensive Wrangling Operations:** Apply common data transformations declaratively via a YAML configuration:
    *   **Monthly Imputation:** Fill missing numeric values with cross-sectional medians and categorical values with modes.
    *   **Min-Max Scaling:** Normalize features to a specified range (e.g., `[-1, 1]`) on a monthly cross-sectional basis.
    *   **Dummy Variable Generation:** Create one-hot encoded (dummy) columns from a categorical feature (e.g., industry codes).
    *   **Dataset Merging:** Combine different datasets (e.g., firm-level with macro-level data) using various join types.
    *   **Lagging/Leading:** Create lagged or lead versions of columns for time-series analysis, with support for panel data grouping.
    *   **Interaction Terms:** Generate interaction features between different sets of columns (e.g., firm characteristics and macro indicators).
*   **Configuration-Driven Pipeline:** Define your entire data pipeline (ingestion, wrangling, export) in a human-readable YAML file, promoting reproducibility and ease of experimentation.
*   **Polars-Native Performance:** Leverage the speed and efficiency of Polars DataFrames for all data manipulation tasks.
*   **Flexible Export:** Export processed data to Parquet format, with optional partitioning by year for efficient downstream consumption.
*   **Integrated Logging:** Detailed logs are written to a file, providing transparency and debugging capabilities without cluttering the console.

---

## 🚀 Installation

`paper-data` is designed to be part of the larger `PAPER` monorepo. You can install it as an optional dependency of `paper-tools` or as a standalone package.

**Recommended (as part of `paper-tools`):**

If you have `paper-tools` installed, you can get `paper-data` and its dependencies using the `data` extra:

```bash
pip install paper-tools[data]
# Or if using uv:
uv add paper-tools[data]
```

**Standalone Installation:**

If you only need `paper-data` and its core functionalities, you can install it directly:

```bash
pip install paper-data
# Or if using uv:
uv add paper-data
```

**From Source (for development within the monorepo):**

Navigate to the root of your `PAPER` monorepo and install `paper-data` in editable mode:

```bash
cd /path/to/your/PAPER_monorepo
uv pip install -e ./paper-data
```

---

## 📖 Usage Example: Synthetic Data Pipeline

This example demonstrates how to use `paper-data` to process synthetic firm-level and macro-economic data.

### 1. Project Setup & Data Generation

First, ensure you have a project structure similar to the `ThesisExample` shown in the main `PAPER` monorepo documentation. For this example, we'll assume you're running from the monorepo root.

Navigate to the `paper-data/examples/synthetic_data` directory and generate the raw CSV files:

```bash
# Assuming you are in the monorepo root, e.g., 'PAPER/'
cd paper-data/examples/synthetic_data

# Generate synthetic firm data
python firm_synthetic.py

# Generate synthetic macro data
python macro_synthetic.py
```

This will create `firm_synthetic.csv` and `macro_synthetic.csv` in the `paper-data/examples/synthetic_data` directory.

### 2. Data Configuration (`data-config.yaml`)

Create a `data-config.yaml` file in your project's `configs` directory (e.g., `ThesisExample/configs/data-config.yaml`). This file defines the entire data processing pipeline.

```yaml
# ThesisExample/configs/data-config.yaml
ingestion:
- name: "firm"
  path: "firm_synthetic.csv" # Path relative to project_root/data/raw
  format: "csv"
  date_column: { "date": "%Y%m%d" }
  to_lowercase_cols: true
- name: "macro"
  path: "macro_synthetic.csv" # Path relative to project_root/data/raw
  format: "csv"
  date_column: { "date": "%Y%m%d" }
  to_lowercase_cols: true

wrangling_pipeline:
- operation: "monthly_imputation"
  dataset: "firm"
  numeric_columns: [ "volume", "marketcap" ]
  categorical_columns: []
  output_name: "imputed_firm"

- operation: "scale_to_range"
  dataset: "imputed_firm"
  range: { min: -1, max: 1 }
  cols_to_scale: [ "volume", "marketcap" ]
  output_name: "scaled_firm"

- operation: "merge"
  left_dataset: "scaled_firm"
  right_dataset: "macro"
  "on": [ "date" ]
  how: "left"
  output_name: "firm_and_macro"

- operation: "lag"
  dataset: "firm_and_macro"
  periods: 1
  columns_to_lag:
  - method: "all_except"
  - columns: [ "date", "permco", "return" ]
  drop_original_cols_after_lag: true
  restore_names: true
  drop_generated_nans: true
  output_name: "lagged_final"

- operation: "create_macro_interactions"
  dataset: "lagged_final"
  macro_columns: [ "gdp_growth", "cpi", "unemployment" ]
  firm_columns: [ "volume", "marketcap" ]
  drop_macro_columns: true
  output_name: "interactions_dataset"

export:
- dataset_name: "lagged_final"
  output_filename_base: "final_dataset"
  format: "parquet"
  partition_by: "year" # 'year' and 'none' are supported
```

**Important:** For this example to work, you need to copy the generated `firm_synthetic.csv` and `macro_synthetic.csv` files into your project's raw data directory (e.g., `ThesisExample/data/raw/`).

```bash
# Assuming you are in paper-data/examples/synthetic_data
cp firm_synthetic.csv ../../ThesisExample/data/raw/
cp macro_synthetic.csv ../../ThesisExample/data/raw/
```

### 3. Running the Data Pipeline

You can run the data pipeline using the `paper-data`'s standalone `run_pipeline.py` script. This script expects the `ThesisExample` project directory to be a sibling of the `paper-data` directory (i.e., both are direct children of your monorepo root).

```bash
# Assuming you are in the monorepo root, e.g., 'PAPER/'
python paper-data/src/paper_data/run_pipeline.py
```

### 4. Expected Output

**Console Output:**

You will see minimal output on the console, primarily indicating the start and successful completion of the data phase, along with the path to the detailed logs.

```
Attempting to load config from: /path/to/your/PAPER_monorepo/ThesisExample/configs/data-config.yaml
Using project root: /path/to/your/PAPER_monorepo/ThesisExample
Detailed logs will be written to: /path/to/your/PAPER_monorepo/ThesisExample/logs.log

Data pipeline completed successfully. Additional information in '/path/to/your/PAPER_monorepo/ThesisExample/logs.log'
```

**`ThesisExample/logs.log` Content (Snippet):**

The `logs.log` file will contain detailed information about each step of the pipeline, including data ingestion, wrangling operations, and export.

```log
2025-06-12 12:30:01,123 - paper_data.run_pipeline - INFO - Starting data pipeline execution via run_pipeline.py script.
2025-06-12 12:30:01,124 - paper_data.run_pipeline - INFO - Config path: /path/to/your/PAPER_monorepo/ThesisExample/configs/data-config.yaml
2025-06-12 12:30:01,125 - paper_data.run_pipeline - INFO - Project root: /path/to/your/PAPER_monorepo/ThesisExample
2025-06-12 12:30:01,126 - paper_data.manager - INFO - Running data pipeline for project: /path/to/your/PAPER_monorepo/ThesisExample
2025-06-12 12:30:01,127 - paper_data.manager - INFO - --- Ingesting Data ---
2025-06-12 12:30:01,128 - paper_data.manager - INFO - For dataset 'macro', 'id_col' is not specified in config. Using 'date' as a fallback. Ensure this is appropriate for your data.
2025-06-12 12:30:01,129 - paper_data.ingestion.local - INFO - Info: Date column 'date' was numeric, cast to string for parsing.
2025-06-12 12:30:01,130 - paper_data.manager - INFO - Dataset 'firm' ingested. Shape: (125, 5)
2025-06-12 12:30:01,131 - paper_data.manager - INFO - Dataset 'macro' ingested. Shape: (25, 4)
2025-06-12 12:30:01,132 - paper_data.manager - INFO - --- Wrangling Data ---
2025-06-12 12:30:01,133 - paper_data.manager - INFO - Performing monthly imputation on dataset 'firm'...
2025-06-12 12:30:01,134 - paper_data.wrangling.cleaner - INFO - Imputing numeric columns by monthly median: ['volume', 'marketcap']
2025-06-12 12:30:01,135 - paper_data.wrangling.cleaner - INFO -   Column 'volume' has 6 nulls to fill.
2025-06-12 12:30:01,136 - paper_data.wrangling.cleaner - INFO -   Column 'marketcap' has no nulls to fill.
2025-06-12 12:30:01,137 - paper_data.manager - INFO - Monthly imputation complete. Resulting shape: (125, 5)
2025-06-12 12:30:01,138 - paper_data.manager - INFO - Performing monthly scaling on dataset 'imputed_firm' for columns ['volume', 'marketcap']...
2025-06-12 12:30:01,139 - paper_data.wrangling.cleaner - INFO - Scaling columns ['volume', 'marketcap'] to range [-1.0, 1.0]...
2025-06-12 12:30:01,140 - paper_data.manager - INFO - Scaling complete. Resulting shape: (125, 5)
2025-06-12 12:30:01,141 - paper_data.manager - INFO - Dataset 'imputed_firm' after wrangling. Shape: (125, 5)
2025-06-12 12:30:01,142 - paper_data.manager - INFO - Dataset 'scaled_firm' after wrangling. Shape: (125, 5)
2025-06-12 12:30:01,143 - paper_data.manager - INFO - Merging datasets on columns: ['date'] with how='left'
2025-06-12 12:30:01,144 - paper_data.wrangling.augmenter - INFO - Merge complete. Resulting shape: (125, 8)
2025-06-12 12:30:01,145 - paper_data.manager - INFO - Dataset 'firm_and_macro' after wrangling. Shape: (125, 8)
2025-06-12 12:30:01,146 - paper_data.manager - INFO - Performing lag operation on dataset 'firm_and_macro' for columns: ['volume', 'marketcap', 'gdp_growth', 'cpi', 'unemployment'] with periods=1...
2025-06-12 12:30:01,147 - paper_data.wrangling.augmenter - INFO - Sorting DataFrame by 'permco' and 'date' for panel operations.
2025-06-12 12:30:01,148 - paper_data.wrangling.augmenter - INFO - Dropping rows with NaN values generated by lagging in columns: ['volume_lag_1', 'marketcap_lag_1', 'gdp_growth_lag_1', 'cpi_lag_1', 'unemployment_lag_1']
2025-06-12 12:30:01,149 - paper_data.wrangling.augmenter - INFO - Dropped 5 rows due to generated NaNs.
2025-06-12 12:30:01,150 - paper_data.wrangling.augmenter - INFO - Dropping original columns after lagging: ['volume', 'marketcap', 'gdp_growth', 'cpi', 'unemployment']
2025-06-12 12:30:01,151 - paper_data.wrangling.augmenter - INFO - Restoring original column names: ['volume_lag_1', 'marketcap_lag_1', 'gdp_growth_lag_1', 'cpi_lag_1', 'unemployment_lag_1'] -> ['volume', 'marketcap', 'gdp_growth', 'cpi', 'unemployment']
2025-06-12 12:30:01,152 - paper_data.wrangling.augmenter - INFO - Lag operation complete. Resulting shape: (120, 8)
2025-06-12 12:30:01,153 - paper_data.manager - INFO - Dataset 'lagged_final' after wrangling. Shape: (120, 8)
2025-06-12 12:30:01,154 - paper_data.manager - INFO - Creating macro-firm interaction columns for dataset 'lagged_final'...
2025-06-12 12:30:01,155 - paper_data.wrangling.augmenter - INFO - Created 6 new interaction columns.
2025-06-12 12:30:01,156 - paper_data.wrangling.augmenter - INFO - Dropping original macro columns: ['gdp_growth', 'cpi', 'unemployment']
2025-06-12 12:30:01,157 - paper_data.wrangling.augmenter - INFO - Interaction creation complete. Resulting shape: (120, 11)
2025-06-12 12:30:01,158 - paper_data.manager - INFO - Dataset 'interactions_dataset' after wrangling. Shape: (120, 11)
2025-06-12 12:30:01,159 - paper_data.manager - INFO - --- Exporting Data ---
2025-06-12 12:30:01,160 - paper_data.manager - INFO - Exporting 'lagged_final' by year to separate files:
2025-06-12 12:30:01,161 - paper_data.manager - INFO -   Exported data for year 2024 to '/path/to/your/PAPER_monorepo/ThesisExample/data/processed/final_dataset_2024.parquet'.
2025-06-12 12:30:01,162 - paper_data.manager - INFO -   Exported data for year 2025 to '/path/to/your/PAPER_monorepo/ThesisExample/data/processed/final_dataset_2025.parquet'.
2025-06-12 12:30:01,163 - paper_data.manager - INFO - Data pipeline completed successfully.
2025-06-12 12:30:01,164 - paper_data.run_pipeline - INFO - Data pipeline completed successfully.
2025-06-12 12:30:01,165 - paper_data.run_pipeline - INFO - --- Final Processed Datasets ---
2025-06-12 12:30:01,166 - paper_data.run_pipeline - INFO - Dataset 'lagged_final':
2025-06-12 12:30:01,167 - paper_data.run_pipeline - INFO -   Shape: (120, 8)
2025-06-12 12:30:01,168 - paper_data.run_pipeline - INFO -   Columns: ['date', 'permco', 'return', 'volume', 'marketcap', 'gdp_growth', 'cpi', 'unemployment']
2025-06-12 12:30:01,169 - paper_data.run_pipeline - INFO - Head shape: (5, 8)
2025-06-12 12:30:01,170 - paper_data.run_pipeline - INFO - Head:
shape: (5, 8)
┌────────────┬────────┬────────┬───────────┬───────────┬────────────┬───────────┬──────────────┐
│ date       ┆ permco ┆ return ┆ volume    ┆ marketcap ┆ gdp_growth ┆ cpi       ┆ unemployment │
│ ───────────┆ ────────┆ ────────┆ ───────────┆ ───────────┆ ───────────┆ ───────────┆ ──────────────│
│ date       ┆ i64    ┆ f64    ┆ f64       ┆ f64       ┆ f64        ┆ f64       ┆ f64          │
╞════════════╪════════╪════════╪═══════════╪═══════════╪════════════╪═══════════╪══════════════╡
│ 2024-01-31 ┆ 1      ┆ 0.09   ┆ -0.000000 ┆ -0.000000 ┆ 0.000000   ┆ 0.000000  ┆ 0.000000     │
│ 2024-01-31 ┆ 2      ┆ 0.15   ┆ -0.000000 ┆ -0.000000 ┆ 0.000000   ┆ 0.000000  ┆ 0.000000     │
│ 2024-01-31 ┆ 3      ┆ 0.02   ┆ -0.000000 ┆ -0.000000 ┆ 0.000000   ┆ 0.000000  ┆ 0.000000     │
│ 2024-01-31 ┆ 4      ┆ 0.18   ┆ -0.000000 ┆ -0.000000 ┆ 0.000000   ┆ 0.000000  ┆ 0.000000     │
│ 2024-01-31 ┆ 5      ┆ 0.05   ┆ -0.000000 ┆ -0.000000 ┆ 0.000000   ┆ 0.000000  ┆ 0.000000     │
└────────────┴────────┴────────┴───────────┴───────────┴────────────┴───────────┴──────────────┘
2025-06-12 12:30:01,171 - paper_data.run_pipeline - INFO - ------------------------------
```

### 5. Processed Data Output

After successful execution, you will find the processed Parquet files in your project's `data/processed` directory:

```
ThesisExample/data/processed/
├── final_dataset_2024.parquet
└── final_dataset_2025.parquet
```

---

## ⚙️ Configuration Reference

The `data-config.yaml` file is the heart of `paper-data`. Here's a breakdown of its main sections:

### `ingestion`

A list of datasets to ingest. Each item defines a source:

*   `name` (string, required): A unique identifier for the dataset within the pipeline.
*   `path` (string, required): Relative path to the raw data file (from `project_root/data/raw/`).
*   `format` (string, required): Currently supports `"csv"`.
*   `date_column` (object, required): Specifies the date column and its format. E.g., `{ "date": "%Y%m%d" }`.
*   `to_lowercase_cols` (boolean, optional, default: `false`): Whether to convert all column names to lowercase after ingestion.

### `wrangling_pipeline`

A sequential list of operations to apply to your datasets. Operations are applied in the order they appear.

*   **`operation: "monthly_imputation"`**
    *   `dataset` (string, required): The name of the dataset to apply imputation to.
    *   `numeric_columns` (list of strings, optional): Columns to impute with monthly cross-sectional median.
    *   `categorical_columns` (list of strings, optional): Columns to impute with monthly cross-sectional mode.
    *   `output_name` (string, required): The name for the resulting dataset.
*   **`operation: "scale_to_range"`**
    *   `dataset` (string, required): The name of the dataset to scale.
    *   `range` (object, required): Defines the target `min` and `max` for scaling. E.g., `{ min: -1, max: 1 }`.
    *   `cols_to_scale` (list of strings, required): Numeric columns to apply min-max scaling to.
    *   `output_name` (string, required): The name for the resulting dataset.
*   **`operation: "dummy_generation"`**
    *   `dataset` (string, required): The name of the dataset to use.
    *   `column_to_dummy` (string, required): The name of the categorical column to convert into dummy variables.
    *   `drop_original_col` (boolean, optional, default: `false`): If `true`, the original categorical column is removed from the output.
    *   `output_name` (string, required): The name for the resulting dataset.
*   **`operation: "merge"`**
    *   `left_dataset` (string, required): The name of the left dataset.
    *   `right_dataset` (string, required): The name of the right dataset.
    *   `on` (list of strings, required): Columns to merge on (e.g., `[ "date", "permco" ]`).
    *   `how` (string, required): Type of join (`"left"`, `"inner"`, `"outer"`, `"full"`).
    *   `output_name` (string, required): The name for the resulting merged dataset.
*   **`operation: "lag"`**
    *   `dataset` (string, required): The name of the dataset to apply lagging to.
    *   `periods` (integer, required): Number of periods to shift (positive for lag, negative for lead).
    *   `columns_to_lag` (list of objects, required): Specifies which columns to lag.
        *   `method` (string, required): Currently only `"all_except"` is supported.
        *   `columns` (list of strings, required): Columns to *exclude* from lagging when `method` is `"all_except"`.
    *   `drop_original_cols_after_lag` (boolean, optional, default: `false`): If `true`, original columns are dropped.
    *   `restore_names` (boolean, optional, default: `false`): If `true` and `drop_original_cols_after_lag` is `true`, lagged columns are renamed to their original names.
    *   `drop_generated_nans` (boolean, optional, default: `false`): If `true`, rows with NaNs introduced by lagging are dropped.
    *   `output_name` (string, required): The name for the resulting dataset.
*   **`operation: "create_macro_interactions"`**
    *   `dataset` (string, required): The name of the dataset containing both macro and firm columns.
    *   `macro_columns` (list of strings, required): Columns identified as macro characteristics.
    *   `firm_columns` (list of strings, required): Columns identified as firm characteristics.
    *   `drop_macro_columns` (boolean, optional, default: `false`): If `true`, original macro columns are dropped after interaction.
    *   `output_name` (string, required): The name for the resulting dataset.

### `export`

A list of processed datasets to export.

*   `dataset_name` (string, required): The name of the dataset (from `ingestion` or `wrangling_pipeline` outputs) to export.
*   `output_filename_base` (string, required): The base name for the output file(s).
*   `format` (string, required): Currently supports `"parquet"`.
*   `partition_by` (string, optional): How to partition the output. Supports `"year"` (creates `_YYYY.parquet` files) or `"none"` (single file).

---

## 🤝 Contributing

We welcome contributions to `paper-data`! If you have suggestions for new data connectors, wrangling operations, or performance improvements, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes and write tests.
4.  Commit your changes (`git commit -am 'Add new feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Create a new Pull Request.

---

## 📄 License

`paper-data` is distributed under the MIT License. See the `LICENSE` file for more information.

---