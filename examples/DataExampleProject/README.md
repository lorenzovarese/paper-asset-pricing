# DataExampleProject

This project demonstrates the end-to-end data processing pipeline of the `paper-data` component within the **P.A.P.E.R** (Platform for Asset Pricing Experimentation and Research) framework.

The goal of this example is to showcase how to ingest multiple raw data sources (synthetic firm and macro data), apply a sequence of cleaning and feature engineering steps, and export a final, analysis-ready dataset, all orchestrated through a single configuration file.

- **Initialized on:** 2025-06-01
- **P.A.P.E.R Tools Version:** 0.1.0

---

## 📖 Workflow Overview

This example follows a simple, three-step process:

1.  **Generate Raw Data:** Run Python scripts to create synthetic firm-level and macro-level CSV files.
2.  **Configure the Pipeline:** Review the `data-config.yaml` file, which defines every step of the data transformation process.
3.  **Execute the Pipeline:** Run a single `paper execute data` command to process the data and generate the final output.

---

## 🚀 Getting Started

This guide assumes you have cloned the `paper-asset-pricing` monorepo and have set up the Python environment as described in the main `README.md`.

### 1. Generate Raw Data

The raw data for this example is generated synthetically. The scripts are located in the `scripts/` directory of this project.

Navigate to the `scripts` directory and run the generation scripts:

```bash
# From the root of this project (DataExampleProject/)
cd scripts/

python firm_synthetic.py
# Expected output: Dataset saved to 'firm_synthetic.csv'

python macro_synthetic.py
# Expected output: Dataset saved to 'macro_synthetic.csv'
```

### 2. Place Raw Data Files

The `paper-data` pipeline expects raw data files to be in the `data/raw/` directory. Move the newly generated files there.

```bash
# From the scripts/ directory
mv *.csv ../data/raw/

# Navigate back to the project root
cd ..
```

Your `data/raw/` directory should now contain `firm_synthetic.csv` and `macro_synthetic.csv`.

### 3. Review the Configuration

The entire logic for the data pipeline is defined in `configs/data-config.yaml`. This file instructs the tool on what data to load, how to transform it, and what to export.

```yaml
# configs/data-config.yaml

# Ingestion section: defines the input datasets
ingestion:
- name: "firm" # Logical name for the dataset
  path: "firm_synthetic.csv" # Path to the CSV file relative to the data/raw directory
  format: "csv"
  date_column: { "date": "%Y%m%d" }
  firm_id_column: "permco"

- name: "macro" # Second dataset: macroeconomic data
  path: "macro_synthetic.csv"
  format: "csv"
  date_column: { "date": "%Y%m%d" }

# Wrangling pipeline: series of transformations applied to the ingested datasets
wrangling_pipeline:
- operation: "monthly_imputation"
  dataset: "firm"
  numeric_columns: [ "volume", "marketcap" ]
  output_name: "imputed_firm"

- operation: "scale_to_range"
  dataset: "imputed_firm"
  range: { min: -1, max: 1 }
  cols_to_scale: [ "volume", "marketcap" ]
  output_name: "scaled_firm"

- operation: "merge"
  left_dataset: "scaled_firm"
  right_dataset: "macro"
  on: [ "date" ]
  how: "left"
  output_name: "firm_and_macro"

- operation: "lag"
  dataset: "firm_and_macro"
  periods: 1
  columns_to_lag:
    - method: "all_except"
      columns: [ "date", "permco", "return" ]
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

# Export section: defines what processed data to save and how
export:
- dataset_name: "interactions_dataset"
  output_filename_base: "final_dataset"
  format: "parquet"
  partition_by: "year"
```

### 4. Execute the Data Pipeline

From the root of the `DataExampleProject` directory, run the data execution command:

```bash
paper execute data
```

---

## ✅ Expected Output

After the command finishes, you can verify the results.

**Console Output:**

The console will show a simple success message, directing you to the log file for details.

```
>>> Executing Data Phase <<<
Auto-detected project root: /path/to/DataExampleProject
Data phase completed successfully. Additional information in '/path/to/DataExampleProject/logs.log'
```

**Log File (`logs.log`):**

The `logs.log` file will contain a detailed, step-by-step record of the entire process, including the shape of the dataframe after each wrangling operation, confirming that all steps were executed as configured.

**Processed Data:**

The `data/processed/` directory will be populated with the final output files. Because `partition_by: "year"` was specified, the tool creates a separate Parquet file for each year in the dataset.

```
data/processed/
├── final_dataset_2024.parquet
└── final_dataset_2025.parquet
```

These Parquet files contain the fully processed data, including the generated interaction terms, and are ready to be used as input for the `paper-model` pipeline.