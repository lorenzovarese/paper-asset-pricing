# paper-data

**PAPER** (Platform for Asset Pricing Experimentation and Research) is a modular toolkit for data ingestion, profiling, cleaning and transformation in asset pricing applications. The **paper-data** package provides the data-related functionality. For model building, see the companion package **paper-model**.

---

## Features

- **Data ingestion** from diverse sources:
  - HTTP(S) URLs (CSV or Parquet)
  - Local files (CSV, Parquet, ZIP)
  - Google Drive links
  - WRDS (Wharton Research Data Services)
  - Hugging Face Hub
- **Data profiling** via an interactive HTML report (powered by **ydata-profiling**)
- **Data cleaning** pipelines for firm-level and macroeconomic series
- **Programmable CLI** for reproducible workflows
- **Schema validation** (via **pandera**) for firm and macro datasets

---

## Installation

```bash
pip install paper-data
````

You will also need:

* Python 3.11+
* `pandas`, `tqdm`, `requests`
* For profiling: `ydata-profiling>=5.0`
* For WRDS ingestion: the `wrds` package and valid credentials
* For Hugging Face ingestion: `datasets>=2.0`

---

## Quick Start

1. **Ingest** one or more datasets

   ```bash
   paper-data ingest --config path/to/config.yaml
   ```
2. **Profile** a raw dataset

   ```bash
   paper-data profile data/raw/my_data.csv \
     --output my_data_profile.html \
     --n-rows 1000 \
     --explorative
   ```
3. **Clean** according to a YAML pipeline

   ```bash
   paper-data clean data/raw/my_data.csv \
     --config path/to/cleaning.yaml \
     --output-dir data/cleaned \
     --objective firm
   ```

All outputs are placed under a top-level `PAPER/` directory by default.

---

## Examples

See the `examples/` directory for full scripts:

* **Ingestion**:
  `examples/ingestion/example_ingestion.py` demonstrates downloading:

  * Welch & Goyal predictors (HTTP CSV)
  * Empirical Asset Pricing ML dataset (manual download)
  * CRSP monthly returns (WRDS)
* **Profiling**:
  `examples/wrangling/example_analyze.py` shows how to generate a profiling report on `sample_data.csv`.

---

## Command-Line Interface

The `paper-data` CLI exposes three commands:

### `ingest`

```bash
paper-data ingest --config CONFIG_PATH
```

* **CONFIG\_PATH**: YAML file defining one or more datasets:

  ```yaml
  output_dir: PAPER/data
  datasets:
    - name: welch_goyal
      connector: http
      params:
        url: https://…
      save_as: welch.csv
    - name: crsp
      connector: wrds
      params:
        query: "SELECT permno, date, ret FROM crsp.msf …"
        user: prompt
        password: prompt
  ```

### `profile`

```bash
paper-data profile SOURCE_PATH [--output REPORT.html]
                   [--n-rows N] [--explorative] [--no-minimal]
```

* **SOURCE\_PATH**: CSV or Parquet file to profile.
* Generates an HTML report under `PAPER/`.

### `clean`

```bash
paper-data clean SOURCE_PATH --config CLEAN.yaml
                   [--output-dir OUTPUT_DIR]
                   [--objective firm|macro]
```

* Applies a sequence of cleaning steps defined in `CLEAN.yaml`, for `"firm"` or `"macro"` objectives.
* Writes cleaned Parquet files to `OUTPUT_DIR`.

---

## Core API

### Ingestion Connectors

* **BaseConnector**: abstract interface.
* **HTTPConnector**: download from HTTP(S).
* **LocalConnector**: read CSV/Parquet or extract from ZIP.
* **GoogleDriveConnector**: fetch shared Drive files.
* **WRDSConnector**: query WRDS via SQL.
* **HuggingFaceConnector**: load from Hugging Face Hub.

Usage example:

```python
from paper_data.ingestion.http import HTTPConnector

df = HTTPConnector("https://…/data.csv").get_data()
```

### Data Profiling

* **analyze\_dataframe**: returns a `ProfileReport`.
* **display\_report**: save HTML and open in browser.

```python
from paper_data.wrangling.analyzer.ui import display_report

report_path = display_report("data.csv", output_path="report.html")
```

### Data Cleaning

* **RawDataset**: wraps DataFrame with objective (`"firm"` or `"macro"`).
* **CleanerFactory**: dispatches to `FirmCleaner` or `MacroCleaner`.
* **BaseCleaner**: column normalization, date parsing, numeric coercion, constant imputation.
* **FirmCleaner**: cross-section median/mean/mode imputation.
* **MacroCleaner**: (future extensions).

Example pipeline:

```yaml
cleaning:
  firm:
    - normalize_columns: {}
    - rename_date_column:
        candidates: ["yyyymm"]
        target: "date"
    - parse_date:
        date_format: "%Y%m"
        monthly_option: start
    - impute_cross_section_median:
        cols: ["feature1", "feature2"]
```

---

## Data Schemas

* **firm\_schema**: enforces `date`, `company_id`, `ret`, and numeric features.
* **macro\_schema**: enforces `date` and numeric macro variables.

Example:

```python
from paper_data.schema.firm import firm_schema

validated_df = firm_schema(raw_df)
```

---

## Testing

A comprehensive test suite is included under `tests/`. To run all tests:

```bash
pytest
```

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request. Ensure new functionality includes tests and documentation.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
