# Examples of paper-data CLI

This tutorial demonstrates how to use the `paper-data` CLI to ingest, profile, and clean your datasets. We'll walk through creating a YAML configuration file, running ingestion, generating profiling reports, and cleaning data with pre-defined pipelines.

## Create the config

First, scaffold a configuration for the **ingest** command. Create a `PAPER/data/config.yaml` file with your dataset specifications:

```yaml
# PAPER/data/config.yaml

# Base output directory for all ingested files
o utput_dir: PAPER/data

datasets:
  - name: welch_goyal_monthly
    connector: http
    params:
      url: "https://docs.google.com/spreadsheets/d/1OIZg6htTK60wtnCVXvxAujvG1aKEOVYv/export?format=csv&gid=1660564386"
    save_as: welch_goyal_monthly.csv

  - name: datashare
    connector: local
    params:
      path: "PAPER/datashare.zip"
      member_name: "datashare.csv"
    save_as: datashare.csv

  - name: crsp_monthly_returns
    connector: wrds
    params:
      query: |
        SELECT permno, date, ret
          FROM crsp.msf
         WHERE date BETWEEN '1957-01-01' AND '2021-12-31'
      user: "${WRDS_USER:-}"
      password: prompt
    save_as: crsp_monthly_returns.parquet
```

* **`output_dir`**: target directory for downloads
* **`datasets`**: list each dataset with

  * `name`: identifier
  * `connector`: one of `http`, `local`, `wrds`, etc.
  * `params`: connection-specific arguments
  * `save_as`: desired filename

## Run the following examples

1. **Ingest all datasets**

   ```bash
   uv run paper-data ingest \
       --config PAPER/data/config.yaml
   ```

   You should see progress output and files saved under `PAPER/data/`:

   ```text
   [ingest] → welch_goyal_monthly via http
   [ingest] ✔ PAPER/data/welch_goyal_monthly.csv
   [ingest] → datashare via local
   [ingest] ✔ PAPER/data/datashare.csv
   [ingest] → crsp_monthly_returns via wrds
   Enter your WRDS password: ********
   [ingest] ✔ PAPER/data/crsp_monthly_returns.parquet
   ```

2. **Generate a profiling report**

   ```bash
   uv run paper-data profile \
       PAPER/data/welch_goyal_monthly.csv \
       --output welch_profile.html \
       --n-rows 500 \
       --explorative
   ```

   This creates `PAPER/welch_profile.html` and opens it in your browser.

3. **Clean a dataset**

   First, define a cleaning pipeline in `PAPER/data/cleaning.yaml`:

   ```yaml
   cleaning:
     firm:
       - normalize_columns: {}
       - rename_date_column:
           candidates: ["yyyymm"]
           target: date
       - parse_date:
           date_format: "%Y%m"
           monthly_option: "start"
       - clean_numeric_column:
           col: "value"
       - impute_constant:
           cols: ["feature1", "feature2"]
           value: 0
   ```

   Then run:

   ```bash
   uv run paper-data clean \
       PAPER/data/some_raw.csv \
       --config PAPER/data/cleaning.yaml \
       --objective firm
   ```

   Outputs land in `PAPER/cleaned/some_raw_cleaned.parquet`.
