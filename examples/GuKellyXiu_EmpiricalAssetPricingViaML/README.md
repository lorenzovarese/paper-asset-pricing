# GuKellyXiu_EmpiricalAssetPricingViaML

A P.A.P.E.R (Platform for Asset Pricing Experimentation and Research) project.

Initialized on: 2025-31-01
P.A.P.E.R Tools Version: 0.1.0

Replication of: "Empirical Asset Pricing via Machine Learning" (2021) by Shihao Gu, Bryan T. Kelly, and Dacheng Xiu

Full citation:
 Gu, Shihao and Kelly, Bryan T. and Xiu, Dacheng, Empirical Asset Pricing via Machine Learning (September 13, 2019). Chicago Booth Research Paper No. 18-04, 31st Australasian Finance and Banking Conference 2018, Yale ICF Working Paper No. 2018-09, Available at SSRN: https://ssrn.com/abstract=3159577 or http://dx.doi.org/10.2139/ssrn.3159577 

## Project Structure

- `configs/paper-project.yaml`: Main project configuration.
- `configs/`: Directory for component-specific YAML configurations:
    - `data-config.yaml`: For `paper-data` processing.
    - `models-config.yaml`: For `paper-model` tasks.
    - `portfolio-config.yaml`: For `paper-portfolio` strategies.
- `data/`: Data storage.
    - `raw/`: Place for raw input data.
    - `processed/`: Output for processed data from `paper-data`.
- `models/`: Model-related files.
    - `saved/`: Output for trained models from `paper-model`.
- `portfolios/`: Portfolio-related files.
    - `results/`: Output for portfolio backtests/results from `paper-portfolio`.
- `logs.log`: Project-level log file.
- `.gitignore`: Specifies files for Git to ignore.
- `README.md`: This file.

## Getting Started

1.  **Navigate to the project directory:**
    ```bash
    cd "GuKellyXiu_EmpiricalAssetPricingViaML"
    ```

2.  **Set up your Python environment** and install P.A.P.E.R components:
    ```bash
    # Example:
    # uv venv
    # source .venv/bin/activate
    pip install paper-data paper-model paper-portfolio # Or use paper-asset-pricing[all]
    ```

3.  **Create Component Configurations:**
    - In the `configs/` directory, create and populate:
        - `data-config.yaml` (for `paper-data`)
        - `models-config.yaml` (for `paper-model`)
        - `portfolio-config.yaml` (for `paper-portfolio`)
    - Refer to the documentation of each P.A.P.E.R component for its specific YAML structure.
    - **Example for `data-config.yaml`:**
      ```yaml
      # configs/data-config.yaml
      sources:
        - name: my_firm_data
          connector: local
          path: "data/raw/your_firm_data.csv" # Relative to project root
          # ... other source parameters
        # - name: my_macro_data ...
      transformations:
        # - type: clean_numeric ...
        # - type: one_hot ...
      output:
        format: parquet
        # directory: "data/processed" # Often inferred by paper-data
        filename_prefix: "master_dataset"
      ```


4.  **Place Raw Data:**
    - Put your raw data files into the `data/raw/` directory.

5.  **Execute Project Phases:**
    Use `paper execute` from the project root:
    ```bash
    paper execute data      # Runs the data processing phase
    paper execute models    # Runs the modeling phase
    paper execute portfolio # Runs the portfolio phase
    ```
    You can also run them sequentially:
    ```bash
    paper execute data && paper execute models && paper execute portfolio
    ```

Refer to `configs/paper-project.yaml` to see how `paper` locates component configurations and CLI tools.