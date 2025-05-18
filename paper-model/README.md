# paper-model: Model Implementation & Evaluation for Asset Pricing 🧠

`paper-model` is a crucial component of the P.A.P.E.R (Platform for Asset Pricing Experimentation and Research) monorepo. It is designed to serve as the central hub for implementing, training, evaluating, and managing various asset pricing models.

Its primary objective is to bridge the gap between the clean, processed data from `paper-data` and the portfolio construction phase in `paper-portfolio`, by providing robust model evaluations and generating actionable checkpoints.

---

## ✨ Goals & Features

`paper-model` aims to provide a flexible and extensible framework for quantitative researchers:

*   **Model Implementation:** 🏗️
    *   Support for implementing a wide range of asset pricing models, from traditional factor models (e.g., Fama-French) to more advanced machine learning approaches for return prediction.
    *   Modular design to easily add new model architectures.
*   **Comprehensive Model Evaluation:** 📊
    *   Evaluate model performance against a suite of relevant metrics (e.g., R-squared, information ratio, Sharpe ratio, statistical significance of factors).
    *   Generate detailed evaluation reports in a standardized format (e.g., text files, tables).
    *   Facilitate comparison across different models and specifications.
*   **Checkpoint Generation for Portfolio Construction:** 💾
    *   Produce "checkpoints" – structured outputs (e.g., predicted returns, factor exposures, model weights) that are directly consumable by the `paper-portfolio` module.
    *   These checkpoints will serve as the foundation for constructing long-short portfolios and analyzing their cross-sectional performance.
*   **Configuration-Driven Workflow:** ⚙️
    *   Define model specifications, training parameters, evaluation criteria, and checkpointing logic declaratively via a `models-config.yaml` file.
    *   Ensures reproducibility and simplifies experimentation by allowing researchers to easily switch between model setups.
*   **Seamless Integration:** 🔗
    *   Designed to work hand-in-hand with `paper-data` for input data and `paper-portfolio` for downstream analysis.
    *   Orchestrated by `paper-tools` for a unified command-line experience.

---

## 🚧 Status: Under Development

`paper-model` is currently under active development. While the core architecture is being laid out, specific model implementations and advanced evaluation functionalities are being built. We are committed to delivering a robust and feature-rich module.

---

## 📦 Installation

`paper-model` is designed to be part of the larger `PAPER` monorepo. You can install it as an optional dependency of `paper-tools` or as a standalone package.

**Recommended (as part of `paper-tools`):**

If you have `paper-tools` installed, you can get `paper-model` and its dependencies using the `models` extra:

```bash
pip install paper-tools[models]
# Or if using uv:
uv pip install paper-tools[models]
```

**Standalone Installation:**

If you only need `paper-model` and its core functionalities (once available), you can install it directly:

```bash
pip install paper-model
# Or if using uv:
uv pip install paper-model
```

**From Source (for development within the monorepo):**

Navigate to the root of your `PAPER` monorepo and install `paper-model` in editable mode:

```bash
cd /path/to/your/PAPER_monorepo
uv pip install -e ./paper-model
```

---

## 📖 Intended Usage Example

The typical workflow for `paper-model` will involve:

1.  **Data Preparation:** Use `paper-data` (orchestrated by `paper execute data`) to process your raw financial data into a clean, ready-to-use format. This processed data will be the input for `paper-model`.
2.  **Configuration:** Create and populate your `models-config.yaml` file (e.g., `ThesisExample/configs/models-config.yaml`). This file will specify:
    *   Which processed dataset(s) from `paper-data` to use.
    *   The type of model to train (e.g., "Fama-French 3-Factor", "Linear Regression", "Random Forest").
    *   Model-specific parameters (e.g., factor definitions, hyper-parameters).
    *   Evaluation metrics to compute.
    *   Output paths for evaluation reports and checkpoints.
3.  **Model Execution:** Run the models phase using the `paper-tools` CLI:

    ```bash
    # From your project root (e.g., ThesisExample/)
    paper execute models
    ```
    This command will trigger `paper-model` to load the configuration, train the specified models, evaluate them, and save the results and checkpoints.

4.  **Review Outputs:**
    *   Check the `logs.log` file in your project root for detailed execution information.
    *   Review the generated evaluation reports (e.g., in `ThesisExample/models/evaluations/`).
    *   Access the model checkpoints (e.g., in `ThesisExample/models/saved/`) which will then be used by `paper-portfolio`.

---

## ⚙️ Configuration (`models-config.yaml`)

The `models-config.yaml` file will define the models to be trained and evaluated. While the exact schema is evolving, it will likely include sections for:

*   **`input_data`**: Reference to the processed dataset(s) from `paper-data`.
*   **`models`**: A list of model definitions, each with:
    *   `name`: Unique identifier for the model run.
    *   `type`: The class or function implementing the model (e.g., `linear_regression`, `ff3_factor`).
    *   `parameters`: Model-specific arguments (e.g., `features`, `target`, `hyperparameters`).
*   **`evaluation`**: Metrics to compute (e.g., `r_squared`, `sharpe_ratio`, `t_stats`).
*   **`checkpointing`**: Settings for saving model outputs for `paper-portfolio` (e.g., `output_path`, `format`).

---

## 🤝 Contributing

We welcome contributions to `paper-model`! As this module is under active development, your input can significantly shape its future. If you have suggestions for new models, evaluation techniques, or architectural improvements, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourModel`).
3.  Make your changes and write tests.
4.  Commit your changes (`git commit -am 'Add new model type'`).
5.  Push to the branch (`git push origin feature/YourModel`).
6.  Create a new Pull Request.

---

## 📄 License

`paper-model` is distributed under the MIT License. See the `LICENSE` file for more information.

---