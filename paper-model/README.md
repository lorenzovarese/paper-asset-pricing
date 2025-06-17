# paper-model: Advanced Model Implementation & Evaluation for Asset Pricing 🧠

[![codecov](https://codecov.io/github/lorenzovarese/paper-asset-pricing/graph/badge.svg?token=ZUDEPEPJFK)](https://codecov.io/github/lorenzovarese/paper-asset-pricing)
[![PyPI version](https://badge.fury.io/py/paper-model.svg)](https://badge.fury.io/py/paper-model)

`paper-model` is a powerful and extensible component of the P.A.P.E.R (Platform for Asset Pricing Experimentation and Research) toolchain. It is the engine for implementing, training, evaluating, and managing a wide array of asset pricing models, from classic linear regressions to deep neural networks.

Its primary objective is to bridge the gap between the clean, processed data from `paper-data` and the portfolio construction phase in `paper-portfolio`, by providing robust model evaluations and generating actionable out-of-sample predictions.

---

## ✨ Features

`paper-model` provides a comprehensive, configuration-driven framework for quantitative researchers, enabling the replication and extension of sophisticated asset pricing studies.

*   **Broad Model Support:** 🏗️
    *   **Linear Models**: `ols` (Ordinary Least Squares), `enet` (Elastic Net), `pcr` (Principal Component Regression), and `pls` (Partial Least Squares).
    *   **Non-Linear & Tree-Based Models**: `glm` (Generalized Linear Models with splines and Group Lasso), `rf` (Random Forest), and `gbrt` (Gradient Boosted Regression Trees).
    *   **Deep Learning**: `nn` (Feed-Forward Neural Networks) with extensive regularization and ensembling options.
*   **Advanced Feature Implementation:** ⚙️
    *   **Robust Objective Functions**: Support for both standard `l2` (least squares) and robust `huber` loss across most model types.
    *   **Adaptive Hyperparameter Tuning**: A validation-set-driven grid search to find optimal hyperparameters (e.g., regularization strength, number of components, tree depth) for each rolling window.
    *   **Sample Weighting**: OLS implementation supports weighting by `inv_n_stocks` (inverse number of stocks per month) or `mkt_cap` (market capitalization).
    *   **Specialized Regularization**: Implements Group Lasso for GLMs and a full suite of NN regularization techniques (L1 penalty, Early Stopping, Batch Normalization, and Ensembling).
*   **Rigorous Model Evaluation:** 📊
    *   Implements a standard rolling-window methodology for true out-of-sample testing.
    *   Calculates standard asset pricing metrics like out-of-sample R² (`r2_oos`) and Mean Squared Error (`mse`).
    *   Generates detailed evaluation reports and time-series metrics for each model.
*   **Reproducible, Configuration-Driven Workflow:** 📝
    *   Define all aspects of the modeling pipeline—data inputs, evaluation windows, and all model specifications—declaratively in a single `models-config.yaml` file.
    *   Ensures perfect reproducibility and simplifies experimentation.
*   **Seamless Integration:** 🔗
    *   Designed to work hand-in-hand with `paper-data` for input and `paper-portfolio` for downstream portfolio construction.
    *   Orchestrated by `paper-asset-pricing` for a unified command-line experience.

---

## 📦 Installation

`paper-model` is designed to be part of the larger `PAPER` monorepo.

**Recommended (as part of `paper-asset-pricing`):**

This method ensures `paper-model` is available to the main `paper` CLI orchestrator.

```bash
# Using pip
pip install "paper-asset-pricing[models]"

# Using uv
uv pip install "paper-asset-pricing[models]"
```

**Standalone Installation:**

If you only need `paper-model` and its core functionalities for a different project.

```bash
# Using pip
pip install paper-model

# Using uv
uv pip install paper-model
```

**From Source (for development within the monorepo):**

Navigate to the root of your `PAPER` monorepo and install `paper-model` in editable mode. This will also install all required dependencies like `scikit-learn`, `torch`, and `group-lasso`.

```bash
# Using pip
pip install -e ./paper-model

# Using uv
uv pip install -e ./paper-model
```

---

## 📖 Usage Workflow

The typical workflow for `paper-model` involves:

1.  **Data Preparation:** Use `paper-data` to process your raw financial data. The resulting Parquet files in `data/processed/` are the direct input for `paper-model`.
2.  **Configuration:** Define your entire experiment in the `models-config.yaml` file. This includes the evaluation window, metrics, and a list of all models to be trained and compared.
3.  **Model Execution:** Run the models phase using the `paper-asset-pricing` CLI from your project's root directory:

    ```bash
    paper execute models
    ```
    This command triggers `paper-model` to:
    *   Load and validate the configuration.
    *   Iterate through each rolling window defined by your parameters.
    *   For each model and each window: perform hyperparameter tuning (if configured), train the best model, and generate out-of-sample predictions.
    *   Save evaluation reports, detailed metrics, prediction files, and optional model checkpoints.

4.  **Review Outputs:**
    *   Check `logs.log` for detailed execution information.
    *   Review summary evaluation reports in `models/evaluations/`.
    *   Analyze detailed, per-window metrics from the Parquet files in the same directory.
    *   Use the generated predictions from `models/predictions/` as input for the `paper-portfolio` stage.

---

## ⚙️ Configuration (`models-config.yaml`)

The `models-config.yaml` file is the heart of `paper-model`. It defines the entire experiment structure.

### Top-Level Configuration

*   **`input_data`**: Specifies the dataset name and key column identifiers.
*   **`evaluation`**: Defines the rolling window structure (`train_month`, `validation_month`, `testing_month`, `step_month`) and the list of metrics to compute (e.g., `r2_oos`, `mse`).

### Model Configuration

The `models` section is a list where each item defines a model to be run. Common keys for all models include `name`, `type`, `target_column`, `features`, `save_model_checkpoints`, and `save_prediction_results`.

#### OLS (`type: "ols"`)
*   **`weighting_scheme`**: `none` (default), `inv_n_stocks`, or `mkt_cap`.
*   **`market_cap_column`**: Required if `weighting_scheme` is `mkt_cap`.
*   **`objective_function`**: `l2` (default) or `huber`.
*   **`huber_epsilon_quantile`**: If using Huber loss, sets the `epsilon` adaptively based on a quantile of residuals (e.g., `0.999`).

#### Elastic Net (`type: "enet"`)
*   **`alpha`**: Regularization strength (λ). Can be a float or a list for tuning (e.g., `[0.01, 0.1, 1.0]`).
*   **`l1_ratio`**: Mixing parameter (ρ). Can be a float or a list for tuning (e.g., `[0.1, 0.5, 0.9]`).
*   **`objective_function`**: `l2` (default) or `huber`.

#### PCR & PLS (`type: "pcr"`, `type: "pls"`)
*   **`n_components`**: Number of components (K). Can be an integer or a list for tuning (e.g., `[5, 10, 15]`).
*   **`objective_function`**: `l2` or `huber` (for the final regression step in PCR).

#### Generalized Linear Model (`type: "glm"`)
*   **`n_knots`**: Number of knots for the quadratic spline transformer. Fixed value (e.g., `3`).
*   **`alpha`**: Group Lasso regularization strength (λ). Can be a float or a list for tuning.
*   **`objective_function`**: Must be `l2`.

#### Random Forest (`type: "rf"`)
*   **`n_estimators`**: Number of trees (B). Typically a fixed integer (e.g., `300`).
*   **`max_depth`**: Tree depth (L). Can be an integer or a list for tuning.
*   **`max_features`**: Features per split. Can be a string (`"sqrt"`), float, or list for tuning.

#### Gradient Boosted Trees (`type: "gbrt"`)
*   **`n_estimators`**: Number of trees (B). Can be an integer or a list for tuning.
*   **`max_depth`**: Tree depth (L). Can be an integer or a list for tuning.
*   **`learning_rate`**: Shrinkage parameter (ν). Can be a float or a list for tuning.
*   **`objective_function`**: `l2` or `huber`.
*   **`use_hist_implementation`**: `true` to use scikit-learn's faster `HistGradientBoostingRegressor`.

#### Neural Network (`type: "nn"`)
*   **`hidden_layer_sizes`**: A tuple defining the architecture (e.g., `[32, 16, 8]`).
*   **`alpha`**: L1 penalty (λ). Can be a float or a list for tuning.
*   **`learning_rate`**: Adam optimizer learning rate. Can be a float or a list for tuning.
*   **`batch_size`**: e.g., `10000`.
*   **`epochs`**: Max epochs, e.g., `100`.
*   **`patience`**: For early stopping, e.g., `5`.
*   **`n_ensembles`**: Number of models to train and average, e.g., `10`.

---

## 🤝 Contributing

We welcome contributions to `paper-model`! If you have suggestions for new models, evaluation techniques, or architectural improvements, please feel free to open an issue or submit a pull request.

---

## 📄 License

`paper-model` is distributed under the MIT License. See the `LICENSE` file for more information.

---