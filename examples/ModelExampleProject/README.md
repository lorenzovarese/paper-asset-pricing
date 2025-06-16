# ModelExampleProject

This project serves as a comprehensive example of the **`paper-model`** component within the **P.A.P.E.R** (Platform for Asset Pricing Experimentation and Research) framework.

The primary goal of this example is to demonstrate how to configure and run a large-scale, comparative study of multiple machine learning models for asset pricing. It showcases the tool's core functionalities, including its rolling-window backtesting engine, automated hyperparameter tuning, and the generation of predictive outputs for a wide array of algorithms.

- **Initialized on:** 2025-06-01
- **P.A.P.E.R Tools Version:** 0.1.0

---

## 📖 Workflow Overview

This example focuses exclusively on the modeling phase of the P.A.P.E.R workflow. It assumes that the data processing has already been completed.

1.  **Data Setup:** The project uses a pre-processed, anonymized dataset that mimics the structure of real financial panel data.
2.  **Configure the Models:** The `models-config.yaml` file is extensively configured to train and evaluate nine different models, from simple linear regressions to deep neural networks, many with automated hyperparameter tuning.
3.  **Execute the Pipeline:** A single `paper execute models` command triggers the entire backtesting and evaluation process.

---

## 🚀 Getting Started

This guide assumes you have cloned the `paper-asset-pricing` monorepo and have set up the Python environment as described in the main `README.md`.

### 1. Data Setup

This example uses a pre-processed and anonymized dataset to focus on the modeling capabilities. The data is already included in the project's `data/processed/` directory as a set of year-partitioned Parquet files (e.g., `final_dataset_model_1992.parquet`). No data generation or processing steps are required.

### 2. Review the Model Configuration

The heart of this example is the `configs/models-config.yaml` file. It provides a detailed specification for a comparative study of nine different predictive models.

**Key aspects of the configuration include:**

-   **Rolling Window:** A 10-year training window, a 2-year validation window, and a 1-year testing window, which slides forward by one year at each step.
-   **Model Suite:**
    -   `ols_huber_adaptive`: A robust OLS model using an adaptive Huber loss.
    -   `elastic_net_model`: A regularized linear model with fixed hyperparameters.
    -   `pcr_tuned_model` & `pls_tuned_model`: Dimensionality reduction models with automated tuning of the number of components.
    -   `glm_tuned_model`: A non-linear model with splines and Group Lasso regularization, with tuning for the penalty strength.
    -   `random_forest_tuned` & `gbrt_huber_tuned`: Tree-based ensembles with extensive hyperparameter tuning.
    -   `nn1_model` & `nn3_model`: Shallow and deep neural networks with tuning for regularization and learning rate, and a multi-faceted regularization strategy including ensembling and early stopping.
-   **Outputs:** Most models are configured to save both their out-of-sample predictions and their trained model checkpoints for each window.

You can review the full, annotated configuration file in the `configs/` directory to see the detailed setup for each model.

### 3. Execute the Model Pipeline

From the root of the `ModelExampleProject` directory, run the model execution command. Note that this is a computationally intensive process, as it involves training and tuning nine different models over multiple rolling windows.

```bash
paper execute models
```

---

## ✅ Expected Output

After the command finishes, the `models/` directory will be populated with a rich set of artifacts.

**Console Output:**

The console will show a simple success message. All detailed output is captured in `logs.log`.

```
>>> Executing Models Phase <<<
Auto-detected project root: /path/to/ModelExampleProject
Models phase completed successfully. Additional information in '/path/to/ModelExampleProject/logs.log'
```

**Log File (`logs.log`):**

The `logs.log` file provides a verbose, step-by-step account of the entire backtest. For each rolling window, you can see:
- Which data files are being loaded.
- The start of training for each of the nine models.
- The results of the hyperparameter tuning for models like `pcr_tuned_model` (e.g., `Best params for 'pcr_tuned_model': {'pcr_pipeline__pca__n_components': 10}`).
- Early stopping messages for the neural network ensembles.
- Confirmation that model checkpoints and predictions are being saved.

**Output Artifacts:**

The `models/` directory will contain the following subdirectories filled with results:

-   `models/predictions/`: A set of Parquet files (e.g., `elastic_net_model_predictions.parquet`, `nn3_model_predictions.parquet`) containing the out-of-sample return predictions for each stock and month. These files are the primary input for the `paper-portfolio` phase.
-   `models/evaluations/`:
    -   A Parquet file for each model (e.g., `pcr_tuned_model_evaluation_metrics.parquet`) containing the performance metrics (`mse`, `r2_oos`, etc.) for each individual testing period.
    -   A text report for each model (e.g., `pcr_tuned_model_evaluation_report.txt`) with aggregated performance statistics.
-   `models/saved/`: A collection of `.joblib` files containing the saved model objects for each training window, which can be loaded later for inspection or custom analysis.