# paper-asset-pricing: P.A.P.E.R.
Platform for Asset Pricing Experiment and Research

```bash
├── core/           # abstract interfaces
│   ├─ model.py
│   └─ portfolio.py         # abstract PortfolioConstructor
├── datasets/       # CRSP loader stub (user‑provided CSVs) - connectors (WRDS, OSAP, CRSP, custom CSV)
├── models/         # OLS, OLS‑3, Ridge, Lasso, Enet, XGBoost, NN, RNN, LSTM - concrete implementations (CAPM, FF3, ML…)
├── portfolios/             # concrete portfolio implementations
│   ├─ long_short_90.py
│   ├─ long_short_95.py
│   └─ …  
├── pipelines/      # CLI or Prefect/airflow DAGs
├── metrics/        # Sharpe, RMSE, GRS, α‑t‑stats, out-of-sample R² …
├── experiments/    # YAML specs wiring data+model+metrics
└─ ui/              # optional Streamlit or REST interface
```

### Troubleshooting

Mock data generation:
```bash
uv run python -m scripts.generate_mock_data
```