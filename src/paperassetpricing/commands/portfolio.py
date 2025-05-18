import yaml
import typer
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from paperassetpricing.commands.experiment import (
    prepare_data_source,
    window_generator,
    load_window,
    get_features_and_target,
    instantiate_model,
)
from paperassetpricing.portfolios.performance import (
    monthly_portfolio_backtest,
    compute_performance_metrics,
)


app = typer.Typer(help="Portfolio construction & backtest commands")


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@app.command("run")
def run_portfolio(
    exp_config: Path = typer.Option(..., "-e", "--exp-config", exists=True),
    pf_config: Path = typer.Option(..., "-p", "--pf-config", exists=True),
) -> None:
    exp_cfg = load_config(exp_config)
    pf_cfg = load_config(pf_config)

    ds_conf = exp_cfg["dataset"]
    ev = exp_cfg["evaluation"]
    model = instantiate_model(exp_cfg)

    pf = pf_cfg["portfolio"]
    sig_col = pf["signal_column"]
    ret_col = pf["return_column"]
    long_lq = pf.get("long_lower_quantile", 0.90)
    long_uq = pf.get("long_upper_quantile")  # may be None
    short_lq = pf.get("short_lower_quantile")  # may be None
    short_uq = pf.get("short_upper_quantile", 0.10)
    weight_md = pf.get("weight_mode", "equal")
    weight_col = pf.get("weight_column")
    out_dir = Path(pf["output_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(ds_conf["path"])
    date_col = ds_conf.get("date_column", "date")
    is_pq, ds, start, end, all_cols = prepare_data_source(data_path, date_col)
    features, _ = get_features_and_target(exp_cfg, all_cols)
    windows = window_generator(
        start,
        end,
        ev["train_years"],
        ev["val_years"],
        ev["test_years"],
        ev["roll_years"],
    )

    all_monthly = []
    for w in windows:
        typer.echo(f"Window {w[0].date()}→{w[2].date()}: backtesting…")

        cols = [date_col] + features.copy()
        if weight_md == "value" and weight_col and weight_col not in cols:
            cols.append(weight_col)

        train_df, test_df = load_window(
            is_pq,
            ds,
            data_path,
            date_col,
            cols,  # must include date_col & any weight_col
            ret_col,
            w,
        )

        monthly = monthly_portfolio_backtest(
            model=model,
            train_df=train_df,
            test_df=test_df,
            features=features,
            date_col=date_col,
            signal_col=sig_col,
            ret_col=ret_col,
            long_lower_q=long_lq,
            long_upper_q=long_uq,
            short_lower_q=short_lq,
            short_upper_q=short_uq,
            weight_mode=weight_md,
            weight_col=weight_col,
            val_years=ev["val_years"],
        )
        all_monthly.append(monthly)

    df_monthly = pd.concat(all_monthly, ignore_index=True).sort_values(date_col)

    # save returns & perf
    csv_ret = out_dir / "portfolio_monthly_returns.csv"
    csv_perf = out_dir / "portfolio_performance.csv"
    df_monthly.to_csv(csv_ret, index=False)
    pd.Series(compute_performance_metrics(df_monthly["port_ret"])).to_frame(
        "value"
    ).to_csv(csv_perf)
    typer.secho(f"✅ Returns → {csv_ret}", fg=typer.colors.GREEN)
    typer.secho(f"✅ Metrics → {csv_perf}", fg=typer.colors.GREEN)

    # cumulative plot
    df_monthly["cum_ret"] = (1 + df_monthly["port_ret"]).cumprod() - 1
    png = out_dir / "cumulative_returns.png"
    plt.figure(figsize=(10, 6))
    plt.plot(df_monthly[date_col], df_monthly["cum_ret"], label="Portfolio")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Portfolio Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png)
    plt.close()
    typer.secho(f"✅ Plot → {png}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()

"""
# Example usage:
  paper portfolio \
  -e configs/experiment/OLS3_20years.yaml \
  -p configs/portfolio/portfolio_config_VAL95.yaml
"""
