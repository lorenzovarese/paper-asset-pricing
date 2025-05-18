import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from typing import Optional, List


def monthly_portfolio_backtest(
    model,
    train_df: DataFrame,
    test_df: DataFrame,
    *,
    features: List[str],
    date_col: str,
    signal_col: str,
    ret_col: str,
    long_lower_q: float = 0.90,
    long_upper_q: Optional[float] = None,
    short_lower_q: Optional[float] = None,
    short_upper_q: float = 0.10,
    weight_mode: str = "equal",
    weight_col: Optional[str] = None,
    val_years: int = 0,
) -> DataFrame:
    """
    As before, but if `long_upper_q` is given, we
    take the band [long_lower_q, long_upper_q];
    otherwise top tail >= long_lower_q.
    Similarly for shorts: if `short_lower_q` is given,
    we take [short_lower_q, short_upper_q], else bottom <= short_upper_q.
    """
    train = train_df.copy()
    test = test_df.copy()
    train[date_col] = pd.to_datetime(train[date_col])
    test[date_col] = pd.to_datetime(test[date_col])

    # carve out validation
    train_end = train[date_col].max()
    val_end = train_end + pd.DateOffset(years=val_years)
    calib = pd.concat([train, test[test[date_col] <= val_end]], ignore_index=True)
    forecast = test[test[date_col] > val_end].copy()

    # fit once
    model.fit(calib[features].values, calib[ret_col].values)

    forecast["ym"] = forecast[date_col].dt.to_period("M")
    out = []
    for ym in sorted(forecast["ym"].unique()):
        sub = forecast[forecast["ym"] == ym]
        if sub.empty:
            continue

        # predict
        sub = sub.copy()
        sub[signal_col] = model.predict(sub[features].values)

        # longs
        if long_upper_q is not None:
            lo_thr = sub[signal_col].quantile(long_lower_q)
            hi_thr = sub[signal_col].quantile(long_upper_q)
            long_leg = sub[(sub[signal_col] >= lo_thr) & (sub[signal_col] <= hi_thr)]
        else:
            lo_thr = sub[signal_col].quantile(long_lower_q)
            long_leg = sub[sub[signal_col] >= lo_thr]

        # shorts
        if short_lower_q is not None:
            lo_s = sub[signal_col].quantile(short_lower_q)
            hi_s = sub[signal_col].quantile(short_upper_q)
            short_leg = sub[(sub[signal_col] >= lo_s) & (sub[signal_col] <= hi_s)]
        else:
            hi_s = sub[signal_col].quantile(short_upper_q)
            short_leg = sub[sub[signal_col] <= hi_s]

        # compute returns
        if weight_mode == "equal":
            lr = long_leg[ret_col].mean() if not long_leg.empty else 0.0
            sr = short_leg[ret_col].mean() if not short_leg.empty else 0.0
        else:
            w_l = long_leg[weight_col]
            w_s = short_leg[weight_col]
            lr = (long_leg[ret_col] * w_l / w_l.sum()).sum() if w_l.sum() > 0 else 0.0
            sr = (short_leg[ret_col] * w_s / w_s.sum()).sum() if w_s.sum() > 0 else 0.0

        out.append(
            {
                date_col: sub[date_col].iloc[0],
                "long_ret": lr,
                "short_ret": sr,
                "port_ret": lr - sr,
            }
        )

    return pd.DataFrame(out)


def compute_performance_metrics(
    returns: Series, periods_per_year: int = 12
) -> dict[str, float]:
    r = returns.dropna()
    ann_ret = r.mean() * periods_per_year
    ann_vol = r.std(ddof=0) * np.sqrt(periods_per_year)
    # force exact zero if the computed vol is just floating-point noise
    if np.isclose(ann_vol, 0.0):
        ann_vol = 0.0
        sharpe = np.nan
    else:
        sharpe = ann_ret / ann_vol

    cum = (1 + r).cumprod()
    drawdown = (cum / cum.cummax() - 1).min()

    return {
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(drawdown),
    }
