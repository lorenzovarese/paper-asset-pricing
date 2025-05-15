import yaml
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import typer


class PerformanceBuilder:
    def __init__(
        self,
        data_path: Path,
        date_col: str,
        id_col: str,
        return_col: str,
        signal_col: str,
        n_quantiles: int,
        long_q: int,
        short_q: int,
        out_dir: Path,
    ):
        self.data_path = data_path
        self.date_col = date_col
        self.id_col = id_col
        self.return_col = return_col
        self.signal_col = signal_col
        self.nq = n_quantiles
        self.long_q = long_q
        self.short_q = short_q
        self.out_dir = out_dir
        self._df: Optional[pd.DataFrame] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "PerformanceBuilder":
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        dc = cfg["dataset"]
        pc = cfg["portfolio"]
        od = Path(cfg["output"]["dir"])
        return cls(
            data_path=Path(dc["path"]),
            date_col=dc.get("date_column", "date"),
            id_col=dc.get("id_column", "permno"),
            return_col=dc["return_column"],
            signal_col=dc["signal_column"],
            n_quantiles=pc.get("n_quantiles", 10),
            long_q=pc.get("long_quantile", 10),
            short_q=pc.get("short_quantile", 1),
            out_dir=od,
        )

    def _load(self) -> pd.DataFrame:
        """Load the entire dataset into memory once."""
        if self._df is None:
            if self.data_path.suffix.lower() in (".pq", ".parquet"):
                df = pd.read_parquet(self.data_path)
            else:
                df = pd.read_csv(self.data_path, parse_dates=[self.date_col])
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            self._df = df
        return self._df

    def build(self) -> pd.DataFrame:
        """
        Compute monthly long/short portfolio returns.
        Returns a DataFrame with columns:
          [date, long_return, short_return, excess_return]
        """
        df = self._load()

        if self.signal_col not in df.columns:
            available = df.columns.tolist()
            raise typer.BadParameter(
                f"Signal column '{self.signal_col}' not found in data. "
                f"Available columns: {available}"
            )

        results = []
        for dt, group in df.groupby(self.date_col):
            # copy so we don’t clobber the master DataFrame
            g = group.copy()

            # assign to quantiles by signal strength
            g["q"] = pd.qcut(
                g[self.signal_col],
                q=self.nq,
                labels=range(1, self.nq + 1),
            )

            long_ret = g.loc[g["q"] == self.long_q, self.return_col].mean()
            short_ret = g.loc[g["q"] == self.short_q, self.return_col].mean()

            results.append(
                {
                    self.date_col: dt,
                    "long_return": long_ret,
                    "short_return": short_ret,
                }
            )

        perf = pd.DataFrame(results).sort_values(self.date_col).reset_index(drop=True)
        perf["excess_return"] = perf["long_return"] - perf["short_return"]
        return perf

    def save_csv(self, perf_df: pd.DataFrame, filename: Optional[Path] = None) -> Path:
        """Save the built performance DataFrame to CSV."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        out = filename or (self.out_dir / "portfolio_monthly.csv")
        perf_df.to_csv(out, index=False)
        return out

    def plot_cumulative(
        self, perf_df: pd.DataFrame, filename: Optional[Path] = None
    ) -> Path:
        """
        Plot cumulative log‐returns of the long vs. short portfolios.
        Saves a figure and returns its Path.
        """
        # set index for time‐series plotting
        ts = perf_df.set_index(self.date_col)

        # cumulative log returns: log(1 + r) cumulatively summed
        cum_long = np.log1p(ts["long_return"]).cumsum()
        cum_short = np.log1p(ts["short_return"]).cumsum()

        fig, ax = plt.subplots()
        ax.plot(cum_long.index, cum_long.values, label="Long Decile")
        ax.plot(cum_short.index, cum_short.values, label="Short Decile")
        ax.legend()
        ax.set_title("Cumulative Log‐Returns: Long vs Short")
        ax.set_xlabel(self.date_col)
        ax.set_ylabel("Cumulative log return")

        self.out_dir.mkdir(parents=True, exist_ok=True)
        out = filename or (self.out_dir / "portfolio_cumulative.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return out
