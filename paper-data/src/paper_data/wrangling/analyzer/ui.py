from typing import Optional
import webbrowser
from pathlib import Path
import polars as pl


from .analyzer import analyze_dataframe  # type: ignore[import-untyped]


def display_report(
    df: pl.DataFrame | str,
    output_path: str = "data_profile.html",
    max_rows: Optional[int] = None,
    minimal: bool = True,
    explorative: bool = False,
) -> Path:
    """
    Accept either a DataFrame or path to a CSV/Parquet file, produce
    an HTML profiling report, save it under PAPER/, and open it in the default browser.

    If `max_rows` is provided and the dataset exceeds it, only the last `max_rows` rows are profiled.
    Explorative mode is automatically capped at 1000 rows by analyse_dataframe.

    Returns:
    - Path to the generated report.

    Raises:
    - ValueError: If explorative is True and the effective row count exceeds 1000.
    """
    # 1) Load file if a filepath is provided
    if isinstance(df, (str, Path)):
        ext = Path(df).suffix.lower()
        df = (
            pl.read_csv(df)
            if ext == ".csv"
            else pl.read_parquet(df)
            if ext in {".parquet", ".pq"}
            else pl.read_csv(df)
        )

    # 2) Generate profile (analyze_dataframe will handle max_rows and explorative cap)
    report = analyze_dataframe(
        df,
        minimal=minimal,
        explorative=explorative,
        n_rows=max_rows,
    )

    # 3) Save report and open in browser
    out_dir = Path("PAPER")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / output_path
    report.to_file(out)

    # Convert to absolute file URI so webbrowser.open() works
    uri = out.resolve().as_uri()
    webbrowser.open(uri)

    return out
