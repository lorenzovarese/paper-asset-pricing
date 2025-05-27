import webbrowser
from pathlib import Path
import pandas as pd

from .analyzer import analyze_dataframe


def display_report(
    df: pd.DataFrame | str, output_path: str = "data_profile.html"
) -> Path:
    """
    Accept either a DataFrame or path to a CSV/Parquet file, produce
    an HTML profiling report, save it, and open it in the default browser.

    Returns the Path to the generated report.
    """
    # 1) Load if given a filename
    if isinstance(df, (str, Path)):
        ext = Path(df).suffix.lower()
        df = (
            pd.read_csv(df)
            if ext == ".csv"
            else pd.read_parquet(df)
            if ext in {".parquet", ".pq"}
            else pd.read_csv(df)  # fallback
        )

    # 2) Analyze
    profile = analyze_dataframe(df)

    # 3) Save & open
    out = Path(output_path)
    profile.to_file(out)
    webbrowser.open(out.as_uri())

    return out
