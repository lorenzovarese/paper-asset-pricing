import pandas as pd
from ydata_profiling import ProfileReport  # type: ignore[import-untyped]
from typing import Optional


def analyze_dataframe(
    df: pd.DataFrame,
    title: str = "PAPER Data Profile",
    minimal: bool = True,
    explorative: bool = False,
    n_rows: Optional[int] = None,
) -> ProfileReport:
    """
    Generate and return a ProfileReport for `df`, which includes:
      - shape & memory usage
      - missing values
      - type inference (heterogeneous columns)
      - descriptive statistics
      - interactive visuals per column

    Parameters:
    - df: DataFrame to analyze.
    - title: Title of the report.
    - minimal: If True, uses minimal mode.
    - explorative: If True, includes explorative analysis.
    - n_rows: If provided, limits the dataset to the last `n_rows` rows.

    Raises:
    - ValueError: If explorative is True and the number of rows exceeds 1000.
    """
    # Limit to last n_rows if specified
    if n_rows is not None and len(df) > n_rows:
        # Ensure a copy to avoid SettingWithCopyWarning in profiling
        df = df.tail(n_rows).copy()

    # Prevent explorative mode on large datasets
    if explorative and len(df) > 1000:
        raise ValueError("Explorative mode is limited to 1000 rows or fewer.")

    report = ProfileReport(
        df,
        title=title,
        explorative=explorative,
        minimal=minimal,
        correlations={"pearson": {"calculate": True}},
        html={"style": {"full_width": True}},
    )
    return report
