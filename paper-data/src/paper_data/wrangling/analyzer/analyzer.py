import pandas as pd
from ydata_profiling import ProfileReport


def analyze_dataframe(
    df: pd.DataFrame, title: str = "PAPER Data Profile", explorative: bool = True
) -> ProfileReport:
    """
    Generate and return a ProfileReport for `df`, which includes:
      - shape & memory usage
      - missing values
      - type inference (heterogeneous columns)
      - descriptive statistics
      - interactive visuals per column
    """
    report = ProfileReport(
        df,
        title=title,
        explorative=explorative,
        correlations={"pearson": {"calculate": True}},
        html={"style": {"full_width": True}},
    )
    return report
