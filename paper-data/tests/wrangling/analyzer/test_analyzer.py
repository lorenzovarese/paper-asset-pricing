import pandas as pd
from ydata_profiling import ProfileReport
from paper_data.wrangling.analyzer.analyzer import analyze_dataframe


def test_analyze_dataframe_returns_profile_report():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    report = analyze_dataframe(df, title="Test Report")
    assert isinstance(report, ProfileReport)
