import pandas as pd
from ydata_profiling import ProfileReport  # type: ignore[import-untyped]
import pytest
from paper_data.wrangling.analyzer.analyzer import analyze_dataframe  # type: ignore[import-untyped]


def test_analyze_dataframe_returns_profile_report():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    report = analyze_dataframe(df, title="Test Report")
    assert isinstance(report, ProfileReport)


def test_analyze_dataframe_limits_rows_with_n_rows(monkeypatch):
    df = pd.DataFrame({"num": list(range(10))})
    captured = {}

    class DummyReport:
        def __init__(self, df_arg, **kwargs):
            captured["df"] = df_arg

        # no other methods required for initialization

    # Monkey-patch ProfileReport to capture the DataFrame passed
    monkeypatch.setattr(
        "paper_data.wrangling.analyzer.analyzer.ProfileReport",
        DummyReport,
    )
    analyze_dataframe(df, n_rows=5)
    assert "df" in captured
    # Ensure only the last 5 rows were used
    assert list(captured["df"]["num"]) == list(range(5, 10))


def test_analyze_dataframe_explorative_with_small_dataframe():
    df = pd.DataFrame({"x": list(range(100))})
    report = analyze_dataframe(df, explorative=True)
    assert isinstance(report, ProfileReport)


def test_analyze_dataframe_explorative_too_many_rows_raises():
    df = pd.DataFrame({"x": list(range(1001))})
    with pytest.raises(
        ValueError, match="Explorative mode is limited to 1000 rows or fewer"
    ):
        analyze_dataframe(df, explorative=True)
