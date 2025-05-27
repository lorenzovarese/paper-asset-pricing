import pandas as pd
import tempfile
import pytest
from pathlib import Path
from paper_data.wrangling.analyzer.ui import display_report


def test_display_report_from_dataframe(monkeypatch):
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})

    # Prevent actual browser from opening
    monkeypatch.setattr(
        "paper_data.wrangling.analyzer.ui.webbrowser.open", lambda url: True
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "report.html"
        result_path = display_report(df, output_path=str(output_path))
        assert result_path.exists()
        assert result_path.name == "report.html"


def test_display_report_from_csv_path(monkeypatch, tmp_path):
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        "paper_data.wrangling.analyzer.ui.webbrowser.open", lambda url: True
    )

    output_html = tmp_path / "report.html"
    result_path = display_report(str(csv_path), output_path=str(output_html))

    assert result_path.exists()
    assert result_path.name == "report.html"


def test_display_report_from_parquet_path(monkeypatch, tmp_path):
    df = pd.DataFrame({"a": [10, 20], "b": [30, 40]})
    parquet_path = tmp_path / "data.parquet"
    df.to_parquet(parquet_path)

    monkeypatch.setattr(
        "paper_data.wrangling.analyzer.ui.webbrowser.open", lambda url: True
    )

    output_html = tmp_path / "report_from_parquet.html"
    result_path = display_report(str(parquet_path), output_path=str(output_html))

    assert result_path.exists()
    assert result_path.name == "report_from_parquet.html"


def test_display_report_invalid_path(monkeypatch, tmp_path):
    bad_path = tmp_path / "unknown_format.xyz"
    bad_path.write_text("irrelevant content")

    monkeypatch.setattr(
        "paper_data.wrangling.analyzer.ui.webbrowser.open", lambda url: True
    )

    output_html = tmp_path / "should_fail.html"
    # Despite the fallback in UI, invalid content may raise downstream errors in ydata-profiling
    with pytest.raises(Exception):
        display_report(str(bad_path), output_path=str(output_html))
