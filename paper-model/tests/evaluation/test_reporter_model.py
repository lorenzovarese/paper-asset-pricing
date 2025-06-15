import polars as pl
from paper_model.evaluation.reporter import EvaluationReporter  # type: ignore


def test_generate_text_report(tmp_path):
    out = tmp_path / "reports"
    reporter = EvaluationReporter(out)

    metrics = {
        "mse": 0.1234,
        "r2_oos": [0.9, 0.8, 0.85],
    }
    reporter.generate_text_report("mymodel", metrics)

    rpt = out / "mymodel_evaluation_report.txt"
    assert rpt.exists()
    text = rpt.read_text()
    assert "--- Model Evaluation Report: mymodel ---" in text
    assert "mse: 0.1234" in text
    # average of r2_oos = 0.85
    assert "r2_oos: 0.8500" in text


def test_save_metrics_to_parquet(tmp_path):
    out = tmp_path / "reports"
    reporter = EvaluationReporter(out)

    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    reporter.save_metrics_to_parquet("mymodel", data)

    pq = out / "mymodel_evaluation_metrics.parquet"
    assert pq.exists()
    df = pl.read_parquet(pq)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_save_metrics_empty(tmp_path, caplog):
    out = tmp_path / "reports"
    reporter = EvaluationReporter(out)

    reporter.save_metrics_to_parquet("mymodel", [])
    assert "No metrics data to save" in caplog.text
    assert not (out / "mymodel_evaluation_metrics.parquet").exists()
