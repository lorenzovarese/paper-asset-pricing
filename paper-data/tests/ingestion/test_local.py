import pytest
import pandas as pd
from paper_data.ingestion.local import LocalConnector  # type: ignore[import-untyped]


def test_read_csv(tmp_path):
    data = pd.DataFrame({"a": [1, 2, 3]})
    csv_path = tmp_path / "test.csv"
    data.to_csv(csv_path, index=False)
    conn = LocalConnector(str(csv_path))
    df = conn.get_data()
    pd.testing.assert_frame_equal(df, data)


def test_read_parquet(tmp_path):
    data = pd.DataFrame({"b": [4.0, 5.5]})
    pq_path = tmp_path / "test.parquet"
    data.to_parquet(pq_path)
    conn = LocalConnector(pq_path)
    df = conn.get_data()
    pd.testing.assert_frame_equal(df, data)


def test_file_not_found(tmp_path):
    missing = tmp_path / "no.csv"
    conn = LocalConnector(missing)
    with pytest.raises(FileNotFoundError):
        conn.get_data()


def test_unsupported_suffix(tmp_path):
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("hello")
    conn = LocalConnector(txt_path)
    with pytest.raises(ValueError):
        conn.get_data()
