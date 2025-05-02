import pandas as pd
import pytest
import requests  # type: ignore[import-untyped]

from paper_data.ingestion.http import HTTPConnector  # type: ignore[import-untyped]
from paper_data.ingestion.local import LocalConnector  # type: ignore[import-untyped]


class DummyResponse:
    def __init__(self, data: bytes):
        self._data = data

    def raise_for_status(self):
        # Simulate a successful status
        pass

    def iter_content(self, chunk_size=1):
        yield self._data


class ErrResponse(DummyResponse):
    def raise_for_status(self):
        # Simulate HTTP error on status
        raise requests.HTTPError("download failed")


def test_get_data_csv(monkeypatch, tmp_path):
    # Prepare a sample CSV file
    sample_csv = tmp_path / "data.csv"
    sample_csv.write_text("a,b\n1,2\n3,4")

    # Monkey-patch requests.get to return our dummy CSV bytes
    monkeypatch.setattr(
        "paper_data.ingestion.http.requests.get",
        lambda url, stream, timeout: DummyResponse(sample_csv.read_bytes()),
    )
    # Monkey-patch LocalConnector to read from our sample file
    monkeypatch.setattr(
        LocalConnector,
        "get_data",
        lambda self: pd.read_csv(sample_csv),
    )

    conn = HTTPConnector("https://example.com/data.csv")
    df = conn.get_data()
    expected = pd.DataFrame({"a": [1, 3], "b": [2, 4]})
    pd.testing.assert_frame_equal(df, expected)


def test_get_data_parquet(monkeypatch, tmp_path):
    # Prepare a sample Parquet file
    df_orig = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
    sample_pq = tmp_path / "data.parquet"
    df_orig.to_parquet(sample_pq)

    # Monkey-patch requests.get to return our dummy Parquet bytes
    monkeypatch.setattr(
        "paper_data.ingestion.http.requests.get",
        lambda url, stream, timeout: DummyResponse(sample_pq.read_bytes()),
    )
    # Monkey-patch LocalConnector to read from our sample file
    monkeypatch.setattr(
        LocalConnector,
        "get_data",
        lambda self: pd.read_parquet(sample_pq),
    )

    conn = HTTPConnector("http://example.org/data.parquet")
    df = conn.get_data()
    pd.testing.assert_frame_equal(df, df_orig)


def test_download_http_error(monkeypatch):
    # Simulate an HTTP error response
    monkeypatch.setattr(
        "paper_data.ingestion.http.requests.get",
        lambda url, stream, timeout: ErrResponse(b""),
    )
    conn = HTTPConnector("https://fail.example/file.csv")
    with pytest.raises(requests.HTTPError):
        conn.get_data()
