import polars as pl
import pytest
import requests  # type: ignore[import-untyped]

from paper_data.ingestion.http import HTTPConnector  # type: ignore[import-untyped]


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


def test_download_http_error(monkeypatch):
    # Simulate an HTTP error response
    monkeypatch.setattr(
        "paper_data.ingestion.http.requests.get",
        lambda url, stream, timeout: ErrResponse(b""),
    )
    conn = HTTPConnector("https://fail.example/file.csv")
    with pytest.raises(requests.HTTPError):
        conn.get_data()
