import pandas as pd
import pytest
from paper_data.ingestion.google_drive import (  # type: ignore[import-untyped]
    GoogleDriveConnector,
    _extract_file_id,
)


class DummyResponse:
    def __init__(self, data: bytes):
        self._data = data

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=1):
        yield self._data


class DummySession:
    def __init__(self, data):
        self.data = data

    def get(self, url, stream, timeout):
        return DummyResponse(self.data)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_extract_file_id_valid():
    url = "https://drive.google.com/file/d/ABCDEFGHIJ1234567890/view?usp=sharing"
    assert _extract_file_id(url) == "ABCDEFGHIJ1234567890"


def test_extract_file_id_invalid():
    with pytest.raises(ValueError):
        _extract_file_id("https://example.com/no-id")


def test_download_and_get_data(monkeypatch, tmp_path):
    # create a small CSV and point LocalConnector at it
    sample = tmp_path / "data.csv"
    sample.write_text("x,y\n1,2\n3,4")
    monkeypatch.setenv("PAPERASSETPRICING_LOCAL_PATH", str(sample))
    # monkeypatch requests.Session
    monkeypatch.setattr(
        "paper_data.ingestion.google_drive.requests.Session",
        lambda: DummySession(sample.read_bytes()),
    )
    # monkeypatch LocalConnector to read our sample
    from paper_data.ingestion.local import LocalConnector  # type: ignore[import-untyped]

    monkeypatch.setattr(LocalConnector, "get_data", lambda self: pd.read_csv(sample))
    conn = GoogleDriveConnector("https://drive.google.com/file/d/ABCDEFGHIJ/view")
    df = conn.get_data()
    assert list(df.columns) == ["x", "y"]
