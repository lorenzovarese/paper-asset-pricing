import polars as pl
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
