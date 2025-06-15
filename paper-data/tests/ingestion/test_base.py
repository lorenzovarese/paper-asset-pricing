import polars as pl
from paper_data.ingestion.base import DataConnector  # type: ignore


class DummyConnector(DataConnector):
    def get_data(self):
        return pl.DataFrame({"x": [1, 2, 3]})


def test_call_and_repr():
    d = DummyConnector()
    df = d()
    assert isinstance(df, pl.DataFrame)
    assert "<DummyConnector>" in repr(d)
