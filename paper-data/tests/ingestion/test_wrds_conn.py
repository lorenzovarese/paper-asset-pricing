import pandas as pd
from paper_data.ingestion.wrds_conn import WRDSConnector  # type: ignore[import-untyped]


class DummyConnection:
    def __init__(self, **kwargs):
        pass

    def raw_sql(self, query):
        return pd.DataFrame({"x": [10, 20]})

    def close(self):
        pass


def test_get_data(monkeypatch):
    monkeypatch.setattr(
        "paper_data.ingestion.wrds_conn.wrds.Connection",
        lambda **kw: DummyConnection(),
    )
    conn = WRDSConnector("SELECT 1 AS x", user="u", password="p", max_rows=1)
    df = conn.get_data()
    pd.testing.assert_frame_equal(df, pd.DataFrame({"x": [10, 20]}))
