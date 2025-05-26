from paper_data.ingestion.base import BaseConnector  # type: ignore[import-untyped]


class DummyConnector(BaseConnector):
    def get_data(self):
        return "dummy"


def test_get_data_and_call():
    conn = DummyConnector()
    # The real implementation would return a DataFrame, but here we just return a string for simplicity.
    assert conn.get_data() == "dummy"
    assert conn() == "dummy"  # type: ignore[return-value]


def test_repr():
    conn = DummyConnector()
    assert repr(conn) == "<DummyConnector>"
