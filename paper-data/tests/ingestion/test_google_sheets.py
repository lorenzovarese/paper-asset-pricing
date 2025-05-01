import pandas as pd
from types import SimpleNamespace
from paper_data.ingestion.google_sheets import GoogleSheetsConnector  # type: ignore[import-untyped]


class DummyWorksheet:
    def __init__(self, records):
        self._records = records

    def get_all_records(self):
        return self._records


class DummySheet:
    def __init__(self, records):
        self.records = records

    def get_worksheet(self, idx):
        return DummyWorksheet(self.records)

    def worksheet(self, name):
        return DummyWorksheet(self.records)


def test_public_sheet(monkeypatch):
    data = [{"a": 1}, {"a": 2}]
    dummy = DummySheet(data)
    monkeypatch.setattr(
        "paper_data.ingestion.google_sheets.gspread",
        SimpleNamespace(public=lambda: SimpleNamespace(open_by_url=lambda url: dummy)),
    )
    conn = GoogleSheetsConnector("some_url")
    df = conn.get_data()
    pd.testing.assert_frame_equal(df, pd.DataFrame(data))


def test_named_worksheet(monkeypatch):
    data = [{"b": 3}]
    dummy = DummySheet(data)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    # simulate service account path None => public()
    monkeypatch.setattr(
        "paper_data.ingestion.google_sheets.gspread",
        SimpleNamespace(public=lambda: SimpleNamespace(open_by_url=lambda url: dummy)),
    )
    conn = GoogleSheetsConnector("url", worksheet_name="Sheet1")
    df = conn.get_data()
    pd.testing.assert_frame_equal(df, pd.DataFrame(data))


def test_welch_goyal_dataset_multi_sheets(monkeypatch):
    from types import SimpleNamespace
    import pandas as pd
    from paper_data.ingestion.google_sheets import GoogleSheetsConnector

    url = (
        "https://docs.google.com/spreadsheets/"
        "d/1OIZg6htTK60wtnCVXvxAujvG1aKEOVYv/edit?gid=1660564386"
    )

    # Common header definitions
    monthly_header = [
        "yyyymm",
        "Index",
        "D12",
        "E12",
        "b/m",
        "tbl",
        "AAA",
        "BAA",
        "lty",
        "ntis",
        "Rfree",
        "infl",
        "ltr",
        "corpr",
        "svar",
        "csp",
        "CRSP_SPvw",
        "CRSP_SPvwx",
    ]
    quarterly_header = ["yyyyq"] + monthly_header[1:]
    annual_header = ["yyyy"] + monthly_header[1:]

    # Dummy data for each sheet
    monthly = [{h: i for i, h in enumerate(monthly_header, start=1)}]
    quarterly = [{h: i for i, h in enumerate(quarterly_header, start=1)}]
    annual = [{h: i for i, h in enumerate(annual_header, start=1)}]

    class DummyWS:
        def __init__(self, records):
            self._records = records

        def get_all_records(self):
            return self._records

    class DummySheet:
        def __init__(self, sheets):
            self._sheets = sheets

        def get_worksheet(self, idx):
            return DummyWS(self._sheets[idx])

        def worksheet(self, name):
            return DummyWS(self._sheets[name])

    sheets_map = {
        0: monthly,
        1: quarterly,
        2: annual,
        "Monthly": monthly,
        "Quarterly": quarterly,
        "Annual": annual,
    }

    monkeypatch.setattr(
        "paper_data.ingestion.google_sheets.gspread",
        SimpleNamespace(
            public=lambda: SimpleNamespace(open_by_url=lambda u: DummySheet(sheets_map))
        ),
    )

    # Default (Monthly)
    conn_m = GoogleSheetsConnector(url)
    df_m = conn_m.get_data()
    assert list(df_m.columns) == monthly_header
    pd.testing.assert_frame_equal(df_m, pd.DataFrame(monthly))

    # Quarterly by name
    conn_q = GoogleSheetsConnector(url, worksheet_name="Quarterly")
    df_q = conn_q.get_data()
    assert list(df_q.columns) == quarterly_header
    pd.testing.assert_frame_equal(df_q, pd.DataFrame(quarterly))

    # Annual by name
    conn_a = GoogleSheetsConnector(url, worksheet_name="Annual")
    df_a = conn_a.get_data()
    assert list(df_a.columns) == annual_header
    pd.testing.assert_frame_equal(df_a, pd.DataFrame(annual))
