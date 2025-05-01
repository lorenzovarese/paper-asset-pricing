"""
Connector for a public or private Google Sheets document.
Requires the packages `gspread` and `google-auth`.
"""

from __future__ import annotations

import pandas as pd
import gspread  # type: ignore[import-untyped]
from google.oauth2.service_account import Credentials  # type: ignore[import-untyped]

from .base import BaseConnector


class GoogleSheetsConnector(BaseConnector):
    """
    Connector for a public or private Google Sheets document.
    Requires `gspread` and `google-auth`.
    """

    def __init__(
        self,
        spreadsheet_url: str,
        worksheet_name: str | None = None,
        service_account_json: str | None = None,
    ) -> None:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        if service_account_json:
            creds = Credentials.from_service_account_file(
                service_account_json, scopes=scopes
            )
            gc = gspread.authorize(creds)
        else:  # anonymous / public sheet
            gc = gspread.public()  # type: ignore[attr-defined]

        self.sheet = gc.open_by_url(spreadsheet_url)
        self.worksheet_name = worksheet_name

    def get_data(self) -> pd.DataFrame:
        ws = (
            self.sheet.worksheet(self.worksheet_name)
            if self.worksheet_name
            else self.sheet.get_worksheet(0)
        )
        data = ws.get_all_records()
        return pd.DataFrame(data)
