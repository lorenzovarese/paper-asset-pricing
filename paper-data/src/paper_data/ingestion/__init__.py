"""ingestion package."""

__author__ = "Lorenzo Varese"
__version__ = "0.1.0"

from .base import DataConnector
from .local import CSVLoader
from .http import GoogleSheetConnector
from .wrds_conn import WRDSConnector


__all__ = ["DataConnector", "CSVLoader", "GoogleSheetConnector", "WRDSConnector"]
