import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal

from paper_data.ingestion.wrds_conn import WRDSConnector  # type: ignore

# --- Fixtures ---


@pytest.fixture
def mock_cache_path(tmp_path):
    """Provides a temporary path for the cache file."""
    return tmp_path / "wrds_cache.csv"


@pytest.fixture
def sample_pandas_df():
    """A sample pandas DataFrame, as often returned by the wrds library."""
    return pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})


@pytest.fixture
def sample_polars_df():
    """The Polars equivalent of the sample pandas DataFrame."""
    return pl.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})


# --- Tests for WRDSConnector ---


def test_wrds_connector_initialization(mock_cache_path, monkeypatch):
    """Tests that the connector initializes correctly, including credential handling."""
    # Test with explicit credentials
    connector = WRDSConnector(
        query="SELECT * FROM table",
        cache_path=mock_cache_path,
        user="testuser",
        password="testpassword",
    )
    assert connector.user == "testuser"
    assert connector.password == "testpassword"

    # Test with environment variables
    monkeypatch.setenv("WRDS_USER", "envuser")
    monkeypatch.setenv("WRDS_PASSWORD", "envpass")
    connector_env = WRDSConnector(
        query="SELECT * FROM table", cache_path=mock_cache_path
    )
    assert connector_env.user == "envuser"
    assert connector_env.password == "envpass"


@patch("paper_data.ingestion.wrds_conn.wrds.Connection")
def test_get_data_cache_miss(
    mock_wrds_conn, mock_cache_path, sample_pandas_df, sample_polars_df
):
    """
    Tests the 'cache miss' scenario: the cache file doesn't exist,
    so it should connect to WRDS, query, create the cache, and return the data.
    """
    # --- Arrange ---
    # Mock the wrds.Connection object and its methods
    mock_db_instance = MagicMock()
    mock_db_instance.raw_sql.return_value = sample_pandas_df
    mock_wrds_conn.return_value = mock_db_instance

    assert not mock_cache_path.exists()  # Pre-condition: cache is empty

    connector = WRDSConnector(
        query="SELECT * FROM table",
        cache_path=mock_cache_path,
        user="user",
        password="pw",
    )

    # --- Act ---
    result_df = connector.get_data()

    # --- Assert ---
    # 1. A connection to WRDS was made
    mock_wrds_conn.assert_called_once_with(wrds_username="user", wrds_password="pw")
    # 2. The SQL query was executed
    mock_db_instance.raw_sql.assert_called_once_with("SELECT * FROM table")
    # 3. The cache file was created
    assert mock_cache_path.is_file()
    # 4. The returned DataFrame is correct
    assert_frame_equal(result_df, sample_polars_df)


def test_get_data_cache_hit(mock_cache_path, sample_polars_df):
    """
    Tests the 'cache hit' scenario: the cache file exists, so it should
    load directly from the file without connecting to WRDS.
    """
    # --- Arrange ---
    # Pre-populate the cache file
    sample_polars_df.write_csv(mock_cache_path)
    assert mock_cache_path.is_file()

    connector = WRDSConnector(query="SELECT * FROM table", cache_path=mock_cache_path)

    # --- Act ---
    # We use a patch here to ensure wrds.Connection is NOT called
    with patch("paper_data.ingestion.wrds_conn.wrds.Connection") as mock_wrds_conn:
        result_df = connector.get_data()

        # --- Assert ---
        # 1. No connection to WRDS was made
        mock_wrds_conn.assert_not_called()
        # 2. The returned DataFrame is correct
        assert_frame_equal(result_df, sample_polars_df)


@patch("paper_data.ingestion.wrds_conn.wrds.Connection")
def test_query_and_cache_handles_connection_error(mock_wrds_conn, mock_cache_path):
    """
    Tests that if the WRDS query fails, an exception is raised and no
    partial cache file is left behind.
    """
    # --- Arrange ---
    # Make the connection fail by raising an exception
    mock_db_instance = MagicMock()
    mock_db_instance.raw_sql.side_effect = Exception("Authentication failed")
    mock_wrds_conn.return_value = mock_db_instance

    # Create a dummy file to ensure it gets deleted on failure
    mock_cache_path.touch()

    connector = WRDSConnector(
        query="SELECT * FROM table",
        cache_path=mock_cache_path,
        user="user",
        password="pw",
    )

    # --- Act & Assert ---
    with pytest.raises(ConnectionError, match="Failed to execute WRDS query"):
        connector._query_and_cache()

    # Assert that the potentially corrupted cache file was removed
    assert not mock_cache_path.exists()


@patch("paper_data.ingestion.wrds_conn.wrds.Connection")
def test_query_and_cache_handles_unsupported_type(mock_wrds_conn, mock_cache_path):
    """
    Tests that a ConnectionError is raised if WRDS returns an unexpected data type.
    """
    # --- Arrange ---
    # Make the query return something other than a pandas or polars DataFrame
    mock_db_instance = MagicMock()
    mock_db_instance.raw_sql.return_value = {"data": "not a dataframe"}
    mock_wrds_conn.return_value = mock_db_instance

    connector = WRDSConnector(
        query="SELECT * FROM table",
        cache_path=mock_cache_path,
        user="user",
        password="pw",
    )

    # --- Act & Assert ---
    # This correctly tests the behavior of the broad `except Exception` block.
    with pytest.raises(ConnectionError, match="Unsupported data type from WRDS"):
        connector._query_and_cache()
