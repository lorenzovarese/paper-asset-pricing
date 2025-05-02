import pytest
import pandas as pd
import zipfile
from pathlib import Path

from paper_data.ingestion.local import LocalConnector  # type: ignore[import-untyped]


def test_read_csv(tmp_path):
    data = pd.DataFrame({"a": [1, 2, 3]})
    csv_path = tmp_path / "test.csv"
    data.to_csv(csv_path, index=False)
    conn = LocalConnector(str(csv_path))
    df = conn.get_data()
    pd.testing.assert_frame_equal(df, data)


def test_read_parquet(tmp_path):
    data = pd.DataFrame({"b": [4.0, 5.5]})
    pq_path = tmp_path / "test.parquet"
    data.to_parquet(pq_path)
    conn = LocalConnector(pq_path)
    df = conn.get_data()
    pd.testing.assert_frame_equal(df, data)


def test_file_not_found(tmp_path):
    missing = tmp_path / "no.csv"
    conn = LocalConnector(missing)
    with pytest.raises(FileNotFoundError):
        conn.get_data()


def test_unsupported_suffix(tmp_path):
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("hello")
    conn = LocalConnector(txt_path)
    with pytest.raises(ValueError):
        conn.get_data()


def _make_zip(tmp_path, files: dict[str, str], zip_name="test.zip") -> Path:
    """
    Helper: create a ZIP at tmp_path/zip_name containing {filename: content}.
    Returns path to the zip.
    """
    zip_path = tmp_path / zip_name
    with zipfile.ZipFile(zip_path, mode="w") as zf:
        for fname, content in files.items():
            # write the string as bytes
            zf.writestr(fname, content)
    return zip_path


def test_read_single_csv_in_zip(tmp_path):
    # Create a ZIP with one CSV file
    csv_content = "x,y\n1,2\n3,4"
    zip_path = _make_zip(tmp_path, {"data.csv": csv_content})
    conn = LocalConnector(str(zip_path))
    df = conn.get_data()
    expected = pd.DataFrame({"x": [1, 3], "y": [2, 4]})
    pd.testing.assert_frame_equal(df, expected)


def test_multiple_files_without_member_name_raises(tmp_path):
    # ZIP with two CSVs; ambiguous without member_name
    zip_path = _make_zip(
        tmp_path,
        {
            "a.csv": "a,b\n1,2",
            "b.csv": "a,b\n3,4",
        },
    )
    conn = LocalConnector(zip_path)
    with pytest.raises(ValueError):
        conn.get_data()


def test_read_specific_member_in_zip(tmp_path):
    # ZIP with two CSVs; specify member_name
    files = {
        "first.csv": "c,d\n5,6",
        "second.csv": "c,d\n7,8",
    }
    zip_path = _make_zip(tmp_path, files)
    conn = LocalConnector(zip_path, member_name="second.csv")
    df = conn.get_data()
    expected = pd.DataFrame({"c": [7], "d": [8]})
    pd.testing.assert_frame_equal(df, expected)


def test_member_name_not_found(tmp_path):
    # ZIP with one CSV, but wrong member_name
    zip_path = _make_zip(tmp_path, {"only.csv": "m,n\n9,10"})
    conn = LocalConnector(zip_path, member_name="nope.csv")
    with pytest.raises(FileNotFoundError):
        conn.get_data()
