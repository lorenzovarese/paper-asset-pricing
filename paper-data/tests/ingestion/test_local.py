import pytest
import polars as pl
from paper_data.ingestion.local import CSVLoader  # type: ignore


def test_csv_loader(tmp_path):
    data = "date,id,val\n20200101,1,10\n20200201,2,20"
    file = tmp_path / "data.csv"
    file.write_text(data)
    loader = CSVLoader(path=file, date_col="date", id_col="id")
    df = loader.get_data(date_format="%Y%m%d")
    assert isinstance(df, pl.DataFrame)
    assert list(df.columns) == ["date", "id", "val"]
    assert df.shape == (2, 3)


def test_missing_cols(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("x,y\n1,2")
    loader = CSVLoader(path=file, date_col="date", id_col="id")
    with pytest.raises(ValueError):
        loader.get_data()
