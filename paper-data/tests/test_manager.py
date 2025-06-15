import pytest
import shutil
import pandas as pd
import polars as pl

from paper_data.manager import DataManager  # type: ignore


@pytest.fixture
def simple_csv(tmp_path):
    """
    Create a tiny CSV with a YYYYMMDD date column and an integer ID.
    """
    df = pd.DataFrame(
        {
            "date": ["20200101", "20200201"],
            "id": [1, 2],
            "val": [10, 20],
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_resolve_data_path_without_project_root(tmp_path):
    # Write a minimal valid config so DataManager.__init__ won't fail
    cfg = tmp_path / "data-config.yaml"
    cfg.write_text(
        """ingestion: []
wrangling_pipeline: []
export: []
"""
    )

    manager = DataManager(config_path=cfg)
    # project_root is still None, so _resolve_data_path must raise
    with pytest.raises(ValueError):
        manager._resolve_data_path("some/path.csv")


def test_ingest_data_missing_project_root(tmp_path):
    # An empty config (no 'ingestion') but project_root unset → ValueError
    cfg = tmp_path / "data-config.yaml"
    cfg.write_text(
        """wrangling_pipeline: []
export: []
"""
    )
    manager = DataManager(config_path=cfg)
    with pytest.raises(ValueError):
        manager._ingest_data()


def test_ingest_and_date_parsing(tmp_path, simple_csv):
    # 1) Prepare a fake project folder and copy CSV into data/raw
    project = tmp_path / "proj"
    raw_dir = project / "data" / "raw"
    raw_dir.mkdir(parents=True)
    dest = raw_dir / "data.csv"
    shutil.copy(simple_csv, dest)

    # 2) Write a minimal config pointing at our CSV
    cfg_dir = project / "configs"
    cfg_dir.mkdir()
    cfg = cfg_dir / "data-config.yaml"
    cfg.write_text(
        """ingestion:
  - name: test
    format: csv
    path: data.csv
    date_column:
      date: "%Y%m%d"
    id_column: id
wrangling_pipeline: []
export: []
"""
    )

    # 3) Instantiate, set project_root, and ingest
    manager = DataManager(config_path=cfg)
    manager._project_root = project  # simulate what run() does internally

    # resolve path
    resolved = manager._resolve_data_path("data.csv")
    assert resolved == dest

    # ingest data
    manager._ingest_data()
    assert "test" in manager.datasets

    df = manager.datasets["test"]
    assert isinstance(df, pl.DataFrame)
    # columns and types
    assert list(df.columns) == ["date", "id", "val"]
    assert df["date"].dtype == pl.Date


def test_run_pipeline_creates_dirs_and_returns_empty(tmp_path):
    project = tmp_path / "proj"
    cfg_dir = project / "configs"
    cfg_dir.mkdir(parents=True)
    cfg = cfg_dir / "data-config.yaml"
    cfg.write_text(
        """ingestion: []
wrangling_pipeline: []
export: []
"""
    )

    manager = DataManager(config_path=cfg)
    result = manager.run(project_root=project)

    # should return empty dict
    assert result == {}
    # raw and processed directories created
    assert (project / "data" / "raw").is_dir()
    assert (project / "data" / "processed").is_dir()
