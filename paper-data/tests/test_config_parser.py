import pytest
import yaml
from paper_data.config_parser import load_config  # type: ignore


def test_load_valid_config(tmp_path):
    cfg = {"key": 42}
    file = tmp_path / "cfg.yaml"
    file.write_text(yaml.dump(cfg))
    result = load_config(file)
    assert result == cfg


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_empty_file(tmp_path):
    file = tmp_path / "empty.yaml"
    file.write_text("")
    with pytest.raises(ValueError):
        load_config(file)
