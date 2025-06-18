import polars as pl
import pytest
from datetime import date

from paper_portfolio.manager import PortfolioManager  # type: ignore[import]
from paper_portfolio.config_parser import load_config  # type: ignore[import]

# A minimal but complete YAML config string for tests
MINIMAL_YAML_CONFIG = """
input_data:
  prediction_model_names: ["m1"]
  prediction_extraction_method: "precomputed_prediction_files"
  precomputed_prediction_files:
    date_column: "date"
    id_column: "permno"
  risk_free_dataset:
    file_name: "risk_free.csv"
    date_column: "date"
    return_column: "rf"
    date_format: "%Y-%m-%d"
  company_value_weights:
    file_name: "company_values.csv"
    date_column: "date"
    id_column: "permno"
    value_weight_col: "mktcap"
    date_format: "%Y-%m-%d"

portfolio_strategies:
- name: "s1"
  weighting_scheme: "equal"
  long_quantiles: [0.0, 1.0]
  short_quantiles: [0.0, 1.0]

metrics: ["sharpe_ratio"]
"""


@pytest.fixture
def setup_project_files(tmp_path):
    """A fixture to create a standard project structure and mock data files."""
    proj_root = tmp_path / "proj"

    # Create directories
    (proj_root / "models" / "predictions").mkdir(parents=True)
    (proj_root / "portfolios" / "additional_datasets").mkdir(parents=True)
    (proj_root / "configs").mkdir(parents=True)

    # Create config file
    (proj_root / "configs" / "portfolio-config.yaml").write_text(MINIMAL_YAML_CONFIG)

    # Create mock prediction file
    preds = pl.DataFrame(
        {
            "date": [date(2020, 1, 31), date(2020, 2, 29)],
            "permno": [101, 102],
            "predicted_ret": [0.05, -0.02],
            "actual_ret": [0.04, -0.03],
        }
    )
    preds.write_parquet(proj_root / "models" / "predictions" / "m1_predictions.parquet")

    # Create mock additional data files
    (proj_root / "portfolios" / "additional_datasets" / "risk_free.csv").write_text(
        "date,rf\n2020-01-31,0.001\n2020-02-29,0.002"
    )
    (
        proj_root / "portfolios" / "additional_datasets" / "company_values.csv"
    ).write_text("date,permno,mktcap\n2020-01-31,101,1000\n2020-02-29,102,2000")

    return proj_root


def test_load_and_merge_data_success(setup_project_files):
    """Tests that data from multiple sources is loaded and merged correctly."""
    proj_root = setup_project_files
    config = load_config(proj_root / "configs" / "portfolio-config.yaml")
    mgr = PortfolioManager(config)

    data = mgr._load_and_merge_data(proj_root)

    assert "m1" in data
    out_df = data["m1"]

    # Check for columns from all sources
    assert "predicted_ret" in out_df.schema.names()
    assert "actual_ret" in out_df.schema.names()
    assert "risk_free_rate" in out_df.schema.names()
    assert "value_weight" in out_df.schema.names()
    assert out_df.height == 2


def test_load_data_missing_prediction_file(setup_project_files):
    """Tests that the manager handles a missing prediction file gracefully."""
    proj_root = setup_project_files
    config = load_config(proj_root / "configs" / "portfolio-config.yaml")
    # Add a model for which no prediction file exists
    config.input_data.prediction_model_names.append("m2_missing")
    mgr = PortfolioManager(config)

    data = mgr._load_and_merge_data(proj_root)

    # Should successfully load m1 but skip m2
    assert "m1" in data
    assert "m2_missing" not in data


def test_load_data_missing_risk_free_file(setup_project_files):
    """Tests that the process fails if a required file like risk_free is missing."""
    proj_root = setup_project_files
    # Delete the required risk-free file
    (proj_root / "portfolios" / "additional_datasets" / "risk_free.csv").unlink()

    config = load_config(proj_root / "configs" / "portfolio-config.yaml")
    mgr = PortfolioManager(config)

    with pytest.raises(
        FileNotFoundError, match="Risk-free rate dataset could not be loaded"
    ):
        mgr._load_and_merge_data(proj_root)


def test_calculate_monthly_returns_equal_weight(setup_project_files):
    """Tests the calculation of equal-weighted portfolio returns."""
    proj_root = setup_project_files
    config = load_config(proj_root / "configs" / "portfolio-config.yaml")
    mgr = PortfolioManager(config)

    # Create a single-month dataframe for testing
    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 31)] * 4,
            "permno": [1, 2, 3, 4],
            "predicted_ret": [0.1, 0.4, 0.2, 0.3],  # Ranks: 1, 4, 2, 3
            "actual_ret": [0.01, 0.04, 0.02, 0.03],
            "risk_free_rate": [0.001] * 4,
        }
    )

    # Modify strategy to be a simple top/bottom half split
    mgr.config.portfolio_strategies[0].long_quantiles = [0.5, 1.0]  # Top 2
    mgr.config.portfolio_strategies[0].short_quantiles = [0.0, 0.5]  # Bottom 2

    out = mgr._calculate_monthly_returns(df)
    assert not out.is_empty()

    # Expected long return: mean(0.04, 0.03) = 0.035
    # Expected short return: mean(0.01, 0.02) = 0.015
    # Expected portfolio return: 0.035 - 0.015 = 0.02
    assert out["long_return"][0] == pytest.approx(0.035)
    assert out["short_return"][0] == pytest.approx(0.015)
    assert out["portfolio_return"][0] == pytest.approx(0.02)


def test_calculate_monthly_returns_value_weight(setup_project_files):
    """Tests the calculation of value-weighted portfolio returns."""
    proj_root = setup_project_files
    config = load_config(proj_root / "configs" / "portfolio-config.yaml")
    # Add a value-weighted strategy
    config.portfolio_strategies.append(
        type(config.portfolio_strategies[0])(
            name="s2_value",
            weighting_scheme="value",
            long_quantiles=[0.5, 1.0],
            short_quantiles=[0.0, 0.5],
        )
    )
    mgr = PortfolioManager(config)

    df = pl.DataFrame(
        {
            "date": [date(2020, 1, 31)] * 4,
            "permno": [1, 2, 3, 4],
            "predicted_ret": [0.1, 0.4, 0.2, 0.3],  # Ranks: 1, 4, 2, 3
            "actual_ret": [0.01, 0.04, 0.02, 0.03],
            "risk_free_rate": [0.001] * 4,
            "value_weight": [
                100,
                300,
                200,
                400,
            ],  # Weights for permno 2 and 4 are 300, 400
        }
    )

    out = mgr._calculate_monthly_returns(df)
    value_strat_out = out.filter(pl.col("strategy") == "s2_value")

    # Long portfolio: permno 2 (ret 0.04, w 300) and 4 (ret 0.03, w 400)
    # Total weight = 700. Expected long ret = (0.04*300 + 0.03*400) / 700 = 24/700
    # Short portfolio: permno 1 (ret 0.01, w 100) and 3 (ret 0.02, w 200)
    # Total weight = 300. Expected short ret = (0.01*100 + 0.02*200) / 300 = 5/300
    expected_long = 24 / 700
    expected_short = 5 / 300

    assert value_strat_out["long_return"][0] == pytest.approx(expected_long)
    assert value_strat_out["short_return"][0] == pytest.approx(expected_short)
    assert value_strat_out["portfolio_return"][0] == pytest.approx(
        expected_long - expected_short
    )
