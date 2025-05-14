import polars as pl
import polars.testing  # this brings in the testing submodule at runtime

from .main import run_data_pipeline_from_config

# Expose pl.testing.assert_frame_equal
pl.testing = polars.testing  # type: ignore[attr-defined]

__version__ = "0.1.0"
