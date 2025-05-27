import polars as pl
import polars.testing  # this brings in the testing submodule at runtime

# Expose pl.testing.assert_frame_equal
pl.testing = polars.testing  # type: ignore[attr-defined]
