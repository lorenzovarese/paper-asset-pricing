from __future__ import annotations

from pathlib import Path
import pandas as pd
import zipfile
import tempfile

from .base import BaseConnector


class LocalConnector(BaseConnector):
    """
    Connector for local CSV, Parquet, or ZIP files containing one or more
    CSV/Parquet files. If the path is a ZIP, it unpacks to a tempdir,
    locates the target file, and loads it as a DataFrame.
    """

    def __init__(
        self,
        path: str | Path,
        member_name: str | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        # Optional: exact filename inside ZIP to load
        self.member_name = member_name

    def get_data(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Path not found: {self.path!s}")

        suffix = self.path.suffix.lower()
        # 1) Basic CSV
        if suffix == ".csv":
            return pd.read_csv(self.path)
        # 2) Basic Parquet
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(self.path)
        # 3) ZIP support
        if suffix == ".zip":
            return self._read_from_zip()
        # 4) Unsupported
        raise ValueError(f"Unsupported file type: {suffix}")

    def _read_from_zip(self) -> pd.DataFrame:
        # Unzip to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(self.path, "r") as zf:
                # List all non-directory entries
                all_files = [name for name in zf.namelist() if not name.endswith("/")]
                # Filter to CSV / Parquet
                candidates = [
                    name
                    for name in all_files
                    if Path(name).suffix.lower() in {".csv", ".parquet", ".pq"}
                ]

                if self.member_name:
                    # Match by basename
                    matches = [
                        name
                        for name in candidates
                        if Path(name).name == self.member_name
                    ]
                    if not matches:
                        raise FileNotFoundError(
                            f"Member '{self.member_name}' not found in ZIP."
                        )
                    if len(matches) > 1:
                        raise ValueError(
                            f"Multiple entries named '{self.member_name}' in ZIP."
                        )
                    target = matches[0]
                else:
                    if len(candidates) == 0:
                        raise ValueError("No CSV or Parquet files found in ZIP.")
                    if len(candidates) > 1:
                        raise ValueError(
                            "Multiple data files in ZIP; please specify member_name."
                        )
                    target = candidates[0]

                ext = Path(target).suffix.lower()
                # Extract the selected file
                zf.extract(target, path=tmpdir)
                extracted_path = Path(tmpdir) / target

                if ext == ".csv":
                    return pd.read_csv(extracted_path)
                # .parquet or .pq
                return pd.read_parquet(extracted_path)
