from __future__ import annotations

from pathlib import Path
import pandas as pd
import zipfile
import tempfile
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from .base import BaseConnector


class LocalConnector(BaseConnector):
    def __init__(
        self,
        path: str | Path,
        member_name: str | None = None,
        csv_chunk_size: int = 100_000,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.member_name = member_name
        self.csv_chunk_size = csv_chunk_size

    def get_data(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Path not found: {self.path}")

        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            return self._read_csv_with_progress(self.path)
        if suffix in {".parquet", ".pq"}:
            return self._read_parquet_with_progress(self.path)
        if suffix == ".zip":
            return self._read_from_zip()
        raise ValueError(f"Unsupported file type: {suffix}")

    def _read_csv_with_progress(self, csv_path: Path) -> pd.DataFrame:
        size_bytes = csv_path.stat().st_size
        threshold = 2000 * 1024**2  # 2 GB
        print(f"CSV file size: {size_bytes / 1024**2:.1f} MB")
        print(f"Threshold for direct load: {threshold / 1024**2:.1f} MB")
        if size_bytes > threshold:
            raise ValueError(
                f"The CSV file size is {size_bytes / 1024**3:.1f} GB, which is too large to load directly.\n"
                f"Consider converting it to Parquet first (e.g., using `paper-data convert`).\n"
                f"Don't forget to update the path in your configuration file afterward."
            )
        chunks: list[pd.DataFrame] = []
        with tqdm(
            total=size_bytes, unit="B", unit_scale=True, desc=f"Reading {csv_path.name}"
        ) as pbar:
            for chunk in pd.read_csv(csv_path, chunksize=self.csv_chunk_size):
                chunks.append(chunk)
                # approximate bytes processed
                pbar.update(chunk.memory_usage(deep=True).sum())
        return pd.concat(chunks, ignore_index=True)

    def _read_parquet_with_progress(self, pq_path: Path) -> pd.DataFrame:
        parquet_file = pq.ParquetFile(str(pq_path))
        n_groups = parquet_file.num_row_groups
        dfs: list[pd.DataFrame] = []
        with tqdm(total=n_groups, desc=f"Reading {pq_path.name} (row-groups)") as pbar:
            for rg in range(n_groups):
                table = parquet_file.read_row_group(rg)
                dfs.append(table.to_pandas())
                pbar.update(1)
        return pd.concat(dfs, ignore_index=True)

    def _read_from_zip(self) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(self.path, "r") as zf:
                all_files = [n for n in zf.namelist() if not n.endswith("/")]
                candidates = [
                    n
                    for n in all_files
                    if Path(n).suffix.lower() in {".csv", ".parquet", ".pq"}
                ]

                if self.member_name:
                    matches = [
                        n for n in candidates if Path(n).name == self.member_name
                    ]
                    if not matches:
                        raise FileNotFoundError(
                            f"Member '{self.member_name}' not in ZIP"
                        )
                    if len(matches) > 1:
                        raise ValueError(
                            f"Multiple entries named '{self.member_name}' in ZIP"
                        )
                    target = matches[0]
                else:
                    if not candidates:
                        raise ValueError("No CSV/Parquet in ZIP")
                    if len(candidates) > 1:
                        raise ValueError(
                            "Multiple data files in ZIP; specify member_name"
                        )
                    target = candidates[0]

                info = zf.getinfo(target)
                out_path = Path(tmpdir) / target
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # extract with progress
                with tqdm(
                    total=info.file_size,
                    unit="B",
                    unit_scale=True,
                    desc=f"Extracting {target}",
                ) as pbar:
                    with zf.open(target, "r") as src, open(out_path, "wb") as dst:
                        for block in iter(lambda: src.read(64 * 1024), b""):
                            dst.write(block)
                            pbar.update(len(block))

                # now read extracted file
                ext = out_path.suffix.lower()
                if ext == ".csv":
                    return self._read_csv_with_progress(out_path)
                return self._read_parquet_with_progress(out_path)
