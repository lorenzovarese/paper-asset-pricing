from __future__ import annotations

from pathlib import Path
import polars as pl
import zipfile
import tempfile
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pyarrow import Table
from pyarrow.parquet import ParquetWriter
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

    def get_data(self) -> pl.DataFrame:
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

    def _read_csv_with_progress(self, csv_path: Path) -> pl.DataFrame:
        # 1) File size check
        size_bytes = csv_path.stat().st_size
        threshold = 2 * 1024**3  # 2 GB
        print(f"CSV file size: {size_bytes / 1024**2:.1f} MB")
        print(f"Threshold for direct load: {threshold / 1024**2:.1f} MB")

        if size_bytes > threshold:
            raise ValueError(
                f"The CSV file size is {size_bytes / 1024**3:.1f} GB, which is too large to load directly.\n"
                "Consider converting it to Parquet first (e.g., using `paper-data convert`).\n"
                "Don't forget to update the path in your configuration file afterward."
            )

        # 2) Lazy scan & streaming collect
        lazy = pl.scan_csv(str(csv_path))
        df_streamed = lazy.collect(engine="streaming")
        record_batches = df_streamed.to_arrow().to_batches()  # type: ignore[attr-defined]

        # 3) Iterate batches, coerce to DataFrame, track progress
        chunks: list[pl.DataFrame] = []
        with tqdm(
            total=size_bytes, unit="B", unit_scale=True, desc=f"Reading {csv_path.name}"
        ) as pbar:
            for rb in record_batches:
                # convert RecordBatch → Polars (might be Series if single column)
                tmp = pl.from_arrow(rb)
                # ensure DataFrame
                if isinstance(tmp, pl.Series):
                    df_chunk = tmp.to_frame()
                else:
                    df_chunk = tmp
                chunks.append(df_chunk)
                # approximate bytes processed
                pbar.update(df_chunk.estimated_size())

        # 4) concatenate all chunks
        return pl.concat(chunks)

    def _read_parquet_with_progress(self, pq_path: Path) -> pl.DataFrame:
        parquet_file = pq.ParquetFile(str(pq_path))
        n_groups = parquet_file.num_row_groups
        tables: list[pa.Table] = []
        with tqdm(total=n_groups, desc="Reading row-groups") as pbar:
            for rg in range(n_groups):
                tables.append(parquet_file.read_row_group(rg))
                pbar.update(1)

        print("Merging tables…")
        table = pa.concat_tables(tables)
        return pl.from_arrow(table)

    def _read_from_zip(self) -> pl.DataFrame:
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
