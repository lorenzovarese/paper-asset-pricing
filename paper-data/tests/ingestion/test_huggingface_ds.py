import pandas as pd
from types import SimpleNamespace
from paper_data.ingestion.huggingface_ds import HuggingFaceConnector  # type: ignore[import-untyped]


def test_get_data(monkeypatch):
    sample_df = pd.DataFrame({"col": [1, 2, 3]})
    dummy_ds = SimpleNamespace(to_pandas=lambda: sample_df)
    monkeypatch.setattr(
        "paper_data.ingestion.huggingface_ds.load_dataset",
        lambda repo_id, split, **kwargs: dummy_ds,
    )
    conn = HuggingFaceConnector("repo", split="train")
    df = conn.get_data()
    pd.testing.assert_frame_equal(df, sample_df)
