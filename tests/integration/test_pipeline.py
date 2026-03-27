from pathlib import Path

import pandas as pd

from entity_data_lakehouse.pipeline import run_pipeline


def test_pipeline_builds_expected_outputs() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    results = run_pipeline(repo_root)

    assert results["entity_master_rows"] == 6
    assert results["asset_master_rows"] == 5
    assert results["relationship_edge_rows"] == 37
    assert results["gold_rows"] == 21

    gold_path = repo_root / "gold" / "owner_infrastructure_exposure_snapshot.parquet"
    gold_df = pd.read_parquet(gold_path)
    assert {"NEW", "CHANGED", "UNCHANGED", "DROPPED"} <= set(gold_df["change_status_vs_prior_snapshot"])

    lifecycle_path = repo_root / "gold" / "dw" / "ownership_lifecycle.parquet"
    lifecycle_df = pd.read_parquet(lifecycle_path)
    assert "INTERMITTENT" in set(lifecycle_df["lifecycle_status"])
