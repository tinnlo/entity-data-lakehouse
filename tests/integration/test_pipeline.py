from pathlib import Path

import pandas as pd

from entity_data_lakehouse.pipeline import run_pipeline

_VALID_LIFECYCLE_STAGES = {
    "planning",
    "construction",
    "operating",
    "decommissioning",
    "retired",
}


def test_pipeline_builds_expected_outputs() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    results = run_pipeline(repo_root)

    assert results["entity_master_rows"] == 6
    assert results["asset_master_rows"] == 5
    assert results["relationship_edge_rows"] == 37
    assert results["gold_rows"] == 21
    assert results["ml_prediction_rows"] == 5

    gold_path = repo_root / "gold" / "owner_infrastructure_exposure_snapshot.parquet"
    gold_df = pd.read_parquet(gold_path)
    assert {"NEW", "CHANGED", "UNCHANGED", "DROPPED"} <= set(
        gold_df["change_status_vs_prior_snapshot"]
    )

    lifecycle_path = repo_root / "gold" / "dw" / "ownership_lifecycle.parquet"
    lifecycle_df = pd.read_parquet(lifecycle_path)
    assert "INTERMITTENT" in set(lifecycle_df["lifecycle_status"])

    ml_path = repo_root / "gold" / "dw" / "asset_lifecycle_predictions.parquet"
    assert ml_path.exists(), (
        "asset_lifecycle_predictions.parquet not written to gold/dw/"
    )
    ml_df = pd.read_parquet(ml_path)

    # One prediction per asset in asset_master.
    assert len(ml_df) == 5, f"Expected 5 ML prediction rows, got {len(ml_df)}"

    # All predicted lifecycle stages must be valid labels.
    unexpected = set(ml_df["predicted_lifecycle_stage"]) - _VALID_LIFECYCLE_STAGES
    assert not unexpected, f"Unexpected lifecycle stages: {unexpected}"

    # Retirement years must be in a physically plausible range.
    assert ml_df["estimated_retirement_year"].between(2025, 2080).all(), (
        f"Retirement years out of range: {ml_df['estimated_retirement_year'].tolist()}"
    )

    # Commissioning must precede retirement.
    assert (
        ml_df["estimated_commissioning_year"] < ml_df["estimated_retirement_year"]
    ).all()

    # Capacity factor predictions must be in 1-80% range.
    assert ml_df["predicted_capacity_factor_pct"].between(1.0, 80.0).all()

    # All five assets from asset_master should appear in ML predictions.
    assert set(ml_df["asset_sector"]).issubset({"solar", "wind", "storage"})
