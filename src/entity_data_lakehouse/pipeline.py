from __future__ import annotations

from pathlib import Path

from .bronze import ingest_sample_data
from .gold import build_gold_outputs
from .public_safety import scan_public_safety
from .silver import build_silver_outputs


def run_pipeline(repo_root: Path) -> dict[str, int]:
    contracts_root = repo_root / "contracts"
    sample_root = repo_root / "sample_data"

    ingest_sample_data(
        sample_root=sample_root,
        bronze_root=repo_root / "bronze",
        contract_path=contracts_root / "bronze_source_record.schema.json",
    )

    silver_outputs = build_silver_outputs(
        sample_root=sample_root,
        silver_root=repo_root / "silver",
        contract_paths={
            "entity_observations": contracts_root / "entity_observations.schema.json",
            "entity_master": contracts_root / "entity_master.schema.json",
            "asset_master": contracts_root / "asset_master.schema.json",
            "ownership_observations": contracts_root / "ownership_observations.schema.json",
            "relationship_edges": contracts_root / "relationship_edges.schema.json",
        },
    )

    gold_outputs = build_gold_outputs(
        gold_root=repo_root / "gold",
        silver_outputs=silver_outputs,
        contract_paths={
            "entity_master_comprehensive_scd4": contracts_root / "entity_master_comprehensive_scd4.schema.json",
            "entity_master_current": contracts_root / "entity_master_current.schema.json",
            "entity_master_event_log": contracts_root / "entity_master_event_log.schema.json",
            "ownership_comprehensive_scd4": contracts_root / "ownership_comprehensive_scd4.schema.json",
            "ownership_lifecycle": contracts_root / "ownership_lifecycle.schema.json",
            "ownership_history_scd2": contracts_root / "ownership_history_scd2.schema.json",
            "ownership_current": contracts_root / "ownership_current.schema.json",
            "owner_infrastructure_exposure_snapshot": contracts_root / "owner_infrastructure_exposure_snapshot.schema.json",
        },
    )

    safety_findings = scan_public_safety(repo_root)
    if safety_findings:
        raise ValueError("Public-safety scan failed:\n" + "\n".join(safety_findings))

    return {
        "entity_master_rows": len(silver_outputs["entity_master"]),
        "asset_master_rows": len(silver_outputs["asset_master"]),
        "relationship_edge_rows": len(silver_outputs["relationship_edges"]),
        "gold_rows": len(gold_outputs["owner_infrastructure_exposure_snapshot"]),
    }
