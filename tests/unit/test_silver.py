from pathlib import Path

from entity_data_lakehouse.silver import build_silver_outputs


def test_silver_outputs_resolve_six_canonical_entities(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    outputs = build_silver_outputs(
        sample_root=repo_root / "sample_data",
        silver_root=tmp_path,
        contract_paths={
            "entity_observations": repo_root / "contracts" / "entity_observations.schema.json",
            "entity_master": repo_root / "contracts" / "entity_master.schema.json",
            "asset_master": repo_root / "contracts" / "asset_master.schema.json",
            "ownership_observations": repo_root / "contracts" / "ownership_observations.schema.json",
            "relationship_edges": repo_root / "contracts" / "relationship_edges.schema.json",
        },
    )

    assert outputs["entity_master"]["entity_id"].nunique() == 6
    assert outputs["asset_master"]["asset_id"].nunique() == 5
    assert len(outputs["entity_observations"]) == 68
    assert len(outputs["ownership_observations"]) == 19
    assert {"OWNS_ASSET", "OPERATES_ASSET", "PARENT_OF_ENTITY"} == set(outputs["relationship_edges"]["relationship_type"])
