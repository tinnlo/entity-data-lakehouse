# Data Warehouse Design

The gold layer uses a hybrid design inspired by the original reference repo.

## Why Both SCD4 And SCD2 Exist

The public source snapshots can change in two different ways:

- the real world changes, such as ownership percentages moving
- the source scope changes, such as a relationship disappearing and then returning later

The demo uses:

- **SCD4** where full snapshot preservation matters
- **SCD2** where consumers need a stable current/history contract

## Entity Master: SCD4

The entity side follows the original firm-list pattern:

- `entity_master_comprehensive_scd4`: all observation snapshots preserved
- `entity_master_current`: one current row per `entity_id`
- `entity_master_event_log`: one per-entity event per snapshot

Entity observations use a deterministic observation-key cascade:

1. `registry_entity_id`
2. `lei`
3. `source_entity_id`
4. `normalized_name + country_code`

This preserves the distinction between:

- bronze `source_business_key` for source-level lineage
- DW observation identity for warehouse-level matching

## Ownership: SCD4 + Lifecycle

Ownership observations are modeled separately from the convenience relationship edge output.

`ownership_comprehensive_scd4` stores every ownership observation snapshot plus explicit dropped rows. `ownership_lifecycle` then aggregates by `owner_entity_id|asset_id` to calculate:

- first and last appearance
- presence rate
- gap periods
- reliability score
- lifecycle status (`ACTIVE`, `NEW`, `DROPPED`, `INTERMITTENT`)

This is the adaptive-data part of the warehouse design: it can show whether a relationship looks stable, newly introduced, dropped, or intermittent across releases.

## Ownership: SCD2 Current And History

The repo also publishes:

- `ownership_history_scd2`
- `ownership_current`

These follow the original repo’s incumbent ownership-contract idea: keep a stable current/history table for downstream consumers while preserving explicit version rows.

The business key is source-isolated:

```text
owner_entity_id | asset_id | observation_source
```

Changes in ownership percentage or asset-facing business attributes create a new SCD2 version. Missing later snapshots close current rows instead of deleting them.

## Derived Mart

The existing public output, `owner_infrastructure_exposure_snapshot`, is retained as a derived mart. It is rebuilt from the SCD2 history table using as-of logic for every snapshot date present in the SCD4 pipeline.
