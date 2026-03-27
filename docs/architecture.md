# Architecture

`entity-data-lakehouse` is a compact medallion-style demo for public entity and infrastructure data.

## Flow

1. `sample_data/` stores bundled public-safe CSV snapshots for:
   - registry-style entities
   - parent-child entity hierarchy
   - infrastructure asset ownership
2. `bronze/` receives a standardized envelope per source record with typed matching fields and a `raw_payload` JSON blob.
3. `silver/` resolves canonical entities, standardizes asset dimensions, and emits both observation-grain tables and convenience outputs.
4. `gold/` publishes a hybrid warehouse-oriented model and a local DuckDB database for ad hoc analysis.

## Entity Resolution

The silver layer uses a fixed match hierarchy:

1. `registry_entity_id`
2. `lei`
3. `source_entity_id`
4. `(normalized_name, country_code)`

Names are normalized with accent stripping, punctuation removal, case folding, and whitespace collapsing.

## Silver Outputs

- `silver/entity_observations.parquet`
- `silver/entity_master.parquet`
- `silver/asset_master.parquet`
- `silver/ownership_observations.parquet`
- `silver/relationship_edges.parquet`

## Gold Outputs

- `gold/dw/entity_master_comprehensive_scd4.parquet`
- `gold/dw/entity_master_current.parquet`
- `gold/dw/entity_master_event_log.parquet`
- `gold/dw/ownership_comprehensive_scd4.parquet`
- `gold/dw/ownership_lifecycle.parquet`
- `gold/dw/ownership_history_scd2.parquet`
- `gold/dw/ownership_current.parquet`
- `gold/owner_infrastructure_exposure_snapshot.parquet`
- `gold/entity_lakehouse.duckdb`

## History Awareness

The demo includes three snapshots.

- entity master uses SCD4 to preserve all observation snapshots and derive a current master plus event log
- ownership uses SCD4 plus lifecycle metrics to measure presence, gaps, and reliability across releases
- downstream ownership consumption uses SCD2 current/history tables
- the public mart still reports `NEW`, `CHANGED`, `UNCHANGED`, and `DROPPED` by snapshot
