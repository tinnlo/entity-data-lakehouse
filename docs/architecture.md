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
5. `ml/` (executed as the final pipeline step) enriches assets with geographic and economic features from `reference_data/`, trains three scikit-learn models on a synthetic reference dataset, and writes lifecycle predictions to `gold/dw/asset_lifecycle_predictions.parquet`.

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

## ML Outputs

- `gold/dw/asset_lifecycle_predictions.parquet` — per-asset lifecycle stage, retirement year, and capacity factor predictions with all enrichment features for explainability
- `entity_lakehouse.duckdb` → table `ml_asset_lifecycle_predictions`

## ML Enrichment Sources

- `reference_data/country_attributes.csv` — 29 countries with geographic and economic attributes (latitude/longitude, altitude, territorial type, GDP tier, solar irradiance, wind speed, regulatory stability)
- `reference_data/sector_lifecycle.csv` — sector lifecycle parameters for solar, wind, and storage (lifespan ranges, construction/decommissioning duration, base capacity factor, geographic sensitivity coefficients)

## History Awareness

The demo includes three snapshots.

- entity master uses SCD4 to preserve all observation snapshots and derive a current master plus event log
- ownership uses SCD4 plus lifecycle metrics to measure presence, gaps, and reliability across releases
- downstream ownership consumption uses SCD2 current/history tables
- the public mart still reports `NEW`, `CHANGED`, `UNCHANGED`, and `DROPPED` by snapshot
- ML lifecycle predictions use the ownership lifecycle signal (presence rate, reliability score, snapshot count) as features alongside geographic enrichment
