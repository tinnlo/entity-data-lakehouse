# Airflow DAG — local dev guide

This directory contains the Airflow DAG that wraps the entity-data-lakehouse pipeline
for orchestration demo purposes.

## Architecture

The DAG `entity_lakehouse_pipeline` runs three tasks in sequence:

1. **run_pipeline_stages** — PythonOperator that calls `run_pipeline(repo_root)` from
   `entity_data_lakehouse.pipeline`. Executes bronze → silver → gold → ML.
2. **run_dbt** — BashOperator that runs `dbt run` + `dbt test` against the gold DuckDB,
   materialising the `main_analytics.*` models.
3. **run_public_safety_scan** — BashOperator that runs `verify_public_safety.py` as a
   final gate.

The DAG is trigger-only (`schedule=None`). It uses `SequentialExecutor` + SQLite, which
is the recommended configuration for single-machine demo use.

## Running locally with Docker

```bash
# Build the custom Airflow image (installs repo deps via constraints file)
docker compose build airflow

# Start scheduler + webserver + triggerer (airflow standalone)
docker compose up airflow

# Open http://localhost:8080 — login: admin / admin
# Trigger the DAG from the UI or via CLI:
docker compose exec airflow airflow dags trigger entity_lakehouse_pipeline
```

## Stopping

```bash
docker compose down
# or via Makefile:
make airflow-down
```

## Notes

- The whole repo is mounted at `/opt/airflow/repo` so all sample data, contracts,
  reference data, and dbt models are available at runtime.
- `PYTHONPATH=/opt/airflow/repo/src` makes `entity_data_lakehouse` importable without
  installing the package inside the container.
- Generated artefacts (`bronze/`, `silver/`, `gold/`) are written inside the container
  mount and appear on the host.
