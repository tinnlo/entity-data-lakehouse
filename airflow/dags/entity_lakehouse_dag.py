from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

REPO_ROOT = Path("/opt/airflow/repo")


def _run_pipeline() -> None:
    from entity_data_lakehouse.pipeline import run_pipeline

    run_pipeline(repo_root=REPO_ROOT)


with DAG(
    dag_id="entity_lakehouse_pipeline",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["entity-data-lakehouse"],
) as dag:
    run_pipeline_stages = PythonOperator(
        task_id="run_pipeline_stages",
        python_callable=_run_pipeline,
    )

    run_dbt = BashOperator(
        task_id="run_dbt",
        bash_command="cd /opt/airflow/repo/dbt && dbt run --profiles-dir . && dbt test --profiles-dir .",
    )

    run_public_safety_scan = BashOperator(
        task_id="run_public_safety_scan",
        bash_command="python /opt/airflow/repo/scripts/verify_public_safety.py",
    )

    run_pipeline_stages >> run_dbt >> run_public_safety_scan
