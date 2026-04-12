"""Smoke-test that the Airflow DAG file can be imported and has the correct dag_id.

This test is skipped when `apache-airflow` is not installed, so base CI (which does not
install the airflow extra) stays green.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Guard: skip unless the real apache-airflow package (not the local airflow/ namespace)
# is present. We check for airflow.DAG which only exists in the real package.
airflow_dag = pytest.importorskip(
    "airflow.models.dag", reason="apache-airflow not installed"
)


def test_dag_imports() -> None:
    dag_file = (
        Path(__file__).resolve().parents[2]
        / "airflow"
        / "dags"
        / "entity_lakehouse_dag.py"
    )
    assert dag_file.exists(), f"DAG file not found: {dag_file}"
    spec = importlib.util.spec_from_file_location("entity_lakehouse_dag", dag_file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    assert module.dag.dag_id == "entity_lakehouse_pipeline"
