"""Unit tests for the PostgreSQL serving layer sink.

All psycopg2 calls are mocked — no running PostgreSQL server required.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest

from entity_data_lakehouse.postgres_sink import (
    _parse_ddl_columns,
    _prepare_insert_frame,
    validate_sink_schema,
    write_gold_to_postgres,
)


# ---------------------------------------------------------------------------
# Minimal test DataFrames matching the exact gold contracts
# ---------------------------------------------------------------------------

def _entity_master_event_log_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": ["entity-1"],
            "snapshot_date": ["2024-01-01"],
            "snapshot_sequence_number": [1],
            "event_type": ["NEW"],
        }
    )


def _exposure_snapshot_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "owner_entity_id": ["owner-1"],
            "asset_country": ["GB"],
            "asset_sector": ["solar"],
            "asset_count": [3],
            "controlled_asset_count": [2],
            "owned_capacity_mw": [150.0],
            "average_ownership_pct": [65.0],
            "relationship_count": [4],
            "snapshot_date": ["2024-01-01"],
            "change_status_vs_prior_snapshot": ["NEW"],
        }
    )


def _make_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    return (
        {
            "entity_master_event_log": _entity_master_event_log_df(),
            "owner_infrastructure_exposure_snapshot": _exposure_snapshot_df(),
        },
        {},  # ml_outputs not used by PostgreSQL sink
    )


# ---------------------------------------------------------------------------
# Shared mock fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def _pg_mock(monkeypatch):
    """Mock psycopg2 connection pool and connections."""
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    mock_pool = MagicMock()
    mock_pool.getconn.return_value = mock_conn

    mock_pool_class = MagicMock(return_value=mock_pool)

    # Mock execute_values from psycopg2.extras
    mock_execute_values = MagicMock()

    mock_module = MagicMock()
    mock_module.pool.ThreadedConnectionPool = mock_pool_class
    mock_module.extras.execute_values = mock_execute_values

    monkeypatch.setitem(sys.modules, "psycopg2", mock_module)
    monkeypatch.setitem(sys.modules, "psycopg2.pool", mock_module.pool)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", mock_module.extras)

    monkeypatch.setenv("USE_POSTGRES", "true")
    monkeypatch.setenv("POSTGRES_DATABASE", "lakehouse")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_USER", "postgres")
    monkeypatch.setenv("POSTGRES_PASSWORD", "")
    monkeypatch.setenv("POSTGRES_POOL_MIN", "2")
    monkeypatch.setenv("POSTGRES_POOL_MAX", "10")

    return {
        "pool": mock_pool,
        "conn": mock_conn,
        "cursor": mock_cursor,
        "pool_class": mock_pool_class,
        "execute_values": mock_execute_values,
    }


# ---------------------------------------------------------------------------
# No-op path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["false", "False", "FALSE", "", "0"])
def test_noop_when_flag_is_false(monkeypatch, flag) -> None:
    monkeypatch.setenv("USE_POSTGRES", flag)
    gold, ml = _make_inputs()

    # Temporarily poison psycopg2 so any attempted import raises.
    saved = sys.modules.pop("psycopg2", None)
    sys.modules["psycopg2"] = None  # type: ignore[assignment]
    try:
        result = write_gold_to_postgres(gold, ml)
        assert result["status"] == "skipped"
        assert result["batch_id"] is None
        assert result["tables_refreshed"] == []
    finally:
        if saved is not None:
            sys.modules["psycopg2"] = saved
        else:
            sys.modules.pop("psycopg2", None)


def test_noop_when_flag_unset(monkeypatch) -> None:
    monkeypatch.delenv("USE_POSTGRES", raising=False)
    gold, ml = _make_inputs()
    sys.modules["psycopg2"] = None  # type: ignore[assignment]
    try:
        result = write_gold_to_postgres(gold, ml)
        assert result["status"] == "skipped"
    finally:
        sys.modules.pop("psycopg2", None)


# ---------------------------------------------------------------------------
# Connection pool tests
# ---------------------------------------------------------------------------

def test_connection_pool_created_once(_pg_mock) -> None:
    """Connection pool should be created once and reused."""
    gold, ml = _make_inputs()

    # Reset the global pool
    import entity_data_lakehouse.postgres_sink as mod
    mod._pool = None

    write_gold_to_postgres(gold, ml)

    # Pool class should be called exactly once
    assert _pg_mock["pool_class"].call_count == 1

    # Verify pool configuration
    call_kwargs = _pg_mock["pool_class"].call_args[1]
    assert call_kwargs["minconn"] == 2
    assert call_kwargs["maxconn"] == 10
    assert call_kwargs["host"] == "localhost"
    assert call_kwargs["port"] == 5432
    assert call_kwargs["database"] == "lakehouse"
    assert call_kwargs["user"] == "postgres"


def test_connection_checkout_and_return(_pg_mock) -> None:
    """Connection should be checked out from pool and returned."""
    gold, ml = _make_inputs()

    import entity_data_lakehouse.postgres_sink as mod
    mod._pool = None

    write_gold_to_postgres(gold, ml)

    # Connection should be checked out
    assert _pg_mock["pool"].getconn.call_count == 1

    # Connection should be returned
    assert _pg_mock["pool"].putconn.call_count == 1
    assert _pg_mock["pool"].putconn.call_args[0][0] == _pg_mock["conn"]


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------

def test_validate_sink_schema_all_pass() -> None:
    """Schema validation should pass when all DataFrames match DDL."""
    gold, ml = _make_inputs()
    results = validate_sink_schema(gold, ml)

    assert len(results) == 2
    assert all(r["status"] == "passed" for r in results)
    assert all(r["error"] is None for r in results)


def test_validate_sink_schema_missing_column() -> None:
    """Schema validation should fail when a required column is missing."""
    gold, ml = _make_inputs()
    # Remove a required column
    gold["entity_master_event_log"] = gold["entity_master_event_log"].drop(columns=["event_type"])

    results = validate_sink_schema(gold, ml)

    # Find the failed result
    failed = [r for r in results if r["status"] == "failed"]
    assert len(failed) == 1
    assert "entity_master_event_log" in failed[0]["table"]
    assert "Missing:" in failed[0]["error"]
    assert "event_type" in failed[0]["error"]


def test_validate_sink_schema_extra_column() -> None:
    """Schema validation should fail when an extra column is present."""
    gold, ml = _make_inputs()
    # Add an extra column
    gold["entity_master_event_log"]["extra_col"] = "value"

    results = validate_sink_schema(gold, ml)

    failed = [r for r in results if r["status"] == "failed"]
    assert len(failed) == 1
    assert "Extra:" in failed[0]["error"]
    assert "extra_col" in failed[0]["error"]


def test_validate_sink_schema_wrong_dtype() -> None:
    """Schema validation should fail when column dtype is incompatible."""
    gold, ml = _make_inputs()
    # Change integer column to string
    gold["entity_master_event_log"]["snapshot_sequence_number"] = ["not_an_int"]

    results = validate_sink_schema(gold, ml)

    failed = [r for r in results if r["status"] == "failed"]
    assert len(failed) == 1
    assert "dtype mismatch" in failed[0]["error"]


def test_validate_sink_schema_missing_dataframe() -> None:
    """Schema validation should fail when expected DataFrame is missing."""
    gold, ml = _make_inputs()
    del gold["entity_master_event_log"]

    results = validate_sink_schema(gold, ml)

    failed = [r for r in results if r["status"] == "failed"]
    assert len(failed) == 1
    assert "entity_master_event_log" in failed[0]["table"]
    assert "missing" in failed[0]["error"].lower()


# ---------------------------------------------------------------------------
# Atomic refresh tests
# ---------------------------------------------------------------------------

def test_atomic_refresh_sequence(_pg_mock) -> None:
    """Each table must go through: create live, drop staging, create staging,
    insert into staging, atomic swap (rename sequence), update metadata."""
    gold, ml = _make_inputs()

    import entity_data_lakehouse.postgres_sink as mod
    mod._pool = None

    write_gold_to_postgres(gold, ml)

    # Get all execute calls
    execute_calls = [str(c.args[0]) if c.args else "" for c in _pg_mock["cursor"].execute.call_args_list]

    for table in ("entity_master_event_log", "owner_infrastructure_exposure_snapshot"):
        # Live table DDL
        assert any(f"CREATE TABLE IF NOT EXISTS {table}" in cmd for cmd in execute_calls)

        # Staging cleanup (SELECT to find orphaned tables, checking for locks to avoid concurrent runs)
        assert any(
            f"SELECT c.relname" in cmd and f"pg_locks" in cmd and f"{table}_staging_%" in cmd
            for cmd in execute_calls
        )

        # Staging DDL (CREATE)
        assert any(f"CREATE TABLE IF NOT EXISTS {table}_staging_" in cmd for cmd in execute_calls)

        # Atomic swap using ALTER TABLE RENAME
        assert any(f"ALTER TABLE {table} RENAME TO {table}_old_" in cmd for cmd in execute_calls)
        assert any(f"ALTER TABLE {table}_staging_" in cmd and "RENAME TO" in cmd for cmd in execute_calls)

        # Drop old table
        assert any(f"DROP TABLE IF EXISTS {table}_old_" in cmd for cmd in execute_calls)


def test_refresh_metadata_table_created(_pg_mock) -> None:
    """_refresh_metadata table should be created."""
    gold, ml = _make_inputs()

    import entity_data_lakehouse.postgres_sink as mod
    mod._pool = None

    write_gold_to_postgres(gold, ml)

    execute_calls = [str(c.args[0]) if c.args else "" for c in _pg_mock["cursor"].execute.call_args_list]

    assert any("CREATE TABLE IF NOT EXISTS _refresh_metadata" in cmd for cmd in execute_calls)


def test_refresh_metadata_updated(_pg_mock) -> None:
    """Freshness metadata should be updated for each table."""
    gold, ml = _make_inputs()

    import entity_data_lakehouse.postgres_sink as mod
    mod._pool = None

    write_gold_to_postgres(gold, ml)

    execute_calls = [str(c.args[0]) if c.args else "" for c in _pg_mock["cursor"].execute.call_args_list]

    # Should have INSERT ... ON CONFLICT for metadata
    metadata_inserts = [cmd for cmd in execute_calls if "INSERT INTO _refresh_metadata" in cmd]
    assert len(metadata_inserts) >= 2  # One per table


# ---------------------------------------------------------------------------
# Batch coordination tests
# ---------------------------------------------------------------------------

def test_all_tables_succeed_batch_published(_pg_mock) -> None:
    """When all tables succeed, batch_id should be published."""
    gold, ml = _make_inputs()

    import entity_data_lakehouse.postgres_sink as mod
    mod._pool = None

    result = write_gold_to_postgres(gold, ml)

    assert result["status"] == "success"
    assert result["batch_id"] is not None
    assert len(result["tables_refreshed"]) == 2
    assert result["rollback_status"] == "clean"


def test_one_table_fails_rollback_triggered(_pg_mock) -> None:
    """When one table fails, all should be rolled back."""
    gold, ml = _make_inputs()

    import entity_data_lakehouse.postgres_sink as mod
    mod._pool = None

    # Make the second table fail
    _pg_mock["cursor"].execute.side_effect = [
        None,  # CREATE _refresh_metadata
        None,  # CREATE live table 1
        None,  # DROP staging 1
        None,  # CREATE staging 1
        None,  # COPY data 1
        None,  # DO $$ (rename old)
        None,  # ALTER TABLE (rename staging to live)
        None,  # DROP old
        None,  # INSERT metadata
        None,  # CREATE live table 2
        None,  # DROP staging 2
        Exception("Simulated failure"),  # Fail on CREATE staging 2
    ]

    with pytest.raises(Exception, match="Simulated failure"):
        write_gold_to_postgres(gold, ml)

    # Rollback should be called
    assert _pg_mock["conn"].rollback.call_count >= 1


def test_rollback_metadata_attached_to_exception(_pg_mock) -> None:
    """Failed sink should attach __postgres_sink_summary__ to exception."""
    gold, ml = _make_inputs()

    import entity_data_lakehouse.postgres_sink as mod
    mod._pool = None

    # Make execution fail before any table swap
    _pg_mock["cursor"].execute.side_effect = Exception("Simulated failure")

    with pytest.raises(Exception) as exc_info:
        write_gold_to_postgres(gold, ml)

    # Check __postgres_sink_summary__ attribute
    assert hasattr(exc_info.value, "__postgres_sink_summary__")
    summary = exc_info.value.__postgres_sink_summary__
    assert summary["status"] == "failed"
    assert summary["batch_id"] is None
    # When failure happens before any table swap, rollback_status should be not_applicable
    assert summary["rollback_status"] == "not_applicable"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_dataframe_handled(_pg_mock) -> None:
    """Empty DataFrames should be handled gracefully (clear table)."""
    gold, ml = _make_inputs()
    # Make one DataFrame empty
    gold["entity_master_event_log"] = pd.DataFrame(columns=gold["entity_master_event_log"].columns)

    import entity_data_lakehouse.postgres_sink as mod
    mod._pool = None

    result = write_gold_to_postgres(gold, ml)

    assert result["status"] == "success"
    # Should still refresh both tables
    assert len(result["tables_refreshed"]) == 2


def test_missing_dataframe_raises_error(_pg_mock) -> None:
    """Missing expected DataFrame should raise ValueError."""
    gold, ml = _make_inputs()
    del gold["entity_master_event_log"]

    import entity_data_lakehouse.postgres_sink as mod
    mod._pool = None

    with pytest.raises(ValueError, match="expected key.*but it was missing"):
        write_gold_to_postgres(gold, ml)


# ---------------------------------------------------------------------------
# DDL parsing tests
# ---------------------------------------------------------------------------

def test_parse_ddl_columns() -> None:
    """DDL parser should extract column names and types."""
    ddl = """
        CREATE TABLE IF NOT EXISTS test_table (
            col1 TEXT NOT NULL,
            col2 INTEGER NOT NULL,
            col3 NUMERIC NOT NULL
        )
    """
    columns = _parse_ddl_columns(ddl)

    assert len(columns) == 3
    assert columns[0] == ("col1", "TEXT")
    assert columns[1] == ("col2", "INTEGER")
    assert columns[2] == ("col3", "NUMERIC")


def test_parse_ddl_normalizes_types() -> None:
    """DDL parser should normalize type names."""
    ddl = """
        CREATE TABLE IF NOT EXISTS test_table (
            col1 BIGINT,
            col2 FLOAT,
            col3 DOUBLE PRECISION
        )
    """
    columns = _parse_ddl_columns(ddl)

    # BIGINT → INTEGER, FLOAT/DOUBLE PRECISION → NUMERIC
    assert columns[0] == ("col1", "INTEGER")
    assert columns[1] == ("col2", "NUMERIC")
    assert columns[2] == ("col3", "NUMERIC")


# ---------------------------------------------------------------------------
# Prepare insert frame tests
# ---------------------------------------------------------------------------

def test_prepare_insert_frame_adds_batch_id() -> None:
    """Prepare should add batch_id column."""
    df = _entity_master_event_log_df()
    run_id = "test_run_123"

    result = _prepare_insert_frame("entity_master_event_log", df, run_id)

    assert "batch_id" in result.columns
    assert (result["batch_id"] == run_id).all()


def test_prepare_insert_frame_projects_to_ddl_order() -> None:
    """Prepare should reorder columns to match DDL."""
    df = _entity_master_event_log_df()
    run_id = "test_run_123"

    result = _prepare_insert_frame("entity_master_event_log", df, run_id)

    # Check column order matches DDL
    expected_order = [
        "entity_id",
        "snapshot_date",
        "snapshot_sequence_number",
        "event_type",
        "batch_id",
    ]
    assert list(result.columns) == expected_order


# ---------------------------------------------------------------------------
# Dry-run mode tests
# ---------------------------------------------------------------------------

def test_dry_run_validates_without_connection() -> None:
    """Dry-run mode should validate schemas without connecting."""
    gold, ml = _make_inputs()

    # No mocking needed — should not attempt to connect
    results = validate_sink_schema(gold, ml)

    assert len(results) == 2
    assert all(r["status"] == "passed" for r in results)


def test_dry_run_catches_schema_errors() -> None:
    """Dry-run mode should catch schema mismatches."""
    gold, ml = _make_inputs()
    gold["entity_master_event_log"]["extra_col"] = "value"

    results = validate_sink_schema(gold, ml)

    failed = [r for r in results if r["status"] == "failed"]
    assert len(failed) == 1
    assert "Extra:" in failed[0]["error"]
