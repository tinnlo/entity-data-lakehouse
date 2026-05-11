"""Optional PostgreSQL serving layer sink.

When ``USE_POSTGRES=true`` is set, selected gold outputs are loaded into
PostgreSQL after the DuckDB pipeline completes. When the flag is absent or
``false``, this module is a no-op and adds no import-time overhead to the
default pipeline path.

Architecture note
-----------------
DuckDB (``gold/entity_lakehouse.duckdb``) is the primary analytics store and
source of truth. PostgreSQL is a **write-through serving layer** — it receives
selected rows that were already written to DuckDB and is intended for
operational serving queries where connection pooling and low-latency point
lookups matter. It is not a backend switch; queries, validation, and failure
handling all centre on DuckDB.

Tables loaded
-------------
- ``entity_master_event_log``           — from gold_outputs["entity_master_event_log"]
- ``owner_infrastructure_exposure_snapshot``
                                        — from gold_outputs["owner_infrastructure_exposure_snapshot"]

Refresh strategy
----------------
Each sink run performs an **atomic full-refresh** using a staging-table swap:

1. Ensure the live table exists (CREATE TABLE IF NOT EXISTS).
2. Drop any leftover staging table from a previous failed run.
3. Create a staging table (``<table>_staging_<uuid>``) with identical schema.
4. Insert all rows into the staging table.
5. Atomically swap staging → live in a transaction:
   - Rename live → old
   - Rename staging → live
   - Drop old
6. Update freshness metadata in ``_refresh_metadata`` table.

If anything fails between steps 3 and 5, the live table still contains the
last successful load. The staging table is cleaned up on the next run (step 2).

Schema validation
-----------------
The DDL column sets are derived directly from the gold output contracts.
No renaming or default-value fabrication is performed. The sink:

- validates that the DataFrame has **exactly** the declared columns (missing
  *and* extra columns are rejected)
- validates that each column's pandas dtype is compatible with the declared
  PostgreSQL type (e.g. INTEGER column must not contain strings)
- projects the DataFrame into DDL declaration order before insert

If any check fails a ``ValueError`` is raised immediately so schema drift is
visible before any data touches PostgreSQL.

Connection pooling
------------------
Uses ``psycopg2.pool.ThreadedConnectionPool`` for connection reuse. The pool
is created once on first use and reused across all table writes. Configure
pool size via environment variables:

- ``POSTGRES_POOL_MIN`` — minimum connections (default: 2)
- ``POSTGRES_POOL_MAX`` — maximum connections (default: 10)

Connection configuration (environment variables)
-------------------------------------------------
USE_POSTGRES         true | false (default: false)
POSTGRES_HOST        hostname or IP (default: localhost)
POSTGRES_PORT        port (default: 5432)
POSTGRES_DATABASE    target database name (default: lakehouse)
POSTGRES_USER        username (default: postgres)
POSTGRES_PASSWORD    password (default: empty string)
POSTGRES_POOL_MIN    minimum pool connections (default: 2)
POSTGRES_POOL_MAX    maximum pool connections (default: 10)

Requires the [postgres] optional dependency group::

    pip install -e '.[postgres]'

Public API
----------
- ``write_gold_to_postgres(gold_outputs, ml_outputs)`` — full sink with
  atomic refresh and rollback. Returns a sink summary dict.
- ``validate_sink_schema(gold_outputs, ml_outputs)`` — schema-only validation
  without connecting to PostgreSQL. Safe for ``dry_run`` mode.

Usage (from pipeline.py)::

    from entity_data_lakehouse.postgres_sink import (
        validate_sink_schema,
        write_gold_to_postgres,
    )
    # dry_run: validate only
    schema_results = validate_sink_schema(gold_outputs, ml_outputs)
    # commit: full atomic refresh
    sink_summary = write_gold_to_postgres(gold_outputs, ml_outputs)
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import psycopg2.pool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection pool singleton
# ---------------------------------------------------------------------------

_pool: psycopg2.pool.ThreadedConnectionPool | None = None


def _get_connection_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Get or create the PostgreSQL connection pool singleton.

    The pool is created once on first use and reused for all subsequent
    operations. Connection parameters are read from environment variables.
    """
    global _pool
    if _pool is None:
        import psycopg2.pool

        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=int(os.getenv("POSTGRES_POOL_MIN", "2")),
            maxconn=int(os.getenv("POSTGRES_POOL_MAX", "10")),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DATABASE", "lakehouse"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )
        logger.info(
            "PostgreSQL connection pool created (min=%s, max=%s, host=%s, database=%s)",
            os.getenv("POSTGRES_POOL_MIN", "2"),
            os.getenv("POSTGRES_POOL_MAX", "10"),
            os.getenv("POSTGRES_HOST", "localhost"),
            os.getenv("POSTGRES_DATABASE", "lakehouse"),
        )
    return _pool


# ---------------------------------------------------------------------------
# DDL templates
#
# Column sets match the actual gold output contracts exactly.
# Derived from the parquet schemas produced by gold.py.
# Any change to the upstream gold schema MUST be reflected here.
# ---------------------------------------------------------------------------

_DDL: dict[str, str] = {
    # gold_outputs["entity_master_event_log"]
    # Columns: entity_id, snapshot_date, snapshot_sequence_number, event_type, batch_id
    "entity_master_event_log": """
        CREATE TABLE IF NOT EXISTS {table} (
            entity_id                   TEXT NOT NULL,
            snapshot_date               TEXT NOT NULL,
            snapshot_sequence_number    INTEGER NOT NULL,
            event_type                  TEXT NOT NULL,
            batch_id                    TEXT NOT NULL
        )
    """,
    # gold_outputs["owner_infrastructure_exposure_snapshot"]
    # Columns: owner_entity_id, asset_country, asset_sector, asset_count,
    #          controlled_asset_count, owned_capacity_mw, average_ownership_pct,
    #          relationship_count, snapshot_date, change_status_vs_prior_snapshot,
    #          batch_id
    "owner_infrastructure_exposure_snapshot": """
        CREATE TABLE IF NOT EXISTS {table} (
            owner_entity_id                 TEXT NOT NULL,
            asset_country                   TEXT NOT NULL,
            asset_sector                    TEXT NOT NULL,
            asset_count                     INTEGER NOT NULL,
            controlled_asset_count          INTEGER NOT NULL,
            owned_capacity_mw               NUMERIC NOT NULL,
            average_ownership_pct           NUMERIC NOT NULL,
            relationship_count              INTEGER NOT NULL,
            snapshot_date                   TEXT NOT NULL,
            change_status_vs_prior_snapshot TEXT NOT NULL,
            batch_id                        TEXT NOT NULL
        )
    """,
}

# Map from DDL table key → (dict_name, key_in_dict).
_TABLE_SOURCES: dict[str, tuple[str, str]] = {
    "entity_master_event_log": ("gold_outputs", "entity_master_event_log"),
    "owner_infrastructure_exposure_snapshot": (
        "gold_outputs",
        "owner_infrastructure_exposure_snapshot",
    ),
}

# ---------------------------------------------------------------------------
# PostgreSQL type → pandas dtype families that are compatible.
# Used by _validate_dtypes() for early, clear failure on type mismatches.
# ---------------------------------------------------------------------------
_PG_TYPE_FAMILIES: dict[str, tuple[str, ...]] = {
    "TEXT": ("object", "string", "str"),
    "NUMERIC": ("float64", "float32", "Float64", "Float32"),
    "INTEGER": ("int64", "int32", "int16", "int8", "Int64", "Int32", "Int16", "Int8"),
}


def _dtype_matches_postgres(pg_type: str, series: "pd.Series") -> bool:
    """Return True when a pandas Series is compatible with a PostgreSQL type."""
    from pandas.api.types import (
        is_float_dtype,
        is_integer_dtype,
        is_string_dtype,
    )

    if pg_type == "TEXT":
        return bool(is_string_dtype(series))
    if pg_type == "NUMERIC":
        return bool(is_float_dtype(series))
    if pg_type == "INTEGER":
        return bool(is_integer_dtype(series))
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_sink_schema(
    gold_outputs: dict[str, "pd.DataFrame"],
    ml_outputs: dict[str, "pd.DataFrame"],
) -> list[dict]:
    """Validate DataFrame schemas against the PostgreSQL DDL contracts.

    Checks column-set match and dtype compatibility for all sink tables
    without connecting to PostgreSQL or mutating any state. Safe to call in
    ``dry_run`` mode.

    Parameters
    ----------
    gold_outputs:
        Dict of DataFrames produced by ``build_gold_outputs`` (gold layer).
    ml_outputs:
        Dict of DataFrames produced by ``build_ml_predictions`` (ML layer).

    Returns
    -------
    list[dict]
        One entry per sink table::

            [
                {"table": "entity_master_event_log", "status": "passed", "error": None},
                {"table": "owner_infrastructure_exposure_snapshot", "status": "failed",
                 "error": "PostgreSQL sink schema mismatch ..."},
                ...
            ]

        All tables are always evaluated; a failure in one does not short-
        circuit the others.
    """
    all_frames = {"gold_outputs": gold_outputs, "ml_outputs": ml_outputs}
    results: list[dict] = []

    # Use a deterministic placeholder run_id for schema-only validation so
    # _prepare_insert_frame can stamp batch_id without actually connecting.
    placeholder_run_id = "dryrun000000"

    for table_name, (dict_name, key) in _TABLE_SOURCES.items():
        df = all_frames[dict_name].get(key)
        if df is None:
            results.append(
                {
                    "table": table_name,
                    "status": "failed",
                    "error": (
                        f"PostgreSQL sink expected key '{key}' in {dict_name}, "
                        "but it was missing."
                    ),
                }
            )
            continue

        try:
            _validate_schema(table_name, df, placeholder_run_id)
            results.append({"table": table_name, "status": "passed", "error": None})
        except ValueError as exc:
            results.append(
                {
                    "table": table_name,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return results


def write_gold_to_postgres(
    gold_outputs: dict[str, "pd.DataFrame"],
    ml_outputs: dict[str, "pd.DataFrame"],
) -> dict:
    """Write selected gold outputs to PostgreSQL with atomic refresh and rollback.

    No-ops immediately when ``USE_POSTGRES`` is not ``"true"``.

    Performs a full-refresh of all sink tables using staging-table swap for
    atomicity. If any table fails, all successfully-swapped tables are rolled
    back to their previous state.

    Parameters
    ----------
    gold_outputs:
        Dict of DataFrames produced by ``build_gold_outputs`` (gold layer).
    ml_outputs:
        Dict of DataFrames produced by ``build_ml_predictions`` (ML layer).

    Returns
    -------
    dict
        Sink summary::

            {
                "tables_refreshed": ["entity_master_event_log", ...],
                "batch_id": "20260510_173045_a1b2c3d4",
                "status": "success",
                "rollback_status": "clean",
            }

        On failure, the exception has a ``__postgres_sink_summary__`` attribute with
        rollback metadata.

    Raises
    ------
    ValueError
        Schema validation failed.
    Exception
        PostgreSQL connection or write error. The exception has a
        ``__postgres_sink_summary__`` attribute with rollback status.
    """
    flag = os.environ.get("USE_POSTGRES", "false").strip().lower()
    if flag != "true":
        return {
            "tables_refreshed": [],
            "batch_id": None,
            "status": "skipped",
            "rollback_status": "not_applicable",
        }

    all_frames = {"gold_outputs": gold_outputs, "ml_outputs": ml_outputs}
    run_id = _generate_run_id()

    logger.info("PostgreSQL sink starting (run_id=%s).", run_id)

    # Track staging tables for rollback
    staging_tables: list[tuple[str, str]] = []  # [(live_table, staging_table)]
    refreshed_tables: list[str] = []  # live table names successfully swapped

    try:
        pool = _get_connection_pool()
        conn = pool.getconn()
    except Exception:
        # Connection setup failed - attach summary before re-raising
        _sink_summary = {
            "tables_refreshed": [],
            "batch_id": None,
            "status": "failed",
            "rollback_status": "not_applicable",
        }
        try:
            raise
        except Exception as _exc:
            _exc.__postgres_sink_summary__ = _sink_summary  # type: ignore[attr-defined]
            raise

    try:
        metadata_table_ensured = False
        for table_name, (dict_name, key) in _TABLE_SOURCES.items():
            df = all_frames[dict_name].get(key)
            if df is None:
                raise ValueError(
                    f"PostgreSQL sink expected key '{key}' in {dict_name}, but it was missing. "
                    "This indicates an upstream pipeline contract change or bug."
                )

            # Ensure refresh metadata table exists before first table refresh.
            # Deferred until after all frame presence checks to avoid committing
            # the metadata table if validation fails.
            if not metadata_table_ensured:
                _ensure_refresh_metadata_table(conn)
                metadata_table_ensured = True

            staging_table = _atomic_refresh(conn, table_name, df, run_id=run_id)
            staging_tables.append((table_name, staging_table))
            refreshed_tables.append(table_name)

        # All tables loaded successfully — commit the transaction.
        # Note: This defers commit until after all table swaps, which means the first
        # table's ACCESS EXCLUSIVE lock is held for the entire batch duration. With
        # concurrent readers, this can block queries. However, committing after each
        # table would break all-or-nothing atomicity: if the second table fails after
        # the first commits, PostgreSQL serves a mixed snapshot and rollback cannot
        # undo the first table. Correctness takes precedence over availability.
        try:
            conn.commit()
            logger.info("PostgreSQL sink complete (run_id=%s).", run_id)
        except Exception as commit_exc:
            # Ambiguous commit failure: if the socket dropped during commit, psycopg2
            # can raise even though the server durably committed the transaction.
            # Check transaction status to determine if we can safely rollback.
            import psycopg2.extensions

            txn_status = conn.get_transaction_status()
            if txn_status == psycopg2.extensions.TRANSACTION_STATUS_IDLE:
                # Transaction committed successfully despite the exception
                logger.warning(
                    "PostgreSQL commit raised exception but transaction is IDLE (committed). "
                    "Treating as success: %s",
                    commit_exc,
                )
                # Return success - don't attempt rollback
                return {
                    "tables_refreshed": list(_TABLE_SOURCES.keys()),
                    "batch_id": run_id,
                    "status": "success",
                    "rollback_status": "clean",
                }
            else:
                # Transaction is still active or in error state - safe to rollback
                raise

    except Exception:
        # Roll back: restore previous state for all successfully-swapped tables
        conn.rollback()
        rollback_failed = False

        for live_table, staging_table in staging_tables[: len(refreshed_tables)]:
            try:
                # The staging table now holds the old data after rollback
                # Clean it up
                with conn.cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {staging_table}")
                logger.warning(
                    "Rolled back %s to previous live data after partial failure.",
                    live_table,
                )
            except Exception as rb_exc:
                rollback_failed = True
                logger.error(
                    "Rollback cleanup of %s failed: %s — manual recovery may be needed.",
                    live_table,
                    rb_exc,
                )

        conn.commit()  # Commit the cleanup

        # Only report rolled_back if we actually swapped tables; otherwise not_applicable
        if refreshed_tables:
            rollback_status = (
                "partial_rollback_failed" if rollback_failed else "rolled_back"
            )
        else:
            rollback_status = "not_applicable"
        # Attach structured summary to the exception so pipeline.py can read it
        # via exc.__postgres_sink_summary__, then re-raise so callers still see the
        # original exception type.
        _sink_summary = {
            "tables_refreshed": list(refreshed_tables),
            "batch_id": None,
            "status": "failed",
            "rollback_status": rollback_status,
        }
        try:
            raise
        except Exception as _exc:
            _exc.__postgres_sink_summary__ = _sink_summary  # type: ignore[attr-defined]
            raise

    finally:
        # Check connection health before returning to pool.
        # Broken connections should be discarded to prevent cascading failures in long-lived workers.
        # IMPORTANT: Do not use 'return' here - it would suppress the active exception.
        try:
            import psycopg2.extensions

            # Check 1: Connection closed by server or client
            if conn.closed != 0:
                logger.warning(
                    "PostgreSQL connection is closed (closed=%d). Discarding instead of returning to pool.",
                    conn.closed,
                )
                try:
                    conn.close()
                except Exception:
                    pass
                pool.putconn(conn, close=True)
            # Check 2: Transaction in error or unknown state
            elif conn.get_transaction_status() in (
                psycopg2.extensions.TRANSACTION_STATUS_INERROR,
                psycopg2.extensions.TRANSACTION_STATUS_UNKNOWN,
            ):
                logger.warning(
                    "PostgreSQL connection in error state (txn_status=%d). Discarding instead of returning to pool.",
                    conn.get_transaction_status(),
                )
                try:
                    conn.close()
                except Exception:
                    pass
                pool.putconn(conn, close=True)
            else:
                # Connection appears healthy - return to pool for reuse
                pool.putconn(conn)

        except Exception as health_check_exc:
            # If health check itself fails, assume connection is broken
            logger.warning(
                "Failed to check PostgreSQL connection health: %s. Discarding connection.",
                health_check_exc,
            )
            try:
                conn.close()
            except Exception:
                pass
            try:
                pool.putconn(conn, close=True)
            except Exception:
                pass

    return {
        "tables_refreshed": list(_TABLE_SOURCES.keys()),
        "batch_id": run_id,
        "status": "success",
        "rollback_status": "clean",
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _generate_run_id() -> str:
    """Generate a unique run identifier for this sink execution.

    Format: YYYYMMDD_HHMMSS_<8-char-uuid>
    Example: 20260510_173045_a1b2c3d4
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"{timestamp}_{short_uuid}"


def _parse_ddl_columns(ddl: str) -> list[tuple[str, str]]:
    """Extract (column_name, pg_type) pairs from a CREATE TABLE DDL string.

    Returns
    -------
    list[tuple[str, str]]
        [(column_name, pg_type), ...] in declaration order.
    """
    # Match lines like: column_name TYPE [NOT NULL] [,]
    pattern = re.compile(
        r"^\s*(\w+)\s+(TEXT|INTEGER|NUMERIC|BIGINT|FLOAT|DOUBLE PRECISION)",
        re.IGNORECASE,
    )
    columns = []
    for line in ddl.splitlines():
        match = pattern.match(line)
        if match:
            col_name = match.group(1)
            pg_type = match.group(2).upper()
            # Normalize type names
            if pg_type in ("BIGINT",):
                pg_type = "INTEGER"
            if pg_type in ("FLOAT", "DOUBLE PRECISION"):
                pg_type = "NUMERIC"
            columns.append((col_name, pg_type))
    return columns


def _validate_schema(table_name: str, df: "pd.DataFrame", run_id: str) -> None:
    """Validate DataFrame schema against DDL contract.

    Raises
    ------
    ValueError
        Column set mismatch or dtype incompatibility.
    """
    ddl = _DDL[table_name]
    expected_cols = _parse_ddl_columns(ddl)
    expected_names = [col for col, _ in expected_cols]

    # Add batch_id if not present (will be stamped during insert)
    df_cols = list(df.columns)
    if "batch_id" not in df_cols:
        df_cols.append("batch_id")

    # Check column set match
    missing = set(expected_names) - set(df_cols)
    extra = set(df_cols) - set(expected_names)

    if missing or extra:
        raise ValueError(
            f"PostgreSQL sink schema mismatch for {table_name}. "
            f"Missing: {sorted(missing) or 'none'}. "
            f"Extra: {sorted(extra) or 'none'}."
        )

    # Check dtype compatibility and NULL constraints
    for col_name, pg_type in expected_cols:
        if col_name == "batch_id":
            continue  # Will be stamped as string
        if col_name not in df.columns:
            continue  # Already caught by column set check

        series = df[col_name]
        # Skip dtype check for empty DataFrames (pandas may not preserve dtypes)
        if len(series) == 0:
            continue
        if not _dtype_matches_postgres(pg_type, series):
            raise ValueError(
                f"PostgreSQL sink dtype mismatch for {table_name}.{col_name}: "
                f"expected {pg_type}-compatible, got {series.dtype}"
            )

        # Check for NULL values (all PostgreSQL sink columns are NOT NULL)
        if series.isna().any():
            null_count = series.isna().sum()
            raise ValueError(
                f"PostgreSQL sink NULL constraint violation for {table_name}.{col_name}: "
                f"found {null_count} NULL value(s), but column is NOT NULL"
            )


def _prepare_insert_frame(
    table_name: str, df: "pd.DataFrame", run_id: str
) -> "pd.DataFrame":
    """Prepare DataFrame for insert: add batch_id and project to DDL order.

    Returns
    -------
    pd.DataFrame
        Copy of df with batch_id column added and columns reordered to match DDL.
    """
    ddl = _DDL[table_name]
    expected_cols = _parse_ddl_columns(ddl)
    expected_names = [col for col, _ in expected_cols]

    # Add batch_id column
    df = df.copy()
    df["batch_id"] = run_id

    # Project to DDL order
    return df[expected_names]


def _ensure_refresh_metadata_table(conn) -> None:
    """Ensure the _refresh_metadata table exists."""
    ddl = """
        CREATE TABLE IF NOT EXISTS _refresh_metadata (
            table_name TEXT PRIMARY KEY,
            last_refreshed_at TIMESTAMPTZ DEFAULT NOW(),
            batch_id TEXT NOT NULL,
            row_count INTEGER NOT NULL
        )
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def _atomic_refresh(conn, table_name: str, df: "pd.DataFrame", run_id: str) -> str:
    """Perform atomic staging-table swap for a single table.

    Returns
    -------
    str
        The staging table name (for rollback tracking).

    Raises
    ------
    ValueError
        Schema validation failed.
    Exception
        PostgreSQL DDL or write error.
    """
    # Validate schema before touching the database
    _validate_schema(table_name, df, run_id)

    # Prepare insert frame
    insert_df = _prepare_insert_frame(table_name, df, run_id)

    # Generate staging table name
    staging_table = f"{table_name}_staging_{uuid.uuid4().hex[:8]}"

    with conn.cursor() as cur:
        # 1. Ensure live table exists
        ddl = _DDL[table_name].format(table=table_name)
        cur.execute(ddl)

        # 2. Clean up orphaned staging tables from previous failed runs
        # Because staging_table gets a fresh UUID suffix on every invocation,
        # DROP TABLE IF EXISTS {staging_table} only targets the current run.
        # We need to find and drop all old staging tables with this pattern.
        # To avoid interfering with concurrent runs, only drop tables that are not
        # currently locked by ANY session (not just AccessExclusiveLock - concurrent
        # inserts hold RowExclusiveLock).
        cur.execute(
            f"""
            SELECT c.relname
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            LEFT JOIN pg_locks l ON l.relation = c.oid AND l.pid != pg_backend_pid()
            WHERE n.nspname = 'public'
              AND c.relkind = 'r'
              AND c.relname LIKE '{table_name}_staging_%'
              AND l.relation IS NULL
            """
        )
        orphaned_staging = [row[0] for row in cur.fetchall()]
        for orphan in orphaned_staging:
            # Use a savepoint to isolate orphan cleanup from the main transaction.
            # If DROP blocks or fails (another session grabbed the table), we can
            # roll back to the savepoint without aborting the entire refresh.
            try:
                cur.execute("SAVEPOINT drop_orphan")
                cur.execute(f"DROP TABLE IF EXISTS {orphan}")
                cur.execute("RELEASE SAVEPOINT drop_orphan")
                logger.debug("Dropped orphaned staging table: %s", orphan)
            except Exception as drop_exc:
                # Roll back to savepoint to keep the main transaction alive
                try:
                    cur.execute("ROLLBACK TO SAVEPOINT drop_orphan")
                except Exception:
                    pass
                logger.warning(
                    "Could not drop orphaned staging table %s: %s", orphan, drop_exc
                )

        # 3. Create staging table with identical schema
        staging_ddl = _DDL[table_name].format(table=staging_table)
        cur.execute(staging_ddl)

        # 4. Insert all rows into staging table
        if len(insert_df) > 0:
            # Use COPY for bulk insert (much faster than INSERT)
            # Note: We use psycopg2's execute_values for proper escaping instead of
            # raw COPY to avoid format issues with tabs/newlines/backslashes in data
            from psycopg2.extras import execute_values

            columns = list(insert_df.columns)
            values = [
                tuple(row) for row in insert_df.itertuples(index=False, name=None)
            ]

            # Build INSERT statement with placeholders
            cols_str = ", ".join(columns)
            insert_sql = f"INSERT INTO {staging_table} ({cols_str}) VALUES %s"

            execute_values(cur, insert_sql, values, page_size=1000)

        # 5. Atomically swap staging → live in transaction
        # PostgreSQL doesn't have EXCHANGE TABLES, so we use rename sequence
        old_table = f"{table_name}_old_{uuid.uuid4().hex[:8]}"

        # Rename live → old (if exists)
        cur.execute(f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables
                          WHERE table_name = '{table_name}') THEN
                    ALTER TABLE {table_name} RENAME TO {old_table};
                END IF;
            END $$;
        """)

        # Rename staging → live
        cur.execute(f"ALTER TABLE {staging_table} RENAME TO {table_name}")

        # Drop old table
        cur.execute(f"DROP TABLE IF EXISTS {old_table}")

        # 6. Update freshness metadata
        cur.execute(
            """
            INSERT INTO _refresh_metadata (table_name, last_refreshed_at, batch_id, row_count)
            VALUES (%s, NOW(), %s, %s)
            ON CONFLICT (table_name)
            DO UPDATE SET
                last_refreshed_at = NOW(),
                batch_id = EXCLUDED.batch_id,
                row_count = EXCLUDED.row_count
        """,
            (table_name, run_id, len(insert_df)),
        )

    logger.info(
        "Refreshed %s: %d rows (run_id=%s)",
        table_name,
        len(insert_df),
        run_id,
    )

    return staging_table
