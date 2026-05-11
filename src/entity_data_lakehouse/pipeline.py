from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from pathlib import Path

import duckdb

from .bronze import ingest_sample_data
from .gold import build_gold_outputs
from .ml import build_ml_predictions
from .public_safety import scan_public_safety
from .silver import build_silver_outputs

logger = logging.getLogger(__name__)

_VALID_PUBLISH_MODES = {"dry_run", "commit"}

# Tables that the ClickHouse sink will attempt to refresh (in order).
_CLICKHOUSE_SINK_TABLES = [
    "ownership_current",
    "owner_infrastructure_exposure_snapshot",
    "ml_asset_lifecycle_predictions",
]

# Tables that the PostgreSQL sink will attempt to refresh (in order).
_POSTGRES_SINK_TABLES = [
    "entity_master_event_log",
    "owner_infrastructure_exposure_snapshot",
]


def run_pipeline(
    repo_root: Path,
    *,
    publish_mode: str = "commit",
    report_path: Path | None = None,
) -> dict[str, int]:
    """Run the full bronze → silver → gold → ML pipeline.

    Parameters
    ----------
    repo_root:
        Absolute path to the repository root.  All data paths are resolved
        relative to this directory.
    publish_mode:
        ``"commit"`` (default) — full pipeline with all disk writes and
        optional ClickHouse sink.  Behaviour is identical to the pre-publish-
        mode baseline.

        ``"dry_run"`` — all computation and contract validation runs but
        **no disk writes** are performed (no parquet files, no DuckDB tables,
        no ClickHouse mutations).  The only artifact produced is
        ``publish_report.json``.  Useful for CI preflight, demo review, and
        manual pre-publish checks.
    report_path:
        Where to write ``publish_report.json``.  Defaults to
        ``{repo_root}/gold/publish_report.json``.  The parent directory is
        created if it does not exist.

    Returns
    -------
    dict[str, int]
        Row counts keyed by layer:
        ``entity_master_rows``, ``asset_master_rows``,
        ``relationship_edge_rows``, ``gold_rows``, ``ml_prediction_rows``.

    Raises
    ------
    ValueError
        On invalid ``publish_mode``, failed contract validations, or a failed
        public-safety scan.
    """
    if publish_mode not in _VALID_PUBLISH_MODES:
        raise ValueError(
            f"Invalid publish_mode {publish_mode!r}. "
            f"Must be one of: {sorted(_VALID_PUBLISH_MODES)}"
        )

    dry_run = publish_mode == "dry_run"
    run_id = uuid.uuid4().hex[:12]
    started_at = (
        datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    )

    if report_path is None:
        report_path = repo_root / "gold" / "publish_report.json"

    clickhouse_enabled = (
        os.environ.get("USE_CLICKHOUSE", "false").strip().lower() == "true"
    )
    postgres_enabled = os.environ.get("USE_POSTGRES", "false").strip().lower() == "true"

    # Determine which tables will be attempted based on enabled sinks.
    # In dry-run mode, include tables from sinks that will be validated even if disabled.
    tables_attempted = []
    if clickhouse_enabled or dry_run:
        tables_attempted.extend(_CLICKHOUSE_SINK_TABLES)
    if postgres_enabled:
        # Add PostgreSQL tables, avoiding duplicates
        for table in _POSTGRES_SINK_TABLES:
            if table not in tables_attempted:
                tables_attempted.append(table)

    # Initialise the report skeleton — fields are filled in as the pipeline
    # progresses.  Written at the end regardless of success or failure.
    report: dict = {
        "schema_version": "1",
        "report_timestamp": started_at,
        "run_id": run_id,
        "publish_mode": publish_mode,
        "status": "failed",  # overwritten to "success" at end
        "tables_attempted": tables_attempted,
        "row_counts": {},
        "rollback_status": "not_applicable",
        "sink_target": {
            "clickhouse_enabled": clickhouse_enabled,
            "tables_refreshed": [],
            "batch_id": None,
            "status": "skipped"
            if not clickhouse_enabled
            else ("dry_run_validated" if dry_run else "not_started"),
            "schema_validations": [],
        },
        "public_safety": {"status": "pending", "findings": []},
        "artifacts_written": [],
    }

    contracts_root = repo_root / "contracts"
    sample_root = repo_root / "sample_data"
    reference_root = repo_root / "reference_data"

    try:
        # Initialize dual-sink coordination state
        ch_ex_live_tables = []  # ClickHouse rollback tables (populated when defer_cleanup=True)

        # ------------------------------------------------------------------
        # Bronze
        # ------------------------------------------------------------------
        bronze_parquet_root = repo_root / "bronze"
        ingest_sample_data(
            sample_root=sample_root,
            bronze_root=bronze_parquet_root,
            contract_path=contracts_root / "bronze_source_record.schema.json",
            dry_run=dry_run,
        )

        # ------------------------------------------------------------------
        # Silver
        # ------------------------------------------------------------------
        silver_outputs = build_silver_outputs(
            sample_root=sample_root,
            silver_root=repo_root / "silver",
            contract_paths={
                "entity_observations": contracts_root
                / "entity_observations.schema.json",
                "entity_master": contracts_root / "entity_master.schema.json",
                "asset_master": contracts_root / "asset_master.schema.json",
                "ownership_observations": contracts_root
                / "ownership_observations.schema.json",
                "relationship_edges": contracts_root / "relationship_edges.schema.json",
            },
            dry_run=dry_run,
        )

        # ------------------------------------------------------------------
        # Gold
        # ------------------------------------------------------------------
        gold_outputs, gold_artifacts = build_gold_outputs(
            gold_root=repo_root / "gold",
            silver_outputs=silver_outputs,
            contract_paths={
                "entity_master_comprehensive_scd4": contracts_root
                / "entity_master_comprehensive_scd4.schema.json",
                "entity_master_current": contracts_root
                / "entity_master_current.schema.json",
                "entity_master_event_log": contracts_root
                / "entity_master_event_log.schema.json",
                "ownership_comprehensive_scd4": contracts_root
                / "ownership_comprehensive_scd4.schema.json",
                "ownership_lifecycle": contracts_root
                / "ownership_lifecycle.schema.json",
                "ownership_history_scd2": contracts_root
                / "ownership_history_scd2.schema.json",
                "ownership_current": contracts_root / "ownership_current.schema.json",
                "owner_infrastructure_exposure_snapshot": contracts_root
                / "owner_infrastructure_exposure_snapshot.schema.json",
            },
            dry_run=dry_run,
        )

        # gold_artifacts is collected incrementally inside build_gold_outputs().
        # On success this reflects every file written.  If build_gold_outputs()
        # raises mid-write it attaches a partial list to __gold_artifacts__ on the
        # exception; the except-block below picks that up instead, so failure
        # reports are always accurate regardless of where gold writing stops.
        report["artifacts_written"] = list(gold_artifacts)

        # ------------------------------------------------------------------
        # Safety gate — runs before ML so telemetry never fires on failure
        # ------------------------------------------------------------------
        safety_findings = scan_public_safety(repo_root)
        if safety_findings:
            report["public_safety"] = {"status": "failed", "findings": safety_findings}
            raise ValueError(
                "Public-safety scan failed:\n" + "\n".join(safety_findings)
            )
        report["public_safety"] = {"status": "passed", "findings": []}

        # ------------------------------------------------------------------
        # ML
        # ------------------------------------------------------------------
        ml_outputs = build_ml_predictions(
            gold_root=repo_root / "gold",
            silver_outputs=silver_outputs,
            gold_outputs=gold_outputs,
            reference_root=reference_root,
            contract_paths={
                "asset_lifecycle_predictions": contracts_root
                / "asset_lifecycle_predictions.schema.json",
            },
            dry_run=dry_run,
        )

        ml_predictions = ml_outputs["asset_lifecycle_predictions"]

        # ML parquet is written by build_ml_predictions(); record it now.
        if not dry_run:
            report["artifacts_written"].append(
                "gold/dw/asset_lifecycle_predictions.parquet"
            )

        # ------------------------------------------------------------------
        # Row counts (computed from in-memory frames — always available)
        # ------------------------------------------------------------------
        row_counts = {
            "entity_master_rows": len(silver_outputs["entity_master"]),
            "asset_master_rows": len(silver_outputs["asset_master"]),
            "relationship_edge_rows": len(silver_outputs["relationship_edges"]),
            "gold_rows": len(gold_outputs["owner_infrastructure_exposure_snapshot"]),
            "ml_prediction_rows": len(ml_predictions),
        }
        report["row_counts"] = row_counts

        # ------------------------------------------------------------------
        # DuckDB registration (commit only)
        # ------------------------------------------------------------------
        if not dry_run:
            duckdb_path = repo_root / "gold" / "entity_lakehouse.duckdb"
            con = duckdb.connect(str(duckdb_path))
            try:
                con.execute(
                    "CREATE OR REPLACE TABLE ml_asset_lifecycle_predictions "
                    "AS SELECT * FROM ml_predictions"
                )
            finally:
                con.close()
            logger.info(
                "Registered ML predictions in DuckDB: %d rows.", len(ml_predictions)
            )

        # ------------------------------------------------------------------
        # Sink preflight checks (when both sinks are enabled)
        # ------------------------------------------------------------------
        # When both ClickHouse and PostgreSQL are enabled, validate connectivity
        # and schemas for both before publishing either one. This prevents a scenario
        # where ClickHouse publishes successfully but PostgreSQL fails, leaving the
        # two sinks serving different snapshots.
        if not dry_run and clickhouse_enabled and postgres_enabled:
            from .postgres_sink import (
                validate_sink_schema as validate_postgres_schema,
                _get_connection_pool as get_postgres_pool,
            )

            # Test PostgreSQL connectivity with actual round-trip query.
            # In long-lived workers, pooled connections can become stale after PostgreSQL
            # restarts, so getconn()/putconn() alone isn't sufficient - we need a real query.
            try:
                pg_pool = get_postgres_pool()
                conn = pg_pool.getconn()
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                    # Reset connection state before returning to pool to avoid leaving it
                    # "idle in transaction" which can hold snapshots and interfere with vacuum.
                    conn.rollback()
                finally:
                    # Check connection health before returning to pool
                    try:
                        import psycopg2.extensions

                        if conn.closed != 0:
                            logger.warning(
                                "Preflight PostgreSQL connection is closed (closed=%d). Discarding.",
                                conn.closed,
                            )
                            try:
                                conn.close()
                            except Exception:
                                pass
                            pg_pool.putconn(conn, close=True)
                        else:
                            txn_status = conn.get_transaction_status()
                            if txn_status in (
                                psycopg2.extensions.TRANSACTION_STATUS_INERROR,
                                psycopg2.extensions.TRANSACTION_STATUS_UNKNOWN,
                            ):
                                logger.warning(
                                    "Preflight PostgreSQL connection in error state (txn_status=%d). Discarding.",
                                    txn_status,
                                )
                                try:
                                    conn.close()
                                except Exception:
                                    pass
                                pg_pool.putconn(conn, close=True)
                            else:
                                pg_pool.putconn(conn)
                    except Exception as health_check_exc:
                        logger.warning(
                            "Failed to check preflight PostgreSQL connection health: %s. Discarding.",
                            health_check_exc,
                        )
                        try:
                            conn.close()
                        except Exception:
                            pass
                        try:
                            pg_pool.putconn(conn, close=True)
                        except Exception:
                            pass
            except Exception as exc:
                raise RuntimeError(
                    "Preflight check failed: cannot connect to PostgreSQL. "
                    "When both sinks are enabled, both must be reachable before publishing."
                ) from exc

            # Validate PostgreSQL schemas before allowing ClickHouse to publish
            pg_schema_validations = validate_postgres_schema(gold_outputs, ml_outputs)
            pg_all_passed = all(v["status"] == "passed" for v in pg_schema_validations)
            if not pg_all_passed:
                failed = [
                    v["table"] for v in pg_schema_validations if v["status"] != "passed"
                ]
                raise ValueError(
                    f"Preflight check failed: PostgreSQL schema validation failed for tables: {failed}. "
                    "When both sinks are enabled, both must pass schema validation before publishing."
                )

            logger.info(
                "Preflight check passed: both ClickHouse and PostgreSQL are reachable and schemas are valid."
            )

        # ------------------------------------------------------------------
        # ClickHouse sink
        # ------------------------------------------------------------------
        from .clickhouse_sink import validate_sink_schema, write_gold_to_clickhouse

        if dry_run:
            # Validate schemas only — no connection, no mutation.
            schema_validations = validate_sink_schema(gold_outputs, ml_outputs)
            all_passed = all(v["status"] == "passed" for v in schema_validations)
            report["sink_target"]["schema_validations"] = schema_validations
            report["sink_target"]["status"] = (
                "dry_run_validated" if all_passed else "dry_run_schema_failed"
            )
            logger.info(
                "dry_run: ClickHouse schema validation %s.",
                "passed" if all_passed else "FAILED",
            )
            if not all_passed:
                failed = [
                    v["table"] for v in schema_validations if v["status"] != "passed"
                ]
                raise ValueError(
                    f"dry_run: ClickHouse schema validation failed for tables: {failed}"
                )
        else:
            # Full atomic refresh — only runs when USE_CLICKHOUSE=true.
            # In dual-sink mode, defer cleanup to preserve rollback capability.
            sink_summary = write_gold_to_clickhouse(
                gold_outputs, ml_outputs, defer_cleanup=postgres_enabled
            )
            report["sink_target"]["tables_refreshed"] = sink_summary["tables_refreshed"]
            report["sink_target"]["batch_id"] = sink_summary["batch_id"]
            report["sink_target"]["status"] = sink_summary["status"]
            report["sink_target"]["rollback_status"] = sink_summary["rollback_status"]
            ch_ex_live_tables = sink_summary.get("ex_live_tables", [])

        # ------------------------------------------------------------------
        # PostgreSQL sink
        # ------------------------------------------------------------------
        if postgres_enabled:
            from .postgres_sink import (
                validate_sink_schema as validate_postgres_schema,
                write_gold_to_postgres,
            )

            if dry_run:
                # Validate schemas only — no connection, no mutation.
                pg_schema_validations = validate_postgres_schema(
                    gold_outputs, ml_outputs
                )
                pg_all_passed = all(
                    v["status"] == "passed" for v in pg_schema_validations
                )
                pg_status = (
                    "dry_run_validated" if pg_all_passed else "dry_run_schema_failed"
                )
                report["postgres_sink"] = {
                    "schema_validations": pg_schema_validations,
                    "status": pg_status,
                }

                # If PostgreSQL is the only sink, also populate sink_target for backward compatibility
                if not clickhouse_enabled:
                    report["sink_target"]["schema_validations"] = pg_schema_validations
                    report["sink_target"]["status"] = pg_status

                logger.info(
                    "dry_run: PostgreSQL schema validation %s.",
                    "passed" if pg_all_passed else "FAILED",
                )
                if not pg_all_passed:
                    failed = [
                        v["table"]
                        for v in pg_schema_validations
                        if v["status"] != "passed"
                    ]
                    raise ValueError(
                        f"dry_run: PostgreSQL schema validation failed for tables: {failed}"
                    )
            else:
                # Full atomic refresh — only runs when USE_POSTGRES=true.
                pg_sink_summary = write_gold_to_postgres(gold_outputs, ml_outputs)
                report["postgres_sink"] = {
                    "tables_refreshed": pg_sink_summary["tables_refreshed"],
                    "batch_id": pg_sink_summary["batch_id"],
                    "status": pg_sink_summary["status"],
                    "rollback_status": pg_sink_summary["rollback_status"],
                }

                # If PostgreSQL is the only sink, also populate sink_target for backward compatibility
                # with consumers that still key off sink_target. Update fields instead of replacing
                # the object to preserve clickhouse_enabled and schema_validations keys.
                if not clickhouse_enabled:
                    report["sink_target"]["tables_refreshed"] = pg_sink_summary[
                        "tables_refreshed"
                    ]
                    report["sink_target"]["batch_id"] = pg_sink_summary["batch_id"]
                    report["sink_target"]["status"] = pg_sink_summary["status"]
                    report["sink_target"]["rollback_status"] = pg_sink_summary[
                        "rollback_status"
                    ]

                # Dual-sink coordination: now that PostgreSQL has committed, publish the
                # ClickHouse batch_id to make the new snapshot visible. Then clean up rollback
                # tables. Batch publication failure is a correctness issue (divergent snapshots),
                # so we mark it as partial_publish and re-raise to fail the run. Cleanup failure is non-fatal.
                if clickhouse_enabled and ch_ex_live_tables:
                    from .clickhouse_sink import (
                        cleanup_clickhouse_rollback,
                        publish_clickhouse_batch_id,
                    )

                    # Publish batch_id first to make the snapshot visible
                    try:
                        batch_id = report["sink_target"]["batch_id"]
                        publish_clickhouse_batch_id(batch_id)
                    except Exception as publish_exc:
                        logger.error(
                            "Failed to publish ClickHouse batch_id after successful PostgreSQL commit: %s. "
                            "ClickHouse is serving the old snapshot while PostgreSQL serves the new one. "
                            "This is a correctness failure requiring manual recovery.",
                            publish_exc,
                        )
                        # Mark as partial_publish since the sinks are now serving different snapshots
                        report["sink_target"]["status"] = "partial_publish"
                        report["sink_target"]["rollback_status"] = "partial_publish"
                        report["rollback_status"] = "partial_publish"

                        # Write report before re-raising so failure state is recorded
                        try:
                            _write_report(report, report_path, dry_run=dry_run)
                        except Exception as report_exc:
                            logger.error(
                                "Failed to write publish report after batch_id publication failure: %s",
                                report_exc,
                            )

                        # Re-raise to propagate failure to caller (CLI/Airflow exits non-zero)
                        raise RuntimeError(
                            f"Dual-sink coordination failed: PostgreSQL committed but ClickHouse batch_id "
                            f"publication failed. The sinks are serving different snapshots (partial_publish). "
                            f"Manual recovery required: publish batch_id {batch_id} to ClickHouse."
                        ) from publish_exc

                    # Clean up rollback tables (non-fatal)
                    try:
                        cleanup_clickhouse_rollback(ch_ex_live_tables)
                    except Exception as cleanup_exc:
                        logger.warning(
                            "Failed to clean up ClickHouse rollback tables after successful dual-sink publish: %s. "
                            "This is non-fatal; tables can be cleaned up manually.",
                            cleanup_exc,
                        )

        # Update top-level rollback_status based on which sinks ran
        ch_rollback = report.get("sink_target", {}).get(
            "rollback_status", "not_applicable"
        )
        pg_rollback = report.get("postgres_sink", {}).get(
            "rollback_status", "not_applicable"
        )

        # Aggregate rollback states with priority:
        # partial_rollback_failed > rolled_back > partial_publish > clean > not_applicable
        # Only check for partial_publish when both sinks are enabled
        both_enabled = clickhouse_enabled and postgres_enabled

        if (
            ch_rollback == "partial_rollback_failed"
            or pg_rollback == "partial_rollback_failed"
        ):
            report["rollback_status"] = "partial_rollback_failed"
        elif ch_rollback == "rolled_back" or pg_rollback == "rolled_back":
            report["rollback_status"] = "rolled_back"
        elif both_enabled and (
            (ch_rollback == "clean" and pg_rollback != "clean")
            or (pg_rollback == "clean" and ch_rollback != "clean")
        ):
            # One sink succeeded, the other did not run or failed without rollback.
            # This is a partial publish: the sinks are on different snapshots.
            report["rollback_status"] = "partial_publish"
        elif ch_rollback == "clean" or pg_rollback == "clean":
            report["rollback_status"] = "clean"
        else:
            report["rollback_status"] = "not_applicable"

        report["status"] = "success"
        logger.info(
            "Pipeline complete (publish_mode=%s, run_id=%s).", publish_mode, run_id
        )

    except Exception as exc:
        report["status"] = "failed"
        logger.error(
            "Pipeline failed (publish_mode=%s, run_id=%s): %s",
            publish_mode,
            run_id,
            exc,
        )
        # If build_gold_outputs() failed mid-write it attaches whatever was
        # already written to __gold_artifacts__.  Use that to overwrite the
        # (possibly empty) report list so recovery/cleanup info is accurate.
        gold_partial = getattr(exc, "__gold_artifacts__", None)
        if gold_partial is not None:
            report["artifacts_written"] = list(gold_partial)
        # If the ClickHouse sink attached structured rollback metadata, inject
        # it into the failure report before writing.
        sink_summary = getattr(exc, "__sink_summary__", None)
        if sink_summary is not None:
            report["sink_target"]["tables_refreshed"] = sink_summary.get(
                "tables_refreshed", []
            )
            report["sink_target"]["batch_id"] = sink_summary.get("batch_id")
            report["sink_target"]["status"] = sink_summary.get("status", "failed")
            report["sink_target"]["rollback_status"] = sink_summary.get(
                "rollback_status", "not_applicable"
            )
        elif report["sink_target"]["status"] == "not_started":
            # Failure occurred before the sink was reached or before __sink_summary__
            # was attached (e.g. ClickHouse config/connection error).  Overwrite the
            # non-terminal initialisation value with a terminal "failed" so the report
            # is always machine-readable without ambiguity.
            report["sink_target"]["status"] = "failed"

        # If the PostgreSQL sink attached structured rollback metadata, inject
        # it into the failure report before writing.
        pg_sink_summary = getattr(exc, "__postgres_sink_summary__", None)
        if pg_sink_summary is not None:
            report["postgres_sink"] = {
                "tables_refreshed": pg_sink_summary.get("tables_refreshed", []),
                "batch_id": pg_sink_summary.get("batch_id"),
                "status": pg_sink_summary.get("status", "failed"),
                "rollback_status": pg_sink_summary.get(
                    "rollback_status", "not_applicable"
                ),
            }

            # If PostgreSQL is the only sink, also mirror the failure into sink_target
            # for backward compatibility with consumers that still key off sink_target
            if not clickhouse_enabled:
                report["sink_target"]["tables_refreshed"] = pg_sink_summary.get(
                    "tables_refreshed", []
                )
                report["sink_target"]["batch_id"] = pg_sink_summary.get("batch_id")
                report["sink_target"]["status"] = pg_sink_summary.get(
                    "status", "failed"
                )
                report["sink_target"]["rollback_status"] = pg_sink_summary.get(
                    "rollback_status", "not_applicable"
                )

        # Dual-sink coordination: if ClickHouse succeeded but PostgreSQL failed,
        # roll back ClickHouse to restore atomicity across both sinks.
        # We check for ch_ex_live_tables (meaning ClickHouse published with deferred cleanup)
        # rather than relying on pg_status, because PostgreSQL can crash before setting
        # report["postgres_sink"], leaving pg_status as "not_started" even though it failed.
        ch_status = report.get("sink_target", {}).get("status", "not_started")
        pg_status = report.get("postgres_sink", {}).get("status", "not_started")
        if (
            clickhouse_enabled
            and postgres_enabled
            and ch_status == "success"
            and pg_status != "success"
        ):
            if ch_ex_live_tables:
                from .clickhouse_sink import rollback_clickhouse

                ch_batch_id = report["sink_target"]["batch_id"]
                try:
                    ch_rollback_status = rollback_clickhouse(
                        ch_ex_live_tables,
                        report["sink_target"]["tables_refreshed"],
                        ch_batch_id,
                    )
                    report["sink_target"]["rollback_status"] = ch_rollback_status

                    # Clear the ClickHouse success markers after rollback.
                    # The rollback restored previous live tables and deleted the batch marker,
                    # so sink_target should reflect that the publish was rolled back, not successful.
                    if ch_rollback_status == "rolled_back":
                        report["sink_target"]["status"] = "rolled_back"
                        report["sink_target"]["batch_id"] = None
                    elif ch_rollback_status == "partial_rollback_failed":
                        report["sink_target"]["status"] = "partial_rollback_failed"
                        # Keep batch_id for manual recovery reference

                    logger.warning(
                        "Rolled back ClickHouse after PostgreSQL failure (pg_status=%s, rollback=%s).",
                        pg_status,
                        ch_rollback_status,
                    )
                except Exception as rollback_exc:
                    # If rollback fails (e.g., ClickHouse connection lost), preserve the original
                    # PostgreSQL failure and report the rollback failure as secondary context.
                    # Mark ClickHouse as failed since manual recovery is needed.
                    report["sink_target"]["status"] = "partial_rollback_failed"
                    report["sink_target"]["rollback_status"] = "partial_rollback_failed"
                    logger.error(
                        "Failed to rollback ClickHouse after PostgreSQL failure: %s. "
                        "Original PostgreSQL failure preserved in report. Manual recovery needed.",
                        rollback_exc,
                    )

        # Update top-level rollback_status based on which sinks failed/rolled back
        ch_rollback = report.get("sink_target", {}).get(
            "rollback_status", "not_applicable"
        )
        pg_rollback = report.get("postgres_sink", {}).get(
            "rollback_status", "not_applicable"
        )

        # Aggregate rollback states with priority:
        # partial_rollback_failed > rolled_back > partial_publish > clean > not_applicable
        # Only check for partial_publish when both sinks are enabled
        both_enabled = clickhouse_enabled and postgres_enabled

        if (
            ch_rollback == "partial_rollback_failed"
            or pg_rollback == "partial_rollback_failed"
        ):
            report["rollback_status"] = "partial_rollback_failed"
        elif ch_rollback == "rolled_back" or pg_rollback == "rolled_back":
            report["rollback_status"] = "rolled_back"
        elif both_enabled and (
            (ch_rollback == "clean" and pg_rollback != "clean")
            or (pg_rollback == "clean" and ch_rollback != "clean")
        ):
            # One sink succeeded, the other did not run or failed without rollback.
            # This is a partial publish: the sinks are on different snapshots.
            report["rollback_status"] = "partial_publish"
        elif ch_rollback == "clean" or pg_rollback == "clean":
            report["rollback_status"] = "clean"
        else:
            report["rollback_status"] = "not_applicable"

        _write_report(report, report_path, dry_run=dry_run)
        raise

    _write_report(report, report_path, dry_run=dry_run)
    return row_counts


def _write_report(report: dict, report_path: Path, *, dry_run: bool = False) -> None:
    """Write publish_report.json to *report_path*, creating parent dirs as needed.

    In ``dry_run`` mode the report is the *only* permitted artifact, so a write
    failure is fatal and raises ``RuntimeError``.  In ``commit`` mode a write
    failure is non-fatal (all real artifacts are already on disk), so only a
    warning is logged.
    """
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Publish report written to %s", report_path)
    except Exception as exc:
        msg = f"Could not write publish report to {report_path}: {exc}"
        if dry_run:
            raise RuntimeError(msg) from exc
        logger.warning(msg)
