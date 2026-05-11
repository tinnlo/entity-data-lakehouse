"""Unit tests for ClickHouse rollback_clickhouse() function.

Tests the concurrent publish detection and rollback safety logic.
All clickhouse_connect calls are mocked — no running ClickHouse server required.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

import pytest

from entity_data_lakehouse.clickhouse_sink import rollback_clickhouse


@pytest.fixture
def mock_client():
    """Create a mock ClickHouse client."""
    return MagicMock()


@pytest.fixture
def mock_clickhouse_env(monkeypatch, mock_client):
    """Mock ClickHouse environment and client."""
    # Set required environment variables
    monkeypatch.setenv("CLICKHOUSE_ENABLED", "true")
    monkeypatch.setenv("CLICKHOUSE_HOST", "localhost")
    monkeypatch.setenv("CLICKHOUSE_PORT", "8123")
    monkeypatch.setenv("CLICKHOUSE_DATABASE", "test_db")
    monkeypatch.setenv("CLICKHOUSE_USER", "default")
    monkeypatch.setenv("CLICKHOUSE_PASSWORD", "")

    # Mock clickhouse_connect module
    import sys
    mock_clickhouse_module = MagicMock()
    sys.modules['clickhouse_connect'] = mock_clickhouse_module

    # Mock _get_client to return our mock client
    with patch("entity_data_lakehouse.clickhouse_sink._get_client", return_value=mock_client):
        yield mock_client
        # Cleanup
        if 'clickhouse_connect' in sys.modules:
            del sys.modules['clickhouse_connect']


def test_rollback_succeeds_when_no_concurrent_publish(mock_clickhouse_env):
    """Test successful rollback when no concurrent publish detected."""
    mock_client = mock_clickhouse_env

    # Setup: batch_log exists, batch_id matches, no other staging tables
    mock_client.query.side_effect = [
        # Initial batch_log existence check
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Initial batch_id check
        MagicMock(result_rows=[("batch-123",)]),
        # Initial staging table check
        MagicMock(result_rows=[]),
        # Final batch_log existence check before EXCHANGE
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Final batch_id check before EXCHANGE
        MagicMock(result_rows=[("batch-123",)]),
        # Final staging table check before EXCHANGE
        MagicMock(result_rows=[]),
        # Post-EXCHANGE batch_log existence check
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Post-EXCHANGE batch_id check
        MagicMock(result_rows=[("batch-123",)]),
        # Post-EXCHANGE staging table check
        MagicMock(result_rows=[]),
    ]

    ex_live_tables = [
        ("test_db", "ownership_current__ex_live_batch-123"),
        ("test_db", "owner_infrastructure_exposure_snapshot__ex_live_batch-123"),
    ]

    result = rollback_clickhouse(
        ex_live_tables,
        ["ownership_current", "owner_infrastructure_exposure_snapshot"],
        "batch-123",
    )

    assert result == "rolled_back"
    # Verify EXCHANGE was called for both tables
    assert mock_client.command.call_count == 5  # 2 EXCHANGE + 1 DELETE + 2 DROP
    exchange_calls = [
        c for c in mock_client.command.call_args_list if "EXCHANGE TABLES" in str(c)
    ]
    assert len(exchange_calls) == 2


def test_rollback_aborts_when_batch_id_changed_before_exchange(mock_clickhouse_env):
    """Test rollback aborts when concurrent publish detected before EXCHANGE."""
    mock_client = mock_clickhouse_env

    # Setup: batch_id changes between initial check and final verification
    mock_client.query.side_effect = [
        # Initial batch_log existence check
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Initial batch_id check - shows our batch
        MagicMock(result_rows=[("batch-123",)]),
        # Initial staging table check
        MagicMock(result_rows=[]),
        # Final batch_log existence check before EXCHANGE
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Final batch_id check before EXCHANGE - NOW SHOWS DIFFERENT BATCH
        MagicMock(result_rows=[("batch-456",)]),
    ]

    ex_live_tables = [("test_db", "ownership_current__ex_live_batch-123")]

    result = rollback_clickhouse(
        ex_live_tables,
        ["ownership_current"],
        "batch-123",
    )

    assert result == "partial_rollback_failed"
    # Verify EXCHANGE was NOT called
    exchange_calls = [
        c for c in mock_client.command.call_args_list if "EXCHANGE TABLES" in str(c)
    ]
    assert len(exchange_calls) == 0


def test_rollback_aborts_when_concurrent_publish_already_happened_at_start(
    mock_clickhouse_env
):
    """Test rollback aborts when another run already published before rollback starts.

    This is the case where:
    1. This run published expected_batch_id successfully
    2. Another run published a newer batch_id before rollback starts
    3. expected_batch_id exists in lakehouse_batch_log but is no longer current

    This must abort immediately - we cannot rollback over a newer published snapshot.
    """
    mock_client = mock_clickhouse_env

    mock_client.query.side_effect = [
        # Initial batch_log existence check
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Initial batch_id check - shows NEWER batch (not ours)
        MagicMock(result_rows=[("batch-456",)]),
        # Check if expected_batch_id exists in log - YES (we published it, but it's no longer current)
        MagicMock(result_rows=[(1,)]),  # COUNT(*) = 1, expected_batch_id exists
    ]

    ex_live_tables = [("test_db", "ownership_current__ex_live_batch-123")]

    result = rollback_clickhouse(
        ex_live_tables,
        ["ownership_current"],
        "batch-123",
    )

    assert result == "partial_rollback_failed"
    # Verify EXCHANGE was NOT called
    exchange_calls = [
        c for c in mock_client.command.call_args_list if "EXCHANGE TABLES" in str(c)
    ]
    assert len(exchange_calls) == 0


def test_rollback_aborts_when_recent_staging_tables_found_before_exchange(
    mock_clickhouse_env
):
    """Test rollback aborts when recent unknown staging tables detected before EXCHANGE."""
    mock_client = mock_clickhouse_env
    now = datetime.datetime.now()

    mock_client.query.side_effect = [
        # Initial batch_log existence check
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Initial batch_id check
        MagicMock(result_rows=[("batch-123",)]),
        # Initial staging table check - finds unknown staging table
        MagicMock(result_rows=[("ownership_current__staging_batch-456",)]),
        # Get all batch_ids to check if staging is from known batch
        MagicMock(result_rows=[("batch-123",)]),  # batch-456 NOT in list
        # Check staging table age - RECENT (< 1 hour)
        MagicMock(result_rows=[(now,)]),
    ]

    ex_live_tables = [("test_db", "ownership_current__ex_live_batch-123")]

    result = rollback_clickhouse(
        ex_live_tables,
        ["ownership_current"],
        "batch-123",
    )

    assert result == "partial_rollback_failed"
    # Verify EXCHANGE was NOT called
    exchange_calls = [
        c for c in mock_client.command.call_args_list if "EXCHANGE TABLES" in str(c)
    ]
    assert len(exchange_calls) == 0


def test_rollback_proceeds_when_staging_tables_are_stale(mock_clickhouse_env):
    """Test rollback proceeds when staging tables are old (>1 hour)."""
    mock_client = mock_clickhouse_env
    old_time = datetime.datetime.now() - datetime.timedelta(hours=2)

    mock_client.query.side_effect = [
        # Initial batch_log existence check
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Initial batch_id check
        MagicMock(result_rows=[("batch-123",)]),
        # Initial staging table check - finds unknown staging table
        MagicMock(result_rows=[("ownership_current__staging_batch-old",)]),
        # Get all batch_ids
        MagicMock(result_rows=[("batch-123",)]),
        # Check staging table age - OLD (> 1 hour)
        MagicMock(result_rows=[(old_time,)]),
        # Final batch_log existence check before EXCHANGE
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Final batch_id check before EXCHANGE
        MagicMock(result_rows=[("batch-123",)]),
        # Final staging table check before EXCHANGE
        MagicMock(result_rows=[("ownership_current__staging_batch-old",)]),
        # Get all batch_ids (final check)
        MagicMock(result_rows=[("batch-123",)]),
        # Check staging table age again (final check)
        MagicMock(result_rows=[(old_time,)]),
        # Post-EXCHANGE batch_log existence check
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Post-EXCHANGE batch_id check
        MagicMock(result_rows=[("batch-123",)]),
        # Post-EXCHANGE staging table check
        MagicMock(result_rows=[]),
    ]

    ex_live_tables = [("test_db", "ownership_current__ex_live_batch-123")]

    result = rollback_clickhouse(
        ex_live_tables,
        ["ownership_current"],
        "batch-123",
    )

    assert result == "rolled_back"
    # Verify EXCHANGE was called
    exchange_calls = [
        c for c in mock_client.command.call_args_list if "EXCHANGE TABLES" in str(c)
    ]
    assert len(exchange_calls) == 1


def test_rollback_detects_concurrent_publish_after_exchange(
    mock_clickhouse_env
):
    """Test rollback detects (but cannot prevent) concurrent publish that completes during EXCHANGE."""
    mock_client = mock_clickhouse_env
    mock_client.query.side_effect = [
        # Initial batch_log existence check
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Initial batch_id check
        MagicMock(result_rows=[("batch-123",)]),
        # Initial staging table check
        MagicMock(result_rows=[]),
        # Final batch_log existence check before EXCHANGE
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Final batch_id check before EXCHANGE - still our batch
        MagicMock(result_rows=[("batch-123",)]),
        # Final staging table check before EXCHANGE
        MagicMock(result_rows=[]),
        # Post-EXCHANGE batch_log existence check
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Post-EXCHANGE batch_id check - NOW SHOWS DIFFERENT BATCH
        MagicMock(result_rows=[("batch-456",)]),
    ]

    ex_live_tables = [("test_db", "ownership_current__ex_live_batch-123")]

    result = rollback_clickhouse(
        ex_live_tables,
        ["ownership_current"],
        "batch-123",
    )

    assert result == "partial_rollback_failed"
    # Verify EXCHANGE was called (damage already done)
    exchange_calls = [
        c for c in mock_client.command.call_args_list if "EXCHANGE TABLES" in str(c)
    ]
    assert len(exchange_calls) == 1


def test_rollback_succeeds_on_first_publish_when_batch_log_missing(
    mock_clickhouse_env
):
    """Test rollback succeeds when lakehouse_batch_log doesn't exist yet (first publish)."""
    mock_client = mock_clickhouse_env
    mock_client.query.side_effect = [
        # Initial batch_log existence check - DOESN'T EXIST
        MagicMock(result_rows=[]),
        # Initial staging table check
        MagicMock(result_rows=[]),
        # Final batch_log existence check before EXCHANGE - STILL DOESN'T EXIST
        MagicMock(result_rows=[]),
        # Final staging table check before EXCHANGE
        MagicMock(result_rows=[]),
        # Post-EXCHANGE batch_log existence check - STILL DOESN'T EXIST
        MagicMock(result_rows=[]),
        # Post-EXCHANGE staging table check
        MagicMock(result_rows=[]),
    ]

    ex_live_tables = [("test_db", "ownership_current__ex_live_batch-123")]

    result = rollback_clickhouse(
        ex_live_tables,
        ["ownership_current"],
        "batch-123",
    )

    assert result == "rolled_back"
    # Verify EXCHANGE was called
    exchange_calls = [
        c for c in mock_client.command.call_args_list if "EXCHANGE TABLES" in str(c)
    ]
    assert len(exchange_calls) == 1


def test_rollback_aborts_when_concurrent_first_publish_creates_staging_tables(
    mock_clickhouse_env
):
    """Test rollback aborts when concurrent first-time publish creates staging tables."""
    mock_client = mock_clickhouse_env
    now = datetime.datetime.now()

    mock_client.query.side_effect = [
        # Initial batch_log existence check - DOESN'T EXIST
        MagicMock(result_rows=[]),
        # Initial staging table check - no staging tables yet
        MagicMock(result_rows=[]),
        # Final batch_log existence check before EXCHANGE - STILL DOESN'T EXIST
        MagicMock(result_rows=[]),
        # Final staging table check before EXCHANGE - NOW HAS STAGING TABLES
        MagicMock(result_rows=[("ownership_current__staging_batch-456",)]),
        # Check staging table age - RECENT
        MagicMock(result_rows=[(now,)]),
    ]

    ex_live_tables = [("test_db", "ownership_current__ex_live_batch-123")]

    result = rollback_clickhouse(
        ex_live_tables,
        ["ownership_current"],
        "batch-123",
    )

    assert result == "partial_rollback_failed"
    # Verify EXCHANGE was NOT called
    exchange_calls = [
        c for c in mock_client.command.call_args_list if "EXCHANGE TABLES" in str(c)
    ]
    assert len(exchange_calls) == 0


def test_rollback_aborts_when_batch_log_check_fails(mock_clickhouse_env):
    """Test rollback aborts when batch_log existence check fails."""
    mock_client = mock_clickhouse_env
    mock_client.query.side_effect = Exception("Connection failed")

    ex_live_tables = [("test_db", "ownership_current__ex_live_batch-123")]

    result = rollback_clickhouse(
        ex_live_tables,
        ["ownership_current"],
        "batch-123",
    )

    assert result == "partial_rollback_failed"
    # Verify EXCHANGE was NOT called
    exchange_calls = [
        c for c in mock_client.command.call_args_list if "EXCHANGE TABLES" in str(c)
    ]
    assert len(exchange_calls) == 0


def test_rollback_handles_deferred_publication_mode(mock_clickhouse_env):
    """Test rollback succeeds in deferred publication mode (batch_id not published yet)."""
    mock_client = mock_clickhouse_env
    mock_client.query.side_effect = [
        # Initial batch_log existence check - EXISTS
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Initial batch_id check - shows DIFFERENT batch (not ours)
        MagicMock(result_rows=[("batch-old",)]),
        # Check if expected_batch_id exists in log - NO (deferred publication)
        MagicMock(result_rows=[(0,)]),  # COUNT(*) = 0, expected_batch_id doesn't exist yet
        # Check batch_id in ex-live table - should match initial_batch_id
        MagicMock(result_rows=[("batch-old",)]),  # ex-live data has batch-old
        # Initial staging table check
        MagicMock(result_rows=[]),
        # Final batch_log existence check before EXCHANGE
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Final batch_id check before EXCHANGE - still old batch
        MagicMock(result_rows=[("batch-old",)]),
        # Final staging table check before EXCHANGE
        MagicMock(result_rows=[]),
        # Post-EXCHANGE batch_log existence check
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Post-EXCHANGE batch_id check - still old batch
        MagicMock(result_rows=[("batch-old",)]),
        # Post-EXCHANGE staging table check
        MagicMock(result_rows=[]),
    ]

    ex_live_tables = [("test_db", "ownership_current__ex_live_batch-123")]

    result = rollback_clickhouse(
        ex_live_tables,
        ["ownership_current"],
        "batch-123",
    )

    assert result == "rolled_back"
    # Verify EXCHANGE was called
    exchange_calls = [
        c for c in mock_client.command.call_args_list if "EXCHANGE TABLES" in str(c)
    ]
    assert len(exchange_calls) == 1


def test_rollback_aborts_when_concurrent_publish_happened_before_rollback_in_deferred_mode(
    mock_clickhouse_env
):
    """Test rollback aborts when concurrent publish completed before rollback starts in deferred mode.

    This is the case where:
    1. Our run did EXCHANGE (expected_batch_id not published yet - deferred mode)
    2. Another run completed EXCHANGE + PostgreSQL + published its batch_id
    3. Now initial_batch_id is the newer batch, but expected_batch_id was never published
    4. The ex-live tables contain the old batch data, which doesn't match initial_batch_id

    This must abort - we cannot rollback over a newer published snapshot.
    """
    mock_client = mock_clickhouse_env

    mock_client.query.side_effect = [
        # Initial batch_log existence check
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Initial batch_id check - shows NEWER batch (from concurrent run)
        MagicMock(result_rows=[("batch-newer",)]),
        # Check if expected_batch_id exists in log - NO (we never published in deferred mode)
        MagicMock(result_rows=[(0,)]),  # COUNT(*) = 0
        # Check batch_id in ex-live table - shows OLD batch (what was live before our EXCHANGE)
        MagicMock(result_rows=[("batch-old",)]),  # ex-live data has batch-old, NOT batch-newer
    ]

    ex_live_tables = [("test_db", "ownership_current__ex_live_batch-123")]

    result = rollback_clickhouse(
        ex_live_tables,
        ["ownership_current"],
        "batch-123",
    )

    assert result == "partial_rollback_failed"
    # Verify EXCHANGE was NOT called
    exchange_calls = [
        c for c in mock_client.command.call_args_list if "EXCHANGE TABLES" in str(c)
    ]
    assert len(exchange_calls) == 0


def test_rollback_detects_deferred_concurrent_publish_after_exchange(mock_clickhouse_env):
    """Test rollback detects concurrent publish that finished EXCHANGE but hasn't published batch_id yet.

    This is the P1 race condition: another dual-sink run finishes EXCHANGE during our rollback window
    but hasn't published its batch_id to lakehouse_batch_log yet (deferred publication mode).
    The batch_id check would miss this, but the staging table scan should catch it.
    """
    mock_client = mock_clickhouse_env
    now = datetime.datetime.now()

    mock_client.query.side_effect = [
        # Initial batch_log existence check
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Initial batch_id check
        MagicMock(result_rows=[("batch-123",)]),
        # Initial staging table check
        MagicMock(result_rows=[]),
        # Final batch_log existence check before EXCHANGE
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Final batch_id check before EXCHANGE - still our batch
        MagicMock(result_rows=[("batch-123",)]),
        # Final staging table check before EXCHANGE
        MagicMock(result_rows=[]),
        # Post-EXCHANGE batch_log existence check
        MagicMock(result_rows=[("lakehouse_batch_log",)]),
        # Post-EXCHANGE batch_id check - STILL shows our batch (deferred publication)
        MagicMock(result_rows=[("batch-123",)]),
        # Post-EXCHANGE staging table check - NOW HAS NEW STAGING TABLES
        MagicMock(result_rows=[("ownership_current__staging_batch-456",)]),
        # Get all batch_ids to check if staging is from known batch
        MagicMock(result_rows=[("batch-123",)]),  # batch-456 NOT in list
        # Check staging table age - RECENT (< 1 hour)
        MagicMock(result_rows=[(now,)]),
    ]

    ex_live_tables = [("test_db", "ownership_current__ex_live_batch-123")]

    result = rollback_clickhouse(
        ex_live_tables,
        ["ownership_current"],
        "batch-123",
    )

    assert result == "partial_rollback_failed"
    # Verify EXCHANGE was called (damage already done)
    exchange_calls = [
        c for c in mock_client.command.call_args_list if "EXCHANGE TABLES" in str(c)
    ]
    assert len(exchange_calls) == 1
