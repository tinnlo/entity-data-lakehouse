# HANDOVER: Optional PostgreSQL Serving Layer

**Project:** `entity-data-lakehouse`  
**Priority:** OPTIONAL (Track C)  
**Purpose:** Track C for Deriv interview prep - only pursue if Track A is fully shipped, tested, and feels routine  
**Estimated time:** 8-10 hours  
**Target completion:** Wednesday, May 14, 2026 (only if Track A completes by Sunday)

---

## Strategic Context: Track C Is Deferred by Default

**The default decision is to skip this implementation.**

The stronger interview signal is:
- one clean Redis implementation in `evidence-enrichment-engine`
- disciplined meeting-prep updates
- rehearsal and follow-up-question practice

**Only pursue this implementation if ALL of these conditions are true:**

1. Track A (Redis in `evidence-enrichment-engine`) is fully complete by Sunday
2. Track A is tested and demoable in under 2 minutes
3. Track A feels routine, not fragile
4. You have energy for 8-10 more hours of implementation

**If any condition is false, skip this and use Tue-Wed for rehearsal instead.**

One clean Redis implementation plus confident delivery beats two implementations where one feels rushed.

---

## Why The Earlier Priority-Queue Direction Was Dropped

The earlier idea was to add:
- Redis priority queues
- worker coordination
- PostgreSQL pooling
- staleness-driven task routing

That scope is now considered **too broad and too weakly aligned** to this repo's identity for the available time.

Problems with the earlier direction:
- too much implementation surface for a pre-interview week
- harder to defend cleanly under technical probing
- turns the repo into a second orchestration project
- lower interview ROI than rehearsing one strong shipped Redis implementation

---

## Scope: What To Build (If Track C Is Reopened)

Add an **optional PostgreSQL serving/write path** for selected lakehouse outputs:

1. **Keep DuckDB authoritative**
   - DuckDB remains the source of truth for analytics
   - PostgreSQL is an optional serving layer, not a replacement

2. **Select narrow output contract**
   - Choose 1-2 gold-layer outputs to publish to PostgreSQL
   - Keep the contract small and explainable

3. **Connection pooling**
   - Use `psycopg2.pool.ThreadedConnectionPool` for repeated writes
   - Show practical understanding of connection reuse

4. **Idempotent refresh semantics**
   - Reruns should behave predictably
   - Use staging-table swap or upsert pattern

5. **Freshness/audit metadata**
   - Track when each output was last refreshed
   - Make data age visible

---

## Why This Scope Is Intentionally Narrow

This task is **not** trying to:
- Rewrite the lakehouse architecture
- Add a Redis queue or worker coordination system
- Build a second orchestration layer
- Replace DuckDB with PostgreSQL

The interview value comes from being able to explain:
- why analytics (DuckDB) and serving (PostgreSQL) are separate concerns
- how connection pooling reduces operational overhead
- how idempotent refresh semantics keep operational data publishing safe
- how freshness metadata makes data age visible

If a feature does not strengthen one of those answers directly, it is out of scope.

---

## Current Repo Identity To Preserve

This repo already demonstrates:
- medallion pipeline structure (bronze/silver/gold)
- DuckDB-centered analytical outputs
- optional ClickHouse OLAP sink
- ML pipeline integration and eval reporting
- rollback-safe publication patterns
- Airflow orchestration

Any future work here must strengthen that storage and publication story rather than compete with it.

---

## Explicit Non-Goals

Do **not** implement:
- Redis priority queues
- BLPOP worker coordinators
- broad concurrency infrastructure
- a second "agent runtime" story
- a large rewrite of the lakehouse architecture
- write-through or write-back persistence
- complex CDC or streaming patterns

This is intentionally **one optional serving layer**, not a platform expansion.

---

## Current Repo Touchpoints

These are the most likely places to inspect before editing:
- `docker-compose.yml`
- `README.md`
- `src/entity_data_lakehouse/clickhouse_sink.py` - mirror this pattern
- `src/entity_data_lakehouse/gold.py` - gold-layer outputs
- `src/entity_data_lakehouse/pipeline.py` - main pipeline
- `src/entity_data_lakehouse/contracts.py` - data contracts
- `tests/` - existing test coverage
- `dags/` - Airflow DAG definitions (if integration needed)

The existing ClickHouse sink and publication patterns are the most likely places to mirror style and semantics.

---

## Implementation Shape

### Docker / config

- Add an optional PostgreSQL service to `docker-compose.yml`
- Add a small config surface for PostgreSQL connection settings
- Keep PostgreSQL optional for the base demo path

### Code

- Add or update a serving module responsible for:
  - connection pool management
  - schema creation/migration
  - idempotent write logic
  - freshness metadata tracking

- Select 1-2 outputs to publish:
  - Gold-layer outputs: entity_master_event_log, ownership_current, owner_infrastructure_exposure_snapshot
  - ML outputs (post-ML stage): asset_lifecycle_predictions
  - Choose narrow, explainable outputs with clear schemas

- Keep the core lakehouse pipeline unchanged unless integration requires minimal changes

### Database design expectations

- Use **connection pooling**, not per-query connections
- Keep schema simple and explainable
- Include `last_refreshed_at` or similar freshness metadata
- **PostgreSQL write errors must fail the pipeline run** (mirrors ClickHouse sink behavior)
- Failed sink status must be reported in `publish_report.json` with rollback metadata
- Make refresh semantics idempotent (staging-table swap or upsert)
- Support `publish_mode=dry_run` for schema validation without connecting
- Wire into Airflow `PUBLISH_MODE` orchestration path

### Suggested shape

- One small serving module for PostgreSQL access (mirror ClickHouse sink pattern)
- One place for schema definitions
- Integration point after ML stage (mirrors ClickHouse sink placement in pipeline.py)
- Handles both gold outputs and ML predictions in single atomic batch
- Freshness metadata visible in the serving tables

---

## Step-by-Step Delivery Plan

### Step 1: Inspect current gold-layer outputs and ClickHouse sink

Goal:
- Identify 1-2 narrow outputs suitable for PostgreSQL serving
- Understand existing publication patterns to mirror

Expected outcome:
- Clear edit list for code, tests, and README

### Step 2: Add PostgreSQL service and config surface

Goal:
- Make PostgreSQL runnable locally
- Keep it optional

Expected outcome:
- Local PostgreSQL path exists
- Base lakehouse flow still works without PostgreSQL enabled

### Step 3: Implement connection pooling

Goal:
- Show practical connection reuse pattern

Expected outcome:
- Connection pool is configured and used for writes
- Pool exhaustion is handled gracefully

### Step 4: Implement idempotent write path

Goal:
- Reruns should not create duplicate data or fail

Expected outcome:
- Staging-table swap or upsert pattern works
- Refresh semantics are explainable

### Step 5: Add freshness metadata

Goal:
- Make data age visible in serving tables

Expected outcome:
- `last_refreshed_at` or similar metadata is tracked
- Stale data is identifiable

### Step 6: Add tests

Minimum coverage:
- PostgreSQL-enabled write path
- Idempotent refresh behavior
- No-PostgreSQL fallback path
- Connection pool reuse
- **Dry-run mode validation** (schema validation without connecting)
- **Publish report integration** (sink status, rollback metadata)
- **Failed sink behavior** (write errors fail the run, rollback triggered)

### Step 7: Wire into publish_report.json and dry-run mode

Goal:
- Ensure PostgreSQL sink follows repo's fail-closed publication contract

Expected outcome:
- PostgreSQL sink status appears in `publish_report.json`
- Failed writes surface as terminal run failures with rollback metadata
- `publish_mode=dry_run` validates PostgreSQL schema without connecting
- Airflow `PUBLISH_MODE` orchestration includes PostgreSQL path

### Step 8: Update README

README changes should explain:
- What PostgreSQL adds (serving layer)
- What it does not add (not replacing DuckDB)
- How to run it locally
- Why this separation matters

---

## Success Criteria

- [ ] PostgreSQL path is optional, not a rewrite of the repo
- [ ] DuckDB remains authoritative
- [ ] Connection pooling is configured and reused
- [ ] Reruns are idempotent and explainable
- [ ] **PostgreSQL write failures fail the pipeline run** (no silent degradation)
- [ ] **Sink status appears in `publish_report.json`** with rollback metadata on failure
- [ ] **Dry-run mode validates PostgreSQL schema** without connecting
- [ ] **Airflow orchestration includes PostgreSQL** via `PUBLISH_MODE` environment variable
- [ ] Freshness metadata is visible in serving tables
- [ ] Base lakehouse flow still works when PostgreSQL is disabled
- [ ] README explains what PostgreSQL adds and what it does not add
- [ ] Can explain why this belongs in `entity-data-lakehouse` in under 3 minutes
- [ ] Can explain why analytics (DuckDB) and serving (PostgreSQL) are separate concerns

---

## Interview Translation

After this implementation (if completed), you can answer:

**Q: "Walk me through your database design for an agentic system at scale"**

> "In production systems, we separate state/audit (PostgreSQL), retrieval (vector search), graph relationships, and raw documents (object storage). In my public lakehouse demo, I extended that thinking by adding an optional PostgreSQL serving layer distinct from the DuckDB analytics layer. DuckDB is authoritative for analytics queries, PostgreSQL is for operational serving where connection pooling and idempotent refresh semantics matter. This week I implemented connection pooling, staging-table swap for idempotent writes, and explicit freshness metadata. For trading systems, I'd extend this to: Redis for hot data caching, PostgreSQL for state/audit, vector DB for retrieval, and time-series DB for market data - each layer serving a distinct purpose."

**Q: "Why separate analytics and serving databases?"**

> "Analytics queries (aggregations, historical analysis) have different access patterns than serving queries (point lookups, recent state). DuckDB is optimized for OLAP workloads - fast aggregations over large datasets. PostgreSQL is optimized for OLTP workloads - fast point lookups with ACID guarantees. In the lakehouse demo, DuckDB handles the medallion pipeline and analytics, PostgreSQL handles operational serving where connection pooling and low-latency point queries matter. Mixing these concerns in one database forces tradeoffs that hurt both use cases."

**Q: "How do you handle connection pooling?"**

> "In the lakehouse demo, I use `psycopg2.pool.ThreadedConnectionPool` with a configured max connection limit. Workers reuse connections from the pool rather than opening new connections per write. This reduces connection overhead and prevents connection exhaustion under load. The pool handles connection lifecycle - checkout, reuse, and return - transparently. For production scale, I'd extend this with connection pool monitoring and circuit breaker patterns to handle pool exhaustion gracefully."

---

## What This Proves in the Interview

Before this week, the repo already proved:
- medallion lakehouse architecture
- ML pipeline integration
- lifecycle and freshness thinking
- optional OLAP sink (ClickHouse)

After this week (if Track C is pursued), the repo should also prove:
- separation of analytics and serving concerns
- connection pooling for operational writes
- idempotent refresh semantics
- freshness metadata in serving context
