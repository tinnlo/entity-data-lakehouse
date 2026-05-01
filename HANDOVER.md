# HANDOVER — ML / AI FinOps Extension

Repo: `entity-data-lakehouse`

## Objective

Add a **secondary AI FinOps angle** to the lakehouse repo so it complements the primary FinOps story in `evidence-enrichment-engine`.

This repo should demonstrate:

- ML runtime and cost-aware benchmarking
- sklearn vs LoRA comparison with cost or runtime proxies
- observability that ties ML quality to latency and cost signals

This is **not** the primary FinOps demo. It is the supporting demo for:

- ML workload efficiency
- batch inference and evaluation economics
- analytics-serving cost awareness

## Required shipped outcome

Minimum shipped outcome:

1. Extend the existing ML/eval story with explicit FinOps metrics.
2. Add runtime and cost-proxy fields to benchmark outputs.
3. Update docs so the lakehouse is visibly cost-aware, not only model-aware.

## In scope

- Extend sklearn vs LoRA eval outputs with:
  - runtime
  - estimated inference cost or cost proxy
  - estimated training cost or cost proxy when relevant
  - quality vs runtime comparison
- Add explicit documentation for:
  - local default path
  - optional ClickHouse serving path
  - ML observability via Langfuse
  - benchmark-backed cost-awareness
- Optional lightweight cost attribution for:
  - training runs
  - eval runs
  - batch prediction runs

## Preferred public framing

Frame this repo as:

- medallion lakehouse
- DuckDB default, ClickHouse optional
- ML extrapolation with reproducible benchmarking
- **ML workload FinOps support signal**

Good phrasing:

- “runtime- and cost-aware ML benchmarking”
- “quality / latency / cost reporting for sklearn vs LoRA”
- “batch AI workload observability”

Avoid turning this repo into:

- a cloud billing demo
- a generic FinOps platform
- a Kafka/streaming cost architecture exercise

## Implementation guidance

### 1. Extend the eval artifact

The repo already has an eval harness. Extend the existing report rather than creating a disconnected FinOps subsystem.

Suggested fields:

- `sklearn_runtime_s`
- `lora_runtime_s`
- `sklearn_estimated_cost_usd` or `sklearn_cost_proxy`
- `lora_estimated_cost_usd` or `lora_cost_proxy`
- `sklearn_cost_per_sample`
- `lora_cost_per_sample`
- `quality_latency_tradeoff_summary`

If exact USD estimates are weak for local runs, a documented cost proxy is acceptable.
Example proxies:

- compute-seconds
- GPU-seconds
- normalized training/inference units

If you use proxies instead of USD, name them clearly and document them honestly.

### 2. Langfuse integration

The repo already has Langfuse-linked ML observability. Strengthen that angle by ensuring traces or summaries make quality / runtime / cost relationships visible.

Examples:

- training trace includes duration and dataset size
- eval trace includes backend, runtime, score summary, and cost proxy
- inference trace includes batch size and per-batch runtime

### 3. ClickHouse framing

Do not make ClickHouse the FinOps centerpiece.

Correct framing:

- DuckDB remains local default and source of truth
- ClickHouse remains optional analytics serving
- FinOps value here is visibility into AI/ML workload cost and runtime, not warehouse billing

### 4. Docs

Update docs so reviewers can see:

- this repo measures ML quality
- this repo measures ML runtime
- this repo now exposes ML cost-awareness too

Likely doc surfaces:

- `README.md`
- `docs/architecture.md`
- maybe Airflow or eval docs if they mention benchmarking

## Likely files to modify

These are likely, not mandatory:

- `README.md`
- `docs/architecture.md`
- `evals/run_evals.py`
- `evals/output/latest_report.json`
- ML observability helpers under `src/entity_data_lakehouse/...`
- tests covering eval schema and benchmark fields

## Out of scope

- account-level cloud FinOps
- cloud billing ingestion
- multi-team cost allocation
- real-time stream cost controls
- turning ClickHouse into the primary analytics backend

## Verification requirements

At minimum, verify:

1. Existing default DuckDB path still works.
2. Existing eval harness still works.
3. New runtime / cost metrics appear in the report schema.
4. Optional ClickHouse path remains optional and unaffected by the FinOps extension.

Suggested commands:

```bash
pytest tests/
make eval
python evals/run_evals.py
ruff check .
```

If ClickHouse-specific docs or examples change, also verify:

```bash
USE_CLICKHOUSE=true docker compose --profile clickhouse up --build
```

## Guardrails

- Do not break the DuckDB-default path.
- Do not over-claim exact cost if the implementation is using runtime proxies.
- Keep ClickHouse as optional serving infrastructure, not the default path.
- Keep the FinOps story here secondary and ML-focused.
- Prefer extending existing benchmark artifacts over inventing unrelated new ones.

## Acceptance standard

This handover is complete only when a reviewer can look at the repo and say:

“This lakehouse demo does not just compare model quality. It also exposes the runtime and cost tradeoffs of the ML workloads in a reproducible way.”
