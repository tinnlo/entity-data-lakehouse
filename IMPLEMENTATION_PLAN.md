# Implementation Plan: ML Benchmark FinOps Extension

## Stage 1: Benchmark Economics Contract
**Goal**: Centralize all runtime-to-proxy and runtime-to-USD logic in one small shared helper and lock the benchmark semantics before touching reports or telemetry.
**Success Criteria**: All cost calculations come from one code path; the pricing profile is explicit and reproducible; no billing math is duplicated across scripts.
**Tests**: Deterministic unit coverage using monkeypatched rates and fixed runtimes.
**Status**: Complete

## Stage 2: Eval Harness Normalization And Report Expansion
**Goal**: Make the benchmark economically fair and extend `evals/output/latest_report.json` with proxy and USD fields.
**Success Criteria**: sklearn and LoRA inference runtimes are comparable; the report contains the new fields with stable semantics; the no-adapter case still writes a valid report with `lora_available=false`.
**Tests**: Extend `tests/unit/test_evals.py` for required keys, null/non-null LoRA behavior, per-sample cost consistency, amortization fields, and round-trip JSON validity.
**Status**: Complete

## Stage 3: LoRA Training Metadata And Langfuse Telemetry
**Goal**: Persist LoRA training economics and surface runtime/cost context on the existing telemetry objects.
**Success Criteria**: Old adapters remain loadable; new adapters carry training-benchmark metadata; Langfuse shows runtime/cost context on training, eval, and LoRA batch inference; telemetry failures remain non-fatal.
**Tests**: Extend `tests/unit/test_ml_lora.py` and `tests/unit/test_ml.py` for metadata additions and telemetry enrichment.
**Status**: Complete

## Stage 4: Documentation And Repo Framing
**Goal**: Make the cost-aware ML story visible and accurate while preserving DuckDB-first and ClickHouse-optional framing.
**Success Criteria**: A reviewer can understand the benchmark economics without reading the code; docs do not over-claim exact cost; ClickHouse is not repositioned as a FinOps centerpiece.
**Tests**: Doc verification is manual, backed by regenerated report output.
**Status**: Complete

## Stage 5: Regression Verification
**Goal**: Prove the FinOps extension does not break the default local workflow or optional surfaces.
**Success Criteria**: All tests pass; the DuckDB-default path still works; the benchmark report includes runtime, proxy, and USD fields; optional ClickHouse behavior remains unaffected.
**Tests**: `pytest tests/`, `ruff check .`, `python evals/run_evals.py`, `python scripts/run_demo.py`.
**Status**: Complete
