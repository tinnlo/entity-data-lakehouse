"""Reproducible eval harness: sklearn baseline vs LoRA adapter.

Can be used as an importable module or run directly as a CLI script.

Importable usage::

    from evals.run_evals import run_evals
    report = run_evals()           # writes evals/output/latest_report.json
    print(report["sklearn_accuracy"])

CLI usage::

    python3 evals/run_evals.py [--adapter DIR] [--samples N] [--test-split F]
    make eval

Output
------
``evals/output/latest_report.json`` — machine-readable JSON with quality,
runtime, and cost fields for both backends.  ``*_runtime_s`` fields measure
inference runtime only; training runtime is reported separately.

When no LoRA adapter exists the ``lora_*`` fields are ``null`` and
``lora_available`` is ``false``; the sklearn half still runs and the report
is still written.

When Langfuse credentials are configured (LANGFUSE_PUBLIC_KEY /
LANGFUSE_SECRET_KEY), aggregate accuracy scores and benchmark metadata are
logged to a Langfuse trace.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from entity_data_lakehouse.benchmark_costs import (  # noqa: E402
    build_lora_section,
    build_sklearn_section,
    build_tradeoff_summary,
    load_pricing,
)
from entity_data_lakehouse.ml import (  # noqa: E402
    _FEATURE_COLS,
    _generate_synthetic_training_data,
    _load_country_attributes,
    _load_sector_lifecycle,
    _train_models,
)
from entity_data_lakehouse.ml_lora import (  # noqa: E402
    BASE_MODEL_REVISION,
    DEFAULT_ADAPTER_REL,
    LIFECYCLE_STAGES,
    load_lora_model,
    predict_lifecycle_lora_batch,
)
from entity_data_lakehouse.observability import get_langfuse  # noqa: E402


def _load_adapter_provenance(adapter_dir: Path) -> dict[str, Any] | None:
    """Return the raw ``training_benchmark`` dict from adapter metadata, or None.

    Unlike :func:`_load_adapter_training_benchmark` this helper returns the
    dict even when ``training_runtime_s`` is absent or ``None``, so pricing
    provenance fields (e.g. ``lora_train_usd_per_hour``) are always recovered
    from the adapter that was actually trained, not from the current env.
    """
    meta_path = adapter_dir / "adapter_metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        tb = meta.get("training_benchmark")
        if isinstance(tb, dict):
            return tb
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _load_adapter_training_benchmark(adapter_dir: Path) -> dict[str, Any] | None:
    """Return the training benchmark dict from adapter metadata, or None.

    Only returns the dict when ``training_runtime_s`` is present and is a
    real number — adapters saved by ``train_lora_adapter()`` before the
    training script backfills the measured runtime will have it set to
    ``None``, which must not be passed into cost-proxy helpers.

    Use :func:`_load_adapter_provenance` when you only need pricing/rate
    fields and do not require a measured runtime.
    """
    meta_path = adapter_dir / "adapter_metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        tb = meta.get("training_benchmark")
        if isinstance(tb, dict) and isinstance(tb.get("training_runtime_s"), (int, float)):
            return tb
    except (json.JSONDecodeError, OSError):
        pass
    return None


def run_evals(
    adapter_dir: Path | None = None,
    samples: int = 300,
    test_split: float = 0.2,
    seed_train: int = 42,
    seed_test: int = 99,
    output_path: Path = _REPO_ROOT / "evals" / "output" / "latest_report.json",
) -> dict[str, Any]:
    """Run the full sklearn vs LoRA eval suite and write a JSON report.

    Parameters
    ----------
    adapter_dir:
        Path to the saved PEFT adapter directory.  Defaults to
        ``<repo>/models/lifecycle_lora_adapter``.  If the directory does not
        exist, the LoRA half is skipped and ``lora_available`` is set to
        ``false`` in the report.
    samples:
        Total synthetic sample count used for training + test generation.
    test_split:
        Fraction of ``samples`` reserved as held-out test data.
    seed_train:
        RNG seed used for *training* data generation and model fitting.
    seed_test:
        RNG seed used for *test* data generation (must differ from
        ``seed_train`` to avoid data leakage).
    output_path:
        File path where ``latest_report.json`` is written.

    Returns
    -------
    dict
        The report dictionary (same content as the written JSON file).
    """
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    if adapter_dir is None:
        adapter_dir = _REPO_ROOT / DEFAULT_ADAPTER_REL

    pricing = load_pricing()

    reference_root = _REPO_ROOT / "reference_data"
    country_attrs = _load_country_attributes(reference_root)
    sector_params = _load_sector_lifecycle(reference_root)

    train_df = _generate_synthetic_training_data(
        country_attrs=country_attrs,
        sector_params=sector_params,
        n_samples=samples,
        seed=seed_train,
    )
    test_df_full = _generate_synthetic_training_data(
        country_attrs=country_attrs,
        sector_params=sector_params,
        n_samples=samples,
        seed=seed_test,
    )
    _, test_df = train_test_split(
        test_df_full,
        test_size=test_split,
        random_state=seed_train,
        stratify=test_df_full["lifecycle_stage"],
    )
    test_df = test_df.reset_index(drop=True)

    y_true = test_df["lifecycle_stage"].tolist()
    X_test = test_df[_FEATURE_COLS].values
    n_test = len(test_df)

    # -----------------------------------------------------------------------
    # sklearn baseline — split training and inference timing
    # -----------------------------------------------------------------------
    t_train_start = time.perf_counter()
    models, le = _train_models(train_df, seed=seed_train)
    sklearn_training_runtime_s = round(time.perf_counter() - t_train_start, 4)

    t_infer_start = time.perf_counter()
    sk_pred_encoded = models["lifecycle_stage_clf"].predict(X_test)
    sk_pred_raw = le.inverse_transform(sk_pred_encoded)
    sk_pred = list(sk_pred_raw)
    _sklearn_inference_raw_s = time.perf_counter() - t_infer_start
    sklearn_inference_runtime_s = round(_sklearn_inference_raw_s, 4)

    sklearn_accuracy = float(accuracy_score(y_true, sk_pred))
    sklearn_f1 = _per_class_f1(y_true, sk_pred)

    schema_valid = _validate_sklearn_predictions(models, train_df, test_df, le)

    sklearn_section = build_sklearn_section(
        accuracy=sklearn_accuracy,
        f1_per_class=sklearn_f1,
        training_runtime_s=sklearn_training_runtime_s,
        inference_runtime_s=sklearn_inference_runtime_s,
        usd_per_hour=pricing["sklearn_usd_per_hour"],
        n_samples=n_test,
    )

    # -----------------------------------------------------------------------
    # LoRA evaluation (optional — skipped if adapter absent or all rows fail)
    # -----------------------------------------------------------------------
    adapter_present = Path(adapter_dir).exists()
    lora_available = False
    lora_inference_healthy: bool | None = None
    lora_successful_predictions: int | None = None
    lora_failed_predictions: int | None = None
    lora_accuracy: float | None = None
    lora_f1: dict[str, float] | None = None
    lora_inference_runtime_s: float | None = None
    lora_training_runtime_s: float | None = None
    lora_model_load_s: float | None = None
    _lora_inference_raw_s: float | None = None

    # Rate used for training cost — prefer stored provenance so historical
    # reports are not retroactively changed by env-var edits after training.
    lora_train_usd_per_hour = pricing["lora_train_usd_per_hour"]
    # Full pricing provenance from the adapter that was actually trained.
    # Surfaced in the report so historical reports cannot silently mix stored
    # train rates with a different rate-card that is live in the current env.
    stored_pricing_profile: str | None = None
    stored_pricing_notes: str | None = None

    if adapter_present:
        # Always load provenance so stored pricing rate is recovered even when
        # runtime backfill is missing (incomplete train run).
        adapter_provenance = _load_adapter_provenance(Path(adapter_dir))
        if adapter_provenance is not None:
            stored_rate = adapter_provenance.get("lora_train_usd_per_hour")
            if isinstance(stored_rate, (int, float)) and stored_rate > 0:
                lora_train_usd_per_hour = float(stored_rate)
            _sp = adapter_provenance.get("pricing_profile")
            if isinstance(_sp, str) and _sp:
                stored_pricing_profile = _sp
            _sn = adapter_provenance.get("notes")
            if isinstance(_sn, str) and _sn:
                stored_pricing_notes = _sn

        # Runtime-bearing benchmark dict — only valid when runtime was backfilled.
        adapter_tb = _load_adapter_training_benchmark(Path(adapter_dir))
        if adapter_tb is not None:
            lora_training_runtime_s = adapter_tb["training_runtime_s"]

        # Time model/tokenizer load separately so lora_inference_runtime_s
        # measures pure scoring and is directly comparable to sklearn timing.
        # Use the same revision resolution as predict_lifecycle_lora_batch so
        # the preload hits the same lru_cache entry and the split is accurate.
        import os as _os
        _revision = _os.environ.get("LORA_BASE_MODEL_REVISION", BASE_MODEL_REVISION)
        t_load_start = time.perf_counter()
        try:
            load_lora_model(str(Path(adapter_dir)), _revision)
        except Exception:
            pass  # load errors surface again inside predict_lifecycle_lora_batch
        lora_model_load_s = round(time.perf_counter() - t_load_start, 4)

        t_lora_start = time.perf_counter()
        batch_results = predict_lifecycle_lora_batch(
            test_df, adapter_dir=Path(adapter_dir)
        )
        _lora_inference_raw_s = time.perf_counter() - t_lora_start
        lora_inference_runtime_s = round(_lora_inference_raw_s, 4)

        n_lora_success = sum(1 for r in batch_results if r is not None)
        lora_successful_predictions = n_lora_success
        lora_failed_predictions = len(batch_results) - n_lora_success
        lora_inference_healthy = n_lora_success > 0

        if not lora_inference_healthy:
            lora_available = False
        else:
            lora_available = True
            # Scoring policy: exclude rows where LoRA inference failed so that
            # accuracy / F1 reflect only actual model predictions.
            paired = [
                (y, r[0])
                for y, r in zip(y_true, batch_results)
                if r is not None
            ]
            y_true_scored, lora_pred = zip(*paired) if paired else ([], [])
            lora_accuracy = float(accuracy_score(y_true_scored, lora_pred))
            lora_f1 = _per_class_f1(list(y_true_scored), list(lora_pred))

    lora_section = build_lora_section(
        adapter_present=adapter_present,
        inference_healthy=lora_inference_healthy,
        successful_predictions=lora_successful_predictions,
        failed_predictions=lora_failed_predictions,
        available=lora_available,
        accuracy=lora_accuracy,
        f1_per_class=lora_f1,
        training_runtime_s=lora_training_runtime_s,
        inference_runtime_s=lora_inference_runtime_s,
        training_usd_per_hour=lora_train_usd_per_hour,
        inference_usd_per_hour=pricing["lora_infer_usd_per_hour"],
        n_samples=n_test,
        amortization_samples=pricing["lora_amortization_samples"],
        model_load_s=lora_model_load_s,
        effective_train_usd_per_hour=lora_train_usd_per_hour,
    )

    # -----------------------------------------------------------------------
    # Derived comparison fields
    # -----------------------------------------------------------------------
    accuracy_delta: float | None = None
    runtime_ratio: float | None = None
    cost_ratio: float | None = None

    sk_infer_cost = sklearn_section["cost_per_sample_usd"]
    lora_infer_cost = lora_section["cost_per_sample_usd"]

    if lora_accuracy is not None:
        accuracy_delta = round(lora_accuracy - sklearn_accuracy, 4)
    # Only populate ratios when LoRA inference actually succeeded (available=True).
    # A failed LoRA run consumed compute but is not a valid backend comparison.
    if lora_available and _lora_inference_raw_s is not None and _sklearn_inference_raw_s > 0:
        runtime_ratio = round(_lora_inference_raw_s / _sklearn_inference_raw_s, 4)
    if (
        lora_available
        and lora_infer_cost is not None
        and sk_infer_cost is not None
        and sk_infer_cost > 0
    ):
        cost_ratio = round(lora_infer_cost / sk_infer_cost, 4)

    tradeoff_summary = build_tradeoff_summary(
        sklearn_accuracy=sklearn_accuracy,
        sklearn_inference_runtime_s=sklearn_inference_runtime_s,
        sklearn_cost_per_sample_usd=sk_infer_cost,
        lora_accuracy=lora_accuracy,
        lora_inference_runtime_s=lora_inference_runtime_s,
        lora_cost_per_sample_usd=lora_infer_cost,
        lora_available=lora_available,
        adapter_present=adapter_present,
        lora_inference_healthy=lora_inference_healthy,
    )

    # -----------------------------------------------------------------------
    # Assemble report (nested sklearn / lora / comparison sections)
    # -----------------------------------------------------------------------
    report: dict[str, Any] = {
        "schema_version": "2",
        "report_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "test_samples": n_test,
        "cost_estimate_method": "equivalent_cloud_rate_card",
        "pricing_profile": pricing["pricing_profile"],
        "pricing_assumptions": {
            "sklearn_usd_per_hour": pricing["sklearn_usd_per_hour"],
            "lora_train_usd_per_hour": lora_train_usd_per_hour,
            "lora_infer_usd_per_hour": pricing["lora_infer_usd_per_hour"],
            "lora_amortization_samples": pricing["lora_amortization_samples"],
            "notes": pricing["notes"],
            "stored_pricing_profile": stored_pricing_profile,
            "stored_pricing_notes": stored_pricing_notes,
        },
        "cost_proxy_unit": pricing["cost_proxy_unit"],
        "sklearn": sklearn_section,
        "lora": lora_section,
        "comparison": {
            "accuracy_delta_lora_minus_sklearn": accuracy_delta,
            "runtime_ratio_lora_to_sklearn": runtime_ratio,
            "cost_ratio_lora_to_sklearn": cost_ratio,
        },
        "quality_latency_tradeoff_summary": tradeoff_summary,
        "schema_valid": schema_valid,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    # -----------------------------------------------------------------------
    # Optional Langfuse score logging + benchmark metadata
    # -----------------------------------------------------------------------
    try:
        lf = get_langfuse()
        trace = lf.trace(name="evals_run")
        trace.score(name="sklearn_accuracy", value=sklearn_accuracy)
        if lora_accuracy is not None:
            trace.score(name="lora_accuracy", value=lora_accuracy)
        trace.update(metadata={
            "benchmark": {
                "sklearn_inference_runtime_s": sklearn_section["runtime_s"],
                "sklearn_training_runtime_s": sklearn_section["training_runtime_s"],
                "sklearn_estimated_cost_usd": sklearn_section["estimated_cost_usd"],
                "lora_inference_runtime_s": lora_section["runtime_s"],
                "lora_training_runtime_s": lora_section["training_runtime_s"],
                "lora_estimated_cost_usd": lora_section["estimated_cost_usd"],
                "pricing_profile": pricing["pricing_profile"],
            },
        })
        lf.flush()
    except Exception:
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "Langfuse telemetry failed during eval; report is unaffected.",
            exc_info=True,
        )

    return report


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _per_class_f1(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    """Return per-class F1 scores as a dict keyed by stage label."""
    from sklearn.metrics import f1_score

    scores = f1_score(y_true, y_pred, labels=LIFECYCLE_STAGES, average=None, zero_division=0)
    return {stage: round(float(s), 4) for stage, s in zip(LIFECYCLE_STAGES, scores)}


def _validate_sklearn_predictions(models, train_df, test_df, le) -> bool:
    """Check that all predicted stages are valid lifecycle stage labels."""
    X_test = test_df[_FEATURE_COLS].values
    sk_pred_encoded = models["lifecycle_stage_clf"].predict(X_test)
    sk_pred_raw = le.inverse_transform(sk_pred_encoded)
    sk_pred = list(sk_pred_raw)
    valid_set = set(LIFECYCLE_STAGES)
    return all(s in valid_set for s in sk_pred)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible sklearn vs LoRA eval and write a JSON report."
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help=(
            "Path to the saved LoRA adapter directory. "
            f"Defaults to <repo>/{DEFAULT_ADAPTER_REL}."
        ),
    )
    parser.add_argument(
        "--samples", type=int, default=300, help="Total synthetic samples (default 300)."
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Held-out test fraction (default 0.2).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "evals" / "output" / "latest_report.json",
        help="Output path for the JSON report.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    report = run_evals(
        adapter_dir=args.adapter,
        samples=args.samples,
        test_split=args.test_split,
        output_path=args.output,
    )
    print(f"Report written to {args.output}")
    print(f"  sklearn accuracy : {report['sklearn']['accuracy']:.4f}")
    if report["lora"]["available"]:
        print(f"  lora accuracy    : {report['lora']['accuracy']:.4f}")
    elif report["lora"]["adapter_present"]:
        print("  lora             : adapter present but inference failed")
    else:
        print("  lora             : not available (no adapter found)")
    print(f"  schema valid     : {report['schema_valid']}")
