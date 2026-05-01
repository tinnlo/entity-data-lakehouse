"""Shared benchmark cost model for ML workload economics.

Provides a single code path for converting measured runtime into cost proxies
and equivalent-cloud USD estimates.  All benchmark-facing code (eval harness,
training scripts, Langfuse telemetry) should call into this module instead of
duplicating cost math.

Pricing is loaded from environment variables with repo-owned defaults so
results are reproducible across machines.  Every USD value is an *estimate*
derived from a declared rate card, not actual billed cost.

Environment overrides
---------------------
BENCHMARK_SKLEARN_USD_PER_HOUR
    Equivalent hourly rate for sklearn/CPU workloads.  Default: 0.20.
BENCHMARK_LORA_TRAIN_USD_PER_HOUR
    Equivalent hourly rate for LoRA training (GPU-equivalent).  Default: 1.00.
BENCHMARK_LORA_INFER_USD_PER_HOUR
    Equivalent hourly rate for LoRA inference (GPU-equivalent).  Default: 1.00.
BENCHMARK_LORA_AMORTIZATION_SAMPLES
    Number of served samples over which to amortise LoRA training cost.  Default: 10000.
"""

from __future__ import annotations

import os
from typing import Any


_PRICING_PROFILE = "benchmark_local_equivalent_v1"
_COST_PROXY_UNIT = "compute_seconds"


def _parse_positive_float(env_var: str, default: str) -> float:
    """Parse *env_var* as a positive float, raising ``ValueError`` on bad input."""
    raw = os.environ.get(env_var, default)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {env_var}={raw!r} is not a valid float."
        )
    if value <= 0:
        raise ValueError(
            f"Environment variable {env_var}={raw!r} must be > 0 (got {value})."
        )
    return value


def _parse_positive_int(env_var: str, default: str) -> int:
    """Parse *env_var* as a positive integer, raising ``ValueError`` on bad input."""
    raw = os.environ.get(env_var, default)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"Environment variable {env_var}={raw!r} is not a valid integer."
        )
    if value <= 0:
        raise ValueError(
            f"Environment variable {env_var}={raw!r} must be > 0 (got {value})."
        )
    return value


def load_pricing() -> dict[str, Any]:
    """Return the current pricing profile as a plain dict.

    Values are read from environment variables with repo-owned fallbacks so
    every call site sees the same assumptions.

    Raises
    ------
    ValueError
        If any pricing env variable is set to a non-numeric or non-positive value.
    """
    return {
        "pricing_profile": _PRICING_PROFILE,
        "cost_proxy_unit": _COST_PROXY_UNIT,
        "sklearn_usd_per_hour": _parse_positive_float(
            "BENCHMARK_SKLEARN_USD_PER_HOUR", "0.20"
        ),
        "lora_train_usd_per_hour": _parse_positive_float(
            "BENCHMARK_LORA_TRAIN_USD_PER_HOUR", "1.00"
        ),
        "lora_infer_usd_per_hour": _parse_positive_float(
            "BENCHMARK_LORA_INFER_USD_PER_HOUR", "1.00"
        ),
        "lora_amortization_samples": _parse_positive_int(
            "BENCHMARK_LORA_AMORTIZATION_SAMPLES", "10000"
        ),
        "notes": (
            "Equivalent benchmark rates for local runs; "
            "not actual billed cloud cost."
        ),
    }


def cost_proxy(runtime_s: float) -> float:
    """Return ``compute_seconds`` proxy for a given wall-clock runtime."""
    return round(runtime_s, 4)


def estimated_cost_usd(runtime_s: float, usd_per_hour: float) -> float:
    """Convert runtime to an equivalent USD estimate.

    Formula: ``runtime_s / 3600 * usd_per_hour``.
    """
    return round(runtime_s / 3600.0 * usd_per_hour, 8)


def cost_per_sample(total_cost_usd: float, n_samples: int) -> float | None:
    """Return per-sample cost, or ``None`` when *n_samples* is zero."""
    if n_samples <= 0:
        return None
    return round(total_cost_usd / n_samples, 10)


def amortized_cost_per_sample(
    training_cost_usd: float,
    inference_cost_per_sample_usd: float | None,
    amortization_samples: int,
) -> float:
    """Return amortised per-sample cost including training spread.

    Formula::

        (training_cost_usd / amortization_samples) + inference_cost_per_sample_usd

    Raises
    ------
    ValueError
        If *amortization_samples* is not positive.
    """
    if amortization_samples <= 0:
        raise ValueError(
            f"amortization_samples must be > 0, got {amortization_samples}."
        )
    train_share = training_cost_usd / amortization_samples
    infer_share = inference_cost_per_sample_usd if inference_cost_per_sample_usd is not None else 0.0
    return round(train_share + infer_share, 10)


def build_sklearn_section(
    *,
    accuracy: float,
    f1_per_class: dict[str, float],
    training_runtime_s: float,
    inference_runtime_s: float,
    usd_per_hour: float,
    n_samples: int,
) -> dict[str, Any]:
    """Return the ``sklearn`` section of a benchmark report.

    Centralises all sklearn field derivation so the eval harness cannot
    produce a state where runtime is present but cost is absent.
    """
    training_cost_usd = estimated_cost_usd(training_runtime_s, usd_per_hour)
    inference_cost_usd = estimated_cost_usd(inference_runtime_s, usd_per_hour)
    return {
        "accuracy": round(accuracy, 4),
        "f1_per_class": f1_per_class,
        "training_runtime_s": training_runtime_s,
        "runtime_s": inference_runtime_s,
        "training_cost_proxy": cost_proxy(training_runtime_s),
        "cost_proxy": cost_proxy(inference_runtime_s),
        "training_estimated_cost_usd": training_cost_usd,
        "estimated_cost_usd": inference_cost_usd,
        "cost_per_sample_usd": cost_per_sample(inference_cost_usd, n_samples),
    }


def build_lora_section(
    *,
    adapter_present: bool,
    inference_healthy: bool | None,
    successful_predictions: int | None,
    failed_predictions: int | None,
    available: bool,
    accuracy: float | None,
    f1_per_class: dict[str, float] | None,
    training_runtime_s: float | None,
    inference_runtime_s: float | None,
    training_usd_per_hour: float,
    inference_usd_per_hour: float,
    n_samples: int,
    amortization_samples: int,
    partial_failure_scoring_policy: str = "exclude_failed",
    model_load_s: float | None = None,
    effective_train_usd_per_hour: float | None = None,
) -> dict[str, Any]:
    """Return the ``lora`` section of a benchmark report.

    Cost fields are always derived from measured runtime when available so
    failed-inference runs still report consumed compute.  Scoring fields
    (accuracy, f1) remain None when inference was unhealthy.

    ``model_load_s`` records adapter/tokenizer load time separately from pure
    inference so ``runtime_s`` is comparable to sklearn inference timing.

    ``effective_train_usd_per_hour`` records the rate actually used for
    training cost derivation (may differ from current env when stored adapter
    provenance is used).
    """
    # Training costs — only when training runtime was measured.
    training_cost_usd: float | None = None
    training_cost_proxy_val: float | None = None
    if training_runtime_s is not None:
        training_cost_usd = estimated_cost_usd(training_runtime_s, training_usd_per_hour)
        training_cost_proxy_val = cost_proxy(training_runtime_s)

    # Inference costs — always derived from runtime when measured.
    inference_cost_usd: float | None = None
    inference_cost_proxy_val: float | None = None
    inference_cost_per_sample: float | None = None
    if inference_runtime_s is not None:
        inference_cost_usd = estimated_cost_usd(inference_runtime_s, inference_usd_per_hour)
        inference_cost_proxy_val = cost_proxy(inference_runtime_s)
        inference_cost_per_sample = cost_per_sample(inference_cost_usd, n_samples)

    # Amortised cost — only when both training and inference costs are known.
    amortized: float | None = None
    if training_cost_usd is not None and inference_cost_per_sample is not None:
        amortized = amortized_cost_per_sample(
            training_cost_usd, inference_cost_per_sample, amortization_samples
        )

    return {
        "adapter_present": adapter_present,
        "inference_healthy": inference_healthy,
        "successful_predictions": successful_predictions,
        "failed_predictions": failed_predictions,
        "partial_failure_scoring_policy": partial_failure_scoring_policy,
        "available": available,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "f1_per_class": f1_per_class,
        "training_runtime_s": training_runtime_s,
        "model_load_s": round(model_load_s, 4) if model_load_s is not None else None,
        "runtime_s": inference_runtime_s,
        "training_cost_proxy": training_cost_proxy_val,
        "cost_proxy": inference_cost_proxy_val,
        "training_estimated_cost_usd": training_cost_usd,
        "estimated_cost_usd": inference_cost_usd,
        "cost_per_sample_usd": inference_cost_per_sample,
        "amortized_cost_per_sample_usd": amortized,
        "training_amortization_samples": amortization_samples,
        "effective_train_usd_per_hour": effective_train_usd_per_hour,
    }


def build_tradeoff_summary(
    *,
    sklearn_accuracy: float,
    sklearn_inference_runtime_s: float,
    sklearn_cost_per_sample_usd: float | None,
    lora_accuracy: float | None,
    lora_inference_runtime_s: float | None,
    lora_cost_per_sample_usd: float | None,
    lora_available: bool,
    adapter_present: bool = False,
    lora_inference_healthy: bool | None = None,
) -> str:
    """Return a plain-English quality/latency/cost tradeoff summary.

    Designed to be machine-written into the benchmark report so reviewers can
    understand the economics at a glance without parsing numeric fields.
    """
    if not adapter_present:
        parts = [
            f"sklearn accuracy {sklearn_accuracy:.4f}",
            f"inference {sklearn_inference_runtime_s:.4f}s",
        ]
        if sklearn_cost_per_sample_usd is not None:
            parts.append(f"${sklearn_cost_per_sample_usd:.6f}/sample")
        return "sklearn only — " + ", ".join(parts) + ". No LoRA adapter present."

    if not lora_available or lora_accuracy is None:
        parts = [
            f"sklearn accuracy {sklearn_accuracy:.4f}",
            f"inference {sklearn_inference_runtime_s:.4f}s",
        ]
        if sklearn_cost_per_sample_usd is not None:
            parts.append(f"${sklearn_cost_per_sample_usd:.6f}/sample")
        healthy_note = "inference failed" if lora_inference_healthy is False else "unknown"
        return "sklearn only — " + ", ".join(parts) + f". LoRA adapter present but {healthy_note}."

    delta = lora_accuracy - sklearn_accuracy
    direction = "improved" if delta >= 0 else "decreased"
    abs_delta = abs(delta)

    parts = [
        f"LoRA {direction} accuracy by {abs_delta:.4f}",
    ]

    if lora_inference_runtime_s is not None and sklearn_inference_runtime_s > 0:
        ratio = lora_inference_runtime_s / sklearn_inference_runtime_s
        parts.append(f"runtime {ratio:.1f}x sklearn")

    if (
        lora_cost_per_sample_usd is not None
        and sklearn_cost_per_sample_usd is not None
        and sklearn_cost_per_sample_usd > 0
    ):
        cost_ratio = lora_cost_per_sample_usd / sklearn_cost_per_sample_usd
        parts.append(f"cost {cost_ratio:.1f}x sklearn/sample")

    return "; ".join(parts) + "."
