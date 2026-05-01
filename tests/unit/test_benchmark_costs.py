"""Unit tests for the shared benchmark cost model (benchmark_costs.py).

Verifies:
  - load_pricing returns repo-owned defaults when env vars are unset.
  - load_pricing reads environment overrides correctly.
  - load_pricing raises ValueError for non-numeric or non-positive env values.
  - cost_proxy returns rounded compute_seconds.
  - estimated_cost_usd converts runtime + hourly rate to USD.
  - cost_per_sample divides total by count and returns None for zero samples.
  - amortized_cost_per_sample spreads training cost and adds per-sample inference cost.
  - amortized_cost_per_sample raises ValueError for non-positive amortization_samples.
  - build_tradeoff_summary produces correct English for sklearn-only and comparison cases.
"""

from __future__ import annotations

import math
import pytest

from entity_data_lakehouse.benchmark_costs import (
    amortized_cost_per_sample,
    build_lora_section,
    build_tradeoff_summary,
    cost_per_sample,
    cost_proxy,
    estimated_cost_usd,
    load_pricing,
)


# ---------------------------------------------------------------------------
# load_pricing
# ---------------------------------------------------------------------------


def test_load_pricing_defaults(monkeypatch) -> None:
    for var in (
        "BENCHMARK_SKLEARN_USD_PER_HOUR",
        "BENCHMARK_LORA_TRAIN_USD_PER_HOUR",
        "BENCHMARK_LORA_INFER_USD_PER_HOUR",
        "BENCHMARK_LORA_AMORTIZATION_SAMPLES",
    ):
        monkeypatch.delenv(var, raising=False)

    p = load_pricing()
    assert p["pricing_profile"] == "benchmark_local_equivalent_v1"
    assert p["cost_proxy_unit"] == "compute_seconds"
    assert p["sklearn_usd_per_hour"] == 0.20
    assert p["lora_train_usd_per_hour"] == 1.00
    assert p["lora_infer_usd_per_hour"] == 1.00
    assert p["lora_amortization_samples"] == 10000
    assert "not actual billed" in p["notes"]


def test_load_pricing_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("BENCHMARK_SKLEARN_USD_PER_HOUR", "0.50")
    monkeypatch.setenv("BENCHMARK_LORA_TRAIN_USD_PER_HOUR", "2.00")
    monkeypatch.setenv("BENCHMARK_LORA_INFER_USD_PER_HOUR", "1.50")
    monkeypatch.setenv("BENCHMARK_LORA_AMORTIZATION_SAMPLES", "5000")

    p = load_pricing()
    assert p["sklearn_usd_per_hour"] == 0.50
    assert p["lora_train_usd_per_hour"] == 2.00
    assert p["lora_infer_usd_per_hour"] == 1.50
    assert p["lora_amortization_samples"] == 5000


def test_load_pricing_rejects_non_numeric_float(monkeypatch) -> None:
    monkeypatch.setenv("BENCHMARK_SKLEARN_USD_PER_HOUR", "not_a_number")
    with pytest.raises(ValueError, match="BENCHMARK_SKLEARN_USD_PER_HOUR"):
        load_pricing()


def test_load_pricing_rejects_zero_float(monkeypatch) -> None:
    monkeypatch.setenv("BENCHMARK_LORA_INFER_USD_PER_HOUR", "0")
    with pytest.raises(ValueError, match="must be > 0"):
        load_pricing()


def test_load_pricing_rejects_negative_float(monkeypatch) -> None:
    monkeypatch.setenv("BENCHMARK_LORA_TRAIN_USD_PER_HOUR", "-1.0")
    with pytest.raises(ValueError, match="must be > 0"):
        load_pricing()


def test_load_pricing_rejects_non_integer_amortization(monkeypatch) -> None:
    monkeypatch.setenv("BENCHMARK_LORA_AMORTIZATION_SAMPLES", "abc")
    with pytest.raises(ValueError, match="BENCHMARK_LORA_AMORTIZATION_SAMPLES"):
        load_pricing()


def test_load_pricing_rejects_zero_amortization(monkeypatch) -> None:
    monkeypatch.setenv("BENCHMARK_LORA_AMORTIZATION_SAMPLES", "0")
    with pytest.raises(ValueError, match="must be > 0"):
        load_pricing()


# ---------------------------------------------------------------------------
# cost_proxy
# ---------------------------------------------------------------------------


def test_cost_proxy_returns_rounded_seconds() -> None:
    assert cost_proxy(1.23456) == 1.2346
    assert cost_proxy(0.0) == 0.0


# ---------------------------------------------------------------------------
# estimated_cost_usd
# ---------------------------------------------------------------------------


def test_estimated_cost_usd_basic() -> None:
    assert estimated_cost_usd(3600.0, 1.0) == 1.0


def test_estimated_cost_usd_small_runtime() -> None:
    result = estimated_cost_usd(1.8, 0.20)
    expected = 1.8 / 3600.0 * 0.20
    assert math.isclose(result, expected, rel_tol=1e-6)


def test_estimated_cost_usd_zero_runtime() -> None:
    assert estimated_cost_usd(0.0, 1.0) == 0.0


# ---------------------------------------------------------------------------
# cost_per_sample
# ---------------------------------------------------------------------------


def test_cost_per_sample_basic() -> None:
    assert cost_per_sample(0.001, 100) == 0.00001


def test_cost_per_sample_zero_samples() -> None:
    assert cost_per_sample(0.001, 0) is None


def test_cost_per_sample_negative_samples() -> None:
    assert cost_per_sample(0.001, -5) is None


# ---------------------------------------------------------------------------
# amortized_cost_per_sample
# ---------------------------------------------------------------------------


def test_amortized_cost_per_sample_basic() -> None:
    training_cost = 1.0
    infer_per_sample = 0.001
    n = 10000
    result = amortized_cost_per_sample(training_cost, infer_per_sample, n)
    expected = 1.0 / 10000 + 0.001
    assert math.isclose(result, expected, rel_tol=1e-6)


def test_amortized_cost_per_sample_none_inference() -> None:
    result = amortized_cost_per_sample(1.0, None, 10000)
    assert math.isclose(result, 0.0001, rel_tol=1e-6)


def test_amortized_cost_per_sample_zero_amort() -> None:
    with pytest.raises(ValueError, match="amortization_samples must be > 0"):
        amortized_cost_per_sample(1.0, 0.0, 0)


# ---------------------------------------------------------------------------
# build_tradeoff_summary
# ---------------------------------------------------------------------------


def test_summary_sklearn_only() -> None:
    s = build_tradeoff_summary(
        sklearn_accuracy=0.82,
        sklearn_inference_runtime_s=1.5,
        sklearn_cost_per_sample_usd=0.000005,
        lora_accuracy=None,
        lora_inference_runtime_s=None,
        lora_cost_per_sample_usd=None,
        lora_available=False,
        adapter_present=False,
    )
    assert "sklearn only" in s
    assert "0.82" in s
    assert "No LoRA adapter present" in s


def test_summary_adapter_present_inference_failed() -> None:
    s = build_tradeoff_summary(
        sklearn_accuracy=0.82,
        sklearn_inference_runtime_s=1.5,
        sklearn_cost_per_sample_usd=0.000005,
        lora_accuracy=None,
        lora_inference_runtime_s=None,
        lora_cost_per_sample_usd=None,
        lora_available=False,
        adapter_present=True,
        lora_inference_healthy=False,
    )
    assert "sklearn only" in s
    assert "adapter present but inference failed" in s


def test_summary_lora_improved() -> None:
    s = build_tradeoff_summary(
        sklearn_accuracy=0.80,
        sklearn_inference_runtime_s=0.5,
        sklearn_cost_per_sample_usd=0.00001,
        lora_accuracy=0.85,
        lora_inference_runtime_s=5.0,
        lora_cost_per_sample_usd=0.001,
        lora_available=True,
        adapter_present=True,
    )
    assert "improved" in s
    assert "0.0500" in s
    assert "10.0x" in s
    assert "100.0x" in s


def test_summary_lora_decreased() -> None:
    s = build_tradeoff_summary(
        sklearn_accuracy=0.90,
        sklearn_inference_runtime_s=0.5,
        sklearn_cost_per_sample_usd=0.00001,
        lora_accuracy=0.85,
        lora_inference_runtime_s=5.0,
        lora_cost_per_sample_usd=0.001,
        lora_available=True,
        adapter_present=True,
    )
    assert "decreased" in s


def test_summary_no_cost_fields() -> None:
    s = build_tradeoff_summary(
        sklearn_accuracy=0.80,
        sklearn_inference_runtime_s=1.0,
        sklearn_cost_per_sample_usd=None,
        lora_accuracy=0.85,
        lora_inference_runtime_s=2.0,
        lora_cost_per_sample_usd=None,
        lora_available=True,
        adapter_present=True,
    )
    assert "improved" in s
    assert "cost" not in s


# ---------------------------------------------------------------------------
# build_lora_section — model_load_s and effective_train_usd_per_hour
# ---------------------------------------------------------------------------

def _base_lora_kwargs() -> dict:
    """Return minimal valid kwargs for build_lora_section."""
    return dict(
        adapter_present=True,
        inference_healthy=True,
        successful_predictions=10,
        failed_predictions=0,
        available=True,
        accuracy=0.85,
        f1_per_class={"growth": 0.85},
        training_runtime_s=60.0,
        inference_runtime_s=1.0,
        training_usd_per_hour=1.0,
        inference_usd_per_hour=1.0,
        n_samples=10,
        amortization_samples=1000,
    )


def test_lora_section_model_load_s_present() -> None:
    """model_load_s is included in lora section when provided."""
    sec = build_lora_section(**_base_lora_kwargs(), model_load_s=2.5)
    assert sec["model_load_s"] == 2.5


def test_lora_section_model_load_s_absent() -> None:
    """model_load_s is None in lora section when not provided."""
    sec = build_lora_section(**_base_lora_kwargs())
    assert sec["model_load_s"] is None


def test_lora_section_model_load_s_does_not_affect_runtime_s() -> None:
    """runtime_s reflects only inference time; model_load_s is separate."""
    sec = build_lora_section(**_base_lora_kwargs(), model_load_s=99.0)
    assert sec["runtime_s"] == 1.0
    assert sec["model_load_s"] == 99.0


def test_lora_section_effective_train_usd_per_hour_stored() -> None:
    """effective_train_usd_per_hour records the rate actually used (stored provenance)."""
    sec = build_lora_section(**_base_lora_kwargs(), effective_train_usd_per_hour=0.75)
    assert sec["effective_train_usd_per_hour"] == 0.75


def test_lora_section_effective_train_usd_per_hour_absent() -> None:
    """effective_train_usd_per_hour is None when not supplied."""
    sec = build_lora_section(**_base_lora_kwargs())
    assert sec["effective_train_usd_per_hour"] is None


def test_lora_section_training_cost_uses_training_usd_per_hour() -> None:
    """Training estimated cost is derived from training_usd_per_hour, not inference rate."""
    sec = build_lora_section(**_base_lora_kwargs())
    # 60s training at $1/hr = $1/3600 * 60 ≈ 0.01667
    assert sec["training_estimated_cost_usd"] == pytest.approx(60 / 3600, rel=1e-4)
