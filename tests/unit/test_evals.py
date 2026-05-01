"""Unit tests for the eval harness (evals/run_evals.py).

All ML functions are mocked so these tests run without any GPU or heavy deps.
Verifies:
  - run_evals() returns a dict with all required keys including cost fields.
  - The JSON report is written to the expected path.
  - lora_available=false and null lora fields when no adapter exists.
  - lora_available=true and non-null lora fields when adapter dir exists.
  - report is valid JSON parseable back to the same dict.
  - schema_valid reflects whether all predicted labels are in LIFECYCLE_STAGES.
  - sklearn cost fields are populated and internally consistent.
  - lora cost fields are null when no adapter exists.
  - quality_latency_tradeoff_summary is present and non-empty.
  - per-sample cost consistency: total cost / n_samples equals cost_per_sample.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from entity_data_lakehouse.ml_lora import LIFECYCLE_STAGES  # noqa: E402

# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------

_N_SAMPLES = 50
_TEST_SPLIT = 0.2
_N_TEST = int(_N_SAMPLES * _TEST_SPLIT)


def _make_synthetic_df(n: int, seed: int = 42) -> pd.DataFrame:
    """Minimal synthetic DataFrame that satisfies run_evals internals."""
    from entity_data_lakehouse.ml import _FEATURE_COLS

    import numpy as np

    rng = np.random.default_rng(seed)
    stages = (LIFECYCLE_STAGES * (n // len(LIFECYCLE_STAGES) + 1))[:n]
    data = {col: rng.random(n) for col in _FEATURE_COLS}
    data["lifecycle_stage"] = stages
    return pd.DataFrame(data)


def _make_stub_models() -> dict:
    """Return stub sklearn model dict that predicts 'operating' for every row."""
    import numpy as np

    clf = MagicMock()
    clf.predict.return_value = np.zeros(_N_TEST, dtype=int)
    reg1 = MagicMock()
    reg1.predict.return_value = np.full(_N_TEST, 2040.0)
    reg2 = MagicMock()
    reg2.predict.return_value = np.full(_N_TEST, 0.35)
    return {
        "lifecycle_stage_clf": clf,
        "retirement_year_reg": reg1,
        "capacity_factor_reg": reg2,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _patched_ml(monkeypatch):
    """Patch all heavy ML functions in evals.run_evals namespace."""
    import evals.run_evals as run_evals_mod

    synthetic_train = _make_synthetic_df(_N_SAMPLES, seed=42)
    synthetic_test = _make_synthetic_df(_N_SAMPLES, seed=99)

    monkeypatch.setattr(
        run_evals_mod,
        "_generate_synthetic_training_data",
        lambda **kw: synthetic_train if kw.get("seed") == 42 else synthetic_test,
    )

    stub_models = _make_stub_models()
    le_mock = MagicMock()
    le_mock.inverse_transform.return_value = ["operating"] * _N_TEST

    monkeypatch.setattr(
        run_evals_mod, "_train_models", lambda df, seed=42: (stub_models, le_mock)
    )

    monkeypatch.setattr(
        run_evals_mod,
        "_load_country_attributes",
        lambda root: {},
    )
    monkeypatch.setattr(
        run_evals_mod,
        "_load_sector_lifecycle",
        lambda root: {},
    )

    return {"models": stub_models, "le": le_mock}


# ---------------------------------------------------------------------------
# Required report keys
# ---------------------------------------------------------------------------

_BASE_REQUIRED_KEYS = {
    "schema_version",
    "report_timestamp",
    "test_samples",
    "cost_estimate_method",
    "pricing_profile",
    "pricing_assumptions",
    "cost_proxy_unit",
    "sklearn",
    "lora",
    "comparison",
    "quality_latency_tradeoff_summary",
    "schema_valid",
}

_SKLEARN_SECTION_KEYS = {
    "accuracy",
    "f1_per_class",
    "training_runtime_s",
    "runtime_s",
    "training_cost_proxy",
    "cost_proxy",
    "training_estimated_cost_usd",
    "estimated_cost_usd",
    "cost_per_sample_usd",
}

_LORA_SECTION_KEYS = {
    "adapter_present",
    "inference_healthy",
    "successful_predictions",
    "failed_predictions",
    "partial_failure_scoring_policy",
    "available",
    "accuracy",
    "f1_per_class",
    "training_runtime_s",
    "model_load_s",
    "runtime_s",
    "training_cost_proxy",
    "cost_proxy",
    "training_estimated_cost_usd",
    "estimated_cost_usd",
    "cost_per_sample_usd",
    "amortized_cost_per_sample_usd",
    "training_amortization_samples",
    "effective_train_usd_per_hour",
}

_COMPARISON_SECTION_KEYS = {
    "accuracy_delta_lora_minus_sklearn",
    "runtime_ratio_lora_to_sklearn",
    "cost_ratio_lora_to_sklearn",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_report_has_all_required_keys(tmp_path, _patched_ml) -> None:
    """run_evals() must return a dict with every required key in all sections."""
    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=None,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    missing = _BASE_REQUIRED_KEYS - set(report.keys())
    assert not missing, f"Report missing top-level keys: {missing}"

    missing_sk = _SKLEARN_SECTION_KEYS - set(report["sklearn"].keys())
    assert not missing_sk, f"sklearn section missing keys: {missing_sk}"

    missing_lora = _LORA_SECTION_KEYS - set(report["lora"].keys())
    assert not missing_lora, f"lora section missing keys: {missing_lora}"

    missing_cmp = _COMPARISON_SECTION_KEYS - set(report["comparison"].keys())
    assert not missing_cmp, f"comparison section missing keys: {missing_cmp}"


def test_report_written_to_output_path(tmp_path, _patched_ml) -> None:
    """run_evals() must write valid JSON to the specified output_path."""
    from evals.run_evals import run_evals

    out = tmp_path / "sub" / "latest_report.json"
    report = run_evals(
        adapter_dir=None,
        samples=_N_SAMPLES,
        output_path=out,
    )

    assert out.exists(), "Report file was not created."
    parsed = json.loads(out.read_text())
    assert parsed == report, "Written JSON does not match returned dict."


def test_lora_fields_null_when_no_adapter(tmp_path, _patched_ml) -> None:
    """When adapter_dir does not exist, lora fields must be null / false."""
    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=tmp_path / "nonexistent_adapter",
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    assert report["lora"]["adapter_present"] is False
    assert report["lora"]["inference_healthy"] is None
    assert report["lora"]["successful_predictions"] is None
    assert report["lora"]["available"] is False
    assert report["lora"]["accuracy"] is None
    assert report["lora"]["f1_per_class"] is None
    assert report["lora"]["runtime_s"] is None
    assert report["lora"]["training_runtime_s"] is None
    assert report["lora"]["training_cost_proxy"] is None
    assert report["lora"]["cost_proxy"] is None
    assert report["lora"]["training_estimated_cost_usd"] is None
    assert report["lora"]["estimated_cost_usd"] is None
    assert report["lora"]["cost_per_sample_usd"] is None
    assert report["lora"]["amortized_cost_per_sample_usd"] is None
    assert report["comparison"]["accuracy_delta_lora_minus_sklearn"] is None
    assert report["comparison"]["runtime_ratio_lora_to_sklearn"] is None
    assert report["comparison"]["cost_ratio_lora_to_sklearn"] is None


def test_lora_fields_populated_when_adapter_exists(tmp_path, monkeypatch, _patched_ml) -> None:
    """When an adapter dir exists and predict_lifecycle_lora_batch is stubbed,
    lora_* fields must be non-null and lora_available must be True."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    import evals.run_evals as run_evals_mod

    monkeypatch.setattr(
        run_evals_mod,
        "predict_lifecycle_lora_batch",
        lambda df, adapter_dir: [("operating", 0.8)] * len(df),
    )

    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=adapter_dir,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    assert report["lora"]["adapter_present"] is True
    assert report["lora"]["inference_healthy"] is True
    assert report["lora"]["successful_predictions"] == report["test_samples"]
    assert report["lora"]["available"] is True
    assert report["lora"]["accuracy"] is not None
    assert isinstance(report["lora"]["accuracy"], float)
    assert report["lora"]["f1_per_class"] is not None
    assert report["lora"]["runtime_s"] is not None
    assert report["lora"]["cost_proxy"] is not None
    assert report["lora"]["estimated_cost_usd"] is not None
    assert report["lora"]["cost_per_sample_usd"] is not None


def test_sklearn_accuracy_in_unit_interval(tmp_path, _patched_ml) -> None:
    """sklearn_accuracy must be a float in [0, 1]."""
    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=None,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    assert 0.0 <= report["sklearn"]["accuracy"] <= 1.0


def test_f1_per_class_has_all_stages(tmp_path, _patched_ml) -> None:
    """sklearn_f1_per_class must contain a key for every LIFECYCLE_STAGE."""
    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=None,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    f1 = report["sklearn"]["f1_per_class"]
    for stage in LIFECYCLE_STAGES:
        assert stage in f1, f"Stage {stage!r} missing from sklearn_f1_per_class"
        assert 0.0 <= f1[stage] <= 1.0


def test_report_is_valid_json_roundtrip(tmp_path, _patched_ml) -> None:
    """The written file must be parseable JSON whose values round-trip cleanly."""
    from evals.run_evals import run_evals

    out = tmp_path / "report.json"
    run_evals(adapter_dir=None, samples=_N_SAMPLES, output_path=out)

    parsed = json.loads(out.read_text())
    assert "report_timestamp" in parsed
    assert "cost_estimate_method" in parsed


def test_sklearn_cost_fields_populated(tmp_path, _patched_ml) -> None:
    """sklearn cost fields must be non-null floats when sklearn runs."""
    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=None,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    assert isinstance(report["sklearn"]["training_runtime_s"], float)
    assert report["sklearn"]["training_runtime_s"] >= 0
    assert isinstance(report["sklearn"]["runtime_s"], float)
    assert report["sklearn"]["runtime_s"] >= 0
    assert isinstance(report["sklearn"]["training_cost_proxy"], float)
    assert isinstance(report["sklearn"]["cost_proxy"], float)
    assert isinstance(report["sklearn"]["training_estimated_cost_usd"], float)
    assert isinstance(report["sklearn"]["estimated_cost_usd"], float)
    assert isinstance(report["sklearn"]["cost_per_sample_usd"], float)
    assert report["sklearn"]["cost_per_sample_usd"] >= 0


def test_cost_per_sample_consistency(tmp_path, _patched_ml) -> None:
    """sklearn_cost_per_sample_usd should equal inference cost / test_samples."""
    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=None,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    expected_per = report["sklearn"]["estimated_cost_usd"] / report["test_samples"]
    assert math.isclose(
        report["sklearn"]["cost_per_sample_usd"], expected_per, rel_tol=1e-6
    )


def test_tradeoff_summary_present(tmp_path, _patched_ml) -> None:
    """quality_latency_tradeoff_summary must be a non-empty string."""
    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=None,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    assert isinstance(report["quality_latency_tradeoff_summary"], str)
    assert len(report["quality_latency_tradeoff_summary"]) > 0


def test_pricing_assumptions_structure(tmp_path, _patched_ml) -> None:
    """pricing_assumptions must contain all rate-card keys."""
    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=None,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    pa = report["pricing_assumptions"]
    for key in (
        "sklearn_usd_per_hour",
        "lora_train_usd_per_hour",
        "lora_infer_usd_per_hour",
        "lora_amortization_samples",
        "notes",
    ):
        assert key in pa, f"Missing pricing assumption key: {key}"


def test_lora_all_rows_failed_sets_available_false(tmp_path, monkeypatch, _patched_ml) -> None:
    """When all LoRA batch results are None, lora_available must be False
    but adapter_present must be True and lora_inference_healthy False."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    import evals.run_evals as run_evals_mod

    monkeypatch.setattr(
        run_evals_mod,
        "predict_lifecycle_lora_batch",
        lambda df, adapter_dir: [None] * len(df),
    )

    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=adapter_dir,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    assert report["lora"]["adapter_present"] is True
    assert report["lora"]["inference_healthy"] is False
    assert report["lora"]["successful_predictions"] == 0
    assert report["lora"]["available"] is False
    assert report["lora"]["accuracy"] is None
    # Cost fields must still be populated — failed inference consumed compute.
    assert report["lora"]["cost_proxy"] is not None
    assert report["lora"]["estimated_cost_usd"] is not None
    assert report["lora"]["cost_per_sample_usd"] is not None


def test_amortized_cost_does_not_double_count_training(tmp_path, monkeypatch, _patched_ml) -> None:
    """lora_amortized_cost_per_sample_usd must be (train/amort) + infer/sample,
    not include training cost twice."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    import json as _json
    (adapter_dir / "adapter_metadata.json").write_text(_json.dumps({
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "abc123",
        "training_benchmark": {
            "training_runtime_s": 3600.0,
            "dataset_rows": 200,
            "epochs": 1,
        },
    }))

    import evals.run_evals as run_evals_mod

    monkeypatch.setattr(
        run_evals_mod,
        "predict_lifecycle_lora_batch",
        lambda df, adapter_dir: [("operating", 0.8)] * len(df),
    )

    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=adapter_dir,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    assert report["lora"]["amortized_cost_per_sample_usd"] is not None
    train_cost = report["lora"]["training_estimated_cost_usd"]
    infer_per_sample = report["lora"]["cost_per_sample_usd"]
    amort_samples = report["lora"]["training_amortization_samples"]
    n_test = report["test_samples"]
    expected_infer_per_sample = report["lora"]["estimated_cost_usd"] / n_test
    assert math.isclose(
        infer_per_sample, expected_infer_per_sample, rel_tol=1e-6
    ), f"LoRA per-sample cost {infer_per_sample} != expected {expected_infer_per_sample}"
    expected = train_cost / amort_samples + infer_per_sample
    assert math.isclose(
        report["lora"]["amortized_cost_per_sample_usd"], expected, rel_tol=1e-6
    ), (
        f"Amortized cost {report['lora']['amortized_cost_per_sample_usd']} != "
        f"expected {expected} (train={train_cost}, infer/sample={infer_per_sample})"
    )


def test_telemetry_failure_does_not_break_report(tmp_path, monkeypatch, _patched_ml) -> None:
    """If Langfuse calls raise, the report must still be written and valid."""
    from unittest.mock import MagicMock

    import evals.run_evals as run_evals_mod

    bad_lf = MagicMock()
    bad_lf.trace.side_effect = RuntimeError("Langfuse down")
    monkeypatch.setattr(run_evals_mod, "get_langfuse", lambda: bad_lf)

    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=None,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    assert (tmp_path / "report.json").exists()
    assert "sklearn" in report


def test_adapter_with_none_training_runtime_no_crash(tmp_path, monkeypatch, _patched_ml) -> None:
    """Adapter metadata with training_runtime_s=None must not crash cost helpers."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    import json as _json
    (adapter_dir / "adapter_metadata.json").write_text(_json.dumps({
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "abc123",
        "training_benchmark": {
            "training_runtime_s": None,
            "dataset_rows": 200,
            "epochs": 1,
        },
    }))

    import evals.run_evals as run_evals_mod

    monkeypatch.setattr(
        run_evals_mod,
        "predict_lifecycle_lora_batch",
        lambda df, adapter_dir: [("operating", 0.8)] * len(df),
    )

    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=adapter_dir,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    assert report["lora"]["available"] is True
    assert report["lora"]["accuracy"] is not None
    assert report["lora"]["training_runtime_s"] is None
    assert report["lora"]["training_estimated_cost_usd"] is None
    assert report["lora"]["amortized_cost_per_sample_usd"] is None
    assert report["lora"]["cost_per_sample_usd"] is not None


def test_partial_lora_failure_cost_uses_n_test_denominator(tmp_path, monkeypatch, _patched_ml) -> None:
    """LoRA cost_per_sample_usd must use n_test denominator, not n_success."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    import evals.run_evals as run_evals_mod

    n_rows = _N_TEST
    half_results = [("operating", 0.8)] * (n_rows // 2) + [None] * (n_rows - n_rows // 2)
    monkeypatch.setattr(
        run_evals_mod,
        "predict_lifecycle_lora_batch",
        lambda df, adapter_dir: half_results,
    )

    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=adapter_dir,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    assert report["lora"]["adapter_present"] is True
    assert report["lora"]["inference_healthy"] is True
    assert report["lora"]["successful_predictions"] == n_rows // 2
    assert report["lora"]["available"] is True
    n_test = report["test_samples"]
    expected_per = report["lora"]["estimated_cost_usd"] / n_test
    assert math.isclose(
        report["lora"]["cost_per_sample_usd"], expected_per, rel_tol=1e-6
    ), (
        f"LoRA cost_per_sample {report['lora']['cost_per_sample_usd']} != "
        f"expected {expected_per} (used n_test={n_test}, not n_success={n_rows // 2})"
    )


def test_preload_uses_same_revision_as_inference(tmp_path, monkeypatch, _patched_ml) -> None:
    """Preload in run_evals must pass the same revision as predict_lifecycle_lora_batch.

    When LORA_BASE_MODEL_REVISION is unset both calls must fall back to
    BASE_MODEL_REVISION (the pinned SHA), not to the string "main".  We verify
    this by capturing the revision argument passed to load_lora_model and
    confirming it matches BASE_MODEL_REVISION.
    """
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    import evals.run_evals as run_evals_mod
    from entity_data_lakehouse.ml_lora import BASE_MODEL_REVISION

    captured_revisions: list[str] = []

    def _fake_load_lora_model(adapter_dir_str: str, revision: str) -> None:
        captured_revisions.append(revision)

    monkeypatch.delenv("LORA_BASE_MODEL_REVISION", raising=False)
    monkeypatch.setattr(run_evals_mod, "load_lora_model", _fake_load_lora_model)
    monkeypatch.setattr(
        run_evals_mod,
        "predict_lifecycle_lora_batch",
        lambda df, adapter_dir: [("operating", 0.8)] * len(df),
    )

    from evals.run_evals import run_evals

    run_evals(
        adapter_dir=adapter_dir,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    assert len(captured_revisions) == 1, (
        f"Expected exactly one preload call, got {len(captured_revisions)}"
    )
    assert captured_revisions[0] == BASE_MODEL_REVISION, (
        f"Preload used revision {captured_revisions[0]!r}, "
        f"expected BASE_MODEL_REVISION={BASE_MODEL_REVISION!r}"
    )


def test_comparison_fields_null_when_all_lora_rows_fail(tmp_path, monkeypatch, _patched_ml) -> None:
    """When all LoRA rows fail lora.available is False and comparison ratios must be null.

    A fully-failed LoRA run consumed compute but is not a valid backend
    comparison; downstream consumers must not treat it as one.
    """
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    import evals.run_evals as run_evals_mod

    monkeypatch.setattr(
        run_evals_mod,
        "load_lora_model",
        lambda adapter_dir_str, revision: None,
    )
    monkeypatch.setattr(
        run_evals_mod,
        "predict_lifecycle_lora_batch",
        lambda df, adapter_dir: [None] * len(df),
    )

    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=adapter_dir,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    assert report["lora"]["available"] is False
    assert report["lora"]["inference_healthy"] is False
    assert report["comparison"]["runtime_ratio_lora_to_sklearn"] is None, (
        "runtime_ratio must be null when LoRA is unavailable"
    )
    assert report["comparison"]["cost_ratio_lora_to_sklearn"] is None, (
        "cost_ratio must be null when LoRA is unavailable"
    )
    # accuracy_delta should also be null — no valid predictions to compare
    assert report["comparison"]["accuracy_delta_lora_minus_sklearn"] is None


def test_stored_training_rate_recovered_without_runtime(tmp_path, monkeypatch, _patched_ml) -> None:
    """Stored lora_train_usd_per_hour must be used even when training_runtime_s is missing.

    Adapters saved before the training-script runtime backfill will have
    training_runtime_s=None in their metadata.  The stored pricing rate must
    still be recovered from adapter_metadata.json so historical reports are
    not silently inflated/deflated by later env-var changes.
    """
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    # Write metadata with a custom rate but no measured runtime.
    stored_rate = 0.42
    meta = {
        "base_model": "test-model",
        "revision": "abc123",
        "training_benchmark": {
            "training_runtime_s": None,      # not yet backfilled
            "lora_train_usd_per_hour": stored_rate,
            "pricing_profile": "test_profile",
        },
    }
    (adapter_dir / "adapter_metadata.json").write_text(
        __import__("json").dumps(meta)
    )

    import evals.run_evals as run_evals_mod

    monkeypatch.setattr(
        run_evals_mod,
        "load_lora_model",
        lambda adapter_dir_str, revision: None,
    )
    monkeypatch.setattr(
        run_evals_mod,
        "predict_lifecycle_lora_batch",
        lambda df, adapter_dir: [("operating", 0.8)] * len(df),
    )

    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=adapter_dir,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    # Effective rate in report must match stored adapter provenance, not env default.
    assert report["pricing_assumptions"]["lora_train_usd_per_hour"] == stored_rate, (
        f"Expected stored rate {stored_rate}, "
        f"got {report['pricing_assumptions']['lora_train_usd_per_hour']}"
    )
    assert report["lora"]["effective_train_usd_per_hour"] == stored_rate

    # training_runtime_s must be null (not backfilled yet) — no cost invented.
    assert report["lora"]["training_runtime_s"] is None
    assert report["lora"]["training_estimated_cost_usd"] is None


def test_report_has_schema_version(tmp_path, _patched_ml) -> None:
    """Every report must carry a schema_version field."""
    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=None,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    assert "schema_version" in report
    assert report["schema_version"] == "2"


# ---------------------------------------------------------------------------
# Stored pricing provenance surfaced in pricing_assumptions
# ---------------------------------------------------------------------------


def test_stored_pricing_provenance_surfaced_in_report(tmp_path, monkeypatch, _patched_ml) -> None:
    """pricing_assumptions must include stored_pricing_profile and stored_pricing_notes
    when the adapter metadata contains them, so historical reports cannot silently mix
    a stored train rate with a different rate-card's metadata.
    """
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    meta = {
        "base_model": "test-model",
        "revision": "abc123",
        "training_benchmark": {
            "training_runtime_s": 42.0,
            "lora_train_usd_per_hour": 1.5,
            "pricing_profile": "gpu_spot_v2",
            "notes": "GPU spot rate, eu-west-1, 2025-Q1",
        },
    }
    (adapter_dir / "adapter_metadata.json").write_text(
        __import__("json").dumps(meta)
    )

    import evals.run_evals as run_evals_mod

    monkeypatch.setattr(run_evals_mod, "load_lora_model", lambda *a, **kw: None)
    monkeypatch.setattr(
        run_evals_mod,
        "predict_lifecycle_lora_batch",
        lambda df, adapter_dir: [("operating", 0.9)] * len(df),
    )

    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=adapter_dir,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    pa = report["pricing_assumptions"]
    assert "stored_pricing_profile" in pa, "stored_pricing_profile must be in pricing_assumptions"
    assert "stored_pricing_notes" in pa, "stored_pricing_notes must be in pricing_assumptions"
    assert pa["stored_pricing_profile"] == "gpu_spot_v2"
    assert pa["stored_pricing_notes"] == "GPU spot rate, eu-west-1, 2025-Q1"


def test_stored_pricing_provenance_null_when_no_adapter(tmp_path, _patched_ml) -> None:
    """When no adapter is present, stored_pricing_profile and stored_pricing_notes
    must be None — not absent from the report.
    """
    from evals.run_evals import run_evals

    report = run_evals(
        adapter_dir=None,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report.json",
    )

    pa = report["pricing_assumptions"]
    assert "stored_pricing_profile" in pa
    assert pa["stored_pricing_profile"] is None
    assert "stored_pricing_notes" in pa
    assert pa["stored_pricing_notes"] is None


# ---------------------------------------------------------------------------
# Repeated run_evals() — model_load_s cache semantics
# ---------------------------------------------------------------------------


def test_repeated_run_evals_model_load_s_is_non_negative(tmp_path, monkeypatch, _patched_ml) -> None:
    """model_load_s must be a non-negative float on every run_evals() call.

    On the first call the lru_cache is cold; on subsequent calls the model is
    already loaded (cache hit).  In both cases model_load_s must be present,
    numeric, and >= 0 — the cache hit produces near-zero but never negative.

    This test documents that the field is stable across repeated calls in the
    same process rather than becoming None or raising after the first call.
    """
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    meta = {
        "base_model": "test-model",
        "revision": "abc123",
        "training_benchmark": {
            "training_runtime_s": 10.0,
            "lora_train_usd_per_hour": 1.0,
            "pricing_profile": "benchmark_local_equivalent_v1",
            "notes": "test",
        },
    }
    (adapter_dir / "adapter_metadata.json").write_text(
        __import__("json").dumps(meta)
    )

    import evals.run_evals as run_evals_mod

    load_call_count = {"n": 0}

    def _stub_load(adapter_dir_str, revision):
        load_call_count["n"] += 1
        return None

    monkeypatch.setattr(run_evals_mod, "load_lora_model", _stub_load)
    monkeypatch.setattr(
        run_evals_mod,
        "predict_lifecycle_lora_batch",
        lambda df, adapter_dir: [("operating", 0.9)] * len(df),
    )

    from evals.run_evals import run_evals

    report1 = run_evals(
        adapter_dir=adapter_dir,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report1.json",
    )
    report2 = run_evals(
        adapter_dir=adapter_dir,
        samples=_N_SAMPLES,
        output_path=tmp_path / "report2.json",
    )

    for i, report in enumerate((report1, report2), start=1):
        mls = report["lora"]["model_load_s"]
        assert mls is not None, f"Call {i}: model_load_s must not be None when adapter is present"
        assert isinstance(mls, (int, float)), f"Call {i}: model_load_s must be numeric, got {type(mls)}"
        assert mls >= 0, f"Call {i}: model_load_s must be >= 0, got {mls}"
