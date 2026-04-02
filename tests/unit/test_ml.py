"""Unit tests for the ML asset lifecycle extrapolation module.

Tests verify:
  - Synthetic training data generation: shape, label distribution, determinism.
  - Feature enrichment: correct join of country and sector attributes.
  - Model training: models fit and expose required sklearn interface.
  - Predictions: valid lifecycle stages, plausible retirement year range,
    capacity factor within 1-80%, internal consistency (commissioning < retirement).
  - Determinism: identical seed produces identical results across two runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from entity_data_lakehouse.ml import (
    _LIFECYCLE_STAGES,
    _MODEL_VERSION,
    _build_sector_encoding,
    _encode_sector,
    _encode_territorial_type,
    _enrich_asset_features,
    _generate_synthetic_training_data,
    _predict_for_assets,
    _train_models,
    _load_country_attributes,
    _load_sector_lifecycle,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REFERENCE_ROOT = Path(__file__).resolve().parents[2] / "reference_data"


@pytest.fixture(scope="module")
def country_attrs():
    return _load_country_attributes(REFERENCE_ROOT)


@pytest.fixture(scope="module")
def sector_params():
    return _load_sector_lifecycle(REFERENCE_ROOT)


@pytest.fixture(scope="module")
def training_data(country_attrs, sector_params):
    return _generate_synthetic_training_data(
        country_attrs, sector_params, n_samples=300, seed=42
    )


@pytest.fixture(scope="module")
def fitted_models(training_data):
    models, label_encoder = _train_models(training_data, seed=42)
    return models, label_encoder


# ---------------------------------------------------------------------------
# Reference data tests
# ---------------------------------------------------------------------------


class TestReferenceDataLoading:
    def test_country_attributes_loads(self, country_attrs):
        assert len(country_attrs) >= 20, (
            "Expected at least 20 countries in reference data"
        )

    def test_required_countries_present(self, country_attrs):
        """All countries used in sample_data must have attributes."""
        for code in ("GB", "US", "ES", "DE", "FR"):
            assert code in country_attrs, (
                f"Country {code} missing from country_attributes.csv"
            )

    def test_country_attributes_have_required_keys(self, country_attrs):
        required = {
            "latitude_centroid",
            "longitude_centroid",
            "altitude_avg_m",
            "territorial_type",
            "economic_level",
            "gdp_tier",
            "solar_irradiance_kwh_m2_yr",
            "wind_speed_avg_ms",
            "regulatory_stability_score",
        }
        for code, attrs in country_attrs.items():
            missing = required - set(attrs.keys())
            assert not missing, f"Country {code} missing fields: {missing}"

    def test_latitudes_in_range(self, country_attrs):
        for code, attrs in country_attrs.items():
            assert -90 <= attrs["latitude_centroid"] <= 90, (
                f"Latitude out of range for {code}"
            )

    def test_sector_lifecycle_loads(self, sector_params):
        assert set(sector_params.keys()) >= {"solar", "wind", "storage"}

    def test_sector_typical_lifespans_plausible(self, sector_params):
        assert sector_params["solar"]["typical_lifespan_years"] >= 20
        assert sector_params["wind"]["typical_lifespan_years"] >= 15
        assert sector_params["storage"]["typical_lifespan_years"] >= 10


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


class TestEncoders:
    def test_build_sector_encoding_assigns_stable_ordinals(self, sector_params):
        """Encoding is alphabetically sorted and 1-indexed."""
        encoding = _build_sector_encoding(sector_params)
        # Current sectors: solar, storage, wind (alphabetical)
        assert encoding["solar"] == 1
        assert encoding["storage"] == 2
        assert encoding["wind"] == 3

    def test_build_sector_encoding_covers_all_sectors(self, sector_params):
        encoding = _build_sector_encoding(sector_params)
        assert set(encoding.keys()) == set(sector_params.keys())

    def test_encode_sector_known_values(self, sector_params):
        encoding = _build_sector_encoding(sector_params)
        assert _encode_sector("solar", encoding) == encoding["solar"]
        assert _encode_sector("wind", encoding) == encoding["wind"]
        assert _encode_sector("storage", encoding) == encoding["storage"]

    def test_encode_sector_unknown_returns_zero(self, sector_params):
        encoding = _build_sector_encoding(sector_params)
        assert _encode_sector("geothermal", encoding) == 0

    def test_territorial_type_encoding_inland_highest(self):
        assert _encode_territorial_type("inland") > _encode_territorial_type("coastal")

    def test_territorial_type_encoding_unknown_returns_midpoint(self):
        result = _encode_territorial_type("unknown_type")
        assert result in (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# Synthetic training data generation
# ---------------------------------------------------------------------------


class TestSyntheticTrainingData:
    def test_shape(self, training_data):
        assert len(training_data) == 300
        assert "lifecycle_stage" in training_data.columns
        assert "retirement_year" in training_data.columns
        assert "capacity_factor" in training_data.columns

    def test_lifecycle_stage_values_are_valid(self, training_data):
        valid_stages = set(_LIFECYCLE_STAGES)
        generated = set(training_data["lifecycle_stage"].unique())
        assert generated <= valid_stages, (
            f"Unexpected stages: {generated - valid_stages}"
        )

    def test_operating_is_dominant_stage(self, training_data):
        """Most synthetic assets should be in the 'operating' stage since
        commissioning years span 2000-2030 and typical lifespans are 15-30 yrs."""
        stage_counts = training_data["lifecycle_stage"].value_counts()
        assert stage_counts.get("operating", 0) > 100, (
            "Expected at least 100/300 assets in 'operating' stage"
        )

    def test_all_five_lifecycle_stages_present(self, training_data):
        """Every stage including 'planning' must appear in training data.

        'planning' requires commissioning_year > _REFERENCE_YEAR, which is
        only reachable when the upper bound of rng.integers exceeds 2025.
        This test would have caught the original off-by-one (2025 exclusive
        → max year 2024 → planning branch dead).
        """
        generated = set(training_data["lifecycle_stage"].unique())
        missing = set(_LIFECYCLE_STAGES) - generated
        assert not missing, (
            f"Lifecycle stages not represented in training data: {missing}. "
            "The classifier cannot predict stages it has never seen."
        )

    def test_capacity_factors_in_physical_range(self, training_data):
        assert training_data["capacity_factor"].between(0.01, 0.85).all()

    def test_retirement_year_plausible(self, training_data):
        assert training_data["retirement_year"].between(2005, 2075).all()

    def test_commissioning_year_before_retirement(self, training_data):
        assert (
            training_data["commissioning_year"] < training_data["retirement_year"]
        ).all()

    def test_determinism(self, country_attrs, sector_params):
        """Same seed must produce identical DataFrames."""
        df1 = _generate_synthetic_training_data(
            country_attrs, sector_params, n_samples=50, seed=7
        )
        df2 = _generate_synthetic_training_data(
            country_attrs, sector_params, n_samples=50, seed=7
        )
        pd.testing.assert_frame_equal(
            df1.reset_index(drop=True), df2.reset_index(drop=True)
        )

    def test_different_seeds_differ(self, country_attrs, sector_params):
        df1 = _generate_synthetic_training_data(
            country_attrs, sector_params, n_samples=50, seed=1
        )
        df2 = _generate_synthetic_training_data(
            country_attrs, sector_params, n_samples=50, seed=2
        )
        # At least some retirement years should differ.
        assert not (
            df1["retirement_year"].values == df2["retirement_year"].values
        ).all()

    def test_sector_distribution_approximately_correct(
        self, country_attrs, sector_params
    ):
        """Sampling fractions should match training_weight values in sector_lifecycle.csv.

        Alphabetical encoding: solar=1 (40%), storage=2 (20%), wind=3 (40%).
        """
        df = _generate_synthetic_training_data(
            country_attrs, sector_params, n_samples=1000, seed=99
        )
        encoding = _build_sector_encoding(sector_params)
        counts = df["sector_encoded"].value_counts(normalize=True)
        solar_pct = counts.get(encoding["solar"], 0.0)
        storage_pct = counts.get(encoding["storage"], 0.0)
        wind_pct = counts.get(encoding["wind"], 0.0)
        assert 0.30 <= solar_pct <= 0.50, (
            f"Solar fraction {solar_pct:.2%} outside 30-50%"
        )
        assert 0.30 <= wind_pct <= 0.50, f"Wind fraction {wind_pct:.2%} outside 30-50%"
        assert 0.10 <= storage_pct <= 0.30, (
            f"Storage fraction {storage_pct:.2%} outside 10-30%"
        )


# ---------------------------------------------------------------------------
# Feature enrichment
# ---------------------------------------------------------------------------


class TestFeatureEnrichment:
    def _make_asset_master(self):
        return pd.DataFrame(
            [
                {
                    "asset_id": "ast-001",
                    "asset_name": "Iberia Solar One",
                    "asset_country": "ES",
                    "asset_sector": "solar",
                    "capacity_mw": 150.0,
                    "operator_entity_id": "ent-001",
                    "source_systems": "infrastructure_assets",
                    "first_seen_snapshot": "2025-01-01",
                    "last_seen_snapshot": "2025-09-01",
                    "is_current": True,
                },
                {
                    "asset_id": "ast-002",
                    "asset_name": "Gulf Wind Hub",
                    "asset_country": "US",
                    "asset_sector": "wind",
                    "capacity_mw": 220.0,
                    "operator_entity_id": "ent-002",
                    "source_systems": "infrastructure_assets",
                    "first_seen_snapshot": "2025-01-01",
                    "last_seen_snapshot": "2025-09-01",
                    "is_current": True,
                },
                {
                    "asset_id": "ast-unknown-country",
                    "asset_name": "Mystery Plant",
                    "asset_country": "XX",  # not in reference data
                    "asset_sector": "solar",
                    "capacity_mw": 50.0,
                    "operator_entity_id": "ent-999",
                    "source_systems": "infrastructure_assets",
                    "first_seen_snapshot": "2025-01-01",
                    "last_seen_snapshot": "2025-01-01",
                    "is_current": False,
                },
            ]
        )

    def _make_ownership_lifecycle(self):
        return pd.DataFrame(
            [
                {
                    "lifecycle_key": "ent-001|ast-001",
                    "owner_entity_id": "ent-001",
                    "asset_id": "ast-001",
                    "total_appearances": 3,
                    "presence_rate": 1.0,
                    "reliability_score": 0.96,
                    "consecutive_appearances_current": 3,
                    "lifecycle_status": "ACTIVE",
                }
            ]
        )

    def test_output_has_one_row_per_asset(self, country_attrs, sector_params):
        assets = self._make_asset_master()
        lifecycle = self._make_ownership_lifecycle()
        enriched = _enrich_asset_features(
            assets, lifecycle, country_attrs, sector_params
        )
        assert len(enriched) == len(assets)

    def test_known_country_lat_lon_populated(self, country_attrs, sector_params):
        assets = self._make_asset_master()
        lifecycle = self._make_ownership_lifecycle()
        enriched = _enrich_asset_features(
            assets, lifecycle, country_attrs, sector_params
        )
        es_row = enriched[enriched["asset_country"] == "ES"].iloc[0]
        assert es_row["latitude"] == pytest.approx(
            country_attrs["ES"]["latitude_centroid"]
        )
        assert es_row["longitude"] == pytest.approx(
            country_attrs["ES"]["longitude_centroid"]
        )

    def test_unknown_country_uses_fallback(self, country_attrs, sector_params):
        assets = self._make_asset_master()
        lifecycle = self._make_ownership_lifecycle()
        enriched = _enrich_asset_features(
            assets, lifecycle, country_attrs, sector_params
        )
        unknown_row = enriched[enriched["asset_country"] == "XX"].iloc[0]
        # Should not raise; latitude should be a number.
        assert isinstance(unknown_row["latitude"], float)

    def test_lifecycle_signal_joined_for_known_asset(
        self, country_attrs, sector_params
    ):
        assets = self._make_asset_master()
        lifecycle = self._make_ownership_lifecycle()
        enriched = _enrich_asset_features(
            assets, lifecycle, country_attrs, sector_params
        )
        solar_row = enriched[enriched["asset_id"] == "ast-001"].iloc[0]
        assert solar_row["total_appearances"] == 3
        assert solar_row["presence_rate"] == pytest.approx(1.0)

    def test_missing_lifecycle_gets_neutral_defaults(
        self, country_attrs, sector_params
    ):
        assets = self._make_asset_master()
        # Provide empty lifecycle DataFrame.
        enriched = _enrich_asset_features(
            assets, pd.DataFrame(), country_attrs, sector_params
        )
        for _, row in enriched.iterrows():
            assert row["total_appearances"] >= 1
            assert 0.0 <= row["presence_rate"] <= 1.0

    def test_required_feature_columns_present(self, country_attrs, sector_params):
        from entity_data_lakehouse.ml import _FEATURE_COLS

        assets = self._make_asset_master()
        lifecycle = self._make_ownership_lifecycle()
        enriched = _enrich_asset_features(
            assets, lifecycle, country_attrs, sector_params
        )
        for col in _FEATURE_COLS:
            assert col in enriched.columns, f"Missing feature column: {col}"

    def test_unsupported_sector_raises_value_error(self, country_attrs, sector_params):
        """An asset with a sector not in sector_lifecycle.csv must raise ValueError
        rather than silently falling back to solar behaviour."""
        unsupported_asset = pd.DataFrame(
            [
                {
                    "asset_id": "ast-bad",
                    "asset_name": "Hydro Dam Alpha",
                    "asset_country": "GB",
                    "asset_sector": "hydro",  # not in reference data
                    "capacity_mw": 200.0,
                    "operator_entity_id": "ent-001",
                    "source_systems": "infrastructure_assets",
                    "first_seen_snapshot": "2025-01-01",
                    "last_seen_snapshot": "2025-01-01",
                    "is_current": True,
                }
            ]
        )
        with pytest.raises(ValueError, match="Unsupported asset sector 'hydro'"):
            _enrich_asset_features(
                unsupported_asset, pd.DataFrame(), country_attrs, sector_params
            )


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


class TestModelTraining:
    def test_returns_three_models(self, fitted_models):
        models, _ = fitted_models
        assert set(models.keys()) == {
            "lifecycle_stage_clf",
            "retirement_year_reg",
            "capacity_factor_reg",
        }

    def test_classifier_has_predict_proba(self, fitted_models):
        models, _ = fitted_models
        assert hasattr(models["lifecycle_stage_clf"], "predict_proba")

    def test_regressors_have_predict(self, fitted_models):
        models, _ = fitted_models
        for name in ("retirement_year_reg", "capacity_factor_reg"):
            assert hasattr(models[name], "predict")

    def test_label_encoder_knows_operating(self, fitted_models):
        _, le = fitted_models
        assert "operating" in le.classes_

    def test_models_deterministic(self, country_attrs, sector_params):
        data = _generate_synthetic_training_data(
            country_attrs, sector_params, n_samples=100, seed=42
        )
        models1, le1 = _train_models(data, seed=0)
        models2, le2 = _train_models(data, seed=0)
        from entity_data_lakehouse.ml import _FEATURE_COLS

        X = data[_FEATURE_COLS].values[:5]
        preds1 = models1["lifecycle_stage_clf"].predict(X)
        preds2 = models2["lifecycle_stage_clf"].predict(X)
        np.testing.assert_array_equal(preds1, preds2)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


class TestPredictions:
    def _make_enriched(self, country_attrs, sector_params):
        """A small enriched DataFrame that can be used in predict tests."""
        assets = pd.DataFrame(
            [
                {
                    "asset_id": "ast-001",
                    "asset_name": "Iberia Solar One",
                    "asset_country": "ES",
                    "asset_sector": "solar",
                    "capacity_mw": 150.0,
                },
                {
                    "asset_id": "ast-002",
                    "asset_name": "Gulf Wind Hub",
                    "asset_country": "US",
                    "asset_sector": "wind",
                    "capacity_mw": 220.0,
                },
                {
                    "asset_id": "ast-003",
                    "asset_name": "Rhine Battery West",
                    "asset_country": "DE",
                    "asset_sector": "storage",
                    "capacity_mw": 90.0,
                },
                {
                    "asset_id": "ast-004",
                    "asset_name": "Civic Water Solar Roof",
                    "asset_country": "FR",
                    "asset_sector": "solar",
                    "capacity_mw": 35.0,
                },
                {
                    "asset_id": "ast-005",
                    "asset_name": "Baltic Wind Pilot",
                    "asset_country": "GB",
                    "asset_sector": "wind",
                    "capacity_mw": 120.0,
                },
            ]
        )
        # Minimal extra columns needed by _build_asset_master
        for col in (
            "operator_entity_id",
            "source_systems",
            "first_seen_snapshot",
            "last_seen_snapshot",
            "is_current",
        ):
            assets[col] = ""
        assets["is_current"] = True
        lifecycle = pd.DataFrame()
        return _enrich_asset_features(assets, lifecycle, country_attrs, sector_params)

    def test_one_prediction_per_asset(
        self, fitted_models, country_attrs, sector_params
    ):
        models, le = fitted_models
        enriched = self._make_enriched(country_attrs, sector_params)
        preds = _predict_for_assets(enriched, models, le)
        assert len(preds) == 5

    def test_lifecycle_stages_valid(self, fitted_models, country_attrs, sector_params):
        models, le = fitted_models
        enriched = self._make_enriched(country_attrs, sector_params)
        preds = _predict_for_assets(enriched, models, le)
        valid = set(_LIFECYCLE_STAGES)
        bad = set(preds["predicted_lifecycle_stage"]) - valid
        assert not bad, f"Invalid lifecycle stages predicted: {bad}"

    def test_retirement_year_in_plausible_range(
        self, fitted_models, country_attrs, sector_params
    ):
        models, le = fitted_models
        enriched = self._make_enriched(country_attrs, sector_params)
        preds = _predict_for_assets(enriched, models, le)
        assert preds["estimated_retirement_year"].between(2025, 2080).all()

    def test_commissioning_year_before_retirement_year(
        self, fitted_models, country_attrs, sector_params
    ):
        models, le = fitted_models
        enriched = self._make_enriched(country_attrs, sector_params)
        preds = _predict_for_assets(enriched, models, le)
        assert (
            preds["estimated_commissioning_year"] < preds["estimated_retirement_year"]
        ).all()

    def test_capacity_factor_in_pct_range(
        self, fitted_models, country_attrs, sector_params
    ):
        models, le = fitted_models
        enriched = self._make_enriched(country_attrs, sector_params)
        preds = _predict_for_assets(enriched, models, le)
        assert preds["predicted_capacity_factor_pct"].between(1.0, 80.0).all()

    def test_confidence_is_probability(
        self, fitted_models, country_attrs, sector_params
    ):
        models, le = fitted_models
        enriched = self._make_enriched(country_attrs, sector_params)
        preds = _predict_for_assets(enriched, models, le)
        assert preds["lifecycle_stage_confidence"].between(0.0, 1.0).all()

    def test_remaining_years_non_negative(
        self, fitted_models, country_attrs, sector_params
    ):
        models, le = fitted_models
        enriched = self._make_enriched(country_attrs, sector_params)
        preds = _predict_for_assets(enriched, models, le)
        assert (preds["predicted_remaining_years"] >= 0.0).all()

    def test_model_version_tag_present(
        self, fitted_models, country_attrs, sector_params
    ):
        models, le = fitted_models
        enriched = self._make_enriched(country_attrs, sector_params)
        preds = _predict_for_assets(enriched, models, le)
        assert (preds["model_version"] == _MODEL_VERSION).all()

    def test_empty_input_returns_empty_dataframe(self, fitted_models):
        models, le = fitted_models
        result = _predict_for_assets(pd.DataFrame(), models, le)
        assert result.empty
