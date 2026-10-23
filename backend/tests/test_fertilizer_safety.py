"""
Unit tests for the fertilizer recommendation layer and the irrigation safety
layer. Both are pure functions; no client, no database, no engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "backend"))

from app.schemas_crop_health import (  # noqa: E402
    FertilizerRecommendationIn,
    ProductIn,
    SoilTestIn,
)
from app.services import fertilizer_service as fs  # noqa: E402
from app.services.safety import SafetyLayer  # noqa: E402

# --------------------------------------------------------------------------
# area and unit conversions
# --------------------------------------------------------------------------


class TestAreaAndUnits:
    def test_one_acre_in_hectares(self):
        assert fs.area_to_hectares(1.0, "acre") == pytest_approx(0.40468564224, 1e-9)

    def test_one_hectare_identity(self):
        assert fs.area_to_hectares(1.0, "hectare") == 1.0

    def test_square_metre(self):
        assert fs.area_to_hectares(10_000.0, "square_metre") == 1.0

    def test_unknown_unit_rejected(self):
        import pytest

        with pytest.raises(ValueError):
            fs.area_to_hectares(5.0, "bigha")

    def test_ppm_to_kg_per_ha_uses_furrow_slice(self):
        # 1 ppm over a 2.24-million-kg furrow slice = 2.24 kg/ha.
        assert fs.soil_to_kg_ha(10.0, "ppm") == pytest_approx(22.4, 1e-6)

    def test_missing_soil_value_stays_none_not_zero(self):
        assert fs.soil_to_kg_ha(None, "ppm") is None
        assert fs.soil_to_kg_ha(None, "kg_per_ha") is None


def pytest_approx(value: float, tolerance: float) -> object:
    """~0.404685... compared with an explicit relative tolerance."""
    import pytest

    return pytest.approx(value, abs=tolerance)


# --------------------------------------------------------------------------
# the recommendation layer
# --------------------------------------------------------------------------


class TestRecommendation:
    def _request(self, **overrides):
        base = {
            "crop": "wheat",
            "areaValue": 2.0,
            "areaUnit": "acre",
        }
        base.update(overrides)
        return FertilizerRecommendationIn(**base)

    def test_no_soil_test_never_invents_a_mass(self):
        response = fs.recommend(self._request(product=ProductIn(name="Urea", nitrogenPct=46)))
        assert response.status == "needs_soil_test"
        assert response.totalProductForFieldKg is None
        assert "soil test" in " ".join(response.missingInputs).lower()
        # Targets are still shown, honestly labelled indicative.
        assert response.nutrients[0].nutrient == "N"
        assert response.nutrients[0].targetKgPerHa == 120.0

    def test_full_computation_wheat_two_acres(self):
        response = fs.recommend(
            self._request(
                soilTest=SoilTestIn(
                    units="kg_per_ha", nitrogen=40.0, phosphorus=30.0, potassium=20.0
                ),
                product=ProductIn(name="Urea", nitrogenPct=46.0),
            )
        )
        assert response.status == "computed"
        area_ha = fs.area_to_hectares(2.0, "acre")
        defect = 120.0 - 40.0
        expected = round(defect / 0.46 * area_ha, 2)
        assert response.totalProductForFieldKg == pytest_approx(expected, 0.01)
        assert response.nutrients[0].deficitKgPerHa == pytest_approx(defect, 1e-6)
        assert response.areaHectares == pytest_approx(area_ha, 1e-4)
        assert response.derivation  # every number is provenanced
        assert response.totalProductForFieldKg is not None

    def test_supply_above_target_does_not_go_negative(self):
        response = fs.recommend(
            self._request(
                soilTest=SoilTestIn(
                    units="kg_per_ha", nitrogen=999.0, phosphorus=30.0, potassium=20.0
                ),
                product=ProductIn(name="Urea", nitrogenPct=46.0),
            )
        )
        nitrogen = response.nutrients[0]
        assert nitrogen.deficitKgPerHa == 0.0
        assert nitrogen.supplyKgPerHa == pytest_approx(999.0, 1e-6)
        # Urea only carries nitrogen and the nitrogen deficit is zero, so the
        # honest quantity is exactly zero — not a blanket filler dose.
        assert response.totalProductForFieldKg == 0.0

    def test_no_product_means_no_product_mass(self):
        response = fs.recommend(
            self._request(
                soilTest=SoilTestIn(
                    units="kg_per_ha", nitrogen=40.0, phosphorus=30.0, potassium=20.0
                )
            )
        )
        assert response.status == "missing_product"
        assert response.totalProductForFieldKg is None

    def test_unknown_crop_needs_a_targeted_source(self):
        response = fs.recommend(self._request(crop="quinoa"))
        assert "quinoa" in " ".join(response.missingInputs)
        assert response.status in ("needs_soil_test", "missing_product")

    def test_splitting_when_a_single_pass_exceeds_the_cap(self):
        response = fs.recommend(
            self._request(
                crop="potato",
                soilTest=SoilTestIn(
                    units="kg_per_ha", nitrogen=50.0, phosphorus=100.0, potassium=110.0
                ),
                product=ProductIn(name="Urea", nitrogenPct=46.0),
            )
        )
        # Potato target 180, supply 50 → deficit 130 kg/ha → 130/0.46 =
        # 282.6 kg/ha, far above the 80 kg/ha per-pass cap → splits.
        assert response.status == "computed"
        assert response.splits, "a large dose must be scheduled in splits"
        assert len(response.splits) >= 2
        total_split_kg = sum(split["kgForField"] for split in response.splits)
        assert total_split_kg == pytest_approx(
            response.totalProductForFieldKg, 0.5
        )

    def test_ppm_soil_values_surface_the_conversion_assumption(self):
        response = fs.recommend(
            self._request(
                soilTest=SoilTestIn(
                    units="ppm", nitrogen=40.0, phosphorus=30.0, potassium=20.0
                ),
                product=ProductIn(name="Urea", nitrogenPct=46.0),
            )
        )
        assert any("ppm" in assumption.lower() for assumption in response.assumptions)

    def test_rounding_is_two_decimals_on_totals(self):
        response = fs.recommend(
            self._request(
                areaValue=0.73,
                areaUnit="hectare",
                soilTest=SoilTestIn(
                    units="kg_per_ha", nitrogen=15.0, phosphorus=5.0, potassium=5.0
                ),
                product=ProductIn(name="Urea", nitrogenPct=46.0),
            )
        )
        mass = response.totalProductForFieldKg
        assert mass is not None
        assert round(mass, 2) == mass

    def test_targets_endpoint_lists_table(self):
        assert "wheat" in fs.CROP_TARGETS_KG_HA
        targets = fs.CROP_TARGETS_KG_HA["rice"]
        assert targets == (120.0, 60.0, 50.0)


# --------------------------------------------------------------------------
# the safety layer
# --------------------------------------------------------------------------


def _advisory(depth_mm: float, duration_s: int = 600) -> dict:
    return {
        "status": "irrigate",
        "headline": "Irrigate now",
        "recommendation": {
            "finalDepthMm": depth_mm,
            "durationSeconds": duration_s,
            "volumeLitres": depth_mm * 2000.0,
        },
        "sensors": {"soilMoisture": 35.0, "soilPh": 6.5},
    }


class TestSafetyLayer:
    def test_depth_is_clamped_to_the_cap(self):
        layer = SafetyLayer(max_depth_mm=45.0, max_runtime_seconds=21_600)
        advisory = layer.apply(_advisory(60.0))
        assert advisory["recommendation"]["finalDepthMm"] == 45.0
        assert any("clamped" in entry for entry in advisory["safety"]["clamps"])
        assert advisory["status"] == "irrigate"

    def test_negligible_command_is_suppressed(self):
        layer = SafetyLayer(max_depth_mm=45.0, max_runtime_seconds=21_600)
        advisory = layer.apply(_advisory(0.2))
        assert advisory["recommendation"]["finalDepthMm"] == 0.0
        assert advisory["status"] == "hold"

    def test_runtime_ceiling(self):
        layer = SafetyLayer(max_depth_mm=45.0, max_runtime_seconds=21_600)
        advisory = layer.apply(_advisory(30.0, duration_s=50_000))
        assert advisory["recommendation"]["durationSeconds"] == 21_600

    def test_impossible_sensor_blocks_the_command(self):
        layer = SafetyLayer(max_depth_mm=45.0, max_runtime_seconds=21_600)
        advisory = _advisory(25.0)
        advisory["sensors"] = {"soilMoisture": 150.0, "soilPh": 6.5}
        result = layer.apply(advisory)
        assert result["status"] == "hold"
        assert result["safety"]["sensorPlausible"] is False
        # The depth is untouched for audit: the block reason, not a silent edit.
        assert result["recommendation"]["finalDepthMm"] == 25.0

    def test_bad_ph_blocks_too(self):
        layer = SafetyLayer(max_depth_mm=45.0, max_runtime_seconds=21_600)
        advisory = _advisory(25.0)
        advisory["sensors"] = {"soilMoisture": 35.0, "soilPh": -2.0}
        result = layer.apply(advisory)
        assert result["safety"]["sensorPlausible"] is False

    def test_healthy_pass_is_annotated_not_altered(self):
        layer = SafetyLayer(max_depth_mm=45.0, max_runtime_seconds=21_600)
        original = _advisory(25.0, duration_s=600)
        result = layer.apply(dict(original))
        assert result["safety"]["applied"] is True
        assert result["safety"]["capped"] is False
        # No actuator exists in this release, and the response says so.
        assert result["safety"]["actuatorConnected"] is False
        assert result["recommendation"]["finalDepthMm"] == 25.0
