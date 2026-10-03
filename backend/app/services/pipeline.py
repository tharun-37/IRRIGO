"""
The inference pipeline: one field in, both learned models out, one fused advisory.

The project ships two learned components and, before this module, they never met.

V2's `NirModel` answers *how much water the field needs over the coming
fortnight* — a residual-on-physics estimate of `nir_horizon_mm` with a
split-conformal interval on its error. It is trained by `scripts/train_model.py`
and was, until now, only ever read back as an evaluation report; nothing loaded
it at serve time.

V1's `Controller` bundle answers *should we irrigate this field, and at what
risk* — the operator-facing decision, with a calibrated probability and depth,
disease risk, yield and nutrient demand. `CompatService` already drives it from
V2's FAO-56 plan.

This module is the join. For each field it runs, in order:

1. the engine's plan for the day (stage, GDD, Kc, ET0/ETc, TAW/RAW, depletion,
   the schedule) — the physical state, from `CompatService`;
2. the V1 controller's decision on that state — the action and the risk;
3. the V2 requirement estimator on the same state, assembled into exactly the
   `FeatureSpec` the model was trained on, with its conformal interval;
4. a fusion that reports whether the two agree, escalates when the decision
   model misses a deficit the requirement model sees, and says so when they
   disagree rather than silently picking one.

Nothing here recomputes the physics. The features fed to `NirModel` are the
engine's own state variables plus the station's observed atmosphere and the
probe values `CompatService` already derives, mapped to the corpus definitions
so that train and serve see the same columns.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from irrigation.data.crops import CROPS
from irrigation.data.soil import texture as lookup_texture
from irrigation.features import STAGE_ORDER, FeatureSpec, encode_categoricals
from irrigation.models import load_model

from .compat import PUMP_RATE_LPM, CompatService, FieldSnapshot

logger = logging.getLogger("iic.pipeline")

#: Conformal coverage reported for the requirement interval. Matches the level
#: `train_model.py` calibrates and `EvaluationService` surfaces.
COVERAGE = 0.90

#: A requirement at or below this is treated as "no irrigation needed". One
#: millimetre is the same trigger threshold the corpus and the evaluation use, so
#: the model's own reported precision/recall apply unchanged.
TRIGGER_MM = 1.0

#: Soil-temperature and soil-chemistry offsets from `corpus.simulate_field`. The
#: corpus adds per-series noise that is irreducible at serve time; the
#: deterministic part is reproduced here so the model is not fed a different
#: quantity than it trained on.
SOIL_TEMP_OFFSET_C = 2.5
EC_BASE_DS_M = 0.35
EC_PER_10MM_DEPLETION = 0.02

#: Human labels for how the two models line up.
AGREEMENT_LABELS: dict[str, str] = {
    "aligned": "Both models agree",
    "agree_hold": "Both models agree on hold",
    "v1_conservative": "Decision model more conservative",
    "v2_only": "Requirement model flags unmet demand",
    "not_applicable": "Out of season",
    "model_unavailable": "Model offline",
}


def _f(value: Any, default: float = 0.0) -> float:
    """A finite float, or the default. Guards None/NaN from the engine."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


class AdvisoryPipeline:
    """
    Runs both learned models over the engine's state and fuses the result.

    Constructed once at startup and held on `app.state.pipeline`. If the V2 model
    artifact is missing or unreadable the pipeline degrades to the V1 decision and
    the FAO-56 plan, and *says so* in every advisory rather than pretending the
    requirement estimate exists.
    """

    def __init__(self, compat: CompatService, evaluation: Any, settings: Any) -> None:
        self.compat = compat
        self.evaluation = evaluation
        self.settings = settings
        self.spec = FeatureSpec()
        self._model = None
        self._error: str | None = None
        self._weather: dict[str, dict[date, dict[str, Any]]] = {}
        self._load_model()

    # --- lifecycle -------------------------------------------------------

    def _load_model(self) -> None:
        path = Path(self.settings.models_dir) / "nir_model.joblib"
        if not path.exists():
            self._error = f"{path} is missing; run scripts/train_model.py"
            logger.warning(
                "V2 requirement model absent at %s; advisor serves V1 + physics only",
                path,
            )
            return
        try:
            self._model = load_model(path)
            logger.info(
                "V2 requirement model loaded: %d features, target %s",
                len(self._model.feature_names),
                self._model.target_name,
            )
        except Exception as error:  # pragma: no cover - defensive
            self._error = str(error)
            logger.exception("could not load the V2 requirement model")

    @property
    def ready(self) -> bool:
        return self._model is not None

    def describe(self) -> dict[str, Any]:
        conformal = (
            self._model.conformal.get(COVERAGE) if self._model is not None else None
        )
        return {
            "modelPresent": self._model is not None,
            "target": "nir_horizon_mm" if self._model is not None else None,
            "coverage": COVERAGE if self._model is not None else None,
            "features": len(self._model.feature_names) if self._model is not None else 0,
            "intervalWidthMm": (
                round(float(conformal.quantile_mm), 3) if conformal is not None else None
            ),
            "error": self._error,
        }

    # --- feature assembly ------------------------------------------------

    def _station_rows(self, station: str) -> dict[date, dict[str, Any]]:
        """
        The station's observed atmosphere, keyed by day.

        Built once per station from the same frame the engine plans against, so
        the model sees the realised weather for the day rather than a re-derived
        estimate. This mirrors `CompatService._ensure_env`, which reads only four
        of these columns for the V1 reading; the estimator needs the full set.
        """
        table = self._weather.get(station)
        if table is not None:
            return table
        table = {}
        try:
            frame = self.compat.plans.context.stations[station].frame
        except Exception:
            self._weather[station] = table
            return table
        for record in frame.to_dict("records"):
            raw = record.get("date")
            key = raw.date() if hasattr(raw, "date") else raw
            if key is not None:
                table[key] = record
        self._weather[station] = table
        return table

    def _feature_frame(self, snap: FieldSnapshot) -> pd.DataFrame:
        """
        One row in the exact shape `FeatureSpec` and the trained model require.

        Every value is either a state variable the engine already computed, an
        observed atmosphere value for the day, or a probe value derived by the
        same rule the corpus used. `soil_ph` is the crop's optimum midpoint and
        `soil_temperature_c` follows the corpus offset; both are the stated
        approximations `CompatService` documents for V1 and are reused here so the
        two models are not fed contradictory soil chemistry.
        """
        event = snap.event
        point = snap.current
        plan = point.plan
        day = point.day
        weather = self._station_rows(event.station).get(day, {})

        soil = lookup_texture(event.soil_type)
        crop = CROPS.get(event.crop)
        ph_low, ph_high = crop.optimal_ph if crop is not None else (6.0, 7.5)

        taw = _f(getattr(plan, "taw_mm", 0.0))
        raw = _f(getattr(plan, "raw_mm", 0.0))
        depletion = _f(getattr(plan, "depletion_mm", 0.0))
        root_depth_cm = _f(getattr(plan, "root_depth_cm", 0.0))
        root_depth_m = root_depth_cm / 100.0
        theta_fc = float(soil.theta_field_capacity)
        theta = float(
            np.clip(
                theta_fc - depletion / max(1e-6, 1000.0 * root_depth_m), 0.0, theta_fc
            )
        )
        air_temp = _f(weather.get("air_temperature_c"), 25.0)

        data: dict[str, Any] = {
            # -- measured atmosphere
            "et0_mm_day": _f(getattr(plan, "et0_mm_day", 0.0)),
            "vpd_kpa": _f(weather.get("vpd_kpa"), 1.2),
            "humidity_pct": _f(weather.get("humidity_pct"), 55.0),
            "air_temperature_c": air_temp,
            "tmax_c": _f(weather.get("tmax_c"), air_temp),
            "tmin_c": _f(weather.get("tmin_c"), air_temp),
            "wind_speed_m_s": _f(weather.get("wind_speed_m_s"), 1.0),
            "solar_radiation_mj_m2_day": _f(
                weather.get("solar_radiation_mj_m2_day"), 15.0
            ),
            "rainfall_mm": _f(
                weather.get("rainfall_mm"), _f(getattr(plan, "rainfall_mm_day", 0.0))
            ),
            "is_monsoon": _f(weather.get("is_monsoon"), 0.0),
            # -- the crop, as the physics already reduced it to
            "kc": _f(getattr(plan, "kc", 0.0)),
            "etc_mm": _f(getattr(plan, "etc_mm_day", 0.0)),
            "root_depth_cm": root_depth_cm,
            "gdd_accumulated": _f(getattr(plan, "gdd_accumulated", 0.0)),
            "gdd_to_next_stage": _f(getattr(plan, "gdd_to_next_stage", None), 0.0),
            "season_progress": _f(getattr(plan, "stage_progress", 0.0)),
            "days_since_sowing": _f(getattr(plan, "days_since_sowing", 0.0)),
            "p_depletion_fraction": (raw / taw) if taw > 0 else 0.0,
            # -- the soil, likewise
            "taw_mm": taw,
            "raw_mm": raw,
            "depletion_mm": depletion,
            "depletion_fraction": _f(getattr(plan, "depletion_fraction", 0.0)),
            "soil_moisture_fraction": theta,
            "soil_temperature_c": air_temp + SOIL_TEMP_OFFSET_C,
            "soil_ph": (float(ph_low) + float(ph_high)) / 2.0,
            "ec_ds_m": float(
                np.clip(
                    EC_BASE_DS_M + EC_PER_10MM_DEPLETION * depletion / 10.0, 0.05, 12.0
                )
            ),
            # -- the management
            "field_area_m2": _f(getattr(event, "field_area_m2", 1000.0), 1000.0),
            "mulched": float(bool(getattr(event, "mulched", False))),
            "nitrogen_regime": _f(getattr(event, "nitrogen_regime", 1.0), 1.0),
            "irrigation_method": str(getattr(event, "method_name", "Sprinkler")),
            "growth_stage": str(getattr(plan, "stage", "unknown")),
        }
        frame = pd.DataFrame([data])
        return encode_categoricals(self.spec.frame(frame), self.spec)

    # --- the two models --------------------------------------------------

    @staticmethod
    def _applicability(snap: FieldSnapshot) -> tuple[bool, str | None]:
        """
        Whether the requirement model may be run on this field's state.

        `NirModel` was trained on a *growing* crop: the corpus simulated each
        season only up to maturity, so `growth_stage` is one of the four
        `STAGE_ORDER` values, `season_progress` never exceeds one, and the root
        profile never collapses. A field past harvest breaks all three — the
        engine reports `stage=post_harvest`, a root depth that has shrunk back to
        the minimum, and a depletion fraction above one because TAW has collapsed
        under it. Feeding that to the estimator is extrapolation, and the smoke
        run confirmed it: post-harvest demo fields drew 18-46 mm predictions
        against a physics requirement of zero. The model is held back and the
        advisory says why.
        """
        plan = snap.current.plan
        stage = str(getattr(plan, "stage", "") or "")
        if stage not in STAGE_ORDER:
            label = stage.replace("_", " ") or "unknown"
            return False, f"{label} — no growing crop to estimate for."
        progress = _f(getattr(plan, "stage_progress", 0.0))
        if progress > 1.0 + 1e-6:
            return False, "The season is complete, so there is no crop demand to estimate."
        depletion_fraction = _f(getattr(plan, "depletion_fraction", 0.0))
        if depletion_fraction > 1.0 + 1e-6:
            return False, (
                "Root-zone depletion exceeds total available water; the profile "
                "has collapsed past harvest."
            )
        if _f(getattr(plan, "root_depth_cm", 0.0)) <= 0.0:
            return False, "Effective rooting depth is zero."
        return True, None

    def _estimate(self, snap: FieldSnapshot) -> dict[str, Any]:
        """The V2 requirement estimate plus its conformal interval."""
        requirement = getattr(snap.current.plan, "requirement", None)
        physics = _f(getattr(requirement, "net_requirement_mm", 0.0))
        single = _f(getattr(requirement, "single_application_mm", 0.0))
        block: dict[str, Any] = {
            "target": "nir_horizon_mm",
            "horizonDays": 14,
            "physicsRequirementMm": round(physics, 2),
            "physicsSingleApplicationMm": round(single, 2),
            "modelPresent": self._model is not None,
        }
        if self._model is None:
            block.update(
                {
                    "predictionMm": None,
                    "intervalMm": None,
                    "coverage": None,
                    "applicable": False,
                    "reason": None,
                }
            )
            return block

        applicable, reason = self._applicability(snap)
        block.update({"applicable": applicable, "reason": reason})
        if not applicable:
            block.update({"predictionMm": None, "intervalMm": None, "coverage": None})
            return block

        try:
            features = self._feature_frame(snap)
            prediction = float(self._model.predict(features)[0])
            if COVERAGE in self._model.conformal:
                low, high = self._model.interval(prediction, COVERAGE)
            else:  # pragma: no cover - a model saved without calibration
                low = high = prediction
            block.update(
                {
                    "predictionMm": round(prediction, 2),
                    "intervalMm": [round(low, 2), round(high, 2)],
                    "coverage": COVERAGE,
                    "residualVsPhysicsMm": round(prediction - physics, 2),
                }
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception(
                "V2 requirement inference failed for %s", snap.event.field_id
            )
            block.update(
                {
                    "predictionMm": None,
                    "intervalMm": None,
                    "coverage": None,
                    "applicable": False,
                    "reason": "inference failed",
                    "error": "inference failed",
                }
            )
        return block

    @staticmethod
    def _v1_block(advisory: dict[str, Any]) -> dict[str, Any]:
        recommendation = advisory["recommendation"]
        return {
            "shouldIrrigate": bool(recommendation.get("shouldIrrigate")),
            "depthMm": recommendation.get("depthMm"),
            "probability": recommendation.get("probability"),
            "threshold": recommendation.get("threshold"),
            "confidence": recommendation.get("confidence"),
            "action": recommendation.get("action"),
        }

    def _fuse(self, advisory: dict[str, Any], nir: dict[str, Any]) -> dict[str, Any]:
        """
        Reconcile the action model with the requirement model.

        The V1 decision stays the primary action — it is calibrated and
        operator-facing — but the requirement model is allowed to *escalate*: when
        the decision says hold while the estimator sees a real deficit beyond the
        trigger, the advisory moves to a scheduled, method-capped application
        rather than silently ignoring it. The converse is reported, not acted on,
        because downgrading a calibrated irrigate on the strength of a
        requirement estimate would trade a wasted irrigation for lost yield.
        """
        prediction = nir.get("predictionMm")
        low, high = (nir.get("intervalMm") or [None, None])[:2]
        v1_should = bool(advisory["recommendation"].get("shouldIrrigate"))
        v1_depth = _f(advisory["recommendation"].get("depthMm"))
        base_status = advisory["status"]
        single = _f(nir.get("physicsSingleApplicationMm"))

        if prediction is None:
            if nir.get("modelPresent"):
                agreement = "not_applicable"
                note = nir.get("reason") or (
                    "The requirement model does not apply to this field's current "
                    "state."
                )
            else:
                agreement = "model_unavailable"
                note = (
                    "The V2 requirement model is not loaded; the advisory uses the V1 "
                    "decision and the FAO-56 plan only."
                )
            final_status = base_status
            final_depth = v1_depth
        else:
            v2_needs = prediction > TRIGGER_MM
            if v1_should and v2_needs:
                agreement = "aligned"
                final_status = base_status
                final_depth = v1_depth
                note = (
                    f"Both models call for water: the decision model asks for "
                    f"{v1_depth:.0f} mm and the requirement model {prediction:.0f} mm "
                    f"over 14 days."
                )
            elif not v1_should and not v2_needs:
                agreement = "agree_hold"
                final_status = base_status
                final_depth = v1_depth
                note = (
                    "Neither model calls for water in the horizon: the root zone "
                    "holds the coming fortnight's demand."
                )
            elif v1_should and not v2_needs:
                agreement = "v1_conservative"
                final_status = base_status
                final_depth = v1_depth
                credible = low is not None and low > TRIGGER_MM
                note = (
                    f"The decision model calls for {v1_depth:.0f} mm while the "
                    f"requirement model sees {prediction:.0f} mm (90% interval "
                    f"{low:.0f}-{high:.0f} mm)"
                    + (
                        "; the interval excludes zero, so the two genuinely disagree."
                        if credible
                        else "; the interval includes zero, so the requirement model "
                        "does not rule the application out."
                    )
                )
            else:
                agreement = "v2_only"
                final_status = "schedule" if base_status == "hold" else base_status
                capped = min(prediction, single) if single > 0 else prediction
                final_depth = round(max(v1_depth, capped), 2)
                note = (
                    f"The requirement model sees {prediction:.0f} mm over the horizon "
                    f"(90% interval {low:.0f}-{high:.0f} mm) that the decision model "
                    f"did not call; escalating to a method-capped "
                    f"{final_depth:.0f} mm application."
                )

        return {
            "agreement": agreement,
            "finalStatus": final_status,
            "finalDepthMm": round(final_depth, 2),
            "v1DepthMm": round(v1_depth, 2),
            "v2RequirementMm": prediction,
            "note": note,
        }

    @staticmethod
    def _rescale_volume(
        snap: FieldSnapshot, advisory: dict[str, Any], depth_mm: float
    ) -> None:
        """
        Restate litres and runtime against the depth the advisory actually shows.

        One millimetre over one square metre is one litre, so litres follow from
        the depth and the field area alone; runtime follows from litres and the
        pump rate. Both are derived here rather than read off the decision so
        that an escalated depth is not displayed next to the un-escalated volume.
        """
        recommendation = advisory["recommendation"]
        area = _f(getattr(snap.event, "field_area_m2", 0.0))
        if area <= 0:
            return
        litres = max(0.0, depth_mm) * area
        recommendation["volumeLitres"] = round(litres, 1)
        recommendation["durationSeconds"] = int(round(litres / PUMP_RATE_LPM))

    # --- enrichment ------------------------------------------------------

    def _enrich(self, snap: FieldSnapshot, advisory: dict[str, Any]) -> dict[str, Any]:
        nir = self._estimate(snap)
        fusion = self._fuse(advisory, nir)
        advisory["models"] = {
            "v1": self._v1_block(advisory),
            "v2": nir,
            "fusion": fusion,
        }

        original_status = advisory["status"]
        new_status = fusion["finalStatus"]
        advisory["status"] = new_status
        advisory["recommendation"]["finalDepthMm"] = fusion["finalDepthMm"]

        # Volume and pump runtime must follow the depth actually being shown.
        # The escalation can raise the depth above what the decision model asked
        # for, and the decision's litres are derived from *its* depth, so leaving
        # them alone produced an advisory that said "irrigate 18 mm / 0 litres".
        self._rescale_volume(snap, advisory, fusion["finalDepthMm"])
        if new_status != original_status:
            depth = fusion["finalDepthMm"]
            advisory["headline"] = {
                "irrigate": f"Irrigate {depth:.0f} mm now",
                "schedule": f"Irrigate {depth:.0f} mm within 24 h",
                "monitor": advisory["headline"],
                "hold": "No irrigation needed",
            }.get(new_status, advisory["headline"])

        if nir.get("modelPresent") and nir.get("predictionMm") is not None:
            low, high = nir["intervalMm"]
            advisory["reasoning"].append(
                {
                    "label": "14-day requirement (V2 model)",
                    "value": f"{nir['predictionMm']:.1f} mm",
                    "detail": (
                        f"90% interval {low:.1f}-{high:.1f} mm; the physics alone "
                        f"asks for {nir['physicsRequirementMm']:.1f} mm."
                    ),
                    "tone": "warn" if nir["predictionMm"] > TRIGGER_MM else "healthy",
                }
            )
        advisory["reasoning"].append(
            {
                "label": "Model agreement",
                "value": AGREEMENT_LABELS.get(fusion["agreement"], fusion["agreement"]),
                "detail": fusion["note"],
                "tone": (
                    "warn"
                    if fusion["agreement"] in {"v1_conservative", "v2_only"}
                    else "healthy"
                ),
            }
        )
        return advisory

    # --- public surface --------------------------------------------------

    def advisory(self, field_id: str) -> dict[str, Any]:
        snapshot = self.compat._find(field_id)  # raises KeyError -> 404 upstream
        return self._enrich(snapshot, self.compat._advisory(snapshot))

    def advisories(self) -> dict[str, Any]:
        self.compat._require_ready()
        fields: list[dict[str, Any]] = []
        for event in self.compat.plans.list_fields():
            try:
                snapshot = self.compat._snapshot(event)
                fields.append(self._enrich(snapshot, self.compat._advisory(snapshot)))
            except Exception:
                logger.exception("pipeline failed for %s", event.field_id)
        fields.sort(key=lambda item: item["priority"], reverse=True)

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return {
            "generatedAt": now,
            "pipeline": self.describe(),
            "fleet": {
                "fields": len(fields),
                "irrigate": sum(1 for item in fields if item["status"] == "irrigate"),
                "monitor": sum(1 for item in fields if item["status"] == "monitor"),
                "hold": sum(1 for item in fields if item["status"] == "hold"),
                "recommendedMm": round(
                    sum(item["recommendation"]["finalDepthMm"] for item in fields), 1
                ),
                "recommendedLitres": round(
                    sum(item["recommendation"]["volumeLitres"] for item in fields), 0
                ),
            },
            "fields": fields,
        }
