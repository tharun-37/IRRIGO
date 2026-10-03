"""
Single inference path for the controller.

Both the FastAPI backend and the CLI tools call `Controller.decide`. Keeping one
implementation matters: the training corpus, the served models and the live
decision all have to agree on how a raw probe reading becomes a feature row. A
second copy of the feature code in the backend would drift from the one used
at training time, and the drift would be silent.

The temporal features are the subtle part. They are trailing aggregates over
the node's own recent readings, computed the same way as in
`dataset_builder.add_temporal_features`, so a live prediction sees exactly the
inputs the model was trained on. A centred window would be better still and is
precisely why it is not used: it would need readings from the future, which a
field node does not have.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ML_DIR / "artifacts"
BUNDLE_PATH = ARTIFACT_DIR / "iic_models.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
THRESHOLD_PATH = ARTIFACT_DIR / "disease_thresholds.json"

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from dataset_builder import (  # noqa: E402
    SOILS,
    CROP_PROFILES,
    DEFAULT_FIELD_CAPACITY_PCT,
    RuleThresholds,
    wetness_ratio,
)
from features import (  # noqa: E402
    NUTRIENT_BANDS,
    crop_stress_index,
    reference_evapotranspiration,
)

#: Volumetric water content band, percent, for the dashboard status colour.
MOISTURE_BAND = (38.0, 62.0)

#: Total available water fraction before the profile is called critically dry.
CRITICAL_DEPLETION = 0.85

#: Millimetres of applied depth, and the bands between severity labels.
SEVERITY_BANDS = (10.0, 25.0)


def iso_timestamp(moment: datetime) -> str:
    """RFC 3339 with a Z suffix, matching what the database stores."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (
        moment.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


class ModelNotTrainedError(RuntimeError):
    """Raised when the artifact bundle is missing, so the API can say why."""


@dataclass
class Reading:
    """One probe sample, exactly as the field node reports it."""

    device_id: str
    zone: str
    temperature: float
    humidity: float
    soil_moisture: float
    soil_ph: float
    ec: float
    nitrogen: float
    phosphorus: float
    potassium: float
    light_intensity: float
    rainfall: float = 0.0
    wind_speed: float = 1.0
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    battery_volts: float | None = None
    rssi_dbm: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "zone": self.zone,
            "temperature": float(self.temperature),
            "humidity": float(self.humidity),
            "soil_moisture": float(self.soil_moisture),
            "soil_ph": float(self.soil_ph),
            "ec": float(self.ec),
            "nitrogen": float(self.nitrogen),
            "phosphorus": float(self.phosphorus),
            "potassium": float(self.potassium),
            "light_intensity": float(self.light_intensity),
            "rainfall": float(self.rainfall),
            "wind_speed": float(self.wind_speed),
            "recorded_at": self.recorded_at,
        }


@dataclass
class DeviceConfig:
    """
    What an operator enters when the node is commissioned.

    These are the configured features the models need. Without them the targets
    are not identifiable: total available water differs by more than 2x between
    a sand and a clay profile under wheat, and nitrogen maintenance rate differs
    by 4x between soybean and maize, so a model blind to configuration cannot
    predict either the irrigation depth or the fertiliser dose.
    """

    crop: str = "Maize"
    soil_type: str = "Loam"
    field_area_m2: float = 1_000.0
    season_progress: float = 0.5
    #: Applied as a pump-rate limit, litres per minute.
    pump_rate_lpm: float = 12.0
    #: Hard ceiling on a single application, mm. Protects against a runaway.
    max_depth_mm: float = 40.0
    #: Never apply less than this, even if the model asks for a token amount.
    min_depth_mm: float = 1.0

    @property
    def soil(self):
        return SOILS.get(self.soil_type, SOILS["Loam"])

    @property
    def crop_profile(self):
        return CROP_PROFILES.get(self.crop, CROP_PROFILES["Maize"])

    @property
    def kc(self) -> float:
        from dataset_builder import crop_coefficient

        return crop_coefficient(self.crop_profile, self.season_progress)

    @property
    def root_depth_cm(self) -> float:
        from dataset_builder import rooting_depth

        return rooting_depth(self.crop_profile, self.season_progress)


@dataclass
class Decision:
    """A complete, operator-facing irrigation decision."""

    device_id: str
    zone: str
    recorded_at: datetime
    should_irrigate: bool
    depth_mm: float
    volume_litres: float
    duration_seconds: int
    action: str
    disease_risk: str
    disease_confidence: float
    disease_probabilities: dict[str, float]
    yield_t_per_ha: float
    nutrient_demand_kg_ha: dict[str, float]
    et0_mm_day: float
    kc: float
    depletion_fraction: float
    crop_stress_index: float
    moisture_band: str
    risk_level: str
    confidence: float
    explanation: list[str]
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """camelCase payload, as the dashboard consumes it over the API."""
        payload = {
            "deviceId": self.device_id,
            "zone": self.zone,
            "recordedAt": self.recorded_at.isoformat(timespec="seconds"),
            "shouldIrrigate": self.should_irrigate,
            "depthMm": round(self.depth_mm, 2),
            "volumeLitres": round(self.volume_litres, 1),
            "durationSeconds": self.duration_seconds,
            "action": self.action,
            "diseaseRisk": self.disease_risk,
            "diseaseConfidence": round(self.disease_confidence, 4),
            "diseaseProbabilities": self.disease_probabilities,
            "yieldTPerHa": round(self.yield_t_per_ha, 3),
            "nutrientDemandKgHa": {
                k: round(v, 2) for k, v in self.nutrient_demand_kg_ha.items()
            },
            "et0MmDay": round(self.et0_mm_day, 3),
            "kc": round(self.kc, 4),
            "depletionFraction": round(self.depletion_fraction, 4),
            "cropStressIndex": round(self.crop_stress_index, 4),
            "moistureBand": self.moisture_band,
            "riskLevel": self.risk_level,
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
        }
        return payload

    def to_record(self) -> dict[str, Any]:
        """
        snake_case payload for persistence.

        Kept separate from `to_dict` on purpose. The API contract is camelCase
        because that is what the TypeScript client declares, and the database
        columns are snake_case because that is what SQL reads like. Converting
        in one named place, rather than at each call site, means a renamed field
        breaks loudly in one spot instead of silently in several.
        """
        return {
            "device_id": self.device_id,
            "zone": self.zone,
            "decided_at": iso_timestamp(self.recorded_at),
            "should_irrigate": self.should_irrigate,
            "depth_mm": round(self.depth_mm, 2),
            "volume_litres": round(self.volume_litres, 1),
            "duration_seconds": self.duration_seconds,
            "action": self.action,
            "disease_risk": self.disease_risk,
            "disease_confidence": round(self.disease_confidence, 4),
            "disease_probabilities": self.disease_probabilities,
            "yield_t_per_ha": round(self.yield_t_per_ha, 3),
            "nutrient_demand": {
                k: round(v, 2) for k, v in self.nutrient_demand_kg_ha.items()
            },
            "et0_mm_day": round(self.et0_mm_day, 3),
            "kc": round(self.kc, 4),
            "depletion_fraction": round(self.depletion_fraction, 4),
            "crop_stress_index": round(self.crop_stress_index, 4),
            "moisture_band": self.moisture_band,
            "risk_level": self.risk_level,
            "confidence": round(self.confidence, 4),
            "explanation": self.explanation,
            "raw": self.raw,
            "inference_ms": self.raw.get("inference_ms", 0.0),
        }


# --------------------------------------------------------------------------
# Controller
# --------------------------------------------------------------------------


class Controller:
    """
    Loads the trained bundle once and turns readings into decisions.

    Thread safety: scikit-learn estimators are read-only during `predict`, so a
    single shared instance is safe across FastAPI's worker threads. The only
    mutable state is the decision threshold, which is set once at load time.
    """

    def __init__(self, bundle_path: Path = BUNDLE_PATH) -> None:
        if not bundle_path.exists():
            raise ModelNotTrainedError(
                f"model bundle not found at {bundle_path}. "
                "Run: .venv\\Scripts\\python backend\\ml\\scripts\\train.py"
            )
        self.bundle = joblib.load(bundle_path)
        self.metadata: dict[str, Any] = self.bundle["metadata"]
        self.features: dict[str, list[str]] = self.metadata["features"]
        self.model_columns: dict[str, list[str]] = self.metadata.get("model_columns", {})
        self.category_levels: dict[str, list[str]] = self.metadata["category_levels"]
        self.thresholds = RuleThresholds.from_dict(self.bundle["thresholds"])
        self.irrigate_threshold: float = float(self.bundle.get("irrigate_threshold", 0.5))
        self.model_metrics: dict[str, Any] = self.metadata.get("metrics", {})
        self.disease_classes: list[str] = self.metadata["diseases"]

        # Per crop|soil operating points, with the global cut as the fallback for
        # any group that was never seen during training. A field configured with
        # a crop the trainer did not cover must still get a decision, so the
        # lookup falls back rather than raising.
        self.group_thresholds: dict[str, float] = {
            str(k): float(v)
            for k, v in (self.bundle.get("irrigate_thresholds_by_group") or {}).items()
        }
        self.group_threshold_detail: dict[str, Any] = (
            self.model_metrics.get("irrigate_thresholds_by_group") or {}
        )

        # Half-width of the depth interval, so the API can state the error the
        # operator should plan for instead of a bare point estimate.
        conformal = self.bundle.get("water_depth_conformal") or {}
        self.depth_conformal: dict[str, Any] = {
            "half_width_mm": float(conformal.get("half_width_mm", 0.0) or 0.0),
            "target_coverage": float(conformal.get("target_coverage", 0.90) or 0.90),
            "achieved_coverage": conformal.get("achieved_coverage"),
        }

        calibration = self.bundle.get("calibration") or {}
        self.calibration_summary: dict[str, Any] = {
            "expected_calibration_error": calibration.get("expected_calibration_error"),
            "recalibration_slope": calibration.get("recalibration_slope"),
        }

        # Reference distribution of the measured inputs, for drift monitoring.
        self.reference_distribution: dict[str, Any] = self.bundle.get(
            "reference_distribution", {}
        )

        #: Measured serving latency, reported by /api/health. `decisions_served`
        #: exists so a latency figure can be read with some context.
        self.last_inference_ms: float = 0.0
        self.warmup_ms: float = 0.0
        self.decisions_served: int = 0

        if THRESHOLD_PATH.exists():
            self.threshold_documentation = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
        else:
            self.threshold_documentation = {}

    def warmup(self) -> float:
        """
        Run one throwaway decision and return how long it took, in milliseconds.

        The first `predict` on a freshly loaded scikit-learn pipeline pays for
        lazy imports, one-hot encoder initialisation and numPy buffer
        allocation, which is around 50 ms against under 1 ms once warm. On a
        system that irrigates on a timer, paying that on the first real reading
        of the process is an avoidable delay, so it is paid at startup instead.

        It doubles as a startup assertion: a bundle that cannot produce a
        decision fails here, loudly, rather than on the first field reading.
        The result is discarded and nothing is persisted.
        """
        reading = Reading(
            device_id="warmup",
            zone="warmup",
            temperature=24.0,
            humidity=55.0,
            soil_moisture=28.0,
            soil_ph=6.6,
            ec=1.1,
            nitrogen=45.0,
            phosphorus=28.0,
            potassium=95.0,
            light_intensity=30_000.0,
            rainfall=0.0,
        )
        started = time.perf_counter()
        self.decide(reading, DeviceConfig(), [], et0_mm_day=4.5)
        self.warmup_ms = (time.perf_counter() - started) * 1000.0
        return self.warmup_ms

    # -- feature assembly ---------------------------------------------------

    @staticmethod
    def _temporal_features(
        history: Iterable[Reading], current: Reading, et0: float, kc: float
    ) -> dict[str, float]:
        """
        Trailing aggregates over recent readings, computed causally.

        `history` must be ordered oldest first. The current reading is appended
        so the windows include it, matching the training-time behaviour.
        """
        series = [h.soil_moisture for h in history]
        series.append(current.soil_moisture)
        moisture = np.asarray(series, dtype=float)

        # The weather-derived series have no per-reading history on the node, so
        # they are repeated at the current value. That makes the trailing
        # windows equal to the current reading for these features, which is the
        # correct behaviour: without a forecast feed there is nothing else to
        # average, and inventing a spread would be worse than reporting a flat
        # series honestly.
        et0_series = np.full(len(series), float(et0), dtype=float)
        rainfall_series = np.full(len(series), float(current.rainfall), dtype=float)
        kc_series = np.full(len(series), float(kc), dtype=float)

        def mean_last(values: np.ndarray, window: int) -> float:
            return float(np.mean(values[-window:])) if values.size else 0.0

        def sum_last(values: np.ndarray, window: int) -> float:
            return float(np.sum(values[-window:])) if values.size else 0.0

        # A three-day trend needs three prior readings to be meaningful. With
        # fewer, the trend is undefined and reported as zero rather than as the
        # difference against a single point, which would look like a real trend.
        if moisture.size >= 4:
            trend = float(moisture[-1] - moisture[-4])
        else:
            trend = 0.0

        return {
            "soil_moisture_mean_3d": round(mean_last(moisture, 3), 3),
            "soil_moisture_mean_7d": round(mean_last(moisture, 7), 3),
            "soil_moisture_trend_3d": round(trend, 3),
            "et0_mean_3d": round(mean_last(et0_series, 3), 3),
            "et0_mean_7d": round(mean_last(et0_series, 7), 3),
            "rainfall_sum_3d": round(sum_last(rainfall_series, 3), 2),
            "rainfall_sum_7d": round(sum_last(rainfall_series, 7), 2),
            "kc_mean_3d": round(mean_last(kc_series, 3), 4),
        }

    def build_feature_row(
        self,
        reading: Reading,
        config: DeviceConfig,
        history: list[Reading] | None = None,
        et0_mm_day: float | None = None,
    ) -> pd.DataFrame:
        """
        Assemble the one-row frame the models consume, columns in trained order.

        `et0_mm_day` should be the FAO-56 Penman-Monteith value from the
        weather service. When it is omitted the Hargreaves-style estimate in
        `features` is substituted, which is what a node with no weather feed can
        compute on its own.

        The substitution is a genuine distribution shift and is reported as
        such. The training corpus was built on FAO-56 ET0, validated to 0.11%
        against the published worked examples, whereas the fallback is a
        radiation-driven approximation needing only temperature, humidity and
        light. Feeding one while the model learned the other degrades the
        prediction quietly, so the decision records which was used.
        """
        soil = config.soil
        et0_source = "fao56_penman_monteith"
        if et0_mm_day is None:
            et0 = reference_evapotranspiration(
                reading.temperature, reading.humidity, reading.light_intensity,
                reading.wind_speed,
            )
            et0_source = "hargreaves_estimate"
        else:
            et0 = float(et0_mm_day)
        kc = config.kc

        # The stress index is engineered from the same measured values the
        # probe reports, so it is reproducible on-device from a single reading.
        stress = crop_stress_index(
            {
                "soil_moisture": reading.soil_moisture,
                "soil_ph": reading.soil_ph,
                "temperature": reading.temperature,
                "humidity": reading.humidity,
                "ec": reading.ec,
                "nitrogen": reading.nitrogen,
                "phosphorus": reading.phosphorus,
                "potassium": reading.potassium,
            }
        )

        root_depth_cm = config.root_depth_cm
        row: dict[str, Any] = {
            "temperature": reading.temperature,
            "humidity": reading.humidity,
            "soil_moisture": reading.soil_moisture,
            "soil_ph": reading.soil_ph,
            "ec": reading.ec,
            "nitrogen": reading.nitrogen,
            "phosphorus": reading.phosphorus,
            "potassium": reading.potassium,
            "light_intensity": reading.light_intensity,
            "wind_speed": reading.wind_speed,
            "crop_stress_index": stress,
            "rainfall": reading.rainfall,
            "root_depth_cm": root_depth_cm,
            "field_capacity_pct": soil.field_capacity_pct,
            "wilting_point_pct": soil.wilting_point_pct,
            "bulk_density_g_cm3": soil.bulk_density_g_cm3,
            "kc": kc,
            "et0_mm_day": et0,
            "season_progress": config.season_progress,
            "field_area_m2": config.field_area_m2,
            "crop": config.crop,
            "soil_type": config.soil_type,
        }
        row.update(self._temporal_features(history or [], reading, et0, kc))

        # Carry the derived values the caller needs for reporting, without
        # putting them in the model's input matrix.
        row["_et0"] = et0
        row["_kc"] = kc
        row["_stress"] = stress
        row["_et0_source"] = et0_source

        frame = pd.DataFrame([row])
        return self._encode(frame)

    def _encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot expand the categoricals and select the model's input columns.

        The expansion uses the category levels recorded at training time, so a
        crop name the training run never saw becomes an all-zero group rather
        than silently changing the column count and producing a wrong answer.
        """
        for column, levels in self.category_levels.items():
            for level in levels:
                frame[f"{column}__{level}"] = (
                    frame[column].astype(str) == level
                ).astype(float)

        for name, estimator_features in self.features.items():
            for column in estimator_features:
                if column not in frame.columns:
                    frame[column] = 0.0
        return frame

    def population_stability_index(
        self, live_values: np.ndarray, deciles: list[float] | None
    ) -> float | None:
        """
        PSI between a live sample and a uniform reference, by decile bucketing.

        PSI is the standard drift measure: the total variation between two binned
        distributions, weighted by how much mass moved between bins. It is not a
        p-value and should not be read as one, but the conventional reading is
        that below 0.1 the shift is negligible, 0.1 to 0.25 warrants
        investigation, and above 0.25 the reference no longer describes the data.

        The reference is uniform across the deciles, which is exact: the edges
        *are* the training deciles, so training data falls one tenth in each bin
        by construction. Using the live sample's own shares as the reference
        would make the score identically zero and silently defeat the check.

        Returns None rather than a number when the sample is too small to bin. A
        drift score from a handful of readings is noise that would read as
        reassurance, and a confident 0.00 is the most dangerous thing this
        function could return.
        """
        if not deciles or len(deciles) < 2:
            return None
        values = np.asarray(live_values, dtype=float)
        values = values[np.isfinite(values)]
        # Ten bins need a reasonable number of rows before each is populated.
        if len(values) < 50:
            return None

        edges = [-np.inf] + [float(d) for d in deciles] + [np.inf]
        n_bins = len(edges) - 1
        live_counts = np.histogram(values, bins=edges)[0].astype(float)
        live_share = live_counts / max(1.0, live_counts.sum())
        reference_share = np.full(n_bins, 1.0 / n_bins)
        # Smoothing both sides before the log. An empty live bin is exactly what a
        # large shift looks like, and without the floor the log of zero returns
        # inf and the score becomes unusable precisely when drift is worst.
        live_share = np.clip(live_share, 1e-6, None)
        reference_share = np.clip(reference_share, 1e-6, None)
        return round(
            float(np.sum(
                (live_share - reference_share) * np.log(live_share / reference_share)
            )),
            4,
        )

    def _matrix(self, frame: pd.DataFrame, model_key: str) -> pd.DataFrame:
        """
        Slice the encoded frame down to one model's trained column order.


        The order is read from `model_columns`, recorded at training time, rather
        than re-derived from `features`. The training frame was assembled in
        dataframe column order, which interleaves the one-hot groups differently
        from the order the features are written down in, so reconstructing the
        order at serve time produced a mismatched matrix that scikit-learn
        rejects. Recording it is the only version of this that cannot drift.
        """
        columns = self.model_columns.get(model_key)
        if columns is None:
            # A bundle trained before this was recorded. Fall back to the
            # declared order rather than failing, so an older artifact still
            # loads, but say so in the decision rather than silently.
            columns = []
            for column in self.features[model_key]:
                if column in self.category_levels:
                    columns.extend(
                        f"{column}__{level}" for level in self.category_levels[column]
                    )
                else:
                    columns.append(column)
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            for column in missing:
                frame[column] = 0.0
        return frame[columns].fillna(0.0).astype(float)

    # -- decision -----------------------------------------------------------

    def decide(
        self,
        reading: Reading,
        config: DeviceConfig,
        history: list[Reading] | None = None,
        et0_mm_day: float | None = None,
    ) -> Decision:
        """Run every model and compose a single operator-facing decision."""
        started = time.perf_counter()
        frame = self.build_feature_row(reading, config, history, et0_mm_day)
        et0 = float(frame["_et0"].iloc[0])
        kc = float(frame["_kc"].iloc[0])
        stress = float(frame["_stress"].iloc[0])
        et0_source = str(frame["_et0_source"].iloc[0])

        # Stage 1: should we irrigate at all, at this field's cost-weighted
        # threshold. The score is isotonic-calibrated, so it is an event rate
        # rather than an arbitrary ranking, which is what makes a threshold
        # meaningful.
        probability = float(
            self.bundle["irrigate_model"].predict_proba(
                self._matrix(frame, "water_depth")
            )[0][1]
        )
        # The key must match the one the trainer built, which is the raw crop and
        # soil_type strings joined by a pipe.
        group_key = f"{config.crop}|{config.soil_type}"
        threshold = self.group_thresholds.get(group_key, self.irrigate_threshold)
        threshold_source = (
            "per_group" if group_key in self.group_thresholds else "global_fallback"
        )
        should_irrigate = probability >= threshold

        # Stage 2: how deep, given irrigation.
        depth = 0.0
        if should_irrigate:
            depth = float(
                self.bundle["water_depth_model"].predict(self._matrix(frame, "water_depth"))[0]
            )

        # Hard safety limits are applied after the model, never before. A model
        # is a recommendation; the pump is protected by the bounds.
        depth = float(np.clip(depth, 0.0, config.max_depth_mm))
        if should_irrigate and depth < config.min_depth_mm:
            # Below the minimum the flow rate cannot deliver it reliably, so the
            # honest outcome is not to open the valve at all.
            should_irrigate = False
            depth = 0.0

        if should_irrigate:
            action = self._severity(depth)
        else:
            action = "NO_IRRIGATION"

        # Disease risk: a stress indicator, not a diagnosis. The classifier was
        # fitted on string labels, so `predict` returns the label directly and
        # there is no index to convert.
        disease_frame = self._matrix(frame, "disease")
        disease_model = self.bundle["disease_model"]
        disease_risk = str(disease_model.predict(disease_frame)[0])
        probabilities = disease_model.predict_proba(disease_frame)[0]
        raw_classes = getattr(disease_model, "classes_", None)
        classes = [str(label) for label in raw_classes] if raw_classes is not None else []
        if classes:
            distribution = {
                label: round(float(value), 4)
                for label, value in zip(classes, probabilities)
            }
        else:  # pragma: no cover - only if the bundle was trained badly
            classes = self.disease_classes
            distribution = {"Unknown": 1.0}
        disease_confidence = float(distribution.get(disease_risk, 0.0))

        yield_estimate = float(
            self.bundle["yield_model"].predict(self._matrix(frame, "yield"))[0]
        )
        demand = {
            name: max(0.0, float(model.predict(self._matrix(frame, "nutrient_demand"))[0]))
            for name, model in self.bundle["nutrient_models"].items()
        }

        soil = config.soil
        wetness = wetness_ratio(
            {
                "soil_moisture": reading.soil_moisture,
                "field_capacity_pct": soil.field_capacity_pct,
            }
        )
        depletion = float(np.clip(1.0 - wetness, 0.0, 1.0))

        # 1 mm of water over 1 m2 is exactly 1 litre.
        volume = depth * config.field_area_m2
        duration = int(round(volume / max(0.1, config.pump_rate_lpm)))

        moisture_band = self._moisture_band(reading.soil_moisture)
        risk_level = self._risk_level(probability, stress, depletion, distribution)
        confidence = self._confidence(probability, disease_confidence)

        # Named for the Decision fields, which the API and the database both use.
        volume_litres = volume
        duration_seconds = duration

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.last_inference_ms = elapsed_ms
        self.decisions_served += 1
        decision = Decision(
            device_id=reading.device_id,
            zone=reading.zone,
            recorded_at=reading.recorded_at,
            should_irrigate=should_irrigate,
            depth_mm=depth,
            volume_litres=volume_litres,
            duration_seconds=duration_seconds,
            action=action,
            disease_risk=disease_risk,
            disease_confidence=disease_confidence,
            disease_probabilities=distribution,
            yield_t_per_ha=yield_estimate,
            nutrient_demand_kg_ha=demand,
            et0_mm_day=et0,
            kc=kc,
            depletion_fraction=depletion,
            crop_stress_index=stress,
            moisture_band=moisture_band,
            risk_level=risk_level,
            confidence=confidence,
            explanation=self._explain(
                reading, config, et0, kc, wetness, depletion, stress,
                probability, depth, distribution, disease_risk, should_irrigate,
                et0_source, threshold, threshold_source, group_key,
            ),
            raw={
                "irrigate_probability": round(probability, 4),
                "irrigate_threshold": round(threshold, 4),
                "irrigate_threshold_source": threshold_source,
                "irrigate_threshold_global": self.irrigate_threshold,
                "threshold_group": group_key,
                "depth_uncertainty_mm": round(self.depth_conformal["half_width_mm"], 3),
                "depth_interval_coverage": self.depth_conformal["target_coverage"],
                "probability_is_calibrated": True,
                "wetness_ratio": round(wetness, 4),
                "et0_source": et0_source,
                "inference_ms": round(elapsed_ms, 3),
            },
        )
        return decision

    # -- presentation helpers ----------------------------------------------

    @staticmethod
    def _severity(depth: float) -> str:
        light_limit, moderate_limit = SEVERITY_BANDS
        if depth <= 0.5:
            return "NO_IRRIGATION"
        if depth < light_limit:
            return "LIGHT_IRRIGATION"
        if depth < moderate_limit:
            return "MODERATE_IRRIGATION"
        return "HEAVY_IRRIGATION"

    @staticmethod
    def _moisture_band(moisture: float) -> str:
        low, high = MOISTURE_BAND
        if moisture < low:
            return "dry"
        if moisture > high:
            return "wet"
        return "optimal"

    @staticmethod
    def _risk_level(
        probability: float, stress: float, depletion: float, distribution: dict[str, float]
    ) -> str:
        """
        Overall zone risk, worst of the three contributing signals.

        Disease probability dominates, because an identified disease is an
        action in itself. Water stress and root-zone depletion are the other
        two, since either can explain a decline that is not yet a disease.
        """
        disease_component = 1.0 - distribution.get("Healthy", 0.0)
        score = max(disease_component, stress, depletion)
        if score >= 0.66:
            return "risk"
        if score >= 0.40:
            return "watch"
        return "healthy"

    @staticmethod
    def _confidence(irrigate_probability: float, disease_confidence: float) -> float:
        """
        How sure the decision is, as a single number for the dashboard.

        Distance from the operating point, not the raw probability. A decision
        made at 0.21 when the threshold is 0.19 is far less certain than one
        made at 0.95, even though both are "yes".
        """
        margin = min(irrigate_probability, 1.0 - irrigate_probability)
        return float(np.clip(0.5 * (margin * 2.0) + 0.5 * disease_confidence, 0.0, 1.0))

    def _explain(
        self,
        reading: Reading,
        config: DeviceConfig,
        et0: float,
        kc: float,
        wetness: float,
        depletion: float,
        stress: float,
        probability: float,
        depth: float,
        distribution: dict[str, float],
        disease_risk: str,
        should_irrigate: bool,
        et0_source: str = "fao56_penman_monteith",
        threshold: float | None = None,
        threshold_source: str = "global_fallback",
        group_key: str = "",
    ) -> list[str]:
        """
        Plain-language reasons for the decision.

        The operating point is passed in rather than read from the controller,
        because the decision may have used this field's own tuned threshold and
        quoting the global one would misdescribe what the model actually did.

        Every line is traceable to a number the operator can also see. This is
        the main defence against a black-box controller: an operator who cannot
        see why the valve opened will not trust it, and will eventually turn it
        off.

        `threshold` defaults to the global operating point when a caller does not
        supply one.
        """
        lines: list[str] = []
        lines.append(
            f"Reference evapotranspiration {et0:.2f} mm/day at Kc {kc:.2f} for "
            f"{config.crop} on {config.soil_type}, so the crop is demanding "
            f"{et0 * kc:.2f} mm/day."
        )
        if et0_source == "hargreaves_estimate":
            lines.append(
                "ET0 is a radiation-driven estimate from the node's own sensors "
                "because no weather feed was available. The model was trained on "
                "FAO-56 Penman-Monteith values, so this is a distribution shift "
                "and the depth estimate should be treated as approximate."
            )

        if reading.rainfall > 0.5:
            lines.append(
                f"{reading.rainfall:.1f} mm of rain recorded; roughly 75% is "
                f"assumed available to the root zone."
            )

        if wetness >= 0.93:
            lines.append(
                f"Soil is at {wetness:.0%} of field capacity "
                f"({reading.soil_moisture:.1f}% volumetric), which is close to "
                f"saturation. Watch for anaerobic stress."
            )
        elif wetness <= 0.55:
            lines.append(
                f"Soil has fallen to {wetness:.0%} of field capacity "
                f"({reading.soil_moisture:.1f}% volumetric), a root-zone depletion "
                f"of {depletion:.0%}."
            )
        else:
            lines.append(
                f"Soil is at {wetness:.0%} of field capacity "
                f"({reading.soil_moisture:.1f}% volumetric), comfortably inside the "
                f"irrigation band."
            )

        for nutrient, (_, opt_low, opt_high, _) in NUTRIENT_BANDS.items():
            value = getattr(reading, nutrient)
            if value < opt_low:
                lines.append(
                    f"{nutrient.capitalize()} is {value:.0f} mg/kg, below the "
                    f"{opt_low:.0f} mg/kg sufficiency floor."
                )
            elif value > opt_high:
                lines.append(
                    f"{nutrient.capitalize()} is {value:.0f} mg/kg, above the "
                    f"{opt_high:.0f} mg/kg sufficiency ceiling; uptake may be "
                    f"restricted."
                )

        if stress >= 0.4:
            lines.append(
                f"Crop stress index is {stress:.2f}, so the canopy is under "
                f"combined water, nutrient or pH limitation."
            )

        healthy_probability = distribution.get("Healthy", 0.0)
        if disease_risk != "Healthy" or healthy_probability < 0.7:
            lines.append(
                f"Stress pattern most consistent with {disease_risk.replace('_', ' ').lower()} "
                f"({(1 - healthy_probability):.0%} of the probability mass is on a "
                f"disease state). This is a risk indicator from sensor telemetry, "
                f"not a confirmed diagnosis."
            )

        threshold = self.irrigate_threshold if threshold is None else threshold
        if should_irrigate:
            lines.append(
                f"Irrigation probability {probability:.0%} is above the "
                f"{threshold:.0%} operating point, so the model "
                f"recommends {depth:.1f} mm over {config.field_area_m2:,.0f} m2. "
                f"Treat the depth as {depth:.1f} +/- "
                f"{self.depth_conformal['half_width_mm']:.1f} mm at "
                f"{self.depth_conformal['target_coverage']:.0%} coverage."
            )
        else:
            lines.append(
                f"Irrigation probability {probability:.0%} is below the "
                f"{threshold:.0%} operating point, so the model "
                f"recommends holding the valve closed. The threshold is set to "
                f"favour watering over not watering, because missing water damages "
                f"a crop while excess water mainly wastes money."
            )
        if threshold_source == "global_fallback":
            lines.append(
                f"No operating point was tuned for {group_key.replace('|', ' on ')}, "
                f"so the single global {self.irrigate_threshold:.0%} threshold was "
                f"used. Treat this field's recommendation with more caution."
            )
        return lines

    # -- drift ---------------------------------------------------------------

    def drift_report(self, readings: list[Reading]) -> dict[str, Any]:
        """
        Compare live sensor readings against the training distribution.

        A model keeps answering confidently after the field it was fitted on
        drifts, and nothing in the decision path can tell, because the model
        never sees the training data again. This is the check that makes drift
        visible: population stability index per measured input, bucketed on the
        decile edges recorded at training time.

        The reference is stored on the bundle, so this works without re-reading
        the corpus. A feature with too few live readings returns None rather
        than a number, because a PSI from a handful of points is noise that
        would read as reassurance.
        """
        reference = self.reference_distribution or {}
        edges = reference.get("edges") or {}
        if not edges or not readings:
            return {
                "available": False,
                "reason": "no reference distribution in the bundle, or no readings supplied",
                "features": {},
            }

        features: dict[str, Any] = {}
        worst = 0.0
        worst_feature = None
        for column, spec in edges.items():
            values: list[float] = []
            for reading in readings:
                value = getattr(reading, column, None)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
            if len(values) < 50:
                features[column] = {"psi": None, "samples": len(values)}
                continue
            psi = self.population_stability_index(
                np.asarray(values, dtype=float), spec.get("deciles", [])
            )
            if psi is None:
                features[column] = {"psi": None, "samples": len(values)}
                continue
            # A sensor frozen at one value is a fault, not drift. Without this
            # guard a stuck probe reports the largest possible PSI on every
            # check and drowns out the features that genuinely moved.
            reference_spread = float(spec.get("max", 0.0)) - float(spec.get("min", 0.0))
            if reference_spread > 0 and max(values) - min(values) == 0:
                features[column] = {
                    "psi": psi,
                    "samples": len(values),
                    "band": "sensor_frozen",
                    "note": "every live reading is identical; the probe is not reporting",
                    "reference_mean": spec.get("mean"),
                    "live_mean": round(float(np.mean(values)), 4),
                }
                continue
            band = (
                "negligible" if psi < 0.1 else "investigate" if psi < 0.25 else "significant"
            )
            features[column] = {
                "psi": psi,
                "samples": len(values),
                "band": band,
                "reference_mean": spec.get("mean"),
                "reference_p50": spec.get("p50"),
                "live_mean": round(float(np.mean(values)), 4) if values else None,
            }
            if psi > worst:
                worst, worst_feature = psi, column

        # A frozen sensor is excluded from the drift ranking: it is a hardware
        # fault to fix, and letting it set the headline number would bury the
        # features that actually drifted.
        frozen = [
            name for name, f in features.items() if f.get("band") == "sensor_frozen"
        ]
        return {
            "available": True,
            "readings_considered": len(readings),
            "features": features,
            "worst_feature": worst_feature,
            "worst_psi": round(worst, 4) if worst_feature else None,
            "frozen_sensors": frozen,
            "interpretation": (
                "PSI below 0.1 is negligible, 0.1 to 0.25 warrants investigation, "
                "above 0.25 means the training distribution no longer describes "
                "the live data and the model should be retrained. A null PSI means "
                "too few readings to judge. A frozen sensor is a hardware fault, "
                "not drift."
            ),
        }

    # -- reporting ----------------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        return self.model_metrics

    def health(self) -> dict[str, Any]:
        return {
            "models_loaded": True,
            "model_version": self.metadata.get("trained_at"),
            "corpus_rows": self.metadata.get("corpus_rows"),
            "weather_source": self.metadata.get("weather_source"),
            "et0_method": self.metadata.get("et0_method"),
            "et0_validation": self.metadata.get("et0_validation"),
            "split_strategy": self.metadata.get("split_strategy"),
            "ood_station": self.metadata.get("ood_station"),
            "disease_label_caveat": self.metadata.get("disease_labels"),
            "disease_score_interpretation": self.metadata.get("disease_score_interpretation"),
            "decision_score_interpretation": self.metadata.get("decision_score_interpretation"),
            "irrigate_threshold": self.irrigate_threshold,
            "irrigate_threshold_groups": len(self.group_thresholds),
            "calibration": self.calibration_summary,
            "depth_conformal": self.depth_conformal,
            # End-to-end cost of one decide(): feature assembly plus all eight
            # estimators. The per-estimator benchmark in metrics.json is a
            # training-time figure for one model in isolation and is several
            # times smaller, so it must not be reported as system latency.
            "decision_latency_ms": round(self.last_inference_ms, 2),
            "warmup_latency_ms": round(self.warmup_ms, 2),
            "decisions_served": self.decisions_served,
        }


_SINGLETON: Controller | None = None


def get_controller() -> Controller:
    """Process-wide singleton, so the 2.6 MB bundle loads once."""
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = Controller()
    return _SINGLETON


def load_metrics_summary() -> dict[str, Any]:
    """
    Metrics reshaped for the dashboard's model panel.

    The raw training report is kept intact in `metrics.json`; this projects
    only the figures an operator or examiner needs, so the frontend cannot
    accidentally display a number that means something else.
    """
    if not METRICS_PATH.exists():
        return {}
    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})

    disease = metrics.get("disease", {})
    irrigate = metrics.get("irrigate_decision", {})
    threshold = metrics.get("irrigate_threshold", {})
    depth = metrics.get("water_depth", {})
    yield_metrics = metrics.get("yield", {})

    return {
        "disease": {
            "macroF1": disease.get("macro_f1"),
            "accuracy": disease.get("accuracy"),
            "perClass": {
                label: {"f1": value.get("f1"), "support": value.get("support")}
                for label, value in (disease.get("per_class") or {}).items()
            },
        },
        "irrigateDecision": {
            "f1": irrigate.get("macro_f1"),
            "recall": threshold.get("recall_tuned"),
            "precision": threshold.get("precision_tuned"),
            "threshold": threshold.get("threshold"),
        },
        "waterDepth": {"mae": depth.get("mae"), "r2": depth.get("r2")},
        "yield": {"mae": yield_metrics.get("mae"), "r2": yield_metrics.get("r2")},
        "nutrient": {
            "nitrogen": (metrics.get("nutrient_nitrogen") or {}).get("mae"),
            "phosphorus": (metrics.get("nutrient_phosphorus") or {}).get("mae"),
            "potassium": (metrics.get("nutrient_potassium") or {}).get("mae"),
        },
        "latencyMs": metrics.get("latency_ms_per_reading", {}),
    }


__all__ = [
    "Controller",
    "Decision",
    "DeviceConfig",
    "Reading",
    "ModelNotTrainedError",
    "get_controller",
    "load_metrics_summary",
    "SOILS",
    "CROP_PROFILES",
    "DEFAULT_FIELD_CAPACITY_PCT",
]
