"""
V1-shaped compatibility surface over the V2 engine.

The dashboard that ships with this project was written against V1's controller
API: `/zones`, `/telemetry/latest`, `/analytics/overview`, `/models/drift`,
`/alerts` and so on, all keyed to a live field node. V2 has no field nodes. It
has a sowing registry and a validated FAO-56 water balance. This module is the
seam between the two: V2 supplies the physical state and the calendar, V1's
learned estimators make the operator-facing decision, and the result is
serialised into the exact JSON the V1 client already understands.

**Where the inputs come from.** V1's models consume a full probe `Reading`,
including nitrogen, phosphorus, potassium, pH, EC and light intensity that V2 has
no sensor for. They are derived from V2's own conventions rather than invented:

* soil moisture is put on V1's field-capacity scale so that V1's depletion equals
  V2's `depletion_fraction` for the same field;
* pH is the midpoint of the crop's published optimum, and EC follows V2's corpus
  rule (`0.35 + 0.002 x depletion_mm`);
* nitrogen is scaled within V1's own sufficiency band by the field's
  `nitrogen_regime`; phosphorus and potassium sit at the midpoint of their bands;
* light intensity is derived from the station's solar radiation.

Every one of these is a stated approximation, not a measurement. The alternative
— fabricating sensor values silently — is the failure mode this project exists to
avoid, so the fields that are derived are documented here and surfaced through the
usual caveats on the model page.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from datetime import date, datetime, time as clock_time, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from irrigation.data.crops import CROPS
from irrigation.data.soil import texture as lookup_texture
from irrigation.sowing import SowingEvent, season_depletion

from ..core.ml_bridge import (
    NUTRIENT_BANDS,
    SOILS,
    DeviceConfig,
    ModelNotTrainedError,
    Reading,
    get_controller,
    load_metrics_summary,
)

logger = logging.getLogger("iic.compat")

#: Days of engine trajectory kept per field and re-decided. Fourteen matches the
#: planning horizon and gives the drift check enough live points to bin.
HISTORY_DAYS = 14

#: Cap on the season walk when recovering a field's recent state.
SEASON_MAX_DAYS = 400

#: Upper bound on how long a snapshot is reused, regardless of whether the
#: inputs moved. This exists only as a backstop so a long-lived process cannot
#: pin memory forever; the real key is `PlanService.version`.
SNAPSHOT_MAX_AGE_SECONDS = 900.0

#: Controller defaults, matching what V1 commissioned its simulated nodes with.
PUMP_RATE_LPM = 12.0
MAX_DEPTH_MM = 40.0
MIN_DEPTH_MM = 1.0

#: Rough lux per MJ/m2/day of solar radiation, used to place the derived
#: `light_intensity` on the scale V1's models were trained on.
LIGHT_LUX_PER_MJ = 5000.0

#: V2 uses twelve FAO-56 textures; V1's corpus knows five. A texture with no
#: direct match is mapped to the closest of V1's profiles so the decision still
#: runs, and the mapping is stated rather than hidden.
SOIL_TO_V1: dict[str, str] = {
    "Sand": "Sand",
    "Loamy_Sand": "Sandy_Loam",
    "Sandy_Loam": "Sandy_Loam",
    "Loam": "Loam",
    "Silt_Loam": "Loam",
    "Silt": "Loam",
    "Sandy_Clay_Loam": "Clay_Loam",
    "Clay_Loam": "Clay_Loam",
    "Silty_Clay_Loam": "Clay_Loam",
    "Sandy_Clay": "Clay",
    "Clay": "Clay",
    "Silty_Clay": "Clay",
}

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


class CompatUnavailable(RuntimeError):
    """The planning data needed to answer a compat request is not loaded."""


@dataclass
class FieldPoint:
    """One day of a field's trajectory, with the V1 decision for that day."""

    day: date
    plan: Any
    reading: Reading
    config: DeviceConfig
    decision: dict[str, Any]


@dataclass
class FieldSnapshot:
    """A field's recent trajectory, newest last."""

    event: SowingEvent
    soil_key: str
    points: list[FieldPoint]

    @property
    def current(self) -> FieldPoint:
        return self.points[-1]


def _stable_id(*parts: Any) -> int:
    """A stable 32-bit integer id, so ids survive a process restart."""
    text = "|".join(str(part) for part in parts)
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if not math.isfinite(number) else number


def _day_timestamp(day: date, hour: int = 6) -> str:
    moment = datetime.combine(day, clock_time(hour, 0), tzinfo=timezone.utc)
    return moment.isoformat(timespec="seconds")


def _as_date(value: Any) -> date:
    """The registry stores dates as ISO strings; the engine returns real dates."""
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _applied_on(plan: Any) -> float:
    """The depth the engine's schedule puts on the plan's own day, if any."""
    schedule = getattr(plan, "schedule", None)
    events = getattr(schedule, "events", None) if schedule is not None else None
    if not events:
        return 0.0
    for event in events:
        if int(getattr(event, "offset", -1)) == 0:
            return float(getattr(event, "apply_mm", 0.0) or 0.0)
    return 0.0


class CompatService:
    """Turns the V2 registry and engine into V1-shaped responses."""

    def __init__(self, plans: Any, evaluation: Any, settings: Any) -> None:
        self.plans = plans
        self.evaluation = evaluation
        self.settings = settings
        self.started_at = time.time()
        self._controller = None
        self._metrics: dict[str, Any] = {}
        self._bundle_error: str | None = None
        self._env: dict[str, dict[date, tuple[float, float, float, float]]] | None = None
        self._cache: dict[str, tuple[float, int, FieldSnapshot]] = {}
        self._acked: set[int] = set()
        self._load_controller()

    # --- lifecycle -------------------------------------------------------

    def _load_controller(self) -> None:
        try:
            controller = get_controller()
            controller.warmup()
            self._controller = controller
            self._metrics = load_metrics_summary()
            logger.info(
                "V1 decision bundle loaded: %s, %d tuned operating points",
                controller.metadata.get("trained_at"),
                len(controller.group_thresholds),
            )
        except ModelNotTrainedError as error:
            self._bundle_error = str(error)
            logger.warning("V1 model bundle unavailable: %s", error)
        except Exception as error:  # pragma: no cover - defensive
            self._bundle_error = str(error)
            logger.exception("could not load the V1 model bundle")

    @property
    def models_loaded(self) -> bool:
        return self._controller is not None

    def _require_ready(self) -> None:
        if self.plans is None or not self.plans.ready:
            raise CompatUnavailable(
                "planning data is not loaded; the weather corpus or the sowing "
                "registry could not be read"
            )

    def _ensure_env(self) -> None:
        """Per-station, per-day atmosphere, read once from the engine's frame."""
        if self._env is not None:
            return
        self._env = {}
        try:
            stations = self.plans.context.stations
        except Exception:
            return
        for name, station in stations.items():
            table: dict[date, tuple[float, float, float, float]] = {}
            for row in station.frame.to_dict("records"):
                raw_day = row.get("date")
                day = raw_day.date() if hasattr(raw_day, "date") else raw_day
                if day is None:
                    continue
                table[day] = (
                    _number(row.get("humidity_pct"), 55.0),
                    _number(row.get("wind_speed_m_s"), 1.0),
                    _number(row.get("solar_radiation_mj_m2_day"), 15.0),
                    _number(row.get("air_temperature_c"), 25.0),
                )
            self._env[name] = table

    # --- snapshot construction ------------------------------------------

    def _soil_key(self, event: SowingEvent) -> str:
        try:
            canonical = lookup_texture(event.soil_type).name
        except Exception:
            canonical = event.soil_type
        return SOIL_TO_V1.get(canonical, "Loam")

    def _nutrition(self, event: SowingEvent) -> tuple[float, float, float]:
        def band(name: str) -> tuple[float, float]:
            values = NUTRIENT_BANDS[name]
            return float(values[1]), float(values[2])

        n_low, n_high = band("nitrogen")
        regime = float(np.clip(event.nitrogen_regime, 0.0, 1.0))
        nitrogen = n_low + (n_high - n_low) * regime
        p_low, p_high = band("phosphorus")
        k_low, k_high = band("potassium")
        return nitrogen, (p_low + p_high) / 2.0, (k_low + k_high) / 2.0

    def _reading(
        self, event: SowingEvent, plan: Any, soil_key: str, day: date
    ) -> tuple[Reading, DeviceConfig]:
        self._ensure_env()
        humidity, wind, solar, air_temp = (self._env or {}).get(event.station, {}).get(
            day, (55.0, 1.0, 15.0, 25.0)
        )
        soil = SOILS[soil_key]
        depletion_fraction = float(np.clip(plan.depletion_fraction, 0.0, 1.0))
        soil_moisture = float(
            np.clip(soil.field_capacity_pct * (1.0 - depletion_fraction), 0.0, soil.field_capacity_pct)
        )
        crop = CROPS.get(event.crop)
        ph_low, ph_high = crop.optimal_ph if crop is not None else (6.0, 7.5)
        soil_ph = (float(ph_low) + float(ph_high)) / 2.0
        ec = 0.35 + 0.002 * float(plan.depletion_mm)
        nitrogen, phosphorus, potassium = self._nutrition(event)
        light = float(np.clip(solar * LIGHT_LUX_PER_MJ, 8000.0, 110_000.0))

        reading = Reading(
            device_id=event.field_id,
            zone=event.field_id,
            temperature=air_temp,
            humidity=humidity,
            soil_moisture=soil_moisture,
            soil_ph=soil_ph,
            ec=ec,
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium,
            light_intensity=light,
            rainfall=float(getattr(plan, "rainfall_mm_day", 0.0) or 0.0),
            wind_speed=wind,
            recorded_at=datetime.combine(day, clock_time(6, 0), tzinfo=timezone.utc),
        )
        config = DeviceConfig(
            crop=event.crop,
            soil_type=soil_key,
            field_area_m2=float(event.field_area_m2 or 1000.0),
            season_progress=float(np.clip(getattr(plan, "stage_progress", 0.5), 0.0, 1.0)),
            pump_rate_lpm=PUMP_RATE_LPM,
            max_depth_mm=MAX_DEPTH_MM,
            min_depth_mm=MIN_DEPTH_MM,
        )
        return reading, config

    def _decide(
        self, plan: Any, reading: Reading, config: DeviceConfig, history: list[Reading]
    ) -> dict[str, Any]:
        if self._controller is None:
            return self._fallback_decision(plan, reading)
        # Passing the engine's FAO-56 ET0 matters: the models were trained on
        # FAO-56 values, so feeding the node's radiation-only estimate would be
        # the distribution shift V1 goes out of its way to avoid.
        return self._controller.decide(
            reading, config, history, et0_mm_day=float(plan.et0_mm_day)
        ).to_dict()

    def _fallback_decision(self, plan: Any, reading: Reading) -> dict[str, Any]:
        """A plan-derived decision, used only when the bundle is missing."""
        should = bool(plan.days_until_stress is not None and plan.days_until_stress <= 0)
        depth = float(getattr(plan.requirement, "single_application_mm", 0.0)) if should else 0.0
        return {
            "deviceId": reading.device_id,
            "zone": reading.zone,
            "recordedAt": reading.recorded_at.isoformat(timespec="seconds"),
            "shouldIrrigate": should,
            "depthMm": round(depth, 2),
            "volumeLitres": round(depth * 1000.0, 1),
            "durationSeconds": int(round(depth * 1000.0 / PUMP_RATE_LPM)),
            "action": "MODERATE_IRRIGATION" if should else "NO_IRRIGATION",
            "diseaseRisk": "Unknown",
            "diseaseConfidence": 0.0,
            "diseaseProbabilities": {"Unknown": 1.0},
            "yieldTPerHa": 0.0,
            "nutrientDemandKgHa": {"nitrogen": 0.0, "phosphorus": 0.0, "potassium": 0.0},
            "et0MmDay": round(float(plan.et0_mm_day), 3),
            "kc": round(float(plan.kc), 4),
            "depletionFraction": round(float(plan.depletion_fraction), 4),
            "cropStressIndex": round(float(plan.depletion_fraction), 4),
            "moistureBand": "dry" if should else "optimal",
            "riskLevel": "watch" if should else "healthy",
            "confidence": 0.0,
            "explanation": ["Model bundle missing; decision derived from the FAO-56 plan."],
        }

    def _evaluation_date(self, event: SowingEvent) -> date | None:
        """
        The day the advisory should describe: the latest day the engine has weather.

        Passing `as_of=None` to `season_depletion` walks the season until maturity
        and stops there, so every field was reported at the end of its own season.
        That is right for a season replay and wrong for an advisor: a field at day
        62 in mid-season was being answered with its day-120 state, which is past
        harvest, depleted past field capacity, and therefore "no water needed"
        for every field in the registry. The engine has weather to 2024-12-31, so
        that is the latest day an honest depletion state can be computed, and it
        is what a caller asking "what should this field do" means by now.

        Returns `None` when the sowing is after that date, which leaves the
        original end-of-season behaviour in place rather than inventing a state
        for a crop that has not been planted yet.
        """
        station = self.plans.context.stations.get(event.station)
        if station is None:
            return None
        frame = getattr(station, "frame", None)
        if frame is None or len(frame) == 0:
            return None
        last = frame["date"].max()
        last_day = last.date() if hasattr(last, "date") else last
        if last_day <= event.sown_on:
            return None
        return last_day

    def _snapshot_field(self, event: SowingEvent) -> FieldSnapshot:
        context = self.plans.context
        station = context.stations.get(event.station)
        if station is None:
            raise KeyError(f"no weather for station {event.station!r}")
        climatology = context.climatologies.get(event.station)
        soil_key = self._soil_key(event)

        series = season_depletion(
            event,
            station,
            climatology,
            as_of=self._evaluation_date(event),
            max_days=SEASON_MAX_DAYS,
        )
        if not series:
            plan = self.plans.plan(event, as_of=None, depletion_mm=0.0, with_season=False)
            start = plan.as_of if isinstance(plan.as_of, date) else date.today()
            series = [(start, float(plan.depletion_mm))]

        points: list[FieldPoint] = []
        seen: list[Reading] = []
        for day, depletion in series[-HISTORY_DAYS:]:
            # The final day is requested with the season summary attached, because
            # the stage windows and the season totals are only computed on that
            # path and the advisor's crop-age card is drawn from them.
            with_season = day == series[-1][0]
            plan = self.plans.plan(
                event,
                as_of=day,
                depletion_mm=float(depletion),
                with_season=with_season,
            )
            reading, config = self._reading(event, plan, soil_key, day)
            decision = self._decide(plan, reading, config, list(seen))
            seen.append(reading)
            points.append(FieldPoint(day, plan, reading, config, decision))
        return FieldSnapshot(event, soil_key, points)

    def _snapshot(self, event: SowingEvent) -> FieldSnapshot:
        """
        Reuse a field's snapshot while the planning inputs are unchanged.

        A snapshot is a pure function of the sowing, the station's archive and the
        registry, so the honest cache key is `PlanService.version`, not a wall
        clock. The previous 60-second TTL threw away a correct answer and rebuilt
        it on a timer: the engine re-walked 14 days of FAO-56 for every field on
        every expiry, which cost roughly 25 seconds and happened once a minute —
        so the dashboard stalled open on a page load every minute of use.
        """
        cached = self._cache.get(event.field_id)
        now = time.monotonic()
        version = getattr(self.plans, "version", None)
        if (
            cached is not None
            and cached[1] == version
            and now - cached[0] < SNAPSHOT_MAX_AGE_SECONDS
        ):
            return cached[2]
        snapshot = self._snapshot_field(event)
        self._cache[event.field_id] = (now, version, snapshot)
        return snapshot

    def warm(self) -> None:
        """
        Build every field's snapshot up front.

        The first `/advisor` otherwise pays the full engine walk while the browser
        waits on it, which reads to the operator as a slow site rather than a slow
        first request. Doing it during startup moves that cost behind the launcher's
        existing readiness probe, so the page is warm by the time it is reachable.
        """
        started = time.monotonic()
        self._snapshots()
        logger.info(
            "advisor snapshots warmed for %d field(s) in %.1fs",
            len(self._cache),
            time.monotonic() - started,
        )

    def _snapshots(self) -> list[FieldSnapshot]:
        self._require_ready()
        snapshots: list[FieldSnapshot] = []
        for event in self.plans.list_fields():
            try:
                snapshots.append(self._snapshot(event))
            except Exception:
                logger.exception("compat snapshot failed for %s", event.field_id)
        return snapshots

    def _find(self, device_id: str) -> FieldSnapshot:
        self._require_ready()
        for event in self.plans.list_fields():
            if event.field_id == device_id:
                return self._snapshot(event)
        raise KeyError(device_id)

    # --- payload builders ------------------------------------------------

    def _reading_payload(self, point: FieldPoint) -> dict[str, Any]:
        reading = point.reading
        return {
            "id": _stable_id("reading", reading.device_id, point.day.isoformat()),
            "deviceId": reading.device_id,
            "zone": reading.zone,
            "recordedAt": _day_timestamp(point.day),
            "temperature": round(reading.temperature, 2),
            "humidity": round(reading.humidity, 2),
            "soilMoisture": round(reading.soil_moisture, 2),
            "soilPh": round(reading.soil_ph, 2),
            "ec": round(reading.ec, 3),
            "nitrogen": round(reading.nitrogen, 1),
            "phosphorus": round(reading.phosphorus, 1),
            "potassium": round(reading.potassium, 1),
            "lightIntensity": round(reading.light_intensity, 0),
            "batteryVolts": None,
            "rssiDbm": None,
        }

    def _device_payload(self, snap: FieldSnapshot) -> dict[str, Any]:
        event = snap.event
        age_days = max(0, (snap.current.day - _as_date(event.sowing_date)).days)
        return {
            "id": event.field_id,
            "name": event.field_id,
            "zone": event.field_id,
            "crop": event.crop,
            "soilType": event.soil_type,
            "fieldAreaM2": float(event.field_area_m2 or 0.0),
            "linkState": "online",
            "firmware": "v2-planner",
            "lastSeen": _day_timestamp(snap.current.day),
            "batteryVolts": None,
            "signalDbm": None,
            "uptimeSeconds": age_days * 86_400,
        }

    def _next_irrigation(self, point: FieldPoint) -> str | None:
        schedule = getattr(point.plan, "schedule", None)
        offset = getattr(schedule, "first_event_day", None) if schedule is not None else None
        if offset is None:
            return None
        return _day_timestamp(point.day + timedelta(days=int(offset)))

    def _zone_payload(self, snap: FieldSnapshot) -> dict[str, Any]:
        current = snap.current
        reading = current.reading
        decision = current.decision
        return {
            "zone": snap.event.field_id,
            "crop": snap.event.crop,
            "deviceId": snap.event.field_id,
            "linkState": "online",
            "soilMoisture": round(reading.soil_moisture, 2),
            "moistureBand": decision.get("moistureBand", "optimal"),
            "soilPh": round(reading.soil_ph, 2),
            "ec": round(reading.ec, 3),
            "nitrogen": round(reading.nitrogen, 1),
            "phosphorus": round(reading.phosphorus, 1),
            "potassium": round(reading.potassium, 1),
            "temperature": round(reading.temperature, 2),
            "humidity": round(reading.humidity, 2),
            "cropStressIndex": decision.get("cropStressIndex"),
            "lastSeen": _day_timestamp(current.day),
            "valveOpen": False,
            "valveRuntimeSeconds": None,
            "nextIrrigationAt": self._next_irrigation(current),
        }

    def _field_alerts(self, snap: FieldSnapshot) -> list[dict[str, Any]]:
        event = snap.event
        current = snap.current
        plan = current.plan
        schedule = getattr(plan, "schedule", None)

        def alert(kind: str, severity: str, message: str) -> dict[str, Any]:
            identifier = _stable_id("alert", event.field_id, kind)
            return {
                "id": identifier,
                "deviceId": event.field_id,
                "zone": event.field_id,
                "raisedAt": _day_timestamp(current.day),
                "kind": kind,
                "severity": severity,
                "message": message,
                "acknowledged": identifier in self._acked,
                "resolvedAt": None,
            }

        alerts: list[dict[str, Any]] = []
        if schedule is not None and getattr(schedule, "infeasible", False):
            alerts.append(
                alert(
                    "infeasible",
                    "critical",
                    f"{event.field_id}: the horizon requirement cannot be delivered "
                    f"within {event.method_name} limits.",
                )
            )
        elif schedule is not None and getattr(schedule, "requirement_insufficient", False):
            alerts.append(
                alert(
                    "requirement_insufficient",
                    "warning",
                    f"{event.field_id}: horizon demand exceeds what the soil can store "
                    "and deliver, so stress is unavoidable this cycle.",
                )
            )
        days = getattr(plan, "days_until_stress", None)
        if days is not None and days <= 3:
            alerts.append(
                alert(
                    "stress_imminent",
                    "critical" if days <= 0 else "warning",
                    f"{event.field_id}: {event.crop} reaches stress in {days} day(s) at "
                    f"{float(plan.depletion_fraction):.0%} depletion.",
                )
            )
        return alerts

    # --- public surface --------------------------------------------------

    def devices(self) -> dict[str, Any]:
        return {"devices": [self._device_payload(snap) for snap in self._snapshots()]}

    def device(self, device_id: str) -> dict[str, Any]:
        return self._device_payload(self._find(device_id))

    def zones(self) -> dict[str, Any]:
        return {"zones": [self._zone_payload(snap) for snap in self._snapshots()]}

    def latest(self, device_id: str | None, limit: int) -> dict[str, Any]:
        snapshots = self._snapshots()
        if device_id:
            snapshots = [snap for snap in snapshots if snap.event.field_id == device_id]
        readings = [self._reading_payload(point) for snap in snapshots for point in snap.points]
        recommendations = [point.decision for snap in snapshots for point in snap.points]
        readings.sort(key=lambda item: item["recordedAt"], reverse=True)
        recommendations.sort(key=lambda item: item["recordedAt"], reverse=True)
        return {
            "readings": readings[:limit],
            "recommendations": recommendations[:limit],
        }

    def history(self, device_id: str, hours: int, limit: int) -> dict[str, Any]:
        snap = self._find(device_id)
        days = int(np.clip(math.ceil(hours / 24.0), 1, 180))
        selected = snap.points[-days:]
        readings = [self._reading_payload(point) for point in selected]
        recommendations = [point.decision for point in selected]
        readings.reverse()
        recommendations.reverse()
        return {
            "readings": readings[:limit],
            "recommendations": recommendations[:limit],
        }

    def analytics(self, hours: int) -> dict[str, Any]:
        snapshots = self._snapshots()
        days = int(np.clip(math.ceil(hours / 24.0), 1, 180))
        buckets: dict[date, dict[str, list[float]]] = {}
        applied_events: list[float] = []
        runtime_seconds = 0.0
        saved_litres = 0.0
        agreements = 0
        compared = 0

        for snap in snapshots:
            area = float(snap.event.field_area_m2 or 0.0)
            for point in snap.points[-days:]:
                applied = _applied_on(point.plan)
                bucket = buckets.setdefault(
                    point.day, {"applied": [], "et0": [], "depletion": []}
                )
                bucket["applied"].append(applied)
                bucket["et0"].append(float(point.plan.et0_mm_day))
                bucket["depletion"].append(float(point.plan.depletion_fraction))

                compared += 1
                agreements += int(
                    bool(point.decision.get("shouldIrrigate")) == (applied > 0.5)
                )
                if applied > 0.5:
                    applied_events.append(applied)
                    runtime_seconds += float(point.decision.get("durationSeconds") or 0.0)
                    # A naive fixed schedule would replace the deficit every three
                    # days; the shortfall against that baseline is the relative
                    # saving, exactly as the page's caption states.
                    baseline = float(point.plan.etc_mm_day) * 3.0
                    saved_litres += max(0.0, baseline - applied) * area

        series = [
            {
                "label": day.strftime("%m-%d"),
                "appliedMm": round(float(np.mean(buckets[day]["applied"])), 2),
                "et0MmDay": round(float(np.mean(buckets[day]["et0"])), 3),
                "depletionFraction": round(float(np.mean(buckets[day]["depletion"])), 4),
            }
            for day in sorted(buckets)
        ]

        change = 0.0
        if len(series) >= 2:
            half = len(series) // 2
            first = float(np.mean([item["appliedMm"] for item in series[:half]]))
            second = float(np.mean([item["appliedMm"] for item in series[half:]]))
            if first > 0:
                change = (second - first) / first * 100.0

        return {
            "window": f"{hours}h",
            "waterAppliedMm": round(float(np.mean(applied_events)), 2) if applied_events else 0.0,
            "waterAppliedPctChange": round(change, 2),
            "estimatedLitresSaved": round(saved_litres, 1),
            "irrigationEvents": len(applied_events),
            "runtimeSeconds": int(round(runtime_seconds)),
            "averageInferenceMs": round(
                float(self._controller.last_inference_ms if self._controller else 0.0), 3
            ),
            "decisionsCorrectPct": round(100.0 * agreements / compared, 1) if compared else 0.0,
            "energyKwh": 0.0,
            "series": series,
        }

    def drift(self, hours: int) -> dict[str, Any]:
        if self._controller is None:
            return {
                "available": False,
                "reason": self._bundle_error or "the model bundle is not loaded",
                "window_hours": hours,
                "readings_considered": 0,
                "features": {},
                "frozen_sensors": [],
            }
        readings = [point.reading for snap in self._snapshots() for point in snap.points]
        report = self._controller.drift_report(readings)
        report["window_hours"] = hours
        return report

    def alerts(self, open_only: bool, limit: int) -> dict[str, Any]:
        alerts = [alert for snap in self._snapshots() for alert in self._field_alerts(snap)]
        if open_only:
            alerts = [alert for alert in alerts if not alert["acknowledged"]]
        alerts.sort(key=lambda alert: (SEVERITY_ORDER.get(alert["severity"], 9), alert["raisedAt"]))
        return {"alerts": alerts[:limit]}

    def acknowledge(self, alert_id: int) -> dict[str, Any]:
        for snap in self._snapshots():
            for alert in self._field_alerts(snap):
                if alert["id"] == alert_id:
                    self._acked.add(alert_id)
                    alert["acknowledged"] = True
                    return alert
        raise KeyError(alert_id)

    def events(self, limit: int) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        for snap in self._snapshots():
            event = snap.event
            events.append(
                {
                    "id": _stable_id("event", event.field_id, "register"),
                    "deviceId": event.field_id,
                    "zone": event.field_id,
                    "kind": "system",
                    "detail": (
                        f"Registered {event.crop} at {event.station} on "
                        f"{_as_date(event.sowing_date).isoformat()}"
                    ),
                    "createdAt": _day_timestamp(_as_date(event.sowing_date)),
                    "source": "registry",
                }
            )
            for alert in self._field_alerts(snap):
                events.append(
                    {
                        "id": _stable_id("event", event.field_id, alert["kind"]),
                        "deviceId": event.field_id,
                        "zone": event.field_id,
                        "kind": "alert",
                        "detail": alert["message"],
                        "createdAt": alert["raisedAt"],
                        "source": "engine",
                    }
                )
        events.sort(key=lambda item: item["createdAt"], reverse=True)
        return {"events": events[:limit]}

    def recommendation(self, device_id: str) -> dict[str, Any]:
        return self._find(device_id).current.decision

    def advisory(self, field_id: str) -> dict[str, Any]:
        """One field's advisory. Same shape the fusion pipeline returns, minus
        the requirement-model block, so the two services are interchangeable."""
        return self._advisory(self._find(field_id))

    def _water_source(self, described: dict[str, Any]) -> dict[str, Any]:
        try:
            weather = self.plans.context.weather
            et0 = pd.to_numeric(weather["et0_mm_day"], errors="coerce")
            dates = pd.to_datetime(weather["date"])
            span_from = dates.min().date().isoformat()
            span_to = dates.max().date().isoformat()
            return {
                "available": bool(described.get("available")),
                "records": int(described.get("records", len(weather))),
                "stations": int(described.get("stations", weather["station"].nunique())),
                "span": {"from": span_from, "to": span_to},
                "et0MeanMmDay": round(float(et0.mean()), 3),
                "et0MaxMmDay": round(float(et0.max()), 3),
                "lastSync": span_to,
                "maxStationDistanceKm": 0.0,
            }
        except Exception:
            return {
                "available": False,
                "records": 0,
                "stations": 0,
                "span": {"from": "", "to": ""},
                "et0MeanMmDay": 0.0,
                "et0MaxMmDay": 0.0,
                "lastSync": None,
                "maxStationDistanceKm": 0.0,
            }

    # --- advisor ---------------------------------------------------------

    #: Plain-language names for the engine's phenological stages. The engine
    #: reports stage as snake_case; an operator should not have to read that.
    STAGE_LABELS: dict[str, str] = {
        "initial": "Emergence and establishment",
        "development": "Vegetative development",
        "mid_season": "Mid-season (peak demand)",
        "late_season": "Late season (ripening)",
        "harvest": "Maturity",
    }

    @classmethod
    def _stage_label(cls, stage: Any) -> str:
        key = str(stage)
        if key in cls.STAGE_LABELS:
            return cls.STAGE_LABELS[key]
        return key.replace("_", " ").strip().title() or "Unknown"

    @staticmethod
    def _probability(decision: dict[str, Any], key: str) -> float:
        """The controller attaches the calibrated probability at the top level
        on newer bundles and under `raw` on older ones; accept either."""
        value = decision.get(key)
        if value is None:
            raw = decision.get("raw")
            if isinstance(raw, dict):
                value = raw.get(key.replace("irrigateProbability", "irrigate_probability"))
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _advisory(self, snap: FieldSnapshot) -> dict[str, Any]:
        """
        One field's advisory: the water decision plus the reasoning behind it.

        This is the payload the operator-facing advisor renders. Everything here
        is computed from the engine's own plan for the field, so the narrative
        and the numbers cannot disagree: the stage, the thermal time, the
        root-zone balance and the schedule all come from the same call.
        """
        event = snap.event
        point = snap.current
        plan = point.plan
        decision = point.decision
        reading = point.reading
        schedule = getattr(plan, "schedule", None)

        age_days = max(0, (point.day - _as_date(event.sowing_date)).days)
        stage = str(getattr(plan, "stage", "unknown"))
        stage_label = self._stage_label(stage)
        stage_progress = float(getattr(plan, "stage_progress", 0.0) or 0.0)
        gdd = float(getattr(plan, "gdd_accumulated", 0.0) or 0.0)
        days_until = getattr(plan, "days_until_stress", None)
        should = bool(decision.get("shouldIrrigate"))
        depth = float(decision.get("depthMm") or 0.0)
        depletion = float(getattr(plan, "depletion_fraction", 0.0) or 0.0)
        et0 = float(getattr(plan, "et0_mm_day", 0.0) or 0.0)
        etc = float(getattr(plan, "etc_mm_day", 0.0) or 0.0)
        rain = float(getattr(plan, "rainfall_mm_day", 0.0) or 0.0)
        taw = float(getattr(plan, "taw_mm", 0.0) or 0.0)
        raw_water = float(getattr(plan, "raw_mm", 0.0) or 0.0)
        root_cm = float(getattr(plan, "root_depth_cm", 0.0) or 0.0)
        kc = float(getattr(plan, "kc", 0.0) or 0.0)

        # Status is the single crisp answer the page leads with. Water stress
        # comes first, then imminent stress, then a scheduled application;
        # everything else is a hold.
        if should and days_until is not None and days_until <= 0:
            status = "irrigate"
        elif should:
            status = "schedule"
        elif days_until is not None and days_until <= 2:
            status = "monitor"
        else:
            status = "hold"

        priority = int(round(min(100.0, max(0.0, depletion * 100.0))))
        if status == "irrigate":
            priority = max(priority, 85)
        elif status == "monitor":
            priority = max(priority, 55)

        if status == "irrigate":
            headline = f"Irrigate {depth:.0f} mm now"
        elif status == "schedule":
            headline = f"Irrigate {depth:.0f} mm within 24 h"
        elif status == "monitor":
            headline = f"Hold today, stress in {days_until} day(s)"
        else:
            headline = "No irrigation needed"

        summary = (
            f"{event.crop} at {event.station} is {age_days} days past sowing, in "
            f"{stage_label.lower()} ({stage_progress:.0%} of the season, "
            f"{gdd:.0f} GDD). Root-zone depletion is {depletion:.0%} of the "
            f"{taw:.0f} mm available; today's demand is {etc:.1f} mm "
            f"(ET0 {et0:.1f} x Kc {kc:.2f})"
            + (f", less {rain:.1f} mm of rain." if rain > 0.05 else ".")
        )

        reasoning: list[dict[str, Any]] = [
            {
                "label": "Crop age",
                "value": f"{age_days} days",
                "detail": "Days since the sowing date on record.",
                "tone": "neutral",
            },
            {
                "label": "Growth stage",
                "value": stage_label,
                "detail": f"{stage_progress:.0%} through the season at {gdd:.0f} GDD.",
                "tone": "neutral",
            },
            {
                "label": "Root-zone water",
                "value": f"{depletion:.0%} depleted",
                "detail": f"{taw:.0f} mm total, {raw_water:.0f} mm readily available.",
                "tone": "warn" if depletion >= 0.55 else "healthy",
            },
            {
                "label": "Atmospheric demand",
                "value": f"{et0:.1f} mm/day ET0",
                "detail": f"Crop use {etc:.1f} mm/day at Kc {kc:.2f}.",
                "tone": "neutral",
            },
            {
                "label": "Rooting depth",
                "value": f"{root_cm:.0f} cm",
                "detail": "Effective depth the balance is computed over.",
                "tone": "neutral",
            },
        ]
        if rain > 0.05:
            reasoning.append(
                {
                    "label": "Rain credit",
                    "value": f"{rain:.1f} mm",
                    "detail": "Rainfall credited against today's demand.",
                    "tone": "healthy",
                }
            )
        if days_until is not None:
            reasoning.append(
                {
                    "label": "Stress horizon",
                    "value": f"{days_until} day(s)",
                    "detail": "Time until the root zone reaches the stress threshold.",
                    "tone": "warn" if days_until <= 2 else "healthy",
                }
            )

        events_by_offset: dict[int, Any] = {}
        if schedule is not None:
            for scheduled in getattr(schedule, "events", None) or []:
                events_by_offset[int(getattr(scheduled, "offset", -1))] = scheduled
        timeline = []
        for offset in range(0, 7):
            scheduled = events_by_offset.get(offset)
            timeline.append(
                {
                    "date": (point.day + timedelta(days=offset)).isoformat(),
                    "dayOffset": offset,
                    "applyMm": round(float(getattr(scheduled, "apply_mm", 0.0) or 0.0), 1)
                    if scheduled
                    else 0.0,
                    "grossMm": round(
                        float(getattr(scheduled, "gross_apply_mm", 0.0) or 0.0), 1
                    )
                    if scheduled
                    else 0.0,
                    "stage": self._stage_label(getattr(plan, "stage", stage)),
                }
            )

        windows = [
            {
                "stage": window.stage,
                "startDate": window.start_date,
                "endDate": window.end_date,
                "days": window.days,
                "meanKc": round(float(window.mean_kc), 3),
                "meanEtcMmDay": round(float(window.mean_etc_mm_day), 2),
                "meanRainMmDay": round(float(window.mean_rain_mm_day), 2),
                "grossRequirementMm": round(float(window.gross_requirement_mm), 1),
            }
            for window in (getattr(plan, "stage_windows", None) or [])
        ]

        return {
            "fieldId": event.field_id,
            "crop": event.crop,
            "station": event.station,
            "soilType": event.soil_type,
            "method": getattr(event, "method_name", None),
            "sowingDate": _as_date(event.sowing_date).isoformat(),
            "areaM2": float(event.field_area_m2 or 0.0),
            "ageDays": age_days,
            "stage": stage,
            "stageLabel": stage_label,
            "stageProgress": round(stage_progress, 4),
            "gdd": round(gdd, 1),
            "status": status,
            "priority": priority,
            "headline": headline,
            "summary": summary,
            "recommendation": {
                "shouldIrrigate": should,
                "depthMm": round(depth, 2),
                "volumeLitres": float(decision.get("volumeLitres") or 0.0),
                "durationSeconds": int(decision.get("durationSeconds") or 0),
                "action": decision.get("action"),
                "confidence": float(decision.get("confidence") or 0.0),
                "probability": self._probability(decision, "irrigateProbability"),
                "threshold": self._probability(decision, "irrigateThreshold"),
            },
            "water": {
                "et0MmDay": round(et0, 2),
                "etcMmDay": round(etc, 2),
                "kc": round(kc, 3),
                "rainfallMmDay": round(rain, 2),
                "tawMm": round(taw, 1),
                "rawMm": round(raw_water, 1),
                "depletionMm": round(float(getattr(plan, "depletion_mm", 0.0) or 0.0), 1),
                "depletionFraction": round(depletion, 4),
                "rootDepthCm": round(root_cm, 1),
                "daysUntilStress": days_until,
            },
            "sensors": self._reading_payload(point),
            "risk": {
                "diseaseRisk": decision.get("diseaseRisk"),
                "diseaseConfidence": float(decision.get("diseaseConfidence") or 0.0),
                "riskLevel": decision.get("riskLevel"),
                "cropStressIndex": float(decision.get("cropStressIndex") or 0.0),
                "moistureBand": decision.get("moistureBand"),
                "yieldTPerHa": float(decision.get("yieldTPerHa") or 0.0),
            },
            "reasoning": reasoning,
            "schedule": timeline,
            "season": {
                "stageWindows": windows,
                "daysInSeason": int(getattr(plan, "season_days", 0) or 0),
                "seasonGrossMm": round(float(getattr(plan, "season_gross_requirement_mm", 0.0) or 0.0), 1),
                "seasonRainMm": round(float(getattr(plan, "season_rainfall_mm", 0.0) or 0.0), 1),
            },
        }

    def advisories(self) -> dict[str, Any]:
        """Every field's advisory, most urgent first, plus a fleet roll-up."""
        self._require_ready()
        fields: list[dict[str, Any]] = []
        for event in self.plans.list_fields():
            try:
                fields.append(self._advisory(self._snapshot(event)))
            except Exception:
                logger.exception("advisor failed for %s", event.field_id)
        fields.sort(key=lambda item: item["priority"], reverse=True)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return {
            "generatedAt": now,
            "fleet": {
                "fields": len(fields),
                "irrigate": sum(1 for item in fields if item["status"] == "irrigate"),
                "monitor": sum(1 for item in fields if item["status"] == "monitor"),
                "hold": sum(1 for item in fields if item["status"] == "hold"),
                "recommendedMm": round(
                    sum(item["recommendation"]["depthMm"] for item in fields), 1
                ),
                "recommendedLitres": round(
                    sum(item["recommendation"]["volumeLitres"] for item in fields), 0
                ),
            },
            "fields": fields,
        }

    def health(self) -> dict[str, Any]:
        described = (
            self.plans.describe()
            if self.plans is not None
            else {"available": False, "records": 0, "stations": 0, "fields": 0}
        )
        snapshots: list[FieldSnapshot] = []
        if self.plans is not None and self.plans.ready:
            try:
                snapshots = self._snapshots()
            except CompatUnavailable:
                snapshots = []
        open_alerts = [
            alert
            for snap in snapshots
            for alert in self._field_alerts(snap)
            if not alert["acknowledged"]
        ]
        controller_health = self._controller.health() if self._controller else {}
        water = self._water_source(described)
        status = "ok" if (described.get("available") and self._controller is not None) else "degraded"
        return {
            "status": status,
            "version": self.settings.app_version,
            "uptimeSeconds": int(max(0.0, time.time() - self.started_at)),
            "modelsLoaded": self._controller is not None,
            "deviceCount": len(snapshots),
            "onlineDevices": len(snapshots),
            "readingsToday": len(snapshots),
            "openAlerts": len(open_alerts),
            "inferenceLatencyMs": round(float(controller_health.get("decision_latency_ms") or 0.0), 3),
            "calibration": self._controller.calibration_summary if self._controller else None,
            "depthUncertaintyMm": (
                self._controller.depth_conformal.get("half_width_mm")
                if self._controller
                else None
            ),
            "irrigateThresholdGroups": (
                len(self._controller.group_thresholds) if self._controller else None
            ),
            "modelMetrics": self._metrics or None,
            "waterSource": water,
            "lastWeatherSync": water.get("lastSync"),
            "provenance": {
                "engine": "FAO-56 water balance (V2)",
                "decisionModel": "IIC gradient-boosted estimators (V1, transferred)",
                "weather": "NASA POWER",
                "fields": len(snapshots),
                "controllerBundle": bool(self._controller),
            },
        }
