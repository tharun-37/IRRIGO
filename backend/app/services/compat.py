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
import pickle
import re
import time
from dataclasses import dataclass, replace
from datetime import date, datetime, time as clock_time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from irrigation.data.crops import CROPS
from irrigation.data.soil import texture as lookup_texture
from irrigation.physics.kc_curve import kc_profile_stage_means, peak_water_stage_span
from irrigation.sowing import SowingEvent, season_depletion

from ..core.database import get_database, parse_iso
from ..db.repository import (
    AlertRepository,
    DecisionRepository,
    EventRepository,
    ReadingRepository,
)
from ..core.ml_bridge import (
    CROP_PROFILES,
    NUTRIENT_BANDS,
    SOILS,
    STATIONS,
    DeviceConfig,
    ModelNotTrainedError,
    Reading,
    get_controller,
    load_metrics_summary,
)

logger = logging.getLogger("iic.compat")

#: The reference stations, keyed by code.
#:
#: The farmer view draws the field on satellite imagery, which needs a place.
#: The registry stores no boundary geometry, so the station's real coordinates
#: are the honest anchor for "roughly here". The engine already owns this table;
#: the advisory carries the two numbers rather than the browser keeping a second
#: copy that can drift from it.
_STATION_BY_CODE = {station.code: station for station in STATIONS}

#: Days of engine trajectory kept per field and re-decided. Fourteen matches the
#: planning horizon and gives the drift check enough live points to bin.
HISTORY_DAYS = 14

#: Cap on the season walk when recovering a field's recent state.
SEASON_MAX_DAYS = 400

#: Upper bound on how long a snapshot is reused, regardless of whether the
#: inputs moved. This exists only as a backstop so a long-lived process cannot
#: pin memory forever; the real key is `PlanService.version`.
SNAPSHOT_MAX_AGE_SECONDS = 900.0

#: How long a node's own reading is authoritative for *planning*. The API
#: overlay uses the same window for display (`LIVE_SENSOR_WINDOW_SECONDS` in
#: `api.compat`); keeping them equal is deliberate — a value authoritative
#: enough to display is authoritative enough to plan from. The bench posts
#: every ~10 s, so a fresh reading is seconds old, and the plan follows it at
#: every request.
LIVE_NODE_FRESH_SECONDS = 1800.0

#: Format marker for the persisted snapshot cache. Folding it into the cache
#: filename's fingerprint means any change to the snapshot's shape - the walk
#: depth, the history window, the caller's year range, or new engine fields a
#: pickled schedule is expected to carry - invalidates every cached file at
#: once, with no manual step.
SNAPSHOT_CACHE_FORMAT = "irrigo-snapshots-v2"

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
        #: Content hash of the planning inputs, computed once per process and
        #: reused for every persisted-snapshot read and write. It is the key
        #: that lets the disk cache survive a restart; the in-memory `version`
        #: counter cannot, because it restarts with the process.
        self._snapshot_fingerprint: str | None = None
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

    # --- snapshot persistence -------------------------------------------

    def _ready_for_disk_cache(self) -> bool:
        """A cache needs a directory and a registry path to fingerprint."""
        return bool(getattr(self.settings, "sowings_path", None) is not None
                    and getattr(self.settings, "logs_dir", None) is not None)

    def _fingerprint(self) -> str:
        """
        A content hash for "the planning inputs are the ones on disk".

        The in-memory `version` counter restarts at one per process, so it
        cannot key a cache that outlives the process. The registry's own bytes
        can: any rewrite through `register` or `remove` produces different
        content, and the weather archive is fixed, so hashing the registry
        plus the constants that shape a snapshot is the honest key. The format
        marker, the year range and the walk depth are folded in, so a code
        change that alters the snapshot's shape invalidates every cached file
        at once without manual action.
        """
        marker = "|".join(
            [
                SNAPSHOT_CACHE_FORMAT,
                str(getattr(self.settings, "weather_start_year", "")),
                str(getattr(self.settings, "weather_end_year", "")),
                str(HISTORY_DAYS),
                str(SEASON_MAX_DAYS),
                # The trigger fraction shapes the pickled schedule itself, so
                # changing `.env` must invalidate the disk cache or a restart
                # would serve plans computed at the previous trigger.
                str(getattr(self.settings, "irrigation_trigger_fraction", "")),
            ]
        )
        digest = hashlib.sha1()
        digest.update(marker.encode("utf-8"))
        try:
            digest.update(Path(self.settings.sowings_path).read_bytes())
        except OSError:
            digest.update(b"<missing-registry>")
        return digest.hexdigest()[:16]

    def _snapshot_path(self, event: SowingEvent, fingerprint: str) -> Path | None:
        directory = Path(self.settings.logs_dir) / "compat_snapshots" if self._ready_for_disk_cache() else None
        if directory is None:
            return None
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", event.field_id).strip("_")
        if not safe:
            return None
        return directory / f"{safe}__{fingerprint}.pkl"

    def _load_persisted(self, event: SowingEvent, fingerprint: str) -> FieldSnapshot | None:
        """
        Yesterday's answer for a field whose inputs did not move.

        A hit removes the ~10 s engine walk at every startup; a miss is
        indistinguishable from the pre-cache behaviour. The stored snapshot is
        checked against the field it claims to be - a user-edited or truncated
        cache file must degrade to a rebuild, never to a wrong answer.
        """
        if self._snapshot_fingerprint is None:
            self._snapshot_fingerprint = self._fingerprint()
        path = self._snapshot_path(event, self._snapshot_fingerprint)
        if path is None or not path.exists():
            return None
        try:
            with path.open("rb") as handle:
                snapshot = pickle.load(handle)
        except Exception:
            # Any unreadable cache entry - a half-written file, a code change
            # that moved a class, a disk fault - reverts to the walk.
            logger.debug("snapshot cache miss for %s: unreadable file", event.field_id)
            return None
        if not isinstance(snapshot, FieldSnapshot) or snapshot.event.field_id != event.field_id:
            logger.debug("snapshot cache miss for %s: content mismatch", event.field_id)
            return None
        return snapshot

    def _persist_snapshot(self, event: SowingEvent, snapshot: FieldSnapshot) -> None:
        """
        Write one snapshot for the next startup. Strictly best-effort: a cache
        that cannot be written must never be a reason the request fails.
        """
        if self._snapshot_fingerprint is None:
            self._snapshot_fingerprint = self._fingerprint()
        path = self._snapshot_path(event, self._snapshot_fingerprint)
        if path is None:
            return
        temp = path.with_suffix(".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with temp.open("wb") as handle:
                pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temp.replace(path)  # atomic on the same volume: no half-written pickles
        except OSError as error:
            logger.debug("snapshot cache write skipped for %s: %s", event.field_id, error)

    def _snapshot(self, event: SowingEvent) -> FieldSnapshot:
        """
        Reuse a field's snapshot while the planning inputs are unchanged.

        A snapshot is a pure function of the sowing, the station's archive and the
        registry, so the honest cache key is `PlanService.version`, not a wall
        clock. A rebuild is consulted against the persisted cache first - the
        walk costs roughly ten seconds per field, and on the launcher's startup
        path the same rebuild happens on every run of an unchanged season, so
        the disk ferry keeps the honest semantics and drops the repeat cost.
        The 15-minute wall-clock backstop still applies inside the process.
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
        if self._snapshot_fingerprint is None:
            self._snapshot_fingerprint = self._fingerprint()
        snapshot = self._load_persisted(event, self._snapshot_fingerprint)
        if snapshot is None:
            snapshot = self._snapshot_field(event)
            self._persist_snapshot(event, snapshot)
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
        # The alert's numbers must describe the soil the plan started from: a
        # probe-backed field must not raise "stress in N days" at a replay
        # depletion the node contradicts.
        plan, _ = self.grounded_plan(event, current.plan, current.day)
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
        recommendations = [
            {**point.decision, "source": "replay"}
            for snap in snapshots
            for point in snap.points
        ]
        readings += self._real_readings(device_id, limit)
        # The ingest POST path also assigns a persisted model decision to every
        # real row, so the recommendation feed gains those at the top: they are
        # models that actually saw the node's measurements and wrote an answer.
        try:
            decisions = DecisionRepository(get_database()).latest(device_id, limit)
        except Exception:
            decisions = []
        recommendations += [{**decision, "source": "telemetry"} for decision in decisions]
        recommendations.sort(key=lambda item: item.get("recordedAt") or item.get("decidedAt") or "", reverse=True)
        readings.sort(key=lambda item: item["recordedAt"], reverse=True)
        return {
            "readings": readings[:limit],
            "recommendations": recommendations[:limit],
        }

    def _real_readings(self, device_id: str | None, limit: int) -> list[dict[str, Any]]:
        """
        Real telemetry from the ingest database, in the replay payload shape.

        The plan replay's `recordedAt` is a replay day in the archive window,
        while a node posts "now", so a single sort on the same key puts
        genuine measurements first — exactly what a live feed should lead
        with. Values the node did not sense stay `None` (the carry-forward and
        derivation rules decide what the *decision* sees; a live feed that
        relabeled those as if they were measurements would be its own lie),
        and an unreadable store degrades to the replay-only answer.
        """
        from ..core.database import get_database
        from ..db.repository import ReadingRepository

        try:
            rows = ReadingRepository(get_database()).latest(device_id, limit)
        except Exception:
            logger.exception("real telemetry unavailable; replay only")
            return []
        readings: list[dict[str, Any]] = []
        for row in rows:
            readings.append(
                {
                    "id": str(row.get("id") or f"live-{row.get('deviceId')}"),
                    "deviceId": row.get("deviceId"),
                    "zone": row.get("zone"),
                    "recordedAt": row.get("recordedAt"),
                    "temperature": row.get("temperature"),
                    "humidity": row.get("humidity"),
                    "soilMoisture": row.get("soilMoisture"),
                    "soilPh": row.get("soilPh"),
                    "ec": row.get("ec"),
                    "nitrogen": row.get("nitrogen"),
                    "phosphorus": row.get("phosphorus"),
                    "potassium": row.get("potassium"),
                    "lightIntensity": row.get("lightIntensity"),
                    "batteryVolts": row.get("batteryVolts"),
                    "rssiDbm": row.get("rssiDbm"),
                    "source": "live-node",
                }
            )
        return readings

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
        # The drift question is about the deployed population, not the replay:
        # the PSI is only honest against rows a real probe produced. The store
        # wins whenever it has rows; the plan-replay readings remain the
        # fallback so a fleet without telemetry still gets its synthetic check
        # rather than an empty one.
        readings = self._real_drift_readings(hours)
        source = "telemetry" if readings else "replay"
        if not readings:
            readings = [point.reading for snap in self._snapshots() for point in snap.points]
        report = self._controller.drift_report(readings)
        report["window_hours"] = hours
        report["source"] = source
        return report

    def _real_drift_readings(self, hours: int, limit: int = 5_000) -> list[Reading]:
        """
        The fleet's real node rows inside the drift window, as Reading objects.

        Quantities the nodes did not sense stay NaN and are skipped per-column
        by `drift_report`'s value filter — a hole in the store must never read
        as a shifted distribution. Unreadable store degrades to an empty list,
        which the caller reads as "fall back to the replay readings".
        """
        try:
            rows = ReadingRepository(get_database()).recent(hours=hours, limit=limit)
        except Exception:
            return []
        nan = float("nan")
        readings: list[Reading] = []
        for row in rows:
            moment = datetime.now(timezone.utc)
            recorded = row.get("recordedAt")
            if recorded:
                try:
                    moment = parse_iso(recorded)
                except (TypeError, ValueError):
                    moment = datetime.now(timezone.utc)
            readings.append(
                Reading(
                    device_id=str(row.get("deviceId") or ""),
                    zone=str(row.get("zone") or ""),
                    temperature=_number(row.get("temperature"), nan),
                    humidity=_number(row.get("humidity"), nan),
                    soil_moisture=_number(row.get("soilMoisture"), nan),
                    soil_ph=_number(row.get("soilPh"), nan),
                    ec=_number(row.get("ec"), nan),
                    nitrogen=_number(row.get("nitrogen"), nan),
                    phosphorus=_number(row.get("phosphorus"), nan),
                    potassium=_number(row.get("potassium"), nan),
                    light_intensity=_number(row.get("lightIntensity"), nan),
                    rainfall=_number(row.get("rainfall"), nan),
                    wind_speed=_number(row.get("windSpeed"), nan),
                    recorded_at=moment,
                )
            )
        return readings

    def alerts(self, open_only: bool, limit: int) -> dict[str, Any]:
        """
        Open alert feed: the planner's replay alerts plus the real telemetry
        alerts the ingest rules raised on every node POST.

        A real row (drought floor, battery, sensor fault, offline) is what the
        ESP32 itself triggered — it outranks a replay-derived alert at equal
        severity because it says something measured, not modelled. Each alert
        carries its `source` so the operator knows which kind it is, and
        acknowledging is routed by the same flag: a telemetry alert is
        acknowledged in the store where it lives, a planner alert in the
        planner's memory.
        """
        planner = [
            {**alert, "source": "planner"}
            for snap in self._snapshots()
            for alert in self._field_alerts(snap)
        ]
        try:
            telemetry = [
                {**alert, "source": "telemetry"}
                for alert in AlertRepository(get_database()).list(open_only, limit)
            ]
        except Exception:
            # Telemetry store unreadable: the replay feed still ships.
            telemetry = []
        merged = telemetry + planner
        if open_only:
            merged = [alert for alert in merged if not alert["acknowledged"]]
        merged.sort(
            key=lambda alert: (SEVERITY_ORDER.get(alert["severity"], 9), alert["raisedAt"]),
        )
        return {"alerts": merged[:limit]}

    def acknowledge(self, alert_id: int) -> dict[str, Any]:
        # Telemetry alerts live in the ingest database; acknowledging them must
        # persist there, so an acknowledged drought warning does not resurface
        # after a restart. Look them up by their recent feed entry first.
        try:
            live = AlertRepository(get_database()).list(True, limit=500)
        except Exception:
            live = []
        open_alert = next(
            (alert for alert in live if alert["id"] == alert_id), None
        )
        if open_alert is not None:
            acknowledged = AlertRepository(get_database()).acknowledge(alert_id)
            if acknowledged is not None:
                return {**acknowledged, "source": "telemetry"}
        for snap in self._snapshots():
            for alert in self._field_alerts(snap):
                if alert["id"] == alert_id:
                    self._acked.add(alert_id)
                    alert["acknowledged"] = True
                    return {**alert, "source": "planner"}
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
        # Real node activity — ingests, valve commands, raised alerts — is the
        # audit trail the operator actually needs to see, and it is written to
        # the ingest store on every POST. It outranks the replay entries at
        # anything like the same time.
        try:
            events += EventRepository(get_database()).list(limit)
        except Exception:
            # Store unreadable: the replay trail still ships.
            pass
        events.sort(key=lambda item: item["createdAt"], reverse=True)
        return {"events": events[:limit]}

    def recommendation(self, device_id: str) -> dict[str, Any]:
        """
        The device's current model decision, on its own measurements when alive.

        With a fresh node reading in the window this is the same evaluation the
        advisory makes: the sense-framed reading (moisture, air, light), the
        node's trailing history behind it, and the plan the probe grounded —
        not the replay decision the field has modelled by weather alone. A
        stale or broken feed degrades to the replay answer rather than
        returning nothing.
        """
        snap = self._find(device_id)
        point = snap.current
        plan, live_row = self.grounded_plan(snap.event, point.plan, point.day)
        if live_row is None:
            return point.decision
        try:
            return self._decide(
                plan,
                self._live_sensed_reading(live_row, point.reading),
                point.config,
                self._live_history(live_row, point.reading),
            )
        except Exception:
            logger.exception("live recommendation for %s failed", device_id)
            return point.decision

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

    @staticmethod
    def _disease_probabilities(decision: dict[str, Any]) -> dict[str, float]:
        """
        The disease class distribution, cleaned and sorted by likelihood.

        Older bundles may attach it under `raw` instead of at the top level, so
        both are accepted. A missing distribution is not an error: the dashboard
        renders the winning label alone and says the breakdown is unavailable.
        """
        source = decision.get("diseaseProbabilities")
        if not isinstance(source, dict):
            raw = decision.get("raw")
            source = raw.get("disease_probabilities") if isinstance(raw, dict) else None
        if not isinstance(source, dict) or not source:
            return {}

        cleaned: dict[str, float] = {}
        for label, value in source.items():
            try:
                cleaned[str(label)] = round(float(value), 4)
            except (TypeError, ValueError):
                continue
        return dict(sorted(cleaned.items(), key=lambda item: item[1], reverse=True))

    @staticmethod
    def _nutrient_demand(decision: dict[str, Any]) -> dict[str, float]:
        """
        The NPK dose the engine asks for, in kg/ha.

        Negative predictions are clamped to zero: a dose cannot be negative, and
        an unclipped value would read to the operator as "apply minus 4 kg".
        """
        source = decision.get("nutrientDemandKgHa")
        if not isinstance(source, dict):
            raw = decision.get("raw")
            source = raw.get("nutrient_demand") if isinstance(raw, dict) else None
        if not isinstance(source, dict):
            return {"nitrogen": 0.0, "phosphorus": 0.0, "potassium": 0.0}

        demand: dict[str, float] = {}
        for name in ("nitrogen", "phosphorus", "potassium"):
            try:
                demand[name] = round(max(0.0, float(source.get(name) or 0.0)), 2)
            except (TypeError, ValueError):
                demand[name] = 0.0
        return demand

    @staticmethod
    def _yield_potential(crop: str | None) -> float:
        """
        The crop's potential yield in tonnes per hectare, from the catalogue.

        Zero when the crop is not in the catalogue: an unknown ceiling would be
        printed as a number, and a wrong number is worse than none.
        """
        if not crop:
            return 0.0
        parameters = CROPS.get(crop)
        if parameters is None:
            return 0.0
        return round(float(getattr(parameters, "yield_potential_t_ha", 0.0) or 0.0), 2)

    @staticmethod
    def _yield_ky(crop: str | None) -> float:
        """
        The crop's FAO-33 yield response factor, from the catalogue.

        The dashboard's live-soil yield adjustment multiplies this against how
        far the node's own reading sits below the moisture band: ky 1.25 (cotton)
        loses 12.5% of potential yield per 10% of evapotranspiration the water
        shortfall costs. Zero when unknown, which the client folds to "no live
        adjustment offered".
        """
        if not crop:
            return 0.0
        parameters = CROPS.get(crop)
        if parameters is None:
            return 0.0
        return round(float(getattr(parameters, "ky", 0.0) or 0.0), 3)

    @staticmethod
    def _peak_stretch(crop: str | None) -> dict[str, str] | None:
        """
        The crop's peak-water stage span, from the crop catalogue alone.

        A (fromStage, toStage) pair naming the stages the crop drinks most in,
        resolved on the profile's own Kc curve over thermal time — deliberately
        not from the per-field season windows, whose stage lengths and ETc rates
        carry the field's station weather in. Two fields of the same crop get
        the same span, whatever their sowing dates or stations, and a field
        registered tomorrow gets the same span as its crop's siblings. Null
        when the crop is outside the catalogue, which the client reads as "no
        field-independent answer exists" and falls back to the field's own
        windows.
        """
        if not crop:
            return None
        parameters = CROPS.get(crop)
        if parameters is None:
            return None
        from_stage, to_stage = peak_water_stage_span(parameters)
        return {"fromStage": from_stage, "toStage": to_stage}

    def _yield_modelled(self, crop: str | None) -> bool:
        """
        Whether the yield model was trained on this crop.

        The crop reaches the yield model as a one-hot encoding, and the bundle
        records the levels it was trained on. A crop outside that list is
        encoded as no crop at all, so the model still answers but what it
        answers is a crop-agnostic estimate rather than a barley estimate.

        That distinction has to survive to the dashboard. Measuring a
        crop-agnostic 1.7 t/ha against potato's 20 t/ha ceiling produces a 91%
        gap that means nothing, so the card is told which of the two it has.

        The bundle's own levels are the authority; the engine's crop catalogue
        is the fallback for a controller that has not been loaded yet.
        """
        if not crop:
            return False

        controller = self._controller
        levels = None
        if controller is not None:
            levels = getattr(controller, "category_levels", {}).get("crop")
        if levels:
            return crop in levels
        return crop in CROP_PROFILES

    def grounded_plan(
        self, event: SowingEvent, replay_plan: Any, day: date
    ) -> tuple[Any, dict[str, Any] | None]:
        """
        The plan a field's advisory should start from, grounded in the probe.

        A fresh node reading replaces the replay's *assumed* soil state: the
        probe's relative-wetness percent (100 at field capacity, 0 when every
        millimetre of plant-available water is gone) maps to a depletion
        fraction of `1 - p/100` of TAW, and a plan built at that measured
        depletion is returned in the first slot. ET, Kc and the rest of the
        physics then run forward from the measured state — the same optimiser,
        the same catalogue, a real starting point.

        The second slot is the fresh reading itself, or `None` when no row was
        fresh (the caller then answers from the replay alone). Telemetry or
        planning failures degrade to the replay plan: a broken probe must
        never stop the plan. Memoised through `plan_cached` on (field, day,
        depletion), so repeated calls and the pipeline's second lookup draw
        the same schedule.
        """
        row = self._live_node_row(event.field_id)
        if row is None:
            return replay_plan, None
        value = row.get("soilMoisture")
        if not isinstance(value, (int, float)):
            return replay_plan, row
        moisture = float(value)
        taw_now = float(getattr(replay_plan, "taw_mm", 0.0) or 0.0)
        if taw_now <= 0.0:
            return replay_plan, row
        measured_fraction = float(np.clip(1.0 - moisture / 100.0, 0.0, 1.0))
        measured_depletion = round(measured_fraction * taw_now, 1)
        replay_depletion = float(getattr(replay_plan, "depletion_mm", 0.0) or 0.0)
        if abs(measured_depletion - replay_depletion) <= 1.0:
            return replay_plan, row
        try:
            grounded = self.plans.plan_cached(
                event.field_id,
                day.isoformat(),
                measured_depletion,
                with_season=False,
            )
            return grounded, row
        except Exception:
            logger.exception(
                "live-soil replan for %s failed; keeping the replay", event.field_id
            )
            return replay_plan, row

    def _live_node_row(self, field_id: str) -> dict[str, Any] | None:
        """
        The field's freshest real node reading, or `None`.

        Mirrors the API overlay's lookup: newest row whose zone is the field,
        authoritative only inside the freshness window. Telemetry problems
        degrade to `None`, which keeps the advisory running on the weather
        replay — a missing probe must never kill the plan.

        The row carries every quantity the node sensed (soil moisture, air
        temperature, humidity, light) exactly as it posted them and the
        derived/unsensed chemistry it inherited at ingest, so callers can
        evaluate the models on the full measurement instead of just moisture.
        """
        try:
            rows = ReadingRepository(get_database()).latest(None, limit=200)
        except Exception:
            # Missing or unreadable telemetry store: plan from the replay.
            return None
        row = next((item for item in rows if item.get("zone") == field_id), None)
        if row is None or not row.get("recordedAt"):
            return None
        try:
            age = (datetime.now(timezone.utc) - parse_iso(row["recordedAt"])).total_seconds()
        except (TypeError, ValueError):
            return None
        if not (0.0 <= age <= LIVE_NODE_FRESH_SECONDS):
            return None
        return row

    def _live_sensed_reading(self, row: dict[str, Any], base: Reading) -> Reading:
        """
        One Reading the models evaluate the node against.

        The node-sensed quantities — soil moisture, air temperature, humidity,
        light — come off the row exactly as posted. Everything the rig cannot
        sense (pH, EC, nutrients) keeps the plan-derived value in `base`,
        which is the document decision already recorded at ingest, not an
        invention here. Luminance is clamped to the training window, the same
        clamp the replay's reading applies to solar-derived light, because a
        photoresistor left in the sun should not push the models outside the
        data they were fitted on.
        """
        moisture = _number(row.get("soilMoisture"), base.soil_moisture)
        moment = base.recorded_at
        recorded = row.get("recordedAt")
        if recorded:
            try:
                moment = parse_iso(recorded)
            except (TypeError, ValueError):
                pass
        return replace(
            base,
            soil_moisture=moisture,
            temperature=_number(row.get("temperature"), base.temperature),
            humidity=_number(row.get("humidity"), base.humidity),
            light_intensity=float(
                np.clip(_number(row.get("lightIntensity"), base.light_intensity), 8000.0, 110_000.0)
            ),
            recorded_at=moment,
        )

    def _live_history(
        self, row: dict[str, Any], base: Reading, limit: int = 7
    ) -> list[Reading]:
        """
        The node's own trailing samples, oldest first, for temporal features.

        The feature row's mean-3/mean-7/trend terms are functions of recent
        soil-moisture readings; the ingest path already evaluates them on the
        node's real history, so the advisory's re-decision must not evaluate
        an empty series and claim the same feature values. Older rows may lag
        on quantities the firmware gained later — unsensed gaps fall back to
        the plan-derived `base` value, the same carry-forward rule ingest
        applies. Spacing is the posting cadence, not a calendar day; that is
        the honest shape of a 10 s bench deployment and matches the ingest
        path's history exactly.
        """
        recent: list[Reading] = []
        try:
            rows = ReadingRepository(get_database()).previous_readings(
                str(row.get("deviceId") or ""), row.get("recordedAt") or "", limit=limit
            )
        except Exception:
            rows = []
        for item in rows:
            moment = base.recorded_at
            recorded = item.get("recordedAt")
            if recorded:
                try:
                    moment = parse_iso(recorded)
                except (TypeError, ValueError):
                    pass
            recent.append(
                replace(
                    base,
                    soil_moisture=_number(item.get("soilMoisture"), base.soil_moisture),
                    temperature=_number(item.get("temperature"), base.temperature),
                    humidity=_number(item.get("humidity"), base.humidity),
                    light_intensity=_number(item.get("lightIntensity"), base.light_intensity),
                    recorded_at=moment,
                )
            )
        return recent

    def _advisory(self, snap: FieldSnapshot) -> dict[str, Any]:
        """
        One field's advisory: the water decision plus the reasoning behind it.

        The stage, thermal time and season come from the engine's replayed plan.
        The soil state, however, comes from the field itself when a probe has
        posted into the freshness window: moisture becomes the plan's starting
        depletion, and the decision is re-made through the same controller on
        the reading the node actually took. Narrative and numbers stay on one
        source of truth per quantity — measured where it can be measured,
        replayed where it cannot.
        """
        event = snap.event
        point = snap.current
        plan = point.plan
        decision = point.decision
        reading = point.reading
        reference = _STATION_BY_CODE.get(event.station)
        schedule = getattr(plan, "schedule", None)

        age_days = max(0, (point.day - _as_date(event.sowing_date)).days)
        stage = str(getattr(plan, "stage", "unknown"))
        stage_label = self._stage_label(stage)
        stage_progress = float(getattr(plan, "stage_progress", 0.0) or 0.0)
        gdd = float(getattr(plan, "gdd_accumulated", 0.0) or 0.0)

        # A fresh probe reading replaces the replay's *assumed* soil state.
        # The node's scale is relative wetness — 100 at field capacity, 0 with
        # every millimetre of plant-available water gone — so a reading of p
        # percent means a depletion fraction of 1 - p/100 of TAW. Everything
        # downstream (ET, Kc, rain credit, the optimiser's search) then runs
        # forward from that measured state, which is exactly what precision
        # scheduling means: the physics is the same, the starting point is real.
        # Stage-based quantities are untouched; a probe cannot move Kc or age.
        # A fresh probe reading replaces the replay's *assumed* soil state:
        # `grounded_plan` maps the node's relative wetness onto a real starting
        # depletion and the full physics runs forward from it. Stage-based
        # quantities are untouched; a probe cannot move Kc or age.
        state_plan, live_row = self.grounded_plan(event, plan, point.day)
        schedule = getattr(state_plan, "schedule", None) or schedule
        moisture = None
        if live_row is not None:
            value = live_row.get("soilMoisture")
            if isinstance(value, (int, float)):
                moisture = float(value)
        if moisture is not None:
            try:
                # The decision is re-made through the same controller, on
                # the reading the probe actually took — moisture, air
                # temperature, humidity and light from the node — with the
                # node's own trailing samples behind it, so the temporal
                # features are the field's real trend. Unsensed chemistry
                # keeps the derived values, as at ingest.
                decision = self._decide(
                    state_plan,
                    self._live_sensed_reading(live_row, point.reading),
                    point.config,
                    self._live_history(live_row, point.reading),
                )
            except Exception:
                logger.exception(
                    "live reading decision for %s failed; keeping the replay",
                    event.field_id,
                )
                decision = point.decision

        days_until = getattr(state_plan, "days_until_stress", None)

        # When the live plan itself begins with an application today, the
        # decision and the schedule have to agree: the same optimiser that
        # combined the measured soil, the crop curve and the forecast into "18
        # mm today" cannot let the decision block say "hold". The factors are
        # re-priced off the day-0 event the engine scheduled, so the action
        # card, the week chart and the pump command describe one event.
        if state_plan is not plan:
            today_events = [
                scheduled
                for scheduled in (getattr(getattr(state_plan, "schedule", None), "events", None) or [])
                if int(getattr(scheduled, "offset", -1)) == 0
                and float(getattr(scheduled, "apply_mm", 0.0) or 0.0) > 0.5
            ]
            if today_events:
                event_depth = float(getattr(today_events[0], "apply_mm", 0.0))
                decision = {
                    **decision,
                    "shouldIrrigate": True,
                    "depthMm": round(event_depth, 2),
                    "volumeLitres": round(
                        event_depth * float(event.field_area_m2 or 1000.0) / 10.0, 1
                    ),
                    "durationSeconds": int(round(event_depth * 1000.0 / PUMP_RATE_LPM)),
                }
                explanation = list(decision.get("explanation") or [])
                explanation.append(
                    "The plan applies water today because the node's own reading "
                    f"({moisture:.0f}% moisture) put the root zone below the "
                    "irrigation trigger."
                )
                decision["explanation"] = explanation

        should = bool(decision.get("shouldIrrigate"))
        depth = float(decision.get("depthMm") or 0.0)
        depletion = float(getattr(state_plan, "depletion_fraction", 0.0) or 0.0)
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
        if moisture is not None and state_plan is not plan:
            reasoning.append(
                {
                    "label": "Probe override",
                    "value": f"{moisture:.0f}% moisture",
                    "detail": "The plan starts from the node's measured soil state, not the weather replay.",
                    "tone": "neutral",
                }
            )

        events_by_offset: dict[int, Any] = {}
        if schedule is not None:
            for scheduled in getattr(schedule, "events", None) or []:
                events_by_offset[int(getattr(scheduled, "offset", -1))] = scheduled
        # The horizon's daily demand and credited rain, straight off the
        # schedule the optimiser planned against. Every day needs water even
        # when the optimiser correctly schedules no pump: the week is a week
        # of requirements with pump days on it, not a week of pump days.
        demand_by_day: list[float] = (
            [float(v) for v in getattr(schedule, "daily_demand_mm", [])]
            if schedule is not None
            else []
        )
        rain_by_day: list[float] = (
            [float(v) for v in getattr(schedule, "daily_rain_mm", [])]
            if schedule is not None
            else []
        )
        timeline = []
        for offset in range(0, 7):
            scheduled = events_by_offset.get(offset)
            demand_today = demand_by_day[offset] if offset < len(demand_by_day) else 0.0
            rain_today = rain_by_day[offset] if offset < len(rain_by_day) else 0.0
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
                    # What the crop needs from this day, net of the rain the
                    # horizon expected to credit it. This is the input the
                    # optimiser planned against — not an invented bar — so the
                    # week chart can show the requirement even on days the
                    # optimiser correctly schedules no pump.
                    "needMm": round(max(0.0, demand_today - rain_today), 2),
                    # The optimiser's own soil bookkeeping: the root-zone
                    # depletion at the end of this day on the planned path,
                    # pump included. Rendered under the bars so the reader can
                    # watch the soil buffer drain towards its stress line —
                    # which is the answer to "why no watering this week".
                    "soilDepletionMm": round(
                        float(getattr(scheduled, "depletion_after_mm", 0.0) or 0.0), 1
                    )
                    if scheduled
                    else None,
                    "stage": self._stage_label(getattr(plan, "stage", stage)),
                }
            )

        # The crop's thermal stage map, for the dashboard's growth track. The
        # shares come from the catalogue's gdd_stage_ends — the engine's own
        # phenology — so every field of one crop renders the same track, the
        # same track widths and the same peak-water bar, however its own
        # weather replay split the calendar. The per-field windows stay for
        # the field's real dates and rates; the profile figures locate the
        # crop on a crop-canonical axis.
        crop_parameters = CROPS.get(event.crop)
        thermal_total = (
            float(crop_parameters.gdd_stage_ends[3])
            if crop_parameters is not None
            and crop_parameters.gdd_stage_ends
            and float(crop_parameters.gdd_stage_ends[3]) > 0
            else 0.0
        )
        thermal_spans: dict[str, float] | None = None
        profile_kc_means: dict[str, float] | None = None
        if crop_parameters is not None and thermal_total > 0:
            ends = crop_parameters.gdd_stage_ends
            thermal_spans = {
                "initial": float(ends[0]),
                "development": float(ends[1]) - float(ends[0]),
                "mid_season": float(ends[2]) - float(ends[1]),
                "late_season": float(ends[3]) - float(ends[2]),
            }
            profile_kc_means = kc_profile_stage_means(crop_parameters)

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
                # This stage's share of the season in thermal time per the crop
                # catalogue, and the catalogue's own mean Kc over the stage. As a
                # function of the crop's parameters alone, these render the same
                # growth track — widths, peak bar and all — for every field of
                # the crop. Omitted for stages or crops the catalogue does not
                # name, which the client reads as "fall back to the field's own
                # replay".
                "profileShare": (
                    round(thermal_spans[window.stage] / thermal_total, 4)
                    if thermal_spans is not None and window.stage in thermal_spans
                    else None
                ),
                "profileKc": (
                    round(profile_kc_means[window.stage], 3)
                    if thermal_spans is not None and window.stage in profile_kc_means
                    else None
                ),
            }
            for window in (getattr(plan, "stage_windows", None) or [])
        ]

        return {
            "fieldId": event.field_id,
            "crop": event.crop,
            "station": event.station,
            "stationName": reference.name if reference else None,
            "stationLat": reference.latitude if reference else None,
            "stationLon": reference.longitude if reference else None,
            # The operator's own GPS pin, when the field carries one, to override
            # the station anchor on the map. getattr because a snapshot pickled
            # before the field existed lacks the attributes — and a missing pin
            # on old data is the correct answer for it, not an error.
            "latitude": getattr(event, "latitude", None),
            "longitude": getattr(event, "longitude", None),
            "soilType": event.soil_type,
            "method": getattr(event, "method_name", None),
            "sowingDate": _as_date(event.sowing_date).isoformat(),
            "areaM2": float(event.field_area_m2 or 0.0),
            "ageDays": age_days,
            "stage": stage,
            "stageLabel": stage_label,
            "stageProgress": round(stage_progress, 4),
            "gdd": round(gdd, 1),
            # Which soil state today's plan started from: the node's own probe
            # when a fresh reading is in the window, the engine's weather
            # replay when it is not. Downstream code (and the dashboard) reads
            # this to open the loop-follower honestly: a probe-driven field
            # moves with the sensor, the others move with the weather archive.
            "plannedFrom": "probe" if state_plan is not plan else "replay",
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
                # The depletion the plan itself ran from — the probe's measured
                # value when a node is fresh, the replay's otherwise.
                "depletionMm": round(float(getattr(state_plan, "depletion_mm", 0.0) or 0.0), 1),
                "depletionFraction": round(depletion, 4),
                "rootDepthCm": round(root_cm, 1),
                "daysUntilStress": days_until,
            },
            "sensors": self._reading_payload(point),
            "risk": {
                "diseaseRisk": decision.get("diseaseRisk"),
                "diseaseConfidence": float(decision.get("diseaseConfidence") or 0.0),
                # The full class distribution, not just the winning label. A
                # farmer deciding whether to spray needs to see how close the
                # runner-up was: 62% rust against 35% blight is a different
                # decision from 62% rust against 2% blight.
                "diseaseProbabilities": self._disease_probabilities(decision),
                "riskLevel": decision.get("riskLevel"),
                "cropStressIndex": float(decision.get("cropStressIndex") or 0.0),
                "moistureBand": decision.get("moistureBand"),
                "yieldTPerHa": float(decision.get("yieldTPerHa") or 0.0),
                # The crop's potential yield, from the engine's own catalogue.
                # Without it the dashboard would have to invent a "potential" to
                # measure the gap against, and an invented ceiling is worse than
                # no ceiling.
                "yieldPotentialTPerHa": self._yield_potential(event.crop),
                # Whether the yield model was trained on this crop. A crop it
                # does not know still gets a number, but a crop-agnostic one,
                # and the gap against that crop's ceiling would be an artefact
                # rather than an opportunity.
                "yieldModelled": self._yield_modelled(event.crop),
                # FAO-33 ky, the loss coefficient the live-soil adjustment and
                # the water-limited estimate both multiply. Companion to the
                # two yield fields above, from the same catalogue entry.
                "yieldKy": self._yield_ky(event.crop),
                # The NPK dose the engine asks for, in kg/ha. Without it the
                # dashboard can only say "nitrogen is low"; with it the advice
                # is a quantity the operator can actually buy.
                "nutrientDemandKgHa": self._nutrient_demand(decision),
            },
            "reasoning": reasoning,
            "schedule": timeline,
            "season": {
                "stageWindows": windows,
                # The crop's peak-water stage span, resolved on the crop
                # catalogue's own Kc curve rather than on this field's windows.
                # This is what the dashboard draws its blue peak-water bar over,
                # so every field of one crop shows the bar over the same
                # stages — the field-specific windows locate the field's weeks,
                # not the crop's thirst.
                "peakStretch": self._peak_stretch(event.crop),
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
