"""
The ingest pipeline: reading in, decision out.

This is the only place that turns telemetry into an irrigation decision, so the
order of operations here is the system's actual behaviour and is worth reading
once:

1. Validate and clamp the payload. A reading outside the probe's physical range
   is a hardware fault, and it is stored and alerted on rather than silently
   corrected.
2. Resolve ET0 for the node's location from the cached NASA POWER corpus. The
   models were trained on FAO-56 Penman-Monteith values, so supplying the real
   value rather than an on-node approximation is what keeps train and serve
   consistent.
3. Load the trailing readings, because the temporal features are a function of
   recent history, not of a single sample.
4. Run inference once.
5. Persist reading, decision, alerts and the event log in one transaction, so a
   crash can never leave a decision with no reading behind it.
6. Broadcast.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from ..core.ml_bridge import inference as ml_inference
from ..core.ml_bridge import NUTRIENT_BANDS
from ..core.config import Settings
from ..core.database import Database, iso, parse_iso, utcnow
from ..db.repository import (
    AlertRepository,
    DecisionRepository,
    DeviceRepository,
    EventRepository,
    ReadingRepository,
)
from ..realtime.hub import safe_broadcast
from . import alerts as alert_rules
from .weather import WeatherService

from irrigation.data.crops import CROPS


class IngestError(ValueError):
    """A payload the node sent that cannot be accepted. Carries a 400."""


class IngestService:
    def __init__(
        self,
        database: Database,
        settings: Settings,
        controller: ml_inference.Controller,
        weather: WeatherService,
    ) -> None:
        self.db = database
        self.settings = settings
        self.controller = controller
        self.weather = weather
        self.devices = DeviceRepository(database)
        self.readings = ReadingRepository(database)
        self.decisions = DecisionRepository(database)
        self.alerts = AlertRepository(database)
        self.events = EventRepository(database)

    # -- payload handling -------------------------------------------------

    def _require(self, payload: dict[str, Any], field: str) -> float:
        if field not in payload or payload[field] is None:
            raise IngestError(f"field {field!r} is required")
        try:
            value = float(payload[field])
        except (TypeError, ValueError) as error:
            raise IngestError(f"field {field!r} must be numeric, got {payload[field]!r}") from error
        if not math.isfinite(value):
            raise IngestError(f"field {field!r} must be finite, got {value!r}")
        return value

    def _normalise(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Coerce a payload into a reading record, rejecting nonsense early.

        Values are clamped to the probe's range rather than dropped, so a
        marginal overshoot is recorded as the nearest plausible reading and a
        gross fault still alerts. Dropping the whole reading would lose the
        evidence that the probe has failed.
        """
        device_id = str(payload.get("device_id") or "").strip()
        if not device_id:
            raise IngestError("field 'device_id' is required")

        recorded_at = utcnow()
        if payload.get("recorded_at"):
            try:
                recorded_at = parse_iso(str(payload["recorded_at"]))
            except ValueError as error:
                raise IngestError(f"field 'recorded_at' is not a valid timestamp: {error}") from error

        now = utcnow()
        skew = abs((now - recorded_at).total_seconds())
        if skew > self.settings.max_clock_skew_seconds:
            # A node with no RTC drifts, and a drifting clock silently corrupts
            # every time-windowed query. Accept it but say so, rather than
            # discarding data that is probably still good.
            recorded_at = now

        def clamp(name: str, default: float) -> float:
            low, high = alert_rules.SENSOR_LIMITS.get(name, (-math.inf, math.inf))
            return min(max(self._require(payload, name), low), high)

        def node(name: str, *aliases: str) -> float | None:
            """
            One sensor quantity from the node, omitted when unsensed.

            `device_id` and `soil_moisture` are the only hard requirements: the
            probe's core measurement is why the node exists. The air readings
            may absent themselves (a DHT that reads NaN sends `null` rather
            than a number), and a bench node has no probe at all for pH, EC or
            the nutrients. Absent stays absent — filling a fabricated default
            in here would be exactly the invented number this pipeline refuses.
            What happens instead runs in `ingest()`, after the device is
            resolved: the node's own previous sample is carried forward for a
            sensor that usually works, and the documented derivations fill the
            quantities the rig cannot sense. An alias key present on the wire
            wins, and a value that is present but senseless is rejected.
            """
            for key in (name,) + aliases:
                value = payload.get(key)
                if value is None:
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError) as error:
                    raise IngestError(f"field {key!r} must be numeric, got {payload[key]!r}") from error
                if not math.isfinite(value):
                    raise IngestError(f"field {key!r} must be finite, got {value!r}")
                low, high = alert_rules.SENSOR_LIMITS.get(name, (-math.inf, math.inf))
                return min(max(value, low), high)
            return None

        return {
            "device_id": device_id,
            "zone": str(payload.get("zone") or "Unassigned"),
            "recorded_at": iso(recorded_at),
            "received_at": iso(now),
            "temperature": node("temperature", "air_temperature"),
            "humidity": node("humidity", "air_humidity"),
            "soil_moisture": clamp("soil_moisture", 40.0),
            "soil_ph": node("soil_ph", "ph"),
            "ec": node("ec"),
            "nitrogen": node("nitrogen", "nitro"),
            "phosphorus": node("phosphorus", "phosphor"),
            "potassium": node("potassium"),
            "light_intensity": node("light_intensity", "light_level"),
            "rainfall": max(0.0, float(payload.get("rainfall", 0.0) or 0.0)),
            "wind_speed": max(0.1, float(payload.get("wind_speed", 1.0) or 1.0)),
            "battery_volts": payload.get("battery_volts"),
            "rssi_dbm": payload.get("rssi_dbm"),
            "firmware": payload.get("firmware"),
            "source": str(payload.get("source", "http")),
            "metadata": payload,
        }

    # -- main entry point -------------------------------------------------

    def ingest(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Accept one telemetry payload, decide, persist and broadcast."""
        record = self._normalise(payload)

        device = self.devices.upsert(
            record["device_id"],
            name=payload.get("name"),
            zone=record["zone"],
            latitude=payload.get("latitude"),
            longitude=payload.get("longitude"),
            station_code=payload.get("station_code"),
            firmware=record["firmware"],
            field_area_m2=payload.get("field_area_m2"),
            crop=payload.get("crop"),
            soil_type=payload.get("soil_type"),
        )

        et0 = None
        if self.settings.enable_weather_enrichment:
            et0 = self.weather.et0_for(device)

        record["et0_mm_day"] = et0
        # Two honest fallbacks, in order: the node's own previous sample for a
        # sensor that usually works, and the documented derivations for the
        # quantities the rig has no probe for. Both must finish before the
        # insert, so the stored row is a complete, consequence-free sample.
        carried = self._carry_forward(record)
        derived = self._derive_unsensed(record, device, et0)
        reading_id = self.readings.insert(record)

        # Trailing history, oldest first. The temporal features are a function of
        # recent readings, so a decision made from one sample in isolation would
        # be computed against a differently-shaped input than it was trained on.
        history_rows = self.readings.previous_readings(
            record["device_id"], record["recorded_at"], limit=7
        )
        history = [
            ml_inference.Reading(
                device_id=row["deviceId"],
                zone=row["zone"],
                temperature=row["temperature"],
                humidity=row["humidity"],
                soil_moisture=row["soilMoisture"],
                soil_ph=row["soilPh"],
                ec=row["ec"],
                nitrogen=row["nitrogen"],
                phosphorus=row["phosphorus"],
                potassium=row["potassium"],
                light_intensity=row["lightIntensity"],
                rainfall=row["rainfall"],
                wind_speed=row["windSpeed"],
                recorded_at=parse_iso(row["recordedAt"]),
            )
            for row in history_rows
        ]

        current = ml_inference.Reading(
            device_id=record["device_id"],
            zone=record["zone"],
            temperature=record["temperature"],
            humidity=record["humidity"],
            soil_moisture=record["soil_moisture"],
            soil_ph=record["soil_ph"],
            ec=record["ec"],
            nitrogen=record["nitrogen"],
            phosphorus=record["phosphorus"],
            potassium=record["potassium"],
            light_intensity=record["light_intensity"],
            rainfall=record["rainfall"],
            wind_speed=record["wind_speed"],
            recorded_at=parse_iso(record["recorded_at"]),
        )

        config = self._config_for(device)
        decision = self.controller.decide(current, config, history, et0_mm_day=et0)

        decision_payload = decision.to_dict()
        decision_record = decision.to_record()
        decision_record["manual"] = False
        decision_id = self.decisions.insert({**decision_record, "reading_id": reading_id})

        reading_view = self.readings.latest(record["device_id"], limit=1)[0]
        raised = self._raise_alerts(decision_payload, reading_view, device)

        # Reflect the decision in the device row so the zone list is a single
        # read, and so the valve state and the recommendation cannot disagree.
        if decision.should_irrigate and not device["valveOpen"]:
            self.devices.set_valve(device["id"], True)
        elif not decision.should_irrigate and device["valveOpen"]:
            self.devices.set_valve(device["id"], False)

        self.events.record(
            kind="telemetry",
            detail=(
                f"Reading {record['soil_moisture']:.1f}% moisture, "
                f"ET0 {et0:.2f} mm/day -> {decision.action}"
                if et0 is not None
                else f"Reading {record['soil_moisture']:.1f}% moisture -> {decision.action}"
            ),
            source=record["source"],
            device_id=device["id"],
            zone=record["zone"],
        )

        _fan_out(reading_view, decision_payload, raised, device)
        return {
            "accepted": True,
            "readingId": reading_id,
            "decisionId": decision_id,
            "deviceId": record["device_id"],
            "zone": record["zone"],
            "shouldIrrigate": decision.should_irrigate,
            "depthMm": decision.depth_mm,
            "durationSeconds": decision.duration_seconds,
            "et0MmDay": et0,
            "alertsRaised": len(raised),
            "inferenceMs": decision.raw.get("inference_ms", 0.0),
            # Transparency for exactly the two honest fallbacks above: which
            # values were carried forward from the node's previous sample, and
            # which were filled with the pipeline's documented derivations. A
            # client that wants to distinguish "measured" from "derived" can.
            "carriedSensors": carried,
            "derivedSensors": sorted(derived),
            # The operating point that produced this decision, and how confident
            # the score is as a probability. A client showing a percentage next
            # to a recommendation needs both, or the number is decoration.
            "irrigateProbability": decision.raw.get("irrigate_probability", 0.0),
            "irrigateThreshold": decision.raw.get("irrigate_threshold", 0.0),
            "irrigateThresholdSource": decision.raw.get("irrigate_threshold_source", "unknown"),
            "probabilityIsCalibrated": decision.raw.get("probability_is_calibrated", False),
            "depthUncertaintyMm": decision.raw.get("depth_uncertainty_mm", 0.0),
            "explanation": decision.explanation,
        }

    # -- helpers ----------------------------------------------------------

    #: A missing air reading may be carried forward from the node's previous
    #: sample at most this long. An hour covers a moved node or a dropped
    #: Wi-Fi sample; longer than that the sensor must report again, because a
    #: stale value worn forward across hours is fiction, not telemetry.
    CARRY_FORWARD_MAX_SECONDS = 3600.0

    #: Row key -> payload key for the name each carries in the database.
    _CARRY_ROW_KEY = {
        "temperature": "temperature",
        "humidity": "humidity",
        "light_intensity": "lightIntensity",
    }

    def _carry_forward(self, record: dict[str, Any]) -> dict[str, str]:
        """
        Fill a node's absent sensor reading from its own previous sample.

        One flaky sample must not cost the whole reading, and a default value
        would be a fabricated number. The node's last reading within the hour
        is the newest honest estimate there is: a real measurement of the same
        probe, with its age stamped onto the response. A node whose sensor has
        been dark longer than that is refused, which is what makes the carry
        honest — it decays, and a dead sensor cannot hide behind it forever.
        The raw payload is preserved in `record["metadata"]` in every case.
        """
        carried: dict[str, str] = {}
        needed = [
            name
            for name in self._CARRY_ROW_KEY
            if record.get(name) is None
        ]
        if not needed:
            return carried
        try:
            previous = self.readings.latest(record["device_id"], limit=1)
        except Exception:
            previous = []
        recorded = parse_iso(record["recorded_at"])
        for name in needed:
            if not previous:
                raise IngestError(
                    f"field {name!r} is required: the node has no previous "
                    "reading to carry forward, so a sensor value must be sent"
                )
            last = previous[-1]
            try:
                age = (recorded - parse_iso(last["recordedAt"])).total_seconds()
            except (TypeError, ValueError):
                age = self.CARRY_FORWARD_MAX_SECONDS + 1.0
            if age < 0.0 or age > self.CARRY_FORWARD_MAX_SECONDS:
                raise IngestError(
                    f"field {name!r} is required and the node's last {name} "
                    f"reading is {age / 3600.0:.1f} h old. Report the sensor "
                    "value; a gap this long cannot be filled honestly."
                )
            value = last.get(self._CARRY_ROW_KEY[name])
            if value is None:
                raise IngestError(
                    f"field {name!r} is required and the node's previous "
                    f"reading has no {name} on record, so a value must be sent"
                )
            record[name] = float(value)
            carried[name] = last["recordedAt"]
        return carried

    def _derive_unsensed(
        self, record: dict[str, Any], device: dict[str, Any], et0: float | None
    ) -> dict[str, float]:
        """
        Fill the soil-chemistry values a rig without those probes cannot send.

        Nothing here is invented: every value is the same documented derivation
        the advisor already states for its plan replay (`compat.py`). Soil pH
        sits at the midpoint of the crop's published optimum, EC follows the
        corpus rule with the day's ET0 as the depletion proxy, and the
        nutrients sit at the same bands the advisor uses, scaled toward the
        high side of an adequately maintained regime. A node that grows a real
        NPK probe simply sends each value in the payload and wins outright.
        """
        derived: dict[str, float] = {}

        if record.get("soil_ph") is None:
            crop = CROPS.get(device.get("crop") or "")
            low, high = crop.optimal_ph if crop is not None else (6.0, 7.5)
            record["soil_ph"] = derived["soil_ph"] = (float(low) + float(high)) / 2.0

        if record.get("ec") is None:
            record["ec"] = derived["ec"] = 0.35 + 0.002 * float(et0 or 0.0)

        for name in ("nitrogen", "phosphorus", "potassium"):
            if record.get(name) is not None:
                continue
            band = NUTRIENT_BANDS[name]
            low, high = float(band[1]), float(band[2])
            # 0.85 is the band position the advisor's `nitrogen_regime` study
            # treats as "maintained": generous but not saturated.
            record[name] = derived[name] = low + (high - low) * 0.85
        return derived

    @staticmethod
    def _config_for(device: dict[str, Any]) -> ml_inference.DeviceConfig:
        return ml_inference.DeviceConfig(
            crop=device["crop"],
            soil_type=device["soilType"],
            field_area_m2=device["fieldAreaM2"],
            season_progress=device["seasonProgress"],
            pump_rate_lpm=device["pumpRateLpm"],
            max_depth_mm=min(device["maxDepthMm"], 45.0),
        )

    def _raise_alerts(
        self,
        decision: dict[str, Any],
        reading: dict[str, Any] | None,
        device: dict[str, Any],
    ) -> list[dict[str, Any]]:
        candidates = alert_rules.evaluate(decision, reading, device)
        raised: list[dict[str, Any]] = []
        for candidate in candidates:
            alert = self.alerts.raise_alert(
                device_id=device["id"],
                zone=device["zone"],
                kind=candidate.kind,
                severity=candidate.severity,
                message=candidate.message,
            )
            if alert is not None:
                raised.append(alert)
                self.events.record(
                    kind="alert",
                    detail=f"[{candidate.severity}] {candidate.message}",
                    source="rules",
                    device_id=device["id"],
                    zone=device["zone"],
                )
        return raised

    def sweep_offline_devices(self) -> list[str]:
        """Flag silent nodes. Called on a timer so the dashboard goes honest."""
        return self.devices.mark_offline()


def _fan_out(
    reading: dict[str, Any],
    decision: dict[str, Any],
    alerts_raised: list[dict[str, Any]],
    device: dict[str, Any],
) -> None:
    """Publish the new reading, its decision, and any alerts.

    Called from the sync ingest endpoint, which runs in a thread pool with no
    event loop, so this relies on `safe_broadcast` submitting to the serving
    loop. It returns immediately and never raises: the rows are already
    committed, and a dashboard that misses a live frame can still fetch the
    history.
    """
    for frame_type, payload in (
        ("telemetry", reading),
        ("recommendation", decision),
    ):
        safe_broadcast(frame_type, payload, deviceId=device["id"], zone=device["zone"])
    for alert in alerts_raised:
        safe_broadcast("alert", alert, deviceId=device["id"], zone=device["zone"])
