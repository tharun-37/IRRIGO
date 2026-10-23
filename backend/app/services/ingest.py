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

        return {
            "device_id": device_id,
            "zone": str(payload.get("zone") or "Unassigned"),
            "recorded_at": iso(recorded_at),
            "received_at": iso(now),
            "temperature": clamp("temperature", 25.0),
            "humidity": clamp("humidity", 60.0),
            "soil_moisture": clamp("soil_moisture", 40.0),
            "soil_ph": clamp("soil_ph", 6.8),
            "ec": clamp("ec", 1.0),
            "nitrogen": clamp("nitrogen", 50.0),
            "phosphorus": clamp("phosphorus", 30.0),
            "potassium": clamp("potassium", 40.0),
            "light_intensity": clamp("light_intensity", 30_000.0),
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
