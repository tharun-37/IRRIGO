"""Data access for the live layer. Every SQL statement in the project lives here."""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from typing import Any

from ..core.database import Database, iso, json_dumps, json_loads, utcnow

SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


class DeviceRepository:
    """Devices, their commissioning config, and valve state."""

    def __init__(self, database: Database) -> None:
        self.db = database

    def upsert(
        self,
        device_id: str,
        *,
        name: str | None = None,
        zone: str | None = None,
        crop: str | None = None,
        soil_type: str | None = None,
        field_area_m2: float | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
        station_code: str | None = None,
        sowing_date: str | None = None,
        firmware: str | None = None,
        season_progress: float | None = None,
        pump_rate_lpm: float | None = None,
        max_depth_mm: float | None = None,
    ) -> dict[str, Any]:
        """
        Insert a device, or fill in anything not already set.

        Only `device_id` is required. A node that phones home before anyone has
        commissioned it is still recorded, so a field that starts reporting on
        day one is never invisible.
        """
        now = iso(utcnow())
        with self.db.transaction() as cursor:
            cursor.execute("SELECT id FROM devices WHERE id = ?", (device_id,))
            exists = cursor.fetchone() is not None
            if not exists:
                cursor.execute(
                    """
                    INSERT INTO devices (
                        id, name, zone, crop, soil_type, field_area_m2,
                        latitude, longitude, station_code, sowing_date, firmware,
                        season_progress, pump_rate_lpm, max_depth_mm,
                        link_state, created_at, last_seen
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'online', ?, ?)
                    """,
                    (
                        device_id,
                        name or device_id,
                        zone or "Unassigned",
                        crop or "Maize",
                        soil_type or "Loam",
                        field_area_m2 or 1000.0,
                        latitude,
                        longitude,
                        station_code,
                        sowing_date,
                        firmware,
                        0.5 if season_progress is None else season_progress,
                        12.0 if pump_rate_lpm is None else pump_rate_lpm,
                        40.0 if max_depth_mm is None else max_depth_mm,
                        now,
                        now,
                    ),
                )
            else:
                cursor.execute(
                    """
                    UPDATE devices SET
                        name          = COALESCE(?, name),
                        zone          = COALESCE(?, zone),
                        crop          = COALESCE(?, crop),
                        soil_type     = COALESCE(?, soil_type),
                        field_area_m2 = COALESCE(?, field_area_m2),
                        latitude      = COALESCE(?, latitude),
                        longitude     = COALESCE(?, longitude),
                        station_code  = COALESCE(?, station_code),
                        sowing_date   = COALESCE(?, sowing_date),
                        firmware      = COALESCE(?, firmware),
                        season_progress = COALESCE(?, season_progress),
                        pump_rate_lpm = COALESCE(?, pump_rate_lpm),
                        max_depth_mm  = COALESCE(?, max_depth_mm),
                        link_state    = 'online',
                        last_seen     = ?
                    WHERE id = ?
                    """,
                    (
                        name, zone, crop, soil_type, field_area_m2, latitude,
                        longitude, station_code, sowing_date, firmware,
                        season_progress, pump_rate_lpm, max_depth_mm, now, device_id,
                    ),
                )
        return self.get(device_id)  # type: ignore[return-value]

    def get(self, device_id: str) -> dict[str, Any] | None:
        with self.db.cursor() as cursor:
            cursor.execute("SELECT * FROM devices WHERE id = ?", (device_id,))
            row = cursor.fetchone()
        return _device_row(row) if row else None

    def list(self) -> list[dict[str, Any]]:
        with self.db.cursor() as cursor:
            cursor.execute("SELECT * FROM devices ORDER BY zone, id")
            return [_device_row(row) for row in cursor.fetchall()]

    def update_config(self, device_id: str, changes: dict[str, Any]) -> dict[str, Any] | None:
        allowed = {
            "name", "zone", "crop", "soil_type", "field_area_m2",
            "latitude", "longitude", "station_code", "sowing_date",
            "season_progress", "pump_rate_lpm", "max_depth_mm",
        }
        fields = {k: v for k, v in changes.items() if k in allowed and v is not None}
        if not fields:
            return self.get(device_id)
        assignments = ", ".join(f"{key} = ?" for key in fields)
        with self.db.transaction() as cursor:
            cursor.execute(
                f"UPDATE devices SET {assignments} WHERE id = ?",
                (*fields.values(), device_id),
            )
        return self.get(device_id)

    def set_valve(self, device_id: str, open_: bool) -> None:
        now = iso(utcnow())
        with self.db.transaction() as cursor:
            if open_:
                cursor.execute(
                    "UPDATE devices SET valve_open = 1, valve_opened_at = ? WHERE id = ?",
                    (now, device_id),
                )
            else:
                cursor.execute(
                    """
                    UPDATE devices
                    SET valve_open = 0,
                        valve_runtime_s = valve_runtime_s + MAX(
                            0, CAST((julianday(?) - julianday(valve_opened_at)) * 86400 AS INTEGER)
                        ),
                        valve_opened_at = NULL
                    WHERE id = ? AND valve_open = 1
                    """,
                    (now, device_id),
                )

    def mark_offline(self, older_than_seconds: int = 300) -> list[str]:
        """Flag devices that have stopped reporting, and return their ids."""
        cutoff = iso(utcnow() - timedelta(seconds=older_than_seconds))
        with self.db.transaction() as cursor:
            cursor.execute(
                "SELECT id FROM devices WHERE last_seen < ? AND link_state != 'offline'",
                (cutoff,),
            )
            ids = [row["id"] for row in cursor.fetchall()]
            if ids:
                placeholders = ", ".join("?" for _ in ids)
                cursor.execute(
                    f"UPDATE devices SET link_state = 'offline' WHERE id IN ({placeholders})",
                    ids,
                )
        return ids


class ReadingRepository:
    """Raw probe telemetry."""

    def __init__(self, database: Database) -> None:
        self.db = database

    def insert(self, payload: dict[str, Any]) -> int:
        with self.db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO readings (
                    device_id, zone, recorded_at, received_at,
                    temperature, humidity, soil_moisture, soil_ph, ec,
                    nitrogen, phosphorus, potassium, light_intensity,
                    rainfall, wind_speed, battery_volts, rssi_dbm,
                    et0_mm_day, crop_age_days, stage, gdd, source, valve_open
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["device_id"], payload["zone"],
                    payload["recorded_at"], payload["received_at"],
                    payload["temperature"], payload["humidity"],
                    payload["soil_moisture"], payload["soil_ph"], payload["ec"],
                    payload["nitrogen"], payload["phosphorus"], payload["potassium"],
                    payload["light_intensity"], payload["rainfall"],
                    payload["wind_speed"], payload.get("battery_volts"),
                    payload.get("rssi_dbm"), payload.get("et0_mm_day"),
                    payload.get("crop_age_days"), payload.get("stage"),
                    payload.get("gdd"), payload.get("source", "http"),
                    # The node's own valve state, when it reports one. It is an
                    # observation of the actuator, not a decision of ours: the
                    # closed loop runs on the node, and the dashboard's job is
                    # to show what the node says it is doing.
                    (
                        int(bool(payload["valve_open"]))
                        if payload.get("valve_open") is not None
                        else None
                    ),
                ),
            )
            return int(cursor.lastrowid or 0)

    def latest(self, device_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self.db.cursor() as cursor:
            if device_id:
                cursor.execute(
                    "SELECT * FROM readings WHERE device_id = ? "
                    "ORDER BY recorded_at DESC, id DESC LIMIT ?",
                    (device_id, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM readings ORDER BY recorded_at DESC, id DESC LIMIT ?", (limit,)
                )
            return [_reading_row(row) for row in cursor.fetchall()]

    def history(
        self, device_id: str, hours: int = 24, limit: int = 2_000
    ) -> list[dict[str, Any]]:
        since = iso(utcnow() - timedelta(hours=hours))
        with self.db.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM readings WHERE device_id = ? AND recorded_at >= ? "
                "ORDER BY recorded_at ASC, id ASC LIMIT ?",
                (device_id, since, limit),
            )
            return [_reading_row(row) for row in cursor.fetchall()]

    def previous_readings(self, device_id: str, before: str, limit: int = 7) -> list[dict[str, Any]]:
        """Readings immediately before a timestamp, oldest first, for the trend features."""
        with self.db.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM readings WHERE device_id = ? AND recorded_at < ? "
                "ORDER BY recorded_at DESC, id DESC LIMIT ?",
                (device_id, before, limit),
            )
            return [_reading_row(row) for row in reversed(cursor.fetchall())]

    def count_since(self, hours: int = 24) -> int:
        since = iso(utcnow() - timedelta(hours=hours))
        with self.db.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS n FROM readings WHERE recorded_at >= ?", (since,))
            return int(cursor.fetchone()["n"])

    def recent(self, hours: int = 168, limit: int = 5_000) -> list[dict[str, Any]]:
        """
        Newest-first readings across every device in the window, for drift checks.

        Pooled across the fleet, because the question is whether the deployed
        population has moved off its training distribution, and a ten-bin
        bucketing needs rows per bin to mean anything.
        """
        since = iso(utcnow() - timedelta(hours=hours))
        with self.db.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM readings WHERE recorded_at >= ? "
                "ORDER BY recorded_at DESC, id DESC LIMIT ?",
                (since, limit),
            )
            return [_reading_row(row) for row in cursor.fetchall()]


class DecisionRepository:
    """Model output, kept so history and analytics survive a restart."""

    def __init__(self, database: Database) -> None:
        self.db = database

    def insert(self, payload: dict[str, Any]) -> int:
        with self.db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO decisions (
                    device_id, zone, decided_at, reading_id, should_irrigate,
                    depth_mm, volume_litres, duration_seconds, action,
                    disease_risk, disease_confidence, disease_probabilities,
                    yield_t_per_ha, nutrient_demand, et0_mm_day, kc,
                    depletion_fraction, crop_stress_index, moisture_band,
                    risk_level, confidence, explanation,
                    crop_age_days, stage, gdd, stage_progress,
                    raw, inference_ms, manual
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["device_id"], payload["zone"], payload["decided_at"],
                    payload.get("reading_id"), 1 if payload["should_irrigate"] else 0,
                    payload["depth_mm"], payload["volume_litres"],
                    payload["duration_seconds"], payload["action"],
                    payload["disease_risk"], payload["disease_confidence"],
                    json_dumps(payload["disease_probabilities"]),
                    payload["yield_t_per_ha"], json_dumps(payload["nutrient_demand"]),
                    payload["et0_mm_day"], payload["kc"],
                    payload["depletion_fraction"], payload["crop_stress_index"],
                    payload["moisture_band"], payload["risk_level"],
                    payload["confidence"], json_dumps(payload["explanation"]),
                    payload.get("crop_age_days", 0.0),
                    payload.get("stage", "unknown"),
                    payload.get("gdd", 0.0),
                    payload.get("stage_progress", 0.0),
                    json_dumps(payload.get("raw", {})),
                    payload.get("inference_ms", 0.0),
                    1 if payload.get("manual") else 0,
                ),
            )
            return int(cursor.lastrowid or 0)

    def latest(self, device_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self.db.cursor() as cursor:
            if device_id:
                cursor.execute(
                    "SELECT * FROM decisions WHERE device_id = ? "
                    "ORDER BY decided_at DESC, id DESC LIMIT ?",
                    (device_id, limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM decisions ORDER BY decided_at DESC, id DESC LIMIT ?", (limit,)
                )
            return [_decision_row(row) for row in cursor.fetchall()]

    def since(self, hours: int = 24) -> list[dict[str, Any]]:
        since = iso(utcnow() - timedelta(hours=hours))
        with self.db.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM decisions WHERE decided_at >= ? ORDER BY decided_at ASC, id ASC",
                (since,),
            )
            return [_decision_row(row) for row in cursor.fetchall()]

    def since_for_device(self, device_id: str, hours: int = 24) -> list[dict[str, Any]]:
        since = iso(utcnow() - timedelta(hours=hours))
        with self.db.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM decisions WHERE device_id = ? AND decided_at >= ? "
                "ORDER BY decided_at ASC, id ASC",
                (device_id, since),
            )
            return [_decision_row(row) for row in cursor.fetchall()]


class AlertRepository:
    """Operator alerts, with de-duplication so a stuck condition is not spammed."""

    #: An alert of the same kind for the same device is not re-raised within
    #: this window, so a probe reading a constant 0 pH does not bury the operator.
    DEDUPE_WINDOW_MINUTES = 60

    def __init__(self, database: Database) -> None:
        self.db = database

    def raise_alert(
        self, device_id: str, zone: str, kind: str, severity: str, message: str
    ) -> dict[str, Any] | None:
        since = iso(utcnow() - timedelta(minutes=self.DEDUPE_WINDOW_MINUTES))
        with self.db.transaction() as cursor:
            cursor.execute(
                "SELECT id FROM alerts WHERE device_id = ? AND kind = ? "
                "AND raised_at >= ? AND acknowledged = 0 LIMIT 1",
                (device_id, kind, since),
            )
            if cursor.fetchone():
                return None
            cursor.execute(
                "INSERT INTO alerts (device_id, zone, raised_at, kind, severity, message) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (device_id, zone, iso(utcnow()), kind, severity, message),
            )
            alert_id = int(cursor.lastrowid or 0)
        return self.get(alert_id)

    def get(self, alert_id: int) -> dict[str, Any] | None:
        with self.db.cursor() as cursor:
            cursor.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,))
            row = cursor.fetchone()
        return _alert_row(row) if row else None

    def list(self, open_only: bool = False, limit: int = 100) -> list[dict[str, Any]]:
        query = "SELECT * FROM alerts"
        params: list[Any] = []
        if open_only:
            query += " WHERE acknowledged = 0"
        query += " ORDER BY raised_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self.db.cursor() as cursor:
            cursor.execute(query, params)
            return [_alert_row(row) for row in cursor.fetchall()]

    def acknowledge(self, alert_id: int) -> dict[str, Any] | None:
        with self.db.transaction() as cursor:
            cursor.execute(
                "UPDATE alerts SET acknowledged = 1, resolved_at = ? WHERE id = ?",
                (iso(utcnow()), alert_id),
            )
        return self.get(alert_id)

    def open_count(self) -> int:
        with self.db.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS n FROM alerts WHERE acknowledged = 0")
            return int(cursor.fetchone()["n"])


class EventRepository:
    """The audit trail: every command, ingest and alert."""

    def __init__(self, database: Database) -> None:
        self.db = database

    def record(
        self, kind: str, detail: str, source: str = "api",
        device_id: str | None = None, zone: str | None = None,
    ) -> int:
        with self.db.transaction() as cursor:
            cursor.execute(
                "INSERT INTO events (device_id, zone, kind, detail, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (device_id, zone, kind, detail, source, iso(utcnow())),
            )
            return int(cursor.lastrowid or 0)

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.db.cursor() as cursor:
            cursor.execute("SELECT * FROM events ORDER BY created_at DESC, id DESC LIMIT ?", (limit,))
            return [
                {
                    "id": row["id"],
                    "deviceId": row["device_id"],
                    "zone": row["zone"],
                    "kind": row["kind"],
                    "detail": row["detail"],
                    "source": row["source"],
                    "createdAt": row["created_at"],
                }
                for row in cursor.fetchall()
            ]

    def count_since(self, hours: int = 24) -> int:
        since = iso(utcnow() - timedelta(hours=hours))
        with self.db.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS n FROM events WHERE created_at >= ?", (since,))
            return int(cursor.fetchone()["n"])


# --------------------------------------------------------------------------
# Row mapping
# --------------------------------------------------------------------------


def _device_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "zone": row["zone"],
        "crop": row["crop"],
        "soilType": row["soil_type"],
        "fieldAreaM2": row["field_area_m2"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "stationCode": row["station_code"],
        "sowingDate": row["sowing_date"],
        "firmware": row["firmware"],
        "linkState": row["link_state"],
        "seasonProgress": row["season_progress"],
        "pumpRateLpm": row["pump_rate_lpm"],
        "maxDepthMm": row["max_depth_mm"],
        "valveOpen": bool(row["valve_open"]),
        "valveOpenedAt": row["valve_opened_at"],
        "valveRuntimeSeconds": row["valve_runtime_s"],
        "lastSeen": row["last_seen"],
        "createdAt": row["created_at"],
    }


def _reading_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "deviceId": row["device_id"],
        "zone": row["zone"],
        "recordedAt": row["recorded_at"],
        "receivedAt": row["received_at"],
        "temperature": row["temperature"],
        "humidity": row["humidity"],
        "soilMoisture": row["soil_moisture"],
        "soilPh": row["soil_ph"],
        "ec": row["ec"],
        "nitrogen": row["nitrogen"],
        "phosphorus": row["phosphorus"],
        "potassium": row["potassium"],
        "lightIntensity": row["light_intensity"],
        "rainfall": row["rainfall"],
        "windSpeed": row["wind_speed"],
        "batteryVolts": row["battery_volts"],
        "rssiDbm": row["rssi_dbm"],
        "et0MmDay": row["et0_mm_day"],
        "cropAgeDays": row["crop_age_days"],
        "stage": row["stage"],
        "gdd": row["gdd"],
        "source": row["source"],
        "valveOpen": None if row["valve_open"] is None else bool(row["valve_open"]),
    }


def _decision_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "deviceId": row["device_id"],
        "zone": row["zone"],
        "decidedAt": row["decided_at"],
        "shouldIrrigate": bool(row["should_irrigate"]),
        "depthMm": row["depth_mm"],
        "volumeLitres": row["volume_litres"],
        "durationSeconds": row["duration_seconds"],
        "action": row["action"],
        "diseaseRisk": row["disease_risk"],
        "diseaseConfidence": row["disease_confidence"],
        "diseaseProbabilities": json_loads(row["disease_probabilities"], {}),
        "yieldTPerHa": row["yield_t_per_ha"],
        "nutrientDemandKgHa": json_loads(row["nutrient_demand"], {}),
        "et0MmDay": row["et0_mm_day"],
        "kc": row["kc"],
        "depletionFraction": row["depletion_fraction"],
        "cropStressIndex": row["crop_stress_index"],
        "moistureBand": row["moisture_band"],
        "riskLevel": row["risk_level"],
        "confidence": row["confidence"],
        "explanation": json_loads(row["explanation"], []),
        "cropAgeDays": row["crop_age_days"],
        "stage": row["stage"],
        "gdd": row["gdd"],
        "stageProgress": row["stage_progress"],
        "raw": json_loads(row["raw"], {}),
        "inferenceMs": row["inference_ms"],
        "manual": bool(row["manual"]),
    }


class CropHealthRepository:
    """
    Image analysis records, following the blueprint's minimum record.

    Rows carry a *reference* into controlled storage, never image bytes, and
    deleting a record returns the reference so the caller can delete the file
    too — the two deletes live in one code path so a record cannot outlive its
    image.
    """

    def __init__(self, database: Database) -> None:
        self.db = database

    def insert(self, record: dict[str, Any]) -> dict[str, Any]:
        import uuid

        analysis_id = record.get("analysis_id") or uuid.uuid4().hex[:16]
        with self.db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO crop_health_analyses (
                    analysis_id, created_at, crop, plant_part, growth_stage,
                    field_id, zone_id, image_reference, image_metadata,
                    quality, gates, task, prediction_label, raw_score,
                    calibrated, supported_classes, result_status,
                    status_reason, model_version, preprocessing_version,
                    validation_status, advice_code, advice_text,
                    advice_text_version, advice_follow_up,
                    advice_expert_referral, treatment, consent_given
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    record["created_at"],
                    record.get("crop") or "unknown",
                    record.get("plant_part"),
                    record.get("growth_stage"),
                    record.get("field_id"),
                    record.get("zone_id"),
                    record.get("image_reference"),
                    json_dumps(record.get("image_metadata") or {}),
                    json_dumps(record.get("quality") or {}),
                    json_dumps(record.get("gates") or []),
                    record.get("task"),
                    record.get("prediction_label"),
                    record.get("raw_score"),
                    1 if record.get("calibrated") else 0,
                    json_dumps(record.get("supported_classes") or []),
                    record.get("result_status") or "processing_error",
                    record.get("status_reason") or "",
                    record.get("model_version") or "unknown",
                    record.get("preprocessing_version") or "unknown",
                    record.get("validation_status") or "",
                    record.get("advice_code") or "service_error",
                    record.get("advice_text") or "",
                    record.get("advice_text_version"),
                    record.get("advice_follow_up"),
                    1 if record.get("advice_expert_referral") else 0,
                    json_dumps(record.get("treatment")) if record.get("treatment") else None,
                    1 if record.get("consent_given") else 0,
                ),
            )
        return self.get(analysis_id)  # type: ignore[return-value]

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        with self.db.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM crop_health_analyses WHERE analysis_id = ?",
                (analysis_id,),
            )
            row = cursor.fetchone()
        return _crop_health_row(row) if row else None

    def list(
        self,
        limit: int = 50,
        *,
        crop: str | None = None,
        status: str | None = None,
        field_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM crop_health_analyses"
        clauses: list[str] = []
        params: list[Any] = []
        if crop:
            clauses.append("crop = ?")
            params.append(crop)
        if status:
            clauses.append("result_status = ?")
            params.append(status)
        if field_id:
            clauses.append("field_id = ?")
            params.append(field_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self.db.cursor() as cursor:
            cursor.execute(query, params)
            return [_crop_health_row(row) for row in cursor.fetchall()]

    def set_feedback(self, analysis_id: str, feedback: str) -> dict[str, Any] | None:
        """
        Optional, consented farmer feedback. It is stored verbatim and never
        treated as ground truth by anything downstream.
        """
        with self.db.transaction() as cursor:
            cursor.execute(
                "UPDATE crop_health_analyses SET farmer_feedback = ? "
                "WHERE analysis_id = ?",
                (feedback, analysis_id),
            )
        return self.get(analysis_id)

    def delete(self, analysis_id: str) -> tuple[bool, str | None]:
        """
        Delete a record, returning (existed, image_reference) so the caller can
        also delete the stored file. Records whose image never arrived carry
        None.
        """
        row = self.get(analysis_id)
        if row is None:
            return False, None
        with self.db.transaction() as cursor:
            cursor.execute(
                "DELETE FROM crop_health_analyses WHERE analysis_id = ?",
                (analysis_id,),
            )
        return True, row.get("imageReference")


class FertilizerRepository:
    """Recommendations kept with their full derivation, for the audit trail."""

    def __init__(self, database: Database) -> None:
        self.db = database

    def insert(self, response: Any, request: Any) -> dict[str, Any]:
        payload = response.model_dump()
        payload["requested"] = request.model_dump()
        created = iso(utcnow())
        with self.db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO fertilizer_recommendations (
                    created_at, crop, growth_stage, area_value, area_unit,
                    area_hectares, status, product_name, total_product_kg,
                    per_ha_kg, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created,
                    payload["crop"],
                    payload.get("growthStage"),
                    float(payload["areaValue"]),
                    payload["areaUnit"],
                    float(payload["areaHectares"]),
                    payload["status"],
                    payload.get("productName"),
                    payload.get("totalProductForFieldKg"),
                    payload.get("applicationPerHaKg"),
                    json_dumps(payload),
                ),
            )
        record = dict(payload)
        record["createdAt"] = created
        return record


def _crop_health_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "analysisId": row["analysis_id"],
        "createdAt": row["created_at"],
        "crop": row["crop"],
        "plantPart": row["plant_part"],
        "growthStage": row["growth_stage"],
        "fieldId": row["field_id"],
        "zoneId": row["zone_id"],
        "imageReference": row["image_reference"],
        "imageMetadata": json_loads(row["image_metadata"], {}),
        "imageBytes": (json_loads(row["image_metadata"], {}) or {}).get("bytes", 0),
        "quality": json_loads(row["quality"], {}),
        "gates": json_loads(row["gates"], []),
        "task": row["task"],
        "prediction": {
            "task": row["task"],
            "classLabel": row["prediction_label"],
            "rawScore": row["raw_score"],
            "calibrated": bool(row["calibrated"]),
            "supportedClasses": json_loads(row["supported_classes"], []),
        },
        "resultStatus": row["result_status"],
        "statusReason": row["status_reason"],
        "modelVersion": row["model_version"],
        "preprocessingVersion": row["preprocessing_version"],
        "validationStatus": row["validation_status"],
        "advice": {
            "adviceCode": row["advice_code"],
            "adviceText": row["advice_text"],
            "textVersion": row["advice_text_version"],
            "followUpQuestion": row["advice_follow_up"],
            "expertReferral": bool(row["advice_expert_referral"]),
        },
        "treatment": json_loads(row["treatment"], None) if row["treatment"] else None,
        "consentGiven": bool(row["consent_given"]),
        "farmerFeedback": row["farmer_feedback"],
    }


def _alert_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "deviceId": row["device_id"],
        "zone": row["zone"],
        "raisedAt": row["raised_at"],
        "kind": row["kind"],
        "severity": row["severity"],
        "message": row["message"],
        "acknowledged": bool(row["acknowledged"]),
        "resolvedAt": row["resolved_at"],
    }
