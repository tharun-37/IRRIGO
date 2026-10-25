"""
V1-compatible routes, mounted under `/api` alongside the V2 surface.

The copied dashboard talks to the endpoints V1's controller exposed. Rather than
rewrite the UI against V2's plan schema, these routes present the V2 engine in
V1's shape. They are additive: no V2 route is displaced, and the paths
(`/health`, `/zones`, `/telemetry`, `/analytics`, `/models`, `/alerts`,
`/events`) do not collide with `/system`, `/fields`, `/plan` or `/model`.

Control endpoints that V2 genuinely cannot honour — opening a valve, or a
free-form prediction — return 409 with an explanation instead of a hollow
success. The dashboard surfaces that message verbatim.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..core.config import get_settings
from ..core.database import get_database, parse_iso
from ..db.repository import ReadingRepository
from ..services.compat import CompatService
from ..services.ingest import IngestError, IngestService

router = APIRouter(tags=["v1-compat"])


def get_compat(request: Request) -> CompatService:
    service = getattr(request.app.state, "compat", None)
    if service is None:
        raise HTTPException(
            status_code=503, detail="the compatibility service has not started"
        )
    return service


def get_advisor(request: Request) -> Any:
    """
    The advisor surface is served by the fusion pipeline when it is up.

    The pipeline wraps the compat service and adds the V2 requirement-model
    estimate; when it is absent (or the model artifact is missing but the
    pipeline object still exists) it is still the right object to call, and it
    falls back to the V1 decision plus the plan internally.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is not None:
        return pipeline
    return get_compat(request)


# --------------------------------------------------------------------------
# system
# --------------------------------------------------------------------------


@router.get("/health")
def health(compat: CompatService = Depends(get_compat)) -> dict[str, Any]:
    return compat.health()


# --------------------------------------------------------------------------
# live telemetry (the V2 ingest pipeline)
# --------------------------------------------------------------------------

INGEST_SERVICE: IngestService | None = None


def get_ingest() -> IngestService:
    """
    The telemetry pipeline, built on first use.

    Every argument is a process-wide singleton — the sqlite database, the
    settings, the 2.6 MB decision bundle, the weather cache — so the lazy
    construction costs almost nothing, and this module stays importable even
    when those pieces are still warming.
    """
    global INGEST_SERVICE
    if INGEST_SERVICE is None:
        from ..core.ml_bridge import get_controller
        from ..services.weather import WeatherService

        INGEST_SERVICE = IngestService(
            database=get_database(),
            settings=get_settings(),
            controller=get_controller(),
            weather=WeatherService(),
        )
    return INGEST_SERVICE


@router.post("/telemetry")
def telemetry(request: Request, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    The node's ingest target: one reading in, the live decision out.

    V1 pushed telemetry over a WebSocket from a client running inside its own
    process; V2 turns the same contract into an HTTP POST so a bench node
    speaks plain HTTP. The response carries the whole decision — should we
    irrigate, at what depth, with what probability — so the node's Serial log
    is the same answer the dashboard will render.

    `device_api_key` in `.env` turns on a shared-secret check via the
    `X-Device-Key` header; empty (the default, a bench setting) disables it.
    """
    settings = get_settings()
    if settings.device_api_key and request.headers.get("x-device-key") != settings.device_api_key:
        raise HTTPException(status_code=401, detail="a valid X-Device-Key header is required")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="a JSON object body is required")
    try:
        return get_ingest().ingest(body)
    except IngestError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


#: How long a node's own reading stays authoritative for its field's `sensors`.
#: The bench posts every ~10 s; the window has to outlast a move or a couple of
#: dropped samples, not a full outage — past it the advisory falls back to the
#: plan-derived values and says so in its `live` block.
LIVE_SENSOR_WINDOW_SECONDS = 1800.0


def _overlay_live(advisory: dict[str, Any]) -> dict[str, Any]:
    """
    Attach the node's freshest real reading to one field's advisory.

    The advisor's `sensors` are derived from the field's plan replay — a
    documented approximation, not a measurement. When a device posts real
    telemetry for the field's zone, the honest answer is the measurement, so
    the quantities the node actually senses replace the plan-derived ones, and
    a `live` block says which reading did it and how old it is. Everything the
    node cannot sense (pH, EC, nutrients) keeps its derived value rather than
    going blank, because downstream visualisations consume every sensor slot.

    A stale node does not fake freshness: past the window its values are left
    as-derived and `live.isLive` turns false, with the last seen time recorded.
    """
    field_id = advisory.get("fieldId")
    if not field_id:
        return advisory
    try:
        rows = ReadingRepository(get_database()).latest(None, limit=200)
    except Exception:
        # Missing or unreadable telemetry store: answer from the plan, the
        # same as before this overlay existed.
        return advisory
    newest = next((row for row in rows if row.get("zone") == field_id), None)
    if newest is None:
        return advisory

    live = {
        "deviceId": newest.get("deviceId"),
        "source": "real-sensor",
        "isLive": False,
    }
    if newest.get("recordedAt"):
        live["lastSeenAt"] = newest["recordedAt"]
        try:
            age = (datetime.now(timezone.utc) - parse_iso(newest["recordedAt"])).total_seconds()
            live["ageSeconds"] = int(max(0.0, age))
            live["isLive"] = 0.0 <= age <= LIVE_SENSOR_WINDOW_SECONDS
        except (TypeError, ValueError):
            pass

    part_sensors = {"temperature", "humidity", "soilMoisture", "lightIntensity"}
    sensors = advisory.setdefault("sensors", {})
    for key in part_sensors:
        value = newest.get(key)
        if live["isLive"] and isinstance(value, (int, float)):
            sensors[key] = float(value)
    if live["isLive"] and newest.get("rssiDbm") is not None:
        sensors["rssiDbm"] = int(newest["rssiDbm"])
    # The node's own valve state, when it reports one, rides in the live block.
    # It is the closed loop seen from the other end: what the actuator is doing
    # right now, stated by the node rather than assumed by the planner.
    if live["isLive"] and newest.get("valveOpen") is not None:
        live["valveOpen"] = bool(newest["valveOpen"])
    advisory["live"] = live
    return advisory


def _overlay_live_all(payload: dict[str, Any]) -> dict[str, Any]:
    """Overlay applied across the whole feed, tolerant of payload changes."""
    for item in payload.get("fields") or []:
        _overlay_live(item)
    return payload


# --------------------------------------------------------------------------
# devices and zones
# --------------------------------------------------------------------------


@router.get("/devices")
def devices(compat: CompatService = Depends(get_compat)) -> dict[str, Any]:
    return compat.devices()


@router.get("/devices/{device_id}")
def device(device_id: str, compat: CompatService = Depends(get_compat)) -> dict[str, Any]:
    try:
        return compat.device(device_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"no device {device_id!r}") from error


@router.get("/zones")
def zones(compat: CompatService = Depends(get_compat)) -> dict[str, Any]:
    return compat.zones()


@router.post("/devices/{device_id}/valve")
def valve(
    device_id: str,
    body: dict[str, Any] | None = None,
    compat: CompatService = Depends(get_compat),
) -> dict[str, Any]:
    try:
        compat.device(device_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"no device {device_id!r}") from error
    raise HTTPException(
        status_code=409,
        detail=(
            "V2 is a planner, not a controller: it has no valve to open. Use the "
            "engine's schedule on the plan endpoint to decide when to irrigate."
        ),
    )


@router.post("/devices/{device_id}/recommendation")
def recommendation(
    device_id: str, compat: CompatService = Depends(get_compat)
) -> dict[str, Any]:
    try:
        return compat.recommendation(device_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"no device {device_id!r}") from error


# --------------------------------------------------------------------------
# telemetry
# --------------------------------------------------------------------------


@router.get("/telemetry/latest")
def latest(
    device_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    compat: CompatService = Depends(get_compat),
) -> dict[str, Any]:
    return compat.latest(device_id, limit)


@router.get("/telemetry/history")
def history(
    device_id: str = Query(...),
    hours: int = Query(default=24, ge=1, le=8760),
    limit: int = Query(default=500, ge=1, le=5000),
    compat: CompatService = Depends(get_compat),
) -> dict[str, Any]:
    try:
        return compat.history(device_id, hours, limit)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"no device {device_id!r}") from error


# --------------------------------------------------------------------------
# analytics and models
# --------------------------------------------------------------------------


@router.get("/analytics/overview")
def analytics(
    hours: int = Query(default=24, ge=1, le=8760),
    compat: CompatService = Depends(get_compat),
) -> dict[str, Any]:
    return compat.analytics(hours)


@router.get("/models/drift")
def drift(
    hours: int = Query(default=168, ge=1, le=8760),
    compat: CompatService = Depends(get_compat),
) -> dict[str, Any]:
    return compat.drift(hours)


# --------------------------------------------------------------------------
# alerts and events
# --------------------------------------------------------------------------


@router.get("/alerts")
def alerts(
    open_only: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=500),
    compat: CompatService = Depends(get_compat),
) -> dict[str, Any]:
    return compat.alerts(open_only, limit)


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge(
    alert_id: int, compat: CompatService = Depends(get_compat)
) -> dict[str, Any]:
    try:
        return compat.acknowledge(alert_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"no alert {alert_id}") from error


@router.get("/events")
def events(
    limit: int = Query(default=100, ge=1, le=500),
    compat: CompatService = Depends(get_compat),
) -> dict[str, Any]:
    return compat.events(limit)


# --------------------------------------------------------------------------
# advisor
# --------------------------------------------------------------------------


@router.get("/advisor")
def advisor(
    service: Any = Depends(get_advisor),
) -> dict[str, Any]:
    """Ranked, plain-language irrigation advice for every field, fusing the V1
    decision model with the V2 requirement model."""
    return _overlay_live_all(service.advisories())


@router.get("/advisor/{field_id}")
def advisor_field(
    field_id: str, service: Any = Depends(get_advisor)
) -> dict[str, Any]:
    try:
        return _overlay_live(service.advisory(field_id))
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"no field {field_id!r}") from error


# --------------------------------------------------------------------------
# out-of-band prediction
# --------------------------------------------------------------------------


@router.post("/predict")
def predict(body: dict[str, Any] | None = None) -> dict[str, Any]:
    raise HTTPException(
        status_code=409,
        detail=(
            "V2 does not accept arbitrary sensor payloads. Register a sowing and "
            "read its decision from /devices/{id}/recommendation instead."
        ),
    )
