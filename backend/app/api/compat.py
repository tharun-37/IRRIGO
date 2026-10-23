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

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..services.compat import CompatService

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
    return service.advisories()


@router.get("/advisor/{field_id}")
def advisor_field(
    field_id: str, service: Any = Depends(get_advisor)
) -> dict[str, Any]:
    try:
        return service.advisory(field_id)
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
