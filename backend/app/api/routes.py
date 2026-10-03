"""
REST API for IRRIGO V2.

Endpoints are grouped by concern: `/api/system` for health and options,
`/api/fields` for the registry, `/api/plan` for planning and `/api/model` for
the evaluation artefacts. All ingest and control in V1 is gone because there is
nothing to ingest: a plan is computed from a registered sowing and station
weather, so the read paths are the product.

Every planning path can carry a `depletionMm`. When it is absent the plan is
computed at field capacity and the response says so via `depletionSource`, because
a plan that silently assumed a wet field would be confidently wrong.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ..core.config import Settings, get_settings
from ..schemas import (
    FieldCreate,
    GrowthOut,
    GrowthPoint,
    HealthOut,
    OptionsOut,
    PlanOut,
    field_to_dict,
    plan_to_dict,
)
from ..services.evaluation import EvaluationService
from ..services.plans import NotFound, PlanService, ServiceUnavailable

logger = logging.getLogger("iic.routes")

router = APIRouter()


# --------------------------------------------------------------------------
# dependencies
# --------------------------------------------------------------------------


def get_plans(request: Request) -> PlanService:
    service = getattr(request.app.state, "plans", None)
    if service is None or not service.ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "planning data is not loaded. Check the server log; the weather "
                "corpus or the sowing registry could not be read."
            ),
        )
    return service


def get_plans_optional(request: Request) -> PlanService | None:
    """
    Planning data, but without the 503.

    Health has to answer when the engine is not ready, otherwise the one endpoint
    that can explain the failure is the endpoint that fails.
    """
    return getattr(request.app.state, "plans", None)


def get_evaluation(request: Request) -> EvaluationService:
    return request.app.state.evaluation


def _parse_as_of(as_of: str | None) -> date | None:
    if not as_of:
        return None
    try:
        return date.fromisoformat(as_of)
    except ValueError as error:
        raise HTTPException(
            status_code=422, detail=f"asOf must be YYYY-MM-DD, got {as_of!r}"
        ) from error


# --------------------------------------------------------------------------
# system
# --------------------------------------------------------------------------


@router.get("/system/health", response_model=HealthOut)
def health(
    plans: PlanService | None = Depends(get_plans_optional),
    evaluation: EvaluationService = Depends(get_evaluation),
) -> HealthOut:
    """
    System health.

    Stays reachable when planning data is missing. That is the point: the
    dashboard uses this to tell the operator what is wrong instead of showing a
    connection error.
    """
    if plans is None:
        described: dict[str, Any] = {
            "available": False,
            "reason": "the plan service has not started",
            "records": 0,
            "stations": 0,
            "fields": 0,
            "horizonDays": 14,
        }
    else:
        described = plans.describe()
    reports = evaluation.describe()["reports"]
    available = bool(described.get("available"))
    settings = get_settings()
    evaluation_health = evaluation.model_health()
    return HealthOut(
        status=(
            "ok"
            if available and reports.get("evaluation")
            else ("degraded" if available else "unavailable")
        ),
        app=settings.app_name,
        version=settings.app_version,
        weatherAvailable=available,
        weatherReason=described.get("reason"),
        weatherRecords=int(described.get("records", 0)),
        weatherStations=int(described.get("stations", 0)),
        fields=int(described.get("fields", 0)),
        weatherStartYear=int(described.get("startYear")) if described.get("startYear") else None,
        weatherEndYear=int(described.get("endYear")) if described.get("endYear") else None,
        horizonDays=int(described.get("horizonDays", 14)),
        modelPresent=evaluation.describe()["modelPresent"],
        evaluationPresent=bool(reports.get("evaluation")),
        groupedR2=(
            evaluation_health.get("grouped", {}).get("r2")
            if evaluation_health.get("grouped")
            else None
        ),
        reports=reports,
    )


@router.get("/system/options", response_model=OptionsOut)
def options(plans: PlanService = Depends(get_plans)) -> OptionsOut:
    """Valid registry values, taken from the engine's own tables."""
    return OptionsOut(**plans.describe_options())


# --------------------------------------------------------------------------
# fields
# --------------------------------------------------------------------------


@router.get("/fields")
def list_fields(
    plans: PlanService = Depends(get_plans),
    as_of: str | None = Query(default=None, description="Plan date, YYYY-MM-DD"),
) -> dict[str, Any]:
    """
    Every registered field with the plan for one date.

    One call for the Overview grid, rather than N. Planning is memoised per
    (field, date, depletion), so this stays cheap, but keeping the overview to a
    single round trip matters more than the arithmetic saved.
    """
    when = _parse_as_of(as_of) or date.today()
    fields = []
    alerts: list[dict[str, Any]] = []
    for event in plans.list_fields():
        try:
            # Memoised: the overview asks for every field on every poll, and the
            # season summary is the expensive part of a plan. Keying the memo on
            # (field, date, depletion, with_season) keeps this identical across
            # calls until the registry or the corpus moves.
            plan = plans.plan_cached(
                event.field_id,
                when.isoformat(),
                0.0,
                with_season=True,
            )
        except Exception as error:
            logger.exception("plan failed for %s", event.field_id)
            fields.append({**field_to_dict(event), "planError": str(error)})
            continue
        payload = plan_to_dict(plan)
        schedule = payload.get("schedule") or {}
        fields.append({**field_to_dict(event), "plan": payload})
        if schedule.get("infeasible"):
            alerts.append(
                {
                    "fieldId": event.field_id,
                    "severity": "high",
                    "kind": "infeasible",
                    "title": f"{event.crop} at {event.station} cannot be scheduled",
                    "detail": (
                        f"{schedule.get('unmetRequirementMm', 0):.1f} mm of the "
                        f"requirement cannot be delivered within the method's limits."
                    ),
                }
            )
        if schedule.get("requirementInsufficient"):
            alerts.append(
                {
                    "fieldId": event.field_id,
                    "severity": "medium",
                    "kind": "requirement_insufficient",
                    "title": f"{event.field_id} requirement does not cover the horizon",
                    "detail": (
                        "The horizon demand exceeds what the soil can hold and "
                        "deliver, so stress is unavoidable with this configuration."
                    ),
                }
            )
        if plan.days_until_stress is not None and plan.days_until_stress <= 3:
            alerts.append(
                {
                    "fieldId": event.field_id,
                    "severity": "medium" if plan.days_until_stress > 0 else "high",
                    "kind": "stress_imminent",
                    "title": f"{event.field_id} reaches stress in {plan.days_until_stress} day(s)",
                    "detail": (
                        f"Depletion is at {plan.depletion_fraction:.0%} of readily "
                        f"available water with {plan.crop} at the {plan.stage.replace('_', ' ')} stage."
                    ),
                }
            )
    return {"asOf": when.isoformat(), "count": len(fields), "fields": fields, "alerts": alerts}


@router.post("/fields", status_code=201)
def register_field(
    body: FieldCreate,
    plans: PlanService = Depends(get_plans),
    replace: bool = Query(default=False),
) -> dict[str, Any]:
    """Register a sowing. Refuses a duplicate id unless `replace=true`."""
    from irrigation.sowing import SowingEvent

    event = SowingEvent(
        field_id=body.fieldId,
        station=body.station,
        crop=body.crop,
        sowing_date=body.sowingDate,
        soil_type=body.soilType or "Loam",
        method_name=body.methodName or "Sprinkler",
        field_area_m2=body.fieldAreaM2,
        mulched=body.mulched,
        nitrogen_regime=body.nitrogenRegime,
        notes=body.notes,
    )
    # The event resolves its tables on construction, so an unknown crop, soil or
    # method fails here with the engine's own message.
    existing = {f.field_id for f in plans.list_fields()}
    if body.fieldId in existing and not replace:
        raise HTTPException(
            status_code=409,
            detail=f"field {body.fieldId!r} already exists; pass replace=true to overwrite",
        )
    stored = plans.register(event, replace=replace)
    return field_to_dict(stored)


@router.get("/fields/{field_id}")
def get_field(field_id: str, plans: PlanService = Depends(get_plans)) -> dict[str, Any]:
    try:
        return field_to_dict(plans.get_field(field_id))
    except NotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.delete("/fields/{field_id}", status_code=204)
def delete_field(field_id: str, plans: PlanService = Depends(get_plans)) -> None:
    try:
        plans.remove(field_id)
    except NotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


# --------------------------------------------------------------------------
# plans
# --------------------------------------------------------------------------


@router.get("/plan/{field_id}", response_model=PlanOut)
def plan_field(
    field_id: str,
    plans: PlanService = Depends(get_plans),
    as_of: str | None = Query(default=None),
    depletion_mm: float | None = Query(
        default=None, ge=0, description="Measured depletion; omit to assume field capacity"
    ),
) -> dict[str, Any]:
    """
    The plan for one field.

    `depletionMm` is a measurement, not a tuning knob, which is why it is optional
    and defaults to a stated assumption rather than an optimistic guess.
    """
    try:
        plan = plans.plan_cached(
            field_id, (_parse_as_of(as_of) or date.today()).isoformat(), depletion_mm or 0.0
        )
    except NotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return plan_to_dict(plan)


@router.get("/plan/{field_id}/growth", response_model=GrowthOut)
def growth(
    field_id: str,
    plans: PlanService = Depends(get_plans),
    days: int = Query(default=180, ge=7, le=400),
    step: int = Query(default=2, ge=1, le=14),
    depletion_mm: float = Query(default=0.0, ge=0),
) -> dict[str, Any]:
    """
    The season as the engine walks it, for the growth chart.

    `depletionSource` is reported so the chart can be captioned honestly: a
    replay seeded from a measurement is a different claim from a replay seeded at
    field capacity.
    """
    try:
        event = plans.get_field(field_id)
    except NotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    points = plans.growth_series(event, days=days, step=step, depletion_mm=depletion_mm)
    return {
        "fieldId": event.field_id,
        "crop": event.crop,
        "station": event.station,
        "sowingDate": event.sowing_date,
        "depletionSource": "measured" if depletion_mm else "field-capacity-assumed",
        "points": points,
    }


# --------------------------------------------------------------------------
# model and validation
# --------------------------------------------------------------------------


@router.get("/model/evaluation")
def model_evaluation(evaluation: EvaluationService = Depends(get_evaluation)) -> dict[str, Any]:
    return evaluation.model_health()


@router.get("/model/external-validation")
def external_validation(evaluation: EvaluationService = Depends(get_evaluation)) -> Any:
    report = evaluation.external_validation()
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="reports/external_validation.json is missing. Run scripts/validate_against_observations.py.",
        )
    return report


@router.get("/model/soil-validation")
def soil_validation(evaluation: EvaluationService = Depends(get_evaluation)) -> Any:
    report = evaluation.soil_validation()
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="reports/soil_validation.json is missing. Run scripts/validate_soil_water.py.",
        )
    return report


@router.get("/model/comparison")
def comparison(evaluation: EvaluationService = Depends(get_evaluation)) -> Any:
    report = evaluation.comparison()
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="reports/v1_vs_v2.json is missing. Run scripts/compare_v1_v2.py.",
        )
    return report
