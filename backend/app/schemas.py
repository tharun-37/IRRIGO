"""
Pydantic schemas for the API surface.

These exist to make the contract explicit and to give the generated OpenAPI
something honest to describe. The plan itself is a dataclass with ~30 fields, and
mirroring all of it here would mean two places to change for one field; instead
the plan is serialised through `plan_to_dict` below, which keeps the wire format
in one place and the type checking where it is cheap.

The camelCase wire names are deliberate. The TypeScript types in
`frontend/src/types.ts` are written by hand and this is the only thing keeping
them honest, so they match.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------
# system
# --------------------------------------------------------------------------


class HealthOut(BaseModel):
    status: str
    app: str
    version: str
    weatherAvailable: bool
    weatherReason: str | None = None
    weatherRecords: int
    weatherStations: int
    fields: int
    weatherStartYear: int | None = None
    weatherEndYear: int | None = None
    horizonDays: int
    modelPresent: bool
    evaluationPresent: bool
    groupedR2: float | None = None
    reports: dict[str, bool]


# --------------------------------------------------------------------------
# fields
# --------------------------------------------------------------------------


class FieldOut(BaseModel):
    fieldId: str
    station: str
    crop: str
    sowingDate: str
    soilType: str
    methodName: str
    fieldAreaM2: float
    mulched: bool
    nitrogenRegime: float
    daysSinceSowing: int
    seasonDay: int
    notes: str = ""


class FieldCreate(BaseModel):
    fieldId: str = Field(min_length=1, max_length=64)
    station: str
    crop: str
    sowingDate: str
    soilType: str | None = None
    methodName: str | None = None
    fieldAreaM2: float = Field(default=1000.0, gt=0)
    mulched: bool = False
    nitrogenRegime: float = Field(default=1.0, gt=0)
    notes: str = ""

    @field_validator("sowingDate")
    @classmethod
    def _date_shape(cls, value: str) -> str:
        from datetime import date

        try:
            date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"sowingDate must be YYYY-MM-DD, got {value!r}") from error
        return value

    @field_validator("station", "crop")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class OptionsOut(BaseModel):
    stations: list[str]
    crops: list[str]
    soils: list[str]
    methods: list[str]


# --------------------------------------------------------------------------
# plans
# --------------------------------------------------------------------------


class ScheduleDayOut(BaseModel):
    offset: int
    applyMm: float
    grossApplyMm: float
    depletionBeforeMm: float
    depletionAfterMm: float
    stressBefore: bool
    stressAfter: bool
    reason: str = ""


class ScheduleOut(BaseModel):
    events: list[ScheduleDayOut]
    totalNetMm: float
    totalGrossMm: float
    totalEvents: int
    firstEventDay: int | None
    unmetRequirementMm: float
    stressDays: int
    percolationMm: float
    infeasible: bool
    requirementInsufficient: bool
    notes: list[str] = []


class StageWindowOut(BaseModel):
    stage: str
    startDate: str
    endDate: str
    days: int
    meanKc: float
    meanEtcMmDay: float
    meanRainMmDay: float
    grossRequirementMm: float


class PlanOut(BaseModel):
    fieldId: str
    station: str
    crop: str
    asOf: str
    sowingDate: str
    daysSinceSowing: int
    seasonDay: str
    gddAccumulated: float
    gddToNextStage: float | None
    stage: str
    stageProgress: float
    emerged: bool
    rootDepthCm: float
    tawMm: float
    rawMm: float
    depletionMm: float
    depletionFraction: float
    daysUntilStress: int | None
    kc: float
    et0MmDay: float
    etcMmDay: float
    rainfallMmDay: float
    requirement: dict[str, Any]
    schedule: ScheduleOut | None = None
    stageWindows: list[StageWindowOut] = []
    seasonDays: int = 0
    seasonGrossRequirementMm: float = 0.0
    seasonRainfallMm: float = 0.0
    seasonGrossVolumeM3: float = 0.0
    dataSource: str = "nasa-power"


class GrowthPoint(BaseModel):
    day: int
    date: str
    stage: str
    gdd: float
    kc: float
    rootCm: float
    tawMm: float
    depletionMm: float
    depletionFraction: float
    needMm: float
    applyMm: float
    grossMm: float
    daysUntilStress: int | None
    requirementInsufficient: bool


class GrowthOut(BaseModel):
    fieldId: str
    crop: str
    station: str
    sowingDate: str
    depletionSource: str
    points: list[GrowthPoint]


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------


def _schedule_to_dict(schedule) -> dict[str, Any] | None:
    if schedule is None:
        return None
    return {
        "events": [
            {
                "offset": day.offset,
                "applyMm": day.apply_mm,
                "grossApplyMm": day.gross_apply_mm,
                "depletionBeforeMm": day.depletion_before_mm,
                "depletionAfterMm": day.depletion_after_mm,
                "stressBefore": day.stress_before,
                "stressAfter": day.stress_after,
                "reason": day.reason,
            }
            for day in schedule.events
        ],
        "totalNetMm": schedule.total_net_mm,
        "totalGrossMm": schedule.total_gross_mm,
        "totalEvents": schedule.total_events,
        "firstEventDay": schedule.first_event_day,
        "unmetRequirementMm": schedule.unmet_requirement_mm,
        "stressDays": schedule.stress_days,
        "percolationMm": schedule.percolation_mm,
        "infeasible": schedule.infeasible,
        "requirementInsufficient": schedule.requirement_insufficient,
        "notes": list(getattr(schedule, "notes", []) or []),
    }


def _requirement_to_dict(requirement) -> dict[str, Any]:
    """The requirement dataclass, flattened for the wire."""
    fields = {}
    for name in dir(requirement):
        if name.startswith("_"):
            continue
        value = getattr(requirement, name)
        if callable(value):
            continue
        fields[name] = value
    return fields


def plan_to_dict(plan) -> dict[str, Any]:
    """One place where a plan becomes JSON."""
    return {
        "fieldId": plan.field_id,
        "station": plan.station,
        "crop": plan.crop,
        "asOf": plan.as_of,
        "sowingDate": plan.sowing_date,
        "daysSinceSowing": plan.days_since_sowing,
        "seasonDay": plan.season_day,
        "gddAccumulated": plan.gdd_accumulated,
        "gddToNextStage": plan.gdd_to_next_stage,
        "stage": plan.stage,
        "stageProgress": plan.stage_progress,
        "emerged": plan.emerged,
        "rootDepthCm": plan.root_depth_cm,
        "tawMm": plan.taw_mm,
        "rawMm": plan.raw_mm,
        "depletionMm": plan.depletion_mm,
        "depletionFraction": plan.depletion_fraction,
        "daysUntilStress": plan.days_until_stress,
        "kc": plan.kc,
        "et0MmDay": plan.et0_mm_day,
        "etcMmDay": plan.etc_mm_day,
        "rainfallMmDay": plan.rainfall_mm_day,
        "requirement": _requirement_to_dict(plan.requirement),
        "schedule": _schedule_to_dict(plan.schedule),
        "stageWindows": [
            {
                "stage": window.stage,
                "startDate": window.start_date,
                "endDate": window.end_date,
                "days": window.days,
                "meanKc": window.mean_kc,
                "meanEtcMmDay": window.mean_etc_mm_day,
                "meanRainMmDay": window.mean_rain_mm_day,
                "grossRequirementMm": window.gross_requirement_mm,
            }
            for window in plan.stage_windows
        ],
        "seasonDays": plan.season_days,
        "seasonGrossRequirementMm": plan.season_gross_requirement_mm,
        "seasonRainfallMm": plan.season_rainfall_mm,
        "seasonGrossVolumeM3": plan.season_gross_volume_m3,
        "dataSource": plan.data_source,
    }


def field_to_dict(event, plan=None) -> dict[str, Any]:
    from datetime import date

    today = date.today()
    return {
        "fieldId": event.field_id,
        "station": event.station,
        "crop": event.crop,
        "sowingDate": event.sowing_date,
        "soilType": event.soil_type,
        "methodName": event.method_name,
        "fieldAreaM2": event.field_area_m2,
        "mulched": event.mulched,
        "nitrogenRegime": event.nitrogen_regime,
        "daysSinceSowing": event.days_since_sowing(today),
        "seasonDay": (today - event.sown_on).days + 1,
        "notes": event.notes,
    }
