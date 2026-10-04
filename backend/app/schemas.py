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

# The engine's own reference tables. They are the single source of truth for what a
# field may be, and importing them here means the API validates a sowing against the
# same catalogue the plan will later be computed from. Validating against a second,
# hand-kept list here would let the two drift, and the drift would only surface as a
# 500 at the first irrigation rather than as a rejected registration.
#
# `app.main` puts the `src/` tree on the import path before this package is reached,
# so the import resolves the same way it does for the service layer.
from irrigation.data.crops import CROPS, crop as lookup_crop
from irrigation.data.cultivation import METHODS, method as lookup_method
from irrigation.data.soil import TEXTURES, texture as lookup_texture


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
    """
    A sowing to register.

    Every reference value is resolved against the engine's catalogue at validation
    time rather than at plan time. The alternative - accepting a free string and
    letting `SowingEvent` refuse it - turns a typo into a 500 with a stack trace,
    and does it after the operator believes the field was saved. Refusing here costs
    nothing: the error names the field, the offending value and the full set of
    accepted values, which is everything a client needs to render the correction
    without a second round trip.
    """

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
    #: Named variety. A short-season hybrid finishes weeks earlier, so a variety is
    #: a different water plan rather than a cosmetic label.
    cropVariant: str | None = None
    #: Depth of a hardpan or gravel, metres. Caps what the roots can reach.
    restrictingDepthM: float | None = Field(default=None, gt=0)
    #: Fraction of the crop considered emerged. Below 1.0 the crop is partly below
    #: ground and transpires less than the curve assumes.
    emergenceFraction: float | None = Field(default=None, gt=0, le=1)

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

    @field_validator("crop")
    @classmethod
    def _known_crop(cls, value: str) -> str:
        try:
            lookup_crop(value)
        except KeyError as error:
            raise ValueError(str(error)) from error
        return value

    @field_validator("soilType")
    @classmethod
    def _known_soil(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            lookup_texture(value)
        except KeyError as error:
            raise ValueError(str(error)) from error
        return value

    @field_validator("methodName")
    @classmethod
    def _known_method(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            lookup_method(value)
        except KeyError as error:
            raise ValueError(str(error)) from error
        return value

    @field_validator("cropVariant")
    @classmethod
    def _known_variant(cls, value: str | None, info) -> str | None:
        # `crop` is declared first, so its validated value is already available.
        if value is None:
            return None
        crop_name = info.data.get("crop")
        if not crop_name:
            raise ValueError("cropVariant requires a crop")
        try:
            lookup_crop(crop_name, value)
        except KeyError as error:
            raise ValueError(str(error)) from error
        return value


class CropOptionOut(BaseModel):
    """
    One crop as the interface needs it: the parameters that change the plan, and
    the methods that suit it.

    The interface has to be able to say *why* a crop behaves differently from the one
    beside it. Four numbers do that work: the crop coefficient the season peaks at,
    the rooting depth, the depletion threshold that sets when irrigation triggers,
    and the length of the season. Sending them means an operator can read a 13 mm
    recommendation for barley and a 4 mm one for a young paddy, and see that the
    difference is the crop rather than a mistake.
    """

    name: str
    kcInitial: float
    kcMid: float
    kcEnd: float
    rootDepthMaxM: float
    depletionFractionP: float
    optimalPh: list[float]
    gddBaseTempC: float
    #: GDD at the end of each FAO-56 stage, and the season total.
    gddStageEnds: list[float]
    seasonGdd: float
    #: Calendar length of each FAO-56 stage in days, and the season total.
    lengthStageDays: list[int]
    seasonDays: int
    #: FAO-33 yield response to water and the season's potential yield. What the
    #: water is actually for, which a depth in millimetres does not say.
    ky: float
    yieldPotentialTHa: float
    #: Named varieties, which each change the season length.
    variants: list[str]
    #: Methods that suit this crop, most efficient first.
    suitableMethods: list[str]
    #: Methods that work but change what the recommendation means, with the reason.
    warnings: dict[str, str]


class CropCatalogueOut(BaseModel):
    crops: list[CropOptionOut]


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
        "cropVariant": event.crop_variant,
        "sowingDate": event.sowing_date,
        "soilType": event.soil_type,
        "methodName": event.method_name,
        "fieldAreaM2": event.field_area_m2,
        "mulched": event.mulched,
        "nitrogenRegime": event.nitrogen_regime,
        "restrictingDepthM": event.restricting_depth_m,
        "emergenceFraction": event.emergence_fraction,
        "daysSinceSowing": event.days_since_sowing(today),
        "seasonDay": (today - event.sown_on).days + 1,
        "notes": event.notes,
    }


def crop_option_to_dict(name: str) -> dict[str, Any]:
    """
    One crop's catalogue entry, assembled from the engine's own parameters.

    The method pairing is computed rather than tabulated, so it cannot fall out of
    step with `crop_and_method_are_compatible`: the reason a pairing is flagged lives
    in exactly one place and this reports it verbatim.

    Suitable methods are ordered by application efficiency, which is what makes the
    form's default a defensible choice rather than an arbitrary one. It also gets
    the two extreme cases right without a per-crop table: rice's only suitable method
    is the ponded one, because the engine flags every alternative, and a shallow-
    rooted crop is offered the frequent light applications its roots can use.
    """
    from irrigation.data.cultivation import crop_and_method_are_compatible

    parameters = CROPS[name]
    suitable: list[str] = []
    warnings: dict[str, str] = {}
    for method_name in sorted(METHODS, key=lambda m: -METHODS[m].application_efficiency):
        ok, reason = crop_and_method_are_compatible(parameters, METHODS[method_name])
        if ok:
            suitable.append(method_name)
        elif reason:
            warnings[method_name] = reason
    stage_days = [
        parameters.length_initial_days,
        parameters.length_development_days,
        parameters.length_mid_season_days,
        parameters.length_late_season_days,
    ]
    return {
        "name": name,
        "kcInitial": parameters.kc_initial,
        "kcMid": parameters.kc_mid,
        "kcEnd": parameters.kc_end,
        "rootDepthMaxM": parameters.root_depth_max_m,
        "depletionFractionP": parameters.depletion_fraction_p,
        "optimalPh": list(parameters.optimal_ph),
        "gddBaseTempC": parameters.gdd_base_temp_c,
        "gddStageEnds": list(parameters.gdd_stage_ends),
        "seasonGdd": parameters.gdd_stage_ends[3],
        "lengthStageDays": stage_days,
        "seasonDays": sum(stage_days),
        "ky": parameters.ky,
        "yieldPotentialTHa": parameters.yield_potential_t_ha,
        "variants": sorted(parameters.variants),
        "suitableMethods": suitable,
        "warnings": warnings,
    }
