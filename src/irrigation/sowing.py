r"""
Sowing registration and the water plan that follows from it.

A farmer's question is never "what is the crop coefficient". It is: I put seed in
the ground on this date, in this soil, at this station, using this method. How
long until it needs water, how much, and when do I come back?

This module answers that. It stores the sowing event, counts the days since it,
accumulates thermal time from the station's own temperature record, resolves the
growth stage, and hands the state to the FAO-56 requirement calculation that
already exists in `physics.nir_target`.

The date the seed went in is the single most consequential input and the one a
model is most often handed without. Two fields sown a fortnight apart are in
different growth stages, have different root systems, and need different water on
the same day. Everything downstream here is anchored to that date.

Forward-looking ET0 uses the station's own monthly climatology rather than the
realised future. Using observed future weather would leak information the farmer
does not have when the decision is made, and would make an optimistic model look
accurate and then behave badly in service. Historical weather is used for the
days that have already happened, because for those the answer is known.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .data.crops import CropParameters, crop as lookup_crop
from .data.cultivation import (
    IrrigationMethod,
    gross_up_mm,
    method as lookup_method,
)
from .data.soil import SoilTexture, texture as lookup_texture
from .phenology.gdd import (
    GrowingDegreeDays,
    GrowthStage,
    effective_root_depth_m,
    stage_from_gdd,
)
from .physics.climate import (
    RainClimatology,
    build_rain_climatology,
    expected_rain_horizon,
)
from .physics.kc_curve import build_breakpoints, kc_at_gdd
from .physics.nir_target import build_horizon, net_requirement
from .physics.water_balance import FieldConfiguration, advance_depletion, effective_rainfall_mm
from .optimizer import horizon_inputs as _horizon_inputs
from .optimizer import optimise as optimise_schedule

#: Days of forecast the plan looks ahead. Fourteen is roughly one irrigation
#: return period for most surface methods.
PLAN_HORIZON_DAYS = 14

#: A sowing is reported as "emerged" once thermal time passes this share of the
#: initial stage. FAO-56 does not publish an emergence GDD threshold, so this is a
#: stated assumption rather than a measured constant, and it is exposed as a
#: parameter rather than buried.
EMERGENCE_STAGE_FRACTION = 0.5


class SowingError(ValueError):
    """Raised when a sowing event is not internally consistent."""


@dataclass
class SowingEvent:
    """
    One sowing: what went in, where, when, and how the field is managed.

    Every field the application knows about starts here. The registry is the
    answer to "which fields exist", and the planner turns one of these records
    plus weather into a schedule.
    """

    field_id: str
    station: str
    crop: str
    sowing_date: str
    soil_type: str = "Loam"
    crop_variant: str | None = None
    method_name: str = "Sprinkler"
    field_area_m2: float = 1_000.0
    mulched: bool = False
    nitrogen_regime: float = 1.0
    restricting_depth_m: float | None = None
    emergence_fraction: float = EMERGENCE_STAGE_FRACTION
    notes: str = ""

    def __post_init__(self) -> None:
        try:
            self.sowing_date = str(
                datetime.strptime(self.sowing_date, "%Y-%m-%d").date().isoformat()
            )
        except (TypeError, ValueError) as exc:
            raise SowingError(
                f"{self.field_id}: sowing_date must be YYYY-MM-DD, got "
                f"{self.sowing_date!r}"
            ) from exc
        if self.field_area_m2 <= 0:
            raise SowingError(f"{self.field_id}: field_area_m2 must be positive")
        if not 0.0 < self.emergence_fraction <= 1.0:
            raise SowingError(
                f"{self.field_id}: emergence_fraction must be in (0, 1], "
                f"got {self.emergence_fraction}"
            )
        # Fail at registration rather than at the first irrigation, where a typo
        # would surface as a wrong recommendation instead of a bad name.
        lookup_crop(self.crop, self.crop_variant)
        lookup_texture(self.soil_type)
        lookup_method(self.method_name)

    # --- resolved parameters
    @property
    def crop_parameters(self) -> CropParameters:
        return lookup_crop(self.crop, self.crop_variant)

    @property
    def soil(self) -> SoilTexture:
        return lookup_texture(self.soil_type)

    @property
    def method(self) -> IrrigationMethod:
        return lookup_method(self.method_name)

    @property
    def sown_on(self) -> date:
        return datetime.strptime(self.sowing_date, "%Y-%m-%d").date()

    def days_since_sowing(self, as_of: date) -> int:
        """Calendar days elapsed. Day 0 is the sowing date itself."""
        return (as_of - self.sown_on).days


@dataclass
class SeasonStageWindow:
    """When one growth stage is expected to run, and how much water it costs."""

    stage: str
    start_date: str
    end_date: str
    days: int
    mean_kc: float
    mean_etc_mm_day: float
    mean_rain_mm_day: float
    gross_requirement_mm: float
    mean_daily_requirement_mm: float


@dataclass
class FieldPlan:
    """
    The water plan for one sowing as of one day.

    `days_since_sowing` and `stage` are the "where is this crop" answer.
    `requirement` is the "how much" answer and `schedule` is the "when and how
    deep" answer, which is a different problem with different constraints: a
    requirement of 60 mm on a basin is one event, and on drip it is three, and
    conflating the two is how a correct number becomes an instruction nobody can
    follow.
    """

    field_id: str
    station: str
    crop: str
    as_of: str
    sowing_date: str
    days_since_sowing: int
    season_day: str
    gdd_accumulated: float
    gdd_to_next_stage: float | None
    stage: str
    stage_progress: float
    emerged: bool
    root_depth_cm: float
    taw_mm: float
    raw_mm: float
    depletion_mm: float
    depletion_fraction: float
    days_until_stress: int | None
    kc: float
    et0_mm_day: float
    etc_mm_day: float
    rainfall_mm_day: float
    requirement: object
    schedule: object = None
    stage_windows: list[SeasonStageWindow] = field(default_factory=list)
    season_days: int = 0
    season_gross_requirement_mm: float = 0.0
    season_rainfall_mm: float = 0.0
    season_gross_volume_m3: float = 0.0
    data_source: str = "nasa-power"

    @property
    def next_irrigation_mm(self) -> float:
        return float(self.requirement.single_application_mm)

    @property
    def gross_application_mm(self) -> float:
        return float(self.requirement.gross_requirement_mm)

    def summary_lines(self) -> list[str]:
        """Plain-language lines for a terminal or a notification."""
        stage_label = self.stage.replace("_", " ")
        lines = [
            f"Field {self.field_id} - {self.crop} at {self.station}",
            f"  sown {self.sowing_date} | day {self.days_since_sowing} of season"
            f" ({self.season_day})",
            f"  stage: {stage_label}"
            + (" (emerged)" if self.emerged else " (pre-emergence)"),
            f"  thermal time {self.gdd_accumulated:.0f} GDD"
            + (
                f", {self.gdd_to_next_stage:.0f} to next stage"
                if self.gdd_to_next_stage is not None
                else ", season complete"
            ),
            f"  root {self.root_depth_cm:.0f} cm | TAW {self.taw_mm:.0f} mm"
            f" | RAW {self.raw_mm:.0f} mm",
            f"  soil now {self.depletion_mm:.0f} mm depleted"
            f" ({100 * self.depletion_fraction:.0f}% of TAW)",
            f"  today ET0 {self.et0_mm_day:.2f} mm, Kc {self.kc:.2f},"
            f" ETc {self.etc_mm_day:.2f} mm, rain {self.rainfall_mm_day:.2f} mm",
        ]
        if self.days_until_stress is None:
            lines.append("  forecast: no water deficit inside the horizon")
        else:
            lines.append(
                f"  forecast: reaches stress in about {self.days_until_stress} day(s)"
            )
        req = self.requirement
        if req.net_requirement_mm <= 0.01:
            lines.append("  ACTION: no irrigation needed in the next "
                         f"{PLAN_HORIZON_DAYS} days")
        else:
            lines.append(
                f"  ACTION: {req.net_requirement_mm:.1f} mm over"
                f" {PLAN_HORIZON_DAYS} days"
                f" -> apply {req.single_application_mm:.1f} mm"
                f" ({req.gross_requirement_mm:.1f} mm gross)"
            )
            if req.requires_multiple_applications:
                lines.append(
                    f"  split over {req.min_interval_days}-day intervals:"
                    " the method cannot place it all in one pass"
                )
        if self.schedule is not None and self.schedule.total_events > 0:
            lines.extend(self.schedule.lines())
        return lines


class SowingRegistry:
    """
    The set of fields the application knows about, persisted as JSON.

    Deliberately a plain file. The number of fields a farmer manages is tens, and
    a database would add a deployment burden to solve a problem that does not
    exist here. The file is the source of truth and can be version controlled,
    which for an application whose calibration is under review is worth more than
    concurrent writes.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._events: dict[str, SowingEvent] = {}

    def register(self, event: SowingEvent, replace: bool = False) -> SowingEvent:
        if event.field_id in self._events and not replace:
            raise SowingError(
                f"{event.field_id} is already registered;"
                " pass replace=True to overwrite"
            )
        self._events[event.field_id] = event
        return event

    def get(self, field_id: str) -> SowingEvent:
        try:
            return self._events[field_id]
        except KeyError:
            known = ", ".join(sorted(self._events)) or "none"
            raise SowingError(
                f"unknown field {field_id!r}; registered fields: {known}"
            ) from None

    def list(self) -> list[SowingEvent]:
        return sorted(self._events.values(), key=lambda e: (e.sowing_date, e.field_id))

    def remove(self, field_id: str) -> None:
        self.get(field_id)
        del self._events[field_id]

    def __len__(self) -> int:
        return len(self._events)

    def __contains__(self, field_id: object) -> bool:
        return field_id in self._events

    # --- persistence
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "fields": [asdict(event) for event in self.list()],
        }

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise SowingError("no path given and this registry has no default path")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "SowingRegistry":
        source = Path(path)
        if not source.exists():
            return cls(source)
        payload = json.loads(source.read_text(encoding="utf-8"))
        registry = cls(source)
        for record in payload.get("fields", []):
            registry.register(SowingEvent(**record), replace=True)
        return registry


class StationWeather:
    """
    Real weather for a station, with a climatological stand-in for the future.

    Two things need weather and they need different weather. Days that have
    already happened have a correct answer, so the observation is used. Days
    ahead do not, so the station's own long-run monthly mean is used. Mixing the
    two is what makes a plan either leak the future or refuse to plan at all.
    """

    def __init__(self, weather: pd.DataFrame, station: str) -> None:
        self.station = station
        frame = weather[weather["station"] == station].sort_values("date")
        if frame.empty:
            raise SowingError(f"no weather records for station {station!r}")
        self.frame = frame.reset_index(drop=True)
        self.frame["date"] = pd.to_datetime(self.frame["date"])
        self._by_date = {
            row.date.date(): index for index, row in self.frame.iterrows()
        }
        self.monthly_et0 = self.frame.groupby("month")["et0_mm_day"].mean()
        self.monthly_rain = self.frame.groupby("month")["rainfall_mm"].mean()
        self.monthly_tmax = self.frame.groupby("month")["tmax_c"].mean()
        self.monthly_tmin = self.frame.groupby("month")["tmin_c"].mean()
        self.default_et0 = float(self.frame["et0_mm_day"].mean())
        self.default_rain = float(self.frame["rainfall_mm"].mean())

    def observed(self, day: date) -> dict | None:
        index = self._by_date.get(day)
        if index is None:
            return None
        row = self.frame.iloc[index]
        return {
            "et0_mm": float(row["et0_mm_day"]),
            "tmax_c": float(row["tmax_c"]),
            "tmin_c": float(row["tmin_c"]),
            "rainfall_mm": float(row["rainfall_mm"]),
        }

    def climatology_for(self, day: date) -> dict:
        month = day.month
        return {
            "et0_mm": float(self.monthly_et0.get(month, self.default_et0)),
            "tmax_c": float(self.monthly_tmax.get(month, self.frame["tmax_c"].mean())),
            "tmin_c": float(self.monthly_tmin.get(month, self.frame["tmin_c"].mean())),
            "rainfall_mm": float(self.monthly_rain.get(month, self.default_rain)),
            "forecast": True,
        }

    def day(self, day: date) -> dict:
        """Observed where available, climatological otherwise."""
        return self.observed(day) or self.climatology_for(day)


def _gdd_series(station: StationWeather, crop: CropParameters, start: date, count: int) -> np.ndarray:
    accumulator = GrowingDegreeDays(crop.gdd_base_temp_c)
    out = np.zeros(count, dtype=float)
    for offset in range(count):
        entry = station.day(start + timedelta(days=offset))
        out[offset] = float(
            accumulator.daily_gdd(np.array([entry["tmax_c"]]), np.array([entry["tmin_c"]]))[0]
        )
    return out


def _elapsed_gdd(
    station: StationWeather, crop: CropParameters, sown: date, as_of: date
) -> float:
    days = (as_of - sown).days + 1
    if days <= 0:
        return 0.0
    return float(_gdd_series(station, crop, sown, days).sum())


def _emitted(stage_progress: float, emergence_fraction: float) -> bool:
    return stage_progress >= emergence_fraction


def _requirement_for(
    event: SowingEvent,
    station: StationWeather,
    climatology: RainClimatology | None,
    as_of: date,
    depletion_mm: float,
    season_progress_scale: float = 1.0,
):
    """
    The FAO-56 requirement over the planning horizon, from known state only.

    `depletion_mm` is what the field is at right now. Defaulting it to zero
    assumes field capacity, which is optimistic; a caller that knows the field is
    dry should say so, and the CLI accepts a probe reading for exactly that.
    """
    crop = event.crop_parameters
    soil = event.soil
    method = event.method
    configuration = FieldConfiguration(
        crop=crop,
        soil=soil,
        restricting_depth_m=event.restricting_depth_m,
        season_scale=season_progress_scale,
    )
    gdd_now = _elapsed_gdd(station, crop, event.sown_on, as_of)
    stage = stage_from_gdd(gdd_now, crop)

    # Past harvest there is no crop drawing water, so the requirement is zero by
    # definition. Left unhandled, the late-season tables keep producing a Kc and a
    # root depth for a field whose grain is already off, and the plan recommended
    # 29 mm of irrigation to a harvested wheat field.
    if stage.stage == GrowthStage.POST_HARVEST:
        empty = np.zeros(PLAN_HORIZON_DAYS, dtype=float)
        return net_requirement(
            build_horizon(empty, empty, empty, empty, gdd_now, crop,
                          season_scale=season_progress_scale),
            configuration,
            configuration.water_state(stage, depletion_mm),
            method,
            method.application_efficiency,
            float(gdd_now),
        )

    state = configuration.water_state(stage, depletion_mm)

    horizon_start = as_of + timedelta(days=1)
    et0, rain, probability = _forward_weather(station, climatology, horizon_start, PLAN_HORIZON_DAYS)
    gdd_forward = _gdd_series(station, crop, horizon_start, PLAN_HORIZON_DAYS)

    horizon = build_horizon(
        et0, rain, probability, gdd_forward, gdd_now, crop,
        season_scale=season_progress_scale,
    )
    return net_requirement(
        horizon,
        configuration,
        state,
        method,
        method.application_efficiency,
        float(gdd_now + gdd_forward.sum()),
    )


def _forward_weather(
    station: StationWeather,
    climatology: RainClimatology | None,
    start: date,
    days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expected ET0, expected effective rain and rain probability over the horizon.

    The rain probability is the part that is easy to get wrong. Crediting the
    full expected rain depth on every day of the window is equivalent to
    assuming it rains every single day for two weeks, which over-credits a dry
    season and under-irrigates. Instead the expected depth is spread over the
    days on which rain of that kind is actually likely, so a 30 mm monthly mean
    arriving in two events is credited as two 15 mm events and not as 0.4 mm on
    each of thirty days.
    """
    et0 = np.zeros(days, dtype=float)
    rain = np.zeros(days, dtype=float)
    probability = np.zeros(days, dtype=float)
    for offset in range(days):
        day = start + timedelta(days=offset)
        et0[offset] = station.day(day)["et0_mm"]
        if climatology is not None:
            window = climatology.expected_rain_over_window_mm(day.month, 1)
            chance = climatology.probability_of_useful_rain(day.month, 1)
            rain[offset] = float(window)
            probability[offset] = float(chance)
    return et0, rain, probability


def _season_windows(
    event: SowingEvent,
    station: StationWeather,
    climatology: RainClimatology | None,
    depletion_start_mm: float = 0.0,
    max_days: int = 400,
) -> tuple[list[SeasonStageWindow], int, float, float, float]:
    """
    Walk the whole season forward and cost each stage.

    The stage boundaries are resolved in calendar dates by accumulating GDD
    until each threshold is crossed, so the output is a calendar a farmer can
    act on rather than four GDD numbers. Water is costed stage by stage using the
    same daily balance the planner uses, which means the season total is the sum
    of the parts the farmer will be told about on any given day.
    """
    crop = event.crop_parameters
    soil = event.soil
    method = event.method
    configuration = FieldConfiguration(
        crop=crop, soil=soil, restricting_depth_m=event.restricting_depth_m
    )
    breakpoints = build_breakpoints(crop)

    depletion = float(depletion_start_mm)
    applied_total = 0.0
    rain_total = 0.0
    windows: list[SeasonStageWindow] = []

    current_stage: str | None = None
    stage_start = event.sown_on
    stage_days = 0
    stage_kc = 0.0
    stage_etc = 0.0
    stage_rain = 0.0
    stage_gross = 0.0

    day = event.sown_on
    gdd_now = 0.0
    last_irrigation = -10_000
    trigger_fraction = 0.55

    for offset in range(max_days):
        entry = station.day(day)
        accumulator = GrowingDegreeDays(crop.gdd_base_temp_c)
        gdd_now += float(
            accumulator.daily_gdd(
                np.array([entry["tmax_c"]]), np.array([entry["tmin_c"]])
            )[0]
        )
        if gdd_now >= float(crop.gdd_stage_ends[3]):
            break

        stage_state = stage_from_gdd(gdd_now, crop)
        stage_name = stage_state.stage.value
        root = configuration.root_depth_m(stage_state)
        water = configuration.water_state(stage_state, depletion)
        taw = water.total_available_water_mm
        raw = min(taw, water.readily_available_water_mm)

        kc = kc_at_gdd(gdd_now, breakpoints)
        etc = kc * entry["et0_mm"]
        pe = effective_rainfall_mm(entry["rainfall_mm"], soil, depletion, taw)

        applied = 0.0
        if depletion >= trigger_fraction * raw and offset - last_irrigation >= method.min_interval_days:
            applied = float(
                max(0.0, min(min(taw - depletion, method.max_depth_per_event_mm), taw))
            )
            last_irrigation = offset

        depletion = advance_depletion(depletion, etc, pe, applied, taw)

        gross = gross_up_mm(max(0.0, etc - pe), method)
        applied_total += applied
        rain_total += entry["rainfall_mm"]

        if stage_name != current_stage:
            if current_stage is not None:
                windows.append(
                    SeasonStageWindow(
                        stage=current_stage,
                        start_date=stage_start.isoformat(),
                        end_date=day.isoformat(),
                        days=stage_days,
                        mean_kc=stage_kc / max(1, stage_days),
                        mean_etc_mm_day=stage_etc / max(1, stage_days),
                        mean_rain_mm_day=stage_rain / max(1, stage_days),
                        gross_requirement_mm=stage_gross,
                        mean_daily_requirement_mm=stage_gross / max(1, stage_days),
                    )
                )
            current_stage = stage_name
            stage_start = day
            stage_days = 0
            stage_kc = 0.0
            stage_etc = 0.0
            stage_rain = 0.0
            stage_gross = 0.0

        stage_days += 1
        stage_kc += kc
        stage_etc += etc
        stage_rain += entry["rainfall_mm"]
        stage_gross += gross
        day += timedelta(days=1)

    if current_stage is not None:
        windows.append(
            SeasonStageWindow(
                stage=current_stage,
                start_date=stage_start.isoformat(),
                end_date=day.isoformat(),
                days=stage_days,
                mean_kc=stage_kc / max(1, stage_days),
                mean_etc_mm_day=stage_etc / max(1, stage_days),
                mean_rain_mm_day=stage_rain / max(1, stage_days),
                gross_requirement_mm=stage_gross,
                mean_daily_requirement_mm=stage_gross / max(1, stage_days),
            )
        )

    total_days = sum(w.days for w in windows)
    total_gross = sum(w.gross_requirement_mm for w in windows)
    # `total_gross` is already a depth in mm over the field. Dividing it by the
    # area, as an earlier version did, is dimensionally meaningless and produced
    # 72,657 mm for a hectare. The area is needed for a volume, not a depth.
    volume_m3 = total_gross * max(0.0, event.field_area_m2) / 1000.0
    return windows, total_days, total_gross, rain_total, volume_m3


def season_depletion(
    event: SowingEvent,
    station: StationWeather,
    climatology: RainClimatology | None = None,
    as_of: date | None = None,
    max_days: int = 400,
    trigger_fraction: float = 0.55,
) -> list[tuple[date, float]]:
    """
    Soil depletion day by day, from sowing, under the operator's own rule.

    This exists because a plan needs an honest soil state, and the only honest
    soil state for day 95 is the one the crop actually reached by day 95. Asking
    for a plan while assuming the field is at field capacity on every single day
    produces a confident and completely wrong answer of "no irrigation needed"
    for a crop in mid-season at 4 mm of ET0, because a full 360 mm profile really
    does contain fourteen days of demand. The profile is not full on day 95. It
    has been drawn down since sowing, and this is what tracks that.

    Uses the same rule the plan uses: irrigate when depletion passes a fraction
    of readily available water, apply what the method can place in one pass, and
    wait the method's minimum interval. The point is a self-consistent state, not
    an optimal one; the optimiser is what decides the best schedule.
    """
    crop = event.crop_parameters
    soil = event.soil
    method = event.method
    configuration = FieldConfiguration(
        crop=crop, soil=soil, restricting_depth_m=event.restricting_depth_m
    )
    breakpoints = build_breakpoints(crop)
    accumulator = GrowingDegreeDays(crop.gdd_base_temp_c)

    start = event.sown_on
    stop = as_of or (event.sown_on + timedelta(days=max_days))
    depletion = 0.0
    gdd_now = 0.0
    last_irrigation = -10_000
    out: list[tuple[date, float]] = []

    day = start
    while day <= stop:
        entry = station.day(day)
        gdd_now += float(
            accumulator.daily_gdd(
                np.array([entry["tmax_c"]]), np.array([entry["tmin_c"]])
            )[0]
        )
        if gdd_now >= float(crop.gdd_stage_ends[3]):
            out.append((day, depletion))
            break

        stage_state = stage_from_gdd(gdd_now, crop)
        water = configuration.water_state(stage_state, depletion)
        taw = water.total_available_water_mm
        raw = min(taw, water.readily_available_water_mm)

        kc = kc_at_gdd(gdd_now, breakpoints)
        etc = kc * entry["et0_mm"]
        pe = effective_rainfall_mm(entry["rainfall_mm"], soil, depletion, taw)

        applied = 0.0
        if (
            depletion >= trigger_fraction * raw
            and (day - start).days - last_irrigation >= method.min_interval_days
        ):
            applied = float(max(0.0, min(taw - depletion, method.max_depth_per_event_mm)))
            last_irrigation = (day - start).days

        depletion = advance_depletion(depletion, etc, pe, applied, taw)
        out.append((day, depletion))
        day += timedelta(days=1)
    return out


def _first_stress_day(
    event: SowingEvent,
    station: StationWeather,
    climatology: RainClimatology | None,
    as_of: date,
    depletion_mm: float,
    season_progress_scale: float = 1.0,
) -> int | None:
    """Days until the soil reaches the FAO-56 stress point, or None if it does not."""
    crop = event.crop_parameters
    soil = event.soil
    configuration = FieldConfiguration(
        crop=crop, soil=soil,
        restricting_depth_m=event.restricting_depth_m,
        season_scale=season_progress_scale,
    )
    gdd_now = _elapsed_gdd(station, crop, event.sown_on, as_of)
    stage = stage_from_gdd(gdd_now, crop)
    # Nothing is drawing water past harvest, so the soil never reaches stress.
    if stage.stage == GrowthStage.POST_HARVEST:
        return None
    water = configuration.water_state(stage, depletion_mm)
    taw = water.total_available_water_mm
    raw = min(taw, water.readily_available_water_mm)

    et0, rain, probability = _forward_weather(
        station, climatology, as_of + timedelta(days=1), PLAN_HORIZON_DAYS
    )
    gdd_forward = _gdd_series(station, crop, as_of + timedelta(days=1), PLAN_HORIZON_DAYS)
    breakpoints = build_breakpoints(crop)
    gdd = gdd_now
    depletion = float(depletion_mm)

    for offset in range(PLAN_HORIZON_DAYS):
        gdd += float(gdd_forward[offset])
        stage_state = stage_from_gdd(gdd, crop)
        w = configuration.water_state(stage_state, depletion)
        taw_d = w.total_available_water_mm
        raw_d = min(taw_d, w.readily_available_water_mm)
        kc = kc_at_gdd(gdd, breakpoints)
        etc = kc * et0[offset]
        pe = effective_rainfall_mm(rain[offset] * probability[offset], soil, depletion, taw_d)
        depletion = float(np.clip(depletion + etc - pe, 0.0, taw_d))
        if depletion > raw_d:
            return offset + 1
    return None


def _schedule_for(
    event: SowingEvent,
    station: StationWeather,
    climatology: RainClimatology | None,
    as_of: date,
    depletion_mm: float,
    gdd_now: float,
    taw: float,
    raw: float,
):
    """
    Turn today's requirement into a dated schedule.

    The stress trigger is passed at 1.0 of the allowable depletion, which is the
    FAO-56 stress point rather than the earlier trigger a human operator usually
    acts on. The optimiser plans so that the crop does not cross stress inside the
    horizon; a farmer who would rather irrigate earlier than that is expressing a
    risk preference, not a physics error, and it belongs in the caller's hands
    rather than hard-coded here.
    """
    crop = event.crop_parameters
    et0, rain, probability = _forward_weather(
        station, climatology, as_of + timedelta(days=1), PLAN_HORIZON_DAYS
    )
    gdd_forward = _gdd_series(station, crop, as_of + timedelta(days=1), PLAN_HORIZON_DAYS)
    demand, effective_rain, _ = _horizon_inputs(
        et0, rain, probability, gdd_forward, gdd_now, crop, taw, raw
    )
    requirement = _requirement_for(event, station, climatology, as_of, depletion_mm)
    return optimise_schedule(
        requirement_mm=requirement.net_requirement_mm,
        demand=demand,
        effective_rain=effective_rain,
        depletion_now=depletion_mm,
        taw=taw,
        raw_mm=raw,
        soil=event.soil,
        method=event.method,
    )


def plan_field(
    event: SowingEvent,
    station: StationWeather,
    climatology: RainClimatology | None = None,
    as_of: date | None = None,
    depletion_mm: float = 0.0,
    with_season: bool = True,
) -> FieldPlan:
    """
    Produce the water plan for one sowing as of one day.

    The default `depletion_mm` of zero is a fresh, full soil profile. It is the
    right default for a sowing date, because on the day of sowing the profile
    *is* at field capacity, and it is the wrong default for a field check three
    weeks later, which is why it is a parameter rather than a constant.
    """
    crop = event.crop_parameters
    as_of = as_of or event.sown_on
    if as_of < event.sown_on:
        raise SowingError(
            f"{event.field_id}: as_of {as_of} precedes sowing {event.sown_on}"
        )

    gdd_now = _elapsed_gdd(station, crop, event.sown_on, as_of)
    stage = stage_from_gdd(gdd_now, crop)
    configuration = FieldConfiguration(
        crop=crop, soil=event.soil, restricting_depth_m=event.restricting_depth_m
    )
    root = configuration.root_depth_m(stage)
    water = configuration.water_state(stage, depletion_mm)
    taw = water.total_available_water_mm
    raw = min(taw, water.readily_available_water_mm)

    today = station.day(as_of)
    kc = kc_at_gdd(gdd_now, build_breakpoints(crop))
    etc = kc * today["et0_mm"]

    requirement = _requirement_for(
        event, station, climatology, as_of, depletion_mm
    )
    stress_day = _first_stress_day(
        event, station, climatology, as_of, depletion_mm
    )

    windows: list[SeasonStageWindow] = []
    season_days = 0
    season_gross = 0.0
    season_rain = 0.0
    season_volume = 0.0
    if with_season:
        windows, season_days, season_gross, season_rain, season_volume = _season_windows(
            event, station, climatology, depletion_start_mm=0.0
        )

    schedule = _schedule_for(event, station, climatology, as_of, depletion_mm, gdd_now,
                             taw, raw)

    emerged = stage.stage == GrowthStage.INITIAL and (
        stage.season_progress >= event.emergence_fraction
    ) or stage.stage != GrowthStage.INITIAL

    return FieldPlan(
        field_id=event.field_id,
        station=event.station,
        crop=event.crop,
        as_of=as_of.isoformat(),
        sowing_date=event.sowing_date,
        days_since_sowing=event.days_since_sowing(as_of),
        season_day=f"{(as_of - event.sown_on).days + 1}",
        gdd_accumulated=round(gdd_now, 1),
        gdd_to_next_stage=(
            round(stage.gdd_to_next_stage, 1)
            if stage.gdd_to_next_stage is not None
            else None
        ),
        stage=stage.stage.value,
        stage_progress=round(stage.season_progress, 4),
        emerged=bool(emerged),
        root_depth_cm=round(root * 100.0, 1),
        taw_mm=round(taw, 1),
        raw_mm=round(raw, 1),
        depletion_mm=round(depletion_mm, 1),
        depletion_fraction=round(depletion_mm / max(1e-6, taw), 4),
        days_until_stress=stress_day,
        kc=round(kc, 3),
        et0_mm_day=round(today["et0_mm"], 3),
        etc_mm_day=round(etc, 3),
        rainfall_mm_day=round(today["rainfall_mm"], 3),
        requirement=requirement,
        schedule=schedule,
        stage_windows=windows,
        season_days=season_days,
        season_gross_requirement_mm=round(season_gross, 1),
        season_rainfall_mm=round(season_rain, 1),
        season_gross_volume_m3=round(season_volume, 1),
    )


def plan_registry(
    registry: SowingRegistry,
    weather: pd.DataFrame,
    as_of: date | None = None,
    depletion_mm: float = 0.0,
    with_season: bool = False,
) -> list[FieldPlan]:
    """Plan every registered field, sharing each station's weather lookups."""
    stations: dict[str, StationWeather] = {}
    climatologies: dict[str, RainClimatology | None] = {}
    plans: list[FieldPlan] = []
    for event in registry.list():
        if event.station not in stations:
            stations[event.station] = StationWeather(weather, event.station)
            try:
                climatologies[event.station] = build_rain_climatology(
                    weather, event.station
                )
            except Exception:
                climatologies[event.station] = None
        plans.append(
            plan_field(
                event,
                stations[event.station],
                climatologies[event.station],
                as_of=as_of,
                depletion_mm=depletion_mm,
                with_season=with_season,
            )
        )
    return plans
