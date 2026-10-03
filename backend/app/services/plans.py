"""
The plan service: everything the dashboard reads goes through here.

The engine is expensive enough that it must not be called once per field per
request, and the expensive part is loading ten years of station weather. So the
weather and the derived per-station rain climatologies are loaded once at
startup and held on the service. Planning a field is then arithmetic.

Two decisions worth stating, because both are places where a dashboard is
tempted to lie:

**Depletion is a parameter, not a guess.** A field plan needs to know how dry the
soil is, and there is no probe in this system. Rather than inventing a value, the
API accepts the depletion the operator measured and marks the plan as assuming
field capacity when it is not supplied. The frontend shows which it is looking
at. A dashboard that quietly assumed a dry field would be confidently wrong.

**Plan results are cached by their inputs.** A plan is a pure function of the
event, the date and the depletion, so it can be memoised safely. The cache is
what makes the season endpoint usable, since it walks a field day by day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from irrigation.data.cultivation import METHODS
from irrigation.data.crops import CROPS
from irrigation.data.soil import TEXTURES
from irrigation.physics.climate import build_rain_climatology, load_weather
from irrigation.sowing import (
    PLAN_HORIZON_DAYS,
    SowingEvent,
    SowingRegistry,
    StationWeather,
    plan_field,
    season_depletion,
)

logger = logging.getLogger("iic.plans")


@dataclass
class PlanContext:
    """Shared, expensive, read-only state."""

    weather: pd.DataFrame
    stations: dict[str, StationWeather]
    climatologies: dict[str, Any]


class PlanService:
    def __init__(
        self,
        sowings_path,
        start_year: int = 2015,
        end_year: int = 2024,
    ) -> None:
        self.sowings_path = sowings_path
        self.start_year = start_year
        self.end_year = end_year
        self._context: PlanContext | None = None
        self._registry: SowingRegistry | None = None
        self._plan_cache: dict[tuple[str, str, float, bool], Any] = {}
        self._plan_cache_limit = 8192
        # Bumped whenever the inputs a plan depends on change. Downstream caches
        # key on this instead of a wall clock, so a plan is recomputed when the
        # weather corpus or the registry actually moves and not merely because
        # time passed. See `CompatService._snapshot`.
        self._version = 0

    # --- lifecycle -------------------------------------------------------

    def load(self) -> None:
        """
        Load weather and the registry. Called once at startup.

        A failure here is logged and tolerated: the app still starts, reports
        `ready: false`, and the health endpoint says why. A monitoring dashboard
        that refuses to come up when its data is missing is a dashboard that
        cannot tell the operator what is wrong.
        """
        self._context = None
        self._registry = None
        self._plan_cache.clear()
        self._version += 1
        try:
            weather = load_weather(self.start_year, self.end_year)
            names = sorted(weather["station"].unique())
            stations = {name: StationWeather(weather, name) for name in names}
            climatologies: dict[str, Any] = {}
            for name in names:
                try:
                    climatologies[name] = build_rain_climatology(weather, name)
                except Exception:
                    # A station with too little history to build a climatology
                    # still plans; it just gets no forward rain credit, which the
                    # plan reports as an assumption.
                    climatologies[name] = None
                    logger.warning("no rain climatology for %s", name)
            self._context = PlanContext(weather, stations, climatologies)
            logger.info(
                "weather ready: %d records, %d stations", len(weather), len(stations)
            )
        except Exception:
            logger.exception("weather corpus unavailable; planning endpoints will 503")
        try:
            self._registry = SowingRegistry.load(self.sowings_path)
        except FileNotFoundError:
            self._registry = SowingRegistry()
            logger.warning("no registry at %s; starting empty", self.sowings_path)
        except Exception:
            logger.exception("registry unreadable; planning endpoints will 503")

    @property
    def ready(self) -> bool:
        return self._context is not None and self._registry is not None

    @property
    def version(self) -> int:
        """
        Monotonic counter for "the planning inputs changed".

        Weather is a fixed archive and the registry only moves through `register`
        or `remove`, so this counter is the honest cache key. Anything downstream
        that memoises on plan results can hold them until it moves, instead of
        rebuilding an identical answer on a timer.
        """
        return self._version

    @property
    def context(self) -> PlanContext:
        if self._context is None:
            raise ServiceUnavailable("weather corpus is not loaded")
        return self._context

    @property
    def registry(self) -> SowingRegistry:
        if self._registry is None:
            raise ServiceUnavailable("sowing registry is not loaded")
        return self._registry

    # --- catalogue -------------------------------------------------------

    def station_names(self) -> list[str]:
        if self._context is None:
            return []
        return sorted(self._context.stations)

    def describe_options(self) -> dict[str, list[str]]:
        """Valid values for the register form, taken from the engine itself."""
        return {
            "stations": self.station_names(),
            "crops": sorted(CROPS),
            "soils": sorted(TEXTURES),
            "methods": sorted(METHODS),
        }

    # --- fields ----------------------------------------------------------

    def list_fields(self) -> list[SowingEvent]:
        return self.registry.list()

    def get_field(self, field_id: str) -> SowingEvent:
        for event in self.registry.list():
            if event.field_id == field_id:
                return event
        raise NotFound(f"no field registered as {field_id!r}")

    def register(self, event: SowingEvent, replace: bool = False) -> SowingEvent:
        self.registry.register(event, replace=replace)
        self.registry.save(self.sowings_path)
        # A new sowing changes what every downstream snapshot should contain, and
        # a replaced one changes it under the same field id. Both invalidate.
        self._plan_cache.clear()
        self._version += 1
        return event

    def remove(self, field_id: str) -> None:
        self.registry.remove(field_id)
        self.registry.save(self.sowings_path)
        self._plan_cache.clear()
        self._version += 1

    # --- planning --------------------------------------------------------

    def plan(
        self,
        event: SowingEvent,
        as_of: date | None = None,
        depletion_mm: float = 0.0,
        with_season: bool = True,
    ):
        context = self.context
        station = context.stations.get(event.station)
        if station is None:
            raise NotFound(f"no weather for station {event.station!r}")
        return plan_field(
            event,
            station,
            context.climatologies.get(event.station),
            as_of=as_of,
            depletion_mm=depletion_mm,
            with_season=with_season,
        )

    def plan_cached(
        self,
        field_id: str,
        as_of: str,
        depletion_mm: float,
        with_season: bool = False,
    ):
        """
        A plan is a pure function of (field, date, depletion, season), so it memoises.

        The growth view walks one field across up to 180 dates; without this the
        dashboard would recompute the whole FAO-56 balance for every cell of
        every chart. A plain dict is used rather than `lru_cache` because the
        latter on a method pins the service instance for the process lifetime and
        gives no way to inspect or clear what is held.

        `with_season` is part of the key because a plan with the season summary
        attached is a different, strictly larger object than one without it. The
        overview grid asks for the summary on every field; without the flag in the
        key it would collide with a chart's cheaper call and serve a plan that is
        missing `stageWindows`.
        """
        key = (field_id, as_of, round(depletion_mm, 3), with_season)
        hit = self._plan_cache.get(key)
        if hit is None:
            event = self.get_field(field_id)
            hit = self.plan(
                event,
                as_of=date.fromisoformat(as_of),
                depletion_mm=depletion_mm,
                with_season=with_season,
            )
            if len(self._plan_cache) > self._plan_cache_limit:
                self._plan_cache.clear()
            self._plan_cache[key] = hit
        return hit

    def growth_series(
        self, event: SowingEvent, days: int, step: int = 2, depletion_mm: float = 0.0
    ) -> list[dict[str, Any]]:
        """
        The season as the engine walks it, one point every `step` days.

        Two engine functions are composed rather than a second water balance being
        written here. `season_depletion` supplies the depletion the engine's own
        operating rule actually reached, and each sampled day is then planned with
        `plan_field` at that depletion, which yields stage, thermal time, Kc, root
        depth, TAW/RAW and that day's requirement.

        The result is the plan the engine would have produced on each of those
        days, which is a real quantity. It is deliberately *not* the training
        corpus: `corpus.simulate_field` models a stochastic operator and draws
        per-series soil chemistry, so replaying it here would be a simulation of a
        hypothetical, not a record of this field.
        """
        context = self.context
        station = context.stations[event.station]
        climatology = context.climatologies.get(event.station)

        series = season_depletion(
            event,
            station,
            climatology,
            as_of=None,
            max_days=days,
        )
        # Seed depletion is the operator's measurement, held constant for the
        # replay; the engine's own season rule then carries it forward.
        if series and depletion_mm:
            first_day, first_value = series[0]
            series[0] = (first_day, depletion_mm)

        points: list[dict[str, Any]] = []
        for index, (day, depletion) in enumerate(series):
            if index % max(1, step) and index != len(series) - 1:
                continue
            if day > date.today():
                continue
            plan = self.plan(
                event, as_of=day, depletion_mm=depletion, with_season=False
            )
            requirement = plan.requirement
            schedule = plan.schedule
            first_event = schedule.events[0] if schedule and schedule.events else None
            points.append(
                {
                    "day": (day - event.sown_on).days,
                    "date": day.isoformat(),
                    "stage": plan.stage,
                    "gdd": plan.gdd_accumulated,
                    "kc": plan.kc,
                    "rootCm": plan.root_depth_cm,
                    "tawMm": plan.taw_mm,
                    "rawMm": plan.raw_mm,
                    "depletionMm": plan.depletion_mm,
                    "depletionFraction": plan.depletion_fraction,
                    "needMm": float(getattr(requirement, "single_application_mm", 0.0)),
                    "applyMm": float(first_event.apply_mm) if first_event else 0.0,
                    "grossMm": float(first_event.gross_apply_mm) if first_event else 0.0,
                    "et0MmDay": plan.et0_mm_day,
                    "etcMmDay": plan.etc_mm_day,
                    "rainfallMmDay": plan.rainfall_mm_day,
                    "daysUntilStress": plan.days_until_stress,
                    "requirementInsufficient": bool(
                        getattr(schedule, "requirement_insufficient", False)
                    )
                    if schedule is not None
                    else False,
                }
            )
        return points

    def describe(self) -> dict[str, Any]:
        if self._context is None:
            return {
                "available": False,
                "reason": "weather corpus not loaded",
                "records": 0,
                "stations": 0,
                "fields": 0,
            }
        return {
            "available": True,
            "records": int(len(self._context.weather)),
            "stations": len(self._context.stations),
            "fields": len(self._registry.list()) if self._registry else 0,
            "horizonDays": PLAN_HORIZON_DAYS,
            "startYear": self.start_year,
            "endYear": self.end_year,
            "source": "nasa-power",
        }


class ServiceUnavailable(RuntimeError):
    pass


class NotFound(LookupError):
    pass
