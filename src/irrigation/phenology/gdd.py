"""
Phenology: where a crop is in its growth cycle, from accumulated heat.

Growth stage is derived from growing-degree-days rather than from position in
the season. This is the single largest departure from a calendar-fraction
approach and it is the one that makes the system genuinely crop-wise.

Why calendar position fails: two wheat crops sown a fortnight apart in the same
field have identical calendar schedules but different thermal histories, so a
model keyed to day-of-season treats them as the same crop on the same day. Worse,
a warm year and a cool year produce the same calendar schedule for a crop that
is phenologically well ahead of it. The consequence is that peak water demand
arrives on the wrong date in exactly the years where getting it wrong is most
expensive.

Why heat works: development rate is close to linear in accumulated thermal time
above a base temperature, with the base differing by species. Maize develops
above roughly 10 C, wheat above 0 to 4 C, cotton above 12 C. Accumulate the
excess and the result is comparable across years, latitudes and sowing dates.

**STATUS OF THE GDD THRESHOLDS: UNCALIBRATED.** FAO-56 expresses stage
boundaries in days, not thermal time. The thresholds in `data.crops` are seeded
from published phenology for reference varieties and are plausible rather than
verified for this project's target region. They are exposed as a single table so
that a field pilot can replace them wholesale, and every consumer reads them
from here rather than hard-coding a second copy. No result that depends on stage
placement should be presented as validated until those thresholds have been
checked against observed stage transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..data.crops import CropParameters


class GrowthStage(str, Enum):
    """
    The four FAO-56 stages.

    Named for the FAO-56 sections rather than for crop calendar terms, because
    "vegetative" and "mid-season" mean the same thing for every crop in this
    framework while "tasselling" means nothing for cotton.
    """

    INITIAL = "initial"
    DEVELOPMENT = "development"
    MID_SEASON = "mid_season"
    LATE_SEASON = "late_season"
    POST_HARVEST = "post_harvest"


#: Ordered stage list, used wherever the sequence itself matters.
STAGE_ORDER: tuple[GrowthStage, ...] = (
    GrowthStage.INITIAL,
    GrowthStage.DEVELOPMENT,
    GrowthStage.MID_SEASON,
    GrowthStage.LATE_SEASON,
)


@dataclass(frozen=True)
class StageState:
    """Where a crop is, and how far through its current stage it has got."""

    stage: GrowthStage
    #: Fraction through the current stage, 0 to 1.
    fraction_through_stage: float
    #: Accumulated GDD since sowing.
    gdd_accumulated: float
    #: GDD remaining to the next stage boundary. None after the final stage.
    gdd_to_next_stage: float | None
    #: GDD accumulated as a fraction of the total season requirement.
    season_progress: float
    #: GDD still required to reach harvest.
    gdd_to_harvest: float
    #: True when the crop has passed the final stage boundary.
    is_mature: bool

    @property
    def days_to_next_stage_hint(self) -> float | None:
        """
        Rough days to the next stage at the recent rate of heat accumulation.

        A hint for the interface only. The rate is taken over the trailing week
        because a single day is far too noisy to extrapolate from, and a warm
        spell should shorten the wait rather than extend it.
        """
        return None  # populated by the daily driver, which holds the history


class GrowingDegreeDays:
    """
    Accumulates thermal time for one crop in one field.

    Simple base-temperature accumulation, the standard method. Two details are
    often got wrong and both are handled here:

    The upper cutoff is the mean daily temperature, not the maximum. FAO-56
    notes that using Tmax alone over-accumulates on hot days, which in a hot
    climate is most days, and it advances the crop through its stages
    prematurely.

    The base temperature is species-specific and not interchangeable. Using
    maize's 10 C for wheat changes the accumulation by roughly a third, because
    wheat's season spans temperatures where maize would have accumulated almost
    nothing.
    """

    def __init__(self, base_temp_c: float, upper_temp_c: float | None = None) -> None:
        self.base_temp_c = float(base_temp_c)
        #: Cutoff for heat stress on development. 30 C is the conventional value
        #: for most field crops and suppresses accumulation during a heat wave.
        self.upper_temp_c = 30.0 if upper_temp_c is None else float(upper_temp_c)

    def daily_gdd(
        self, tmax_c: float | np.ndarray, tmin_c: float | np.ndarray
    ) -> float | np.ndarray:
        """
        Growing degree days for one day, or an array of days.

        `mean_effective = clip((tmax + tmin) / 2, base, upper)`, then subtract the
        base. Clipping to a floor of the base temperature is what makes this
        different from a plain subtraction: on a cold day a cereal accumulates
        zero rather than a negative number, and letting negatives accumulate
        would drive a winter wheat backwards through its stages.
        """
        mean = (np.asarray(tmax_c, dtype=float) + np.asarray(tmin_c, dtype=float)) / 2.0
        effective = np.clip(mean, self.base_temp_c, self.upper_temp_c)
        return effective - self.base_temp_c

    def accumulate(
        self, tmax_c: np.ndarray, tmin_c: np.ndarray
    ) -> np.ndarray:
        """Cumulative GDD across a series of days, same length as the input."""
        return np.cumsum(self.daily_gdd(tmax_c, tmin_c))


def stage_from_gdd(
    gdd_accumulated: float, crop: CropParameters
) -> StageState:
    """
    Locate a crop on its growth curve from accumulated thermal time.

    `gdd_stage_ends` gives four boundaries: the end of initial, development,
    mid-season and late-season. A crop past the fourth is mature and FAO-56 sets
    evapotranspiration to the bare-soil coefficient, which the water balance
    module applies rather than this function guessing at.

    The reported `fraction_through_stage` is linear in GDD within a stage. It is
    an approximation of development within the stage, and it is the one place in
    this module where a linear interpolation is a compromise rather than a
    convention: development rate within the development stage is not actually
    uniform. It is kept linear because the alternative, a published curve per
    stage, needs data this project does not have.
    """
    ends = crop.gdd_stage_ends
    total_required = ends[3]
    # Guard against a season defined entirely in the past, which would otherwise
    # divide by zero below.
    total_required = max(1e-6, total_required)

    if gdd_accumulated >= ends[3]:
        return StageState(
            stage=GrowthStage.POST_HARVEST,
            fraction_through_stage=1.0,
            gdd_accumulated=float(gdd_accumulated),
            gdd_to_next_stage=None,
            season_progress=1.0,
            gdd_to_harvest=0.0,
            is_mature=True,
        )

    if gdd_accumulated < ends[0]:
        stage_index = 0
        lower, upper = 0.0, ends[0]
    elif gdd_accumulated < ends[1]:
        stage_index = 1
        lower, upper = ends[0], ends[1]
    elif gdd_accumulated < ends[2]:
        stage_index = 2
        lower, upper = ends[1], ends[2]
    else:
        stage_index = 3
        lower, upper = ends[2], ends[3]

    span = max(1e-6, upper - lower)
    fraction = float(np.clip((gdd_accumulated - lower) / span, 0.0, 1.0))

    return StageState(
        stage=STAGE_ORDER[stage_index],
        fraction_through_stage=fraction,
        gdd_accumulated=float(gdd_accumulated),
        gdd_to_next_stage=float(upper - gdd_accumulated),
        season_progress=float(np.clip(gdd_accumulated / total_required, 0.0, 1.0)),
        gdd_to_harvest=float(max(0.0, ends[3] - gdd_accumulated)),
        is_mature=False,
    )


def effective_root_depth_m(
    crop: CropParameters, stage_state: StageState
) -> float:
    """
    Root depth at a given stage.

    FAO-56 grows rooting depth through the season and reaches maximum depth near
    the end of development. Using the maximum from day one, which a
    calendar-driven model effectively does, overstates the crop's access to
    stored soil water early in the season and therefore under-irrigates: a field
    with 300 mm of roots is drawing on a smaller reservoir than one with 1.7 m,
    and the same depletion fraction means a very different volume of water.

    Three growth shapes are used, and they differ in where the root front sits
    during the period that matters most:

    initial        linear from 10% to 25% of maximum depth
    development    rapid, reaching full depth at the end of the stage
    mid-season     holds at maximum
    late-season    slight decline, reflecting root senescence

    The decline is real but modest. FAO-56 attributes late-season stress to
    reduced root function, and a model that keeps full depth while also raising
    the depletion threshold late in the season would be doubly optimistic about
    the crop's late-season resilience.
    """
    maximum = crop.root_depth_max_m
    fraction = stage_state.fraction_through_stage

    if stage_state.is_mature:
        return 0.05

    if stage_state.stage is GrowthStage.INITIAL:
        return maximum * (0.10 + 0.15 * fraction)
    if stage_state.stage is GrowthStage.DEVELOPMENT:
        # Concave growth: most of the depth is reached early in development.
        return maximum * (0.25 + 0.75 * fraction**0.5)
    if stage_state.stage is GrowthStage.MID_SEASON:
        return maximum
    # Late season: root senescence, a shallow decline to 90% of maximum.
    return maximum * (1.0 - 0.10 * fraction)


def effective_depletion_fraction(
    crop: CropParameters, stage_state: StageState
) -> float:
    """
    The depletion fraction at which this crop begins to suffer, at this stage.

    `p` in FAO-56's notation. It is the number that sets when irrigation is
    triggered, and it is both crop-specific and stage-specific.

    The stage dependence matters most in late season, where FAO-56 notes drought
    sensitivity rises as the canopy senesces and root function declines. A crop
    held to the same `p` it tolerated at mid-season will lose yield at the end
    of the season for exactly the amount of water someone decided was
    unnecessary. The late-season reduction is encoded per crop in
    `depletion_fraction_p_late_factor`.
    """
    base = crop.depletion_fraction_p
    if stage_state.stage in (GrowthStage.LATE_SEASON, GrowthStage.POST_HARVEST):
        return base * crop.depletion_fraction_p_late_factor
    if stage_state.stage is GrowthStage.INITIAL:
        # A young crop with a shallow root system is more exposed per unit of
        # depletion, since the whole rooting profile is small.
        return base * 0.85
    return base


def days_remaining_in_stage(
    crop: CropParameters,
    stage_state: StageState,
    recent_gdd_per_day: float,
) -> float | None:
    """
    Approximate days until the next stage, given the recent rate of heat.

    For the operator interface and for planning the decision horizon. It uses a
    supplied rate rather than deriving one, because the caller holds the
    temperature history and a single day's rate is far too noisy to extrapolate
    from. Returns None when the crop is mature or the rate is not positive.
    """
    if stage_state.is_mature or stage_state.gdd_to_next_stage is None:
        return None
    if recent_gdd_per_day <= 0.05:
        return None
    return float(stage_state.gdd_to_next_stage / recent_gdd_per_day)


def thermal_time_anomaly(
    gdd_accumulated: float, expected_gdd_for_day_of_season: float
) -> float:
    """
    How far ahead or behind the crop is against a typical season, in GDD.

    A small module, but it is what lets the system notice an unusual year. A
    crop 15% ahead of typical at the end of development will reach its water-demand
    peak a fortnight earlier than the schedule assumed, and on a calendar-driven
    controller that fortnight is watered at the wrong depletion and with the
    wrong depth.

    Expressed as a fraction rather than an absolute difference, because a fixed
    GDD gap matters little early in the season and a great deal at mid-season.
    """
    if expected_gdd_for_day_of_season <= 1e-6:
        return 0.0
    return float((gdd_accumulated - expected_gdd_for_day_of_season) / expected_gdd_for_day_of_season)
