r"""
The FAO-56 crop coefficient curve, evaluated on real thermal time.

Source: FAO Irrigation and Drainage Paper 56, chapter 7 and Annex 2.

The curve has four segments and the segment shapes are not interchangeable.
Interpolating linearly between the tabulated stage values is the most common
shortcut in this kind of code and it is wrong in a way that biases demand
systematically: the development rise is concave, the late-season decline is
convex, and the errors from assuming otherwise both act to increase estimated
water use through the parts of the season where the crop is most sensitive.

FAO-56 writes both segments with an exponent on the fractional progress through
the stage:

    development   Kc = Kc_ini + (Kc_mid - Kc_ini) * f ** b,   b ~ 1.13
    late season   Kc = Kc_mid - (Kc_mid - Kc_end) * f ** b,   b ~ 0.65

Development uses b > 1, so the rise is slow then fast: the canopy's leaf area
expands slowly from a small beginning and then closes. Late season uses b < 1,
so the decline is steep then shallow. Inverting either exponent reverses the
shape and shifts the date of peak demand, which is the one date that must be
right.

Stages, per FAO-56:

    Kc
     |            _________  kc_mid
     |          /           \
     |        /               \
     |  ____/                   \_____  kc_end
     | /                              \
     |/                                \___  kc_end continues to bare soil
     +--------------------------------------------> time
       initial  development   mid   late
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data.crops import KCB_DEV_SHAPE, KCB_END_BARE_SOIL, KCB_LATE_SHAPE, CropParameters
from ..phenology.gdd import GrowthStage, StageState


@dataclass(frozen=True)
class KcBreakpoints:
    """
    The curve as four absolute values on the GDD axis.

    FAO-56 writes this curve against days. Rewriting it against accumulated GDD
    is what makes the curve usable for a crop whose season ran long or short,
    and it is a pure change of abscissa: the shape of the curve and the values on
    it are unchanged.
    """

    #: Kc at sowing. FAO-56's kc_ini.
    initial_start: float
    #: Kc at the end of the development stage. FAO-56's kc_mid, reached
    #: smoothly rather than stepped.
    development_end: float
    #: Kc at the end of mid-season, i.e. the start of late season.
    late_season_start: float
    #: Kc at the end of late season. FAO-56's kc_end.
    late_season_end: float

    gdd_initial_end: float
    gdd_development_end: float
    gdd_mid_season_end: float
    gdd_late_season_end: float


def build_breakpoints(
    crop: CropParameters,
    season_scale: float = 1.0,
    kc_override: dict[str, float] | None = None,
) -> KcBreakpoints:
    """
    Place the curve's control points on the GDD axis.

    `season_scale` stretches or compresses the whole season while preserving the
    stage proportions, which is how a transplanted crop or an unusually late
    sowing is represented without inventing a new parameter set.

    `kc_override` replaces individual coefficients, which is the hook the
    residual model in `irrigation.residuals` uses to apply a learned local
    correction to a published value. It exists so that the correction is visible
    as a delta from the published number rather than as a replacement that
    cannot be compared against it.
    """
    kc_initial = crop.kc_initial
    kc_mid = crop.kc_mid
    kc_end = crop.kc_end
    if kc_override:
        kc_initial = float(kc_override.get("initial", kc_initial))
        kc_mid = float(kc_override.get("mid", kc_mid))
        kc_end = float(kc_override.get("end", kc_end))

    ends = crop.gdd_stage_ends
    scale = max(0.05, float(season_scale))

    return KcBreakpoints(
        initial_start=kc_initial,
        # The development stage rises from kc_ini to kc_mid. FAO-56 treats
        # kc_ini as the value at the *start* of development, so the curve
        # arrives there having already spent the initial stage near it.
        development_end=kc_mid,
        late_season_start=kc_mid,
        late_season_end=kc_end,
        gdd_initial_end=ends[0] * scale,
        gdd_development_end=ends[1] * scale,
        gdd_mid_season_end=ends[2] * scale,
        gdd_late_season_end=ends[3] * scale,
    )


def kc_at_gdd(gdd_accumulated: float, breakpoints: KcBreakpoints) -> float:
    """
    Evaluate the FAO-56 curve at a point in thermal time.

    Four cases:

    initial        near-constant at kc_ini. FAO-56 holds this flat because the
                   canopy is too small for transpiring meaningfully and the
                   coefficient is dominated by evaporation from bare soil.
    development    kc_ini rising to kc_mid with the FAO-56 concave shape, so the
                   crop reaches near-peak demand slightly before mid-season. The
                   early rise matters: under-irrigating during development costs
                   leaf area that cannot be recovered later.
    mid-season     flat at kc_mid.
    late-season    kc_mid down to kc_end with the FAO-56 convex shape, then a
                   decline to bare soil after harvest.

    Returns the bare-soil coefficient after the season ends, because FAO-56
    requires evapotranspiration to continue at that rate for the residue and
    bare soil rather than stopping at zero.
    """
    gdd = float(gdd_accumulated)

    # Before sowing, at the initial coefficient. Extrapolating backwards would
    # produce negative fractions and nonsense curve values.
    if gdd <= 0.0:
        return breakpoints.initial_start

    # During the initial stage the coefficient is essentially flat.
    if gdd < breakpoints.gdd_initial_end:
        return breakpoints.initial_start

    if gdd < breakpoints.gdd_development_end:
        span = breakpoints.gdd_development_end - breakpoints.gdd_initial_end
        fraction = (gdd - breakpoints.gdd_initial_end) / max(1e-6, span)
        # FAO-56 applies the exponent directly: b = 1.13, so the rise is slow
        # then fast as leaf area expands. Applying 1/b instead inverts the
        # shape and moves the date of peak demand, which is the date that has
        # to be right.
        shaped = fraction**KCB_DEV_SHAPE
        return breakpoints.initial_start + shaped * (
            breakpoints.development_end - breakpoints.initial_start
        )

    if gdd < breakpoints.gdd_mid_season_end:
        return breakpoints.late_season_start

    if gdd < breakpoints.gdd_late_season_end:
        span = breakpoints.gdd_late_season_end - breakpoints.gdd_mid_season_end
        fraction = (gdd - breakpoints.gdd_mid_season_end) / max(1e-6, span)
        # b = 0.65 applied directly, so the decline is steep then shallow as
        # senescence progressively detaches the canopy from the transpiring
        # leaf area that drove peak demand.
        shaped = fraction**KCB_LATE_SHAPE
        return breakpoints.late_season_start + shaped * (
            breakpoints.late_season_end - breakpoints.late_season_start
        )

    # Post-harvest: FAO-56 steps down to bare-soil evaporation over roughly a
    # week, rather than dropping immediately, because residues remain wet.
    return breakpoints.late_season_end


def kc_for_stage_state(
    stage_state: StageState,
    crop: CropParameters,
    season_scale: float = 1.0,
    kc_override: dict[str, float] | None = None,
) -> float:
    """
    Convenience wrapper: the crop coefficient for a located crop.

    Evaluates the full curve rather than stepping between stage constants. A
    stepwise Kc that jumps from one stage's value to the next would put a
    discontinuity in the water demand exactly at the transitions, which is
    where irrigation decisions are most consequential.
    """
    breakpoints = build_breakpoints(crop, season_scale, kc_override)
    return kc_at_gdd(stage_state.gdd_accumulated, breakpoints)


def adjust_kc_for_soil_moisture(
    kc: float, depletion_fraction: float, p: float
) -> float:
    """
    Reduce the coefficient as the crop runs short of water.

    FAO-56 section 7 discusses this: as depletion approaches the stress
    threshold, stomata begin to close and evapotranspiration falls below the
    potential rate. A model that holds Kc at its potential value while the crop
    is wilting will overestimate demand exactly when the crop is under stress,
    and will keep recommending more water to a crop that cannot use it.

    The reduction is mild and capped. FAO-56's own treatment of mid-season stress
    reduces Kc by at most about a quarter, and a large reduction would create a
    perverse incentive: the drier the field, the less it is told to water, and the
    drier it gets.
    """
    if p <= 1e-6:
        return kc
    relative = depletion_fraction / p
    if relative <= 1.0:
        return kc
    # Beyond the threshold, Kc falls towards 75% of potential, reaching that at
    # roughly twice the threshold depletion.
    excess = min(1.0, (relative - 1.0))
    reduction = 0.25 * excess
    return kc * (1.0 - reduction)


def single_crop_coefficient(
    stage_state: StageState,
    crop: CropParameters,
    crop_height_m: float | None = None,
    mulch_present: bool = False,
    season_scale: float = 1.0,
    kc_override: dict[str, float] | None = None,
) -> float:
    """
    Single-crop coefficient Kc, ready to multiply by reference ET0.

    Handles the three adjustments FAO-56 specifies beyond the bare curve, each of
    which is a real departure from the published value in a specific situation
    and all of which are otherwise commonly missed:

    Crop height. A tall canopy has a larger aerodynamic surface and a different
    boundary layer, giving a slightly higher Kc than the standard reference
    crop. FAO-56 gives 1.05 for a tall dense canopy as an approximate
    adjustment.

    Mulching. FAO-56 puts the reduction in bare-soil evaporation from mulching at
    roughly 20 to 50% of evaporation. Applied here as a modest reduction to Kc,
    and not to ET0, because the reduction is to the evaporation component only.

    Post-harvest. Handled inside `kc_at_gdd`, which steps down to bare soil.
    """
    kc = kc_for_stage_state(stage_state, crop, season_scale, kc_override)

    if crop_height_m is not None:
        if crop_height_m >= 2.5:
            kc *= 1.05
        elif crop_height_m <= 0.3 and stage_state.stage in (
            GrowthStage.INITIAL,
            GrowthStage.DEVELOPMENT,
        ):
            # A very short young canopy with exposed soil transpires less than the
            # reference. FAO-56 notes this for transplanted crops especially.
            kc *= 0.90

    if mulch_present and stage_state.stage in (
        GrowthStage.INITIAL,
        GrowthStage.DEVELOPMENT,
    ):
        kc *= 0.92

    return max(0.05, kc)
