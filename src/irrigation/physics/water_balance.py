"""
The soil water balance, and the net irrigation requirement.

This is the physics core. It is deterministic, auditable, and every coefficient
in it traces to a numbered table in FAO-56. Nothing here is fitted, because
there is nothing left to fit once the published coefficients are used correctly.

Notation follows FAO-56 chapter 7:

    Dr    root zone depletion, mm. Water removed from the root zone by
          evapotranspiration and not replenished. 0 at field capacity, TAW at
          wilting point.
    TAW   total available water, mm. TAW = (theta_fc - theta_wp) x Zr.
    RAW   readily available water, mm. RAW = p x TAW.
    ETc   crop evapotranspiration, mm/day. ETc = Kc x ET0.
    pe    effective rainfall, mm. The fraction of gross rainfall that infiltrates
          rather than runs off or evaporates.

The previous generation of this project computed its irrigation target as
`min(TAW - Dr, basic_infiltration)`, which meant that once depletion exceeded
the infiltration capacity the target stopped depending on the crop, the season
or the weather at all, and became a constant per soil texture. Two thirds of all
irrigation events in that corpus sat on three values. That target is replaced
here by a horizon net irrigation requirement, which is continuous, crop-specific
and driven by weather.

The distinction matters beyond tidiness. A target that is a per-soil constant
contains no information about the field's climate, so a model fitted to it cannot
possibly transfer to an unseen station, and the one that was fitted to it did
not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.crops import CropParameters
from ..data.soil import SoilTexture
from ..phenology.gdd import GrowthStage, StageState, effective_root_depth_m


@dataclass(frozen=True)
class SoilWaterState:
    """
    Root zone water state at one moment.

    `depletion_mm` is the state variable that gets carried forward day to day.
    Everything else is either a boundary or a derived read-out.
    """

    depletion_mm: float
    total_available_water_mm: float
    readily_available_water_mm: float
    root_depth_m: float

    @property
    def depletion_fraction(self) -> float:
        """Depletion as a fraction of TAW. This is the normalised form."""
        if self.total_available_water_mm <= 1e-6:
            return 0.0
        return float(np.clip(self.depletion_mm / self.total_available_water_mm, 0.0, 1.0))

    @property
    def relative_depletion(self) -> float:
        """
        Depletion relative to the stress threshold, where 1.0 means stress begins.

        Normalising against RAW rather than against TAW is what makes the figure
        comparable between a sandy soil with 80 mm of water and a clay with 200.
        Both reach stress at the same value, which is the correct behaviour and
        is impossible if the model only ever sees TAW.
        """
        if self.readily_available_water_mm <= 1e-6:
            return float("inf")
        return self.depletion_mm / self.readily_available_water_mm

    @property
    def is_stressed(self) -> bool:
        return self.depletion_mm > self.readily_available_water_mm

    @property
    def above_wilting_point(self) -> bool:
        return self.depletion_mm < self.total_available_water_mm


@dataclass(frozen=True)
class FieldConfiguration:
    """Everything about one field that does not change day to day."""

    crop: CropParameters
    soil: SoilTexture
    #: Restricting layer depth, metres, if the profile has a hardpan or gravel.
    restricting_depth_m: float | None = None
    penetration_resistance_mpa: float | None = None
    #: Scale factor on the season, for transplanted or delayed crops.
    season_scale: float = 1.0

    def root_depth_m(self, stage_state: StageState) -> float:
        """
        Root depth this crop can actually reach at this stage.

        The minimum of the crop's stage-appropriate depth and any physical
        restriction in the profile. FAO-56's depths assume an unconstrained
        soil; on a hardpan the crop sits above it and behaves as though the water
        below were not there.
        """
        from ..data.soil import restricted_root_depth_m

        potential = effective_root_depth_m(self.crop, stage_state)
        return restricted_root_depth_m(
            potential, self.restricting_depth_m, self.penetration_resistance_mpa
        )

    def water_state(
        self, stage_state: StageState, depletion_mm: float
    ) -> SoilWaterState:
        """Assemble the water state for a stage and a depletion."""
        from ..phenology.gdd import effective_depletion_fraction

        root_depth = self.root_depth_m(stage_state)
        taw = self.soil.taw_mm(root_depth)
        p = effective_depletion_fraction(self.crop, stage_state)
        return SoilWaterState(
            depletion_mm=float(max(0.0, depletion_mm)),
            total_available_water_mm=taw,
            readily_available_water_mm=float(min(taw, p * taw)),
            root_depth_m=root_depth,
        )


# --------------------------------------------------------------------------
# Daily recurrence
# --------------------------------------------------------------------------


def effective_rainfall_mm(
    rainfall_mm: float,
    soil: SoilTexture,
    current_depletion_mm: float,
    total_available_water_mm: float,
) -> float:
    """
    Rainfall that actually infiltrates and enters the root zone.

    Gross rainfall is not the same as effective rainfall, and using the gross
    figure is one of the commonest ways an irrigation model over-waters. Three
    deductions apply, and FAO-29 and FAO-56 both give them:

    Interception. The canopy holds a millimetre or two before dripping, which on
    a light shower is the entire event.

    Runoff. Water above the soil's infiltration capacity during the rainfall
    runs off rather than infiltrating. Small events on a clay therefore contribute
    almost nothing.

    Soil storage. Water cannot enter a root zone that is already near field
    capacity, and deep percolation below the root zone is not available to the
    crop even though it may have infiltrated.

    The last of these is the one that matters most for the decision. A 25 mm rain
    event on a soil 20 mm below field capacity adds almost nothing the crop can
    use, and recommending irrigation on the strength of the gross figure would
    water a field that the rain has already watered.
    """
    if rainfall_mm <= 0.0:
        return 0.0

    # Interception, small and canopy-dependent. FAO-56 gives 0.5 to 2 mm.
    interception_mm = 1.0
    available = rainfall_mm - interception_mm
    if available <= 0.0:
        return 0.0

    # Surface runoff: only the portion within the soil's intake capacity gets in.
    infiltrating = min(available, soil.basic_infiltration_mm_day)

    # Deep percolation: the root zone can only accept what its remaining storage
    # holds. Anything past that drains below the roots and is lost to the crop.
    if total_available_water_mm <= 1e-6:
        return 0.0
    storage_headroom = max(
        0.0, total_available_water_mm - current_depletion_mm
    )
    return float(min(infiltrating, storage_headroom))


def advance_depletion(
    depletion_mm: float,
    etc_mm: float,
    effective_rain_mm: float,
    applied_mm: float,
    total_available_water_mm: float,
) -> float:
    """
    One day forward in the water balance.

    Dr(t+1) = Dr(t) + ETc(t) - pe(t) - Irr(t)

    clamped to [0, TAW] at both ends. The lower clamp is physical: a root zone
    cannot be drier than wilting point, and any further extraction is stress on a
    root system that has already lost contact with the water. The upper clamp is
    physical too: applying more than the soil can accept pushes water below the
    root zone by drainage, so the profile returns to field capacity while the
    surplus has already left the root zone.

    That upper clamp is precisely what turned the previous generation's target
    into a constant. Here it is a boundary condition on the state, not the
    definition of the target.
    """
    updated = depletion_mm + etc_mm - effective_rain_mm - applied_mm
    return float(np.clip(updated, 0.0, max(0.0, total_available_water_mm)))


def depletion_after_et(
    depletion_mm: float,
    etc_mm: float,
    total_available_water_mm: float,
) -> float:
    """
    Depletion after a day with no rain and no irrigation.

    Used by the look-ahead, where the rainfall is not known yet and the honest
    assumption for a planning horizon is to carry forward the last known state
    and the expected demand. Assuming rain that does not arrive produces a
    recommendation that will not be needed on the day; assuming none produces
    one that may be too small. The policy module weighs both and prefers to
    irrigate early by a little rather than late by a lot, because the
    asymmetric cost is already established upstream.
    """
    return float(np.clip(depletion_mm + etc_mm, 0.0, max(0.0, total_available_water_mm)))


# --------------------------------------------------------------------------
# Stress
# --------------------------------------------------------------------------


def water_stress_index(
    state: SoilWaterState,
    ec_ds_m: float | None = None,
    salinity_threshold_ds_m: float = 6.0,
    salinity_slope: float = 0.07,
) -> float:
    """
    Combined water and salinity stress on the crop, 0 to 1.

    FAO-56 treats salinity as a separate stress pathway, and they compound:
    osmotic stress closes stomata, which reduces transpiration, which makes the
    crop less able to cool itself and more susceptible to heat. A model that
    scores only water will recommend watering a saline field when what the field
    actually needs is leaching, and watering it more will concentrate the salt
    further.

    Water stress ramps from zero at field capacity to 1.0 at wilting point, so it
    reports how hard the crop is being pushed rather than only whether a binary
    threshold has been crossed. Salinity follows FAO-29's linear-threshold form
    above its threshold and is zero below it.

    The two combine as the maximum rather than the product. Multiplying would
    let a mildly stressed and a mildly saline field compound into apparent
    severe stress, which is not how the mechanisms interact: the crop is limited
    by whichever is binding.
    """
    water_term = float(np.clip(state.depletion_fraction, 0.0, 1.0))

    salinity_term = 0.0
    if ec_ds_m is not None and ec_ds_m > salinity_threshold_ds_m:
        salinity_term = float(
            np.clip(
                1.0 - salinity_slope * (ec_ds_m - salinity_threshold_ds_m),
                0.0,
                1.0,
            )
        )

    return float(max(water_term, salinity_term))


def crop_stress_index(stage_state: StageState, stress: float) -> float:
    """
    Fold a stress magnitude into the yield-relevant index the models consume.

    A crop in late season under mild stress is worse off than the same stress at
    mid-season, because the yield is already committed and there is less time to
    recover. Stress is therefore weighted by how much yield is still at stake.
    """
    if stage_state.is_mature:
        return 0.0
    if stage_state.stage is GrowthStage.LATE_SEASON:
        return float(np.clip(stress * 1.25, 0.0, 1.0))
    if stage_state.stage is GrowthStage.INITIAL:
        # Early stress costs leaf area that cannot be recovered, which is severe
        # in its way, but the yield consequence over a whole season is smaller
        # than a mid-season deficit of the same magnitude.
        return float(np.clip(stress * 0.85, 0.0, 1.0))
    return float(np.clip(stress, 0.0, 1.0))


def relative_depletion_allowance(
    state: SoilWaterState,
    season_progress: float,
    recommended_fraction: float = 0.55,
) -> float:
    """
    A single dimensionless trigger for "irrigate now".

    The scheduler needs one number to compare against, and the alternative, a
    raw depletion in millimetres, is not comparable across fields. Two fields on
    the same day, one sandy with 80 mm of available water and one clay with 200,
    need watering at very different depths but for the same reason.

    `recommended_fraction` is the depletion fraction at which irrigating is
    preferred to waiting. Below 0.5 of the readily available water is comfortable;
    FAO-56's own guidance is to irrigate when depletion reaches RAW, which is
    1.0 on this scale. The default of 0.55 irrigates earlier than strictly
    necessary, which is the right bias given the asymmetric cost of a missed
    irrigation, and it is stated rather than buried so a different policy can be
    chosen deliberately.
    """
    return float(recommended_fraction * state.readily_available_water_mm)


def relative_depletion_ratio(
    state: SoilWaterState, allowance_mm: float
) -> float:
    """
    Depletion against the trigger, where 1.0 or more means irrigate.

    Above 1.0 the crop is into stress. The distance above 1.0 is the severity,
    and it is what the policy module uses to scale the cost of waiting rather
    than treating "stressed" and "badly stressed" as the same state.
    """
    if allowance_mm <= 1e-6:
        return float("inf") if state.depletion_mm > 0 else 0.0
    return float(state.depletion_mm / allowance_mm)
