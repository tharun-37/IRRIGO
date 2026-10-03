"""
The net irrigation requirement over a planning horizon.

This is the target the decision model learns, and it is the specific change that
fixes the previous generation's central defect.

What was wrong before
---------------------
The irrigation target was `min(TAW - Dr, basic_infiltration)`. The second term
is a per-soil constant, and it binds whenever depletion exceeds the infiltration
capacity, which is most of the time. In the resulting corpus, 66% of all
irrigation events landed on exactly three values, 22, 12 and 32 mm, one per soil
texture. A gradient-boosting model fitted to it reached an MAE of 1.05 mm, which
looks like a good model and is in fact mostly a soil-texture lookup table: the
lookup alone reproduces 66% of events to within 1 mm.

The consequence was not cosmetic. A target with no weather signal in it cannot
transfer to a new climate, and the model fitted to it scored R-squared 0.082 on
an unseen station. It had learned the soil, not the irrigation.

What replaces it
----------------
A net irrigation requirement over a forward horizon:

    NIR(t) = sum over the horizon of ETc, corrected
             minus effective rainfall expected over the horizon
             minus the soil water the root zone can already supply

clipped to what the field can usefully accept in one application. Each term
varies continuously and each is crop-specific:

    ETc          Kc(stage) x ET0. Rice's Kc of 1.05 against maize's 0.30 in the
                 initial stage means the same field, on the same day, under the
                 same weather, carries a different requirement.
    ET0          measured weather, so the target inherits the climate.
    rain         from the station's climatology, so the target inherits the
                 monsoon.
    soil supply  depletion and RAW, which vary with crop and stage.

The result is a continuous quantity that is different on two fields receiving
identical sensor readings, which is the whole point of a crop-wise system.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.cultivation import IrrigationMethod, effective_max_depth_mm
from ..data.crops import CropParameters
from ..phenology.gdd import (
    GrowthStage,
    StageState,
    effective_depletion_fraction,
    effective_root_depth_m,
)
from .kc_curve import KcBreakpoints, adjust_kc_for_soil_moisture, kc_at_gdd
from .water_balance import FieldConfiguration, SoilWaterState

#: Smallest net application worth opening a pump for, mm. Below this the water
#: saved is smaller than the cost of the event, and starting a pump daily to
#: move two millimetres is a net loss in labour, energy and wear.
MIN_USEFUL_APPLICATION_MM = 2.0


@dataclass(frozen=True)
class HorizonDay:
    """One day of the planning horizon."""

    #: Days from now. 0 is today.
    offset: int
    et0_mm: float
    expected_effective_rain_mm: float
    #: Probability of useful rain on this day, from the station climatology.
    rain_probability: float
    #: Growing degree days that day, used to advance the crop's stage.
    gdd: float
    kc: float
    etc_mm: float


@dataclass(frozen=True)
class IrrigationRequirement:
    """
    The requirement, with the accounting that produced it.

    The decomposition is kept because a single number handed to an operator
    cannot be checked, and a recommendation nobody can verify is one nobody will
    follow.
    """

    #: Net water the root zone needs over the horizon, mm. NOT capped by the
    #: application limit, and that separation is deliberate. Capping the target
    #: produces a plateau at the cap across every soil that hits it, which is the
    #: identical pathology that made the previous generation's target a per-soil
    #: lookup table: 66% of its events sat on three values. How much water a
    #: field needs is a quantity to learn; how it is split into applications is
    #: a constraint for the scheduler to enforce.
    net_requirement_mm: float
    #: What the pump has to deliver for that to reach the roots, mm.
    gross_requirement_mm: float
    #: The largest single application, which is the target capped by the method
    #: and the soil. This is what the valve actually does today.
    single_application_mm: float
    #: Crop demand over the horizon before rainfall and soil supply, mm.
    demand_mm: float
    #: Effective rainfall credited over the horizon, mm.
    rainfall_credit_mm: float
    #: Drawdown of root-zone depletion the irrigation itself causes, mm.
    soil_supply_mm: float
    #: Largest single application this method and soil can take, mm.
    max_application_mm: float
    #: True when one application cannot deliver the whole requirement.
    requires_multiple_applications: bool
    #: Days between applications this method requires at minimum.
    min_interval_days: int
    #: Depletion at the end of the horizon with no irrigation, mm.
    projected_depletion_mm: float
    #: Depletion at the end of the horizon with the first application, mm.
    projected_depletion_after_mm: float
    #: The irrigation method, carried for the explanation text.
    method_name: str = "unknown"

    @property
    def requires_irrigation(self) -> bool:
        """
        Whether to irrigate, on the physical test rather than a tuned threshold.

        Tested against the *single application*, not the total requirement,
        because a few millimetres spread over a fortnight is not an irrigation
        event. It is a rounding difference between one reading and the next, and
        recommending against it would start the pump daily to move nothing.
        """
        return self.single_application_mm > MIN_USEFUL_APPLICATION_MM

    def explanation(self) -> list[str]:
        """
        Operator-facing accounting, in plain language.

        Same purpose as the previous generation's explanation strings: an
        operator who cannot see why the valve opened will eventually stop
        trusting it and turn it off.
        """
        lines = [
            f"Crop demand over the horizon is {self.demand_mm:.1f} mm.",
        ]
        if self.rainfall_credit_mm > 0.5:
            lines.append(
                f"Expected rainfall supplies {self.rainfall_credit_mm:.1f} mm, so "
                f"less needs to be pumped."
            )
        if self.soil_supply_mm > 0.5:
            lines.append(
                f"The root zone can supply {self.soil_supply_mm:.1f} mm of its own "
                f"stored water before reaching the operating trigger."
            )
        lines.append(
            f"Net requirement {self.net_requirement_mm:.1f} mm, grossed up to "
            f"{self.gross_requirement_mm:.1f} mm at the pump for application losses."
        )
        if self.requires_multiple_applications:
            lines.append(
                f"One application can place at most {self.max_application_mm:.1f} mm "
                f"on this soil by {self.min_interval_days_dialect()}, so today's "
                f"pass is {self.single_application_mm:.1f} mm and the balance is "
                f"scheduled for later."
            )
        return lines

    def min_interval_days_dialect(self) -> str:
        """Human phrasing for the method's turn-around, for the explanation text."""
        names = {
            "Drip": "a drip system",
            "Sprinkler": "a sprinkler set",
            "Furrow": "a furrow run",
            "Basin": "a basin",
            "Flood": "a flood",
            "Paddy": "the paddy",
        }
        return names.get(self.method_name, "this method")


def build_horizon(
    et0_series_mm: np.ndarray,
    expected_rain_series_mm: np.ndarray,
    rain_probability_series: np.ndarray,
    gdd_series: np.ndarray,
    gdd_now: float,
    crop: CropParameters,
    season_scale: float = 1.0,
    kc_override: dict[str, float] | None = None,
) -> list[HorizonDay]:
    """
    Evaluate ETc for each day of the horizon, advancing the crop as it goes.

    The crop's stage is re-evaluated per horizon day rather than held fixed at
    today's value. Over a fortnight that matters: a maize field crossing into
    mid-season inside the horizon will need materially more water in the second
    week than in the first, and a single-stage calculation understates the total
    by exactly the amount that decides whether the application is sufficient.

    The Kc used for each day is also adjusted downward as depletion rises, so the
    demand reflects a crop that is already closing stomata rather than a
    hypothetical unstressed one.
    """
    breakpoints = _breakpoints_for(crop, season_scale, kc_override)
    days: list[HorizonDay] = []
    cumulative_gdd = float(gdd_now)

    for offset in range(len(et0_series_mm)):
        if offset > 0:
            cumulative_gdd += float(gdd_series[offset])
        kc = kc_at_gdd(cumulative_gdd, breakpoints)
        etc = kc * float(et0_series_mm[offset])
        days.append(
            HorizonDay(
                offset=offset,
                et0_mm=float(et0_series_mm[offset]),
                expected_effective_rain_mm=float(expected_rain_series_mm[offset]),
                rain_probability=float(rain_probability_series[offset]),
                gdd=float(gdd_series[offset]),
                kc=float(kc),
                etc_mm=float(etc),
            )
        )
    return days


def _breakpoints_for(
    crop: CropParameters,
    season_scale: float,
    kc_override: dict[str, float] | None,
) -> KcBreakpoints:
    from .kc_curve import build_breakpoints

    return build_breakpoints(crop, season_scale, kc_override)


def net_requirement(
    horizon: list[HorizonDay],
    configuration: FieldConfiguration,
    state: SoilWaterState,
    irrigation_method: IrrigationMethod,
    application_efficiency: float,
    gdd_at_horizon_end: float,
    rainfall_credibility: float = 1.0,
    operating_depletion_fraction: float = 0.55,
) -> IrrigationRequirement:
    """
    Compute the net irrigation requirement over a horizon.

    The accounting, in the order it matters:

    1. Sum crop demand, ETc, over the horizon.
    2. Credit expected effective rainfall, downweighted by how much the forecast
       can be trusted. Treating an expected 20 mm as certain leads to
       under-watering on the day, and the asymmetry of the cost makes that the
       wrong way to be wrong.
    3. Subtract what the root zone can supply without being taken to its
       operating trigger. The trigger sits *before* the crop's stress point, not
       at it: a field driven to wilting point and then rescued has already paid
       for the water in yield. Crediting soil only up to the trigger is what
       makes the requirement come due before the stress rather than at it.
    4. The remainder is the net requirement, floored at zero and deliberately
       NOT capped at the application limit.
    5. Gross up for application losses, and separately report what a single
       application can place, which is what the valve actually does.

    `gdd_at_horizon_end` is the crop's accumulated thermal time at the end of the
    horizon, used to fetch the depletion threshold that applies there. It is
    passed in rather than reconstructed, because reconstructing it from the
    horizon's daily increments alone would lose the accumulated total and
    silently return the initial stage's threshold.
    """
    if not horizon:
        return IrrigationRequirement(
            net_requirement_mm=0.0, gross_requirement_mm=0.0, single_application_mm=0.0,
            demand_mm=0.0, rainfall_credit_mm=0.0, soil_supply_mm=0.0,
            max_application_mm=0.0, requires_multiple_applications=False,
            min_interval_days=0, projected_depletion_mm=state.depletion_mm,
            projected_depletion_after_mm=state.depletion_mm,
            method_name=irrigation_method.name,
        )

    demand_mm = float(sum(day.etc_mm for day in horizon))

    # Rain credit, scaled by how much the forecast is trusted, and bounded by the
    # demand it could possibly satisfy: rain cannot supply more water than the
    # atmosphere is removing over the same period.
    rain_credit = float(
        sum(day.expected_effective_rain_mm * day.rain_probability for day in horizon)
    ) * float(np.clip(rainfall_credibility, 0.0, 1.0))
    rain_credit = min(rain_credit, demand_mm)

    # Soil supply, bounded by the operating trigger rather than by wilting.
    from ..phenology.gdd import stage_from_gdd

    end_stage = stage_from_gdd(gdd_at_horizon_end, configuration.crop)
    p = effective_depletion_fraction(configuration.crop, end_stage)
    operating_trigger = (
        float(np.clip(operating_depletion_fraction, 0.05, 1.0))
        * p
        * state.total_available_water_mm
    )
    supply_available = max(0.0, operating_trigger - state.depletion_mm)
    soil_supply = float(min(supply_available, max(0.0, demand_mm - rain_credit)))

    # The requirement itself: continuous, and deliberately uncapped.
    net = max(0.0, demand_mm - rain_credit - soil_supply)

    # What one application can actually place. This is a separate quantity from
    # the requirement because the requirement is a fact about the field and the
    # application limit is a fact about the equipment and the soil. Conflating
    # them puts a plateau in the target at the cap, which is precisely how the
    # previous generation's target came to be a per-soil constant.
    max_application = effective_max_depth_mm(irrigation_method, configuration.soil)
    single_application = float(min(net, max_application))

    efficiency = max(0.1, min(0.99, application_efficiency))
    # Gross the *requirement*, not the capped single pass. The whole requirement
    # has to reach the root zone whatever the number of passes, so every pass
    # carries the loss. Grossing the single application instead reports a pump
    # volume smaller than the field actually needs, and for a split schedule
    # reports a gross depth below the net depth it is meant to deliver.
    gross_requirement = net / efficiency
    gross_single = single_application / efficiency

    # Project the depletion forward so the caller can see what the
    # recommendation does rather than having to re-derive it.
    projected_no_irrigation = float(state.depletion_mm + demand_mm - rain_credit)
    projected_no_irrigation = float(
        np.clip(projected_no_irrigation, 0.0, state.total_available_water_mm)
    )
    projected_after = float(
        np.clip(
            projected_no_irrigation - single_application,
            0.0, state.total_available_water_mm,
        )
    )

    return IrrigationRequirement(
        net_requirement_mm=net,
        gross_requirement_mm=gross_requirement,
        single_application_mm=single_application,
        demand_mm=demand_mm,
        rainfall_credit_mm=rain_credit,
        soil_supply_mm=soil_supply,
        max_application_mm=max_application,
        requires_multiple_applications=bool(net > max_application + 1e-9),
        min_interval_days=irrigation_method.min_interval_days,
        projected_depletion_mm=projected_no_irrigation,
        projected_depletion_after_mm=projected_after,
        method_name=irrigation_method.name,
    )


def single_event_target_mm(
    horizon: list[HorizonDay],
    configuration: FieldConfiguration,
    state: SoilWaterState,
    irrigation_method: IrrigationMethod,
    gdd_at_horizon_end: float,
) -> float:
    """
    Convenience: the depth to apply in one pass, before efficiency grossing-up.

    Present as a named entry point because this is the quantity most callers
    actually want, and burying it inside the full requirement object invites
    callers to reach for `gross_requirement_mm` and double-count the losses.
    """
    from ..data.cultivation import gross_up_mm

    requirement = net_requirement(
        horizon=horizon,
        configuration=configuration,
        state=state,
        irrigation_method=irrigation_method,
        application_efficiency=irrigation_method.application_efficiency,
        gdd_at_horizon_end=gdd_at_horizon_end,
    )
    return gross_up_mm(requirement.single_application_mm, irrigation_method)
