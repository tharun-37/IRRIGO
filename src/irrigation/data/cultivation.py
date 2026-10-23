"""
Cultivation practice: how the water is applied, and what that permits.

Source for the physical limits: FAO-56 chapters 5 and 6, and FAO-29 on water
quality. The application efficiencies are indicative ranges for each method
rather than FAO tabulations, and are marked as such: they vary enormously with
layout, spacing, operator skill and terrain, which is exactly why they are an
input at commissioning rather than a constant in code.

This layer exists because it decides whether a recommendation is physically
executable. A 35 mm recommendation to a sprinkler system with a 25 mm intake
limit is not a suboptimal answer, it is a wrong one: the surplus becomes runoff
and leaching, and the field ends the week wetter at depth but drier at the
surface where the roots are.

It also decides whether grossing up is required. Net requirement and applied
depth are different quantities, and the ratio is the application efficiency.
Recommending net 20 mm to a flood-irrigated field means applying roughly 33 mm,
because roughly 40% of what leaves the pump never reaches the root zone.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..data.crops import CropParameters
from ..data.soil import SoilTexture


@dataclass(frozen=True)
class IrrigationMethod:
    """
    One way of applying water, with its physical and economic constraints.

    `max_depth_per_event_mm` is the binding physical limit. For surface methods
    it is set by the need to avoid prolonged ponding and to limit infiltration
    losses; for sprinkler by the soil intake rate, which is often the real
    constraint; for drip by the wetted volume and the need for frequent small
    applications.

    `application_efficiency` is the fraction of applied water stored in the root
    zone. It is the factor between gross and net requirement, and getting it
    wrong by 20 points is a 25% error in every recommendation on the field.

    `min_interval_days` encodes that the equipment has a turn-around: a drip
    system cannot be asked for one heavy application a week, and a mainline
    cannot be refilled instantly.
    """

    name: str
    #: Largest single net application this method can deliver usefully, mm.
    max_depth_per_event_mm: float
    #: Indicative fraction of applied water reaching the root zone.
    application_efficiency: float
    #: Shortest sensible gap between applications, days.
    min_interval_days: int
    #: Fixed labour and pumping energy per irrigation event, in arbitrary cost
    #: units. Set so that starting a pump is comparable to a few mm of water,
    #: which is the usual field reality and is what stops the optimiser from
    #: recommending a large number of tiny applications.
    fixed_cost_per_event: float = 4.0
    #: Marginal cost of a delivered mm, in the same arbitrary units.
    cost_per_mm: float = 1.0
    #: True where the field is ponded and the relevant state is a standing
    #: water depth rather than a soil depletion.
    is_ponded: bool = False
    #: Target standing depth to maintain, mm, for ponded methods.
    pond_target_depth_mm: float = 50.0
    #: Tolerance band on that depth. Outside it, the recommendation is to top up
    #: or drain, which is a different decision from irrigating a dry root zone.
    pond_tolerance_mm: float = 15.0


METHODS: dict[str, IrrigationMethod] = {
    method.name: method
    for method in (
        IrrigationMethod(
            name="Flood",
            max_depth_per_event_mm=120.0, application_efficiency=0.50,
            min_interval_days=7, fixed_cost_per_event=6.0, cost_per_mm=0.8,
        ),
        IrrigationMethod(
            name="Basin",
            max_depth_per_event_mm=100.0, application_efficiency=0.55,
            min_interval_days=6, fixed_cost_per_event=5.0, cost_per_mm=0.9,
        ),
        IrrigationMethod(
            name="Furrow",
            max_depth_per_event_mm=75.0, application_efficiency=0.60,
            min_interval_days=5, fixed_cost_per_event=4.5, cost_per_mm=0.95,
        ),
        IrrigationMethod(
            name="Sprinkler",
            # The intake rate is the constraint, and it depends on the soil, so
            # this is the nominal figure for a loam under a conventional
            # sprinkler. `effective_max_depth` intersects it with the soil's
            # basic infiltration rather than taking this number on trust.
            max_depth_per_event_mm=35.0, application_efficiency=0.75,
            min_interval_days=3, fixed_cost_per_event=3.0, cost_per_mm=1.3,
        ),
        IrrigationMethod(
            name="Drip",
            max_depth_per_event_mm=18.0, application_efficiency=0.90,
            min_interval_days=1, fixed_cost_per_event=2.5, cost_per_mm=1.6,
        ),
        IrrigationMethod(
            name="Paddy",
            max_depth_per_event_mm=60.0, application_efficiency=0.85,
            min_interval_days=1, fixed_cost_per_event=2.0, cost_per_mm=1.0,
            is_ponded=True, pond_target_depth_mm=50.0, pond_tolerance_mm=15.0,
        ),
    )
}


def method(name: str) -> IrrigationMethod:
    if name not in METHODS:
        raise KeyError(f"unknown irrigation method {name!r}; available: {sorted(METHODS)}")
    return METHODS[name]


def effective_max_depth_mm(
    irrigation_method: IrrigationMethod, soil: SoilTexture
) -> float:
    """
    The application limit that actually binds, given the soil.

    A sprinkler system on clay cannot take 35 mm in one event whatever the
    sprinkler rating says, because the soil will not accept it. Taking the
    minimum of the equipment limit and the soil's basic infiltration is the only
    version of this that is true on the ground.
    """
    return min(
        irrigation_method.max_depth_per_event_mm,
        soil.basic_infiltration_mm_day * 1.5,
    )


def gross_up_mm(net_mm: float, irrigation_method: IrrigationMethod) -> float:
    """
    Convert a net root-zone requirement into the depth to pump.

    The model reasons in net water, because that is what the crop consumes. The
    pump delivers gross water, and the difference is lost to evaporation,
    runoff, percolation beyond the root zone, and non-uniformity. A field that
    ignores this factor is systematically under-watered on every application.
    """
    efficiency = max(0.1, min(0.99, irrigation_method.application_efficiency))
    return net_mm / efficiency


# --------------------------------------------------------------------------
# Management context
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ManagementContext:
    """
    How this particular field is run.

    These are commissioning inputs, not measurements. They do not change day to
    day, and getting them right once is worth more than any amount of model
    refinement. The three that matter most are the method, the efficiency and
    the nitrogen regime, in that order.

    `mulch_present` reduces bare-soil evaporation, which for a maize field in
    the first month after sowing is often a third of total ET. FAO-56 puts the
    reduction from mulching at roughly 20 to 50 percent of evaporation, which is
    the difference between irrigating and not irrigating in the establishment
    period.

    `nitrogen_regime` matters because canopy size drives Kc. The same soil
    moisture on a high-nitrogen maize field and a low-nitrogen one implies
    different decisions, since the former is transpiring more and will reach its
    depletion limit sooner. FAO-56's tabulated Kc assumes adequate nutrition;
    nitrogen stress lowers it.
    """

    method_name: str
    application_efficiency_override: float | None = None
    mulch_present: bool = False
    #: Fraction of Kc attributable to bare-soil evaporation, removed by mulch.
    mulch_evaporation_reduction: float = 0.35
    #: Relative nitrogen supply. 1.0 is adequate, below that is limiting.
    nitrogen_regime: float = 1.0
    #: Standing surface water depth, mm. Only meaningful for ponded methods.
    current_pond_depth_mm: float = 0.0
    #: Days since the previous irrigation, or None if never irrigated.
    days_since_last_irrigation: int | None = None
    #: Depth of the previous application, mm, used to judge residual depletion.
    last_application_mm: float = 0.0

    @property
    def method(self) -> IrrigationMethod:
        return method(self.method_name)

    @property
    def application_efficiency(self) -> float:
        if self.application_efficiency_override is not None:
            return self.application_efficiency_override
        return self.method.application_efficiency


def nitrogen_kc_multiplier(nitrogen_regime: float) -> float:
    """
    Scale FAO-56's Kc by how well the crop is fed.

    FAO-56's tabulated coefficients assume a well-nourished, unstressed crop.
    A nitrogen-limited canopy has less leaf area, transpires less, and has a
    correspondingly lower Kc. Modelling that as a multiplier rather than as a
    separate coefficient keeps the published values intact and makes the
    correction auditable: the published number stays the anchor, and this is the
    local deviation from it.
    """
    # Below 0.6 the crop is visibly nitrogen-deficient and FAO-56's figure
    # overstates demand. Above 1.0 the crop is well supplied but rarely
    # transpiring more than the reference, so the cap prevents an
    # over-fertilised field from inflating the estimate without limit.
    if nitrogen_regime <= 0.0:
        return 0.65
    if nitrogen_regime < 0.6:
        return 0.65 + 0.35 * (nitrogen_regime / 0.6) * 0.5
    if nitrogen_regime <= 1.0:
        return 0.825 + 0.175 * (nitrogen_regime - 0.6) / 0.4
    return min(1.10, 1.0 + 0.10 * (nitrogen_regime - 1.0))


def crop_and_method_are_compatible(
    crop: CropParameters, irrigation_method: IrrigationMethod
) -> tuple[bool, str | None]:
    """
    Flag combinations where the pairing itself is worth checking.

    Not a hard block. A paddy crop under drip is a legitimate (if unusual)
    setup, and a field may be transitioning between methods mid-season. The
    reason to raise it is that the pairing changes what the recommendation means:
    drip on rice is scheduled on a near-daily cycle with small depths, and
    treating it as a threshold-triggered soil-deficit problem would be wrong.
    """
    if crop.name == "Rice" and not irrigation_method.is_ponded:
        return False, (
            "Rice is managed as a maintained pond depth. Sprinkler or drip on rice "
            "is a real practice, but it needs near-daily scheduling with small "
            "depths rather than threshold-triggered soil-deficit irrigation."
        )
    if crop.name != "Rice" and irrigation_method.is_ponded:
        return False, (
            f"{irrigation_method.name} is a ponded method, which suits rice. For "
            f"{crop.name} it means managing standing water, not root-zone "
            f"depletion, and the recommendations would not apply as stated."
        )
    if crop.root_depth_max_m < 0.4 and irrigation_method.name in {"Furrow", "Flood"}:
        return False, (
            f"{crop.name} roots to {crop.root_depth_max_m * 100:.0f} cm. Frequent "
            f"{irrigation_method.name.lower()} application on a shallow-rooted "
            f"crop will waterlog it; drip or sprinkler suits the root depth."
        )
    return True, None
