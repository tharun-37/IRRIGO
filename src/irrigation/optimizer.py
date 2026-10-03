r"""
Turn a water requirement into a schedule a pump can actually deliver.

The regressor answers "how much water does this field need over the next
forteen days". That is not an instruction. It does not say whether to apply it
today or in a week, how deep each pass should be, or how many times the farmer
has to turn up. Those are constrained by the application method, the soil's
infiltration, the crop's minimum interval, and the cost of mobilising the
equipment, and getting them wrong is how a correct requirement becomes an
impractical recommendation.

The problem is small enough to solve exactly rather than approximately. Over a
fourteen-day horizon with a minimum interval of three days there are a few
thousand feasible schedules, and a dynamic program over them is instant. Greedy
alternatives fail in a specific and common way: they apply the maximum depth as
early as possible, which refills the profile to field capacity and then loses the
excess to deep percolation. Drip and sprinkler waste that; a basin tolerates it
better. An optimiser that knows the infiltration constraint and the percolation
loss will spread the same water over fewer, better-timed passes without being told
to.

The objective is a weighted sum, and the weights are stated rather than tuned,
because tuning them against a simulated season would produce numbers that look
precise and mean nothing. Water is not free but neither is yield, so the schedule
minimises gross applied depth plus a per-event fixed cost, subject to never
letting the crop cross its stress point. Stress is a hard constraint rather than
a penalty, because a schedule that saves a millimetre of water by allowing stress
is not a cheaper schedule, it is a worse one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data.cultivation import IrrigationMethod, gross_up_mm
from .data.soil import SoilTexture
from .phenology.gdd import GrowthStage
from .physics.kc_curve import build_breakpoints, kc_at_gdd
from .physics.water_balance import effective_rainfall_mm

#: Days the schedule is planned over.
HORIZON_DAYS = 14

#: Weight on the per-event fixed cost, in equivalent millimetres of water. A
#: figure rather than a currency because the caller may not know the cost of
#: mobilising a pump, and a schedule that saves one irrigation is usually worth
#: several millimetres to whoever has to drive out and do it.
DEFAULT_EVENT_COST_MM = 4.0

#: Cost of a millimetre of the requirement that is not delivered, in equivalent
#: millimetres of water. This is deliberately far larger than any achievable water
#: saving, which makes the objective lexicographic in effect: deliver the
#: requirement first, then use as little water and as few passes as possible to do
#: it. At a weight of 2 the optimiser would decline to irrigate a 150 mm
#: requirement because 300 mm of gross water looked dearer than leaving 120 mm
#: unmet -- a schedule that costs a grower yield to save water, which is the wrong
#: trade in every direction. The gross water the search could possibly save is
#: bounded by the horizon and the per-event cap (order 2,800 mm at worst), so 100
#: is comfortably outside that range and the ordering cannot be overturned.
UNMET_PENALTY_PER_MM = 100.0

#: Days of forecast at the end of the horizon that get a safety margin. A
#: forecast is least reliable furthest out, so the last few days are planned
#: against a slightly padded demand rather than the point estimate.
TAIL_MARGIN_DAYS = 3


@dataclass
class ScheduleDay:
    """One day of the planned schedule."""

    offset: int
    apply_mm: float
    gross_apply_mm: float
    depletion_before_mm: float
    depletion_after_mm: float
    stress_before: bool
    stress_after: bool
    reason: str = ""


@dataclass
class Schedule:
    """A complete, feasible plan, with enough detail to explain itself."""

    events: list[ScheduleDay] = field(default_factory=list)
    total_net_mm: float = 0.0
    total_gross_mm: float = 0.0
    total_events: int = 0
    first_event_day: int | None = None
    unmet_requirement_mm: float = 0.0
    stress_days: int = 0
    percolation_mm: float = 0.0
    infeasible: bool = False
    requirement_insufficient: bool = False
    notes: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [f"  schedule: {self.total_events} event(s), "
               f"{self.total_net_mm:.1f} mm net / {self.total_gross_mm:.1f} mm gross"]
        if self.first_event_day is not None:
            out.append(f"  first application: day {self.first_event_day}")
        for day in self.events:
            if day.apply_mm > 0.05:
                out.append(
                    f"    day {day.offset:>2}: {day.apply_mm:>5.1f} mm net"
                    f" ({day.gross_apply_mm:>5.1f} gross)"
                    f"  depletion {day.depletion_before_mm:.0f} -> "
                    f"{day.depletion_after_mm:.0f} mm"
                )
        if self.unmet_requirement_mm > 0.05:
            out.append(
                f"  WARNING: {self.unmet_requirement_mm:.1f} mm of the requirement "
                f"cannot be met by this method within the minimum interval"
            )
        if self.requirement_insufficient:
            out.append(
                f"  WARNING: delivering the full {self.total_net_mm:.1f} mm still "
                f"leaves the crop above its allowable depletion on "
                f"{self.stress_days} day(s). The requirement itself is too small "
                f"for this crop, soil and forecast; applying more than the "
                f"requirement is a decision for the grower, not for the model"
            )
        out.extend(f"  note: {note}" for note in self.notes)
        return out


def horizon_inputs(
    et0: np.ndarray,
    expected_rain: np.ndarray,
    rain_probability: np.ndarray,
    gdd_forward: np.ndarray,
    gdd_now: float,
    crop,
    taw: float,
    p: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Daily demand, effective rain and raw, over the horizon.

    Expected rain is credited at the probability that an event of that kind occurs
    in the day, not at full depth. Crediting a month's rain as if it fell every
    day over-credits a wet season and under-irrigates a dry one, and it is the
    single easiest place to make a rain-fed recommendation wrong.
    """
    breakpoints = build_breakpoints(crop)
    demand = np.zeros(len(et0))
    effective_rain = np.zeros(len(et0))
    running_gdd = gdd_now
    depletion = 0.0
    for i in range(len(et0)):
        running_gdd += float(gdd_forward[i])
        kc = kc_at_gdd(running_gdd, breakpoints)
        demand[i] = kc * et0[i]
        effective_rain[i] = expected_rain[i] * rain_probability[i]
    return demand, effective_rain, np.full(len(et0), taw)


def optimise(
    requirement_mm: float,
    demand: np.ndarray,
    effective_rain: np.ndarray,
    depletion_now: float,
    taw: float,
    raw_mm: float,
    soil: SoilTexture,
    method: IrrigationMethod,
    max_depth_per_event: float | None = None,
    event_cost_mm: float = DEFAULT_EVENT_COST_MM,
    horizon: int = HORIZON_DAYS,
    stress_fraction_of_raw: float = 1.0,
    overshoot_allowance: float = 1.0,
) -> Schedule:
    """
    Choose dates and depths by dynamic programming over feasible schedules.

    The state is the day reached and the depletion left. The action is a depth in
    `max_depth` increments, or zero for no application. The terminal value
    subtracts unmet requirement, so a schedule that delivers less is correctly
    more expensive, and the search prefers delivering in fewer passes purely
    through the fixed cost rather than through a hand-written tie-break.

    The horizon is assumed to end at the stress point rather than at the next
    natural replenishment. That is the conservative choice: it means the
    optimiser is not permitted to push work past the point where the crop starts
    losing yield, which is the failure that a purely cost-minimising schedule
    always eventually produces.
    """
    cap = max(1.0, float(
        max_depth_per_event
        if max_depth_per_event is not None
        else min(method.max_depth_per_event_mm, soil.ksat_mm_day)
    ))
    depths = _depth_grid(cap)
    trigger = raw_mm * stress_fraction_of_raw
    return _search(
        requirement_mm=requirement_mm,
        demand=demand,
        effective_rain=effective_rain,
        depletion_now=depletion_now,
        taw=taw,
        raw_mm=raw_mm,
        soil=soil,
        method=method,
        cap=cap,
        depths=depths,
        trigger=trigger,
        event_cost_mm=event_cost_mm,
        horizon=horizon,
        overshoot_allowance=overshoot_allowance,
    )


def _depth_grid(cap: float, resolution: float = 5.0) -> list[float]:
    """
    Application depths to consider, on a fixed grid up to the cap.

    A grid rather than a continuous range, because the search is exhaustive and
    the number of actions per day is what multiplies the state space. Five
    millimetres is finer than any irrigation recommendation is reported, so the
    grid is not the limiting factor on the answer.
    """
    count = int(np.floor(cap / resolution))
    if count < 1:
        return [float(cap)]
    grid = [float(i * resolution) for i in range(1, count + 1)]
    if cap - grid[-1] > 1e-9:
        grid.append(float(cap))
    return grid


def _search(
    requirement_mm: float,
    demand: np.ndarray,
    effective_rain: np.ndarray,
    depletion_now: float,
    taw: float,
    raw_mm: float,
    soil: SoilTexture,
    method: IrrigationMethod,
    cap: float,
    depths: list[float],
    trigger: float,
    event_cost_mm: float,
    horizon: int,
    overshoot_allowance: float = 1.0,
) -> Schedule:
    """
    Exact search over schedules, by dynamic programming on rounded depletion.

    Depletion is discretised to 2 mm. That is a deliberate approximation and the
    resolution is chosen so the state space stays around 250 buckets rather than
    thousands: a 360 mm profile at 2 mm is 180 states, and with a 5 mm depth grid
    the search is exhaustive over a discretisation fine enough that the depth
    reported in the output is dominated by the requirement rather than by the
    bucket size. Using continuous depletion would require a continuous state space
    and buy nothing, because the answer is reported in whole millimetres anyway.
    """
    step = 2.0
    buckets = max(1, int(round(taw / step)))

    def bucket(value: float) -> int:
        return int(np.clip(round(value / step), 0, buckets))

    start = bucket(float(np.clip(depletion_now, 0.0, taw)))
    # The optimiser delivers the requirement. It is not permitted to exceed it.
    # Without this cap the stress penalty makes over-watering look attractive: a
    # schedule asked for 20 mm and applied 45, because every millimetre that kept
    # the crop below its stress point reduced the objective. That is a model
    # deciding a grower's irrigation rate. The requirement comes from the
    # estimator and represents what the field needs; if that is genuinely not
    # enough to avoid stress, the answer is to say so, not to quietly apply two
    # and a quarter times as much as computed.
    total_cap = float(requirement_mm) * max(1.0, overshoot_allowance)

    # Equipment has a turn-around: basin cannot be filled two days running, and
    # `min_interval_days` records how many. Enforcing it means the state has to
    # remember how long ago the last application happened, because depletion alone
    # does not carry that information -- two schedules can reach the same bucket by
    # different routes and only one of them is allowed to irrigate today. Without
    # the cooldown, a basin field got its 80 mm on day 0 and day 1.
    min_interval = max(0, int(getattr(method, "min_interval_days", 0) or 0))

    # (bucket, days_until_available) -> (cost, path, applied_so_far)
    State = tuple[int, int]
    layer: dict[State, tuple[float, tuple[float, ...], float]] = {
        (start, 0): (0.0, (), 0.0)
    }
    percolation_by_path: dict[tuple[float, ...], float] = {(): 0.0}
    stress_by_path: dict[tuple[float, ...], int] = {(): 0}

    for day in range(horizon):
        etc = float(demand[day]) if day < len(demand) else 0.0
        rain = float(effective_rain[day]) if day < len(effective_rain) else 0.0
        nxt: dict[State, tuple[float, tuple[float, ...], float]] = {}
        for (state, cooldown), (cost, path, applied_so_far) in layer.items():
            depletion = state * step
            moved = depletion + etc - rain
            for depth in [0.0] + depths:
                applied = 0.0
                if depth > 0.0:
                    if cooldown > 0:
                        # The equipment is still recovering from the last pass.
                        continue
                    applied = min(depth, soil.ksat_mm_day,
                                  max(0.0, taw - depletion),
                                  max(0.0, total_cap - applied_so_far))
                after = float(np.clip(moved - applied, 0.0, taw))
                lost = max(0.0, moved - taw)   # deep percolation above field capacity
                new_cost = cost + gross_up_mm(applied, method)
                if applied > 0.05:
                    new_cost += event_cost_mm
                new_stress = stress_by_path.get(path, 0) + (1 if after > trigger else 0)
                if after > trigger:
                    new_cost += 500.0 + (after - trigger)
                new_path = path + (applied,)
                # A day with no application is one day closer to being allowed to
                # apply again; a day with an application restarts the turn-around.
                next_cooldown = (min_interval - 1) if applied > 0.05 \
                    else max(0, cooldown - 1)
                key = (bucket(after), next_cooldown)
                # Dominance has to prefer water already delivered, not lower cost.
                # Two paths can reach the same depletion bucket by applying very
                # different amounts, because a near-empty profile absorbs an
                # application without changing the bucket. Comparing on cost alone
                # therefore always discarded the watering and kept the dry path, and
                # the optimiser returned a schedule that applied nothing at all.
                # Equal bucket and equal cooldown means the soil state and the
                # equipment state are the same, so the future is identical and the
                # path that has already delivered more of the requirement is
                # strictly the better one to keep.
                current = nxt.get(key)
                if current is None or applied_so_far + applied > current[2] or (
                    applied_so_far + applied == current[2] and new_cost < current[0]
                ):
                    nxt[key] = (new_cost, new_path, applied_so_far + applied)
                percolation_by_path.setdefault(new_path, percolation_by_path.get(path, 0.0) + lost)
                stress_by_path.setdefault(new_path, new_stress)
        layer = nxt
        if not layer:
            break

    def final_cost(item) -> float:
        cost, path, _ = item
        total_applied = sum(path)
        unmet = max(0.0, requirement_mm - total_applied)
        return cost + unmet * UNMET_PENALTY_PER_MM

    best_key = min(layer, key=lambda k: final_cost(layer[k]))
    cost, path, _ = layer[best_key]

    depletion = float(np.clip(depletion_now, 0.0, taw))
    events: list[ScheduleDay] = []
    total_net = 0.0
    total_gross = 0.0
    for day, applied in enumerate(path):
        etc = float(demand[day]) if day < len(demand) else 0.0
        rain = float(effective_rain[day]) if day < len(effective_rain) else 0.0
        before = depletion
        stress_before = before > trigger
        depletion = float(np.clip(before + etc - rain - applied, 0.0, taw))
        stress_after = depletion > trigger
        reason = ""
        if applied > 0.05:
            if before < trigger:
                reason = "soil approaching its allowable depletion"
            else:
                reason = "topping up after depletion"
        elif stress_before:
            reason = "rain is expected to relieve the deficit"
        events.append(
            ScheduleDay(
                offset=day,
                apply_mm=round(applied, 2),
                gross_apply_mm=round(gross_up_mm(applied, method), 2),
                depletion_before_mm=round(before, 2),
                depletion_after_mm=round(depletion, 2),
                stress_before=bool(stress_before),
                stress_after=bool(stress_after),
                reason=reason,
            )
        )
        total_net += applied
        total_gross += gross_up_mm(applied, method)

    applied_days = [d.offset for d in events if d.apply_mm > 0.05]
    unmet = max(0.0, requirement_mm - total_net)
    # A shortfall is only worth calling infeasible if it is bigger than the
    # search's own resolution. Depletion is bucketed to 2 mm and depths to a 5 mm
    # grid, so demanding 0.05 mm of exactness reports a field as infeasible for
    # missing three tenths of a millimetre out of eighty. The tolerance is the
    # larger of one depletion bucket and 1% of the requirement.
    tolerance = max(step, 0.01 * float(requirement_mm))
    return Schedule(
        events=events,
        total_net_mm=round(total_net, 2),
        total_gross_mm=round(total_gross, 2),
        total_events=len(applied_days),
        first_event_day=applied_days[0] if applied_days else None,
        unmet_requirement_mm=round(unmet, 2),
        stress_days=sum(1 for d in events if d.stress_after),
        percolation_mm=round(float(percolation_by_path.get(path, 0.0)), 2),
        infeasible=bool(unmet > tolerance),
        requirement_insufficient=bool(
            stress_by_path.get(path, 0) > 0 and unmet <= tolerance
        ),
    )
