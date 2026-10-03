"""Tests for the schedule optimiser.

The properties tested here are the ones that make a schedule safe to hand to
someone. None of them are about finding a better schedule; a schedule that is
cheap but lets the crop into stress is not a better schedule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from irrigation.data.cultivation import method as lookup_method  # noqa: E402
from irrigation.data.soil import texture as lookup_texture  # noqa: E402
from irrigation.optimizer import (  # noqa: E402
    HORIZON_DAYS,
    Schedule,
    _depth_grid,
    optimise,
)

LOAM = lookup_texture("Loam")
SAND = lookup_texture("Sand")
CLAY = lookup_texture("Clay")


def _demand(peak: float, days: int = HORIZON_DAYS) -> np.ndarray:
    """A flat daily demand, which makes the arithmetic checkable by hand."""
    return np.full(days, peak, dtype=float)


def test_no_water_needed_produces_no_events():
    schedule = optimise(
        requirement_mm=0.0, demand=_demand(2.0), effective_rain=np.zeros(HORIZON_DAYS),
        depletion_now=5.0, taw=200.0, raw_mm=110.0, soil=LOAM,
        method=lookup_method("Sprinkler"),
    )
    assert schedule.total_events == 0
    assert schedule.total_net_mm == 0.0
    assert not schedule.infeasible


def test_single_event_when_method_can_deliver_it_all():
    """Low demand, deep soil, small requirement: one pass is genuinely enough."""
    schedule = optimise(
        requirement_mm=20.0, demand=_demand(2.0), effective_rain=np.zeros(HORIZON_DAYS),
        depletion_now=20.0, taw=300.0, raw_mm=165.0, soil=LOAM,
        method=lookup_method("Sprinkler"),
    )
    assert schedule.total_events == 1
    assert schedule.total_net_mm == pytest.approx(20.0, abs=5.0)
    assert schedule.stress_days == 0


def test_requirement_too_small_for_the_demand_is_reported_as_stress():
    """
    20 mm cannot cover 14 days of 4 mm demand, and the optimiser must say so.

    This is the case a greedy scheduler gets quietly wrong: it applies the
    available water, looks like it has delivered the requirement, and leaves the
    crop in stress from day three without saying anything. The stress days are the
    part that matters and they are reported, not buried in an objective function.
    """
    schedule = optimise(
        requirement_mm=20.0, demand=_demand(4.0), effective_rain=np.zeros(HORIZON_DAYS),
        depletion_now=100.0, taw=200.0, raw_mm=110.0, soil=LOAM,
        method=lookup_method("Sprinkler"),
    )
    assert schedule.stress_days > 0
    assert schedule.total_net_mm <= 20.0 + 5.0


def test_no_application_ever_exceeds_the_method_cap():
    """
    Basin is capped at 100 mm and is infiltration-limited below that; nothing may
    be scheduled deeper than the cap allows.

    The scenario gives a deep, nearly empty profile so that the per-event cap, not
    the available headroom, is what forces the requirement into several passes. An
    earlier version of this test used a 200 mm profile already 150 mm depleted
    under 9 mm/day demand, which cannot work: across Basin's six-day turn-around
    the soil refills to field capacity and there is nowhere to put the water, so it
    asserted two events on a field that has room for one.
    """
    schedule = optimise(
        requirement_mm=150.0, demand=_demand(2.0),
        effective_rain=np.zeros(HORIZON_DAYS),
        depletion_now=0.0, taw=400.0, raw_mm=220.0, soil=LOAM,
        method=lookup_method("Basin"),
    )
    cap = lookup_method("Basin").max_depth_per_event_mm
    for day in schedule.events:
        assert day.apply_mm <= cap + 1e-6
    # Loam admits 50 mm/day, so 150 mm needs three passes however deep the cap is.
    assert schedule.total_events >= 2
    assert schedule.total_net_mm <= 150.0 + 2.0


def test_no_application_exceeds_field_capacity():
    schedule = optimise(
        requirement_mm=300.0, demand=_demand(9.0), effective_rain=np.zeros(HORIZON_DAYS),
        depletion_now=0.0, taw=120.0, raw_mm=66.0, soil=CLAY,
        method=lookup_method("Basin"),
    )
    for day in schedule.events:
        assert day.depletion_after_mm >= -1e-6
        assert day.depletion_before_mm <= 120.0 + 1e-6


def test_infiltration_limit_is_respected_on_sand():
    """Sand cannot accept a 60 mm basin application in a day."""
    schedule = optimise(
        requirement_mm=120.0, demand=_demand(7.0), effective_rain=np.zeros(HORIZON_DAYS),
        depletion_now=40.0, taw=90.0, raw_mm=45.0, soil=SAND,
        method=lookup_method("Basin"),
    )
    for day in schedule.events:
        assert day.apply_mm <= SAND.ksat_mm_day + 1e-6


def test_rain_credit_reduces_the_schedule():
    wet = optimise(
        requirement_mm=40.0, demand=_demand(5.0), effective_rain=np.full(HORIZON_DAYS, 2.0),
        depletion_now=60.0, taw=200.0, raw_mm=110.0, soil=LOAM,
        method=lookup_method("Sprinkler"),
    )
    dry = optimise(
        requirement_mm=40.0, demand=_demand(5.0), effective_rain=np.zeros(HORIZON_DAYS),
        depletion_now=60.0, taw=200.0, raw_mm=110.0, soil=LOAM,
        method=lookup_method("Sprinkler"),
    )
    assert wet.total_net_mm <= dry.total_net_mm


def test_soil_state_stays_physical_across_the_horizon():
    schedule = optimise(
        requirement_mm=80.0, demand=_demand(6.0), effective_rain=np.zeros(HORIZON_DAYS),
        depletion_now=10.0, taw=200.0, raw_mm=110.0, soil=LOAM,
        method=lookup_method("Sprinkler"),
    )
    previous = None
    for day in schedule.events:
        assert 0.0 <= day.depletion_before_mm <= 200.0 + 1e-6
        assert 0.0 <= day.depletion_after_mm <= 200.0 + 1e-6
        if day.apply_mm > 0.05:
            assert day.depletion_after_mm <= day.depletion_before_mm + 1e-6
        if previous is not None:
            assert abs(day.depletion_before_mm - previous) < 1e-6
        previous = day.depletion_after_mm


def test_stress_is_avoided_where_the_method_allows():
    """A reachable requirement should not be scheduled into guaranteed stress."""
    schedule = optimise(
        requirement_mm=90.0, demand=_demand(6.0), effective_rain=np.zeros(HORIZON_DAYS),
        depletion_now=60.0, taw=300.0, raw_mm=165.0, soil=LOAM,
        method=lookup_method("Sprinkler"),
    )
    assert schedule.stress_days == 0


def test_infeasibility_is_reported_not_hidden():
    """A requirement no schedule can meet is flagged rather than silently cut."""
    schedule = optimise(
        requirement_mm=400.0, demand=_demand(8.0), effective_rain=np.zeros(HORIZON_DAYS),
        depletion_now=100.0, taw=150.0, raw_mm=82.0, soil=CLAY,
        method=lookup_method("Furrow"),
    )
    assert schedule.infeasible
    assert schedule.unmet_requirement_mm > 0.0
    assert any("cannot be met" in line for line in schedule.lines())


def test_fewer_events_is_cheaper_at_equal_water():
    """The fixed-cost term must actually change the chosen schedule."""
    one = optimise(
        requirement_mm=25.0, demand=_demand(4.0), effective_rain=np.zeros(HORIZON_DAYS),
        depletion_now=80.0, taw=300.0, raw_mm=165.0, soil=LOAM,
        method=lookup_method("Sprinkler"), event_cost_mm=10.0,
    )
    assert one.total_events == 1


def test_depth_grid_covers_the_cap():
    grid = _depth_grid(60.0)
    assert max(grid) == pytest.approx(60.0)
    assert all(0 < d <= 60.0 for d in grid)
    assert _depth_grid(3.0) == [3.0]
    assert _depth_grid(0.5) == [0.5]


def test_empty_forecast_does_not_crash():
    """
    A missing forecast must not produce an unsafe schedule.

    With no demand data the optimiser still honours the requirement it was given,
    so it may apply; what it must not do is apply without limit, or produce a
    schedule whose depths are unbounded because there is nothing to check them
    against.
    """
    schedule = optimise(
        requirement_mm=10.0, demand=np.array([]), effective_rain=np.array([]),
        depletion_now=20.0, taw=200.0, raw_mm=110.0, soil=LOAM,
        method=lookup_method("Sprinkler"),
    )
    assert isinstance(schedule, Schedule)
    assert schedule.total_net_mm <= lookup_method("Sprinkler").max_depth_per_event_mm + 1e-6
    for day in schedule.events:
        assert 0.0 <= day.depletion_before_mm <= 200.0 + 1e-6
        assert 0.0 <= day.depletion_after_mm <= 200.0 + 1e-6


def test_minimum_interval_is_respected_between_applications():
    """
    Basin has a six-day turn-around and the schedule must honour it.

    The optimiser's state was depletion only, which cannot express "how long ago
    did the equipment last run", so a basin field with 80 mm to place applied it
    on day 0 and day 1. The turn-around is in the method table and was never read.
    """
    method = lookup_method("Basin")
    schedule = optimise(
        requirement_mm=80.0, demand=_demand(5.0),
        effective_rain=np.zeros(HORIZON_DAYS),
        depletion_now=300.0, taw=360.0, raw_mm=198.0, soil=LOAM, method=method,
    )
    applied = [d.offset for d in schedule.events if d.apply_mm > 0.05]
    assert len(applied) >= 2, "this case is meant to need more than one pass"
    gaps = [b - a for a, b in zip(applied, applied[1:])]
    assert gaps, "no second event to check"
    assert min(gaps) >= method.min_interval_days, (
        f"applications {applied} are closer than the "
        f"{method.min_interval_days}-day turn-around for Basin"
    )


def test_minimum_interval_delays_a_pass_without_disabling_it():
    """The turn-around postpones irrigation; it must not forbid it outright."""
    method = lookup_method("Basin")
    schedule = optimise(
        requirement_mm=90.0, demand=_demand(6.0),
        effective_rain=np.zeros(HORIZON_DAYS),
        depletion_now=320.0, taw=360.0, raw_mm=198.0, soil=LOAM, method=method,
    )
    applied = [d.offset for d in schedule.events if d.apply_mm > 0.05]
    assert applied
    assert max(applied) > method.min_interval_days, (
        "Basin was still refusing to run after its turn-around had elapsed"
    )


def test_daily_methods_keep_a_one_day_turn_around():
    """Drip runs daily, so consecutive applications are legitimate for it."""
    method = lookup_method("Drip")
    assert method.min_interval_days == 1
    schedule = optimise(
        requirement_mm=20.0, demand=_demand(4.0),
        effective_rain=np.zeros(HORIZON_DAYS),
        depletion_now=60.0, taw=200.0, raw_mm=110.0, soil=LOAM, method=method,
    )
    for day in schedule.events:
        assert day.apply_mm <= method.max_depth_per_event_mm + 1e-6


def test_shortfall_below_the_search_resolution_is_not_called_infeasible():
    """
    Missing three tenths of a millimetre out of eighty is not a broken method.

    Depletion is bucketed to 2 mm and depths to a 5 mm grid, so the search cannot
    resolve a smaller shortfall. Flagging that as infeasible reported a perfectly
    workable basin schedule as impossible.
    """
    schedule = optimise(
        requirement_mm=80.3, demand=_demand(5.0),
        effective_rain=np.zeros(HORIZON_DAYS),
        depletion_now=300.0, taw=360.0, raw_mm=198.0, soil=LOAM,
        method=lookup_method("Basin"),
    )
    if schedule.unmet_requirement_mm > 0.0:
        assert schedule.unmet_requirement_mm <= 2.0
        assert not schedule.infeasible
        assert not schedule.requirement_insufficient
