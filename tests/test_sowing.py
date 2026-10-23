"""Tests for sowing registration and the plan that follows from it."""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from irrigation.physics.climate import build_rain_climatology, load_weather  # noqa: E402
from irrigation.sowing import (  # noqa: E402
    SowingError,
    SowingEvent,
    SowingRegistry,
    StationWeather,
    plan_field,
    plan_registry,
    season_depletion,
)


@pytest.fixture(scope="module")
def weather():
    return load_weather(2015, 2024)


@pytest.fixture(scope="module")
def station(weather):
    return StationWeather(weather, "LKO")


@pytest.fixture(scope="module")
def climatology(weather):
    return build_rain_climatology(weather, "LKO")


def _event(**overrides) -> SowingEvent:
    payload = dict(
        field_id="T-1", station="LKO", crop="Wheat",
        sowing_date="2023-11-28", soil_type="Silt_Loam", method_name="Basin",
    )
    payload.update(overrides)
    return SowingEvent(**payload)


# --- registration


def test_registration_resolves_parameters():
    event = _event()
    assert event.crop_parameters.name == "Wheat"
    assert event.soil.name == "Silt_Loam"
    assert event.method.name == "Basin"
    assert event.sown_on == date(2023, 11, 28)


def test_soil_names_are_forgiving():
    assert _event(soil_type="sandy loam").soil.name == "Sandy_Loam"
    assert _event(soil_type="SANDY_LOAM").soil.name == "Sandy_Loam"


def test_bad_names_fail_at_registration_not_at_irrigation():
    with pytest.raises(Exception):
        _event(crop="Dragonfruit")
    with pytest.raises(Exception):
        _event(soil_type="Moonsand")
    with pytest.raises(Exception):
        _event(method_name="Telepathy")
    with pytest.raises(SowingError):
        _event(field_area_m2=0)
    with pytest.raises(SowingError):
        _event(sowing_date="28-11-2023")


def test_days_since_sowing_counts_calendar_days():
    event = _event()
    assert event.days_since_sowing(event.sown_on) == 0
    assert event.days_since_sowing(event.sown_on + timedelta(days=30)) == 30


def test_registry_round_trips_through_json(tmp_path):
    path = tmp_path / "sowings.json"
    registry = SowingRegistry(path)
    registry.register(_event())
    registry.register(_event(field_id="T-2", crop="Maize"))
    registry.save()

    reloaded = SowingRegistry.load(path)
    assert len(reloaded) == 2
    assert reloaded.get("T-2").crop == "Maize"
    assert json.loads(path.read_text())["version"] == 1


def test_registry_refuses_silent_overwrite(tmp_path):
    registry = SowingRegistry(tmp_path / "s.json")
    registry.register(_event())
    with pytest.raises(SowingError, match="already registered"):
        registry.register(_event(crop="Maize"))
    registry.register(_event(crop="Maize"), replace=True)
    assert registry.get("T-1").crop == "Maize"


def test_unknown_field_lists_what_exists():
    registry = SowingRegistry()
    registry.register(_event())
    with pytest.raises(SowingError, match="T-1"):
        registry.get("T-99")


# --- planning


def test_plan_before_sowing_is_an_error(station, climatology):
    with pytest.raises(SowingError, match="precedes sowing"):
        plan_field(_event(), station, climatology, as_of=date(2023, 11, 1))


def test_gdd_never_decreases_with_age(station, climatology):
    event = _event()
    previous = -1.0
    for age in range(0, 130, 10):
        plan = plan_field(
            event, station, climatology,
            as_of=event.sown_on + timedelta(days=age), with_season=False,
        )
        assert plan.gdd_accumulated >= previous
        previous = plan.gdd_accumulated


def test_stage_progresses_initial_to_mid(weather):
    """The core requirement: a baby plant and a bigger plant differ in stage."""
    station = StationWeather(weather, "HYD")
    clim = build_rain_climatology(weather, "HYD")
    event = _event(
        field_id="C-1", station="HYD", crop="Cotton",
        sowing_date="2024-05-25", soil_type="Sandy_Loam", method_name="Drip",
    )
    stages = [
        plan_field(
            event, station, clim,
            as_of=event.sown_on + timedelta(days=age), with_season=False,
        )
        for age in range(0, 200, 20)
    ]
    assert stages[0].stage == "initial"
    assert any(s.stage == "mid_season" for s in stages)
    # root depth and Kc both rise with the crop, which is what makes a seedling
    # and a full canopy need different water on the same day
    assert stages[0].root_depth_cm < stages[4].root_depth_cm
    assert stages[0].kc < stages[4].kc
    assert stages[0].taw_mm < stages[4].taw_mm


def test_requirement_respects_application_cap(station, climatology):
    """A large requirement must be reported uncapped, and applied capped."""
    event = _event()
    plan = plan_field(
        event, station, climatology,
        as_of=date(2024, 3, 20), depletion_mm=150.0, with_season=False,
    )
    req = plan.requirement
    assert req.net_requirement_mm >= 0
    assert req.single_application_mm <= req.net_requirement_mm + 1e-6
    assert req.single_application_mm <= req.max_application_mm + 1e-6
    if req.requires_multiple_applications:
        assert req.single_application_mm < req.net_requirement_mm


def test_full_profile_never_needs_water(station, climatology):
    """A field at field capacity holds more than a fortnight of demand."""
    plan = plan_field(
        _event(), station, climatology,
        as_of=date(2024, 3, 20), depletion_mm=0.0, with_season=False,
    )
    assert plan.requirement.net_requirement_mm == pytest.approx(0.0, abs=1e-6)
    assert plan.days_until_stress is None


def test_drier_field_never_needs_less(station, climatology):
    """Monotone in depletion: less soil water cannot reduce the requirement."""
    as_of = date(2024, 3, 20)
    dry = 0.0
    wet = 0.0
    for depletion in (0.0, 40.0, 90.0, 150.0):
        req = plan_field(
            _event(), station, climatology,
            as_of=as_of, depletion_mm=depletion, with_season=False,
        ).requirement.net_requirement_mm
        if depletion == 0.0:
            dry = req
        else:
            assert req >= dry - 1e-6
            wet = req
    assert wet > dry


def test_gross_never_below_net(station, climatology):
    plan = plan_field(
        _event(), station, climatology,
        as_of=date(2024, 3, 20), depletion_mm=120.0, with_season=False,
    )
    assert plan.requirement.gross_requirement_mm >= plan.requirement.net_requirement_mm


def test_season_depletion_stays_within_bounds(station, climatology):
    event = _event()
    series = season_depletion(event, station, climatology)
    assert series[0][0] == event.sown_on
    for day, depletion in series:
        assert depletion >= 0.0
        assert depletion < 1000.0
    days = [d for d, _ in series]
    assert days == sorted(days)
    assert len(set(days)) == len(days)


def test_season_windows_are_contiguous_and_cover_the_season(station, climatology):
    plan = plan_field(_event(), station, climatology, with_season=True)
    assert plan.stage_windows
    for previous, following in zip(plan.stage_windows, plan.stage_windows[1:]):
        assert previous.stage != following.stage
        assert previous.days > 0
    assert plan.season_days == sum(w.days for w in plan.stage_windows)
    assert plan.season_gross_requirement_mm == pytest.approx(
        sum(w.gross_requirement_mm for w in plan.stage_windows), abs=0.1
    )


def test_season_total_is_a_depth_and_volume_needs_the_area(station, climatology):
    """Depth is per unit area and area-independent; volume is not."""
    small = plan_field(
        _event(field_area_m2=1_000.0), station, climatology, with_season=True
    )
    large = plan_field(
        _event(field_area_m2=10_000.0), station, climatology, with_season=True
    )
    assert small.season_gross_requirement_mm == pytest.approx(
        large.season_gross_requirement_mm, rel=1e-6
    )
    assert small.season_gross_volume_m3 * 10 == pytest.approx(
        large.season_gross_volume_m3, rel=1e-3
    )
    # 1 mm of depth over 1 m2 is 0.001 m3, so volume = depth * area / 1000
    assert small.season_gross_requirement_mm * 1_000.0 / 1_000.0 == pytest.approx(
        small.season_gross_volume_m3, rel=1e-3
    )


def test_plan_registry_covers_every_field(weather):
    registry = SowingRegistry()
    registry.register(_event(field_id="A", station="LKO", crop="Wheat"))
    registry.register(
        _event(field_id="B", station="PNQ", crop="Maize",
               sowing_date="2024-06-20")
    )
    plans = plan_registry(registry, weather, as_of=date(2024, 7, 5))
    assert {p.field_id for p in plans} == {"A", "B"}
    assert all(p.as_of == "2024-07-05" for p in plans)


def test_summary_lines_mention_the_action(station, climatology):
    lines = plan_field(
        _event(), station, climatology,
        as_of=date(2024, 3, 20), depletion_mm=200.0, with_season=False,
    ).summary_lines()
    assert any("ACTION" in line for line in lines)
    assert any("sown 2023-11-28" in line for line in lines)


def test_a_harvested_field_is_never_asked_to_be_irrigated(station, climatology):
    """
    Past harvest there is no crop drawing water, so the requirement is zero.

    Unhandled, the late-season tables went on producing a Kc and a root depth for a
    field whose grain was already off, and the plan recommended 29 mm of
    irrigation to harvested wheat. A recommendation to water a field with nothing
    growing in it is the kind of error a grower stops trusting the tool over.
    """
    plan = plan_field(
        _event(), station, climatology,
        as_of=date(2024, 4, 20), depletion_mm=0.0, with_season=False,
    )
    assert plan.stage == "post_harvest"
    assert plan.requirement.net_requirement_mm == pytest.approx(0.0, abs=0.05)
    assert plan.days_until_stress is None
    if plan.schedule is not None:
        assert plan.schedule.total_net_mm == pytest.approx(0.0, abs=0.05)
        assert plan.schedule.total_events == 0


def test_the_same_field_still_needs_water_before_harvest(station, climatology):
    """The guard must not leak backwards into the live part of the season."""
    plan = plan_field(
        _event(), station, climatology,
        as_of=date(2024, 3, 5), depletion_mm=300.0, with_season=False,
    )
    assert plan.stage != "post_harvest"
    assert plan.requirement.net_requirement_mm > 1.0
    assert plan.schedule is not None
    assert plan.schedule.total_net_mm > 0.0
