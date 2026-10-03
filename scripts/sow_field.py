"""CLI to register sowings and read the resulting water plan.

Commands
  demo    register a few example fields and print their plans
  add     register one sowing
  list    show registered fields
  plan    print the plan for a field (or all fields)
  season  print the full season calendar and cost for a field

Run `python scripts/sow_field.py demo` to see it work with no setup.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from irrigation.physics.climate import load_weather  # noqa: E402
from irrigation.sowing import (  # noqa: E402
    SowingError,
    SowingEvent,
    SowingRegistry,
    plan_field,
    plan_registry,
    season_depletion,
    StationWeather,
)

DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "data" / "sowings.json"


def _weather():
    return load_weather(2015, 2024)


def _print_season(plan) -> None:
    print()
    print(f"  season calendar for {plan.field_id} ({plan.crop})")
    print(f"  {'stage':<13}{'from':<12}{'to':<12}{'days':>5}{'Kc':>6}"
          f"{'ETc/d':>7}{'rain/d':>8}{'gross mm':>10}")
    for w in plan.stage_windows:
        print(f"  {w.stage:<13}{w.start_date:<12}{w.end_date:<12}{w.days:>5}"
              f"{w.mean_kc:>6.2f}{w.mean_etc_mm_day:>7.2f}{w.mean_rain_mm_day:>8.2f}"
              f"{w.gross_requirement_mm:>10.0f}")
    print(f"  {'TOTAL':<13}{'':<12}{'':<12}{plan.season_days:>5}{'':>6}{'':>7}"
          f"{'':>8}{plan.season_gross_requirement_mm:>10.0f}")
    print(f"  season rainfall {plan.season_rainfall_mm:.0f} mm;"
          f" gross depth over the field {plan.season_gross_requirement_mm:.1f} mm"
          f" = {plan.season_gross_volume_m3:.0f} m3 over the whole field")


def cmd_demo(args) -> int:
    registry = SowingRegistry(args.registry)
    for event in (
        SowingEvent(field_id="PNQ-wheat-01", station="PNQ", crop="Wheat",
                    sowing_date="2023-11-15", soil_type="Clay",
                    method_name="Furrow", field_area_m2=4_000),
        SowingEvent(field_id="PNQ-maize-01", station="PNQ", crop="Maize",
                    sowing_date="2024-06-20", soil_type="Loam",
                    method_name="Sprinkler", field_area_m2=2_500, mulched=True),
        SowingEvent(field_id="LKO-wheat-02", station="LKO", crop="Wheat",
                    sowing_date="2023-11-28", soil_type="Silt_Loam",
                    method_name="Basin", field_area_m2=6_000),
        SowingEvent(field_id="HYD-cotton-01", station="HYD", crop="Cotton",
                    sowing_date="2024-05-25", soil_type="Sandy Loam",
                    method_name="Drip", field_area_m2=1_800),
    ):
        registry.register(event, replace=True)

    weather = _weather()
    as_of = date(2024, 7, 5)
    plans = plan_registry(registry, weather, as_of=as_of, with_season=True)
    for plan in plans:
        print()
        for line in plan.summary_lines():
            print("  " + line)
        _print_season(plan)
    registry.save()
    print(f"\n  saved {len(registry)} sowings to {registry.path}")
    return 0


def cmd_add(args) -> int:
    registry = SowingRegistry.load(args.registry)
    registry.register(
        SowingEvent(
            field_id=args.field_id, station=args.station, crop=args.crop,
            sowing_date=args.sowing_date, soil_type=args.soil,
            crop_variant=args.variant, method_name=args.method,
            field_area_m2=args.area, mulched=args.mulched,
            nitrogen_regime=args.nitrogen, restricting_depth_m=args.restricting_depth,
        ),
        replace=args.replace,
    )
    registry.save()
    print(f"registered {args.field_id}")
    return 0


def cmd_list(args) -> int:
    registry = SowingRegistry.load(args.registry)
    if not len(registry):
        print("no sowings registered")
        return 0
    print(f"{'field':<18}{'station':<9}{'crop':<11}{'sown':<12}"
          f"{'soil':<14}{'method':<11}{'area m2':>10}")
    for e in registry.list():
        print(f"{e.field_id:<18}{e.station:<9}{e.crop:<11}{e.sowing_date:<12}"
              f"{e.soil_type:<14}{e.method_name:<11}{e.field_area_m2:>10,.0f}")
    return 0


def cmd_plan(args) -> int:
    registry = SowingRegistry.load(args.registry)
    weather = _weather()
    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else None
    if args.field:
        event = registry.get(args.field)
        station = StationWeather(weather, event.station)
        from irrigation.physics.climate import build_rain_climatology
        plan = plan_field(
            event, station, build_rain_climatology(weather, event.station),
            as_of=as_of, depletion_mm=args.depletion,
        )
        for line in plan.summary_lines():
            print(line)
        if args.season:
            _print_season(plan)
    else:
        plans = plan_registry(
            registry, weather, as_of=as_of, depletion_mm=args.depletion
        )
        for plan in plans:
            print()
            for line in plan.summary_lines():
                print("  " + line)
    return 0


def cmd_season(args) -> int:
    registry = SowingRegistry.load(args.registry)
    weather = _weather()
    from irrigation.physics.climate import build_rain_climatology
    event = registry.get(args.field)
    station = StationWeather(weather, event.station)
    plan = plan_field(
        event, station, build_rain_climatology(weather, event.station),
        as_of=args.as_of, depletion_mm=args.depletion, with_season=True,
    )
    for line in plan.summary_lines():
        print(line)
    _print_season(plan)
    return 0


def cmd_compare(args) -> int:
    """The same sowing at several ages, side by side.

    This is the view that answers "does the system actually treat a seedling
    differently from a full plant". It holds the crop, soil, station and method
    fixed and moves only the day count, so any difference in the recommendation
    comes from growth stage and root depth alone.
    """
    registry = SowingRegistry.load(args.registry)
    weather = _weather()
    from irrigation.physics.climate import build_rain_climatology
    event = registry.get(args.field)
    station = StationWeather(weather, event.station)
    clim = build_rain_climatology(weather, event.station)
    days = [int(x) for x in args.days.split(",")]

    print(f"{event.field_id}  {event.crop} / {event.soil_type} / {event.method_name}"
          f"  at {event.station}  sown {event.sowing_date}")
    print("depletion is the value the season actually reached under the operator's"
          " own rule; forecast ET0 is the station's monthly climatology\n")
    header = (f"{'day':>4} {'stage':<12}{'GDD':>7}{'Kc':>6}{'root cm':>9}"
              f"{'TAW':>7}{'depl':>7}{'need mm':>10}{'apply':>8}{'gross':>8}{'stress in':>11}")
    print(header)
    print("-" * len(header))
    depletion = season_depletion(event, station, clim, as_of=event.sown_on + timedelta(days=max(days)))
    by_day = {d: dep for d, dep in depletion}
    for d in days:
        as_of = event.sown_on + timedelta(days=d)
        current = by_day.get(as_of, 0.0)
        plan = plan_field(
            event, station, clim, as_of=as_of,
            depletion_mm=current, with_season=False,
        )
        stress = (
            f"{plan.days_until_stress}d" if plan.days_until_stress else "none"
        )
        print(f"{d:>4} {plan.stage:<12}{plan.gdd_accumulated:>7.0f}{plan.kc:>6.2f}"
              f"{plan.root_depth_cm:>9.0f}{plan.taw_mm:>7.0f}{current:>7.0f}"
              f"{plan.requirement.net_requirement_mm:>10.1f}"
              f"{plan.requirement.single_application_mm:>8.1f}"
              f"{plan.requirement.gross_requirement_mm:>8.1f}{stress:>11}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo"); p.set_defaults(func=cmd_demo)

    p = sub.add_parser("add")
    p.add_argument("--field-id", required=True)
    p.add_argument("--station", required=True)
    p.add_argument("--crop", required=True)
    p.add_argument("--sowing-date", required=True, help="YYYY-MM-DD")
    p.add_argument("--soil", default="Loam")
    p.add_argument("--variant", default=None)
    p.add_argument("--method", default="Sprinkler")
    p.add_argument("--area", type=float, default=1_000.0)
    p.add_argument("--mulched", action="store_true")
    p.add_argument("--nitrogen", type=float, default=1.0)
    p.add_argument("--restricting-depth", type=float, default=None)
    p.add_argument("--replace", action="store_true")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("list"); p.set_defaults(func=cmd_list)

    p = sub.add_parser("plan")
    p.add_argument("--field", default=None)
    p.add_argument("--as-of", default=None, help="YYYY-MM-DD")
    p.add_argument("--depletion", type=float, default=0.0)
    p.add_argument("--season", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("compare")
    p.add_argument("--field", required=True)
    p.add_argument("--days", default="3,10,25,45,70,95")
    p.add_argument("--depletion", type=float, default=0.0)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("season")
    p.add_argument("--field", required=True)
    p.add_argument("--as-of", default=None)
    p.add_argument("--depletion", type=float, default=0.0)
    p.set_defaults(func=cmd_season)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except SowingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
