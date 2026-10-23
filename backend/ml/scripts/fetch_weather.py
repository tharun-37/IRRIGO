"""
Fetch and cache the real NASA POWER weather corpus.

Run once before `train.py`. Subsequent runs are served from
`backend/ml/datasets/cache/`, so training is reproducible offline.

    python backend/ml/scripts/fetch_weather.py
    python backend/ml/scripts/fetch_weather.py --start-year 2010 --end-year 2024
    python backend/ml/scripts/fetch_weather.py --summary-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

import pandas as pd  # noqa: E402

from real_data import STATIONS, CACHE_DIR, climate_summary, load_real_weather  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the NASA POWER weather corpus.")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--summary-only", action="store_true",
                        help="report on the cache without downloading")
    args = parser.parse_args()

    cached = sorted(CACHE_DIR.glob("power_*.csv"))
    if args.summary_only and not cached:
        print(f"cache is empty: {CACHE_DIR}")
        return 1

    weather = load_real_weather(
        start_year=args.start_year, end_year=args.end_year, verbose=True
    )

    print("\n")
    print("=" * 92)
    print("NASA POWER agroclimatology corpus - climate summary")
    print("=" * 92)
    summary = climate_summary(weather)
    print(summary.to_string(index=False))

    print("\nDerived columns computed by this project")
    print("-" * 92)
    for column in ("et0_mm_day", "rn_mj_m2_day", "g_vapour_kpa", "ra_mj_m2_day", "elevation_m"):
        series = weather[column]
        print(f"  {column:<18} min {series.min():>9.3f}   mean {series.mean():>9.3f}   "
              f"max {series.max():>9.3f}")

    routes = weather["vapour_pressure_source"].value_counts() if "vapour_pressure_source" in weather else None
    if routes is not None:
        share = (routes / len(weather)).map("{:.2%}".format)
        print("\n  actual vapour pressure derivation (FAO-56 route)")
        for route, fraction in share.items():
            print(f"    {route:<12} {fraction}")

    print(f"\n  stations     {len(STATIONS)}")
    print(f"  records      {len(weather):,}")
    print(f"  span         {weather['date'].min():%Y-%m-%d} to {weather['date'].max():%Y-%m-%d}")
    print(f"  cache        {CACHE_DIR}")
    print("\n  validate the ET0 implementation against FAO-56 published examples:")
    print("    .venv\\Scripts\\python backend\\ml\\tests\\test_fao56.py")
    _ = pd, summary
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
