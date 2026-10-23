"""Fetch the independent observational source for every station.

Usage
  python scripts/fetch_external.py
  python scripts/fetch_external.py --stations PNQ,HYD --start-year 2020 --end-year 2024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irrigation.data.external import (  # noqa: E402
    StationPoint,
    describe_provenance,
    fetch_open_meteo,
)
from irrigation.physics.climate import load_weather  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", default=None, help="comma separated")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache" / "external"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    print("reading station coordinates from the POWER cache ...")
    weather = load_weather(2015, 2024)
    if args.stations:
        names = [s.strip() for s in args.stations.split(",") if s.strip()]
        weather = weather[weather["station"].isin(names)]
    stations = [
        StationPoint(name, float(g["latitude"].iloc[0]), float(g["longitude"].iloc[0]),
                     float(g["elevation_m"].iloc[0]) if "elevation_m" in g else None)
        for name, g in weather.groupby("station")
    ]
    print(f"  {len(stations)} stations: {', '.join(s.name for s in stations)}\n")

    started = time.time()
    frames = []
    for index, station in enumerate(stations, start=1):
        print(f"  [{index}/{len(stations)}] {station.name}", flush=True)
        try:
            frames.append(
                fetch_open_meteo(
                    station, args.start_year, args.end_year,
                    cache_dir=args.cache_dir, force=args.force,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one station must not kill the run
            print(f"    FAILED: {exc}")

    if not frames:
        print("no external data retrieved", file=sys.stderr)
        return 1

    frame = pd.concat(frames, ignore_index=True)
    target = Path(args.cache_dir) / "external_all.csv"
    frame.to_csv(target, index=False)

    print(f"\nwrote {len(frame):,} rows to {target}")
    has_soil = "soil_moisture_0_to_7cm" in frame.columns
    print(f"  soil moisture present: {has_soil}")
    if has_soil:
        print(f"  0-7cm soil moisture range: "
              f"{frame['soil_moisture_0_to_7cm'].min():.3f} to "
              f"{frame['soil_moisture_0_to_7cm'].max():.3f} m3/m3")

    provenance = Path(args.cache_dir) / "provenance.json"
    provenance.write_text(json.dumps(describe_provenance(), indent=2), encoding="utf-8")
    print(f"  provenance -> {provenance}")
    print(f"  total {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
