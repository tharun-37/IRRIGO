"""Rewrite the per-station aggregate caches from their yearly source files.

Run after adding years to a station so the convenience copies match the yearly
caches, which `load_external` treats as authoritative.
"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irrigation.data.external import _station_years, load_external  # noqa: E402
from irrigation.physics.climate import load_weather  # noqa: E402

CACHE = ROOT / "data" / "cache" / "external"

names = sorted(load_weather(2015, 2024)["station"].unique())
total = 0
for name in names:
    years = _station_years(CACHE, name)
    if not years:
        print("%-6s no yearly caches" % name)
        continue
    frame = pd.concat(
        [pd.read_csv(y, parse_dates=["date"]) for y in years], ignore_index=True
    )
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    has_soil = "soil_moisture_0_to_7cm" in frame.columns
    frame.to_csv(CACHE / ("open_meteo_%s.csv" % name), index=False)
    total += len(frame)
    print("%-6s %5d days  %s" % (name, len(frame), "soil ok" if has_soil else "NO SOIL"))

merged = load_external(names, CACHE)
merged.to_csv(CACHE / "external_all.csv", index=False)
print("total %d rows across %d stations" % (total, merged["station"].nunique()))
