r"""
Compare the project's own numbers against the independent observational source.

Three comparisons, in increasing order of how much they should worry us.

**ET0 against ERA5.** The project computes reference evapotranspiration from
NASA POWER meteorology using its own FAO-56 Penman-Monteith implementation. The
external source computes the same equation from ERA5 with a different
implementation. Two independent routes to the same physical quantity should
agree to within the quality of the inputs. This validates the arithmetic and, more
usefully, localises any disagreement: the script decomposes the gap into the
humidity, wind, radiation and temperature differences, so a bias can be traced to
a meteorological variable rather than merely observed.

**Rainfall against ERA5.** The project's rain comes from POWER's
`PRECTOTCORR`, which is bias-corrected but still satellite-merged. Rainfall
drives the whole credit term in the requirement, so a bias here is a bias in
every recommendation. Comparing against ERA5 also gives the annual totals, where a
systematic multiplier is visible even if the daily correlation is poor.

**Simulated soil water against observed soil water.** This is the one that
matters most and the one most likely to look bad. The soil water state is purely
simulated: there is no public dataset of volumetric water content under a
specific crop, so it comes out of the water balance. ERA5's top layer is a
reanalysis of a real quantity at the right order of magnitude, which makes it the
best available independent check. Disagreement here is not a nuisance; it is
information about whether the water balance is right.

A correlation over a season is reported rather than a calibrated match, because
the two quantities are not the same quantity. ERA5's 0-7 cm layer sits above most
of the root zone, responds faster, and is smooth at a 31 km scale. Expect
correlation in the 0.2-0.6 range from a correct implementation. A correlation near
zero is a genuine failure signal; a correlation of 0.9 would mean something has
gone wrong in the comparison rather than gone right in the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irrigation.data.external import load_external  # noqa: E402
from irrigation.physics.climate import load_weather  # noqa: E402


def _align(weather: pd.DataFrame, external: pd.DataFrame, station: str) -> pd.DataFrame:
    left = weather[weather["station"] == station][
        ["station", "date", "et0_mm_day", "rainfall_mm", "humidity_pct",
         "wind_speed_m_s", "solar_radiation_mj_m2_day", "tmax_c", "tmin_c",
         "air_temperature_c"]
    ].copy()
    right = external[external["station"] == station][
        ["station", "date", "et0_fao_evapotranspiration", "precipitation_sum",
         "relative_humidity_2m_mean", "wind_speed_10m_mean",
         "shortwave_radiation_sum", "temperature_2m_max", "temperature_2m_min"]
    ].copy()
    if "soil_moisture_0_to_7cm" in external.columns:
        soil = external[external["station"] == station][
            ["station", "date", "soil_moisture_0_to_7cm",
             "soil_moisture_7_to_28cm", "soil_moisture_28_to_100cm",
             "soil_temperature_0_to_7cm"]
        ]
        right = right.merge(soil, on=["station", "date"], how="left")
    merged = left.merge(right, on=["station", "date"], how="inner")
    return merged


def _paired_stats(a: pd.Series, b: pd.Series, scale_note: str | None = None) -> dict:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    mask = a.notna() & b.notna()
    a, b = a[mask], b[mask]
    if a.size < 30:
        return {"n": int(a.size), "note": "too few paired records"}
    diff = b - a
    stats = {
        "n": int(a.size),
        "pearson_r": float(a.corr(b)),
        "mean_project": float(a.mean()),
        "mean_external": float(b.mean()),
        "bias_external_minus_project": float(diff.mean()),
        "ratio_external_over_project": float(b.mean() / max(1e-9, a.mean())),
        "mae": float(diff.abs().mean()),
        "p90_abs_error": float(diff.abs().quantile(0.90)),
    }
    if scale_note:
        stats["scale_note"] = scale_note
    return stats


def compare(weather: pd.DataFrame, external: pd.DataFrame) -> dict:
    report: dict = {"per_station": {}}
    shared = sorted(set(weather["station"]) & set(external["station"]))
    for station in shared:
        frame = _align(weather, external, station)
        entry = {
            "et0": _paired_stats(frame["et0_mm_day"], frame["et0_fao_evapotranspiration"]),
            "rain": _paired_stats(frame["rainfall_mm"], frame["precipitation_sum"]),
            "inputs": {
                "humidity_pct": _paired_stats(
                    frame["humidity_pct"], frame["relative_humidity_2m_mean"]
                ),
                # Open-Meteo reports the daily 10 m wind mean in km/h. Comparing it
                # to the project's m/s without converting produces a 3.6x "bias" that
                # is pure units and would point the ET0 diagnosis at the wrong input.
                "wind_speed_m_s": _paired_stats(
                    frame["wind_speed_m_s"],
                    frame["wind_speed_10m_mean"] / 3.6,
                    scale_note="external divided by 3.6 from km/h to m/s",
                ),
                "solar_radiation_mj_m2_day": _paired_stats(
                    frame["solar_radiation_mj_m2_day"], frame["shortwave_radiation_sum"]
                ),
                "tmax_c": _paired_stats(frame["tmax_c"], frame["temperature_2m_max"]),
            },
        }
        for depth in ("0_to_7cm", "7_to_28cm", "28_to_100cm"):
            column = f"soil_moisture_{depth}"
            if column in frame.columns:
                entry[f"soil_moisture_{depth}_stats"] = {
                    "mean": float(frame[column].mean()),
                    "min": float(frame[column].min()),
                    "max": float(frame[column].max()),
                }
        if "soil_moisture_0_to_7cm" in frame.columns:
            entry["soil_depth_profile"] = {
                depth: float(frame[f"soil_moisture_{depth}"].mean())
                for depth in ("0_to_7cm", "7_to_28cm", "28_to_100cm")
                if f"soil_moisture_{depth}" in frame.columns
            }
        report["per_station"][station] = entry

    pooled = []
    for station, entry in report["per_station"].items():
        for key in ("et0", "rain"):
            if "pearson_r" in entry[key]:
                pooled.append(
                    {
                        "station": station,
                        "variable": key,
                        "r": entry[key]["pearson_r"],
                        "bias": entry[key]["bias_external_minus_project"],
                        "ratio": entry[key]["ratio_external_over_project"],
                        "mae": entry[key]["mae"],
                    }
                )
    pooled_frame = pd.DataFrame(pooled)
    summary = {}
    for key in ("et0", "rain"):
        subset = pooled_frame[pooled_frame["variable"] == key]
        if subset.empty:
            continue
        summary[key] = {
            "stations": int(len(subset)),
            "mean_pearson_r": float(subset["r"].mean()),
            "min_pearson_r": float(subset["r"].min()),
            "mean_bias": float(subset["bias"].mean()),
            "mean_ratio": float(subset["ratio"].mean()),
            "worst_station": subset.loc[subset["r"].idxmin(), "station"],
        }
    report["summary"] = summary
    return report


def print_report(report: dict) -> None:
    print("\n=== ET0: project (POWER + FAO-56) vs external (ERA5) ===")
    print(f"{'station':<9}{'n':>7}{'r':>8}{'bias mm/d':>12}{'ratio':>9}{'MAE':>8}")
    for station, entry in report["per_station"].items():
        e = entry["et0"]
        if "pearson_r" not in e:
            print(f"{station:<9}{e.get('n', 0):>7}   insufficient overlap")
            continue
        print(f"{station:<9}{e['n']:>7,}{e['pearson_r']:>8.3f}"
              f"{e['bias_external_minus_project']:>12.3f}"
              f"{e['ratio_external_over_project']:>9.3f}{e['mae']:>8.3f}")
    summary = report["summary"].get("et0")
    if summary:
        print(f"\n  mean r {summary['mean_pearson_r']:.3f} across "
              f"{summary['stations']} stations; worst {summary['worst_station']} "
              f"at r {summary['min_pearson_r']:.3f}")
        print(f"  ERA5 is {summary['mean_ratio']:.3f}x the project's ET0 on average "
              f"(bias {summary['mean_bias']:+.3f} mm/day)")

    print("\n=== rainfall: project (POWER PRECTOTCORR) vs external (ERA5) ===")
    print(f"{'station':<9}{'n':>7}{'r':>8}{'bias mm/d':>12}{'ratio':>9}")
    for station, entry in report["per_station"].items():
        e = entry["rain"]
        if "pearson_r" not in e:
            print(f"{station:<9}{e.get('n', 0):>7}   insufficient overlap")
            continue
        print(f"{station:<9}{e['n']:>7,}{e['pearson_r']:>8.3f}"
              f"{e['bias_external_minus_project']:>12.3f}"
              f"{e['ratio_external_over_project']:>9.3f}")
    rain_summary = report["summary"].get("rain")
    if rain_summary:
        print(f"\n  mean r {rain_summary['mean_pearson_r']:.3f}; "
              f"ERA5 is {rain_summary['mean_ratio']:.3f}x the project's rainfall")

    print("\n=== the inputs behind ET0 (where any ET0 gap comes from) ===")
    print(f"{'station':<9}{'humidity':>10}{'wind':>10}{'radiation':>11}{'tmax':>9}")
    for station, entry in report["per_station"].items():
        inputs = entry["inputs"]
        cells = []
        for key, width in (("humidity_pct", 10), ("wind_speed_m_s", 10),
                           ("solar_radiation_mj_m2_day", 11), ("tmax_c", 9)):
            value = inputs[key].get("pearson_r")
            cells.append(f"{value:>{width}.3f}" if value is not None
                         else f"{'--':>{width}}")
        print(f"{station:<9}" + "".join(cells))

    print("\n=== observed soil moisture depth profile (ERA5, m3/m3) ===")
    print(f"{'station':<9}{'0-7cm':>10}{'7-28cm':>10}{'28-100cm':>11}")
    for station, entry in report["per_station"].items():
        profile = entry.get("soil_depth_profile")
        if not profile:
            continue
        print(f"{station:<9}{profile.get('0_to_7cm', float('nan')):>10.3f}"
              f"{profile.get('7_to_28cm', float('nan')):>10.3f}"
              f"{profile.get('28_to_100cm', float('nan')):>11.3f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache" / "external"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "external_validation.json"))
    args = parser.parse_args(argv)

    print("loading project climate (POWER) ...")
    weather = load_weather(2015, 2024)
    print("loading external climate (ERA5) ...")
    external = load_external(sorted(weather["station"].unique()), args.cache_dir)
    if external.empty:
        print("no external data; run scripts/fetch_external.py first", file=sys.stderr)
        return 1
    print(f"  {len(external):,} external rows, "
          f"{external['station'].nunique()} stations")

    report = compare(weather, external)
    print_report(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
