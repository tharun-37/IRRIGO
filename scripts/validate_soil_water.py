r"""
The check this project most needed and did not have: does the simulated soil
water state look like the real one?

Everything upstream of this is atmosphere. ET0 has now been checked against an
independent implementation and agrees to 0.2% on the mean, which validates the
arithmetic. But the soil water state is a different matter. It comes out of the
water balance, there is no public dataset of volumetric water content beneath a
named crop, and until now nothing in the repository has examined it. A water
balance that is subtly wrong produces plausible numbers, and plausible numbers
are exactly what a validation suite that only checks internal consistency will
happily confirm.

The external source provides observed soil moisture from ERA5, in three layers:
0-7 cm, 7-28 cm and 28-100 cm. That permits two comparisons.

**Seasonal shape.** Does the simulated root-zone moisture rise when it rains and
fall when the crop transpires? This is a correlation over time, within a field
and crop, and it is the comparison with real diagnostic power. A water balance
with the wrong sign, the wrong infiltration, or a missing rain term will fail it
regardless of how well its constants are tuned.

**Depth ordering.** Observed moisture should increase with depth in most of
India, because the surface loses water to evaporation and to the crop while the
subsoil is insulated by the residue and the previous season's water. Getting that
ordering wrong in the simulation would mean the root zone is being modelled as
draining from the top down when in reality the profile is comparatively stable
and the surface layer is the volatile one.

What is deliberately *not* attempted is calibration. Fitting the simulated
profile to ERA5 would be fitting to a 31 km reanalysis of the top metre of a
generic vegetated surface, and would replace a transparent physical model with
an opaque correction that transfers badly. Agreement is evidence. Disagreement is
a question. Both are reported; neither is silently resolved.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irrigation.data.external import load_external  # noqa: E402

#: The corpus column holding the simulated root-zone volumetric water content.
SIMULATED = "soil_moisture_fraction"

#: ERA5 layers, in the order they are reported.
LAYERS = ("soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
          "soil_moisture_28_to_100cm")


def _root_zone_proxy(frame: pd.DataFrame) -> pd.Series:
    """
    A depth-weighted ERA5 moisture for the layer the crop's roots actually reach.

    Weighted by layer thickness where the layers overlap the root zone, which is
    not the same as a plain mean. The 28-100 cm layer is a metre thick and the
    other two are 7 and 21 cm, so an unweighted mean of the three would let the
    deep layer dominate a wheat root zone that barely reaches it.
    """
    available = [c for c in LAYERS if c in frame.columns]
    if not available:
        return pd.Series(np.nan, index=frame.index)
    thickness = np.array(
        [7.0 if "0_to_7" in c else 21.0 if "7_to_28" in c else 72.0 for c in available]
    )
    values = frame[available].apply(pd.to_numeric, errors="coerce").to_numpy()
    # Where a layer is missing, fall back to the mean of the layers that are
    # present rather than dropping the row: ERA5 omits the deep layer in the
    # first years of the reanalysis, and a whole row lost per missing layer would
    # quietly bias the sample towards recent years.
    # np.errstate does not suppress the "Mean of empty slice" RuntimeWarning that
    # numpy raises for all-NaN rows, so the warning filter is needed as well. The
    # rows it complains about are handled explicitly on the next line.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fill = np.nanmean(values, axis=1, keepdims=True)
    all_missing = np.all(np.isnan(values), axis=1)
    fill[all_missing] = np.nan
    values = np.where(np.isnan(values), fill, values)
    return pd.Series(
        np.nansum(values * thickness, axis=1) / thickness.sum(), index=frame.index
    )


def _within_series_correlation(
    simulated: pd.Series, observed: pd.Series, group: pd.Series
) -> dict:
    """
    Correlation of the seasonal shape, computed within each field series.

    Pooling every station into one correlation would be dominated by the
    differences *between* fields, which is the lookup-table skill this project has
    already established it does not need to demonstrate. What is being asked is
    narrower and more useful: within one field, over one season, does the
    simulation move the way the observation moves. That is de-meaned per series,
    which removes the between-field offset and leaves only the response to rain
    and transpiration.
    """
    data = pd.DataFrame(
        {"sim": simulated, "obs": observed, "group": group}
    ).dropna()
    if data["group"].nunique() < 3:
        return {"note": "too few series to de-mean"}
    grouped = data.groupby("group")
    if grouped["sim"].std().mean() == 0 or grouped["obs"].std().mean() == 0:
        return {"note": "no within-series variation"}
    sim_dm = data["sim"] - grouped["sim"].transform("mean")
    obs_dm = data["obs"] - grouped["obs"].transform("mean")
    per_series = {}
    for key, index in data.groupby("group").groups.items():
        if index.size < 30:
            continue
        s, o = sim_dm.loc[index], obs_dm.loc[index]
        if s.std() == 0 or o.std() == 0:
            continue
        per_series[str(key)] = float(s.corr(o))
    if not per_series:
        return {"note": "no series long enough to correlate"}
    values = np.array(list(per_series.values()))
    return {
        "series_scored": int(values.size),
        "mean_within_series_r": float(values.mean()),
        "median_within_series_r": float(np.median(values)),
        "min_within_series_r": float(values.min()),
        "max_within_series_r": float(values.max()),
        "share_positive": float((values > 0).mean()),
    }


def analyse(corpus: pd.DataFrame, external: pd.DataFrame) -> dict:
    soil_columns = [c for c in LAYERS if c in external.columns]
    if not soil_columns:
        return {"error": "external cache has no soil moisture columns"}

    keep = ["station", "date"] + soil_columns
    observed = external[keep].copy()
    observed["date"] = pd.to_datetime(observed["date"])
    observed["obs_root_zone"] = _root_zone_proxy(observed)
    observed["obs_surface"] = pd.to_numeric(
        observed[soil_columns[0]], errors="coerce"
    )

    data = corpus.copy()
    data["date"] = pd.to_datetime(data["date"])
    merged = data.merge(observed, on=["station", "date"], how="inner")
    merged["series"] = (
        merged["station"].astype(str)
        + "|" + merged["crop"].astype(str)
        + "|" + merged["soil_type"].astype(str)
        + "|" + merged["year"].astype(str)
    )

    report: dict = {
        "n_corpus_rows": int(len(data)),
        "n_paired_rows": int(len(merged)),
        "paired_fraction": float(len(merged) / max(1, len(data))),
    }
    if merged.empty:
        report["error"] = "no overlapping dates between corpus and observations"
        return report

    report["simulated_vs_observed"] = {
        "pooled_r": float(
            merged[SIMULATED].corr(merged["obs_root_zone"])
        ),
        "pooled_r_surface_layer": float(
            merged[SIMULATED].corr(merged["obs_surface"])
        ),
        "within_series": _within_series_correlation(
            merged[SIMULATED], merged["obs_root_zone"], merged["series"]
        ),
        "simulated_mean": float(merged[SIMULATED].mean()),
        "observed_root_zone_mean": float(merged["obs_root_zone"].mean()),
        "observed_surface_mean": float(merged["obs_surface"].mean()),
    }

    # --- depth ordering, the physical sign test
    profile = merged[soil_columns].mean()
    report["depth_ordering"] = {
        "layer_means": {c: float(profile[c]) for c in soil_columns},
        "monotonic_with_depth": bool(
            all(profile[a] <= profile[b] for a, b in zip(soil_columns, soil_columns[1:]))
        ),
        "note": (
            "surface should be drier than subsoil in most of these climates;"
            " an inversion means the reanalysis profile is not behaving as a "
            "root-zone proxy here and the depth test should not be over-read"
        ),
    }

    # --- per station
    per_station = {}
    for station, group in merged.groupby("station"):
        entry = {
            "n": int(len(group)),
            "r": float(group[SIMULATED].corr(group["obs_root_zone"])),
            "simulated_mean": float(group[SIMULATED].mean()),
            "observed_mean": float(group["obs_root_zone"].mean()),
            "needs_water_pct": float((group["nir_horizon_mm"] > 1.0).mean() * 100.0),
        }
        entry["within_series"] = _within_series_correlation(
            group[SIMULATED], group["obs_root_zone"], group["series"]
        )
        per_station[station] = entry
    report["per_station"] = per_station

    # --- does agreement track whether the recommendation actually depends on water
    per_stage = {}
    for stage, group in merged.groupby("growth_stage"):
        per_stage[str(stage)] = {
            "n": int(len(group)),
            "r": float(group[SIMULATED].corr(group["obs_root_zone"])),
        }
    report["per_stage"] = per_stage

    return report


def print_report(report: dict) -> None:
    if "error" in report:
        print(f"  {report['error']}")
        return
    print(f"  corpus rows {report['n_corpus_rows']:,}, "
          f"paired with observations {report['n_paired_rows']:,} "
          f"({report['paired_fraction']:.1%})")
    s = report["simulated_vs_observed"]
    print(f"\n  simulated mean {s['simulated_mean']:.4f} m3/m3")
    print(f"  observed root-zone proxy mean {s['observed_root_zone_mean']:.4f} m3/m3")
    print(f"  observed 0-7 cm mean {s['observed_surface_mean']:.4f} m3/m3")
    print(f"\n  pooled r (simulated vs observed root zone): {s['pooled_r']:+.4f}")
    print(f"  pooled r (simulated vs observed 0-7 cm):     {s['pooled_r_surface_layer']:+.4f}")
    w = s["within_series"]
    if "mean_within_series_r" in w:
        print(f"\n  within-series seasonal shape, {w['series_scored']} series:")
        print(f"    mean r {w['mean_within_series_r']:+.4f}  "
              f"median {w['median_within_series_r']:+.4f}  "
              f"range {w['min_within_series_r']:+.3f} to {w['max_within_series_r']:+.3f}")
        print(f"    share of series positively correlated: {w['share_positive']:.1%}")
    else:
        print(f"  within-series: {w.get('note')}")

    d = report["depth_ordering"]
    print("\n  observed depth profile (should be monotonic increasing):")
    for layer, mean in d["layer_means"].items():
        print(f"    {layer:<26} {mean:.4f}")
    print(f"    monotonic with depth: {d['monotonic_with_depth']}")

    print("\n  per station:")
    print(f"    {'station':<9}{'n':>8}{'r':>9}{'sim mean':>10}{'obs mean':>10}{'within-r':>11}")
    for station, entry in report["per_station"].items():
        w = entry["within_series"]
        within = (
            f"{w['mean_within_series_r']:+.3f}" if "mean_within_series_r" in w else "  --"
        )
        print(f"    {station:<9}{entry['n']:>8,}{entry['r']:>+9.4f}"
              f"{entry['simulated_mean']:>10.4f}{entry['observed_mean']:>10.4f}"
              f"{within:>11}")

    print("\n  per growth stage:")
    for stage, entry in report["per_stage"].items():
        print(f"    {stage:<14} n {entry['n']:>7,}  r {entry['r']:+.4f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(ROOT / "data" / "crop_corpus.csv"))
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache" / "external"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "soil_validation.json"))
    args = parser.parse_args(argv)

    print("loading corpus ...")
    corpus = pd.read_csv(args.corpus, low_memory=False)
    print("loading observations ...")
    external = load_external(sorted(corpus["station"].unique()), args.cache_dir)
    if external.empty:
        print("no observations; run scripts/fetch_external.py first", file=sys.stderr)
        return 1

    print("\n=== simulated soil water vs observed soil moisture ===")
    report = analyse(corpus, external)
    print_report(report)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
