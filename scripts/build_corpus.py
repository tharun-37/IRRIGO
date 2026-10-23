"""Build the V2 corpus and audit whether the target is fit to learn from."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

#: Project root, which is the parent of `scripts/`. Using `parent.parent`
#: rather than the script's own directory matters: the package lives in
#: `<root>/src`, and pointing sys.path at `scripts/` makes the import fail with
#: a message that points at the wrong directory entirely.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from irrigation.corpus import FieldSpec, build_corpus  # noqa: E402
from irrigation.data.crops import CROPS  # noqa: E402
from irrigation.data.soil import TEXTURES  # noqa: E402
from irrigation.physics.climate import load_weather, station_climate_profile  # noqa: E402

# Crops that make agronomic sense for the Indian stations in the record, paired
# with the soils they are actually grown on. Not every crop is assigned to every
# station: putting wheat in a Chennai paddy district or rice in arid Rajasthan
# would generate series that are agronomically meaningless, and a corpus full of
# impossible pairings teaches the model relationships that do not hold in a field.
STATION_CROPS: dict[str, list[str]] = {
    "HYD": ["Maize", "Soybean", "Sorghum"],
    "LUD": ["Wheat", "Maize", "Rice", "Cotton"],
    "CHN": ["Rice", "Sugarcane", "Groundnut"],
    "JAI": ["Sorghum", "Maize", "Wheat", "Cotton"],
    "NAG": ["Soybean", "Cotton", "Wheat"],
    "KOL": ["Rice", "Maize", "Groundnut"],
    "PNQ": ["Soybean", "Sugarcane", "Onion"],
    "AMD": ["Cotton", "Wheat", "Groundnut"],
    "BBN": ["Rice", "Groundnut", "Maize"],
    "LKO": ["Wheat", "Rice", "Potato"],
    "PNJ": ["Wheat", "Maize", "Rice", "Sunflower"],
    "IDR": ["Soybean", "Wheat", "Onion", "Soybean"],
}

#: Soils use the FAO-56 Table 19 texture class names. Where a station is
#: associated with a pedological soil series in practice, that series is mapped
#: to the texture class whose hydraulic behaviour matches it, and the mapping is
#: recorded here rather than being applied silently: black cotton soil is a
#: Vertisol with the swelling and cracking behaviour of a heavy clay, and calling
#: it "Clay_Loam" would understate its available water by a third.
STATION_SOILS: dict[str, list[str]] = {
    "HYD": ["Clay_Loam", "Sandy_Loam", "Clay"],
    "LUD": ["Loam", "Clay_Loam", "Sandy_Loam"],
    "CHN": ["Clay", "Clay_Loam", "Silt_Loam"],
    "JAI": ["Sandy_Loam", "Loam", "Sand"],
    "NAG": ["Clay", "Clay_Loam", "Sandy_Loam"],
    "KOL": ["Silt_Loam", "Clay_Loam", "Loam"],
    "PNQ": ["Clay_Loam", "Sandy_Loam", "Black_Cotton_Clay"],
    "AMD": ["Sandy_Loam", "Clay_Loam", "Loam"],
    "BBN": ["Silt_Loam", "Clay_Loam", "Sandy_Clay_Loam"],
    "LKO": ["Loam", "Clay_Loam", "Silt"],
    "PNJ": ["Loam", "Sandy_Loam", "Clay_Loam"],
    "IDR": ["Black_Cotton_Clay", "Clay_Loam", "Loam"],
}

#: Sowing windows as day-of-year, chosen to sit inside the station's own growing
#: season rather than being spread evenly across the year. A rice crop sown in
#: December at 20 degC accumulates almost no GDD and would appear to sit in its
#: initial stage for the whole record, which teaches the model nothing.
SOWING_WINDOWS: dict[str, tuple[int, int]] = {
    "HYD": (175, 230), "LUD": (270, 320), "CHN": (110, 180), "JAI": (200, 260),
    "NAG": (150, 200), "KOL": (110, 180), "PNQ": (120, 190), "AMD": (160, 230),
    "BBN": (110, 180), "LKO": (270, 320), "PNJ": (275, 320), "IDR": (185, 240),
}

METHODS = ["Furrow", "Sprinkler", "Drip", "Basin"]


def build_field_specs(weather: pd.DataFrame, max_fields: int | None = None) -> list[FieldSpec]:
    """Enumerate the field series, pairing each station with plausible crops and soils."""
    available_soils = set(TEXTURES)
    available_crops = set(CROPS)
    rng = np.random.default_rng(20240917)
    specs: list[FieldSpec] = []

    # Black cotton soil is a real pedological series common in central India, not
    # a FAO-56 texture class. It is mapped onto the heavy clay whose hydraulic
    # behaviour it shares, rather than dropped for want of an exact match.
    SOIL_ALIASES = {"Black_Cotton_Clay": "Clay"}

    for station in sorted(weather["station"].unique()):
        crops = [c for c in STATION_CROPS.get(station, []) if c in available_crops]
        soils = [
            SOIL_ALIASES.get(s, s)
            for s in STATION_SOILS.get(station, [])
            if SOIL_ALIASES.get(s, s) in available_soils
        ]
        low, high = SOWING_WINDOWS.get(station, (120, 200))
        if not crops or not soils:
            continue
        for sow in (low, high):
            for crop_name in crops:
                for soil_name in soils:
                    specs.append(
                        FieldSpec(
                            station=station,
                            crop_name=crop_name,
                            soil_name=soil_name,
                            method_name=str(rng.choice(METHODS)),
                            field_area_m2=float(
                                np.clip(rng.lognormal(np.log(1_500.0), 1.0), 200.0, 40_000.0)
                            ),
                            mulched=bool(rng.random() < 0.25),
                            nitrogen_regime=round(float(rng.uniform(0.55, 1.15)), 3),
                            sowing_doy=int(rng.integers(low, high + 1)),
                        )
                    )

    if max_fields is not None and len(specs) > max_fields:
        specs = specs[:max_fields]
    return specs


def audit(frame: pd.DataFrame) -> None:
    """The checks that decide whether this corpus can teach a transferable model."""
    print()
    print("=" * 78)
    print("CORPUS AUDIT")
    print("=" * 78)
    print(f"rows                 : {len(frame):,}")
    print(f"columns              : {len(frame.columns)}")
    print(f"stations             : {frame['station'].nunique()}")
    print(f"crops                : {frame['crop'].nunique()}")
    print(f"soils                : {frame['soil_type'].nunique()}")
    print(f"series               : {frame.groupby(['station','crop','soil_type']).ngroups}")

    target = frame["nir_horizon_mm"]
    nonzero = target[target > 1e-9]

    print()
    print("--- THE V1 DEFECT: is the target a per-soil lookup? ---")
    print(f"  distinct values    : {target.nunique():,}")
    print(f"  zero rows          : {(target <= 1e-9).mean() * 100:.2f}%   "
          f"({int((target <= 1e-9).sum()):,} rows)")

    # V1's corpus had almost no zeros (2.09% positives) and 39.13% of all events
    # sat on a single value, 22 mm. Comparing raw modal share against that would
    # be dishonest in both directions: V1 had no legitimate zero-inflation to hide
    # behind, and this corpus has a real one. The like-for-like number is the
    # modal share among rows that need water at all.
    if len(nonzero):
        nz_modal = float(nonzero.value_counts().max() / len(nonzero))
        nz_top = nonzero.value_counts().head(3)
    else:
        nz_modal, nz_top = float("nan"), pd.Series(dtype=int)
    print(f"  modal share among rows needing water : {nz_modal * 100:.2f}%   "
          f"(V1: 39.13% on 22 mm)")
    print(f"  top 3 non-zero values : {[f'{v:.2f}mm x{c:,}' for v, c in nz_top.items()]}")

    # The decisive test. If a per-soil median can reproduce the target, the target
    # carries no climate, no crop and no season, and a model fitted to it is a
    # lookup table. This is the single number that separates V2 from V1.
    lookup = frame.groupby("soil_type")["nir_horizon_mm"].median()
    predicted = frame["soil_type"].map(lookup)
    within_2 = float(((predicted - target).abs() <= 2.0).mean())
    nz_mask = target > 1e-9
    nz_within_2 = float(((predicted - target).abs() <= 2.0)[nz_mask].mean())
    print(f"  soil-median guess, all rows          : {within_2 * 100:5.2f}% within 2 mm"
          f"   (V1: 66.92%)")
    print(f"  soil-median guess, rows needing water : {nz_within_2 * 100:5.2f}% within 2 mm")

    # And the target must actually respond to the things that should drive it.
    print()
    print("--- required correlations: a transferable target needs all three ---")
    et0_corr = frame["et0_mm_day"].corr(target)
    depletion_corr = frame["depletion_mm"].corr(target)
    stage_means = frame.groupby("growth_stage")["nir_horizon_mm"].mean()
    stage_spread = float(stage_means.max() - stage_means.min())
    print(f"  corr(nir, ET0)          = {et0_corr:+.4f}   must be clearly positive")
    print(f"  corr(nir, depletion)    = {depletion_corr:+.4f}   must be positive")
    print(f"  stage mean spread       = {stage_spread:.2f} mm across four stages")
    if et0_corr < 0.2 or depletion_corr < 0.2 or stage_spread < 3.0:
        print("  [FAIL] the target is not responsive to climate, soil state or season")
    else:
        print("  [ok] responsive to climate, soil state and season")

    print()
    print("--- does the target vary with weather, crop and stage? ---")
    for column in ("et0_mm_day", "vpd_kpa", "rainfall_mm", "air_temperature_c"):
        print(f"  corr(nir, {column:<20}) = {frame[column].corr(target):+.4f}")
    print(f"  corr(nir, depletion_mm)  = {frame['depletion_mm'].corr(target):+.4f}")
    print(f"  corr(nir, kc)            = {frame['kc'].corr(target):+.4f}")

    print()
    print("--- per-crop target spread (each crop must be individually learnable) ---")
    print(f"  {'crop':<14}{'rows':>8}{'mean mm':>10}{'std':>8}{'distinct':>10}{'>0 mm':>9}")
    for crop_name, group in frame.groupby("crop"):
        t = group["nir_horizon_mm"]
        print(f"  {crop_name:<14}{len(group):>8,}{t.mean():>10.2f}{t.std():>8.2f}"
              f"{t.nunique():>10,}{(t > 0.5).mean() * 100:>8.1f}%")

    print()
    print("--- per-stage target spread ---")
    print(f"  {'stage':<16}{'rows':>8}{'mean mm':>10}{'std':>8}")
    for stage_name, group in frame.groupby("growth_stage"):
        t = group["nir_horizon_mm"]
        print(f"  {stage_name:<16}{len(group):>8,}{t.mean():>10.2f}{t.std():>8.2f}")


def main() -> int:
    print("loading NASA POWER weather (reused cache, 12 stations) ...")
    weather = load_weather(2015, 2024)
    profiles = {s: station_climate_profile(weather, s) for s in weather["station"].unique()}
    print(f"  {len(weather):,} daily records, {len(profiles)} stations")

    specs = build_field_specs(weather)
    print(f"field series to simulate: {len(specs)}")
    print()
    print("simulating (this runs a full water balance per series) ...")

    started = time.time()
    frame = build_corpus(weather, specs, progress=True)
    elapsed = time.time() - started

    if frame.empty:
        print("corpus is empty; nothing to audit")
        return 1

    out = ROOT / "data" / "crop_corpus.csv"
    frame.to_csv(out, index=False)
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {elapsed:.1f}s")

    audit(frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
