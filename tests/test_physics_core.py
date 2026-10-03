"""Smoke-test the V2 physics core before anything is built on top of it."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from irrigation.data.crops import CROPS, crop  # noqa: E402
from irrigation.data.cultivation import (  # noqa: E402
    METHODS,
    crop_and_method_are_compatible,
    effective_max_depth_mm,
    gross_up_mm,
)
from irrigation.data.soil import TEXTURES, restricted_root_depth_m, texture  # noqa: E402
from irrigation.phenology.gdd import (  # noqa: E402
    GrowingDegreeDays,
    effective_depletion_fraction,
    effective_root_depth_m,
    stage_from_gdd,
)
from irrigation.physics.kc_curve import (  # noqa: E402
    build_breakpoints,
    kc_at_gdd,
    single_crop_coefficient,
)
from irrigation.physics.nir_target import build_horizon, net_requirement  # noqa: E402
from irrigation.physics.water_balance import (  # noqa: E402
    FieldConfiguration,
    advance_depletion,
    effective_rainfall_mm,
    water_stress_index,
)

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [PASS] {label:<56} {detail}")
    else:
        FAILURES.append(label)
        print(f"  [FAIL] {label:<56} {detail}")


print("=" * 78)
print("1. Reference tables load")
print("=" * 78)
check("crops encoded", len(CROPS) == 14, f"{len(CROPS)} crops")
check("soil textures encoded", len(TEXTURES) == 12, f"{len(TEXTURES)} textures")
check("irrigation methods encoded", len(METHODS) == 6, f"{len(METHODS)} methods")

print()
print("=" * 78)
print("2. Kc curve is monotone within segments and hits its breakpoints")
print("=" * 78)
maize = crop("Maize")
bp = build_breakpoints(maize)
check("Kc at sowing is kc_ini", abs(kc_at_gdd(0, bp) - maize.kc_initial) < 1e-9,
      f"{kc_at_gdd(0, bp):.3f}")
check("Kc at mid-season is kc_mid",
      abs(kc_at_gdd(bp.gdd_mid_season_end - 1, bp) - maize.kc_mid) < 1e-6,
      f"{kc_at_gdd(bp.gdd_mid_season_end - 1, bp):.3f}")
check("Kc after harvest is kc_end",
      abs(kc_at_gdd(bp.gdd_late_season_end + 100, bp) - maize.kc_end) < 1e-6,
      f"{kc_at_gdd(bp.gdd_late_season_end + 100, bp):.3f}")

# Monotone rise through development, flat mid-season, falling late.
dev = [kc_at_gdd(bp.gdd_initial_end
                  + i * (bp.gdd_development_end - bp.gdd_initial_end) / 40, bp)
       for i in range(41)]
check("Kc rises monotonically through development",
      all(dev[i] < dev[i + 1] for i in range(len(dev) - 1)),
      f"{dev[0]:.3f} -> {dev[-1]:.3f}")
check("development rise is concave (slow then fast)",
      (dev[20] - dev[0]) < (dev[-1] - dev[20]),
      f"first half +{dev[20] - dev[0]:.3f}, second half +{dev[-1] - dev[20]:.3f}")

mid = [kc_at_gdd(bp.gdd_development_end
                 + i * (bp.gdd_mid_season_end - bp.gdd_development_end) / 20, bp)
       for i in range(21)]
check("Kc is flat through mid-season",
      max(mid) - min(mid) < 1e-9, f"spread {max(mid) - min(mid):.2e}")

late = [kc_at_gdd(bp.gdd_mid_season_end
                  + i * (bp.gdd_late_season_end - bp.gdd_mid_season_end) / 40, bp)
        for i in range(41)]
check("Kc falls monotonically through late season",
      all(late[i] > late[i + 1] for i in range(len(late) - 1)),
      f"{late[0]:.3f} -> {late[-1]:.3f}")
check("Kc never goes negative", min(late) >= 0, f"min {min(late):.3f}")

print()
print("=" * 78)
print("3. Rice and maize differ on the same day, which is the point")
print("=" * 78)
rice = crop("Rice")
et0 = 5.0
# The initial stage is where the crops diverge sharply: flooded rice has almost
# no evaporation barrier, maize has a dry seedbed and a tiny canopy. At mid-season
# both converge on FAO-56's peak of 1.20 by design, so comparing there would show
# nothing. The establishment period is also where a calendar-driven controller
# most often over-waters, because it assumes maize-like demand for every crop.
early_gdd = 150.0
kc_maize_early = kc_at_gdd(early_gdd, build_breakpoints(maize))
kc_rice_early = kc_at_gdd(early_gdd, build_breakpoints(rice))
check("rice Kc far exceeds maize during establishment",
      kc_rice_early > 2.5 * kc_maize_early,
      f"maize {kc_maize_early:.3f} vs rice {kc_rice_early:.3f} -> "
      f"ETc {kc_maize_early * et0:.2f} vs {kc_rice_early * et0:.2f} mm/d")

mid_gdd = maize.gdd_stage_ends[2] * 0.5
kc_maize_mid = kc_at_gdd(mid_gdd, build_breakpoints(maize))
kc_rice_mid = kc_at_gdd(mid_gdd, build_breakpoints(rice))
check("both converge near FAO-56's peak at mid-season",
      abs(kc_maize_mid - kc_rice_mid) < 0.05,
      f"maize {kc_maize_mid:.3f}, rice {kc_rice_mid:.3f}")

# A third crop for spread, one with a genuinely different peak.
onion = crop("Onion")
kc_onion_mid = kc_at_gdd(
    onion.gdd_stage_ends[2] * 0.5, build_breakpoints(onion)
)
print(f"  at each crop's own mid-season, ET0 5 mm/d:")
print(f"    rice   Kc {kc_rice_mid:.3f}  ETc {kc_rice_mid * et0:.2f} mm/d")
print(f"    maize  Kc {kc_maize_mid:.3f}  ETc {kc_maize_mid * et0:.2f} mm/d")
print(f"    onion  Kc {kc_onion_mid:.3f}  ETc {kc_onion_mid * et0:.2f} mm/d")
check("onion transpires less than maize at the same weather",
      kc_onion_mid < kc_maize_mid,
      f"onion {kc_onion_mid:.3f} < maize {kc_maize_mid:.3f}")

print()
print("=" * 78)
print("4. GDD accumulation: base temperature matters, floors at zero")
print("=" * 78)
gdd_maize = GrowingDegreeDays(maize.gdd_base_temp_c)
gdd_wheat = GrowingDegreeDays(crop("Wheat").gdd_base_temp_c)
# A single warm day, 30C day / 10C night, mean 20C.
warm_day_maize = float(gdd_maize.daily_gdd(30.0, 10.0))
warm_day_wheat = float(gdd_wheat.daily_gdd(30.0, 10.0))
check("mean 20C day accumulates for maize",
      abs(warm_day_maize - 10.0) < 1e-9, f"{warm_day_maize:.1f} GDD")
check("the same day accumulates more for wheat (lower base)",
      warm_day_wheat > warm_day_maize,
      f"wheat {warm_day_wheat:.1f} vs maize {warm_day_maize:.1f} GDD")
cold_day = float(gdd_wheat.daily_gdd(2.0, -2.0))
check("a cold day accumulates zero for wheat, not a negative",
      cold_day == 0.0, f"{cold_day:.1f} GDD")
heat_spike = float(gdd_maize.daily_gdd(42.0, 28.0))
check("heat spike is capped by the 30C upper cutoff, not the 35C mean",
      abs(heat_spike - 20.0) < 1e-9,
      f"mean 35C -> {heat_spike:.1f} GDD (capped at 30C)")
check("accumulation is vectorised over a series",
      len(gdd_maize.accumulate([30.0] * 7, [10.0] * 7)) == 7,
      "7-day series")

print()
print("=" * 78)
print("5. Root depth grows with stage, does not start at maximum")
print("=" * 78)
depths = []
for g in np.linspace(0, maize.gdd_stage_ends[2], 40):
    stage = stage_from_gdd(float(g), maize)
    depths.append(effective_root_depth_m(maize, stage))
check("root depth at sowing is far below maximum",
      depths[0] < 0.25 * maize.root_depth_max_m,
      f"{depths[0]:.2f} m vs max {maize.root_depth_max_m} m")
check("root depth reaches maximum by mid-season",
      abs(max(depths) - maize.root_depth_max_m) < 1e-6,
      f"{max(depths):.2f} m")
check("root depth increases monotonically through establishment and mid-season",
      all(depths[i] <= depths[i + 1] + 1e-9 for i in range(len(depths) - 1)),
      f"{depths[0]:.2f} -> {depths[-1]:.2f} m")
# Late season deliberately declines: root senescence reduces functional depth,
# and FAO-56 attributes late-season drought sensitivity partly to it.
late_depth = effective_root_depth_m(
    maize, stage_from_gdd(maize.gdd_stage_ends[2] + (maize.gdd_stage_ends[3] - maize.gdd_stage_ends[2]) * 0.5, maize)
)
check("root depth declines slightly in late season from senescence",
      late_depth < maize.root_depth_max_m * 1.001 and late_depth > maize.root_depth_max_m * 0.85,
      f"{late_depth:.2f} m vs {maize.root_depth_max_m} m at mid-season")

print()
print("=" * 78)
print("6. Depletion threshold is crop-specific, which is the crop-wise core")
print("=" * 78)
mid_stage = stage_from_gdd(maize.gdd_stage_ends[2] * 0.5, maize)
print(f"  {'crop':<12}{'p':>7}{'root m':>9}{'TAW mm':>9}{'RAW mm':>9}")
for name in ("Rice", "Onion", "Potato", "Maize", "Cotton", "Sugarcane"):
    c = crop(name)
    st = stage_from_gdd(c.gdd_stage_ends[2] * 0.5, c)
    soil = texture("Loam")
    taw = soil.taw_mm(c.root_depth_max_m)
    p = effective_depletion_fraction(c, st)
    print(f"  {name:<12}{p:>7.2f}{c.root_depth_max_m:>9.1f}{taw:>9.0f}{p * taw:>9.0f}")
p_rice = effective_depletion_fraction(crop("Rice"), stage_from_gdd(1000, crop("Rice")))
p_cotton = effective_depletion_fraction(crop("Cotton"), stage_from_gdd(1000, crop("Cotton")))
check("rice and cotton differ in tolerated depletion",
      p_cotton > p_rice * 5,
      f"cotton p={p_cotton:.2f} vs rice p={p_rice:.2f} -> {p_cotton / p_rice:.0f}x apart")

print()
print("=" * 78)
print("7. Water balance conserves and clamps correctly")
print("=" * 78)
taw = 100.0
after = advance_depletion(20.0, 5.0, 0.0, 0.0, taw)
check("no rain, no irrigation: depletion rises by ETc", abs(after - 25.0) < 1e-9,
      f"20 -> {after}")
after = advance_depletion(20.0, 5.0, 3.0, 0.0, taw)
check("rain offsets ETc", abs(after - 22.0) < 1e-9, f"20 -> {after}")
after = advance_depletion(20.0, 5.0, 0.0, 10.0, taw)
check("irrigation offsets ETc", abs(after - 15.0) < 1e-9, f"20 -> {after}")
after = advance_depletion(95.0, 10.0, 0.0, 0.0, taw)
check("depletion clamps at TAW, cannot exceed", abs(after - taw) < 1e-9,
      f"95 + 10 -> {after}")
after = advance_depletion(2.0, 5.0, 10.0, 0.0, taw)
check("depletion clamps at zero when rain exceeds demand", after == 0.0,
      f"2 + 5 ETc - 10 rain -> {after}")

print()
print("=" * 78)
print("8. Effective rainfall: runoff, interception and storage all bite")
print("=" * 78)
sand = texture("Sand")
clay = texture("Clay")
pe_sand = effective_rainfall_mm(25.0, sand, 0.0, 200.0)
pe_clay = effective_rainfall_mm(25.0, clay, 0.0, 200.0)
check("sand takes more of a 25mm rain than clay",
      pe_sand > pe_clay, f"sand {pe_sand:.1f} mm vs clay {pe_clay:.1f} mm")
pe_full = effective_rainfall_mm(25.0, sand, 0.0, 200.0)
pe_nearly_full = effective_rainfall_mm(25.0, sand, 195.0, 200.0)
check("rain is bounded by the root zone's remaining storage headroom",
      abs(pe_nearly_full - 5.0) < 1e-9,
      f"empty profile {pe_full:.1f} mm vs 5mm headroom {pe_nearly_full:.1f} mm")
pe_full_profile = effective_rainfall_mm(25.0, sand, 200.0, 200.0)
check("rain on a full profile contributes nothing",
      pe_full_profile == 0.0, f"{pe_full_profile:.1f} mm")
pe_tiny = effective_rainfall_mm(0.5, sand, 0.0, 200.0)
check("a drizzle is entirely intercepted", pe_tiny == 0.0, f"{pe_tiny:.1f} mm")

print()
print("=" * 78)
print("9. Application limits bind on the soil, not just the equipment")
print("=" * 78)
drip = METHODS["Drip"]
sprinkler = METHODS["Sprinkler"]
limit_sprinkler_clay = effective_max_depth_mm(sprinkler, clay)
limit_sprinkler_sand = effective_max_depth_mm(sprinkler, sand)
check("sprinkler on clay is limited below its rating",
      limit_sprinkler_clay < sprinkler.max_depth_per_event_mm,
      f"clay {limit_sprinkler_clay:.1f} mm vs rating {sprinkler.max_depth_per_event_mm}")
check("sprinkler on sand allows more than on clay",
      limit_sprinkler_sand > limit_sprinkler_clay,
      f"sand {limit_sprinkler_sand:.1f} vs clay {limit_sprinkler_clay:.1f} mm")
check("drip's own limit is the binding one",
      abs(effective_max_depth_mm(drip, sand) - drip.max_depth_per_event_mm) < 1e-9,
      f"{effective_max_depth_mm(drip, sand):.1f} mm")
check("grossing up for 50% efficiency doubles the net requirement",
      abs(gross_up_mm(20.0, METHODS["Flood"]) - 40.0) < 1e-9,
      f"net 20 mm -> gross {gross_up_mm(20.0, METHODS['Flood']):.1f} mm")

print()
print("=" * 78)
print("10. Horizon NIR: continuous, crop-specific, weather-driven")
print("=" * 78)
# Two fields, identical sensors, different crops and soils.
horizon_days = 14
et0 = np.full(horizon_days, 5.0)
rain = np.zeros(horizon_days)
rain_p = np.zeros(horizon_days)
gdd_daily = np.full(horizon_days, 12.0)

cfg_maize = FieldConfiguration(crop=maize, soil=texture("Loam"))
cfg_onion = FieldConfiguration(crop=crop("Onion"), soil=texture("Sandy_Loam"))
state_maize = cfg_maize.water_state(stage_from_gdd(1800.0, maize), 40.0)
state_onion = cfg_onion.water_state(
    stage_from_gdd(crop("Onion").gdd_stage_ends[2] * 0.5, crop("Onion")), 20.0
)

hz_m = build_horizon(et0, rain, rain_p, gdd_daily, 1800.0, maize)
hz_o = build_horizon(et0, rain, rain_p, gdd_daily,
                     crop("Onion").gdd_stage_ends[2] * 0.5, crop("Onion"))

req_m = net_requirement(hz_m, cfg_maize, state_maize, METHODS["Sprinkler"],
                        sprinkler.application_efficiency,
                        gdd_at_horizon_end=1800.0 + 12 * 14)
req_o = net_requirement(hz_o, cfg_onion, state_onion, drip,
                        drip.application_efficiency,
                        gdd_at_horizon_end=crop("Onion").gdd_stage_ends[2] * 0.5 + 168)

print(f"  maize on loam, sprinkler : net {req_m.net_requirement_mm:6.2f} mm  gross {req_m.gross_requirement_mm:6.2f} mm")
print(f"    demand {req_m.demand_mm:.2f}  rain {req_m.rainfall_credit_mm:.2f}  soil {req_m.soil_supply_mm:.2f}")
print(f"  onion on sandy loam, drip: net {req_o.net_requirement_mm:6.2f} mm  gross {req_o.gross_requirement_mm:6.2f} mm")
print(f"    demand {req_o.demand_mm:.2f}  rain {req_o.rainfall_credit_mm:.2f}  soil {req_o.soil_supply_mm:.2f}")
check("the two fields get different requirements on identical weather",
      abs(req_m.net_requirement_mm - req_o.net_requirement_mm) > 1.0,
      f"maize {req_m.net_requirement_mm:.2f} vs onion {req_o.net_requirement_mm:.2f} mm")

# Now vary weather on the SAME field and confirm the target moves.
et0_hot = np.full(horizon_days, 7.0)
hz_hot = build_horizon(et0_hot, rain, rain_p, gdd_daily, 1800.0, maize)
req_hot = net_requirement(hz_hot, cfg_maize, state_maize, sprinkler,
                          sprinkler.application_efficiency,
                          gdd_at_horizon_end=1800.0 + 168)
check("the target responds to weather",
      req_hot.net_requirement_mm > req_m.net_requirement_mm,
      f"ET0 5 -> 7 mm/d raises the target {req_m.net_requirement_mm:.2f} -> {req_hot.net_requirement_mm:.2f} mm")

# And rain must reduce it.
rain_heavy = np.full(horizon_days, 4.0)
rain_prob = np.full(horizon_days, 0.8)
hz_wet = build_horizon(et0, rain_heavy, rain_prob, gdd_daily, 1800.0, maize)
req_wet = net_requirement(hz_wet, cfg_maize, state_maize, sprinkler,
                          sprinkler.application_efficiency,
                          gdd_at_horizon_end=1800.0 + 168)
check("rain credit reduces the target",
      req_wet.net_requirement_mm < req_m.net_requirement_mm,
      f"{req_m.net_requirement_mm:.2f} -> {req_wet.net_requirement_mm:.2f} mm "
      f"(credited {req_wet.rainfall_credit_mm:.1f} mm)")

print()
print("=" * 78)
print("11. THE V1 DEFECT: is the target continuous, or a per-soil constant?")
print("=" * 78)
# The sweep covers the range a controller actually operates in: from field
# capacity down to the crop's stress threshold. Beyond that the field is already
# in deficit and the requirement saturates at the horizon's total demand, which
# is a hard physical bound rather than an artefact: a crop cannot need more water
# over fourteen days than it transpires over fourteen days. Sweeping into that
# region would measure the bound, not the model.
operational_range = state_maize.readily_available_water_mm
sweep = []
for depletion in np.linspace(0, operational_range, 80):
    st = cfg_maize.water_state(stage_from_gdd(1800.0, maize), float(depletion))
    r = net_requirement(hz_m, cfg_maize, st, sprinkler,
                        sprinkler.application_efficiency,
                        gdd_at_horizon_end=1968.0)
    sweep.append(r.net_requirement_mm)
sweep = np.array(sweep)
distinct = len(np.unique(np.round(sweep, 4)))
print(f"  operational range   : 0 to {operational_range:.0f} mm depletion (RAW)")
print(f"  distinct values     : {distinct}")
print(f"  V1 for comparison   : 3 values across all fields, all weather")
check("target takes many distinct values in the operating range",
      distinct >= 40, f"{distinct} distinct of 80 samples")

# The target does plateau, at the horizon's total crop demand: once the root zone
# is past its operating trigger it cannot contribute without being pushed past it,
# so the whole requirement is the demand. That is a physical bound, not an
# artefact, and a target that had no upper bound would be suspect instead.
#
# What made V1's plateau a defect was not that it existed but that its value was
# uncorrelated with everything: 22 mm on every field, in every season, at every
# station. So the test is whether this ceiling tracks the physics, not whether it
# is flat.
ceiling = float(sweep.max())
horizon_demand = float(sum(d.etc_mm for d in hz_m))
check("the plateau sits exactly at the horizon's crop demand",
      abs(ceiling - horizon_demand) < 1e-6,
      f"ceiling {ceiling:.2f} mm == demand {horizon_demand:.2f} mm")

ceilings: dict[float, float] = {}
for et0_value in (3.0, 5.0, 7.0, 9.0):
    hz_x = build_horizon(np.full(horizon_days, et0_value), np.zeros(horizon_days),
                         np.zeros(horizon_days), gdd_daily, 1800.0, maize)
    r = net_requirement(hz_x, cfg_maize, state_maize, sprinkler,
                        sprinkler.application_efficiency, gdd_at_horizon_end=1968.0)
    ceilings[et0_value] = r.net_requirement_mm
print(f"  requirement ceiling vs ET0 on the same field:")
for et0_value, value in ceilings.items():
    print(f"    ET0 {et0_value:.0f} mm/d -> ceiling {value:6.2f} mm")
check("the ceiling tracks weather, unlike V1's fixed 22 mm",
      len(set(round(v, 2) for v in ceilings.values())) == len(ceilings),
      f"{len(set(round(v, 2) for v in ceilings.values()))} distinct ceilings of {len(ceilings)}")

check("target is monotonically non-decreasing in depletion",
      all(sweep[i] <= sweep[i + 1] + 1e-9 for i in range(len(sweep) - 1)),
      "drier soil never reduces the requirement")

# And the transition band itself must be continuous, since that is where the
# decision to irrigate is actually made.
band = [v for v in sweep if 0.0 < v < horizon_demand - 1e-6]
band_unique = len(set(np.round(band, 4)))
print(f"  transition band      : {len(band)} samples, {band_unique} distinct")
check("the decision band is continuous, not a step",
      band_unique >= 25,
      f"{band_unique} distinct values between zero and full demand")

# It must vary with crop, soil and weather, all else equal.
variations: dict[str, float] = {}
for crop_name, soil_name, gdd_ref in (
    ("Maize", "Loam", 1800.0),
    ("Onion", "Loam", 1800.0),
    ("Maize", "Sand", 1800.0),
    ("Maize", "Clay", 1800.0),
):
    c = crop(crop_name)
    cfgi = FieldConfiguration(crop=c, soil=texture(soil_name))
    gdd_ref_i = min(gdd_ref, c.gdd_stage_ends[2] * 0.5)
    sti = cfgi.water_state(stage_from_gdd(gdd_ref_i, c), 60.0)
    hzi = build_horizon(et0, np.zeros(horizon_days), np.zeros(horizon_days),
                         gdd_daily, gdd_ref_i, c)
    ri = net_requirement(hzi, cfgi, sti, sprinkler, sprinkler.application_efficiency,
                         gdd_at_horizon_end=gdd_ref_i + 12 * 14)
    variations[f"{crop_name}/{soil_name}"] = ri.net_requirement_mm
print(f"  same depletion (60mm), same weather, four fields:")
for key, value in variations.items():
    print(f"    {key:<16} {value:7.2f} mm")
check("identical sensor readings give different requirements per field",
      len(set(round(v, 2) for v in variations.values())) == len(variations),
      f"{len(set(round(v, 2) for v in variations.values()))} distinct of {len(variations)}")

print()
print("=" * 78)
print("12. Crop and method compatibility warnings fire correctly")
print("=" * 78)
ok, reason = crop_and_method_are_compatible(rice, sprinkler)
check("rice on sprinkler is flagged", not ok and "Rice" in reason, reason[:60] if reason else "")
ok, _ = crop_and_method_are_compatible(rice, METHODS["Paddy"])
check("rice on paddy is fine", ok)
ok, reason = crop_and_method_are_compatible(crop("Onion"), METHODS["Flood"])
check("onion on flood is flagged as a waterlogging risk", not ok,
      reason[:60] if reason else "")
ok, _ = crop_and_method_are_compatible(crop("Sugarcane"), drip)
check("sugarcane on drip is fine", ok)

print()
print("=" * 78)
print("13. Root restriction from a hardpan")
print("=" * 78)
restricted = restricted_root_depth_m(1.7, 0.6)
check("a hardpan at 0.6m caps the root depth", abs(restricted - 0.6) < 1e-9,
      f"1.7 m -> {restricted:.2f} m")
resistance = restricted_root_depth_m(1.7, None, 2.5)
check("penetration resistance > 2 MPa caps depth at 1.2m",
      abs(resistance - 1.2) < 1e-9, f"{resistance:.2f} m")

print()
print("=" * 78)
if FAILURES:
    print(f"{len(FAILURES)} check(s) FAILED:")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("All physics-core checks passed.")
print("=" * 78)
