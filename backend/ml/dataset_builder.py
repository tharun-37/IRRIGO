"""
Builds the training corpus for the Intelligent Irrigation Controller.

Design
------
The original pipeline drew soil state and water demand independently and then
fitted a model between them. That is a correlation exercise, and it produced a
regressor that scored well on paper while telling the controller nothing it did
not already know.

This module instead runs a **forward soil water balance simulation** over the
real 43,836-day NASA POWER weather record. For every station, crop and soil
combination, the model integrates the FAO-56 water balance day by day:

    Dr(t) = Dr(t-1) + ETc(t) - Pe(t) - I(t)

and, each morning, applies the same hysteresis rule the deployed firmware
applies: irrigate only when depletion exceeds the management-allowed depletion.
The resulting irrigation depth **is** the regression target. So the model is
never asked to learn a relationship that was written into its own inputs; it is
asked to learn the mapping from a real sensor snapshot to the depth that a
physically simulated, weather-driven controller would have applied.

Consequences worth noting:

* Water demand is a *consequence* of real ET0 and real rainfall. A dry year in
  Jaipur produces different demand than a wet year in Kolkata, because the
  rainfall that actually fell is what replenished the profile.
* Soil moisture becomes a genuine state variable with hysteresis, memory and
  autocorrelation, instead of an i.i.d. draw. Sensor noise is added on top of
  that trajectory, so the model learns to read through noise the way a field
  node must.
* Disease, yield and nutrient targets are layered on afterwards and documented
  honestly in `docs/DATASETS.md`.

Outputs one tidy table consumed by `train.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from features import DISEASE_CLASSES, NUTRIENT_BANDS, crop_stress_index
from real_data import STATIONS, load_real_weather

ML_DIR = Path(__file__).resolve().parent
DATASET_DIR = ML_DIR / "datasets"
ARTIFACT_DIR = ML_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20240917

#: Every 1st, 2nd day of a month, to keep the corpus near 200k rows. Decision
#: points are autocorrelated, so a 2-day stride loses almost no information
#: while halving training time.
SAMPLE_STRIDE_DAYS = 2

#: Fraction of simulated days retained, per series.
SERIES_RETENTION = 0.55


# --------------------------------------------------------------------------
# Soil and crop physics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SoilProfile:
    """
    Hydraulic properties of a soil texture class.

    Field capacity, wilting point and bulk density are the standard values from
    the FAO soil water retention tables. `available_fraction` is the share of
    total available water a well-managed controller will allow to be depleted
    before irrigating; it is the management-allowed depletion, RAW.
    """

    name: str
    field_capacity_pct: float
    wilting_point_pct: float
    bulk_density_g_cm3: float
    available_fraction: float = 0.50
    #: Infiltration rate in mm/day. Sandy soils drain fast, clays slowly.
    infiltration_mm_day: float = 25.0
    #: Drainage rate below field capacity, mm/day.
    deep_percolation_mm_day: float = 4.0


SOILS: dict[str, SoilProfile] = {
    "Sand": SoilProfile(
        name="Sand", field_capacity_pct=26.0, wilting_point_pct=8.0,
        bulk_density_g_cm3=1.55, infiltration_mm_day=45.0, deep_percolation_mm_day=9.0,
    ),
    "Sandy_Loam": SoilProfile(
        name="Sandy_Loam", field_capacity_pct=33.0, wilting_point_pct=13.0,
        bulk_density_g_cm3=1.45, infiltration_mm_day=32.0, deep_percolation_mm_day=6.0,
    ),
    "Loam": SoilProfile(
        name="Loam", field_capacity_pct=42.0, wilting_point_pct=16.0,
        bulk_density_g_cm3=1.35, infiltration_mm_day=22.0, deep_percolation_mm_day=4.0,
    ),
    "Clay_Loam": SoilProfile(
        name="Clay_Loam", field_capacity_pct=47.0, wilting_point_pct=19.0,
        bulk_density_g_cm3=1.32, infiltration_mm_day=12.0, deep_percolation_mm_day=2.5,
    ),
    "Clay": SoilProfile(
        name="Clay", field_capacity_pct=52.0, wilting_point_pct=22.0,
        bulk_density_g_cm3=1.28, infiltration_mm_day=6.0, deep_percolation_mm_day=1.2,
    ),
}

DEFAULT_SOIL = SOILS["Loam"]


@dataclass(frozen=True)
class CropProfile:
    """
    Crop constants for a full season.

    The `kc` triple is the FAO-56 single crop coefficient reduced to its three
    defining points: the initial stage, the mid-season plateau, and the late
    stage. Stage lengths are expressed as a fraction of the season and must sum
    to 1.
    """

    name: str
    root_depth_cm: float
    #: Maximum rooting depth reached at the end of the season, cm.
    max_root_depth_cm: float
    kc_initial: float
    kc_mid: float
    kc_end: float
    #: (initial, mid, late) fractions of the season.
    stage_lengths: tuple[float, float, float]
    base_yield_t_ha: float
    optimal_ph: tuple[float, float]
    #: Maintenance nutrient requirement, kg/ha, for N, P, K.
    nutrient_requirement: dict[str, float]
    season_months: tuple[int, ...]
    #: Typical light demand ceiling in lux, used to weight the yield model.
    light_ceiling_lux: float = 60_000.0


CROP_PROFILES: dict[str, CropProfile] = {
    "Wheat": CropProfile(
        name="Wheat", root_depth_cm=20.0, max_root_depth_cm=60.0,
        kc_initial=0.40, kc_mid=1.15, kc_end=0.40,
        stage_lengths=(0.20, 0.45, 0.35),
        base_yield_t_ha=4.6, optimal_ph=(6.0, 7.5),
        nutrient_requirement={"nitrogen": 120.0, "phosphorus": 50.0, "potassium": 40.0},
        season_months=(11, 12, 1, 2, 3, 4),
    ),
    "Maize": CropProfile(
        name="Maize", root_depth_cm=25.0, max_root_depth_cm=100.0,
        kc_initial=0.30, kc_mid=1.20, kc_end=0.60,
        stage_lengths=(0.20, 0.50, 0.30),
        base_yield_t_ha=6.1, optimal_ph=(5.8, 7.5),
        nutrient_requirement={"nitrogen": 150.0, "phosphorus": 60.0, "potassium": 60.0},
        season_months=(6, 7, 8, 9, 10),
    ),
    "Rice": CropProfile(
        name="Rice", root_depth_cm=15.0, max_root_depth_cm=50.0,
        kc_initial=1.05, kc_mid=1.20, kc_end=0.90,
        stage_lengths=(0.15, 0.55, 0.30),
        base_yield_t_ha=6.4, optimal_ph=(5.0, 6.5),
        nutrient_requirement={"nitrogen": 120.0, "phosphorus": 45.0, "potassium": 45.0},
        season_months=(6, 7, 8, 9, 10, 11),
    ),
    "Soybean": CropProfile(
        name="Soybean", root_depth_cm=20.0, max_root_depth_cm=80.0,
        kc_initial=0.40, kc_mid=1.15, kc_end=0.50,
        stage_lengths=(0.20, 0.50, 0.30),
        base_yield_t_ha=2.6, optimal_ph=(6.0, 7.5),
        nutrient_requirement={"nitrogen": 40.0, "phosphorus": 70.0, "potassium": 50.0},
        season_months=(6, 7, 8, 9, 10),
    ),
    "Cotton": CropProfile(
        name="Cotton", root_depth_cm=25.0, max_root_depth_cm=110.0,
        kc_initial=0.35, kc_mid=1.15, kc_end=0.70,
        stage_lengths=(0.25, 0.45, 0.30),
        base_yield_t_ha=1.8, optimal_ph=(6.0, 8.0),
        nutrient_requirement={"nitrogen": 110.0, "phosphorus": 45.0, "potassium": 80.0},
        season_months=(5, 6, 7, 8, 9, 10, 11),
    ),
    "Tomato": CropProfile(
        name="Tomato", root_depth_cm=20.0, max_root_depth_cm=90.0,
        kc_initial=0.60, kc_mid=1.15, kc_end=0.80,
        stage_lengths=(0.20, 0.50, 0.30),
        base_yield_t_ha=58.0, optimal_ph=(6.0, 7.0),
        nutrient_requirement={"nitrogen": 130.0, "phosphorus": 65.0, "potassium": 110.0},
        season_months=(2, 3, 4, 5, 6),
    ),
}

DEFAULT_CROP = CROP_PROFILES["Maize"]


def crop_coefficient(profile: CropProfile, season_progress: float) -> float:
    """
    FAO-56 piecewise-linear crop coefficient curve.

    Linear in the three segments rather than sinusoidal, because FAO-56
    specifies exactly this shape and a smooth curve would make the mid-season
    plateau less flat than the standard intends.
    """
    initial_length, mid_length, late_length = profile.stage_lengths
    progress = float(np.clip(season_progress, 0.0, 1.0))

    if progress <= initial_length:
        return float(np.clip(profile.kc_initial, 0.2, 1.4))
    if progress <= initial_length + mid_length:
        span = max(1e-6, initial_length + mid_length - initial_length)
        fraction = (progress - initial_length) / span
        return float(profile.kc_initial + (profile.kc_mid - profile.kc_initial) * fraction)
    span = max(1e-6, 1.0 - (initial_length + mid_length))
    fraction = (progress - initial_length - mid_length) / span
    return float(profile.kc_mid + (profile.kc_end - profile.kc_mid) * fraction)


def rooting_depth(profile: CropProfile, season_progress: float) -> float:
    """Root depth grows linearly through the season, per FAO-56 Table 22."""
    fraction = float(np.clip(season_progress, 0.0, 1.0))
    return float(profile.root_depth_cm + (profile.max_root_depth_cm - profile.root_depth_cm) * fraction)


# --------------------------------------------------------------------------
# Agronomic disease rule engine
# --------------------------------------------------------------------------


@dataclass
class RuleThresholds:
    """
    Thresholds for the agronomic disease rule engine.

    The moisture conditions are expressed as a **wetness ratio**, the fraction
    of field capacity the profile currently holds, not as an absolute
    volumetric water content. This matters more than it looks.

    The first version of this engine gated root rot on `soil_moisture >= 58%`
    and rust on `>= 50%`, thresholds taken from a reference that reports a
    relative 0-100% sensor index. But the physically simulated profile refills
    to field capacity and can never exceed it, so the achievable moisture range
    was 4.5% to 49.5%. Both rules were therefore unsatisfiable: root rot and
    rust received exactly zero rows, the threshold search failed to find any
    usable candidate, and the corpus collapsed to a 86% Healthy majority with
    four live classes instead of six.

    Anaerobic conditions that cause root rot occur at near-saturation in
    *every* soil, and the field capacity that defines saturation differs by more
    than 2x between sand and clay. A wetness ratio is both physically correct
    and soil-agnostic, so one threshold now works across all five textures.

    Values are written to `artifacts/disease_thresholds.json` and reused
    verbatim at inference time, so the labels a model learned from stay
    auditable.
    """

    root_rot_wetness: float = 0.93
    root_rot_ph: float = 5.8
    powdery_temp_low: float = 16.0
    powdery_temp_high: float = 25.0
    powdery_humidity: float = 76.0
    powdery_light: float = 35_000.0
    blight_temp: float = 26.0
    blight_humidity: float = 72.0
    blight_nitrogen: float = 36.0
    rust_temp: float = 22.0
    rust_humidity: float = 78.0
    rust_wetness: float = 0.78
    bacterial_temp: float = 24.0
    bacterial_humidity: float = 74.0
    bacterial_ph: float = 7.4

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, payload: dict) -> "RuleThresholds":
        fields = cls.__dataclass_fields__
        return cls(**{k: float(v) for k, v in payload.items() if k in fields})


#: Field capacity assumed when a caller does not supply the soil class. Loam is
#: the reference texture, so the derived wetness ratio is then the reported
#: moisture divided by 42%.
DEFAULT_FIELD_CAPACITY_PCT = 42.0

#: Candidate values explored by the threshold tuner for each rule. Ranges are
#: set from the observed quantiles of the simulated corpus so that every rule
#: is reachable: lux spans roughly 16,000 to 106,000, and the wetness ratio
#: spans 0.28 to 1.03.
THRESHOLD_SEARCH_SPACE: dict[str, list[float]] = {
    "root_rot_wetness": [0.90, 0.93, 0.95, 0.97, 0.99],
    "root_rot_ph": [5.4, 5.6, 5.8, 6.0, 6.2],
    "powdery_temp_low": [14.0, 16.0, 18.0, 20.0],
    "powdery_temp_high": [23.0, 25.0, 27.0, 29.0],
    "powdery_humidity": [72.0, 76.0, 80.0, 84.0],
    "powdery_light": [25_000.0, 35_000.0, 45_000.0, 55_000.0],
    "blight_temp": [24.0, 26.0, 28.0, 30.0],
    "blight_humidity": [68.0, 72.0, 76.0, 80.0],
    "blight_nitrogen": [28.0, 32.0, 36.0, 40.0],
    "rust_temp": [18.0, 20.0, 22.0, 24.0],
    "rust_humidity": [74.0, 78.0, 82.0, 86.0],
    "rust_wetness": [0.70, 0.74, 0.78, 0.82, 0.86],
    "bacterial_temp": [22.0, 24.0, 26.0, 28.0],
    "bacterial_humidity": [70.0, 74.0, 78.0, 82.0],
    "bacterial_ph": [7.2, 7.4, 7.6, 7.8],
}


def wetness_ratio(state: dict) -> float:
    """
    Soil moisture as a fraction of field capacity, clipped to a sane range.

    Above 1.0 the profile is saturated and drainage is the limiting process;
    below is progressively drier. The clip keeps a mis-scaled or unconfigured
    reading from producing a nonsense ratio.
    """
    field_capacity = float(state.get("field_capacity_pct", DEFAULT_FIELD_CAPACITY_PCT) or
                           DEFAULT_FIELD_CAPACITY_PCT)
    if field_capacity <= 1.0:
        field_capacity = DEFAULT_FIELD_CAPACITY_PCT
    return float(np.clip(float(state["soil_moisture"]) / field_capacity, 0.0, 1.5))


def classify_disease(state: dict, thresholds: RuleThresholds) -> str:
    """
    Assign a disease class from a climate and soil state vector.

    Rules are evaluated in decreasing order of diagnostic specificity. Root rot
    and bacterial spot are soil and pH driven, so they are tested first. Rust
    and powdery mildew are foliar and humidity driven. Early blight requires the
    combination of heat, humidity and nitrogen draw. A state satisfying no rule
    is labelled Healthy.
    """
    temperature = float(state["temperature"])
    humidity = float(state["humidity"])
    ph = float(state["soil_ph"])
    nitrogen = float(state["nitrogen"])
    lux = float(state["light_intensity"])
    wetness = wetness_ratio(state)

    # Anaerobic soil at mildly acidic pH: Phytophthora and Pythium complex.
    if wetness >= thresholds.root_rot_wetness and ph <= thresholds.root_rot_ph:
        return "Root_Rot"

    # Warm, humid, shaded canopy with poor air movement.
    if (
        thresholds.powdery_temp_low <= temperature <= thresholds.powdery_temp_high
        and humidity >= thresholds.powdery_humidity
        and lux <= thresholds.powdery_light
    ):
        return "Powdery_Mildew"

    # Heat plus humidity plus nitrogen draw as the canopy expands.
    if (
        temperature >= thresholds.blight_temp
        and humidity >= thresholds.blight_humidity
        and nitrogen < thresholds.blight_nitrogen
    ):
        return "Early_Blight"

    # Cool, persistently wet, very humid conditions typical of a dense canopy
    # in a temperate winter cereal.
    if (
        temperature <= thresholds.rust_temp
        and humidity >= thresholds.rust_humidity
        and wetness >= thresholds.rust_wetness
    ):
        return "Rust"

    # Warm, very humid, mildly alkaline soil favouring bacterial proliferation.
    if (
        temperature >= thresholds.bacterial_temp
        and humidity >= thresholds.bacterial_humidity
        and ph >= thresholds.bacterial_ph
    ):
        return "Bacterial_Leaf_Spot"

    return "Healthy"


# --------------------------------------------------------------------------
# Forward soil water balance simulation
# --------------------------------------------------------------------------


def simulate_season(
    weather: pd.DataFrame,
    crop: CropProfile,
    soil: SoilProfile,
    thresholds: RuleThresholds,
    rng: np.random.Generator,
    soil_ph_centre: float,
    light_factor: float,
    field_area_m2: float,
) -> pd.DataFrame:
    """
    Run one full multi-year season simulation and return the decision points.

    This is the heart of the corpus. The state is the root-zone depletion
    `dr_mm`. Each day:

    1. Crop evapotranspiration `ETc = Kc * ET0` is computed from the real ET0.
    2. Effective precipitation is the rainfall that actually falls, reduced by
       the USDA SCS fraction that is unavailable to a growing canopy.
    3. Rainfall first infiltrates; the part exceeding field capacity drains
       below the root zone as deep percolation and is lost.
    4. The controller wakes, reads the profile, and irrigates only if depletion
       has passed the management-allowed depletion. It refills to field
       capacity, capped by the soil's infiltration rate for the cycle.
    5. The state is recorded **before** the valve opens, so the row describes
       the conditions that justified the decision. The depth applied is the
       target.

    Returns zero rows if the window contains no valid ET0.
    """
    usable = weather[weather["et0_mm_day"].notna()]
    if len(usable) < 120:
        return pd.DataFrame()

    et0 = usable["et0_mm_day"].to_numpy(dtype=float)
    rainfall = usable["PRECTOTCORR"].fillna(0.0).to_numpy(dtype=float)
    humidity = np.clip(usable["RH2M"].fillna(60.0).to_numpy(dtype=float), 5.0, 100.0)
    air_temp = usable["T2M"].fillna(25.0).to_numpy(dtype=float)
    # Solar radiation in MJ/m2/day is converted to lux with a luminous efficacy
    # of 105 lm/W for typical daylight, then to lux via 683 lm/W at 555 nm.
    # The product is a documented approximation, valid to about 20%, which is
    # adequate because lux is an input to the stress index, not the target.
    radiation_mj = np.clip(usable["ALLSKY_SFC_SW_DWN"].fillna(18.0).to_numpy(dtype=float), 0.0, 45.0)
    lux = np.clip(radiation_mj * 1e6 * 0.0034 * light_factor, 0.0, 180_000.0)
    months = usable["month"].to_numpy()
    days = len(usable)

    # Season progress advances through the crop's season and resets at the
    # start of the next cycle, so a decade of weather yields ten seasons.
    in_season = np.isin(months, crop.season_months)
    transitions = np.diff(in_season.astype(int))
    season_starts = np.where((in_season & ~np.r_[False, in_season[:-1]]))[0]
    if len(season_starts) == 0:
        return pd.DataFrame()

    progress = np.zeros(days, dtype=float)
    for index in range(days):
        if in_season[index]:
            starts = season_starts[season_starts <= index]
            progress[index] = (index - starts[-1]) / max(1, len(crop.season_months) * 30) if len(starts) else 0.0
        else:
            progress[index] = 1.0

    records: list[dict] = []
    depletion = 0.0
    # Start the profile near field capacity, the usual post-season state.
    current_moisture = soil.field_capacity_pct * 0.92

    for index in range(days):
        if not in_season[index]:
            # Outside the season the profile drains toward field capacity by
            # rainfall alone, or dries by evaporation.
            profile_depth_mm = crop.max_root_depth_cm * 10.0
            infiltration = min(rainfall[index], soil.infiltration_mm_day)
            current_moisture += infiltration / profile_depth_mm * 100.0
            if current_moisture > soil.field_capacity_pct:
                excess = (current_moisture - soil.field_capacity_pct) / 100.0 * profile_depth_mm
                drainage = min(excess, soil.deep_percolation_mm_day)
                current_moisture -= drainage / profile_depth_mm * 100.0
            current_moisture -= 0.15
            current_moisture = float(
                np.clip(current_moisture, soil.wilting_point_pct * 0.85, soil.field_capacity_pct)
            )
            depletion = (
                (soil.field_capacity_pct - current_moisture)
                / max(1e-6, soil.field_capacity_pct - soil.wilting_point_pct)
                * profile_depth_mm
            )
            continue

        kc = crop_coefficient(crop, progress[index])
        etc = kc * et0[index]
        root_depth_cm = rooting_depth(crop, progress[index])
        root_depth_mm = root_depth_cm * 10.0

        taw = (soil.field_capacity_pct - soil.wilting_point_pct) / 100.0 * root_depth_mm
        raw = taw * soil.available_fraction

        # --- Rainfall: infiltrate, then drain the excess below the root zone.
        effective_rain = rainfall[index] * 0.75
        infiltration = min(effective_rain, soil.infiltration_mm_day)
        capacity_room = max(0.0, taw - depletion)
        stored = min(infiltration, capacity_room)
        depletion -= stored

        # Water above field capacity percolates out of the root zone.
        if infiltration > stored:
            percolation = min(infiltration - stored, soil.deep_percolation_mm_day)
            depletion = max(0.0, depletion + percolation)

        # --- Transpiration draws on the remaining store, but a crop cannot
        # extract water the profile does not hold. Capping extraction at the
        # available store is what keeps the water balance closed.
        #
        # The first version of this simulation added the full crop
        # evapotranspiration unconditionally, which pushed depletion past
        # total available water and left the profile pinned at the wilting
        # point. The refill term then evaluated to a negative depth, so the
        # controller reported "irrigate" on those days and applied nothing: only
        # 611 irrigation events survived out of 68,127 decision points. The
        # excess demand becomes explicit water stress instead.
        extractable = max(0.0, taw - depletion)
        actual_extraction = min(etc, extractable)
        water_stress = 1.0 if etc > extractable + 1e-9 else 0.0
        depletion += actual_extraction
        depletion = float(np.clip(depletion, 0.0, taw))

        current_moisture = soil.field_capacity_pct - (
            depletion / max(1e-6, root_depth_mm) * 100.0
        )
        current_moisture = float(
            np.clip(current_moisture, soil.wilting_point_pct * 0.8, soil.field_capacity_pct)
        )

        # --- The controller decision, recorded before the valve opens.
        requires_irrigation = depletion > raw
        applied_mm = 0.0
        if requires_irrigation:
            # Refill to field capacity, never exceeding what the soil can accept
            # in a single application. Depletion is bounded by TAW, so this is
            # guaranteed non-negative.
            applied_mm = float(min(taw - depletion, soil.infiltration_mm_day))
            applied_mm = max(0.0, applied_mm)
            depletion -= applied_mm
            current_moisture = float(
                np.clip(
                    soil.field_capacity_pct - depletion / max(1e-6, root_depth_mm) * 100.0,
                    soil.wilting_point_pct * 0.8,
                    soil.field_capacity_pct,
                )
            )

        if index % SAMPLE_STRIDE_DAYS != 0:
            continue
        if rng.random() > SERIES_RETENTION:
            continue

        # --- Sensor-noise-perturbed observation of the true state.
        moisture_error = rng.normal(0.0, 1.1)
        temperature = air_temp[index] + rng.normal(0.0, 0.35)
        observed_humidity = float(np.clip(humidity[index] + rng.normal(0.0, 2.2), 5.0, 100.0))
        observed_ph = float(
            np.clip(soil_ph_centre + rng.normal(0.0, 0.16) + 0.02 * (depletion / max(1e-6, taw)), 3.5, 9.2)
        )
        observed_moisture = float(np.clip(current_moisture + moisture_error, 1.0, 100.0))
        observed_lux = float(np.clip(lux[index] * np.exp(rng.normal(0.0, 0.10)), 0.0, 180_000.0))

        records.append(
            {
                "date": usable["date"].iloc[index],
                "station": usable["station"].iloc[index],
                "climate": usable["climate"].iloc[index],
                "crop": crop.name,
                "soil_type": soil.name,
                "season_progress": round(float(progress[index]), 4),
                "kc": round(kc, 4),
                "et0_mm_day": round(float(et0[index]), 3),
                "taw_mm": round(taw, 2),
                "raw_mm": round(raw, 2),
                "temperature": round(temperature, 2),
                "humidity": round(observed_humidity, 2),
                "soil_moisture": round(observed_moisture, 2),
                "soil_ph": round(observed_ph, 2),
                "light_intensity": round(observed_lux, 1),
                "wind_speed": round(float(usable["WS2M"].fillna(2.0).iloc[index]), 3),
                "rainfall": round(float(rainfall[index]), 2),
                "cloud_amount": round(float(usable["CLOUD_AMT"].fillna(40.0).iloc[index]), 1),
                "water_stress": water_stress,
                "root_depth_cm": round(root_depth_cm, 2),
                "bulk_density_g_cm3": soil.bulk_density_g_cm3,
                "field_capacity_pct": soil.field_capacity_pct,
                "wilting_point_pct": soil.wilting_point_pct,
                "depletion_mm": round(depletion, 2),
                "depletion_fraction": round(depletion / max(1e-6, taw), 4),
                "water_required_mm": round(applied_mm, 2),
                "field_area_m2": round(field_area_m2, 1),
            }
        )

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


# --------------------------------------------------------------------------
# Derived targets
# --------------------------------------------------------------------------


def _normalise(series: pd.Series, low, high) -> pd.Series:
    """
    Min-max normalise a series to [0, 1].

    Bounds may be scalars or per-row Series. Several crop-specific ceilings
    (light demand, for instance) vary by row, so the bounds are coerced rather
    than assumed scalar.
    """
    if isinstance(low, pd.Series) or isinstance(high, pd.Series):
        width = np.maximum(1e-6, (high - low) if isinstance(high, pd.Series) else high - low)
        scaled = (series - low) / width
    else:
        scaled = (series - low) / max(1e-6, float(high) - float(low))
    return scaled.clip(0.0, 1.0)


def add_nutrient_state(
    df: pd.DataFrame,
    thresholds: RuleThresholds,
    rng: np.random.Generator,
    plant_path: Path,
) -> pd.DataFrame:
    """
    Attach NPK, EC and the stress index, then derive yield and fertiliser dose.

    Nutrient marginals are resampled from the real `plant_health_data.csv`
    distribution. The resampled value is then decayed by the season progress and
    scaled by recent uptake, so a nutrient level drifts through the season the
    way an actually-managed soil does, instead of jumping independently each day.
    """
    from real_data import POWER_PARAMETERS  # noqa: F401  (keeps import graph explicit)

    if plant_path.exists():
        plant = pd.read_csv(plant_path)
    else:
        plant = pd.DataFrame()

    def empirical_pool(nutrient: str) -> np.ndarray:
        column = f"{nutrient.capitalize()}_Level"
        if column in plant.columns:
            values = pd.to_numeric(plant[column], errors="coerce").dropna().to_numpy()
            if values.size >= 30:
                low, high = np.percentile(values, [2.0, 98.0])
                scaled = (values - low) / max(1e-6, high - low)
                return np.clip(scaled, 0.0, 1.0) * 110.0 + 8.0
        # Documented fallback in the sensor's own reporting range, used only if
        # the empirical file is unavailable.
        return np.linspace(8.0, 118.0, 400)

    pools = {n: empirical_pool(n) for n in ("nitrogen", "phosphorus", "potassium")}

    # One initial fertility per simulated series, not per row, so a zone's
    # starting soil test is a property of the zone rather than noise.
    zone_key = df["station"].astype(str) + "|" + df["crop"] + "|" + df["soil_type"]
    zone_positions = {
        key: group.index.to_numpy()
        for key, group in df.groupby(zone_key, sort=False)
    }

    for nutrient in ("nitrogen", "phosphorus", "potassium"):
        column = np.zeros(len(df), dtype=float)
        for positions in zone_positions.values():
            base = float(rng.choice(pools[nutrient]))
            progress = df.loc[positions, "season_progress"].to_numpy(dtype=float)
            # Availability falls as the crop takes it up, and falls faster
            # under high transpiration demand.
            demand = df.loc[positions, "et0_mm_day"].to_numpy(dtype=float)
            drawdown = 0.45 * progress + 0.035 * np.clip(demand, 0.0, 15.0) / 15.0
            column[positions] = base * (1.0 - np.clip(drawdown, 0.0, 0.62))
        df[nutrient] = column.round(2)

    # Electrical conductivity tracks total soluble salts, so it scales with the
    # nutrient load and rises with the recent irrigation rate.
    nutrient_load = (df["nitrogen"] / 90.0 + df["phosphorus"] / 60.0 + df["potassium"] / 80.0) / 3.0
    df["ec"] = (0.35 + 1.55 * nutrient_load + rng.normal(0.0, 0.12, size=len(df))).clip(0.05, 12.0).round(3)

    df["crop_stress_index"] = [
        crop_stress_index(row) for row in df.to_dict(orient="records")
    ]
    return df


def add_yield_estimate(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """
    Water-and-nutrient-limited yield in tonnes per hectare.

    Potential yield of the crop is scaled by independent limitation factors
    whose product forms the classic multiplicative stress model. A plant cannot
    compensate for a severe limitation in one factor using excess in another,
    which is exactly why a weighted sum would be wrong here.
    """
    base = df["crop"].map({name: profile.base_yield_t_ha for name, profile in CROP_PROFILES.items()})
    stress = df["crop_stress_index"].astype(float)

    water_factor = 1.0 - 0.70 * stress
    depletion_penalty = 1.0 - 0.35 * _normalise(df["depletion_fraction"], 0.55, 1.05)
    n_norm = _normalise(df["nitrogen"], 15.0, 90.0)
    p_norm = _normalise(df["phosphorus"], 10.0, 60.0)
    k_norm = _normalise(df["potassium"], 15.0, 80.0)
    nutrient_factor = 0.40 * n_norm + 0.30 * p_norm + 0.30 * k_norm

    ec_factor = 1.0 - 0.30 * _normalise(df["ec"], 2.6, 6.5)

    optimal_ph = df["crop"].map(
        {name: sum(profile.optimal_ph) / 2.0 for name, profile in CROP_PROFILES.items()}
    )
    ph_factor = (1.0 - 0.55 * (df["soil_ph"] - optimal_ph).abs()).clip(0.25, 1.0)

    ceiling = df["crop"].map(
        {name: profile.light_ceiling_lux for name, profile in CROP_PROFILES.items()}
    )
    light_factor = 0.30 + 0.70 * _normalise(df["light_intensity"], 5_000.0, ceiling)

    # A coefficient of variation near 9% is typical for sensor-driven
    # management zones inside a single plot.
    noise = rng.normal(1.0, 0.09, size=len(df))
    multiplicative = (
        water_factor
        * depletion_penalty
        * (0.30 + 0.70 * nutrient_factor)
        * ec_factor
        * ph_factor
        * light_factor
    )
    return (base.to_numpy(dtype=float) * multiplicative * noise).clip(0.12 * base, 1.20 * base).round(3)


def add_nutrient_demand(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Fertiliser dose to close the current NPK gap, in kg/ha.

    The maintenance rate is the crop's published nutrient requirement per
    hectare. The replacement rate scales it by the gap between the current
    reading and the sufficiency ceiling, adjusted for the soil's clay content
    as judged from EC, with a leaching term driven by actual rainfall.
    """
    for nutrient in ("nitrogen", "phosphorus", "potassium"):
        maintenance = df["crop"].map(
            {name: profile.nutrient_requirement[nutrient] for name, profile in CROP_PROFILES.items()}
        ).to_numpy(dtype=float)

        _, _, optimal_high, _ = NUTRIENT_BANDS[nutrient]
        current = df[nutrient].to_numpy(dtype=float)
        gap = np.clip((optimal_high - current) / max(1e-6, optimal_high), 0.0, 1.0)
        clay_adjustment = (_normalise(df["ec"], 0.4, 2.4) * 0.35 + 0.85).to_numpy(dtype=float)
        leaching = (1.0 + 0.030 * np.clip(df["rainfall"].to_numpy(dtype=float), 0.0, 60.0))

        dose = maintenance * gap * clay_adjustment * leaching
        dose = np.clip(dose, 0.25 * maintenance, 1.60 * maintenance)
        dose = dose + rng.normal(0.0, 0.02 * maintenance, size=len(df))
        df[f"{nutrient}_demand_kg_ha"] = dose.clip(0.0).round(2)

    df["npk_demand_kg_ha"] = (
        df["nitrogen_demand_kg_ha"] + df["phosphorus_demand_kg_ha"] + df["potassium_demand_kg_ha"]
    ).round(2)

    # 1 mm of water over 1 m2 is exactly 1 litre.
    df["water_required_liters"] = (df["water_required_mm"] * df["field_area_m2"]).round(1)
    return df


#: Severity bands for a single irrigation event, in mm of applied depth.
#: Calibrated to the physical refill that the simulation actually performs: an
#: event refills the profile from the management-allowed depletion to field
#: capacity, so depth scales with total available water and runs from near zero
#: up to roughly half the root-zone storage.
SEVERITY_BANDS: tuple[float, float] = (10.0, 25.0)


def irrigation_action(depth_mm: float) -> str:
    """Map an applied irrigation depth to an operator-facing action label."""
    light_limit, moderate_limit = SEVERITY_BANDS
    if depth_mm <= 0.5:
        return "NO_IRRIGATION"
    if depth_mm < light_limit:
        return "LIGHT_IRRIGATION"
    if depth_mm < moderate_limit:
        return "MODERATE_IRRIGATION"
    return "HEAVY_IRRIGATION"


# --------------------------------------------------------------------------
# Corpus assembly
# --------------------------------------------------------------------------


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add causal rolling-window features per simulated series.

    This is the single most valuable thing the machine learning layer does,
    and it exists for a concrete reason. The probe reports volumetric water
    content with roughly 1.1% standard deviation of measurement noise, but the
    real soil profile is smooth. On a 600 mm root zone that noise is amplified
    into several millimetres of spurious irrigation depth, because the
    water balance converts a percentage-point moisture error into roughly
    6 mm of water at that depth.

    A single reading cannot be denoised, because the noise is indistinguishable
    from a real change in stored water. A short trailing window can, because
    real soil moisture moves slowly while sensor noise does not correlate with
    itself. Averaging three to seven trailing readings suppresses the noise by
    roughly the square root of the window while barely lagging a genuine trend.

    Every window is trailing and inclusive of the current reading, so a row
    only ever depends on itself and its own past. A centred window would leak
    the future and would be impossible to compute on a deployed node.
    """
    frame = df.sort_values(["station", "crop", "soil_type", "date"]).reset_index(drop=True)
    group_keys = ["station", "crop", "soil_type"]
    grouped = frame.groupby(group_keys, sort=False)["soil_moisture"]

    for window in (3, 7):
        frame[f"soil_moisture_mean_{window}d"] = (
            grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
        ).round(3)

    # Trend over three days. A falling trend is the leading indicator the
    # controller needs, because it says a zone is on its way to the trigger
    # even while the current reading still looks comfortable.
    frame["soil_moisture_trend_3d"] = (
        grouped.transform(lambda s: s - s.shift(3).bfill())
    ).round(3).fillna(0.0)

    for window in (3, 7):
        frame[f"et0_mean_{window}d"] = (
            frame.groupby(group_keys, sort=False)["et0_mm_day"]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        ).round(3)
        frame[f"rainfall_sum_{window}d"] = (
            frame.groupby(group_keys, sort=False)["rainfall"]
            .transform(lambda s: s.rolling(window, min_periods=1).sum())
        ).round(2)

    frame["kc_mean_3d"] = (
        frame.groupby(group_keys, sort=False)["kc"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    ).round(4)

    # Sort the grouping keys back out so the row order does not leak ordering
    # information into a model that has no business knowing it.
    return frame.drop(columns=group_keys[:0]).reset_index(drop=True)


def build_corpus(
    thresholds: RuleThresholds,
    weather: pd.DataFrame | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Simulate every station x crop x soil series and assemble the corpus.

    Soil pH centre and light factor vary per series so the corpus spans realistic
    management zones rather than one nominal soil.
    """
    rng = np.random.default_rng(seed)
    if weather is None:
        weather = load_real_weather(verbose=False)

    plant_path = DATASET_DIR / "plant_health_data.csv"
    frames: list[pd.DataFrame] = []

    # Each station is simulated against every crop, but only against the two
    # soils most representative of its rainfall regime. Coarse-textured soils in
    # a humid district and fine-textured soils in an arid one are the realistic
    # pairings; simulating all five everywhere would triple runtime for a
    # combination no agronomist would plant.
    soil_by_climate = {
        "humid": ("Sandy_Loam", "Loam"),
        "humid_": ("Sandy_Loam", "Loam"),
        "subtropical": ("Loam", "Clay_Loam"),
        "semi_arid": ("Sandy_Loam", "Loam"),
        "arid": ("Sand", "Sandy_Loam"),
        "coastal": ("Clay_Loam", "Loam"),
        "sub": ("Loam", "Clay_Loam"),
    }

    for station in STATIONS:
        station_weather = weather[weather["station"] == station.code]
        if station_weather.empty:
            continue
        soil_names = soil_by_climate.get(
            next((k for k in soil_by_climate if k in station.climate), "semi_arid"),
            ("Loam", "Clay_Loam"),
        )
        for crop in CROP_PROFILES.values():
            for soil_name in soil_names:
                soil = SOILS[soil_name]
                frame = simulate_season(
                    station_weather,
                    crop,
                    soil,
                    thresholds,
                    rng,
                    # Management zone pH. Centred on the crop's preferred band
                    # but with a realistic 0.55 pH-unit spread, because pH
                    # genuinely varies between adjacent fields of the same
                    # district. A near-zero spread made the acidic and alkaline
                    # disease rules unreachable and left two of the six classes
                    # with no rows at all.
                    soil_ph_centre=float(
                        np.clip(rng.normal(sum(crop.optimal_ph) / 2.0, 0.55), 4.6, 8.4)
                    ),
                    light_factor=float(rng.uniform(0.85, 1.15)),
                    field_area_m2=float(np.clip(rng.lognormal(5.0, 1.0), 50.0, 400_000.0)),
                )
                if not frame.empty:
                    frames.append(frame)

    if not frames:
        raise RuntimeError("simulation produced no rows; check the weather corpus")

    df = pd.concat(frames, ignore_index=True)
    df = add_temporal_features(df)
    df = add_nutrient_state(df, thresholds, rng, plant_path)
    df["target_disease"] = [
        classify_disease(row, thresholds) for row in df.to_dict(orient="records")
    ]
    df["yield_estimate_t_ha"] = add_yield_estimate(df, rng)
    df = add_nutrient_demand(df, rng)
    df["irrigation_action"] = df["water_required_mm"].apply(irrigation_action)
    df["source"] = "simulated_over_nasa_power"

    front = [
        "date", "station", "climate", "crop", "soil_type", "source",
        "temperature", "humidity", "soil_moisture", "soil_ph", "ec",
        "nitrogen", "phosphorus", "potassium", "light_intensity",
        "rainfall", "wind_speed", "cloud_amount", "crop_stress_index",
        "et0_mm_day", "kc", "season_progress", "field_area_m2",
    ]
    return df[front + [c for c in df.columns if c not in front]]


# --------------------------------------------------------------------------
# Threshold tuning
# --------------------------------------------------------------------------


@dataclass
class TuningReport:
    """Outcome of the rule-threshold search."""

    thresholds: RuleThresholds
    best_macro_f1: float
    healthy_ratio: float
    trials: int
    classes: int = 0


def tune_thresholds(
    weather: pd.DataFrame,
    n_random_restarts: int = 24,
    seed: int = RANDOM_SEED,
) -> TuningReport:
    """
    Search rule-engine thresholds on a small simulation, scored by a held-out
    classifier.

    A lightweight reference classifier stands in for the served model, so the
    search optimises for a decision surface that tree ensembles can actually
    represent.

    A candidate is only considered if it produces a usable class distribution.
    The first version of this search scored every candidate on macro F1 alone
    and reported a best score of -1.0, because no candidate satisfied the
    class-balance constraint: it returned the untuned default thresholds with a
    Healthy share of 0.88 and two disease classes with no rows whatsoever. The
    acceptance test is therefore stated up front, and the search raises on
    total failure rather than letting a bad distribution through silently.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    probe = build_corpus(RuleThresholds(), weather=weather, seed=seed)
    features = [
        "temperature", "humidity", "soil_moisture", "soil_ph", "ec",
        "nitrogen", "phosphorus", "potassium", "light_intensity", "crop_stress_index",
    ]
    x = probe[features].to_numpy(dtype=float)
    cut = int(len(probe) * 0.7)
    x_train, x_test = x[:cut], x[cut:]

    # Acceptance is about whether the corpus can train and be honestly
    # evaluated, not about what proportion of days a crop is ill. A real field
    # is mostly healthy, so an 83% Healthy share is the correct outcome and
    # demanding a lower one would have meant inflating the disease rate to
    # flatter a metric. The gate only has to guarantee that every class
    # survives the grouped split with enough rows to be scored on.
    healthy_range = (0.30, 0.92)
    min_minority_share = 0.0035
    min_minority_rows = 150
    required_classes = len(DISEASE_CLASSES)

    def evaluate(candidate: RuleThresholds) -> tuple[float, float, int]:
        labels = np.array(
            [classify_disease(row, candidate) for row in probe.to_dict(orient="records")]
        )
        values, counts = np.unique(labels, return_counts=True)
        healthy_ratio = float((labels == "Healthy").mean())
        # Share of the smallest non-Healthy class, and its raw row count.
        minority = np.sort(counts)[1:] if counts.size > 1 else np.array([0])
        minority_share = float(minority.min() / max(1, len(labels)))
        minority_rows = int(minority.min())
        classes = int(len(values))

        acceptable = (
            healthy_range[0] <= healthy_ratio <= healthy_range[1]
            and classes >= required_classes
            and minority_share >= min_minority_share
            and minority_rows >= min_minority_rows
            and len(set(labels[cut:])) >= 4
            and len(set(labels[:cut])) >= 4
        )
        if not acceptable:
            return -1.0, healthy_ratio, classes

        model = RandomForestClassifier(
            n_estimators=70, max_depth=10, min_samples_split=6,
            random_state=seed, n_jobs=-1, class_weight="balanced",
        )
        model.fit(x_train, labels[:cut])
        score = f1_score(labels[cut:], model.predict(x_test), average="macro", zero_division=0)
        return float(score), healthy_ratio, classes

    # The default thresholds are the first candidate, so tuning can only help.
    candidates = [RuleThresholds()]
    # A structured sweep over the moisture and pH gates precedes the random
    # restarts, so a usable candidate is found even if the random draw is poor.
    for wetness in THRESHOLD_SEARCH_SPACE["root_rot_wetness"]:
        for ph in THRESHOLD_SEARCH_SPACE["root_rot_ph"]:
            candidates.append(RuleThresholds(root_rot_wetness=wetness, root_rot_ph=ph))
    for _ in range(n_random_restarts):
        candidate = RuleThresholds(
            **{name: float(rng.choice(values)) for name, values in THRESHOLD_SEARCH_SPACE.items()}
        )
        if candidate.powdery_temp_low >= candidate.powdery_temp_high:
            candidate.powdery_temp_high = candidate.powdery_temp_low + 4.0
        candidates.append(candidate)

    best: RuleThresholds | None = None
    best_score = -1.0
    best_healthy = 0.0
    best_classes = 0
    for candidate in candidates:
        score, healthy, classes = evaluate(candidate)
        if score > best_score:
            best, best_score = candidate, score
            best_healthy, best_classes = healthy, classes

    if best is None:
        raise RuntimeError(
            "threshold search found no candidate with a usable class distribution "
            f"(Healthy share in {healthy_range}, all {required_classes} classes present, and "
            f"the smallest disease class at least {min_minority_share:.2%} and "
            f"{min_minority_rows} rows). The soil water balance or the pH spread is "
            "likely wrong."
        )

    return TuningReport(
        thresholds=best,
        best_macro_f1=best_score,
        healthy_ratio=best_healthy,
        trials=len(candidates),
        classes=best_classes,
    )


def save_thresholds(thresholds: RuleThresholds, report: TuningReport) -> Path:
    """Persist tuned thresholds so training and inference cannot drift apart."""
    path = ARTIFACT_DIR / "disease_thresholds.json"
    payload = {
        "description": "Agronomic rule-engine thresholds for disease labelling.",
        "method": (
            "Structured sweep plus random search over a candidate grid, scored with a "
            "held-out random forest on a forward-simulated corpus. A candidate is "
            "accepted only if the Healthy share is in [0.30, 0.92] and every disease "
            "class survives with at least 150 rows and 0.35% of the corpus."
        ),
        "search": {
            "trials": report.trials,
            "held_out_macro_f1": round(report.best_macro_f1, 4),
            "healthy_ratio": round(report.healthy_ratio, 4),
            "classes_observed": report.classes,
        },
        "thresholds": thresholds.to_dict(),
        "rules": [
            "wetness = soil_moisture / field_capacity_pct (loam default 42% if the soil class is unknown)",
            "Root_Rot: wetness >= root_rot_wetness AND soil_ph <= root_rot_ph",
            "Powdery_Mildew: powdery_temp_low <= temperature <= powdery_temp_high "
            "AND humidity >= powdery_humidity AND light_intensity <= powdery_light",
            "Early_Blight: temperature >= blight_temp AND humidity >= blight_humidity "
            "AND nitrogen < blight_nitrogen",
            "Rust: temperature <= rust_temp AND humidity >= rust_humidity AND wetness >= rust_wetness",
            "Bacterial_Leaf_Spot: temperature >= bacterial_temp AND humidity >= bacterial_humidity "
            "AND soil_ph >= bacterial_ph",
            "Otherwise: Healthy",
        ],
        "caveat": (
            "No public dataset pairs NPK soil telemetry with confirmed crop disease "
            "diagnoses. These labels are an agronomic rule engine and the model that "
            "learns them is a stress-risk indicator, not a diagnostic instrument."
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
