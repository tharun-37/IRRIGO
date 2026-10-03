"""
Feature engineering and the canonical sensor contract for the
Intelligent Irrigation Controller.

Every model in this project consumes the same eight environmental features
produced by the field node. The names below match the JSON payload emitted by
the ESP32 firmware (see `firmware/esp32/src/telemetry.cpp`) exactly, so a
reading can travel from ADC pin to dashboard without translation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Canonical feature contract
# --------------------------------------------------------------------------

#: Features the 7-in-1 NPK probe and the weather station can measure directly.
SENSOR_FEATURES: list[str] = [
    "temperature",     # deg C, soil + air probe average
    "humidity",        # % RH, capacitive ambient sensor
    "soil_moisture",   # % volumetric water content
    "soil_ph",         # pH units
    "ec",              # mS/cm, electrical conductivity
    "nitrogen",        # mg/kg
    "phosphorus",      # mg/kg
    "potassium",       # mg/kg
    "light_intensity", # lux
    "rainfall",        # mm over the last hour
    "wind_speed",      # m/s
]

#: Features accepted by the disease classifier (adds the derived stress index).
DISEASE_FEATURES: list[str] = SENSOR_FEATURES + ["crop_stress_index"]

#: Features accepted by the water-requirement regressor.
WATER_FEATURES: list[str] = SENSOR_FEATURES

#: Features accepted by the NPK nutrient advisor regressor.
NUTRIENT_FEATURES: list[str] = [
    "soil_moisture",
    "soil_ph",
    "ec",
    "nitrogen",
    "phosphorus",
    "potassium",
    "temperature",
    "crop_stress_index",
]

#: Field that must be supplied to convert a per-square-metre depth into volume.
AREA_FEATURE: str = "field_area_m2"

#: Disease classes, ordered. The index is what the label encoder emits.
DISEASE_CLASSES: list[str] = [
    "Healthy",
    "Early_Blight",
    "Root_Rot",
    "Powdery_Mildew",
    "Rust",
    "Bacterial_Leaf_Spot",
]

#: Agronomic bands used for status colour-coding in the dashboard.
#: Each entry is (deficient_low, optimal_low, optimal_high, excessive_high).
NUTRIENT_BANDS: dict[str, tuple[float, float, float, float]] = {
    "nitrogen": (20.0, 40.0, 90.0, 120.0),
    "phosphorus": (15.0, 25.0, 60.0, 90.0),
    "potassium": (20.0, 35.0, 80.0, 110.0),
}

#: pH band for general-purpose (non-legume) row crops.
PH_BAND: tuple[float, float] = (6.0, 7.5)

#: Volumetric water content band, as a percentage.
MOISTURE_BAND: tuple[float, float] = (38.0, 62.0)

#: Physically plausible ranges. Readings outside these bounds are treated as
#: sensor faults rather than real values, and are clamped before inference.
SENSOR_LIMITS: dict[str, tuple[float, float]] = {
    "temperature": (-10.0, 80.0),
    "humidity": (0.0, 100.0),
    "soil_moisture": (0.0, 100.0),
    "soil_ph": (2.5, 9.5),
    "ec": (0.0, 20.0),
    "nitrogen": (0.0, 300.0),
    "phosphorus": (0.0, 250.0),
    "potassium": (0.0, 300.0),
    "light_intensity": (0.0, 200000.0),
    "rainfall": (0.0, 300.0),
    "wind_speed": (0.0, 60.0),
    "field_area_m2": (0.1, 1_000_000.0),
}


# --------------------------------------------------------------------------
# Derived quantities
# --------------------------------------------------------------------------


def reference_evapotranspiration(
    temperature: float,
    humidity: float,
    light_intensity: float,
    wind_speed: float = 1.0,
) -> float:
    """
    Hargreaves-style reference evapotranspiration in mm/day.

    This is the standard radiation-driven form used when the field node has no
    rain gauge and no anemometer. It is deliberately transparent so that the
    dashboard can show the operator exactly why a valve opened.

    Ra  : extraterrestrial radiation, approximated by a daylight-hours factor
    T   : mean daily air temperature, deg C
    D   : inverse relative humidity deficit, dimensionless
    W   : wind speed at 2 m, m/s
    """
    t_mean = max(0.0, float(temperature))
    # Daylight hours from solar geometry are approximated by temperature for the
    # prototype; a 13.5 h mean day is a reasonable mid-latitude annual value.
    daylight_hours = _approx_daylight_hours(t_mean)
    ra = 0.75 * daylight_hours * 0.77  # MJ m-2 day-1, after Duff & Beckman

    # 0.408 converts MJ m-2 day-1 to equivalent mm of evaporation.
    numerator = 0.408 * (ra - 0.34) * (t_mean + 18.0)
    denominator = t_mean + 23.4
    et_radiation = max(0.0, numerator / denominator)

    humidity_deficit = 100.0 - min(100.0, max(0.0, float(humidity)))
    et_aerodynamic = 0.0023 * (t_mean + 17.8) * (
        40.75 * (humidity_deficit / max(1.0, 360.0 * max(0.1, float(wind_speed)))) ** 0.5
    )

    return round(et_radiation + et_aerodynamic, 3)


def _approx_daylight_hours(temperature: float) -> float:
    """Map temperature to a plausible daylight duration for the radiation term."""
    if temperature <= 5.0:
        return 8.5
    if temperature >= 35.0:
        return 13.5
    # Linear ramp across the agriculturally interesting range.
    return 8.5 + (temperature - 5.0) * (5.0 / 30.0)


def crop_coefficient(light_intensity: float, temperature: float) -> float:
    """
    FAO-56 single crop coefficient Kc, estimated from light and temperature.

    Kc rises through vegetative development, plateaus at maximum canopy, and
    falls under heat stress. Without a growth-stage input from the operator this
    is a defensible proxy that keeps the water balance conservative.
    """
    lux = max(0.0, float(light_intensity))
    if lux < 8_000.0:
        kc_light = 0.25
    elif lux < 25_000.0:
        kc_light = 0.25 + 0.45 * (lux - 8_000.0) / 17_000.0
    else:
        kc_light = 0.70

    t = float(temperature)
    if t > 32.0:
        kc_heat = max(0.45, 1.05 - 0.03 * (t - 32.0))
    else:
        kc_heat = 1.0

    return round(min(1.25, kc_light * kc_heat), 3)


def crop_stress_index(reading: dict[str, float]) -> float:
    """
    Single scalar in [0, 1] summarising how far a zone is from ideal conditions.

    This is the engineered feature that lets the disease classifier react to a
    zone that is not yet visibly diseased but is physiologically stressed. It is
    also the interpretable number the dashboard surfaces as "crop stress".
    """
    def band_penalty(value: float, low: float, high: float, tolerance: float) -> float:
        """0 when inside the band, rising to 1 at `tolerance` distance outside."""
        if low <= value <= high:
            return 0.0
        distance = (low - value) if value < low else (value - high)
        return float(np.clip(distance / max(1e-6, tolerance), 0.0, 1.0))

    penalties = [
        band_penalty(reading.get("soil_moisture", 50.0), *MOISTURE_BAND, tolerance=22.0),
        band_penalty(reading.get("soil_ph", 6.8), *PH_BAND, tolerance=1.8),
        band_penalty(reading.get("temperature", 25.0), 15.0, 32.0, tolerance=12.0),
        band_penalty(reading.get("humidity", 60.0), 35.0, 85.0, tolerance=35.0),
        band_penalty(reading.get("ec", 1.5), 0.4, 2.4, tolerance=1.6),
    ]

    for nutrient, (low, opt_low, opt_high, high) in NUTRIENT_BANDS.items():
        value = reading.get(nutrient, opt_low)
        if value < low:
            penalties.append(float(np.clip((low - value) / max(1e-6, low), 0.0, 1.0)))
        elif value > high:
            penalties.append(float(np.clip((value - high) / max(1e-6, high), 0.0, 1.0)))
        else:
            # Reward sitting inside the optimal window rather than merely
            # avoiding deficiency, so a well-balanced zone scores near zero.
            span = max(1e-6, opt_high - opt_low)
            offset = 0.0 if opt_low <= value <= opt_high else min(
                (opt_low - value) / span, (value - opt_high) / span
            )
            penalties.append(float(np.clip(abs(offset), 0.0, 1.0)) * 0.5)

    return round(float(np.clip(np.mean(penalties), 0.0, 1.0)), 4)


def soil_water_deficit_mm(soil_moisture: float, field_capacity: float, wilting_point: float) -> float:
    """
    Current depletion of the plant-available water reservoir, in mm.

    `field_capacity` and `wilting_point` are volumetric water content in
    percent, as reported by the probe. Converting both to a depth requires the
    rooting depth and the soil's bulk density, which the backend derives from
    the crop profile configured for the device.
    """
    available = max(0.0, field_capacity - wilting_point)
    if available <= 0.0:
        return 0.0
    depletion_fraction = (field_capacity - float(soil_moisture)) / available
    return round(max(0.0, depletion_fraction), 4)


# --------------------------------------------------------------------------
# Frame-level helpers
# --------------------------------------------------------------------------


def build_features(records: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of `records` with the derived columns the models expect.

    The legacy training data used capitalised column names (`Moisture`, `PH`,
    `Nitrogen`, ...). Accept both spellings so the historical datasets and the
    live API payloads can flow through one function.
    """
    df = records.copy()
    renames = {
        "Moisture": "soil_moisture",
        "PH": "soil_ph",
        "Nitrogen": "nitrogen",
        "Phosphorus": "phosphorus",
        "Potassium": "potassium",
        "Light_Intensity": "light_intensity",
        "EC": "ec",
        "Soil_EC": "ec",
        "Electrical_Conductivity": "ec",
    }
    for old, new in renames.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    for column in ("ec", "rainfall", "wind_speed"):
        if column not in df.columns:
            df[column] = 0.0

    if "crop_stress_index" not in df.columns:
        df["crop_stress_index"] = [
            crop_stress_index(row) for row in df.to_dict(orient="records")
        ]

    return df


def select_and_clip(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Select `features`, coerce to numeric, clip to physical limits, impute."""
    out = pd.DataFrame(index=df.index)
    for feature in features:
        if feature in df.columns:
            series = pd.to_numeric(df[feature], errors="coerce")
        else:
            series = pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
        low, high = SENSOR_LIMITS.get(feature, (-np.inf, np.inf))
        out[feature] = series.clip(lower=low, upper=high)

    return out.fillna(out.median(numeric_only=True)).fillna(0.0)


def reading_to_frame(reading: dict[str, float]) -> pd.DataFrame:
    """Convert a single API reading into a one-row feature frame."""
    return build_features(pd.DataFrame([reading]))
