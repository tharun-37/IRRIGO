"""
Real data acquisition for the Intelligent Irrigation Controller.

Why this module exists
----------------------
The project brief asks for a system that combines soil moisture with *weather
forecasts*. Inventing weather defeats the point: irrigation decisions are only
interesting when the atmospheric demand varies the way it really does. This
module pulls genuine daily meteorology from the NASA POWER project and derives
reference evapotranspiration with the FAO-56 Penman-Monteith equation.

Source
------
NASA Prediction of Worldwide Energy Resources (POWER), agroclimatology
community, daily point API. Public, no key required, CC BY 4.0.

    https://power.larc.nasa.gov/docs/services/api/temporal/daily

Citing POWER
------------
    The NASA Prediction of Worldwide Energy Resources (POWER) Project was
    accessed via the POWER Project's API. Data are openly available under the
    NASA Open Data Commons.

Everything below the fetch layer is standard published physics:
Allen et al. (1998), FAO Irrigation and Drainage Paper 56, "Crop
evapotranspiration: guidelines for computing crop water requirements".

    https://www.fao.org/3/x0490e/x0490e00.htm

Downloaded series are cached to `datasets/cache` so that training is
reproducible and does not require network access on every run.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATASET_DIR = Path(__file__).resolve().parent / "datasets"
CACHE_DIR = DATASET_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

POWER_API = "https://power.larc.nasa.gov/api/temporal/daily/point"

#: POWER parameter codes used by this project, with their documented units.
POWER_PARAMETERS: dict[str, str] = {
    "T2M": "deg C",                    # mean air temperature at 2 m
    "T2M_MAX": "deg C",                # daily maximum air temperature
    "T2M_MIN": "deg C",                # daily minimum air temperature
    "RH2M": "percent",                 # relative humidity at 2 m
    "PRECTOTCORR": "mm/day",           # bias-corrected total precipitation
    "WS2M": "m/s",                     # wind speed at 2 m
    "ALLSKY_SFC_SW_DWN": "MJ/m2/day",  # all-sky surface shortwave downward
    "PS": "kPa",                       # surface pressure
    "CLOUD_AMT": "percent",            # cloud amount
    "T2MDEW": "deg C",                 # dew point at 2 m
}

#: Sentinel that POWER uses for unavailable observations.
POWER_FILL_VALUE = -999.0


@dataclass(frozen=True)
class Station:
    """
    A real agro-climatic reference station and the district it represents.

    Chosen to span the four broad climate regimes that a field prototype is
    likely to be deployed into, so the water-balance model is trained on
    genuinely different evaporative demand regimes rather than one climate
    resampled with noise.
    """

    code: str
    name: str
    latitude: float
    longitude: float
    climate: str
    #: Typical kc multiplier for the dominant cropping system in the district.
    crop: str


#: Indian agricultural districts, spanning semi-arid, coastal-subhumid,
#: subtropical-humid and tropical-wet regimes. All coordinates are real.
STATIONS: tuple[Station, ...] = (
    Station("HYD", "Hyderabad, Telangana", 17.38, 78.48, "semi_arid", "Maize"),
    Station("LUD", "Ludhiana, Punjab", 30.90, 75.85, "subtropical_humid", "Wheat"),
    Station("CHN", "Chennai, Tamil Nadu", 13.08, 80.27, "coastal_subhumid", "Rice"),
    Station("JAI", "Jaipur, Rajasthan", 26.91, 75.79, "arid", "Pearl_Millet"),
    Station("NAG", "Nagpur, Maharashtra", 21.15, 79.09, "semi_arid", "Soybean"),
    Station("KOL", "Kolkata, West Bengal", 22.57, 88.36, "humid_subtropical", "Rice"),
    Station("PNQ", "Pune, Maharashtra", 18.52, 73.86, "semi_arid", "Soybean"),
    Station("AMD", "Ahmedabad, Gujarat", 23.02, 72.57, "arid_subtropical", "Cotton"),
    Station("BBN", "Bhubaneswar, Odisha", 20.27, 85.82, "humid_subtropical", "Rice"),
    Station("LKO", "Lucknow, Uttar Pradesh", 26.85, 80.95, "subtropical_humid", "Wheat"),
    Station("PNJ", "Panipat, Haryana", 29.39, 76.96, "semi_arid_subtropical", "Wheat"),
    Station("IDR", "Indore, Madhya Pradesh", 22.72, 75.86, "subtropical_dry", "Soybean"),
)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def _cache_path(station: Station, start: date, end: date) -> Path:
    return CACHE_DIR / f"power_{station.code}_{start:%Y%m%d}_{end:%Y%m%d}.csv"


def fetch_power_daily(
    station: Station,
    start: date,
    end: date,
    use_cache: bool = True,
    max_retries: int = 4,
) -> pd.DataFrame:
    """
    Retrieve daily meteorology for one station.

    Results are cached on disk keyed by station and date range, so repeated
    training runs are deterministic and offline. Transient network failures are
    retried with exponential backoff rather than aborting the build.
    """
    cache_file = _cache_path(station, start, end)
    if use_cache and cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["date"])

    query = urllib.parse.urlencode(
        {
            "parameters": ",".join(POWER_PARAMETERS),
            "community": "ag",
            "longitude": f"{station.longitude}",
            "latitude": f"{station.latitude}",
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "format": "JSON",
        }
    )
    url = f"{POWER_API}?{query}"

    payload: dict | None = None
    for attempt in range(max_retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "intelligent-irrigation-controller/1.0"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"NASA POWER request failed for {station.code} after "
                    f"{max_retries} attempts: {error}"
                ) from error
            time.sleep(2.0 * (2**attempt))

    assert payload is not None
    frame = _parse_power_response(payload, station, start, end)
    frame.to_csv(cache_file, index=False)
    return frame


def _parse_power_response(payload: dict, station: Station, start: date, end: date) -> pd.DataFrame:
    """Flatten POWER's nested JSON time series into a tidy daily frame."""
    series = payload["properties"]["parameter"]
    expected_days = (end - start).days + 1
    dates = [start + timedelta(days=i) for i in range(expected_days)]

    frame = pd.DataFrame({"date": dates})
    for name in POWER_PARAMETERS:
        values = series.get(name, {})
        column = []
        for day in dates:
            # POWER keys its daily series by compact date (20240301). The
            # ISO form is accepted too, because the monthly and climatology
            # endpoints use it and a silent miss here would empty the corpus.
            raw = values.get(day.strftime("%Y%m%d"))
            if raw is None:
                raw = values.get(day.isoformat(), POWER_FILL_VALUE)
            value = float(raw)
            column.append(np.nan if value <= POWER_FILL_VALUE + 1e-6 else value)
        frame[name] = column

    # POWER occasionally drops a day at the very start of a satellite record.
    before = len(frame)
    frame = frame.dropna(subset=["T2M", "RH2M"]).reset_index(drop=True)
    if len(frame) == 0:
        raise RuntimeError(
            f"NASA POWER returned no usable daily records for station {station.code}. "
            "Check that the requested date range overlaps the satellite record."
        )
    if before - len(frame) > 0.1 * before:
        print(
            f"    note: dropped {before - len(frame)} of {before} days with missing meteorology",
            flush=True,
        )

    frame["station"] = station.code
    frame["latitude"] = station.latitude
    frame["longitude"] = station.longitude
    frame["climate"] = station.climate
    frame["source"] = "NASA_POWER_daily_ag"

    frame = compute_fao56_et0(frame)
    return frame


# --------------------------------------------------------------------------
# FAO-56 Penman-Monteith reference evapotranspiration
# --------------------------------------------------------------------------

#: Stefan-Boltzmann constant, MJ K^-4 m^-2 day^-1.
STEFAN_BOLTZMANN = 4.903e-9
#: Solar constant, MJ m^-2 min^-1.
SOLAR_CONSTANT = 0.0820
#: Albedo of a reference grass surface, dimensionless.
ALBEDO = 0.23
#: Psychrometric constant coefficient, kPa degC^-1.
PSYCHROMETRIC_COEFF = 0.000665
#: Stefan-Boltzmann exponent offset used in the saturation vapour curve.
VAPOUR_EXPONENT = 17.27
VAPOUR_OFFSET = 237.3


def _saturation_vapour_pressure(temperature_c: float) -> float:
    """Saturation vapour pressure at `temperature_c`, in kPa. FAO-56 Eq. 11."""
    return 0.6108 * math.exp((VAPOUR_EXPONENT * temperature_c) / (temperature_c + VAPOUR_OFFSET))


def _vapour_pressure_slope(temperature_c: float) -> float:
    """Slope of the saturation vapour pressure curve, in kPa degC^-1. Eq. 13."""
    es = _saturation_vapour_pressure(temperature_c)
    return (4098.0 * es) / (temperature_c + VAPOUR_OFFSET) ** 2


def _extraterrestrial_radiation(day_of_year: int, latitude_deg: float) -> float:
    """
    Extraterrestrial radiation Ra, in MJ m^-2 day^-1. FAO-56 Eq. 21.

    Ra is purely astronomical: it depends only on latitude and the day of year,
    which makes it the correct solar-top-of-atmosphere input for the daily
    energy balance when no pyranometer is fitted to the field node.
    """
    phi = math.radians(latitude_deg)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * day_of_year / 365.0)
    declination = 0.409 * math.sin(2.0 * math.pi * day_of_year / 365.0 - 1.39)

    # Sunset hour angle. At the poles, or in polar night, the argument of the
    # arccos leaves its domain; clamp rather than raise, because a controller
    # must not crash on a geometry edge case.
    cosine_argument = -math.tan(phi) * math.tan(declination)
    sunset_hour_angle = math.acos(max(-1.0, min(1.0, cosine_argument)))

    return (
        (24.0 * 60.0 / math.pi)
        * SOLAR_CONSTANT
        * dr
        * (
            sunset_hour_angle * math.sin(phi) * math.sin(declination)
            + math.cos(phi) * math.cos(declination) * math.sin(sunset_hour_angle)
        )
    )


def _estimate_elevation_m(surface_pressure_kpa: float) -> float:
    """
    Site elevation from surface pressure, using the US Standard Atmosphere.

    POWER's daily point response does not return elevation, but it does return
    surface pressure, and the barometric formula inverts cleanly. This is only
    needed to correct clear-sky radiation and wind height, both of which are
    weakly sensitive to it.
    """
    ratio = max(0.05, surface_pressure_kpa / 101.325)
    return 44330.0 * (1.0 - ratio ** (1.0 / 4.2559))


def compute_fao56_et0(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add the FAO-56 Penman-Monteith energy balance and ET0 in mm/day.

    Implemented on the daily time step, which is the granularity irrigation
    scheduling actually needs. Columns added:

    `et0_mm_day`     reference evapotranspiration, mm/day
    `rn_mj_m2_day`   net radiation at the surface, MJ m-2 day-1
    `g_vapour_kpa`   actual vapour pressure, kPa
    `elevation_m`    site elevation inferred from surface pressure

    Rows with any missing input are left with NaN ET0 rather than being
    silently filled, so a gap in the satellite record stays visible downstream.
    """
    if frame.empty:
        return frame

    work = frame.copy()
    t_mean = work["T2M"].to_numpy(dtype=float)
    t_max = work["T2M_MAX"].to_numpy(dtype=float)
    t_min = work["T2M_MIN"].to_numpy(dtype=float)
    humidity = np.clip(work["RH2M"].to_numpy(dtype=float), 5.0, 100.0)
    wind = np.clip(work["WS2M"].to_numpy(dtype=float), 0.5, 12.0)
    solar = np.clip(work["ALLSKY_SFC_SW_DWN"].to_numpy(dtype=float), 0.0, 45.0)
    pressure = np.clip(work["PS"].to_numpy(dtype=float), 60.0, 105.0)
    latitude = float(work["latitude"].iloc[0])

    # Actual vapour pressure. FAO-56 offers four routes; the dew point route
    # (Eq. 14) is the recommended one because it is the only one that does not
    # rely on the non-linear relationship between relative humidity and vapour
    # pressure. Mean relative humidity (Eq. 19) is used only as a fallback and
    # is explicitly documented in FAO-56 as "less recommended".
    dew_point = (
        work["T2MDEW"].to_numpy(dtype=float)
        if "T2MDEW" in work.columns
        else np.full(len(work), np.nan)
    )
    dew_valid = np.isfinite(dew_point) & (dew_point <= t_mean)
    ea_dew = np.array([_saturation_vapour_pressure(t) for t in np.where(dew_valid, dew_point, 0.0)])
    ea_rh = np.array([_saturation_vapour_pressure(t) for t in t_mean]) * humidity / 100.0
    ea_source = np.where(dew_valid, "dewpoint", "mean_rh")
    ea = np.where(dew_valid, ea_dew, ea_rh)

    day_of_year = pd.to_datetime(work["date"]).dt.dayofyear.to_numpy()
    ra = np.array([_extraterrestrial_radiation(int(d), latitude) for d in day_of_year])
    elevation = np.array([_estimate_elevation_m(p) for p in pressure])

    delta = np.array([_vapour_pressure_slope(t) for t in t_mean])
    es = np.array([_saturation_vapour_pressure(t) for t in t_mean])
    gamma = PSYCHROMETRIC_COEFF * pressure

    # Clear-sky radiation, used only for the net longwave term.
    clear_sky = (0.75 + 2.0e-5 * elevation) * ra

    # Net shortwave: albedo-reflected fraction of incoming shortwave.
    net_shortwave = (1.0 - ALBEDO) * solar

    # Net longwave, Eq. 39. The emissivity term falls as actual vapour pressure
    # rises, which is why humid nights radiate less heat back to the sky.
    t_max_k = t_max + 273.16
    t_min_k = t_min + 273.16
    longwave_emissivity = (0.34 - 0.14 * np.sqrt(np.clip(ea, 0.0, None))) * (
        1.35 * (solar / np.clip(clear_sky, 1e-6, None)) - 0.35
    )
    net_longwave = (
        STEFAN_BOLTZMANN
        * ((t_max_k**4 + t_min_k**4) / 2.0)
        * longwave_emissivity
    )

    net_radiation = net_shortwave - net_longwave
    # Soil heat flux is negligible at the daily step, so G is set to zero per
    # FAO-56 Annex 2. It only becomes material at the hourly time step.
    soil_heat_flux = 0.0

    # Penman-Monteith, Eq. 6. u2 is already at the 2 m reference height, so no
    # logarithmic wind adjustment is required.
    u2 = wind
    numerator = (
        0.408 * delta * (net_radiation - soil_heat_flux)
        + gamma * (900.0 / (t_mean + 273.0)) * u2 * (es - ea)
    )
    denominator = delta + gamma * (1.0 + 0.34 * u2)
    et0 = np.where(np.abs(denominator) > 1e-9, numerator / denominator, np.nan)

    # Negative ET0 is physically meaningless; it only arises from numerical
    # noise in the longwave term on very still, overcast nights.
    et0 = np.clip(et0, 0.0, 25.0)

    invalid = ~np.isfinite(et0) | ~np.isfinite(net_radiation)
    work["et0_mm_day"] = np.where(invalid, np.nan, np.round(et0, 3))
    work["rn_mj_m2_day"] = np.where(invalid, np.nan, np.round(net_radiation, 3))
    work["g_vapour_kpa"] = np.round(ea, 4)
    work["vapour_pressure_source"] = ea_source
    work["elevation_m"] = np.round(elevation, 1)
    work["ra_mj_m2_day"] = np.round(ra, 3)
    return work


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def load_real_weather(
    start_year: int = 2015,
    end_year: int = 2024,
    stations: tuple[Station, ...] = STATIONS,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch the real multi-station daily weather corpus used for training.

    The default window covers ten complete years across twelve stations, which
    yields roughly 43,000 daily records. That is enough to span a full
    ten-year climatological cycle of solar geometry and interannual rainfall
    variability, so the water-balance model is not tuned to a single season.
    """
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)

    frames: list[pd.DataFrame] = []
    for index, station in enumerate(stations, start=1):
        if verbose:
            print(
                f"  [{index:>2}/{len(stations)}] NASA POWER {station.code} "
                f"{station.name} ({station.climate}) ...",
                flush=True,
            )
        frames.append(fetch_power_daily(station, start, end, use_cache=use_cache))

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined["doy"] = combined["date"].dt.dayofyear
    combined["month"] = combined["date"].dt.month
    combined["year"] = combined["date"].dt.year
    combined["is_monsoon"] = (
        (combined["month"] >= 6) & (combined["month"] <= 9)
    ).astype(int)

    if verbose:
        valid = int(combined["et0_mm_day"].notna().sum())
        print(
            f"  real weather corpus: {len(combined):,} daily records, "
            f"{valid:,} with valid ET0 "
            f"(ET0 mean {combined['et0_mm_day'].mean():.2f} mm/day)",
            flush=True,
        )
    return combined


def climate_summary(weather: pd.DataFrame) -> pd.DataFrame:
    """Per-station climate statistics, used to report dataset coverage."""
    grouped = weather.groupby(["station", "climate"], observed=True)
    summary = grouped.agg(
        days=("et0_mm_day", "size"),
        et0_mean=("et0_mm_day", "mean"),
        et0_max=("et0_mm_day", "max"),
        temp_mean=("T2M", "mean"),
        temp_max=("T2M_MAX", "max"),
        humidity_mean=("RH2M", "mean"),
        rain_annual=("PRECTOTCORR", "sum"),
        wind_mean=("WS2M", "mean"),
    ).reset_index()
    summary["rain_annual"] = summary["rain_annual"] / summary.groupby("station")["days"].transform("size")
    return summary.round(2)
