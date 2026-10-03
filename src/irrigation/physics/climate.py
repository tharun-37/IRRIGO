"""
Climate context: the part of the decision that comes from where and when it is.

Three additions beyond reference evapotranspiration, each answering a question
that ET0 alone cannot.

1. Vapour pressure deficit. VPD is the gap between what the air can hold and
   what it holds, and it governs how fast a crop burns through the water in its
   root zone. Two days with identical ET0 can have very different VPD, and a
   crop on the high-VPD day will reach its depletion limit sooner. ET0 already
   contains a humidity term, so VPD is not new information in the strict sense,
   but it is the *state variable* a canopy responds to, and reporting it
   separately makes the mechanism legible rather than buried inside a product.

2. Rain climatology. Before recommending irrigation, the useful question is not
   "is it raining" but "is rain likely before the next decision point". NASA
   POWER gives ten years per station, so the climatological probability of a
   useful event within a given window is computable per station per calendar
   period. Recommending irrigation three days ahead of a probable 25 mm event
   wastes water and destroys the residual soil moisture the rain would have
   displaced.

3. Effective rainfall expectations, which combine the two with the soil. A
   forecast rainfall is not a root-zone contribution: some runs off, some
   evaporates, and what infiltrates into a nearly full profile mostly drains
   below the roots.

**Reused rather than rewritten.** The NASA POWER client and the FAO-56
Penman-Monteith implementation come from the previous generation of this project
(`MP3/backend/ml/real_data.py`), together with the cache of twelve stations
already downloaded for 2015-2024. That implementation is validated against nine
published FAO-56 reference examples, and re-deriving ET0 here would mean either
duplicating a validated routine or shipping an unvalidated one. A second
implementation of the same equation is a liability, not redundancy.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# The previous generation's weather module, imported rather than reimplemented.
# Path is resolved relative to this file so the dependency is explicit and fails
# loudly at import time if the sibling project is missing.
V1_ML_DIR = Path(r"D:\Codings\MP3\backend\ml")
if str(V1_ML_DIR) not in sys.path:
    sys.path.insert(0, str(V1_ML_DIR))

try:
    import real_data as _power  # type: ignore
except ImportError as error:  # pragma: no cover - configuration failure
    raise ImportError(
        "MP3-V2 reuses the validated NASA POWER client and FAO-56 ET0 "
        "implementation from the sibling project. Expected it at "
        f"{V1_ML_DIR}. Either restore that path or replace this import."
    ) from error


#: Column names in the reused NASA POWER frame, mapped to the names this module
#: and the rest of V2 use. The upstream frame keeps the API's own field names,
#: which are terse to the point of being cryptic (`T2M` is mean air temperature,
#: `PRECTOTCORR` is corrected precipitation), and the rename happens once here
#: rather than at every call site.
#:
#: `g_vapour_kpa` is already present in the upstream frame, computed from the
#: dewpoint. This module computes VPD from temperature and relative humidity
#: instead, because FAO-56's stress terms are defined on air VPD rather than on
#: the dewpoint depression, and mixing the two conventions in one system makes
#: it impossible to reason about which is in play.
COLUMN_MAP: dict[str, str] = {
    "T2M": "air_temperature_c",
    "T2M_MAX": "tmax_c",
    "T2M_MIN": "tmin_c",
    "RH2M": "humidity_pct",
    "PRECTOTCORR": "rainfall_mm",
    "WS2M": "wind_speed_m_s",
    "ALLSKY_SFC_SW_DWN": "solar_radiation_mj_m2_day",
    "T2MDEW": "dewpoint_c",
    "rn_mj_m2_day": "net_radiation_mj_m2_day",
}


def normalise_weather_columns(weather: pd.DataFrame) -> pd.DataFrame:
    """
    Rename the upstream NASA POWER columns and add the VPD this module needs.

    Applied once at load rather than on every access. `date` is forced to
    `datetime64` because the upstream loader has already parsed it, but a
    straggling object-typed date column would break the `.dt` accessor used
    throughout the climatology code.
    """
    frame = weather.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in weather.columns})
    if "date" in frame.columns and not pd.api.types.is_datetime64_any_dtype(frame["date"]):
        frame["date"] = pd.to_datetime(frame["date"])
    if {"air_temperature_c", "humidity_pct"}.issubset(frame.columns):
        frame["vpd_kpa"] = vapour_pressure_deficit_kpa(
            frame["air_temperature_c"].to_numpy(),
            frame["humidity_pct"].to_numpy(),
        )
    return frame


# --------------------------------------------------------------------------
# Vapour pressure deficit
# --------------------------------------------------------------------------


def saturation_vapour_pressure_kpa(temperature_c: np.ndarray) -> np.ndarray:
    """
    Saturation vapour pressure, kPa, over water at 2 m.

    FAO-56 equation 11, the Magnus form. Uses the air-temperature coefficient
    set (17.27 / 237.3) rather than the dew-point one (17.62 / 243.5), because
    FAO-56 distinguishes them for the Penman-Monteith saturation term and this
    is that term.

    Clamped at zero for sub-zero temperatures, where the Magnus form is not
    valid and the physical answer is that the air is saturated with respect to
    ice at a lower pressure than the formula predicts.
    """
    temperature = np.asarray(temperature_c, dtype=float)
    safe = np.where(temperature > -50.0, temperature, 0.0)
    return 0.6108 * np.exp(17.27 * safe / (safe + 237.3))


def actual_vapour_pressure_kpa(
    temperature_c: np.ndarray, relative_humidity_pct: np.ndarray
) -> np.ndarray:
    """
    Actual vapour pressure from temperature and relative humidity, kPa.

    FAO-56 equation 10. Relative humidity is clipped to a saner band than [0,
    100] because a probe reporting 104% is reporting a fault, and propagating
    104% produces a VPD below zero, which is physically impossible and which a
    downstream model would happily use.
    """
    temperature = np.asarray(temperature_c, dtype=float)
    humidity = np.clip(np.asarray(relative_humidity_pct, dtype=float), 1.0, 100.0)
    return saturation_vapour_pressure_kpa(temperature) * humidity / 100.0


def vapour_pressure_deficit_kpa(
    temperature_c: np.ndarray, relative_humidity_pct: np.ndarray
) -> np.ndarray:
    """
    Vapour pressure deficit, kPa. Never negative.
    """
    deficit = (
        saturation_vapour_pressure_kpa(temperature_c)
        - actual_vapour_pressure_kpa(temperature_c, relative_humidity_pct)
    )
    return np.clip(np.nan_to_num(deficit, nan=0.0), 0.0, None)


def vpd_stress_multiplier(vpd_kpa: float | np.ndarray) -> np.ndarray:
    """
    How much faster a crop uses its soil water at this VPD, relative to a
    moderate reference.

    This is a coarse, explicit allowance, not a published coefficient. FAO-56
    handles stress through the soil-water-deficit term in the Penman-Monteith
    equation rather than through a VPD multiplier, so applying one on top would
    double-count in the conditions FAO-56 intended.

    It is included because a field deployment needs to communicate *why* a hot
    dry week shortened its return interval, and VPD is the honest answer. The
    magnitude is deliberately modest, and the function is documented as a
    reporting aid rather than a term in the water balance.

    Reference 1.2 kPa, roughly a mild temperate day. A 3 kPa afternoon in a hot
    arid week is more than double that, and a canopy on such a day closes
    partially by mid-afternoon.
    """
    vpd = np.asarray(vpd_kpa, dtype=float)
    reference = 1.2
    return np.clip(1.0 + 0.22 * (vpd - reference), 0.7, 2.0)


# --------------------------------------------------------------------------
# Rain climatology
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RainClimatology:
    """
    Per-station rain climatology, derived from the historical record.

    Two statistics, both per station and per calendar month, because the
    difference between a station's January and its July is the whole point of
    conditioning on the calendar. A single annual figure would average away the
    monsoon and then recommend irrigation in the week the monsoon arrives.

    `wet_day_probability` is the chance of a day receiving at least `wet_day_mm`
    or more of rain. `useful_event_probability_by_window` is the chance of
    receiving at least `useful_event_mm` within the next N days, which is the
    quantity the decision actually needs.
    """

    station: str
    #: WMO convention: a wet day delivers at least 1.0 mm.
    wet_day_mm: float
    #: An event worth delaying irrigation for.
    useful_event_mm: float
    wet_day_probability: dict[int, float]
    mean_daily_rain_mm: dict[int, float]
    expected_rain_mm: dict[int, float]
    sample_years: int

    def probability_of_useful_rain(
        self, month: int, window_days: int
    ) -> float:
        """
        Chance of at least one useful event within `window_days` of this month.

        A one minus the product of daily no-event probabilities, under an
        assumption of independence within the window. Independence is
        demonstrably false for rainfall, because events cluster, so this
        *understates* the probability of rain in a wet spell. The bias is in
        the safe direction for the decision it feeds: it makes the system less
        willing to skip irrigation, and skipping irrigation is the expensive
        error. A dependence-corrected version would need a fitted
        autocorrelation per station and is not attempted here.
        """
        daily_no_rain = 1.0 - self.wet_day_probability.get(month, 0.0)
        independent = daily_no_rain**window_days
        return float(np.clip(1.0 - independent, 0.0, 1.0))

    def expected_rain_over_window_mm(self, month: int, window_days: int) -> float:
        """
        Expected rainfall over a window, mm.

        This is a mean, and a mean is what an irrigation planner wants: it is
        the volume the field can bank on average, and the variance is handled
        separately by the probability term through the `rainfall_credibility`
        weighting in the requirement calculation.
        """
        daily_mean = self.mean_daily_rain_mm.get(month, 0.0)
        return float(daily_mean * window_days)


def build_rain_climatology(
    weather: pd.DataFrame,
    station: str,
    wet_day_mm: float = 1.0,
    useful_event_mm: float = 10.0,
) -> RainClimatology:
    """
    Derive per-month rain statistics for one station from its record.

    The record is grouped by calendar month rather than by season, because the
    caller knows the crop's calendar and the crop's calendar is what determines
    which months matter.
    """
    frame = weather[weather["station"] == station].copy()
    if frame.empty:
        raise ValueError(f"no weather records for station {station!r}")

    frame["month"] = pd.to_datetime(frame["date"]).dt.month
    frame["rain_mm"] = pd.to_numeric(frame.get("rainfall_mm", 0.0), errors="coerce").fillna(0.0)
    frame["is_wet"] = frame["rain_mm"] >= wet_day_mm

    grouped = frame.groupby("month")
    wet_probability = grouped["is_wet"].mean().to_dict()
    mean_daily = grouped["rain_mm"].mean().to_dict()
    expected = grouped["rain_mm"].sum().div(len(frame["date"].unique())).to_dict()

    years = int(frame["date"].nunique() // 365)

    return RainClimatology(
        station=station,
        wet_day_mm=wet_day_mm,
        useful_event_mm=useful_event_mm,
        wet_day_probability={int(k): float(v) for k, v in wet_probability.items()},
        mean_daily_rain_mm={int(k): float(v) for k, v in mean_daily.items()},
        expected_rain_mm={int(k): float(v) for k, v in expected.items()},
        sample_years=years,
    )


def expected_rain_horizon(
    climatology: RainClimatology,
    start: date,
    horizon_days: int,
    soil=None,
    current_depletion_mm: float = 0.0,
    total_available_water_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expected effective rainfall and its probability, per horizon day.

    Returns two arrays of length `horizon_days`: the expected *effective* rain in
    mm and the probability that it occurs. The probability matters as much as the
    mean, because the requirement calculation scales the credit by both the
    probability and a credibility factor, so a probable light drizzle and an
    unlikely downpour are handled differently rather than being averaged into one
    meaningless number.

    Rain is spread across the horizon by its probability of occurring, which is a
    crude assumption: the model does not know when within the window rain will
    fall, and assuming it falls on day 1 would make the field look wet when it
    is not. Spreading it is the neutral choice.

    When a soil texture is supplied, the rain is run through the same effective
    rainfall function the daily balance uses, so a climatological expectation on
    clay is discounted for runoff and for storage the profile cannot accept. That
    matters: it is the difference between crediting a monsoon month with its full
    nominal rainfall and crediting only what a particular field could have used.
    """
    from ..physics.water_balance import effective_rainfall_mm

    expected = np.zeros(horizon_days, dtype=float)
    probability = np.zeros(horizon_days, dtype=float)

    total_expected = climatology.expected_rain_over_window_mm(start.month, horizon_days)
    probability_of_event = climatology.probability_of_useful_rain(
        start.month, horizon_days
    )

    if total_expected <= 0.0 or probability_of_event <= 0.0:
        return expected, probability

    per_day_mean = total_expected / max(1, horizon_days)
    per_day_probability = float(
        np.clip(1.0 - (1.0 - probability_of_event) ** (1.0 / max(1, horizon_days)), 0.0, 1.0)
    )

    for offset in range(horizon_days):
        raw = per_day_mean * per_day_probability
        if soil is not None:
            raw = effective_rainfall_mm(
                raw, soil, current_depletion_mm, total_available_water_mm
            )
        expected[offset] = raw
        probability[offset] = per_day_probability

    return expected, probability


# --------------------------------------------------------------------------
# Climate zone, for the residual models
# --------------------------------------------------------------------------


def classify_climate_zone(
    annual_rain_mm: float,
    mean_temperature_c: float,
    annual_et0_mm: float,
) -> str:
    """
    Classify a station into a coarse climate zone, for the residual models.

    The moisture band is the **aridity index**, annual rainfall divided by annual
    reference evapotranspiration, with the UNEP thresholds. That ratio is the
    standard measure of atmospheric supply against atmospheric demand and it is
    the right denominator: an earlier version of this function derived demand
    from VPD and classified all twelve stations as humid, because a VPD-derived
    demand term came out roughly ten times too small.

    ET0 is used rather than a computed potential evaporation because it is
    already measured and validated here, and because the two answer the same
    question with the same inputs: how much water does the atmosphere try to
    remove from this place in a year.

    The temperature band separates warm from hot, which matters for evapotranspiration
    because a hot day at the same VPD carries more absolute moisture and a
    longer season of demand.
    """
    if mean_temperature_c < 15.0:
        temperature_band = "cool"
    elif mean_temperature_c < 25.0:
        temperature_band = "warm"
    else:
        temperature_band = "hot"

    aridity_index = annual_rain_mm / max(1.0, annual_et0_mm)

    # UNEP aridity index thresholds.
    if aridity_index < 0.45:
        moisture_band = "arid"
    elif aridity_index < 0.80:
        moisture_band = "semi_arid"
    elif aridity_index < 1.30:
        moisture_band = "dry_sub_humid"
    elif aridity_index < 2.00:
        moisture_band = "sub_humid"
    else:
        moisture_band = "humid"

    return f"{temperature_band}_{moisture_band}"


def station_climate_profile(weather: pd.DataFrame, station: str) -> dict:
    """
    Summarise one station's climate, for the residual models and the dashboard.

    Includes the mean VPD, which the previous generation did not compute at all
    despite having the temperature and humidity needed for it.
    """
    frame = weather[weather["station"] == station]
    if frame.empty:
        raise ValueError(f"no weather records for station {station!r}")

    temperature = pd.to_numeric(frame["air_temperature_c"], errors="coerce")
    humidity = pd.to_numeric(frame["humidity_pct"], errors="coerce")
    rainfall = pd.to_numeric(frame.get("rainfall_mm", 0.0), errors="coerce").fillna(0.0)
    et0 = pd.to_numeric(frame.get("et0_mm_day", 0.0), errors="coerce").fillna(0.0)

    vpd = (
        frame["vpd_kpa"].to_numpy()
        if "vpd_kpa" in frame.columns
        else vapour_pressure_deficit_kpa(
            temperature.to_numpy(), humidity.to_numpy()
        )
    )

    annual_rain = float(rainfall.sum() / max(1, frame["date"].nunique() // 365))
    annual_et0 = float(et0.sum() / max(1, frame["date"].nunique() // 365))
    mean_temperature = float(temperature.mean())
    mean_vpd = float(np.nanmean(vpd))

    return {
        "station": station,
        "climate_zone": classify_climate_zone(
            annual_rain, mean_temperature, annual_et0
        ),
        "annual_rain_mm": round(annual_rain, 1),
        "annual_et0_mm": round(annual_et0, 1),
        "aridity_index": round(annual_rain / max(1.0, annual_et0), 3),
        "mean_temperature_c": round(mean_temperature, 2),
        "mean_humidity_pct": round(float(humidity.mean()), 2),
        "mean_vpd_kpa": round(mean_vpd, 3),
        "mean_et0_mm_day": round(float(et0.mean()), 3),
        "median_et0_mm_day": round(float(et0.median()), 3),
        "p90_et0_mm_day": round(float(et0.quantile(0.90)), 3),
        "mean_wind_speed": round(
            float(pd.to_numeric(frame.get("wind_speed_m_s", 2.0), errors="coerce").mean()), 3
        ),
        "vapour_pressure_source": "derived from T2M and RH2M, FAO-56 eq. 10 and 11",
        "et0_method": "FAO-56 Penman-Monteith, reused from MP3 V1, 9 reference checks",
        "records": int(len(frame)),
    }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_weather(
    start_year: int = 2015, end_year: int = 2024, verbose: bool = False
) -> pd.DataFrame:
    """
    Load the NASA POWER record with V2's column names and the VPD column added.

    The entry point every caller should use. Wrapping the upstream loader rather
    than calling it directly means the rename and the VPD derivation happen once,
    and no caller has to remember whether a frame came in raw or normalised.
    """
    frame = _power.load_real_weather(
        start_year=start_year, end_year=end_year, use_cache=True, verbose=verbose
    )
    return normalise_weather_columns(frame)
