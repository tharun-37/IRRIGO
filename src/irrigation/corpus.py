r"""
Builds the crop-wise training corpus by running a real water balance per field.

Unlike the previous generation's builder, the water requirement recorded here is
the target the model learns, and it is not a clamp. Two design rules govern it.

**The target is the requirement, not the application.** The model learns how much
water a field needs over a horizon. Whether that can go on in one pass is a
property of the pump and the soil, and it is recorded as a separate column. The
previous generator produced `min(TAW - Dr, infiltration)`, and because the
second term is a per-soil constant it bound on two thirds of all events, so the
corpus recorded 22, 12 and 32 mm over and over. A model fitted to that is a
soil-texture lookup, which is precisely what happened: R-squared 0.082 on an
unseen station.

**Real weather in, simulated soil out.** NASA POWER daily records drive ET0 and
the rain, so the corpus inherits real climate including its interannual
variability. What is simulated is the soil water state, because that is what has
no public dataset. Everything between the two, the crop coefficient, the root
profile and the texture constants, is a published FAO-56 value rather than
something invented here.

Sampling stride is a deliberate compromise. A daily record for 117 series over
five seasons is a few hundred thousand rows, which trains slowly and mostly
teaches the model what it already knows about the diurnal cycle. Every third
day keeps a few tens of thousands of rows, preserves irrigation events, and
keeps the between-event depletion accumulation visible, which is what the model
actually needs to learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data.crops import CropParameters, crop as lookup_crop
from .data.cultivation import (
    IrrigationMethod,
    effective_max_depth_mm,
    nitrogen_kc_multiplier,
)
from .data.soil import SoilTexture, restricted_root_depth_m, texture as lookup_texture
from .phenology.gdd import (
    GrowingDegreeDays,
    GrowthStage,
    effective_depletion_fraction,
    effective_root_depth_m,
    stage_from_gdd,
)
from .physics.climate import build_rain_climatology
from .physics.kc_curve import adjust_kc_for_soil_moisture, build_breakpoints, kc_at_gdd
from .physics.water_balance import (
    SoilWaterState,
    advance_depletion,
    effective_rainfall_mm,
    water_stress_index,
)

#: Days between sampled records. See the module docstring for why.
SAMPLE_STRIDE_DAYS = 3

#: Horizon for the recorded requirement, days. Fourteen is roughly one irrigation
#: return period for most surface methods, so the target answers "what does this
#: field need before the next round" rather than an arbitrary window.
REQUIREMENT_HORIZON_DAYS = 14

#: Seasons simulated per field series. Each restarts the soil profile at field
#: capacity and the thermal clock at zero, so a series contributes repeated
#: observations of the same crop-on-soil pairing under different weather rather
#: than one unbroken run that would confound season with year.
SAMPLE_SEASONS = 5

#: Soil chemistry and microclimate that are properties of a field rather than of a
#: day. Drawn once per series.
SOIL_PH_NOISE = 0.06
SOIL_TEMP_OFFSET_C = 2.5
SOIL_TEMP_NOISE_C = 1.2
MOISTURE_PROBE_NOISE = 0.012
EC_BASE_DS_M = 0.35
EC_PER_10MM_DEPLETION = 0.02
EC_NOISE = 0.08

#: Multipliers applied to Kc, from `data.cultivation`.
MULCH_KC_FACTOR = 0.92
MULCH_STAGES = (GrowthStage.INITIAL, GrowthStage.DEVELOPMENT)

#: Fraction of readily available water at which the simulated operator irrigates.
#: Below FAO-56's stress threshold of 1.0 on the relative scale, deliberately: a
#: real operator irrigates before the crop is visibly suffering, because driving
#: a field to its stress threshold and then rescuing it costs yield that cannot
#: be recovered afterwards.
OPERATOR_TRIGGER_FRACTION = 0.55


@dataclass
class FieldSpec:
    """One simulated field: a crop on a soil, at a station, for a season."""

    station: str
    crop_name: str
    soil_name: str
    crop_variant: str | None = None
    method_name: str = "Sprinkler"
    field_area_m2: float = 1_000.0
    mulched: bool = False
    nitrogen_regime: float = 1.0
    restricting_depth_m: float | None = None
    sowing_doy: int = 100
    crop: CropParameters = field(init=False)
    soil: SoilTexture = field(init=False)

    def __post_init__(self) -> None:
        self.crop = lookup_crop(self.crop_name, self.crop_variant)
        self.soil = lookup_texture(self.soil_name)


def _method(name: str) -> IrrigationMethod:
    from .data.cultivation import method as lookup_method

    return lookup_method(name)


def _state(taw: float, raw: float, depletion: float, root_depth: float) -> SoilWaterState:
    return SoilWaterState(
        depletion_mm=depletion,
        total_available_water_mm=taw,
        readily_available_water_mm=raw,
        root_depth_m=root_depth,
    )


def _kc_multiplier(spec: FieldSpec) -> float:
    """Management factors on Kc, applied on top of the FAO-56 curve."""
    multiplier = nitrogen_kc_multiplier(spec.nitrogen_regime)
    if spec.mulched:
        multiplier *= MULCH_KC_FACTOR
    return float(multiplier)


def _days_to_maturity(
    gdd_daily: np.ndarray, season_start: int, crop: CropParameters, frame_length: int
) -> int:
    """
    Days from sowing until the crop has accumulated its season's thermal time.

    Returns a count rather than a flag, because the caller needs to know where to
    stop. A crop whose season runs past the end of the weather record contributes
    every day it has rather than dropping out.
    """
    required = float(crop.gdd_stage_ends[3])
    accumulated = 0.0
    available = frame_length - season_start
    for offset in range(max(1, available)):
        accumulated += float(gdd_daily[season_start + offset])
        if accumulated >= required:
            return offset + 1
    return max(1, available)


def _season_starts(
    frame: pd.DataFrame, sowing_doy: int, seasons: int
) -> list[int]:
    """
    Row indices at which each simulated season begins.

    The frame is sorted by calendar date, so its day-of-year column runs
    1..365, 1..365, ... and is not monotonic. `np.searchsorted` on it is
    therefore meaningless and silently returned an index deep in the record for
    every sowing. Matching on the (year, day-of-year) pair instead is both
    correct and what the intent was.

    Only years at or after the midpoint of the record are used, which leaves at
    least five seasons for a ten-year record while keeping the chosen years
    recent enough to be inside the weather archive.
    """
    doy = frame["doy"].to_numpy()
    years = frame["year"].to_numpy()
    earliest_year = int(np.median(years))
    matches = np.flatnonzero((years >= earliest_year) & (doy >= sowing_doy))
    if matches.size == 0:
        return []
    first = int(matches[0])
    starts = [first]
    for extra in range(365, len(frame) - 1, 365):
        starts.append(first + extra)
    return starts[:seasons]


def _forward_horizon(
    et0: np.ndarray,
    gdd_daily: np.ndarray,
    gdd_now: float,
    index: int,
    crop: CropParameters,
    soil: SoilTexture,
    depletion: float,
    taw: float,
    p: float,
    climatology,
    method: IrrigationMethod,
    month: int,
    kc_multiplier_value: float,
) -> dict:
    """
    Requirement over the coming horizon, using only information available now.

    The loop advances ET0 with the station's own climatological mean rather than
    the realised future. Using the realised future would be leakage: the target
    would contain weather that the features do not yet describe, and a model
    fitted to it would look excellent in evaluation and be useless in service,
    because at six in the morning nobody knows what the weather will do on
    Thursday.

    Rain is credited at the climatological probability, which is a long-run
    average rather than a forecast, and each day's credit is discounted by the
    fraction of days on which rain is likely at all. A version of this function
    hard-coded January, which would have credited a monsoon field with a few
    millimetres in July and irrigated it through a wet week.

    `month` is passed in because the rain climatology is conditioned on the
    calendar month, and the difference between a station's January and its July
    is the entire reason for conditioning.
    """
    horizon = REQUIREMENT_HORIZON_DAYS
    end = min(len(et0), index + horizon + 1)
    days = max(1, end - index)

    breakpoints = build_breakpoints(crop)
    raw = min(taw, p * taw)
    operating_trigger = OPERATOR_TRIGGER_FRACTION * raw

    if climatology is not None:
        expected_total = climatology.expected_rain_over_window_mm(month, days)
        event_probability = climatology.probability_of_useful_rain(month, days)
    else:
        expected_total, event_probability = 0.0, 0.0

    per_day_rain = expected_total / days
    per_day_probability = (
        float(
            np.clip(
                1.0 - (1.0 - event_probability) ** (1.0 / days), 0.0, 1.0
            )
        )
        if event_probability > 0
        else 0.0
    )

    demand = 0.0
    rain_credit = 0.0
    projected_depletion = depletion
    projected_stress_days = 0
    future_gdd = gdd_now

    for offset in range(1, days):
        i = index + offset
        future_gdd += float(gdd_daily[i])
        kc = kc_at_gdd(future_gdd, breakpoints) * kc_multiplier_value
        # A future crop under water stress transpires less than a well-supplied
        # one, so the look-ahead must fall too or it over-forecasts the demand.
        kc = adjust_kc_for_soil_moisture(
            kc, projected_depletion / max(1e-6, taw), p
        )
        etc_day = kc * float(et0[i])
        demand += etc_day

        effective = effective_rainfall_mm(
            per_day_rain * per_day_probability, soil, projected_depletion, taw
        )
        rain_credit += effective

        projected_depletion = float(
            np.clip(projected_depletion + etc_day - effective, 0.0, taw)
        )
        if projected_depletion > raw:
            projected_stress_days += 1

    rain_credit = min(rain_credit, demand)
    soil_supply = min(
        max(0.0, operating_trigger - depletion),
        max(0.0, demand - rain_credit),
    )
    requirement = max(0.0, demand - rain_credit - soil_supply)
    max_application = effective_max_depth_mm(method, soil)

    return {
        "net_requirement_mm": requirement,
        "single_application_mm": min(requirement, max_application),
        "max_application_mm": max_application,
        "demand_mm": demand,
        "rain_credit_mm": rain_credit,
        "soil_supply_mm": soil_supply,
        "projected_stress_days": projected_stress_days,
    }


def simulate_field(
    weather_for_station: pd.DataFrame,
    spec: FieldSpec,
    climatology,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Run the growing seasons for one field and record the daily state.

    The loop is the FAO-56 daily water balance,

        Dr(t+1) = Dr(t) + ETc(t) - pe(t) - Irr(t)

    with ETc from the crop coefficient curve on real thermal time, pe from
    effective rainfall, and Irr from a rule that models a real operator rather
    than an ideal one.

    Three things about the loop shape are load-bearing and each was a bug first:

    The season's end is located before the loop starts, by accumulating GDD to
    the maturity threshold. Discovering maturity by breaking out of a day loop
    works, but combined with a stride filter it can skip every sampled day of a
    short season and record nothing: an earlier version emitted one row per
    season, all at maturity.

    Thermal time resets at each sowing. Accumulating from the start of the
    weather record made every crop instantly mature on day one, which produced an
    empty corpus and no error at all.

    The stride is applied to days since sowing, not to the absolute row index.
    A row index whose modulus happens to land on the stride skips every day of a
    season that began on an offset day.
    """
    crop = spec.crop
    soil = spec.soil
    method = _method(spec.method_name)

    frame = weather_for_station.sort_values("date").reset_index(drop=True)
    if frame.empty:
        return []

    accumulator = GrowingDegreeDays(crop.gdd_base_temp_c)
    breakpoints = build_breakpoints(crop)
    multiplier = _kc_multiplier(spec)

    # Per-series soil chemistry, drawn once so fields differ from one another in
    # pH the way real fields do. Deriving pH from the texture's field capacity
    # instead, as an earlier version did, produced values near 0.02 that clipped
    # to the 3.5 floor on every row and left a constant column.
    ph_low, ph_high = crop.optimal_ph
    ph_baseline = float(rng.uniform(ph_low - 0.5, ph_high + 0.5))

    et0 = frame["et0_mm_day"].to_numpy(dtype=float)
    tmax = frame["tmax_c"].to_numpy(dtype=float)
    tmin = frame["tmin_c"].to_numpy(dtype=float)
    rain = frame["rainfall_mm"].to_numpy(dtype=float)
    vpd = (
        frame["vpd_kpa"].to_numpy(dtype=float)
        if "vpd_kpa" in frame.columns
        else np.zeros(len(frame))
    )
    humidity = frame["humidity_pct"].to_numpy(dtype=float)
    air_temp = frame["air_temperature_c"].to_numpy(dtype=float)
    months = frame["month"].to_numpy()

    gdd_daily = accumulator.daily_gdd(tmax, tmin)
    max_application = effective_max_depth_mm(method, soil)

    records: list[dict] = []

    for season_start in _season_starts(frame, spec.sowing_doy, SAMPLE_SEASONS):
        # A new season starts from field capacity with a fresh root system and a
        # zeroed thermal clock, not from wherever the previous crop left the soil.
        depletion = 0.0
        last_irrigation_index = -10_000

        season_length = _days_to_maturity(gdd_daily, season_start, crop, len(frame))

        for offset in range(season_length):
            index = season_start + offset
            if index >= len(frame):
                break

            gdd_accumulated = float(np.sum(gdd_daily[season_start: index + 1]))
            stage = stage_from_gdd(gdd_accumulated, crop)

            root_depth = restricted_root_depth_m(
                effective_root_depth_m(crop, stage), spec.restricting_depth_m
            )
            taw = soil.taw_mm(root_depth)
            p = effective_depletion_fraction(crop, stage)
            raw = min(taw, p * taw)

            # --- Demand, with the crop's response to its own water state.
            kc = kc_at_gdd(gdd_accumulated, breakpoints) * multiplier
            kc = adjust_kc_for_soil_moisture(
                kc, depletion / max(1e-6, taw), p
            )
            etc = kc * et0[index]

            pe = effective_rainfall_mm(rain[index], soil, depletion, taw)

            # --- The irrigation rule: a real operator, not an ideal one.
            applied = 0.0
            if (
                depletion >= OPERATOR_TRIGGER_FRACTION * raw
                and index - last_irrigation_index >= method.min_interval_days
            ):
                to_field_capacity = taw - depletion
                applied = float(
                    max(0.0, min(to_field_capacity, max_application))
                )
                last_irrigation_index = index

            depletion = advance_depletion(depletion, etc, pe, applied, taw)

            if offset % SAMPLE_STRIDE_DAYS != 0:
                continue

            horizon = _forward_horizon(
                et0, gdd_daily, gdd_accumulated, index,
                crop, soil, depletion, taw, p,
                climatology, method, int(months[index]), multiplier,
            )
            requirement = horizon["net_requirement_mm"]
            single_application = horizon["single_application_mm"]

            # --- Observed state, as a probe would report it.
            #
            # theta = theta_fc - Dr / (1000 * Zr) with Zr in metres, so the
            # divisor is a thousand times the root depth. Dividing by the depth
            # in centimetres instead, as an earlier version did, understated
            # depletion by a hundredfold and pinned every field at field capacity.
            theta = float(
                np.clip(
                    soil.theta_field_capacity
                    - depletion / max(1e-6, 1000.0 * root_depth),
                    0.0,
                    soil.theta_field_capacity,
                )
            )
            observed_theta = theta + rng.normal(0.0, MOISTURE_PROBE_NOISE)

            ec = float(
                np.clip(
                    EC_BASE_DS_M
                    + EC_PER_10MM_DEPLETION * depletion / 10.0
                    + rng.normal(0.0, EC_NOISE),
                    0.05,
                    12.0,
                )
            )
            ph = float(np.clip(ph_baseline + rng.normal(0.0, SOIL_PH_NOISE), 3.5, 9.2))
            soil_temp = float(
                air_temp[index] + SOIL_TEMP_OFFSET_C + rng.normal(0.0, SOIL_TEMP_NOISE_C)
            )

            stress = water_stress_index(
                _state(taw, raw, depletion, root_depth),
                ec_ds_m=ec,
                salinity_threshold_ds_m=crop.salinity_threshold_ds_m,
                salinity_slope=crop.salinity_slope,
            )

            records.append(
                {
                    # --- identity
                    "station": spec.station,
                    "climate_zone": str(frame["climate"].iloc[0]),
                    "crop": spec.crop_name,
                    "crop_variant": spec.crop_variant or "default",
                    "soil_type": spec.soil_name,
                    "irrigation_method": spec.method_name,
                    "date": frame["date"].iloc[index].strftime("%Y-%m-%d"),
                    "doy": int(frame["doy"].iloc[index]),
                    "month": int(months[index]),
                    "year": int(frame["year"].iloc[index]),
                    "is_monsoon": int(frame["is_monsoon"].iloc[index])
                    if "is_monsoon" in frame.columns
                    else 0,
                    # --- configured
                    "days_since_sowing": offset,
                    "gdd_accumulated": round(gdd_accumulated, 2),
                    "gdd_to_next_stage": round(stage.gdd_to_next_stage, 1)
                    if stage.gdd_to_next_stage is not None
                    else 0.0,
                    "season_progress": round(stage.season_progress, 4),
                    "field_area_m2": spec.field_area_m2,
                    "mulched": int(spec.mulched),
                    "nitrogen_regime": round(spec.nitrogen_regime, 3),
                    # --- measured atmosphere
                    "air_temperature_c": round(float(air_temp[index]), 2),
                    "tmax_c": round(float(tmax[index]), 2),
                    "tmin_c": round(float(tmin[index]), 2),
                    "humidity_pct": round(float(humidity[index]), 2),
                    "vpd_kpa": round(float(vpd[index]), 3),
                    "wind_speed_m_s": round(
                        float(frame["wind_speed_m_s"].iloc[index]), 3
                    ),
                    "solar_radiation_mj_m2_day": round(
                        float(frame["solar_radiation_mj_m2_day"].iloc[index]), 3
                    ),
                    "et0_mm_day": round(float(et0[index]), 4),
                    "rainfall_mm": round(float(rain[index]), 3),
                    # --- measured soil
                    "soil_moisture_fraction": round(float(observed_theta), 5),
                    "soil_moisture_pct": round(float(observed_theta * 100.0), 3),
                    "soil_temperature_c": round(soil_temp, 2),
                    "soil_ph": round(ph, 3),
                    "ec_ds_m": round(ec, 4),
                    # --- derived physical state
                    "growth_stage": stage.stage.value,
                    "kc": round(float(kc), 4),
                    "etc_mm": round(float(etc), 4),
                    "root_depth_cm": round(root_depth * 100.0, 2),
                    "taw_mm": round(float(taw), 2),
                    "raw_mm": round(float(raw), 2),
                    "p_depletion_fraction": round(float(p), 4),
                    "depletion_mm": round(float(depletion), 3),
                    "depletion_fraction": round(float(depletion / max(1e-6, taw)), 4),
                    "effective_rainfall_mm": round(float(pe), 3),
                    "irrigation_applied_mm": round(float(applied), 3),
                    "max_application_mm": round(float(max_application), 2),
                    "water_stress": round(float(stress), 4),
                    # --- the target
                    "nir_horizon_mm": round(float(requirement), 4),
                    "horizon_days": REQUIREMENT_HORIZON_DAYS,
                    "single_application_mm": round(float(single_application), 4),
                    "requires_multiple": int(
                        requirement > max_application + 1e-9
                    ),
                    "horizon_demand_mm": round(horizon["demand_mm"], 3),
                    "horizon_rain_credit_mm": round(horizon["rain_credit_mm"], 3),
                    "horizon_soil_supply_mm": round(horizon["soil_supply_mm"], 3),
                    "projected_stress_days": int(horizon["projected_stress_days"]),
                    # --- V1's column, identical to the single application, so the
                    # two generations can be compared like for like. A model
                    # fitted to it is fitting the application cap.
                    "water_required_mm": round(float(single_application), 4),
                }
            )

    return records


def build_corpus(
    weather: pd.DataFrame,
    fields: list[FieldSpec],
    progress: bool = True,
) -> pd.DataFrame:
    """
    Simulate every field and stack the records into one frame.

    `weather` must carry the renamed columns from
    `irrigation.physics.climate.load_weather`, including `vpd_kpa`.
    """
    all_records: list[dict] = []
    for index, spec in enumerate(fields, start=1):
        station_frame = weather[weather["station"] == spec.station]
        if station_frame.empty:
            continue
        climatology = build_rain_climatology(weather, spec.station)
        # Seed from the series identity so a field gets the same soil chemistry and
        # probe noise every time the corpus is rebuilt. Without this, two runs of
        # the same script produce different corpora and no comparison between
        # them means anything.
        seed = abs(
            hash((spec.station, spec.crop_name, spec.soil_name, spec.sowing_doy))
        ) % (2**32)
        records = simulate_field(
            station_frame, spec, climatology, np.random.default_rng(seed)
        )
        all_records.extend(records)
        if progress:
            print(
                f"  [{index:>3}/{len(fields)}] {spec.station:<5} "
                f"{spec.crop_name:<11}{spec.soil_name:<15}{len(records):>6,} rows",
                flush=True,
            )

    frame = pd.DataFrame(all_records)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["station", "crop", "soil_type", "date"]
    ).reset_index(drop=True)
