r"""
Independent observational sources, fetched over the network and cached to disk.

This module exists because of an uncomfortable admission in the project README:
every water number the system produces traces back to two things, the NASA POWER
atmosphere and the FAO-56 tables, and both of those were written by people who
never saw the fields this system is meant to serve. A simulation that validates
itself proves nothing.

So the project takes its inputs from a second, independent implementation and
compares.

`Open-Meteo Historical Weather API` is the source used here. It serves the
ERA5 reanalysis and, critically, it computes
`et0_fao_evapotranspiration` with its *own* implementation of the FAO-56
Penman-Monteith equation, from ERA5 rather than from POWER. That makes it a
genuine cross-check and not a restatement: if the two ET0 series agree, the
equation and its implementation are probably right, and if they disagree, the
disagreement is in the input meteorology rather than in the arithmetic.

It also serves observed soil moisture and soil temperature. That is the more
valuable half, because the soil water state is the part of this system that is
purely simulated, and it is therefore the part with no independent evidence
behind it. A simulated depletion trace that disagrees with ERA5 soil moisture is
evidence that the water balance is wrong.

Provenance and caveats, which matter for how the results are read:

- ERA5 is a reanalysis: a physics-based model constrained by observations, with
  its own parameterisations, at roughly 31 km resolution. It is not a soil
  probe. Agreement between the two is corroboration, not ground truth, and at a
  station a few tens of kilometres from the ERA5 grid cell, disagreement is
  expected even with a correct implementation.
- ERA5 soil moisture is the top layer, 0-7 cm, while the crop's roots reach
  0.3-1.8 m. Comparing a simulated root-zone mean against a 7 cm layer is
  comparing different quantities, and the comparison is reported as a correlation
  over season rather than as a calibrated match.
- ERA5 is known to be drier than reality in some humid regions and to overstate
  soil moisture in others. A systematic bias here is a property of ERA5, not
  evidence that the water balance should be shifted to match it.
- `CHIRPS` and `TerraClimate` were both evaluated and neither is reachable from
  this environment; CHIRPS is the better rainfall reference and remains the
  recommended upgrade.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

#: Archive endpoint. The forecast host is deliberately not used: it will not
#: serve 2015 and would mix a model's reanalysis with its own nowcast.
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

#: The variables worth having, and why each is here.
#:
#: `et0_fao_evapotranspiration` is the independent cross-check on the demand
#: side. The rest are the inputs to that equation, kept so the comparison can be
#: decomposed: if the two ET0 series disagree, knowing whether the humidity or the
#: wind or the radiation differs turns a mystery into a diagnosis.
OPEN_METEO_DAILY: tuple[str, ...] = (
    "et0_fao_evapotranspiration",
    "precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "dew_point_2m_mean",
    "wind_speed_10m_mean",
    "shortwave_radiation_sum",
    "surface_pressure_mean",
    "cloud_cover_mean",
)

#: Hourly soil variables, aggregated to daily means. These are the observations
#: that make an independent check on the simulated water balance possible at all.
OPEN_METEO_HOURLY: tuple[str, ...] = (
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_temperature_0_to_7cm",
)


@dataclass
class StationPoint:
    """A station's coordinates, needed to request either source."""

    name: str
    latitude: float
    longitude: float
    elevation_m: float | None = None


def _cache_path(cache_dir: Path, name: str) -> Path:
    return Path(cache_dir) / f"{name}.csv"


def _chunked_days(start: date, end: date, chunk: int = 365) -> list[tuple[date, date]]:
    out = []
    cursor = start
    while cursor <= end:
        stop = min(end, cursor + timedelta(days=chunk - 1))
        out.append((cursor, stop))
        cursor = stop + timedelta(days=1)
    return out


def _get_json(url: str, params: dict, retries: int = 4, pause: float = 1.5) -> dict:
    """
    Fetch JSON, with backoff on rate limiting.

    The archive endpoint returns 429 under sustained request, and a 429 during the
    year-chunked fetch used to abandon that station entirely rather than retry
    after backing off. The first run of this lost NAG to it. `Retry-After` is
    honoured when present, because the endpoint does send it, and the backoff
    grows quadratically so a long run against a shared budget eventually gets
    through without hammering.
    """
    import requests

    last = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=90)
            if response.status_code == 200:
                return response.json()
            last = f"HTTP {response.status_code}"
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else pause * (3 ** attempt)
                time.sleep(min(wait, 90.0))
                continue
        except Exception as exc:  # noqa: BLE001 - network errors vary by library
            last = str(exc)
        time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"request failed after {retries} attempts: {last}")


#: Seconds to wait between requests to the same host. The archive endpoint
#: documents fair use and starts refusing sustained clients; a ten-year, twelve
#: station fetch is 240 requests and the default no-delay loop trips the limit.
REQUEST_THROTTLE_SECONDS = 1.2


def fetch_open_meteo(
    station: StationPoint,
    start_year: int = 2015,
    end_year: int = 2024,
    cache_dir: str | Path = "data/cache/external",
    force: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Daily ERA5 meteorology and soil state for one station, cached as CSV.

    Requests are chunked by year. The archive endpoint is generous but a
    ten-year, sixteen-variable request is a few megabytes, and a failed transfer
    partway through loses the lot. Yearly chunks also make a partial cache useful.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    target = _cache_path(cache, f"open_meteo_{station.name}")

    daily_rows: list[pd.DataFrame] = []
    hourly_rows: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        year_cache = _cache_path(cache, f"open_meteo_{station.name}_{year}")
        if year_cache.exists() and not force:
            payload = pd.read_csv(year_cache, parse_dates=["date"])
            daily_rows.append(payload)
            if "soil_moisture_0_to_7cm" in payload.columns:
                hourly_rows.append(payload)
            continue
        if verbose:
            print(f"    {station.name} {year} ...", end=" ", flush=True)
        time.sleep(REQUEST_THROTTLE_SECONDS)
        payload_daily = _get_json(
            OPEN_METEO_ARCHIVE,
            {
                "latitude": station.latitude,
                "longitude": station.longitude,
                "start_date": f"{year}-01-01",
                "end_date": f"{year}-12-31",
                "daily": ",".join(OPEN_METEO_DAILY),
                "timezone": "auto",
            },
        )
        daily = pd.DataFrame(payload_daily["daily"])
        daily["date"] = pd.to_datetime(daily["time"])
        daily = daily.drop(columns=["time"])
        try:
            payload_hourly = _get_json(
                OPEN_METEO_ARCHIVE,
                {
                    "latitude": station.latitude,
                    "longitude": station.longitude,
                    "start_date": f"{year}-01-01",
                    "end_date": f"{year}-12-31",
                    "hourly": ",".join(OPEN_METEO_HOURLY),
                    "timezone": "auto",
                },
            )
            hourly = pd.DataFrame(payload_hourly["hourly"])
            hourly["date"] = pd.to_datetime(hourly["time"]).dt.normalize()
            # Daily mean of the hourly series: the soil layers are reported
            # volumetrically and an hourly mean is the right resolution to compare
            # against a daily water balance step.
            hourly = (
                hourly.drop(columns=["time"])
                .groupby("date", as_index=False)
                .mean(numeric_only=True)
            )
            daily = daily.merge(hourly, on="date", how="left")
        except RuntimeError as exc:
            if verbose:
                print(f"soil unavailable ({exc}); continuing without it")
        daily["station"] = station.name
        daily["latitude"] = station.latitude
        daily["longitude"] = station.longitude
        daily["source"] = "open-meteo-era5"
        year_cache.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(year_cache, index=False)
        daily_rows.append(daily)
        if verbose:
            print(f"{len(daily)} days", flush=True)

    if not daily_rows:
        return pd.DataFrame()
    frame = pd.concat(daily_rows, ignore_index=True)
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    # The per-station aggregate is derived from the yearly caches, which are the
    # source of truth. It is rewritten unconditionally on purpose. An earlier
    # version only wrote it when the file was absent, which meant the first
    # partial fetch froze a truncated aggregate that a later complete fetch could
    # never replace, and every downstream report silently read the truncated file.
    # Writing is cheap; a stale cache that looks complete is not.
    frame.to_csv(target, index=False)
    return frame


def _station_years(cache: Path, name: str) -> list[Path]:
    return sorted(cache.glob(f"open_meteo_{name}_[0-9][0-9][0-9][0-9].csv"))


def load_external(
    station_names: list[str], cache_dir: str | Path = "data/cache/external"
) -> pd.DataFrame:
    """Read a previously cached external fetch, without touching the network.

    Reconstructs each station from its yearly caches when those are present, and
    falls back to the per-station aggregate only if they are not. The yearly files
    are preferred because a station's aggregate is a derived convenience copy, and
    a derived copy is only as current as the last write that happened to pass
    through it.
    """
    cache = Path(cache_dir)
    frames: list[pd.DataFrame] = []
    for name in station_names:
        years = _station_years(cache, name)
        if years:
            frames.append(
                pd.concat(
                    [pd.read_csv(y, parse_dates=["date"]) for y in years],
                    ignore_index=True,
                )
            )
            continue
        target = _cache_path(cache, f"open_meteo_{name}")
        if target.exists():
            frames.append(pd.read_csv(target, parse_dates=["date"]))
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values(["station", "date"]).drop_duplicates(
        ["station", "date"]
    ).reset_index(drop=True)


def describe_provenance() -> dict:
    """Machine-readable record of what this source is, for the reports."""
    return {
        "source": "Open-Meteo Historical Weather API",
        "url": OPEN_METEO_ARCHIVE,
        "underlying_data": "ERA5 reanalysis (ECMWF)",
        "independent_et0": "et0_fao_evapotranspiration, separate FAO-56 implementation",
        "daily_variables": list(OPEN_METEO_DAILY),
        "soil_variables": list(OPEN_METEO_HOURLY),
        "caveats": [
            "ERA5 is a reanalysis at roughly 31 km, not a ground observation",
            "ERA5 soil moisture is the 0-7 cm layer while crop roots reach 0.3-1.8 m",
            "agreement is corroboration, not calibration; do not shift the water "
            "balance to fit ERA5",
            "CHIRPS and TerraClimate were unreachable and are the preferred "
            "upgrades for precipitation and reference ET respectively",
        ],
    }
