"""
ET0 enrichment from the cached NASA POWER corpus.

The models were trained on FAO-56 Penman-Monteith evapotranspiration computed
from real daily meteorology. A node in the field has no barometer, no pyranometer
and no knowledge of the last ten years of climate, so it cannot reproduce that
number on its own. The backend can: the corpus is already downloaded and cached
under `backend/ml/datasets/cache`, and the backend resolves each node to its
nearest station.

The failure mode this avoids is the interesting one. An on-node Hargreaves
approximation is a *different quantity on a different scale* from FAO-56 ET0,
and feeding one where the model learned the other degrades every prediction
quietly. When no station matches, `None` is returned and inference falls back to
the on-node estimate while labelling the decision as approximate, so the
approximation is visible in the UI rather than hidden in a feature value.
"""

from __future__ import annotations

import math
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.ml_bridge import CACHE_DIR, STATIONS, Station, compute_fao56_et0


class WeatherService:
    """
    Serves ET0 by location, from the on-disk NASA POWER cache.

    The cache is read once at first use and held in memory. It is a few hundred
    kilobytes, and re-reading it per telemetry payload would be wasteful; the
    file only changes when someone runs the fetch script.
    """

    #: Beyond this distance from any cached station, the local climate is not
    #: a fair proxy. Roughly 150 km, which is about the spatial resolution at
    #: which POWER's meteorological products are meaningfully local.
    MAX_STATION_DISTANCE_KM = 150.0

    def __init__(self, cache_dir: Path = CACHE_DIR, enable: bool = True) -> None:
        self.cache_dir = cache_dir
        self.enabled = enable
        self._frame: pd.DataFrame | None = None
        self._lock = threading.Lock()
        self._last_sync: str | None = None

    # -- cache ------------------------------------------------------------

    @property
    def frame(self) -> pd.DataFrame | None:
        if self._frame is None:
            with self._lock:
                if self._frame is None:
                    self._frame = self._load()
        return self._frame

    def _load(self) -> pd.DataFrame | None:
        if not self.cache_dir.exists():
            return None
        files = sorted(self.cache_dir.glob("power_*.csv"))
        if not files:
            return None
        frames = []
        for path in files:
            try:
                frame = pd.read_csv(path, parse_dates=["date"])
            except Exception:
                # One corrupt cache file must not take down the whole service.
                continue
            if "et0_mm_day" not in frame.columns or "latitude" not in frame.columns:
                # Cached by an earlier code version, before ET0 was computed.
                continue
            frames.append(frame)
        if not frames:
            return None
        combined = pd.concat(frames, ignore_index=True)
        self._last_sync = combined["date"].max().isoformat()
        return combined

    def reload(self) -> bool:
        with self._lock:
            self._frame = None
        return self.frame is not None

    @property
    def last_sync(self) -> str | None:
        return self._last_sync

    # -- lookup -----------------------------------------------------------

    def et0_for(self, device: dict[str, Any]) -> float | None:
        """
        ET0 in mm/day for a device, or None when no station is close enough.

        Falls back to nearest-neighbour matching when the node has no
        coordinates, using the station code it reports if it has one. A node
        that has never been geolocated gets None, which is the honest answer
        rather than a number from an arbitrary continent.
        """
        if not self.enabled:
            return None
        frame = self.frame
        if frame is None or frame.empty:
            return None

        code = device.get("stationCode")
        if code:
            subset = frame[frame["station"] == code]
            if not subset.empty:
                return self._latest_value(subset)

        latitude = device.get("latitude")
        longitude = device.get("longitude")
        if latitude is None or longitude is None:
            return None

        station = self.nearest_station(float(latitude), float(longitude))
        if station is None:
            return None
        subset = frame[frame["station"] == station.code]
        if subset.empty:
            return None
        return self._latest_value(subset)

    def nearest_station(self, latitude: float, longitude: float) -> Station | None:
        """Nearest cached station, or None if it is too far to be a proxy."""
        best: Station | None = None
        best_km = math.inf
        for station in STATIONS:
            distance = _haversine_km(latitude, longitude, station.latitude, station.longitude)
            if distance < best_km:
                best, best_km = station, distance
        if best is None or best_km > self.MAX_STATION_DISTANCE_KM:
            return None
        return best

    @staticmethod
    def _latest_value(subset: pd.DataFrame) -> float | None:
        valid = subset[subset["et0_mm_day"].notna()]
        if valid.empty:
            return None
        # A mean over the last week is steadier than a single day, and a single
        # cloud-free day can sit far from the week's demand.
        recent = valid.tail(7)["et0_mm_day"]
        return round(float(recent.mean()), 3)

    def forecast(self, device: dict[str, Any], days: int = 3) -> list[dict[str, Any]]:
        """Recent and near-term ET0, for the analytics view."""
        frame = self.frame
        if frame is None or frame["et0_mm_day"].dropna().empty:
            return []
        latitude = device.get("latitude")
        longitude = device.get("longitude")
        code = device.get("stationCode")
        if code:
            subset = frame[frame["station"] == code]
        elif latitude is not None and longitude is not None:
            station = self.nearest_station(float(latitude), float(longitude))
            subset = frame[frame["station"] == station.code] if station else frame.iloc[0:0]
        else:
            subset = frame.iloc[0:0]
        if subset.empty:
            return []

        daily = (
            subset.groupby(subset["date"].dt.date)["et0_mm_day"]
            .mean()
            .dropna()
            .sort_index()
        )
        if daily.empty:
            return []
        recent = daily.tail(days)
        return [
            {
                "date": day.isoformat(),
                "et0MmDay": round(float(value), 2),
                "source": "historical",
            }
            for day, value in recent.items()
        ]

    def describe(self) -> dict[str, Any]:
        frame = self.frame
        if frame is None or frame.empty:
            return {
                "available": False,
                "reason": "no cached NASA POWER corpus. Run "
                          "backend/ml/scripts/fetch_weather.py",
            }
        return {
            "available": True,
            "records": int(len(frame)),
            "stations": int(frame["station"].nunique()),
            "span": {
                "from": frame["date"].min().date().isoformat(),
                "to": frame["date"].max().date().isoformat(),
            },
            "et0MeanMmDay": round(float(frame["et0_mm_day"].mean()), 3),
            "et0MaxMmDay": round(float(frame["et0_mm_day"].max()), 3),
            "lastSync": self._last_sync,
            "maxStationDistanceKm": self.MAX_STATION_DISTANCE_KM,
        }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))
