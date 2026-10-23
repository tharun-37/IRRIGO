"""
Live current conditions for a field's station, from OpenWeather.

The advisory carries the crop's own sensor readings. In a field deployment those
are the node's numbers; in the demo they are a replay of the season's own
meteorology from the NASA POWER corpus. Neither is the sky *right now*, and the
weather card is titled "· now", so this service fetches the actual current
conditions for the station the field sits at.

Two properties matter more than the feature itself:

* **The key never leaves the API process.** It is read from the environment
  here and used as a query parameter against OpenWeather. The payload the
  browser receives carries the observation and nothing else, so the credential
  is never in the bundle, the network tab or the repository.
* **The observation is cached per station.** Every open dashboard polls this
  endpoint, and the free tier allows sixty calls a minute. Without a cache a
  handful of browsers would exhaust the quota and the card would go empty, so
  a fetched observation is reused for ten minutes.

With no key configured, an unknown station, or an upstream that is
unreachable, the service reports itself unavailable. The dashboard then falls
back to the advisory's own readings rather than showing an error, because a
weather card with the season's own numbers is better than no weather card.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from ..core.ml_bridge import STATIONS

logger = logging.getLogger("iic.live_weather")

#: The engine's station table, by code. A sowing records the station code; the
#: coordinates are what OpenWeather is asked for.
_BY_CODE = {station.code: station for station in STATIONS}


@dataclass(frozen=True)
class Observation:
    """One current-conditions reading, in the shape the dashboard renders."""

    station: str
    temperatureC: float
    humidityPct: float
    windMs: float
    rainfallMm: float
    condition: str
    observedAt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "station": self.station,
            "temperatureC": self.temperatureC,
            "humidityPct": self.humidityPct,
            "windMs": self.windMs,
            "rainfallMm": self.rainfallMm,
            "condition": self.condition,
            "observedAt": self.observedAt,
        }


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def observation_from_payload(code: str, payload: dict[str, Any]) -> Observation:
    """
    Turn an OpenWeather current-conditions body into an observation.

    Split out from the fetch so it can be tested against a recorded body
    without a network call. The upstream nests its readings under `main`,
    `wind`, `rain` and `weather`, and reports rain over the last hour or the
    last three hours, so both windows are read and the shorter one wins.
    """
    main = payload.get("main") or {}
    wind = payload.get("wind") or {}
    rain = payload.get("rain") or {}

    if "1h" in rain:
        rainfall = _as_float(rain.get("1h"))
    elif "3h" in rain:
        rainfall = _as_float(rain.get("3h"))
    else:
        rainfall = 0.0

    weather = payload.get("weather") or []
    condition = ""
    if isinstance(weather, list) and weather:
        first = weather[0]
        if isinstance(first, dict):
            condition = str(first.get("description") or "").strip()

    observed = payload.get("dt")
    observed_at = ""
    try:
        if observed is not None:
            observed_at = datetime.fromtimestamp(
                int(observed), tz=timezone.utc
            ).isoformat(timespec="seconds")
    except (TypeError, ValueError, OSError):
        observed_at = ""

    return Observation(
        station=code,
        temperatureC=round(_as_float(main.get("temp")), 1),
        humidityPct=round(_as_float(main.get("humidity")), 0),
        windMs=round(_as_float(wind.get("speed")), 1),
        rainfallMm=round(rainfall, 2),
        condition=condition,
        observedAt=observed_at,
    )


class LiveWeatherService:
    """
    Fetches and caches current conditions per station.

    The cache is in memory and per process. That is enough for the one thing
    this has to survive: an open dashboard polling on a timer.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openweathermap.org/data/2.5/weather",
        cache_seconds: float = 600.0,
        timeout_seconds: float = 6.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url
        self.cache_seconds = cache_seconds
        self.timeout_seconds = timeout_seconds
        self._cache: dict[str, tuple[float, Observation]] = {}
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        """Whether a key is present. Without one the feature is simply off."""
        return bool(self.api_key)

    def known_stations(self) -> list[str]:
        return sorted(_BY_CODE)

    def observation(self, station_code: str) -> Observation | None:
        """
        The current observation for a station, or None when there is none.

        None covers every way this can fail to have an answer: no key, a station
        the engine does not know, and an upstream that is down. The caller shows
        the advisory's own readings in all three cases.
        """
        if not self.configured:
            return None

        station = _BY_CODE.get(str(station_code or "").strip().upper())
        if station is None:
            return None

        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(station.code)
            if cached is not None and cached[0] > now:
                return cached[1]

        try:
            observation = self._fetch(station)
        except Exception as error:  # pragma: no cover - network failure
            # Upstream trouble is expected and must not take the dashboard with
            # it. The card falls back to the advisory's own readings.
            logger.warning("live weather for %s unavailable: %s", station.code, error)
            return None

        with self._lock:
            self._cache[station.code] = (now + self.cache_seconds, observation)
        return observation

    def _fetch(self, station: Any) -> Observation:
        params = {
            "lat": station.latitude,
            "lon": station.longitude,
            "appid": self.api_key,
            "units": "metric",
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(self.base_url, params=params)
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("unexpected payload from the weather service")
        return observation_from_payload(station.code, payload)
