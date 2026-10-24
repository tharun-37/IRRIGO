"""
Live current conditions for a field's station.

The advisory carries the crop's own sensor readings. In a field deployment those
are the node's numbers; in the demo they are a replay of the season's own
meteorology from the NASA POWER corpus. Neither is the sky *right now*, and the
weather card is titled "· now", so this service fetches the actual current
conditions for the station the field sits at.

Two upstreams are used, in a fixed priority order:

* **Open-Meteo is primary.** It needs no key, has no call quota worth fearing
  at dashboard scale, and is the same reanalysis family (ERA5) the corpus was
  externally validated against (`r = 0.911` on ET0), so the live card and the
  training weather speak the same climate language.
* **OpenWeather is the fallback.** With an API key configured, a failed
  Open-Meteo fetch retries the keyed upstream before giving up. No key means
  no fallback, and an Open-Meteo outage simply reports the card unavailable.

Two properties matter more than the feature itself:

* **No credential reaches the client.** The OpenWeather key, when present, is
  read from the environment here and used as a query parameter from the API
  process only. The payload the browser receives carries the observation and
  nothing else.
* **The observation is cached per station.** Every open dashboard polls this
  endpoint, so a fetched observation is reused for ten minutes regardless of
  which upstream supplied it.

When neither upstream can answer, the service reports itself unavailable. The
dashboard then falls back to the advisory's own readings rather than showing
an error, because a weather card with the season's own numbers is better than
no weather card.
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
    #: The upstream the observation came from. The endpoint passes it through,
    #: because a farmer-facing "which sky is this" card is only honest when the
    #: source of the answer is named.
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "station": self.station,
            "temperatureC": self.temperatureC,
            "humidityPct": self.humidityPct,
            "windMs": self.windMs,
            "rainfallMm": self.rainfallMm,
            "condition": self.condition,
            "observedAt": self.observedAt,
            "source": self.source,
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
        source="openweather",
    )


#: World Meteorological Organisation weather codes, the numeric condition
#: encoding Open-Meteo uses. The common Indian-agriculture relevant set is
#: spelled out; anything absent maps to an empty string, which the dashboard
#: already renders as no description rather than an error.
_WMO_CONDITIONS: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "light showers",
    81: "moderate showers",
    82: "violent showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


def _wmo_condition(code: Any) -> str:
    """The readable condition for a WMO weather code, or '' for an unknown one."""
    try:
        return _WMO_CONDITIONS.get(int(code), "")
    except (TypeError, ValueError):
        return ""


def observation_from_openmeteo(code: str, payload: dict[str, Any]) -> Observation:
    """
    Turn an Open-Meteo current-weather body into an observation.

    Split out like the OpenWeather equivalent so it can be tested against a
    recorded body with no network call. The upstream keeps the whole current
    snapshot flat under `current`: a UTC-local ISO timestamp, float fields
    named for what they measure, and a precipitation figure accumulated over
    the last hour. The wind speed arrives in km/h and is converted, because
    the observation contract (`windMs`) has one unit and two upstreams now
    feed it.
    """
    current = payload.get("current") or {}
    if not isinstance(current, dict):
        current = {}

    observed_at = ""
    raw_time = current.get("time")
    try:
        if raw_time is not None:
            moment = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
            if moment.tzinfo is None:  # Open-Meteo returns naive UTC by default
                moment = moment.replace(tzinfo=timezone.utc)
            observed_at = moment.isoformat(timespec="seconds")

    except ValueError:
        observed_at = ""

    return Observation(
        station=code,
        temperatureC=round(_as_float(current.get("temperature_2m")), 1),
        humidityPct=round(_as_float(current.get("relative_humidity_2m")), 0),
        windMs=round(_as_float(current.get("wind_speed_10m")) / 3.6, 1),
        rainfallMm=round(_as_float(current.get("precipitation")), 2),
        condition=_wmo_condition(current.get("weather_code")),
        observedAt=observed_at,
        source="open-meteo",
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
        openmeteo_url: str = "https://api.open-meteo.com/v1/forecast",
        openmeteo_enabled: bool = True,
        cache_seconds: float = 600.0,
        timeout_seconds: float = 6.0,
    ) -> None:
        self.api_key = api_key.strip()
        self.base_url = base_url
        self.openmeteo_url = openmeteo_url
        self.openmeteo_enabled = bool(openmeteo_enabled)
        self.cache_seconds = cache_seconds
        self.timeout_seconds = timeout_seconds
        self._cache: dict[str, tuple[float, Observation]] = {}
        self._lock = threading.Lock()

    @property
    def configured(self) -> bool:
        """
        Whether the service can answer at all.

        With Open-Meteo on it always can - the primary upstream needs neither
        a key nor an account. It streams false only when Open-Meteo is
        disabled *and* no OpenWeather key is present, which is the state the
        offline tests and a deliberately keyless deployment share.
        """
        return self.openmeteo_enabled or bool(self.api_key)

    def known_stations(self) -> list[str]:
        return sorted(_BY_CODE)

    def observation(self, station_code: str) -> Observation | None:
        """
        The current observation for a station, or None when there is none.

        None covers every way this can fail to have an answer: no upstream
        configured, a station the engine does not know, and an upstream that
        is down even after the fallback was tried. The caller shows the
        advisory's own readings in all three cases.
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
        """
        One observation, with the fallback chain the card deserves.

        Open-Meteo is tried first. If it fails and an OpenWeather key exists,
        the keyed upstream gets a turn; only when both refuse does the error
        climb, and `observation` turns it into the graceful unavailable state.
        Without a key there is no second try - the primary's error is the
        honest answer.
        """
        try:
            return self._fetch_openmeteo(station)
        except Exception as openmeteo_error:  # noqa: BLE001 - network failure
            if not self.api_key:
                raise
            try:
                return self._fetch_openweather(station)
            except Exception:  # noqa: BLE001 - the primary's failure is reported
                raise openmeteo_error from None

    def _fetch_openmeteo(self, station: Any) -> Observation:
        """The current-conditions block for the station, from Open-Meteo."""
        params = {
            "latitude": station.latitude,
            "longitude": station.longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,"
            "weather_code,wind_speed_10m",
            "timezone": "UTC",
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.get(self.openmeteo_url, params=params)
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("unexpected payload from the weather service")
        return observation_from_openmeteo(station.code, payload)

    def _fetch_openweather(self, station: Any) -> Observation:
        """The keyed fallback: OpenWeather's current-conditions endpoint."""
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
