"""
Tests for the live weather service.

The upstream is never contacted. The parsing is checked against a recorded
OpenWeather body, and the cache is checked by counting how many times a stubbed
fetch is called.

The endpoint test does not enter the application lifespan: that warms ten years
of weather and the V1 decision bundle, and this endpoint needs none of it. The
service is placed on `app.state` directly, and the previous one is restored
afterwards so the module cannot leak into another test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "backend"))

from app.main import app  # noqa: E402
from app.services.live_weather import (  # noqa: E402
    LiveWeatherService,
    Observation,
    observation_from_openmeteo,
    observation_from_payload,
)

#: A recorded OpenWeather current-conditions body, trimmed to the fields the
#: service reads. The shape is what the upstream actually sends: readings
#: nested under `main`, `wind` and `rain`, a description list under `weather`,
#: and an epoch timestamp under `dt`.
RECORDED = {
    "dt": 1767225600,
    "main": {"temp": 21.37, "humidity": 68},
    "wind": {"speed": 3.4},
    "rain": {"1h": 1.2},
    "weather": [{"description": "light rain"}],
}

#: A recorded Open-Meteo current-weather body, again as the upstream sends it:
#: everything flat under `current`, an ISO timestamp, precipitation for the
#: last hour, and wind in km/h rather than m/s.
RECORDED_METEO = {
    "latitude": 26.8467,
    "longitude": 80.9462,
    "current": {
        "time": "2026-10-24T09:45",
        "temperature_2m": 29.7,
        "relative_humidity_2m": 68,
        "precipitation": 0.9,
        "weather_code": 61,
        "wind_speed_10m": 12.2,
    },
}


def test_a_recorded_body_becomes_an_observation():
    observation = observation_from_payload("LKO", RECORDED)

    assert observation.station == "LKO"
    assert observation.temperatureC == pytest.approx(21.4)
    assert observation.humidityPct == pytest.approx(68)
    assert observation.windMs == pytest.approx(3.4)
    assert observation.rainfallMm == pytest.approx(1.2)
    assert observation.condition == "light rain"
    assert observation.observedAt.startswith("2026-")


def test_the_shorter_rain_window_wins():
    # The upstream reports the last hour and the last three hours. The hour is
    # the fresher figure, so it is the one a "right now" card should show.
    payload = {**RECORDED, "rain": {"1h": 1.2, "3h": 9.0}}
    assert observation_from_payload("LKO", payload).rainfallMm == pytest.approx(1.2)

    # With only the three-hour window there is still rain to report.
    payload = {**RECORDED, "rain": {"3h": 9.0}}
    assert observation_from_payload("LKO", payload).rainfallMm == pytest.approx(9.0)

    # And a dry hour under a wet three-hour window reads as dry, which is the
    # honest answer for "how much is falling now".
    payload = {**RECORDED, "rain": {"1h": 0.0, "3h": 9.0}}
    assert observation_from_payload("LKO", payload).rainfallMm == pytest.approx(0.0)


def test_a_body_missing_everything_still_parses():
    observation = observation_from_payload("LKO", {})

    assert observation.temperatureC == 0.0
    assert observation.humidityPct == 0.0
    assert observation.rainfallMm == 0.0
    assert observation.condition == ""


def test_the_open_meteo_body_becomes_an_observation():
    observation = observation_from_openmeteo("LKO", RECORDED_METEO)

    assert observation.station == "LKO"
    assert observation.temperatureC == pytest.approx(29.7)
    assert observation.humidityPct == pytest.approx(68)
    # km/h in, m/s out: 12.2 km/h is 3.4 m/s, the same card either way.
    assert observation.windMs == pytest.approx(3.4)
    assert observation.rainfallMm == pytest.approx(0.9)
    # Code 61 is the WMO's "light rain", and the card shows words, not numbers.
    assert observation.condition == "light rain"
    assert observation.observedAt.startswith("2026-10-24T09:45")
    # The card names its upstream, because the sky's source is part of the truth.
    assert observation.source == "open-meteo"


def test_an_unknown_open_meteo_code_reads_as_no_condition():
    payload = {
        "current": {**RECORDED_METEO["current"], "weather_code": 9999},
    }
    assert observation_from_openmeteo("LKO", payload).condition == ""


def test_without_a_key_there_is_no_observation():
    # With the keyless primary off and no fallback key, no upstream can be
    # asked, so the service reports itself off rather than reaching out.
    service = LiveWeatherService(api_key="", openmeteo_enabled=False)

    assert service.configured is False
    assert service.observation("LKO") is None


def test_even_without_a_key_open_meteo_is_the_primary():
    service = LiveWeatherService(api_key="")

    # The keyless primary is on, so the service is configured and will fetch.
    assert service.configured is True


def test_an_unknown_station_has_no_observation():
    service = LiveWeatherService(api_key="a-key")

    assert service.observation("XXX") is None


def test_the_observation_is_cached_so_the_quota_is_not_burned():
    service = LiveWeatherService(api_key="a-key", cache_seconds=600.0)
    calls: list[str] = []

    def fake_fetch(station):
        calls.append(station.code)
        return Observation(
            station=station.code,
            temperatureC=20.0,
            humidityPct=50.0,
            windMs=1.0,
            rainfallMm=0.0,
            condition="clear sky",
            observedAt="2026-01-01T00:00:00+00:00",
        )

    service._fetch = fake_fetch  # type: ignore[method-assign]

    first = service.observation("LKO")
    second = service.observation("LKO")

    assert first is not None and second is not None
    assert first.temperatureC == 20.0
    assert second is first
    # One call for two lookups: every open dashboard polls this endpoint, and
    # the free tier allows sixty calls a minute.
    assert calls == ["LKO"]


def test_a_failed_open_meteo_fetch_falls_back_to_the_keyed_upstream():
    """
    Priority with a parachute: Open-Meteo first, OpenWeather second.

    The stubs raise for the primary and answer for the fallback, so both
    upstream callers are exercised without a network. The cache is bypassed
    (fresh service, one call) so the dispatch itself is what is observed.
    """
    service = LiveWeatherService(api_key="a-key")
    meteo_calls: list[str] = []
    weather_calls: list[str] = []

    def failing_meteo_fetch(station):
        meteo_calls.append(station.code)
        raise ConnectionError("open-meteo unreachable")

    def working_weather_fetch(station):
        weather_calls.append(station.code)
        return Observation(
            station=station.code,
            temperatureC=21.4,
            humidityPct=68,
            windMs=3.4,
            rainfallMm=1.2,
            condition="light rain",
            observedAt="2026-01-01T00:00:00+00:00",
            source="openweather",
        )

    service._fetch_openmeteo = failing_meteo_fetch  # type: ignore[method-assign]
    service._fetch_openweather = working_weather_fetch  # type: ignore[method-assign]

    observation = service.observation("LKO")

    assert observation is not None
    assert observation.source == "openweather"
    # The primary was tried once, the fallback was tried once, nothing more.
    assert meteo_calls == ["LKO"]
    assert weather_calls == ["LKO"]


def test_without_a_key_the_fallback_is_never_tried():
    service = LiveWeatherService(api_key="")
    meteo_calls: list[str] = []
    weather_calls: list[str] = []

    def failing_meteo_fetch(station):
        meteo_calls.append(station.code)
        raise ConnectionError("open-meteo unreachable")

    def working_weather_fetch(station):
        weather_calls.append(station.code)
        return Observation(
            station=station.code,
            temperatureC=21.4,
            humidityPct=68,
            windMs=3.4,
            rainfallMm=1.2,
            condition="light rain",
            observedAt="2026-01-01T00:00:00+00:00",
        )

    service._fetch_openmeteo = failing_meteo_fetch  # type: ignore[method-assign]
    service._fetch_openweather = working_weather_fetch  # type: ignore[method-assign]

    # No key means no fallback: the primary's failure is the honest answer,
    # and the keyed upstream is not silently chosen to paper over it.
    assert service.observation("LKO") is None
    assert meteo_calls == ["LKO"]
    assert weather_calls == []


def test_both_upstreams_failing_reports_unavailable():
    service = LiveWeatherService(api_key="a-key")

    def always_raise(station):
        raise ConnectionError("upstream unreachable")

    service._fetch = always_raise  # type: ignore[method-assign]

    # A down sky is not a crashed dashboard: the card falls back to the plan.
    assert service.observation("LKO") is None


def test_a_station_the_engine_does_not_know_is_refused():
    service = LiveWeatherService(api_key="a-key")

    assert "LKO" in service.known_stations()
    assert "XXX" not in service.known_stations()


@pytest.fixture()
def client():
    """
    A client without the lifespan.

    The lifespan is what makes the API tests slow, and this endpoint does not
    need any of it. The live weather service is put on `app.state` directly and
    the previous one is put back afterwards.
    """
    previous = getattr(app.state, "live_weather", None)
    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        if previous is None:
            try:
                del app.state.live_weather
            except AttributeError:
                pass
        else:
            app.state.live_weather = previous
        test_client.close()


def test_the_endpoint_reports_itself_unavailable_with_no_upstream(client):
    # Keyless primary off and no fallback key: nothing can be asked, so the
    # card reports itself out before any network is touched.
    app.state.live_weather = LiveWeatherService(api_key="", openmeteo_enabled=False)

    body = client.get("/api/weather/live/LKO").json()

    assert body["available"] is False
    assert body["station"] == "LKO"
    assert body["reason"]


def test_the_endpoint_returns_the_observation_when_one_is_available(client):
    service = LiveWeatherService(api_key="a-key")
    service._fetch = lambda station: Observation(
        station="LKO",
        temperatureC=21.4,
        humidityPct=68,
        windMs=3.4,
        rainfallMm=1.2,
        condition="light rain",
        observedAt="2026-01-01T00:00:00+00:00",
        source="openweather",
    )  # type: ignore[method-assign]
    app.state.live_weather = service

    body = client.get("/api/weather/live/LKO").json()

    assert body["available"] is True
    assert body["source"] == "openweather"
    assert body["temperatureC"] == pytest.approx(21.4)
    assert body["humidityPct"] == pytest.approx(68)
    assert body["condition"] == "light rain"
    # The credential is never in the response.
    assert "a-key" not in str(body)
    assert "appid" not in str(body)


def test_the_endpoint_never_fails_for_an_unknown_station(client):
    service = LiveWeatherService(api_key="a-key")
    app.state.live_weather = service

    response = client.get("/api/weather/live/XXX")

    # Not a 500, and not a 404 the operator has to interpret: the card falls
    # back to the plan's own readings.
    assert response.status_code == 200
    assert response.json()["available"] is False
