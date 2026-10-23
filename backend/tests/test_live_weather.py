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


def test_without_a_key_there_is_no_observation():
    service = LiveWeatherService(api_key="")

    assert service.configured is False
    assert service.observation("LKO") is None


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


def test_the_endpoint_reports_itself_unavailable_without_a_key(client):
    app.state.live_weather = LiveWeatherService(api_key="")

    body = client.get("/api/weather/live/LKO").json()

    assert body["available"] is False
    assert body["station"] == "LKO"
    assert body["reason"]


def test_the_endpoint_returns_the_observation_when_one_is_available(client):
    observation = Observation(
        station="LKO",
        temperatureC=21.4,
        humidityPct=68,
        windMs=3.4,
        rainfallMm=1.2,
        condition="light rain",
        observedAt="2026-01-01T00:00:00+00:00",
    )
    service = LiveWeatherService(api_key="a-key")
    service._fetch = lambda station: observation  # type: ignore[method-assign]
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
