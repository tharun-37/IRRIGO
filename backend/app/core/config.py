"""
Settings for the API, read from the environment with sensible local defaults.

Everything is overridable so the same code runs on a laptop and on a field
gateway without edits. Paths default to the repository layout rather than the
process working directory, because the two are not the same once the app is
started from `run.cmd`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository root. This file is `backend/app/core/config.py`, so the root is four
#: levels up, not three: core -> app -> backend -> repo. Getting this wrong points
#: `reports_dir` at `backend/reports` and every report silently 404s.
ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Absolute, because the process working directory is not the repository
        # root once the app is started from `run.cmd`. A `.env` that fails to
        # load looks exactly like a missing key, which is a slow thing to debug.
        env_prefix="IRRIGO_V2_",
        env_file=str(ROOT / ".env"),
        extra="ignore",
    )

    app_name: str = "IRRIGO V2"
    app_version: str = "2.0.0"

    #: Where the launcher writes runtime output, and where this app may keep
    #: its own derived, rebuildable artefacts (the advisor snapshot cache).
    #: Runtime noise, never project data: anything here can be deleted without
    #: changing what the app would answer.
    logs_dir: Path = ROOT / "logs"

    #: Origin allowed to call the API. The Vite dev server proxies `/api`, so in
    #: normal use the browser talks same-origin and this rarely matters.
    cors_origins: list[str] = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ]

    # --- live state, ported from the V1 controller -----------------------

    #: SQLite is the source of truth for telemetry once a node reports. The
    #: planner stays stateless; this is the running record on top of it.
    database_url: str = f"sqlite:///{(ROOT / 'backend' / 'data' / 'controller.db').as_posix()}"

    #: Shared secret the firmware must send on ingest. Empty disables the check,
    #: which is only ever acceptable on a bench, never on a field deployment.
    device_api_key: str = ""

    #: Readings further in the future than this are rejected as a clock fault.
    max_clock_skew_seconds: int = 300

    #: Cap on a single irrigation application, independent of the model.
    hard_max_depth_mm: float = 45.0

    #: Valve runtime ceiling, so a stuck command cannot flood a field.
    hard_max_runtime_seconds: int = 21_600

    #: Derive ET0 for an ingested reading from the engine's own climate corpus
    #: rather than trusting the node. The models were fitted on FAO-56 values.
    enable_weather_enrichment: bool = True

    #: Irrigation trigger as a fraction of the crop's readily available water.
    #: 1.0 is the FAO-56 stress point — the engine schedules only when the
    #: physics demands water. A lower fraction irrigates earlier than stress:
    #: the operator's risk preference, for a demo or a farm that acts before
    #: the crop feels it. Plan caches key on the plan inputs, so restart the
    #: backend after changing this for it to take effect.
    irrigation_trigger_fraction: float = 1.0

    # --- live weather, Open-Meteo primary / OpenWeather fallback ----------
    #: Open-Meteo needs no key and no account, so the live card works out of
    #: the box. An OpenWeather key, when present in `.env`, turns the keyed
    #: upstream into the fallback for the days Open-Meteo is unreachable.
    #: The key is a billable, per-account credential: read here, called from
    #: the API process only, never in source or the browser bundle.
    openweather_api_key: str = ""

    #: Base URL for Open-Meteo's current-weather endpoint. Overridable so a
    #: test can point at a stub instead of the real service.
    openmeteo_url: str = "https://api.open-meteo.com/v1/forecast"

    #: Base URL for OpenWeather's current-conditions endpoint. Overridable so
    #: a test can point at a stub instead of the real service.
    openweather_url: str = "https://api.openweathermap.org/data/2.5/weather"

    #: Seconds a fetched observation is reused before going back upstream.
    #: Every open dashboard polls this, so without a cache a handful of
    #: browsers would burn an upstream's goodwill and start returning 429s.
    #: Ten minutes is far finer than the weather itself changes. The cache
    #: is shared by both upstreams.
    openweather_cache_seconds: float = 600.0

    #: Where `.env` is looked for is fixed in `model_config` below, not here:
    #: `env_file` is a pydantic-settings key, so a field of the same name would
    #: shadow the setting that decides where the file is read from.


    # -- demonstration feed ------------------------------------------------
    #: V2 has no field nodes, so a simulator replays each registered sowing's
    #: real season through the ingest pipeline. This makes the whole live stack
    #: (ingest -> decision -> alerts -> analytics -> websocket) exercisable with
    #: no hardware. Set `IRRIGO_V2_DEMO_SIMULATOR=false` to run headless.
    demo_simulator: bool = True
    #: Wall-clock seconds between simulated readings per field.
    demo_tick_seconds: float = 15.0
    #: How much of the recent past the startup back-fill spans.
    demo_history_days: int = 7

    # --- data locations, absolute by default -----------------------------

    src_path: Path = ROOT / "src"
    sowings_path: Path = ROOT / "data" / "sowings.json"
    reports_dir: Path = ROOT / "reports"
    models_dir: Path = ROOT / "models"

    #: Years of POWER weather to load. Ten years is enough for the monthly
    #: climatology the forward forecast uses and keeps startup under a second.
    weather_start_year: int = 2015
    weather_end_year: int = 2024

    #: Controlled storage for crop-health uploads. Deliberately inside the
    #: backend's data directory, not any web-servable location, and addressed
    #: by content-addressed keys rather than client-controlled names.
    crop_health_images_dir: Path = (
        ROOT / "backend" / "data" / "crop_health" / "images"
    )

    #: Planning horizon in days. Matches `PLAN_HORIZON_DAYS` in the engine; the
    #: dashboard shows the same number the model was fitted on, so it is stated
    #: once here rather than hard-coded in the frontend.
    horizon_days: int = 14

    def summary(self) -> dict[str, object]:
        return {
            "app": self.app_name,
            "version": self.app_version,
            "sowings": str(self.sowings_path),
            "reports": str(self.reports_dir),
        }

    @property
    def backend_dir(self) -> Path:
        return ROOT / "backend"

    @property
    def data_dir(self) -> Path:
        return self.backend_dir / "data"


@lru_cache
def get_settings() -> Settings:
    return Settings()
