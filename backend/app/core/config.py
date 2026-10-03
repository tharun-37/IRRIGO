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
        env_prefix="IRRIGO_V2_", env_file=".env", extra="ignore"
    )

    app_name: str = "IRRIGO V2"
    app_version: str = "2.0.0"

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
