"""
SQLite access for the live layer, using the standard library only.

The planner is stateless: a plan is a pure function of a sowing and the weather.
The live layer is the opposite — it is the running record of what nodes reported
and what the advisor decided. It lives in SQLite so history, alerts and the
audit trail survive a restart.

No ORM. The schema is explicit and a reviewer can read the SQL. That matters
most for the one property worth checking by eye: a reading, its decision and its
event row are written in the right order, inside one transaction.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import get_settings

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS devices (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    zone              TEXT NOT NULL,
    crop              TEXT NOT NULL DEFAULT 'Maize',
    soil_type         TEXT NOT NULL DEFAULT 'Loam',
    field_area_m2     REAL NOT NULL DEFAULT 1000,
    latitude          REAL,
    longitude         REAL,
    station_code      TEXT,
    sowing_date       TEXT,
    firmware          TEXT,
    link_state        TEXT NOT NULL DEFAULT 'online',
    season_progress   REAL NOT NULL DEFAULT 0.5,
    pump_rate_lpm     REAL NOT NULL DEFAULT 12.0,
    max_depth_mm      REAL NOT NULL DEFAULT 40.0,
    valve_open        INTEGER NOT NULL DEFAULT 0,
    valve_opened_at   TEXT,
    valve_runtime_s   INTEGER NOT NULL DEFAULT 0,
    last_seen         TEXT,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS readings (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id         TEXT NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
    zone              TEXT NOT NULL,
    recorded_at       TEXT NOT NULL,
    received_at       TEXT NOT NULL,
    temperature       REAL NOT NULL,
    humidity          REAL NOT NULL,
    soil_moisture     REAL NOT NULL,
    soil_ph           REAL NOT NULL,
    ec                REAL NOT NULL,
    nitrogen          REAL NOT NULL,
    phosphorus        REAL NOT NULL,
    potassium         REAL NOT NULL,
    light_intensity   REAL NOT NULL,
    rainfall          REAL NOT NULL DEFAULT 0,
    wind_speed        REAL NOT NULL DEFAULT 1.0,
    battery_volts     REAL,
    rssi_dbm          INTEGER,
    et0_mm_day        REAL,
    crop_age_days     REAL,
    stage             TEXT,
    gdd               REAL,
    source            TEXT NOT NULL DEFAULT 'http'
);

CREATE INDEX IF NOT EXISTS idx_readings_device_time
    ON readings(device_id, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_readings_time
    ON readings(recorded_at DESC);

CREATE TABLE IF NOT EXISTS decisions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id         TEXT NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
    zone              TEXT NOT NULL,
    decided_at        TEXT NOT NULL,
    reading_id        INTEGER REFERENCES readings(id) ON DELETE SET NULL,
    should_irrigate   INTEGER NOT NULL,
    depth_mm          REAL NOT NULL,
    volume_litres     REAL NOT NULL,
    duration_seconds  INTEGER NOT NULL,
    action            TEXT NOT NULL,
    disease_risk      TEXT NOT NULL,
    disease_confidence REAL NOT NULL,
    disease_probabilities TEXT NOT NULL,
    yield_t_per_ha    REAL NOT NULL,
    nutrient_demand   TEXT NOT NULL,
    et0_mm_day        REAL NOT NULL,
    kc                REAL NOT NULL,
    depletion_fraction REAL NOT NULL,
    crop_stress_index REAL NOT NULL,
    moisture_band     TEXT NOT NULL,
    risk_level        TEXT NOT NULL,
    confidence        REAL NOT NULL,
    explanation       TEXT NOT NULL,
    crop_age_days     REAL NOT NULL DEFAULT 0,
    stage             TEXT NOT NULL DEFAULT 'unknown',
    gdd               REAL NOT NULL DEFAULT 0,
    stage_progress    REAL NOT NULL DEFAULT 0,
    raw               TEXT NOT NULL,
    inference_ms      REAL NOT NULL,
    manual            INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_decisions_device_time
    ON decisions(device_id, decided_at DESC);

CREATE TABLE IF NOT EXISTS alerts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id         TEXT NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
    zone              TEXT NOT NULL,
    raised_at         TEXT NOT NULL,
    kind              TEXT NOT NULL,
    severity          TEXT NOT NULL,
    message           TEXT NOT NULL,
    acknowledged      INTEGER NOT NULL DEFAULT 0,
    resolved_at       TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerts_open
    ON alerts(acknowledged, raised_at DESC);

CREATE TABLE IF NOT EXISTS events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id         TEXT,
    zone              TEXT,
    kind              TEXT NOT NULL,
    detail            TEXT NOT NULL,
    source            TEXT NOT NULL,
    created_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_time
    ON events(created_at DESC);
"""


class Database:
    """
    A small connection-per-thread pool over SQLite.

    SQLite connections are not safe to share across threads, and FastAPI runs
    sync endpoints on a worker pool. One connection per thread with WAL gives
    concurrent reads against a single writer, which is exactly this workload.
    """

    def __init__(self, url: str) -> None:
        self._path = _sqlite_path(url)
        self._local = threading.local()
        self._write_lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self._path, timeout=15.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 15000")
            self._local.connection = connection
        return connection

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        yield self._connect().cursor()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """
        A write transaction, serialised across threads.

        SQLite allows one writer at a time. The Python lock avoids the
        intermittent `database is locked` the busy timeout alone turns into a
        slow failure under a continuously reporting fleet.
        """
        connection = self._connect()
        with self._write_lock:
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection.cursor()
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def initialise(self) -> None:
        connection = self._connect()
        connection.executescript(SCHEMA)
        self._migrate(connection)
        connection.commit()

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        """
        Add columns introduced after a database was first created.

        `CREATE TABLE IF NOT EXISTS` is silent about a table that already exists
        with an older shape, and an existing install must not need manual SQL.
        """
        wanted = {
            "devices": {"sowing_date": "TEXT"},
            "readings": {
                "crop_age_days": "REAL",
                "stage": "TEXT",
                "gdd": "REAL",
            },
            "decisions": {
                "crop_age_days": "REAL NOT NULL DEFAULT 0",
                "stage": "TEXT NOT NULL DEFAULT 'unknown'",
                "gdd": "REAL NOT NULL DEFAULT 0",
                "stage_progress": "REAL NOT NULL DEFAULT 0",
            },
        }
        for table, columns in wanted.items():
            present = {
                row["name"]
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for name, declaration in columns.items():
                if name not in present:
                    connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
                    )

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None


def _sqlite_path(url: str) -> Path:
    if url.startswith("sqlite:///"):
        return Path(url.removeprefix("sqlite:///"))
    raise ValueError(f"only sqlite URLs are supported in this project, got {url!r}")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    """RFC 3339 with a Z suffix, which is what the firmware and the UI expect."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_iso(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), default=str)


def json_loads(value: str, default: Any = None) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


_database: Database | None = None


def get_database() -> Database:
    global _database
    if _database is None:
        _database = Database(get_settings().database_url)
    return _database


__all__ = [
    "Database",
    "get_database",
    "utcnow",
    "iso",
    "parse_iso",
    "json_dumps",
    "json_loads",
    "timedelta",
]
