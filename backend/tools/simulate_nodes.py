"""
Field node simulator.

Stands in for a real ESP32 so the dashboard, the alert rules, the WebSocket feed
and the analytics endpoints can all be exercised without hardware on the desk.

It is not a random number generator. Each simulated zone integrates the same
FAO-56 soil water balance used to build the training corpus, driven by real
meteorology from the cached NASA POWER corpus. That matters, because a
simulator producing i.i.d. readings would never exercise the parts most likely
to be wrong: the hysteresis around the irrigation trigger, the way a zone dries
over consecutive days, and the alert rules that depend on trend.

    python backend/tools/simulate_nodes.py --zones 4 --interval 5 --minutes 2880

The `--minutes` option back-fills history so the analytics charts are populated
on first run, rather than starting from an empty database.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

ML_DIR = Path(__file__).resolve().parents[1] / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from dataset_builder import CROP_PROFILES, SOILS, crop_coefficient  # noqa: E402
from real_data import CACHE_DIR, STATIONS, compute_fao56_et0  # noqa: E402

#: Zone templates, each pinned to a real station so ET0 is genuine rather than
#: invented. Four distinct climates, so the dashboard shows meaningfully
#: different irrigation behaviour per zone.
ZONE_TEMPLATES: list[dict[str, Any]] = [
    {"zone": "North Field", "station": "LUD", "crop": "Wheat", "soil": "Loam",
     "field_area_m2": 4_200, "season_progress": 0.62, "moisture": 0.58},
    {"zone": "South Field", "station": "HYD", "crop": "Maize", "soil": "Sandy_Loam",
     "field_area_m2": 2_800, "season_progress": 0.48, "moisture": 0.34},
    {"zone": "East Terrace", "station": "JAI", "crop": "Cotton", "soil": "Sand",
     "field_area_m2": 6_500, "season_progress": 0.55, "moisture": 0.24},
    {"zone": "West Plot", "station": "KOL", "crop": "Rice", "soil": "Clay",
     "field_area_m2": 1_500, "season_progress": 0.70, "moisture": 0.86},
    {"zone": "Block C", "station": "NAG", "crop": "Soybean", "soil": "Loam",
     "field_area_m2": 3_100, "season_progress": 0.40, "moisture": 0.46},
    {"zone": "Block D", "station": "BBN", "crop": "Tomato", "soil": "Clay_Loam",
     "field_area_m2": 900, "season_progress": 0.66, "moisture": 0.52},
]


class ZoneSimulator:
    """
    One zone, integrating the FAO-56 water balance over real daily weather.

    The probe measurement error is deliberately included. A simulator without
    noise would make the temporal-smoothing features look useless, when in the
    field they are one of the few things protecting the controller from a single
    bad reading.
    """

    #: Standard deviation of the probe's volumetric water content error, percent.
    MOISTURE_NOISE = 1.1

    def __init__(
        self,
        template: dict[str, Any],
        weather: dict[str, Any],
        rng: random.Random,
    ) -> None:
        self.template = template
        self.weather = weather
        self.rng = rng

        self.device_id = f"node-{template['zone'].lower().replace(' ', '-')}"
        self.zone = template["zone"]
        self.crop = CROP_PROFILES[template["crop"]]
        self.soil = SOILS[template["soil"]]
        self.field_area_m2 = float(template["field_area_m2"])
        self.season_progress = float(template["season_progress"])

        # Starting soil moisture, as a fraction of plant-available water, so the
        # zones begin at genuinely different points in their cycle.
        available = self.soil.field_capacity_pct - self.soil.wilting_point_pct
        self.moisture = self.soil.wilting_point_pct + float(template["moisture"]) * available

        # Soil chemistry drifts slowly around its commissioning value.
        self.soil_ph = float(np.clip(rng.gauss(6.7, 0.45), 4.8, 8.4))
        self.nitrogen = float(np.clip(rng.gauss(55.0, 22.0), 10.0, 160.0))
        self.phosphorus = float(np.clip(rng.gauss(38.0, 15.0), 8.0, 120.0))
        self.potassium = float(np.clip(rng.gauss(48.0, 18.0), 12.0, 140.0))
        self.ec = float(np.clip(0.5 + (self.nitrogen / 90.0 + self.potassium / 80.0) * 0.9, 0.1, 8.0))

        # 3.7 V lithium cell, sagging under load.
        self.battery = float(np.clip(rng.gauss(3.92, 0.05), 3.30, 4.15))
        self.rssi = int(np.clip(rng.gauss(-62, 8), -92, -38))

    # -- physics ----------------------------------------------------------

    def step(self) -> None:
        """Advance one day, applying rainfall, transpiration and any irrigation."""
        soil, crop = self.soil, self.crop
        kc = crop_coefficient(crop, self.season_progress)
        et0 = float(self.weather.get("et0", 4.5))
        etc = kc * et0

        rainfall = float(self.weather.get("rainfall", 0.0))
        infiltration = min(rainfall * 0.75, soil.infiltration_mm_day)
        root_depth_mm = self._root_depth_mm()
        available = (soil.field_capacity_pct - soil.wilting_point_pct) / 100.0 * root_depth_mm

        self.moisture += infiltration / root_depth_mm * 100.0
        if self.moisture > soil.field_capacity_pct:
            excess = (self.moisture - soil.field_capacity_pct) / 100.0 * root_depth_mm
            self.moisture -= min(excess, soil.deep_percolation_mm_day) / root_depth_mm * 100.0

        self.moisture -= etc / root_depth_mm * 100.0

        # The valve opens when depletion passes the management-allowed point,
        # exactly as the training simulation does.
        if self.moisture < self._refill_threshold(available):
            self.moisture = soil.field_capacity_pct * rng.uniform(0.94, 1.0)
            self.battery = max(3.30, self.battery - self.rng.uniform(0.002, 0.012))

        self.moisture = float(
            np.clip(self.moisture, soil.wilting_point_pct * 0.75, soil.field_capacity_pct)
        )

        # Nutrients decline through the season and are partly replenished.
        drawdown = 0.006 + 0.004 * (etc / 8.0)
        self.nitrogen = float(np.clip(self.nitrogen * (1.0 - drawdown) + self.rng.gauss(0, 0.6), 5.0, 240.0))
        self.phosphorus = float(np.clip(self.phosphorus * (1.0 - drawdown * 0.6) + self.rng.gauss(0, 0.4), 4.0, 180.0))
        self.potassium = float(np.clip(self.potassium * (1.0 - drawdown * 0.8) + self.rng.gauss(0, 0.5), 6.0, 220.0))
        self.ec = float(
            np.clip(0.4 + (self.nitrogen / 90.0 + self.potassium / 80.0) * 0.9 + self.rng.gauss(0, 0.05), 0.05, 12.0)
        )
        # pH is near-stable, nudged by the acidifying effect of nitrogen.
        self.soil_ph = float(np.clip(self.soil_ph - 0.0012 + self.rng.gauss(0, 0.008), 4.2, 8.6))
        self.rssi = int(np.clip(self.rssi + self.rng.randint(-2, 2), -92, -38))

    def _root_depth_mm(self) -> float:
        from dataset_builder import rooting_depth

        return rooting_depth(self.crop, self.season_progress) * 10.0

    def _refill_threshold(self, available_mm: float) -> float:
        """Moisture below which the valve opens, from the depletion fraction."""
        raw = available_mm * self.soil.available_fraction
        return self.soil.field_capacity_pct - (raw / self._root_depth_mm() * 100.0)

    # -- payload ----------------------------------------------------------

    def reading(self, recorded_at: datetime) -> dict[str, Any]:
        """One telemetry payload, with probe noise applied to the moisture."""
        station = STATIONS[0]
        for candidate in STATIONS:
            if candidate.code == self.template["station"]:
                station = candidate
                break

        observed = float(
            np.clip(self.moisture + self.rng.gauss(0, self.MOISTURE_NOISE), 0.5, 100.0)
        )
        return {
            "device_id": self.device_id,
            "zone": self.zone,
            "recorded_at": recorded_at.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "temperature": round(float(self.weather.get("t2m", 27.0) + self.rng.gauss(0, 0.8)), 2),
            "humidity": round(float(self.rng.gauss(self.weather.get("rh2m", 62.0), 2.5)), 2),
            "soil_moisture": round(observed, 2),
            "soil_ph": round(self.soil_ph, 2),
            "ec": round(self.ec, 3),
            "nitrogen": round(self.nitrogen, 2),
            "phosphorus": round(self.phosphorus, 2),
            "potassium": round(self.potassium, 2),
            "light_intensity": round(float(max(0.0, self.weather.get("sw", 22.0) * 1e6 * 0.0034 * self.rng.uniform(0.8, 1.2))), 1),
            "rainfall": round(float(self.weather.get("rainfall", 0.0)), 2),
            "wind_speed": round(float(max(0.2, self.weather.get("ws2m", 2.0) + self.rng.gauss(0, 0.3))), 2),
            "battery_volts": round(self.battery, 3),
            "rssi_dbm": self.rssi,
            "crop": self.crop.name,
            "soil_type": self.soil.name,
            "field_area_m2": self.field_area_m2,
            "latitude": station.latitude,
            "longitude": station.longitude,
            "station_code": station.code,
            "firmware": "1.0.0-sim",
            "source": "simulator",
        }


def load_weather_for_station(station_code: str, day_offset: int) -> dict[str, Any]:
    """Real meteorology for a station on a given day offset from today."""
    import pandas as pd

    files = sorted(CACHE_DIR.glob(f"power_{station_code}_*.csv"))
    if not files:
        return {"et0": 4.5, "t2m": 27.0, "rh2m": 62.0, "sw": 22.0, "rainfall": 0.0, "ws2m": 2.0}
    frame = pd.read_csv(files[0], parse_dates=["date"])
    if "et0_mm_day" not in frame.columns:
        return {"et0": 4.5, "t2m": 27.0, "rh2m": 62.0, "sw": 22.0, "rainfall": 0.0, "ws2m": 2.0}

    end = frame["date"].max()
    target = end - timedelta(days=abs(day_offset))
    window = frame[frame["date"] <= target].tail(1)
    if window.empty:
        return {"et0": 4.5, "t2m": 27.0, "rh2m": 62.0, "sw": 22.0, "rainfall": 0.0, "ws2m": 2.0}
    row = window.iloc[0]
    return {
        "et0": float(row.get("et0_mm_day", 4.5) or 4.5),
        "t2m": float(row.get("T2M", 27.0) or 27.0),
        "rh2m": float(row.get("RH2M", 62.0) or 62.0),
        "sw": float(row.get("ALLSKY_SFC_SW_DWN", 22.0) or 22.0),
        "rainfall": float(row.get("PRECTOTCORR", 0.0) or 0.0),
        "ws2m": float(row.get("WS2M", 2.0) or 2.0),
    }


def post(url: str, payload: dict[str, Any], api_key: str = "") -> int:
    """POST one reading, returning the HTTP status."""
    data = payload
    request = urllib.request.Request(
        url,
        data=__import__("json").dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json", **({"X-Device-Key": api_key} if api_key else {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code
    except urllib.error.URLError:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate ESP32 field nodes.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/telemetry")
    parser.add_argument("--zones", type=int, default=4, help="how many zones to simulate")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between cycles")
    parser.add_argument("--minutes", type=int, default=0,
                        help="back-fill this many minutes of history before going live")
    parser.add_argument("--step", type=int, default=15,
                        help="minutes between back-filled readings")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--seed", type=int, default=20240917)
    parser.add_argument("--once", action="store_true",
                        help="back-fill and exit, instead of streaming live")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    templates = ZONE_TEMPLATES[: max(1, min(args.zones, len(ZONE_TEMPLATES)))]

    print("Intelligent Irrigation Controller - node simulator")
    print(f"  target      {args.url}")
    print(f"  zones       {len(templates)}")
    for template in templates:
        print(f"    {template['zone']:<14} {template['crop']:<8} "
              f"{template['soil']:<11} station {template['station']}  "
              f"{template['field_area_m2']:>6,.0f} m2")
    if args.minutes:
        print(f"  back-fill   {args.minutes} minutes at {args.step}-minute steps")
    print()

    now = datetime.now(timezone.utc)

    # -- back-fill history ----------------------------------------------
    if args.minutes:
        cycles = max(1, args.minutes // args.step)
        simulators = [
            ZoneSimulator(template, load_weather_for_station(template["station"], 40 - i), rng)
            for i, template in enumerate(templates)
        ]
        sent = failed = 0
        print(f"back-filling {cycles} cycles ...", flush=True)
        for step in range(cycles):
            moment = now - timedelta(minutes=args.minutes - step * args.step)
            for index, simulator in enumerate(simulators):
                day_offset = cycles - step
                simulator.weather = load_weather_for_station(
                    simulator.template["station"], day_offset // max(1, 24 * 60 // args.step)
                )
                status = post(args.url, simulator.reading(moment), args.api_key)
                if status in (200, 201):
                    sent += 1
                else:
                    failed += 1
            if (step + 1) % 24 == 0:
                print(f"  {step + 1}/{cycles} cycles, {sent} accepted, {failed} rejected", flush=True)
            time.sleep(0.01)
        print(f"back-fill complete: {sent} accepted, {failed} rejected\n")

    # -- live loop ------------------------------------------------------
    if args.once:
        # Back-fill only. Used to populate the dashboard for a demo or a test
        # without leaving a process streaming in the background.
        print("back-fill only, not entering the live loop\n", flush=True)
        return 0

    print("streaming live telemetry. Ctrl-C to stop.\n", flush=True)
    simulators = [
        ZoneSimulator(template, load_weather_for_station(template["station"], 0), rng)
        for template in templates
    ]
    try:
        while True:
            for simulator in simulators:
                simulator.weather = load_weather_for_station(simulator.template["station"], 0)
                payload = simulator.reading(now)
                status = post(args.url, payload, args.api_key)
                marker = "ok " if status in (200, 201) else f"!{status}"
                print(
                    f"  {marker} {simulator.zone:<14} "
                    f"moisture {payload['soil_moisture']:5.1f}%  "
                    f"pH {payload['soil_ph']:.2f}  "
                    f"N {payload['nitrogen']:5.1f}  "
                    f"rain {payload['rainfall']:.1f}mm",
                    flush=True,
                )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
