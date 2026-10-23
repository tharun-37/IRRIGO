"""
Analytics aggregation for the dashboard.

Everything here is derived from the stored `decisions` and `devices` rows, so
the figures on screen are computed from the same records the system acted on
rather than recomputed separately in the browser.

Two of the numbers are estimates and are labelled as such in the UI:

* `estimated_litres_saved` compares applied water against a fixed-ET0 baseline
  with no rainfall and no leaching. It is a relative indication of scheduling
  efficiency, not a metered volume.
* `energy_kwh` assumes a nominal pump power and lifts the applied volume once.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

#: Nominal pump rating for the energy estimate. A 1 hp centrifugal pump at
#: typical field head is the common case; override per deployment if measured.
PUMP_POWER_WATTS = 750.0

#: Litres per cubic metre, for the volume/depth conversion.
LITRES_PER_M3 = 1000.0

#: Field area assumed when a device has no configured area, in square metres.
DEFAULT_AREA_M2 = 100.0

#: How many buckets the series is divided into.
SERIES_BUCKETS = 24


def _parse(value: str) -> datetime:
    text = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _percent_change(current: float, previous: float) -> float:
    """Percentage change from previous to current. Zero base returns 0.0."""
    if previous <= 0:
        return 0.0
    return round((current - previous) / previous * 100.0, 2)


class AnalyticsService:
    def __init__(self, devices, decisions) -> None:
        self.devices = devices
        self.decisions = decisions

    def overview(self, hours: int = 24) -> dict[str, Any]:
        current = self.decisions.since(hours=hours)
        previous = self.decisions.since(hours=hours * 2)
        cutoff = _parse(previous[0]["decidedAt"]) if previous else None
        if cutoff is not None:
            previous = [row for row in previous if _parse(row["decidedAt"]) < cutoff]

        area_by_device = {
            device["id"]: float(device.get("fieldAreaM2") or DEFAULT_AREA_M2)
            for device in self.devices.list()
        }

        runtime = sum(float(row["durationSeconds"] or 0) for row in current)
        latency = [
            float(row["inferenceMs"] or 0.0)
            for row in current
            if row.get("inferenceMs") is not None
        ]

        # Mean depth per irrigation event. Summing depth x area across zones
        # would produce a volume, not the millimetre figure the label implies.
        applied_current = self._mean_applied_mm(current)
        applied_previous = self._mean_applied_mm(previous)
        events_current = [row for row in current if row["shouldIrrigate"]]

        # Against a fixed-ET0 baseline with no rain and no leaching, the water a
        # naive schedule would have applied for an event is ET0 * Kc * area.
        # Anything the controller skipped is the saving.
        baseline_litres = 0.0
        for row in events_current:
            area = area_by_device.get(row["deviceId"], DEFAULT_AREA_M2)
            baseline_litres += float(row["et0MmDay"] or 0.0) * float(row["kc"] or 0.0) * area

        applied_litres = self._applied_litres(events_current, area_by_device)
        saved = max(0.0, baseline_litres - applied_litres)

        energy_kwh = (PUMP_POWER_WATTS * runtime) / 3_600_000.0

        return {
            "window": f"last {hours}h",
            "waterAppliedMm": round(applied_current, 2),
            "waterAppliedPctChange": _percent_change(applied_current, applied_previous),
            "estimatedLitresSaved": round(saved, 1),
            "irrigationEvents": len(events_current),
            "runtimeSeconds": int(runtime),
            "averageInferenceMs": round(
                (sum(latency) / len(latency)) if latency else 0.0, 4
            ),
            "decisionsCorrectPct": self._agreement_pct(current),
            "energyKwh": round(energy_kwh, 4),
            "series": self._series(current, hours),
        }

    @staticmethod
    def _mean_applied_mm(rows: list[dict[str, Any]]) -> float:
        """Mean applied depth per irrigation event, in millimetres."""
        depths = [float(row["depthMm"] or 0.0) for row in rows if row["shouldIrrigate"]]
        if not depths:
            return 0.0
        return sum(depths) / len(depths)

    @staticmethod
    def _applied_litres(rows: list[dict[str, Any]], area_by_device: dict[str, float]) -> float:
        total = 0.0
        for row in rows:
            area = area_by_device.get(row["deviceId"], DEFAULT_AREA_M2)
            volume_m3 = (float(row["depthMm"] or 0.0) / 1000.0) * area
            total += volume_m3 * LITRES_PER_M3
        return total

    @staticmethod
    def _agreement_pct(rows: list[dict[str, Any]]) -> float:
        """
        Share of decisions that agree with a simple rule check.

        The rule is intentionally transparent: irrigate when the depletion
        fraction exceeds the threshold implied by the moisture band. Agreement is
        a sanity signal on the classifier, not a measure of agronomic truth.
        """
        if not rows:
            return 0.0
        agreed = 0
        for row in rows:
            depletion = float(row["depletionFraction"] or 0.0)
            band = row["moistureBand"]
            rule_fires = depletion >= 0.55 if band != "wet" else False
            if rule_fires == bool(row["shouldIrrigate"]):
                agreed += 1
        return round(agreed / len(rows) * 100.0, 2)

    @staticmethod
    def _series(rows: list[dict[str, Any]], hours: int) -> list[dict[str, Any]]:
        """Bucket decisions and ET0 into evenly spaced time slots."""
        if not rows:
            return []

        start = _parse(rows[0]["decidedAt"])
        end = _parse(rows[-1]["decidedAt"])
        if end <= start:
            end = start + timedelta(hours=hours)
        span = (end - start).total_seconds()
        width = max(span / SERIES_BUCKETS, 1.0)

        buckets: dict[int, dict[str, float]] = {}
        for row in rows:
            index = int((_parse(row["decidedAt"]) - start).total_seconds() // width)
            bucket = buckets.setdefault(
                index, {"appliedMm": 0.0, "et0MmDay": 0.0, "depletionFraction": 0.0, "n": 0.0}
            )
            if row["shouldIrrigate"]:
                bucket["appliedMm"] += float(row["depthMm"] or 0.0)
            bucket["et0MmDay"] += float(row["et0MmDay"] or 0.0)
            bucket["depletionFraction"] += float(row["depletionFraction"] or 0.0)
            bucket["n"] += 1.0

        series = []
        for index in sorted(buckets):
            bucket = buckets[index]
            count = bucket["n"] or 1.0
            moment = start + timedelta(seconds=width * (index + 0.5))
            series.append(
                {
                    "label": moment.strftime("%H:%M"),
                    "appliedMm": round(bucket["appliedMm"], 2),
                    "et0MmDay": round(bucket["et0MmDay"] / count, 3),
                    "depletionFraction": round(bucket["depletionFraction"] / count, 4),
                }
            )
        return series
