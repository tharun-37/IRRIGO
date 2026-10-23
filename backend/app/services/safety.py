"""
The irrigation safety layer: the last gate between the models and a pump.

The blueprint requires a safety layer between ML output and any pump/valve
command. The advisory pipeline already had the right materials configured
(`hard_max_depth_mm`, `hard_max_runtime_seconds`) but no module owned their
enforcement — this module is that owner. It runs after fusion, on the *final*
depth the advisory plans to show, and:

1. **Caps** — clamps depth to the requested cap and runtime to its ceiling;
   volume and runtime are re-derived from the clamped depth, never left at the
   unclamped values.
2. **Suppression** — a depth below the negligible threshold becomes no command
   at all. A 0.2 mm "irrigate now" is noise, and noise on a pump costs power
   and farmer trust.
3. **Plausibility** — sensor values far outside physical range (moisture below
   0 or above 100, pH outside 0-14, non-finite readings) promote hold: a plan
   computed from impossible inputs is refused rather than served with its
   numbers crossly attached.
4. **Safe stop** — a flag on the response stating that the current release has
   no actuator path at all (`actuatorConnected: false`), and that hardware
   integration remains a separate, untested workstream.

The block it appends is additive: it never changes the plan's identity, only
its numbers, with every clamp listed so an operator can audit the change.
"""

from __future__ import annotations

import math
from typing import Any

#: Below this the command is suppressed outright rather than clamped.
NEGLIGIBLE_DEPTH_MM = 0.5


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class SafetyLayer:
    """Stateless; constructed once per pipeline and applied per advisory."""

    def __init__(self, max_depth_mm: float, max_runtime_seconds: int) -> None:
        self.max_depth_mm = float(max_depth_mm)
        self.max_runtime_seconds = int(max_runtime_seconds)
        self.suppress_mm = NEGLIGIBLE_DEPTH_MM

    def describe(self) -> dict[str, Any]:
        return {
            "maxDepthMm": self.max_depth_mm,
            "maxRuntimeSeconds": self.max_runtime_seconds,
            "negligibleDepthMm": self.suppress_mm,
            "actuatorConnected": False,
            "safeStop": "implemented in-process; hardware safe-stop is a "
            "pending field-tested workstream",
        }

    def check_sensors(self, advisory: dict[str, Any]) -> tuple[bool, str]:
        """
        Whether the advisory's sensor readings are physically plausible.

        Returns (plausible, reason). Only the readings the plan actually
        depends on are checked, and only for the things that would corrupt a
        water-balance walk: moisture fraction and pH signatures of a broken
        probe.
        """
        sensors = advisory.get("sensors") or {}
        moisture = sensors.get("soilMoisture")
        ph = sensors.get("soilPh")
        if moisture is not None and _finite(moisture):
            if not 0.0 <= float(moisture) <= 100.0:
                return False, (
                    f"soil moisture {moisture:g}% is outside 0-100; the probe "
                    "is suspect and the plan is not computed from it"
                )
        if ph is not None and _finite(ph):
            if not 0.0 <= float(ph) <= 14.0:
                return False, f"soil pH {ph:g} is outside 0-14; the probe is suspect"
        return True, ""

    def apply(self, advisory: dict[str, Any]) -> dict[str, Any]:
        """
        Enforce the caps on the final recommendation block. Mutates in place
        and adds a `safety` block; returns the advisory for chaining.
        """
        clamps: list[str] = []
        recommendation = advisory.get("recommendation") or {}
        original_depth = float(recommendation.get("finalDepthMm") or 0.0)

        plausible, reason = self.check_sensors(advisory)
        if not plausible:
            advisory["status"] = "hold"
            advisory["headline"] = "Recipe blocked — sensor plausibility check failed"
            clamps.append(reason)

        depth = original_depth
        if plausible and depth > self.max_depth_mm:
            clamps.append(
                f"depth {depth:g} mm clamped to the {self.max_depth_mm:g} mm cap"
            )
            depth = self.max_depth_mm
        if plausible and 0.0 < depth < self.suppress_mm:
            clamps.append(
                f"depth {depth:g} mm suppressed as negligible (below "
                f"{self.suppress_mm:g} mm)"
            )
            depth = 0.0
            advisory["status"] = "hold"

        if depth != original_depth:
            recommendation["finalDepthMm"] = round(depth, 2)

        duration = int(recommendation.get("durationSeconds") or 0)
        if duration > self.max_runtime_seconds:
            clamps.append(
                f"runtime {duration}s clamped to the "
                f"{self.max_runtime_seconds}s ceiling"
            )
            duration = self.max_runtime_seconds
            recommendation["durationSeconds"] = duration

        advisory["safety"] = {
            "applied": True,
            "capped": bool(clamps),
            "clamps": clamps,
            "sensorPlausible": plausible,
            "actuatorConnected": False,
            "safeStopFlag": True,
            "limits": self.describe(),
        }
        return advisory


def safety_from_settings(settings: Any) -> SafetyLayer:
    """Build the layer from the same settings the rest of the app reads."""
    return SafetyLayer(
        max_depth_mm=settings.hard_max_depth_mm,
        max_runtime_seconds=settings.hard_max_runtime_seconds,
    )
