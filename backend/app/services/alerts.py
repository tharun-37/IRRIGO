"""
Alert rules.

Deliberately a small, explicit rule set rather than a framework. Each rule reads
a decision and optionally a device, and returns zero or more alerts. The value of
keeping it this plain is that the whole alerting behaviour of the system can be
read in one screen, which matters more here than configurability.

Two rules are about the *sensor* rather than the crop, and that distinction is
the important one: a reading outside the probe's physical range is a hardware
fault and must never be presented to the operator as a crop condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: Plausibility bounds for a 7-in-1 NPK probe. Outside these the reading is a
#: fault, not a value. The NPK ceiling is the sensor's own reported range.
SENSOR_LIMITS: dict[str, tuple[float, float]] = {
    "temperature": (-10.0, 80.0),
    "humidity": (0.0, 100.0),
    "soil_moisture": (0.0, 100.0),
    "soil_ph": (2.5, 9.5),
    "ec": (0.0, 20.0),
    "nitrogen": (0.0, 3000.0),
    "phosphorus": (0.0, 3000.0),
    "potassium": (0.0, 3000.0),
    "light_intensity": (0.0, 200_000.0),
}

#: Sustained pH deviation that warrants attention. Soil pH moves slowly, so a
#: single excursion is usually the electrode rather than the field.
PH_ALERT_LOW = 4.5
PH_ALERT_HIGH = 8.5

#: Salinity above this restricts uptake regardless of moisture.
EC_CRITICAL = 6.0

#: Battery below this on a 3.7 V lithium cell is near end of discharge.
BATTERY_LOW = 3.45

#: At or below this moisture the profile is critically dry on any soil.
MOISTURE_CRITICAL = 12.0


@dataclass
class AlertCandidate:
    kind: str
    severity: str
    message: str


def evaluate(
    decision: dict[str, Any],
    reading: dict[str, Any] | None = None,
    device: dict[str, Any] | None = None,
) -> list[AlertCandidate]:
    """
    Run every rule against one decision and return the alerts it warrants.

    Rules are ordered most to least severe so the dashboard can show the worst
    condition first without sorting on its own.
    """
    alerts: list[AlertCandidate] = []

    # -- sensor faults ----------------------------------------------------
    if reading is not None:
        for field, (low, high) in SENSOR_LIMITS.items():
            value = reading.get(field)
            if value is None:
                continue
            if not (low <= value <= high):
                alerts.append(
                    AlertCandidate(
                        kind="sensor_out_of_range",
                        severity="critical",
                        message=(
                            f"{field.replace('_', ' ').title()} reading {value:g} is "
                            f"outside the probe's usable range {low:g} to {high:g}. "
                            f"Treat as a hardware fault, not a crop condition."
                        ),
                    )
                )

        battery = reading.get("batteryVolts")
        if battery is not None and battery < BATTERY_LOW:
            alerts.append(
                AlertCandidate(
                    kind="battery_low",
                    severity="warning",
                    message=(
                        f"Node battery at {battery:.2f} V, below the {BATTERY_LOW} V "
                        f"threshold. Schedule a charge before readings become unreliable."
                    ),
                )
            )

    # -- water ------------------------------------------------------------
    moisture = decision.get("moistureBand")
    depletion = float(decision.get("depletionFraction", 0.0))
    if reading is not None and float(reading.get("soilMoisture", 100.0)) <= MOISTURE_CRITICAL:
        alerts.append(
            AlertCandidate(
                kind="soil_critically_dry",
                severity="critical",
                message=(
                    f"Soil moisture {reading['soilMoisture']:.1f}% is at or below the "
                    f"{MOISTURE_CRITICAL:.0f}% critical floor. The crop is at risk of "
                    f"irreversible wilting if not watered soon."
                ),
            )
        )
    elif depletion >= 0.85:
        alerts.append(
            AlertCandidate(
                kind="root_zone_depleted",
                severity="warning",
                message=(
                    f"Root zone is {depletion:.0%} depleted of plant-available water. "
                    f"Rain will be less effective than an irrigation event now."
                ),
            )
        )

    if not decision.get("shouldIrrigate") and moisture == "wet" and depletion < 0.2:
        alerts.append(
            AlertCandidate(
                kind="waterlogging_risk",
                severity="warning",
                message=(
                    "Profile is near field capacity with the valve closed. If rainfall "
                    "continues, aeration and nutrient leaching become a risk."
                ),
            )
        )

    # -- chemistry --------------------------------------------------------
    ph = None
    ec = None
    if reading is not None:
        ph = reading.get("soilPh")
        ec = reading.get("ec")

    if ph is not None and (ph < PH_ALERT_LOW or ph > PH_ALERT_HIGH):
        direction = "acidic" if ph < PH_ALERT_LOW else "alkaline"
        alerts.append(
            AlertCandidate(
                kind="ph_out_of_range",
                severity="warning",
                message=(
                    f"Soil pH {ph:.2f} is strongly {direction}. Nutrient availability is "
                    f"restricted whatever the reported NPK values suggest."
                ),
            )
        )

    if ec is not None and ec >= EC_CRITICAL:
        alerts.append(
            AlertCandidate(
                kind="salinity_high",
                severity="warning",
                message=(
                    f"Electrical conductivity {ec:.2f} mS/cm is above the "
                    f"{EC_CRITICAL} mS/cm salinity threshold, which restricts water "
                    f"uptake regardless of moisture."
                ),
            )
        )

    # -- disease risk -----------------------------------------------------
    probabilities = decision.get("diseaseProbabilities") or {}
    healthy = float(probabilities.get("Healthy", 1.0))
    if healthy < 0.6:
        top = max(
            ((k, v) for k, v in probabilities.items() if k != "Healthy"),
            key=lambda item: item[1],
            default=("unknown", 0.0),
        )
        alerts.append(
            AlertCandidate(
                kind="disease_risk",
                severity="critical" if healthy < 0.35 else "warning",
                message=(
                    f"Stress pattern is {1 - healthy:.0%} consistent with "
                    f"{top[0].replace('_', ' ').lower()}. This is a risk indicator from "
                    f"sensor telemetry, not a confirmed diagnosis; confirm in the field "
                    f"before treating."
                ),
            )
        )

    # -- nutrients --------------------------------------------------------
    demand = decision.get("nutrientDemandKgHa") or {}
    for nutrient, ceiling in (("nitrogen", 180.0), ("potassium", 130.0), ("phosphorus", 80.0)):
        value = float(demand.get(nutrient, 0.0))
        if value > ceiling:
            alerts.append(
                AlertCandidate(
                    kind=f"{nutrient}_deficient",
                    severity="info",
                    message=(
                        f"Recommended {nutrient} dose is {value:.0f} kg/ha, above the "
                        f"{ceiling:.0f} kg/ha maintenance ceiling for this crop. Split the "
                        f"application rather than applying it in one pass."
                    ),
                )
            )

    # -- link -------------------------------------------------------------
    if device is not None and device.get("linkState") == "offline":
        alerts.append(
            AlertCandidate(
                kind="node_offline",
                severity="critical",
                message="Node has stopped reporting. Check power and link before trusting "
                        "the last reading.",
            )
        )

    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda alert: severity_rank.get(alert.severity, 3))
    return alerts
