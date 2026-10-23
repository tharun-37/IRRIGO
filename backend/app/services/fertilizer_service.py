"""
The fertilizer recommendation layer.

This is deliberately *not* an ML module. The blueprint separates them: a model
may estimate nutrient demand, but a recommendation a farmer can act on has to
be built from explicit inputs — a soil test, a crop stage, a field area and the
product's actual nutrient percentages — with every quantity's unit shown and
every gap stated instead of filled with an invented number.

The arithmetic is intentionally boring, because boring is what can be unit
tested against an agronomist's worked example:

    deficit (kg/ha)   = crop target (kg/ha) − soil supply (kg/ha), floored at 0
    product per ha    = deficit ÷ product nutrient fraction
    product for field = per-ha mass × area in hectares

Missing soil test → `needs_soil_test`. Missing formulation → `missing_product`.
Neither ever produces a product mass. PPM soil values are converted with a
stated assumption, not pretended to be exact.

Areas: hectare, acre and square metre only. Acreage converts through the exact
factor (1 acre = 4046.8564224 m²), and the area used is echoed in the response,
so a wrong unit can be spotted from the output alone.
"""

from __future__ import annotations

from typing import Any

from ..schemas_crop_health import (
    FertilizerRecommendationIn,
    FertilizerRecommendationOut,
    NutrientLineOut,
)

M2_PER_ACRE = 4046.8564224
M2_PER_HA = 10_000.0

#: Weight of the furrow slice (soil 15 cm deep over one hectare) in kg, the
#: standard basis for converting mg/kg soil test values to kg/ha.
FURROW_SLICE_KG_PER_HA = 2_240_000.0

#: The indicative crop nutrient targets. These come from common Indian
#: recommendation tables (the classical N:P:K ratios for the big crops) and are
#: labelled indicative everywhere they surface: a real pilot must overlay the
#: state agriculture department's recommendation for the district.
CROP_TARGETS_KG_HA: dict[str, tuple[float, float, float]] = {
    # crop: (N, P2O5, K2O) kg/ha, full-season, at recommended basal + splits
    "rice": (120.0, 60.0, 50.0),
    "wheat": (120.0, 60.0, 40.0),
    "maize": (120.0, 60.0, 40.0),
    "cotton": (90.0, 45.0, 45.0),
    "potato": (180.0, 100.0, 120.0),
    "tomato": (80.0, 60.0, 60.0),
    "barley": (60.0, 30.0, 20.0),
    "sugarcane": (200.0, 60.0, 40.0),
    "groundnut": (20.0, 40.0, 40.0),
    "onion": (80.0, 40.0, 50.0),
}

REMINDERS = [
    "Cross-check the crop's dose with your state agriculture department's "
    "recommendation before buying or applying anything.",
    "Read and follow the product label; the percentages it states override "
    "any entered here.",
    "This calculation came from a soil test plus entered percentages — have "
    "it checked by an agricultural officer or agronomist before field use.",
]

#: A single application of urea-class product is commonly held near this cap
#: in Indian practice; more than one split is then scheduled. Stated as an
#: assumption, not an agronomic truth.
SPLIT_CAP_KG_HA = 80.0


def area_to_hectares(value: float, unit: str) -> float:
    """Exact conversion, one place, unit types enforced upstream."""
    if unit == "hectare":
        return value
    if unit == "acre":
        return value * M2_PER_ACRE / M2_PER_HA
    if unit == "square_metre":
        return value / M2_PER_HA
    raise ValueError(f"unsupported area unit {unit!r}")


def soil_to_kg_ha(value: float | None, units: str) -> float | None:
    """
    A soil-test value in kg/ha, or None when the input is absent.

    PPM/available-K values convert through the furrow-slice weight; the
    phosphorus basis ambiguity (P vs P2O5, Olsen vs Bray) is surfaced as an
    assumption by the caller, not hidden here.
    """
    if value is None:
        return None
    if units == "kg_per_ha":
        return float(value)
    if units == "ppm":
        return float(value) * FURROW_SLICE_KG_PER_HA / 1_000_000.0
    raise ValueError(f"unsupported soil units {units!r}")


def _line(nutrient: str, target: float, supply: float | None) -> NutrientLineOut:
    deficit = None if supply is None else max(0.0, target - supply)
    return NutrientLineOut(
        nutrient=nutrient,
        targetKgPerHa=target,
        supplyKgPerHa=supply,
        deficitKgPerHa=deficit,
    )


def recommend(request: FertilizerRecommendationIn) -> FertilizerRecommendationOut:
    """
    One recommendation. Pure: the request in, the fully-provenanced response
    out. No database, no model, no network.
    """
    crop_key = (request.crop or "").strip().lower()
    area_ha = area_to_hectares(request.areaValue, request.areaUnit)
    derivation: list[str] = [
        f"Field area: {request.areaValue:g} {request.areaUnit.replace('_', ' ')} "
        f"= {area_ha:.4f} ha.",
    ]
    missing: list[str] = []
    assumptions: list[str] = []
    splits: list[dict[str, Any]] = []

    targets = CROP_TARGETS_KG_HA.get(crop_key)
    if targets is not None:
        derivation.append(
            f"Indicative crop targets used for {crop_key}: "
            f"N {targets[0]:g}, P2O5 {targets[1]:g}, K2O {targets[2]:g} kg/ha."
        )
        assumptions.append(
            "The nutrient targets are generic indicative values for the crop, "
            "not a district-specific recommendation."
        )
    else:
        derivation.append(
            f"No indicative target table covers {crop_key!r}; the response "
            "needs a target from an agronomic source to become a plan."
        )
        missing.append(
            f"a verified nutrient target for {request.crop} — none is on file "
            "for this crop"
        )
        targets = (0.0, 0.0, 0.0)

    soil = request.soilTest
    soil_supplies: list[float | None]
    if soil is None:
        soil_supplies = [None, None, None]
        derivation.append(
            "No soil test was supplied. Without one, the recommendation stops "
            "at the crop targets: blanket quantities are not invented here."
        )
        missing.append("a soil test (N, P, K) with its source and sampling date")
    else:
        converted = [
            soil_to_kg_ha(soil.nitrogen, soil.units),
            soil_to_kg_ha(soil.phosphorus, soil.units),
            soil_to_kg_ha(soil.potassium, soil.units),
        ]
        soil_supplies = converted
        if any(value is None for value in converted):
            gaps = [
                name
                for name, value in (
                    ("N", converted[0]), ("P", converted[1]), ("K", converted[2])
                )
                if value is None
            ]
            missing.append(
                "soil-test values for: " + ", ".join(gaps) + " (absent is not zero)"
            )
        if soil.units == "ppm":
            assumptions.append(
                "Soil values were reported as ppm and converted at "
                "2.24 million kg furrow-slice per ha; phosphorus results are "
                "assumed to be extractable P, stated as P2O5 equivalent is "
                "not verified."
            )
        if soil.source:
            derivation.append(
                f"Soil test from {soil.source}"
                + (f", sampled {soil.sampledOn}." if soil.sampledOn else ".")
            )
        if soil.ph is not None and not 5.5 <= soil.ph <= 8.5:
            derivation.append(
                f"Soil pH {soil.ph:g} is outside the 5.5-8.5 band where "
                "nutrients behave as the table assumes; ask the officer to "
                "review pH before applying."
            )

    nutrients = [
        _line(name, target, supply)
        for name, target, supply in zip(("N", "P2O5", "K2O"), targets, soil_supplies)
    ]

    product = request.product
    deficits = [item.deficitKgPerHa for item in nutrients]
    if all(value is not None for value in deficits) and all(v is not None for v in soil_supplies):
        for item in nutrients:
            item.totalForFieldKg = round(
                (item.deficitKgPerHa or 0.0) * area_ha, 2
            )
        derivation.append(
            "Deficit per hectare = target − supply, floored at zero; the "
            "total for the field is that deficit × the area above."
        )

    status: str
    total_mass: float | None = None
    per_ha_mass: float | None = None
    product_name: str | None = None

    if product is None or (
        product.nitrogenPct is None
        and product.p2o5Pct is None
        and product.k2oPct is None
    ):
        status = "missing_product"
        missing.append(
            "the fertilizer product's formulation (N %, P2O5 %, K2O %) — "
            "without it no product mass can be calculated"
        )
        derivation.append(
            "No product formulation given, so no product mass is calculated; "
            "only the nutrient quantities above stand."
        )
    elif any(value is None for value in soil_supplies) or targets == (0.0, 0.0, 0.0):
        status = "needs_soil_test"
        derivation.append(
            "With the soil test incomplete, no product mass should be "
            "calculated even though a formulation was given."
        )
        product_name = product.name
    else:
        status = "computed"
        product_name = product.name

        def _mass(nutrient_index: int, pct: float | None) -> float | None:
            deficit = deficits[nutrient_index]
            if pct is None or pct <= 0 or deficit is None:
                return None
            return (deficit / (pct / 100.0)) * area_ha

        contributions = [
            _mass(0, product.nitrogenPct),
            _mass(1, product.p2o5Pct),
            _mass(2, product.k2oPct),
        ]
        quantities = [value for value in contributions if value is not None]
        if not quantities:
            status = "missing_product"
            missing.append(
                "a product that actually supplies one of the deficient "
                "nutrients (its percentages are all absent or zero)"
            )
            derivation.append(
                "The formulation carries none of the nutrients in deficit, so "
                "no quantity is calculated against it."
            )
        else:
            total_mass = round(sum(quantities), 2)
            per_ha_mass = (
                round(sum(q / area_ha for q in quantities), 2) if area_ha > 0 else None
            )
            derivation.append(
                f"Product mass = deficit ÷ nutrient fraction, per nutrient, "
                f"summed: {total_mass:g} kg of {product.name} for the whole "
                f"field ({per_ha_mass:g} kg/ha)."
            )
            if per_ha_mass is not None and per_ha_mass > SPLIT_CAP_KG_HA:
                n_splits = int(-(-per_ha_mass // SPLIT_CAP_KG_HA))
                single = round(total_mass / n_splits, 2)
                for index in range(n_splits):
                    splits.append(
                        {"sequence": index + 1, "kgForField": single}
                    )
                derivation.append(
                    f"At more than {SPLIT_CAP_KG_HA:g} kg/ha per pass the "
                    f"application is split into {n_splits} doses of about "
                    f"{single:g} kg each."
                )

    if status == "computed":
        derivation.insert(
            0,
            "Everything below is derived; check the units on every line.",
        )

    return FertilizerRecommendationOut(
        status=status,
        crop=request.crop,
        growthStage=request.growthStage,
        areaValue=request.areaValue,
        areaUnit=request.areaUnit,
        areaHectares=round(area_ha, 4),
        productName=product_name,
        nutrients=nutrients,
        totalProductForFieldKg=total_mass,
        applicationPerHaKg=per_ha_mass,
        splits=splits,
        derivation=derivation,
        missingInputs=missing,
        assumptions=assumptions,
        reminders=REMINDERS,
    )
