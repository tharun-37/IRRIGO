"""
Pydantic schemas for the crop-health surface and the fertilizer recommendation
layer.

These exist for the same reason `schemas.py` does: the wire format is camelCase
and the TypeScript types are written by hand, so the contract is stated once
here. The status enum is the blueprint's, not an invention of this codebase:

    result | uncertain | unsupported_crop | poor_image | invalid_input
    | processing_error

An analysis that reaches the farmer carries one of those values, and a raw
score is never implied to be a calibrated probability (`calibrated` is false
unless a validation report says otherwise).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

#: The five statuses the blueprint names plus the internal error surface. They
#: are exposed as plain strings through the API rather than as an OpenAPI enum
#: so an added state is never a breaking change for a hand-written client.
RESULT = "result"
UNCERTAIN = "uncertain"
UNSUPPORTED_CROP = "unsupported_crop"
POOR_IMAGE = "poor_image"
INVALID_INPUT = "invalid_input"
PROCESSING_ERROR = "processing_error"

#: Image constraints. Enforced before any decode work happens on the bytes.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ACCEPTED_MIME_TYPES = {
    "image/jpeg": (".jpg", ".jpeg"),
    "image/png": (".png",),
    "image/webp": (".webp",),
}

#: A decoded image larger than this is treated as invalid rather than decoded:
#: a decompression bomb has no business in a service that runs on a Pi-class
#: gateway. 24 Mpx is ~4x a demo phone photo.
MAX_DECODED_PIXELS = 24_000_000
MAX_DIMENSION_PX = 8_000
MIN_DIMENSION_PX = 240


class CropHealthGateOut(BaseModel):
    """One image-quality check that the analysis ran."""

    check: str
    passed: bool
    detail: str


class CropHealthPredictionOut(BaseModel):
    """
    The model's answer, or the absence of one.

    `rawScore` is whatever the model emits — never labelled a probability here.
    `calibrated` stays false until an evaluation report documents the method;
    the typography of honesty is deliberately cheap to see.
    """

    task: str | None = None
    classLabel: str | None = None
    rawScore: float | None = None
    calibrated: bool = False
    supportedClasses: list[str] = Field(default_factory=list)


class CropHealthAdviceOut(BaseModel):
    """Farmer-facing text, versioned so the wording can be traced."""

    adviceCode: str
    adviceText: str
    textVersion: str
    followUpQuestion: str | None = None
    expertReferral: bool


class CropHealthAnalysisOut(BaseModel):
    """The full minimum analysis record the blueprint requires."""

    analysisId: str
    createdAt: str
    crop: str
    plantPart: str | None = None
    growthStage: str | None = None
    fieldId: str | None = None
    zoneId: str | None = None
    resultStatus: str
    statusReason: str
    prediction: CropHealthPredictionOut
    advice: CropHealthAdviceOut
    quality: dict[str, Any] = Field(default_factory=dict)
    gates: list[CropHealthGateOut] = Field(default_factory=list)
    modelVersion: str
    preprocessingVersion: str
    validationStatus: str
    imageReference: str | None = None
    imageBytes: int = 0
    consentGiven: bool


class CropHealthStatusOut(BaseModel):
    """What the service can and cannot do right now, stated plainly."""

    serviceAvailable: bool
    deployedModel: bool
    modelVersion: str
    preprocessingVersion: str
    validationStatus: str
    supportedClasses: list[str] = Field(default_factory=list)
    supportedCrops: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    statusLabels: dict[str, str] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# fertilizer recommendation
# --------------------------------------------------------------------------

AreaUnit = Literal["hectare", "acre", "square_metre"]
SoilUnits = Literal["kg_per_ha", "ppm"]


class SoilTestIn(BaseModel):
    """
    A soil-test result, with the units it was reported in.

    Missing values are `None`, never zero — the blueprint is explicit that a
    blank row must not read as "no nutrient present".
    """

    units: SoilUnits = "kg_per_ha"
    nitrogen: float | None = None
    phosphorus: float | None = None
    potassium: float | None = None
    ph: float | None = None
    source: str | None = None
    sampledOn: str | None = None


class ProductIn(BaseModel):
    """A fertilizer formulation, stated as oxidise-form percentages."""

    name: str
    nitrogenPct: float | None = None
    p2o5Pct: float | None = None
    k2oPct: float | None = None


class FertilizerRecommendationIn(BaseModel):
    crop: str
    growthStage: str | None = None
    areaValue: float = Field(gt=0)
    areaUnit: AreaUnit = "acre"
    soilTest: SoilTestIn | None = None
    product: ProductIn | None = None
    method: str | None = None


class NutrientLineOut(BaseModel):
    """One nutrient, in one unit system: per hectare and for the field."""

    nutrient: str
    targetKgPerHa: float
    supplyKgPerHa: float | None = None
    deficitKgPerHa: float | None = None
    totalForFieldKg: float | None = None


class FertilizerRecommendationOut(BaseModel):
    """
    A recommendation with every number's provenance attached.

    `status` is `computed` only when the derivation is complete. A missing soil
    test yields `needs_soil_test`, a missing formulation yields
    `missing_product`; neither invents a quantity.
    """

    status: Literal["computed", "needs_soil_test", "missing_product"]
    crop: str
    growthStage: str | None
    areaValue: float
    areaUnit: str
    areaHectares: float
    productName: str | None = None
    nutrients: list[NutrientLineOut]
    totalProductForFieldKg: float | None = None
    applicationPerHaKg: float | None = None
    splits: list[dict[str, Any]] = Field(default_factory=list)
    derivation: list[str] = Field(default_factory=list)
    missingInputs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    reminders: list[str] = Field(default_factory=list)
