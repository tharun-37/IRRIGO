"""
Crop-health and fertilizer API routes.

`POST /api/crop-health/analyze` is the blueprint's contract: a multipart image
with the selected crop, optional field/zone context, returning a status, a
prediction-or-absence, quality gates, advice and a persisted record id.

The upload constraints (MIME types, size ceiling, decode limits) live in
`schemas_crop_health.py`; every violation becomes a *record* with
`invalid_input`/`poor_image` rather than a bare 4xx, because the history page
should explain a failed upload the same way it explains an analysed one. Only
truly hostile requests (unreadable multipart, oversize at the HTTP layer)
return 4xx with a plain detail.

Consent: a photo stored without consent is not stored — it is analysed from
the request bytes alone and the stored image reference is deleted before the
response returns. Retention is real: `DELETE /api/crop-health/analyses/{id}`
removes both record and file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from ..schemas_crop_health import (
    MAX_UPLOAD_BYTES,
    CropHealthAnalysisOut,
    CropHealthStatusOut,
    FertilizerRecommendationIn,
    FertilizerRecommendationOut,
)
from ..services.crop_health_service import ImageError
from ..services.fertilizer_service import CROP_TARGETS_KG_HA, recommend

logger = logging.getLogger("iic.crop_health_routes")

router = APIRouter()

#: The snake_case → camelCase renames for a record that could not be persisted
#: through the repository, so the client sees one wire shape either way.
FALLBACK_RENAMES = {
    "created_at": "createdAt",
    "plant_part": "plantPart",
    "growth_stage": "growthStage",
    "field_id": "fieldId",
    "zone_id": "zoneId",
    "image_metadata": "imageMetadata",
    "prediction_label": "classLabel",
    "raw_score": "rawScore",
    "supported_classes": "supportedClasses",
    "calibrated": "calibrated",
    "result_status": "resultStatus",
    "status_reason": "statusReason",
    "model_version": "modelVersion",
    "preprocessing_version": "preprocessingVersion",
    "validation_status": "validationStatus",
    "consent_given": "consentGiven",
}


def _service(request: Request) -> Any:
    service = getattr(request.app.state, "crop_health", None)
    if service is None:  # pragma: no cover - lifespan always builds it
        raise HTTPException(status_code=503, detail="crop-health service is not loaded")
    return service


def _repository(request: Request) -> Any:
    service = _service(request)
    repo = getattr(service, "repository", None)
    if repo is None:  # pragma: no cover - persistence degrades, not dies
        raise HTTPException(
            status_code=503,
            detail="crop-health persistence is not available; see server log",
        )
    return repo


def _persist(request: Request, record: dict[str, Any]) -> dict[str, Any]:
    """
    Persist with graceful degradation; an unpersisted record still returns.
    The repository returns the camelCase wire row; the in-memory fallback is
    reshaped to the same form here so the client sees one wire shape either way.
    """
    try:
        return _repository(request).insert(record)
    except Exception:  # pragma: no cover - DB never blocks a response
        logger.exception("could not persist the analysis record")
        shaped = dict(record)
        shaped.pop("analysis_id", None)
        shaped.pop("image_reference", None)
        for source, target in FALLBACK_RENAMES.items():
            if source in shaped:
                shaped[target] = shaped.pop(source)
        # The wire shape nests prediction and advice; the in-memory record does
        # not, so rebuild them rather than answering with a broken shape.
        shaped.setdefault("analysisId", "unpersisted")
        shaped.setdefault("prediction", {
            "task": shaped.pop("task", None),
            "classLabel": shaped.get("classLabel"),
            "rawScore": shaped.get("rawScore"),
            "calibrated": shaped.pop("calibrated", False),
            "supportedClasses": shaped.pop("supportedClasses", []),
        })
        shaped.setdefault("advice", {
            "adviceCode": shaped.pop("advice_code", "service_error"),
            "adviceText": shaped.pop("advice_text", ""),
            "textVersion": shaped.pop("advice_text_version", ""),
            "followUpQuestion": shaped.pop("advice_follow_up"),
            "expertReferral": shaped.pop("advice_expert_referral", False),
        })
        shaped.setdefault("consentGiven", bool(shaped.pop("consent_given", False)))
        shaped.setdefault("gates", [])
        shaped.setdefault("quality", {})
        shaped.setdefault("imageReference", None)
        shaped.setdefault("imageBytes", 0)
        return shaped


# --------------------------------------------------------------------------
# crop health
# --------------------------------------------------------------------------


@router.post("/crop-health/analyze", response_model=CropHealthAnalysisOut)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    crop: str = Form(...),
    plant_part: str | None = Form(None),
    growth_stage: str | None = Form(None),
    field_id: str | None = Form(None),
    zone_id: str | None = Form(None),
    consent: bool = Form(False),
) -> dict[str, Any]:
    service = _service(request)
    payload = await file.read(MAX_UPLOAD_BYTES + 1)
    content_type = file.content_type or "application/octet-stream"
    filename = file.filename or "upload"
    try:
        record = service.analyze(
            filename=filename,
            content_type=content_type,
            payload=payload,
            crop=crop,
            plant_part=plant_part or None,
            growth_stage=growth_stage or None,
            field_id=field_id or None,
            zone_id=zone_id or None,
            consent=consent,
        )
    except ImageError as error:
        # Client-problem types were already caught inside the service and
        # turned into records; reaching here means the storage layer itself
        # failed.
        raise HTTPException(status_code=500, detail=str(error)) from error

    if not consent and record.get("image_reference"):
        # No consent, no retention — the photo is analysed from this request's
        # bytes alone and the stored copy is gone before anything persists.
        try:
            service.store.delete(record["image_reference"])
        except OSError:  # pragma: no cover
            logger.exception("could not delete the non-consented upload")
        record.pop("image_reference", None)
        record.pop("image_metadata", None)
        record.pop("quality", None)

    return _persist(request, record)


@router.get("/crop-health/status", response_model=CropHealthStatusOut)
def status(request: Request) -> dict[str, Any]:
    return _service(request).describe()


@router.get("/crop-health/analyses")
def list_analyses(
    request: Request,
    limit: int = 50,
    crop: str | None = None,
    status: str | None = None,
    field_id: str | None = None,
) -> dict[str, Any]:
    limit = max(1, min(limit, 200))
    rows = _repository(request).list(
        limit, crop=crop, status=status, field_id=field_id
    )
    return {"analyses": rows, "count": len(rows)}


@router.get("/crop-health/analyses/{analysis_id}")
def get_analysis(request: Request, analysis_id: str) -> dict[str, Any]:
    row = _repository(request).get(analysis_id)
    if row is None:
        raise HTTPException(status_code=404, detail="no such analysis record")
    return row


@router.post("/crop-health/analyses/{analysis_id}/feedback")
def feedback(request: Request, analysis_id: str, comment: str) -> dict[str, Any]:
    """Farmer feedback, consented and stored verbatim; never ground truth."""
    row = _repository(request).set_feedback(analysis_id, comment)
    if row is None:
        raise HTTPException(status_code=404, detail="no such analysis record")
    return row


@router.delete("/crop-health/analyses/{analysis_id}")
def delete_analysis(request: Request, analysis_id: str) -> dict[str, Any]:
    existed, reference = _repository(request).delete(analysis_id)
    if not existed:
        raise HTTPException(status_code=404, detail="no such analysis record")
    deleted_file = False
    if reference:
        try:
            deleted_file = _service(request).store.delete(reference)
        except OSError:  # pragma: no cover
            logger.exception("could not delete the stored file for %s", analysis_id)
    return {"deleted": True, "analysisId": analysis_id, "imageDeleted": deleted_file}


@router.get("/crop-health/manifest")
def manifest(request: Request) -> dict[str, Any]:
    """The dataset manifest from `backend/ml/crop_health`, if it has been built."""
    path = Path(__file__).resolve().parents[2] / "ml" / "crop_health" / "manifest.json"
    if not path.exists():  # pragma: no cover - manifest ships with the repo
        return {"generated": False, "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"manifest unreadable: {error}")


# --------------------------------------------------------------------------
# fertilizer recommendations
# --------------------------------------------------------------------------


@router.get("/fertilizer/targets")
def fertilizer_targets() -> dict[str, Any]:
    return {
        "source": "indicative classical recommendation table (verify against "
        "the state agriculture department for the district)",
        "units": "kg/ha (N, P2O5, K2O), full-season totals",
        "status": "indicative — under review; not district-verified",
        "targets": [
            {"crop": crop, "nitrogenKgHa": n, "p2o5KgHa": p, "k2oKgHa": k}
            for crop, (n, p, k) in sorted(CROP_TARGETS_KG_HA.items())
        ],
    }


@router.post("/fertilizer/recommend", response_model=FertilizerRecommendationOut)
def fertilizer_recommend(request: Request, asked: FertilizerRecommendationIn) -> Any:
    response = recommend(asked)
    try:
        repository = getattr(request.app.state, "fertilizer_repository", None)
        if repository is not None:
            repository.insert(response, asked)
    except Exception:  # pragma: no cover - persistence is never fatal
        logger.exception("could not persist the fertilizer recommendation")
    return response
