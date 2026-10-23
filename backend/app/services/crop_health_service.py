"""
The crop-health analysis service.

What this is, honestly: an image-receiving, quality-gating, record-keeping
service with an inference *slot*. No validated disease/pet/nutrient image model
exists in this repository yet, so unless a versioned artifact is installed the
service never returns `result` — it runs the quality gates and returns
`uncertain` with the reason stated. A raw heuristic statistic is not a
diagnosis, and the advice text says so.

The status machine follows the blueprint:

- `invalid_input`   — empty/oversized/mismatched upload, unknown crop id
- `poor_image`      — decoded but blurry, too dark, too bright, flat or tiny
- `unsupported_crop`— a *deployed* model does not cover the selected crop
- `uncertain`       — gates passed, model absent or confidence below threshold
- `result`          — a deployed, validated model returned a supported class
- `processing_error`— anything the service itself broke on

`descriptor`-style labels ("proposed", "under test", "pending") travel with
every response through `validationStatus`, so nothing can be quoted as
accurate before it has an independent test set.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ..schemas_crop_health import (
    POOR_IMAGE,
    PROCESSING_ERROR,
    RESULT,
    UNCERTAIN,
    UNSUPPORTED_CROP,
    INVALID_INPUT,
    CropHealthGateOut,
    CropHealthPredictionOut,
)
from .image_storage import ImageError, ImageStore, analyse_pixels

logger = logging.getLogger("iic.crop_health")

#: Version strings. The preprocessing version changes whenever the statistics
#: above change; the model content version comes from the artifact, not here.
PREPROCESSING_VERSION = "quality-gates-v1"
SERVICE_MODEL_VERSION = "none-deployed"

#: Status of the validation work behind the deployed artifact. Nothing has run
#: on an independent field test set, so this never reads better than the truth.
VALIDATION_STATUS = "not validated — quality checks only, under test"

#: Sharpness/level thresholds. Chosen by hand against demo photos, never
#: against a labelled set — so they are labelled "under test" too.
BLUR_VARIANCE_MIN = 90.0
LUMA_DARK_MAX = 45.0
LUMA_BRIGHT_MIN = 230.0
LUMA_STD_FLAT_MAX = 8.0

#: Below this calibrated-equivalent confidence a deployed model abstains.
ABSTENTION_CONFIDENCE = 0.55

#: The advice text set. Each entry has a version so a wording change is
#: traceable from the analysis record.
ADVICE_VERSION = "en-1"
ADVICE: dict[str, dict[str, Any]] = {
    "poor_image": {
        "adviceCode": "retake_photo",
        "adviceText": (
            "We could not examine this photo clearly. Move closer, hold the "
            "camera steady, use even daylight and take another photo of the "
            "whole leaf area."
        ),
        "followUpQuestion": None,
        "expertReferral": False,
    },
    "invalid_input": {
        "adviceCode": "check_upload",
        "adviceText": (
            "The upload could not be used. Send a square-on JPEG, PNG or WebP "
            "photo under 10 MB of a single leaf or plant."
        ),
        "followUpQuestion": None,
        "expertReferral": False,
    },
    "unsupported_crop": {
        "adviceCode": "crop_not_covered",
        "adviceText": (
            "IRRIIGO does not yet have a validated image model for this crop, "
            "so it cannot name the problem. Do not apply treatment based on "
            "the photo alone. Ask a local agricultural officer if the problem "
            "continues."
        ),
        "followUpQuestion": "Which crop is this? The officer will need the crop and the pattern of the damage.",
        "expertReferral": True,
    },
    "uncertain": {
        "adviceCode": "no_reliable_diagnosis",
        "adviceText": (
            "IRRIIGO could not identify this problem reliably. Do not apply "
            "treatment based only on this result. Check the plant again in "
            "a day — whether spots or marks are spreading is useful information "
            "for an expert."
        ),
        "followUpQuestion": "Is the same pattern appearing on neighbouring plants?",
        "expertReferral": True,
    },
    "result": {
        "adviceCode": "possible_sighting_check",
        "adviceText": (
            "This result is a model's reading, not a confirmed diagnosis. "
            "Verify in the field before spraying or applying anything, and "
            "contact your agricultural officer for a confirmed identification."
        ),
        "followUpQuestion": None,
        "expertReferral": True,
    },
    "processing_error": {
        "adviceCode": "service_error",
        "adviceText": (
            "The analysis service failed on this photo. Nothing was assumed. "
            "If this repeats, the operator should check the service logs."
        ),
        "followUpQuestion": None,
        "expertReferral": False,
    },
}


class CropHealthArtifact:
    """
    A deployed image model, loaded from disk.

    The contract is deliberately small: a joblib object exposing `predict` (and
    optionally `predict_proba`) over the six scalar statistics from
    `analyse_pixels`, plus a side-car JSON with `classes`, `crops`,
    `validationStatus` and `modelVersion`. The same artifact contract is what
    `backend/ml/crop_health` documents, so training and serving agree.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.meta: dict[str, Any] = {}
        self.model: Any = None
        self.model_version = SERVICE_MODEL_VERSION
        self.validation_status = VALIDATION_STATUS
        self.error: str | None = None

    @property
    def present(self) -> bool:
        return self.model is not None and bool(self.meta.get("classes"))

    def load(self) -> None:
        path = self.directory / "crop_health_model.joblib"
        meta_path = self.directory / "model_meta.json"
        if not path.exists() or not meta_path.exists():
            return
        try:
            import json

            self.model = joblib.load(path)
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.model_version = str(self.meta.get("modelVersion", "artifact-unknown"))
            self.validation_status = str(
                self.meta.get("validationStatus", VALIDATION_STATUS)
            )
            logger.info("crop-health artifact loaded: %s", self.model_version)
        except Exception as error:  # pragma: no cover - defensive
            self.model = None
            self.error = str(error)
            logger.exception("crop-health artifact at %s failed to load", path)

    def classes(self) -> list[str]:
        return list(self.meta.get("classes", []))

    def crops(self) -> list[str]:
        return list(self.meta.get("crops", []))


class CropHealthService:
    """Runs one request through gates → inference slot → record."""

    def __init__(self, settings: Any, repository: Any = None) -> None:
        self.settings = settings
        self.repository = repository
        self.store = ImageStore(Path(settings.crop_health_images_dir))
        self.artifact = CropHealthArtifact(
            Path(settings.models_dir) / "crop_health"
        )
        self.artifact.load()
        self._feature_order = (
            "lumaMean", "lumaStd", "sharpness",
            "greenFraction", "darkFraction", "yellowFraction",
        )

    def describe(self) -> dict[str, Any]:
        """The `GET /api/crop-health/status` payload."""
        deployed = self.artifact.present
        return {
            "serviceAvailable": True,
            "deployedModel": deployed,
            "modelVersion": self.artifact.model_version,
            "preprocessingVersion": PREPROCESSING_VERSION,
            "validationStatus": self.artifact.validation_status,
            "supportedClasses": self.artifact.classes(),
            "supportedCrops": self.artifact.crops(),
            "limitations": (
                self.artifact.meta.get("limitations", []) if deployed else [
                    "No disease, pest or nutrient image model is deployed. "
                    "Uploads are checked for photo quality only.",
                    "The analysis never confirms a disease and never names a "
                    "pesticide or dose; that needs a trained model plus "
                    "agronomic review.",
                ]
            ),
            "statusLabels": {
                "model": "pending (no validated artifact deployed)",
                "qualityGates": "implemented (under test)",
                "endpoint": "implemented",
            },
        }

    # --- gates -----------------------------------------------------------

    def _gate(self, check: str, passed: bool, detail: str) -> CropHealthGateOut:
        return CropHealthGateOut(check=check, passed=passed, detail=detail)

    def _run_gates(self, stats: dict[str, Any]) -> tuple[list[CropHealthGateOut], bool]:
        """
        The four gates named in the blueprint: blur, level, saturation, size.

        Returns the gate list and whether the photo is fit for analysis at all.
        """
        gates = [
            self._gate(
                "resolution",
                stats["width"] >= 240 and stats["height"] >= 240,
                f"{stats['width']}x{stats['height']} (min 240px each side)",
            ),
            self._gate(
                "blur",
                stats["sharpness"] >= BLUR_VARIANCE_MIN,
                (
                    f"texture variance {stats['sharpness']:.0f} "
                    f"(min {BLUR_VARIANCE_MIN:.0f})"
                ),
            ),
            self._gate(
                "exposure",
                LUMA_DARK_MAX <= stats["lumaMean"] <= LUMA_BRIGHT_MIN,
                (
                    f"mean brightness {stats['lumaMean']:.0f}/255 "
                    f"(needs {LUMA_DARK_MAX:.0f}-{LUMA_BRIGHT_MIN:.0f})"
                ),
            ),
            self._gate(
                "content",
                stats["lumaStd"] >= LUMA_STD_FLAT_MAX,
                f"brightness variation {stats['lumaStd']:.0f} (min {LUMA_STD_FLAT_MAX:.0f}, "
                "a flat image is not a leaf)",
            ),
        ]
        return gates, all(gate.passed for gate in gates)

    # --- the analysis -----------------------------------------------------

    def analyze(
        self,
        *,
        filename: str,
        content_type: str,
        payload: bytes,
        crop: str,
        plant_part: str | None = None,
        growth_stage: str | None = None,
        field_id: str | None = None,
        zone_id: str | None = None,
        consent: bool = False,
    ) -> dict[str, Any]:
        """
        Full request path, pure: bytes in, an analysis record out.

        Every failure becomes a *complete record* with the right status — a
        farmer's failed upload is still a record, because the retention and
        feedback story needs it. The caller persists the returned dict through
        the repository; nothing here touches the database. ImageError-free:
        even a malformed upload is a record, per the blueprint.
        """
        status_reason = ""
        status = UNCERTAIN
        prediction = CropHealthPredictionOut()
        gates: list[CropHealthGateOut] = []
        quality: dict[str, Any] = {}
        stored = None
        source = ADVICE[UNCERTAIN]

        try:
            try:
                stored = self.store.sanitise(filename, content_type, payload)
            except ImageError as error:
                status = INVALID_INPUT
                status_reason = str(error)
                return self._record(
                    status, status_reason, prediction, gates, quality, stored,
                    crop, plant_part, growth_stage, field_id, zone_id, consent,
                )

            try:
                image, quality = self.store.pixels(stored.reference)
            except ImageError as error:
                status = INVALID_INPUT
                status_reason = str(error)
                return self._record(
                    status, status_reason, prediction, gates, quality, stored,
                    crop, plant_part, growth_stage, field_id, zone_id, consent,
                )
            _ = image
            gates, fit = self._run_gates(quality)
            if not fit:
                status = POOR_IMAGE
                failed = [gate.check for gate in gates if not gate.passed]
                status_reason = (
                    "photo quality failed: " + ", ".join(failed) + ". "
                    "No diagnosis could be attempted."
                )
                return self._record(
                    status, status_reason, prediction, gates, quality, stored,
                    crop, plant_part, growth_stage, field_id, zone_id, consent,
                )

            if self.artifact.present:
                # A deployed artifact covers a crop list; anything outside it
                # is the blueprint's unsupported_crop state, not a guess.
                crop_key = (crop or "").strip().lower()
                if self.artifact.crops() and crop_key not in self.artifact.crops():
                    status = UNSUPPORTED_CROP
                    status_reason = (
                        f"The deployed image model covers: "
                        f"{', '.join(self.artifact.crops())}. "
                        f"{crop!r} is not covered, so no class is named."
                    )
                    return self._record(
                        status, status_reason, prediction, gates, quality,
                        stored, crop, plant_part, growth_stage, field_id,
                        zone_id, consent,
                    )

                prediction.task = str(self.artifact.meta.get("task", "disease"))
                prediction.supportedClasses = self.artifact.classes()
                features = np.array(
                    [quality[name] for name in self._feature_order],
                    dtype=np.float32,
                ).reshape(1, -1)
                if hasattr(self.artifact.model, "predict_proba"):
                    probs = self.artifact.model.predict_proba(features)[0]
                    best = int(np.argmax(probs))
                    prediction.rawScore = round(float(probs[best]), 4)
                    confidence = float(probs[best])
                else:
                    best = int(self.artifact.model.predict(features)[0])
                    prediction.rawScore = None
                    confidence = None
                prediction.calibrated = bool(
                    self.artifact.meta.get("calibrated", False)
                )
                if confidence is not None and confidence < ABSTENTION_CONFIDENCE:
                    # The blueprint's abstention path: a low-confidence score
                    # never reaches the farmer as a label.
                    status = UNCERTAIN
                    status_reason = (
                        f"The model's confidence is {confidence:.2f}, below the "
                        f"abstention threshold {ABSTENTION_CONFIDENCE:.2f}; no "
                        "reliable class is reported."
                    )
                    prediction.classLabel = None
                    prediction.rawScore = round(confidence, 4)
                    return self._record(
                        status, status_reason, prediction, gates, quality,
                        stored, crop, plant_part, growth_stage, field_id,
                        zone_id, consent,
                    )
                prediction.classLabel = self.artifact.classes()[best]
                status = RESULT
                status_reason = (
                    "A deployed model returned a supported class. This is not "
                    "a confirmed diagnosis; verify in the field."
                )
            else:
                status = UNCERTAIN
                status_reason = (
                    "The photo passed the quality checks, but no validated "
                    "disease or pest model is deployed for it, so no class "
                    "can be named. Status: pending."
                )
        except Exception:  # pragma: no cover - defensive: log + processing_error
            logger.exception("crop-health analysis failed")
            status = PROCESSING_ERROR
            status_reason = "The analysis raised unexpectedly; see the server log."

        source = ADVICE.get(status, ADVICE["processing_error"])
        return self._record(
            status, status_reason, prediction, gates, quality, stored,
            crop, plant_part, growth_stage, field_id, zone_id, consent, source,
        )

    def _record(
        self,
        status: str,
        status_reason: str,
        prediction: CropHealthPredictionOut,
        gates: list[CropHealthGateOut],
        quality: dict[str, Any],
        stored: Any,
        crop: str,
        plant_part: str | None,
        growth_stage: str | None,
        field_id: str | None,
        zone_id: str | None,
        consent: bool,
        source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The minimum analysis record, exactly as the blueprint lists it."""
        advice_key = status if status in ADVICE else "processing_error"
        source = source or ADVICE[advice_key]
        return {
            "created_at": datetime.now(timezone.utc)
            .isoformat(timespec="seconds"),
            "crop": crop,
            "plant_part": plant_part,
            "growth_stage": growth_stage,
            "field_id": field_id,
            "zone_id": zone_id,
            "image_reference": stored.reference if stored else None,
            "image_metadata": stored.as_metadata() if stored else None,
            "quality": quality,
            "gates": [gate.model_dump() for gate in gates],
            "task": prediction.task,
            "prediction_label": prediction.classLabel,
            "raw_score": prediction.rawScore,
            "calibrated": prediction.calibrated,
            "supported_classes": prediction.supportedClasses,
            "result_status": status,
            "status_reason": status_reason,
            "model_version": self.artifact.model_version,
            "preprocessing_version": PREPROCESSING_VERSION,
            "validation_status": self.artifact.validation_status,
            "advice_code": source["adviceCode"],
            "advice_text": source["adviceText"],
            "advice_text_version": ADVICE_VERSION,
            "advice_follow_up": source["followUpQuestion"],
            "advice_expert_referral": source["expertReferral"],
            "consent_given": bool(consent),
        }
