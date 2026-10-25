"""
Crop-health API tests.

Every status the blueprint names is exercised here against the real service:
valid upload (no model deployed → honest `uncertain`), poor image, invalid
input, unsupported type, consent behaviour, persistence, retention (DELETE)
and the manifest endpoint.

The image bytes are synthesised with Pillow so the suite never depends on a
binary fixture sitting in git.
"""

from __future__ import annotations

import io
import random
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "backend"))

import app.core.database as database_module  # noqa: E402
from app.core.database import Database  # noqa: E402
from app.main import app  # noqa: E402


def _sharp_photo() -> bytes:
    """A 640x480 photo full of texture: sharp, mid-lit, gate-passing."""
    random.seed(7)
    # Deterministic noise makes the Laplacian variance high and the mean luma
    # comfortable, so the gates pass and the request reaches the model slot.
    image = Image.frombytes(
        "RGB", (640, 480),
        bytes(random.randrange(60, 200) for _ in range(640 * 480 * 3)),
    )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _dark_flat_photo() -> bytes:
    """A uniformly dark, texture-free square: fails blur/exposure/flat."""
    image = Image.new("RGB", (400, 400), (12, 12, 12))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import app.main as main_module

    db_path = tmp_path / "crop_health.db"
    monkeypatch.setattr(
        database_module, "_database",
        Database(f"sqlite:///{db_path.as_posix()}"),
    )
    images_dir = tmp_path / "images"
    main_module.settings.crop_health_images_dir = images_dir

    # Point the artifact loader at an empty directory so tests exercise the
    # no-deployed-model service exactly as written; a repo checkout that has a
    # model deployed must not flip these expectations.
    models_dir = tmp_path / "models"
    main_module.settings.models_dir = models_dir

    with TestClient(app) as test_client:
        yield test_client


def test_status_reports_no_deployed_model(client):
    response = client.get("/api/crop-health/status")
    assert response.status_code == 200
    body = response.json()
    assert body["deployedModel"] is False
    assert "pending" in body["statusLabels"]["model"]
    assert body["supportedClasses"] == []


def test_valid_upload_is_honest_uncertain_and_persisted(client):
    response = client.post(
        "/api/crop-health/analyze",
        files={"file": ("leaf.jpg", _sharp_photo(), "image/jpeg")},
        data={"crop": "Rice", "field_id": "Rice-A", "consent": "true"},
    )
    assert response.status_code == 200
    body = response.json()
    # No validated image model exists — nothing may pretend otherwise.
    assert body["resultStatus"] == "uncertain"
    assert body["prediction"]["classLabel"] is None
    assert "no validated" in body["statusReason"].lower()
    assert body["modelVersion"] == "none-deployed"
    assert body["advice"]["expertReferral"] is True
    # The record and its image are real.
    analysis_id = body["analysisId"]
    assert analysis_id
    listing = client.get("/api/crop-health/analyses").json()
    assert listing["count"] == 1
    assert listing["analyses"][0]["analysisId"] == analysis_id
    reference = body["imageReference"]
    stored = Path(client.app.state.settings.crop_health_images_dir) / reference
    assert stored.exists()


def test_no_consent_means_no_retained_image(client):
    response = client.post(
        "/api/crop-health/analyze",
        files={"file": ("leaf.jpg", _sharp_photo(), "image/jpeg")},
        data={"crop": "Rice", "consent": "false"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body.get("imageReference") in (None, "")
    images_dir = Path(client.app.state.settings.crop_health_images_dir)
    stored_files = list(images_dir.rglob("*.jpg")) if images_dir.exists() else []
    assert stored_files == []


def test_dark_flat_photo_returns_poor_image(client):
    response = client.post(
        "/api/crop-health/analyze",
        files={"file": ("leaf.jpg", _dark_flat_photo(), "image/jpeg")},
        data={"crop": "Rice"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["resultStatus"] == "poor_image"
    failed = {gate["check"] for gate in body["gates"] if not gate["passed"]}
    assert "blur" in failed and "exposure" in failed
    assert body["advice"]["adviceCode"] == "retake_photo"


def test_unsupported_type_is_recorded_invalid_input(client):
    response = client.post(
        "/api/crop-health/analyze",
        files={"file": ("notes.txt", b"definitely not an image", "text/plain")},
        data={"crop": "Rice"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["resultStatus"] == "invalid_input"


def test_empty_upload_is_invalid(client):
    response = client.post(
        "/api/crop-health/analyze",
        files={"file": ("leaf.jpg", b"", "image/jpeg")},
        data={"crop": "Rice"},
    )
    assert response.status_code == 200
    assert response.json()["resultStatus"] == "invalid_input"


def test_missing_file_rejected(client):
    response = client.post("/api/crop-health/analyze", data={"crop": "Rice"})
    assert response.status_code == 422


def test_delete_removes_record_and_image(client):
    created = client.post(
        "/api/crop-health/analyze",
        files={"file": ("leaf.jpg", _sharp_photo(), "image/jpeg")},
        data={"crop": "Wheat", "consent": "true"},
    ).json()
    reference = created["imageReference"]
    images_dir = Path(client.app.state.settings.crop_health_images_dir)

    deleted = client.delete(f"/api/crop-health/analyses/{created['analysisId']}")
    assert deleted.status_code == 200
    body = deleted.json()
    assert body["deleted"] is True and body["imageDeleted"] is True
    assert not (images_dir / reference).exists()
    assert client.get(f"/api/crop-health/analyses/{created['analysisId']}").status_code == 404
    assert client.get("/api/crop-health/analyses").json()["count"] == 0


def test_feedback_is_stored(client):
    created = client.post(
        "/api/crop-health/analyze",
        files={"file": ("leaf.jpg", _sharp_photo(), "image/jpeg")},
        data={"crop": "Wheat", "consent": "true"},
    ).json()
    response = client.post(
        f"/api/crop-health/analyses/{created['analysisId']}/feedback",
        params={"comment": "No disease visible after two days"},
    )
    assert response.status_code == 200
    assert response.json()["farmerFeedback"] == "No disease visible after two days"


def test_manifest_is_served_with_pending_image_collections(client):
    response = client.get("/api/crop-health/manifest")
    assert response.status_code == 200
    manifest = response.json()
    assert manifest["tabularDatasets"], "existing CSV corpora must be listed"
    pending = [c for c in manifest["imageCollections"] if c["status"] == "pending"]
    assert pending, "the image work must be visible as pending, not hidden"
    assert manifest["imageModelStatus"]["deployed"] is False


def test_advisory_carries_the_safety_block(client):
    """The safety layer must appear on the dashboard-facing advisory feed."""
    response = client.get("/api/advisor")
    assert response.status_code == 200
    feed = response.json()
    fields = feed.get("fields") or []
    assert fields
    assert all("safety" in field for field in fields)
    assert all(field["safety"]["applied"] for field in fields)


def test_a_named_class_carries_its_nutrition_word(tmp_path):
    """
    A named photo read means something for the fertiliser plan, not just the
    spray can. The mapping is the same general practice the treatment text
    distils, structured so the advisory's fertiliser chip can act on it; the
    table is audited here so a wording edit cannot quietly break the contract.
    """
    from app.schemas_crop_health import CropHealthPredictionOut
    from app.services.crop_health_service import (
        DISEASE_NUTRITION,
        TREATMENTS,
        CropHealthService,
    )

    assert set(DISEASE_NUTRITION) <= set(TREATMENTS), (
        "a nutrition word must belong to a class that also carries treatment text"
    )
    for entry in DISEASE_NUTRITION.values():
        assert entry["nutrient"] in {"nitrogen", "phosphorus", "potassium"}
        assert entry["action"] in {"withhold", "reduce", "check", "balance", "steady"}
        assert entry["note"].strip(), "the word carries the agronomy; no empty notes"

    # A real service on empty paths: no artifact, no store traffic — these
    # records go through _record alone, which is where the word attaches.
    class _Paths:
        crop_health_images_dir = str(tmp_path / "images")
        models_dir = str(tmp_path / "models")

    service = CropHealthService(_Paths())
    named = CropHealthPredictionOut()
    named.classLabel = "Rice Bacterial leaf blight"
    record = service._record(
        "result", "reason", named, [], {}, None,
        "Rice", None, None, None, None, False,
    )
    nutrition = record["treatment"]["nutrition"]
    assert nutrition["nutrient"] == "nitrogen"
    assert nutrition["action"] == "withhold"
    assert nutrition["textVersion"]

    # A class outside the table constrains nothing — a bared disease says no
    # nutrient word, not a guessed one.
    unnamed = CropHealthPredictionOut()
    unnamed.classLabel = "Potato Late blight"
    record = service._record(
        "result", "reason", unnamed, [], {}, None,
        "Potato", None, None, None, None, False,
    )
    assert record["treatment"]["nutrition"] is None

    # No class named → no treatment at all, so nothing for a chip to act on.
    blank = CropHealthPredictionOut()
    blank.classLabel = None
    record = service._record(
        "result", "reason", blank, [], {}, None,
        "Rice", None, None, None, None, False,
    )
    assert record["treatment"] is None


def test_named_results_query_filters_age_and_status(tmp_path):
    """
    Only fresh, named reads are food for the plan: a blight named last month
    is not a blight the crop still carries, and a sub-floor "normal" read
    constrains nothing. `list_named_results` is the gate.
    """
    from app.db.repository import CropHealthRepository

    database = Database(f"sqlite:///{(tmp_path / 'named.db').as_posix()}")
    database.initialise()
    repository = CropHealthRepository(database)

    recent = repository.insert(
        {
            "created_at": "2026-10-24T09:00:00+00:00",
            "field_id": "Live-Field",
            "prediction_label": "Rice Brown spot",
            "result_status": "result",
            "treatment": {"heading": "h", "text": "t", "disclaimer": "d",
                          "textVersion": "treat-en-1",
                          "nutrition": {"nutrient": "potassium", "action": "check",
                                        "note": "n", "textVersion": "nutr-en-1"}},
        }
    )
    older = repository.insert(
        {
            "created_at": "2026-09-01T09:00:00+00:00",
            "field_id": "Live-Field",
            "prediction_label": "Rice Brown spot",
            "result_status": "result",
        }
    )
    # A read that names nothing: it reaches the table but not the query —
    # a "normal" answer never constrains a fertiliser plan.
    repository.insert(
        {
            "created_at": "2026-10-24T10:00:00+00:00",
            "field_id": "Live-Field",
            "prediction_label": None,
            "result_status": "result",
        }
    )

    rows = repository.list_named_results("2026-10-18T00:00:00+00:00")
    ids = [row["analysisId"] for row in rows]
    assert ids == [recent["analysisId"]], (
        "the query must keep exactly the fresh named reads newest-first, "
        f"dropping the obsolete and the unnamed ones, got {ids}"
    )
    # The treatment the service attached travels with the row, so the overlay
    # can hand the fertiliser chip the structured word.
    assert rows[0]["treatment"]["nutrition"]["action"] == "check"


def test_the_advisor_lets_a_fresh_photo_constrain_the_plan(client):
    """
    The photo joins the plan end to end: a named, fresh read for a field in
    the feed rides the advisory as `photoDisease`, carrying its class and the
    structured nutrition word the fertiliser chip acts on — and no photo for
    a field means no constraint for that field, so no chip invents work from
    a photo another field earned.
    """
    from app.db.repository import CropHealthRepository
    from datetime import datetime, timezone

    repository = CropHealthRepository(app.state.health_repository.db)
    field_id = (client.get("/api/advisor").json()["fields"])[0]["fieldId"]
    named = datetime.now(timezone.utc).isoformat(timespec="seconds")
    repository.insert(
        {
            "created_at": named,
            "field_id": field_id,
            "crop": "Rice",
            "prediction_label": "Rice Bacterial leaf blight",
            "result_status": "result",
            "treatment": {"heading": "h", "text": "t", "disclaimer": "d",
                          "textVersion": "treat-en-1",
                          "nutrition": {"nutrient": "nitrogen", "action": "withhold",
                                        "note": "n", "textVersion": "nutr-en-1"}},
        }
    )

    feed = client.get("/api/advisor").json()
    fields = {item["fieldId"]: item for item in feed["fields"]}
    photo = fields[field_id]["photoDisease"]
    assert photo["classLabel"] == "Rice Bacterial leaf blight"
    assert photo["photoOn"] == named
    assert photo["nutrition"]["nutrient"] == "nitrogen"
    assert photo["nutrition"]["action"] == "withhold"
    for other_id, item in fields.items():
        if other_id != field_id:
            assert "photoDisease" not in item, (
                f"{other_id} must not inherit a photo read taken for {field_id}"
            )


def test_the_advisor_ignores_a_photo_beyond_the_freshness_window(client):
    """
    A named read stops constraining the plan after the window: a blight
    named last month is not evidence this week's feed choice answers to.
    The farmer takes a fresh photo; the plan does not age its advice on a
    stale one.
    """
    from app.db.repository import CropHealthRepository

    repository = CropHealthRepository(app.state.health_repository.db)
    field_id = (client.get("/api/advisor").json()["fields"])[0]["fieldId"]
    repository.insert(
        {
            "created_at": "2025-01-01T00:00:00+00:00",
            "field_id": field_id,
            "crop": "Rice",
            "prediction_label": "Rice Bacterial leaf blight",
            "result_status": "result",
            "treatment": None,
        }
    )

    fields = client.get("/api/advisor").json()["fields"]
    constrained = [
        item for item in fields
        if item["fieldId"] == field_id and item.get("photoDisease")
    ]
    assert not constrained, (
        "a read from outside the freshness window must not constrain the plan"
    )
