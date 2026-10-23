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
