"""
Tests for the V1-shaped compatibility surface.

These assert the contract the copied dashboard depends on, against the real
engine and the real (temp-copied) registry. The V1 decision bundle is loaded by
the same lifespan, so a passing run also proves the ML bridge resolves the
sibling `MP3/backend/ml` tree.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "backend"))

from app.main import app  # noqa: E402

_REGISTRY_FILE = ROOT / "data" / "sowings.json"
REAL_REGISTRY = _REGISTRY_FILE.read_text(encoding="utf-8") if _REGISTRY_FILE.exists() else None


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """
    One app lifespan for the whole module.

    The compat surface is expensive to warm (ten years of weather plus the V1
    bundle), so a fresh client per test would dominate the suite. The registry is
    still redirected to a temp copy so `acknowledge`-style writes cannot touch
    project data.
    """
    import app.main as main_module

    registry = tmp_path_factory.mktemp("compat") / "sowings.json"
    source = ROOT / "data" / "sowings.json"
    if source.exists():
        shutil.copy(source, registry)
    else:
        registry.write_text(json.dumps({"version": 1, "fields": []}), encoding="utf-8")
    original = main_module.settings.sowings_path
    main_module.settings.sowings_path = registry

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        main_module.settings.sowings_path = original

    if _REGISTRY_FILE.exists():
        assert _REGISTRY_FILE.read_text(encoding="utf-8") == REAL_REGISTRY, (
            f"{_REGISTRY_FILE} was modified by the compat tests"
        )


@pytest.fixture()
def device_id(client):
    devices = client.get("/api/devices").json()["devices"]
    if not devices:
        pytest.skip("registry is empty")
    return devices[0]["id"]


def test_health_reports_the_model_bundle(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["modelsLoaded"] is True
    assert body["deviceCount"] >= 1
    assert body["modelsLoaded"] and body["inferenceLatencyMs"] >= 0.0
    assert "calibration" in body and "depthUncertaintyMm" in body


def test_devices_and_zones_are_shaped(client):
    devices = client.get("/api/devices").json()["devices"]
    assert devices, "the demo registry must list fields"
    for entry in devices:
        for key in ("id", "name", "zone", "crop", "soilType", "linkState", "lastSeen"):
            assert key in entry

    zones = client.get("/api/zones").json()["zones"]
    assert len(zones) == len(devices)
    for entry in zones:
        for key in ("zone", "crop", "deviceId", "soilMoisture", "soilPh", "nitrogen"):
            assert key in entry


def test_latest_returns_readings_and_recommendations(client, device_id):
    body = client.get("/api/telemetry/latest", params={"device_id": device_id}).json()
    assert body["readings"] and body["recommendations"]
    reading = body["readings"][0]
    assert reading["deviceId"] == device_id
    assert 0.0 <= reading["soilMoisture"] <= 100.0
    recommendation = body["recommendations"][0]
    for key in ("deviceId", "shouldIrrigate", "depthMm", "action", "diseaseRisk", "et0MmDay"):
        assert key in recommendation


def test_history_is_scoped_to_the_device(client, device_id):
    body = client.get(
        "/api/telemetry/history", params={"device_id": device_id, "hours": 48}
    ).json()
    assert body["readings"]
    assert all(item["deviceId"] == device_id for item in body["readings"])


def test_recommendation_endpoint_and_missing_device(client, device_id):
    body = client.post(f"/api/devices/{device_id}/recommendation").json()
    assert body["deviceId"] == device_id
    assert body["action"] in {"NO_IRRIGATION", "LIGHT_IRRIGATION", "MODERATE_IRRIGATION", "HEAVY_IRRIGATION"}
    assert client.post("/api/devices/nope/recommendation").status_code == 404


def test_valve_is_refused_with_a_reason(client, device_id):
    response = client.post(f"/api/devices/{device_id}/valve", json={"open": True})
    assert response.status_code == 409
    assert "planner" in response.json()["detail"]
    assert client.post("/api/devices/nope/valve", json={"open": True}).status_code == 404


def test_alerts_shaped_and_acknowledgeable(client):
    body = client.get("/api/alerts", params={"open_only": True, "limit": 50}).json()
    alerts = body["alerts"]
    if not alerts:
        pytest.skip("no open alerts for the demo registry")
    first = alerts[0]
    for key in ("id", "deviceId", "kind", "severity", "message", "acknowledged", "raisedAt"):
        assert key in first
    acked = client.post(f"/api/alerts/{first['id']}/acknowledge")
    assert acked.status_code == 200
    assert acked.json()["acknowledged"] is True
    assert client.post("/api/alerts/999999999/acknowledge").status_code == 404


def test_events_and_analytics_and_drift(client):
    events = client.get("/api/events", params={"limit": 20}).json()["events"]
    assert events
    assert all("createdAt" in item and "kind" in item for item in events)

    overview = client.get("/api/analytics/overview", params={"hours": 72}).json()
    assert overview["window"] == "72h"
    assert "series" in overview and overview["series"]

    drift = client.get("/api/models/drift", params={"hours": 168}).json()
    assert "available" in drift
    if drift["available"]:
        assert "features" in drift


def test_predict_is_refused_with_a_reason(client):
    response = client.post("/api/predict", json={"device_id": "x"})
    assert response.status_code == 409
    assert "sowing" in response.json()["detail"]


def test_advisor_returns_ranked_stage_aware_advice(client):
    body = client.get("/api/advisor").json()

    fleet = body["fleet"]
    assert fleet["fields"] == len(body["fields"]) >= 1
    for name in ("irrigate", "monitor", "hold", "recommendedMm", "recommendedLitres"):
        assert name in fleet

    priorities = [field["priority"] for field in body["fields"]]
    assert priorities == sorted(priorities, reverse=True)

    field = body["fields"][0]
    for key in (
        "fieldId",
        "crop",
        "station",
        "ageDays",
        "stage",
        "stageLabel",
        "stageProgress",
        "gdd",
        "status",
        "priority",
        "headline",
        "summary",
        "recommendation",
        "water",
        "sensors",
        "risk",
        "reasoning",
        "schedule",
    ):
        assert key in field

    assert field["status"] in {"irrigate", "schedule", "monitor", "hold"}
    assert field["ageDays"] >= 0
    assert 0.0 <= field["stageProgress"] <= 1.0
    assert field["reasoning"]
    assert len(field["schedule"]) == 7

    # The season plan the crop-age card is drawn from.
    season = field["season"]
    assert set(season) == {
        "stageWindows",
        "daysInSeason",
        "seasonGrossMm",
        "seasonRainMm",
    }
    for window in season["stageWindows"]:
        assert window["days"] >= 1
        assert window["meanEtcMmDay"] >= 0
        assert window["startDate"] <= window["endDate"]
        assert "stage" in window

    # All seven readings the sensor head reports must reach the payload, since
    # the field board renders them as the primary data on the page.
    sensors = field["sensors"]
    for key in (
        "soilMoisture",
        "temperature",
        "humidity",
        "soilPh",
        "ec",
        "nitrogen",
        "phosphorus",
        "potassium",
    ):
        assert sensors[key] is not None, f"sensor {key} missing"
        assert isinstance(sensors[key], (int, float))
    # The narrative and the numbers are drawn from one plan, so the age in the
    # sentence must be the age in the field.
    assert f"{field['ageDays']} days past sowing" in field["summary"]
    for key in ("et0MmDay", "etcMmDay", "kc", "tawMm", "rawMm", "depletionFraction"):
        assert key in field["water"]


def test_advisor_single_field_and_missing(client, device_id):
    body = client.get(f"/api/advisor/{device_id}").json()
    assert body["fieldId"] == device_id
    assert client.get("/api/advisor/does-not-exist").status_code == 404


def test_advisor_pipeline_reports_both_models(client):
    """The advisor fuses V1's decision bundle with V2's requirement model."""
    body = client.get("/api/advisor").json()

    assert body["pipeline"]["modelPresent"] is True
    assert body["pipeline"]["target"] == "nir_horizon_mm"

    field = body["fields"][0]
    models = field["models"]
    assert set(models) == {"v1", "v2", "fusion"}

    v1 = models["v1"]
    assert isinstance(v1["shouldIrrigate"], bool)
    assert v1["depthMm"] is not None

    v2 = models["v2"]
    assert v2["target"] == "nir_horizon_mm"
    assert v2["modelPresent"] is True

    fusion = models["fusion"]
    for key in ("agreement", "finalStatus", "finalDepthMm", "note"):
        assert key in fusion
    assert fusion["finalStatus"] in {"irrigate", "schedule", "monitor", "hold"}
    assert "finalDepthMm" in field["recommendation"]

    # A depth shown without a matching volume is the bug this guards: the
    # escalation can raise the depth above the decision's, and the litres and
    # runtime are derived from the depth, so all three must agree.
    depth = field["recommendation"]["finalDepthMm"]
    volume = field["recommendation"]["volumeLitres"]
    assert volume == pytest.approx(depth * field["areaM2"], rel=0.02), (
        f"depth {depth} mm over {field['areaM2']} m2 should be "
        f"{depth * field['areaM2']:.0f} litres, got {volume}"
    )
    if depth > 0:
        assert field["recommendation"]["durationSeconds"] > 0
    else:
        assert field["recommendation"]["durationSeconds"] == 0

    labels = [item["label"] for item in field["reasoning"]]
    assert "Model agreement" in labels


def test_advisor_pipeline_holds_back_fields_out_of_season(client, device_id):
    """
    Past harvest there is no requirement to estimate, so none is offered.

    The estimator was fitted on a growing crop. A field whose season has ended
    has a collapsed root profile and a depletion fraction above one, which is
    outside the training envelope; extrapolating it produced 18-46 mm
    recommendations against a physics requirement of zero. The pipeline must
    report the state rather than a number.
    """
    pipeline = client.app.state.pipeline
    snapshot = pipeline.compat._find(device_id)

    import dataclasses

    finished = dataclasses.replace(
        snapshot.current.plan,
        stage="post_harvest",
        stage_progress=1.0,
        depletion_fraction=1.9,
    )
    point = dataclasses.replace(snapshot.current, plan=finished)
    harvested = dataclasses.replace(snapshot, points=[*snapshot.points[:-1], point])

    assert pipeline._applicability(harvested)[0] is False
    estimate = pipeline._estimate(harvested)
    assert estimate["predictionMm"] is None
    assert estimate["intervalMm"] is None

    advisory = {
        "status": "hold",
        "recommendation": {"shouldIrrigate": False, "depthMm": 0.0},
    }
    fusion = pipeline._fuse(advisory, estimate)
    assert fusion["agreement"] == "not_applicable"
    # Held back, not escalated: no water is invented for a harvested field.
    assert fusion["finalStatus"] == "hold"
    assert fusion["finalDepthMm"] == 0.0


def test_pipeline_runs_the_requirement_model_in_season(client, device_id):
    """
    In season, the model actually runs and returns a bounded interval.

    Exercised against a real engine plan for a field the registry keeps in
    season, so the prediction path is covered without inventing weather.
    """
    pipeline = client.app.state.pipeline
    snapshot = pipeline.compat._find(device_id)

    assert pipeline._applicability(snapshot)[0] is True

    estimate = pipeline._estimate(snapshot)
    assert estimate["modelPresent"] is True
    assert estimate["applicable"] is True
    assert estimate["predictionMm"] is not None and estimate["predictionMm"] >= 0.0
    low, high = estimate["intervalMm"]
    assert 0.0 <= low <= estimate["predictionMm"] <= high
    assert estimate["coverage"] == 0.9


def test_advisor_fields_get_different_amounts(client):
    """
    Two fields must never be given the same advice for the same reason.

    The seeded registry holds a young paddy, two mid-season cereals and a
    late-season cotton on four different soils and methods. Their depths are
    driven by crop coefficient against days of growth and by how much water the
    soil can hold, so an identical answer across all four would mean one of those
    inputs had stopped reaching the model.
    """
    body = client.get("/api/advisor").json()
    depths = {field["fieldId"]: field["recommendation"]["finalDepthMm"] for field in body["fields"]}
    assert len(body["fields"]) >= 3
    assert len(set(depths.values())) == len(depths), f"depths collided: {depths}"

    ages = {field["fieldId"]: field["ageDays"] for field in body["fields"]}
    assert len(set(ages.values())) == len(ages), f"ages collided: {ages}"

    # The stage drives daily demand, so equal crop coefficients at different ages
    # would mean the season clock had stopped reaching the plan.
    stages = {field["fieldId"]: field["stage"] for field in body["fields"]}
    assert len(set(stages.values())) >= 2, f"every field is in the same stage: {stages}"

    for field in body["fields"]:
        assert field["water"]["etcMmDay"] >= 0
        if field["recommendation"]["finalDepthMm"] > 0:
            assert field["recommendation"]["volumeLitres"] > 0, (
                f"{field['fieldId']} recommends water but reports no volume"
            )

    # A young paddy transpires like a mature crop because rice holds a high crop
    # coefficient from the start; a cereal at the same age does not.
    paddy = next((f for f in body["fields"] if f["crop"] == "Rice"), None)
    if paddy is not None:
        assert paddy["water"]["kc"] >= 1.0, (
            "rice should carry a high crop coefficient even when young, "
            f"got {paddy['water']['kc']}"
        )


def test_fusion_escalates_when_only_the_requirement_model_sees_demand(client):
    pipeline = client.app.state.pipeline
    advisory = {
        "status": "hold",
        "recommendation": {"shouldIrrigate": False, "depthMm": 0.0},
    }
    nir = {
        "modelPresent": True,
        "predictionMm": 20.0,
        "intervalMm": [10.0, 30.0],
        "physicsSingleApplicationMm": 15.0,
    }
    fusion = pipeline._fuse(advisory, nir)
    assert fusion["agreement"] == "v2_only"
    assert fusion["finalStatus"] == "schedule"
    # Escalation is capped by what one application can place.
    assert fusion["finalDepthMm"] == 15.0


def test_fusion_agrees_and_holds_without_escalating(client):
    pipeline = client.app.state.pipeline
    advisory = {
        "status": "irrigate",
        "recommendation": {"shouldIrrigate": True, "depthMm": 12.0},
    }
    aligned = pipeline._fuse(
        advisory,
        {
            "modelPresent": True,
            "predictionMm": 12.0,
            "intervalMm": [5.0, 19.0],
            "physicsSingleApplicationMm": 15.0,
        },
    )
    assert aligned["agreement"] == "aligned"
    assert aligned["finalStatus"] == "irrigate"
    assert aligned["finalDepthMm"] == 12.0

    held = pipeline._fuse(
        {"status": "hold", "recommendation": {"shouldIrrigate": False, "depthMm": 0.0}},
        {
            "modelPresent": True,
            "predictionMm": 0.0,
            "intervalMm": [0.0, 9.0],
            "physicsSingleApplicationMm": 0.0,
        },
    )
    assert held["agreement"] == "agree_hold"
    assert held["finalStatus"] == "hold"


def test_advisor_pipeline_assembles_the_feature_contract(client, device_id):
    """
    The estimator is fed the exact FeatureSpec columns, at serve time.

    A mismatch between the training contract and the served frame is the failure
    that would otherwise only show up as a silently degraded prediction, so the
    pipeline is asserted to produce one row with every trained feature present.
    """
    pipeline = client.app.state.pipeline
    snapshot = pipeline.compat._find(device_id)
    features = pipeline._feature_frame(snapshot)
    assert len(features) == 1
    assert list(features.columns) == list(pipeline._model.feature_names)
    assert features.notna().all().all()
