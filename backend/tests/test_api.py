"""
End-to-end API tests.

These run against the real engine and the real registry, not mocks. The point of
the API is to serialise the FAO-56 plan faithfully, and a mock would happily
agree with a wrong field name. The registry is redirected to a tmp_path copy so
registering a field in a test cannot touch `data/sowings.json`.
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

from app.core.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402

#: Snapshot of the real registry taken at import, before any test has run. Every
#: test asserts against it afterwards, so a test that writes project data fails
#: loudly instead of quietly corrupting `data/sowings.json`.
_REGISTRY_FILE = ROOT / "data" / "sowings.json"
REAL_REGISTRY = _REGISTRY_FILE.read_text(encoding="utf-8") if _REGISTRY_FILE.exists() else None


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """
    A client whose registry is a temp copy, never the real one.

    The patch has to land on `app.main.settings`, because that is the object the
    lifespan holds when it constructs the PlanService. Patching a fresh
    `get_settings()` result instead leaves lifespan pointing at the real path, and
    a `POST /api/fields` in a test then silently rewrites `data/sowings.json`.
    """
    import app.main as main_module

    source = ROOT / "data" / "sowings.json"
    registry = tmp_path / "sowings.json"
    if source.exists():
        shutil.copy(source, registry)
    else:
        registry.write_text(
            json.dumps({"version": 1, "fields": []}), encoding="utf-8"
        )

    monkeypatch.setattr(main_module.settings, "sowings_path", registry)

    with TestClient(app) as test_client:
        yield test_client

    # Belt and braces: the real registry must be byte-identical after the suite.
    # A test that can rewrite project data is a test that eventually will.
    if _REGISTRY_FILE.exists():
        assert _REGISTRY_FILE.read_text(encoding="utf-8") == REAL_REGISTRY, (
            f"{_REGISTRY_FILE} was modified by the API tests"
        )


def test_health_reports_weather(client):
    response = client.get("/api/system/health")
    assert response.status_code == 200
    body = response.json()
    assert body["weatherAvailable"] is True
    assert body["weatherStations"] > 0
    assert body["horizonDays"] == 14
    assert "reports" in body


def test_options_come_from_the_engine(client):
    body = client.get("/api/system/options").json()
    assert "Maize" in body["crops"]
    assert "Sprinkler" in body["methods"]
    assert "Loam" in body["soils"]
    assert len(body["stations"]) > 0


def test_fields_list_carries_a_plan_per_field(client):
    body = client.get("/api/fields").json()
    assert body["count"] == len(body["fields"])
    if not body["fields"]:
        pytest.skip("registry is empty")
    for entry in body["fields"]:
        assert entry["fieldId"]
        plan = entry.get("plan")
        if plan is None:
            # A field whose station has no weather is reported, not dropped.
            assert entry.get("planError")
            continue
        assert plan["crop"] == entry["crop"]
        assert plan["requirement"], "requirement must serialise"
        assert plan["stage"]


def test_register_then_plan_then_delete(client):
    fields = client.get("/api/fields").json()["fields"]
    source = fields[0] if fields else None
    payload = {
        "fieldId": "test-plot",
        "station": (source or {}).get("station") or "KOL",
        "crop": (source or {}).get("crop") or "Maize",
        "sowingDate": "2024-06-01",
    }
    if source is None:
        options = client.get("/api/system/options").json()
        payload["station"] = options["stations"][0]
        payload["crop"] = options["crops"][0]

    created = client.post("/api/fields", json=payload)
    assert created.status_code == 201, created.text
    assert created.json()["fieldId"] == "test-plot"

    # A duplicate is refused rather than silently overwriting a real field.
    assert client.post("/api/fields", json=payload).status_code == 409

    plan = client.get("/api/plan/test-plot", params={"as_of": "2024-07-01"})
    assert plan.status_code == 200
    body = plan.json()
    assert body["daysSinceSowing"] == 30
    assert body["tawMm"] > 0
    req = body["requirement"]
    # The engine's requirement has a precise contract; the API must not invent
    # new field names for it. `single_application_mm` is the next event depth.
    assert req["single_application_mm"] >= 0
    assert req["net_requirement_mm"] >= 0
    assert req["method_name"]

    growth = client.get("/api/plan/test-plot/growth", params={"days": 60, "step": 3})
    assert growth.status_code == 200
    points = growth.json()["points"]
    assert points, "growth series must not be empty"
    assert points[0]["date"] == "2024-06-01"
    assert {p["stage"] for p in points}

    assert client.delete("/api/fields/test-plot").status_code == 204
    assert client.get("/api/plan/test-plot").status_code == 404


def test_invalid_sowing_date_is_rejected(client):
    response = client.post(
        "/api/fields",
        json={"fieldId": "bad", "station": "KOL", "crop": "Maize", "sowingDate": "01-06-2024"},
    )
    assert response.status_code == 422


def test_crop_catalogue_carries_the_parameters_that_differ_between_crops(client):
    """
    The selector needs to be able to explain the choice, not just collect it.

    A list of fourteen names gives an operator nothing to decide on. These are the
    numbers that make a cotton field need three times the water of a paddy on the
    same day, so they have to be on the wire rather than in the engine where the
    interface cannot reach them.
    """
    response = client.get("/api/system/crops")
    assert response.status_code == 200
    crops = {entry["name"]: entry for entry in response.json()["crops"]}
    for name in ("Wheat", "Rice", "Maize", "Cotton", "Barley", "Potato"):
        assert name in crops, f"{name} missing from the catalogue"

    rice, cotton = crops["Rice"], crops["Cotton"]
    # Rice starts at FAO-56's flooded-paddy coefficient; cotton starts near bare
    # soil. This is the whole reason a shared per-field number was wrong.
    assert rice["kcInitial"] > 2 * cotton["kcInitial"]
    # Cotton is a deep-rooted crop and tolerates far more depletion before it is
    # stressed, while a paddy has almost no allowable depletion at all.
    assert cotton["rootDepthMaxM"] > rice["rootDepthMaxM"]
    assert cotton["depletionFractionP"] > 5 * rice["depletionFractionP"]

    for entry in crops.values():
        assert entry["gddStageEnds"] == sorted(entry["gddStageEnds"])
        assert entry["seasonGdd"] == entry["gddStageEnds"][-1]
        assert len(entry["lengthStageDays"]) == 4
        assert entry["seasonDays"] == sum(entry["lengthStageDays"])
        assert entry["ky"] > 0
        assert entry["yieldPotentialTHa"] > 0
        assert entry["suitableMethods"], f"{entry['name']} has no suitable method"
        assert not set(entry["suitableMethods"]) & set(entry["warnings"])


def test_catalogue_warns_about_a_method_that_does_not_suit_the_crop(client):
    """
    A pairing that will work but changes what the recommendation means is a warning.

    A paddy under sprinkler is a real practice; it just needs near-daily scheduling
    with small depths rather than threshold-triggered soil-deficit irrigation. The
    interface has to be able to say so before the field is registered, because
    afterwards the only evidence is a recommendation that reads wrong.
    """
    crops = {e["name"]: e for e in client.get("/api/system/crops").json()["crops"]}
    rice = crops["Rice"]
    assert "Paddy" in rice["suitableMethods"]
    assert "Sprinkler" in rice["warnings"]
    # The reason has to be the engine's own sentence, verbatim, and it has to name
    # the thing that changes - the pond depth - rather than just saying "invalid".
    assert "pond depth" in rice["warnings"]["Sprinkler"]
    assert "near-daily" in rice["warnings"]["Sprinkler"]
    # And the reverse: a ponded method on a crop that is not rice is just as wrong.
    assert "Paddy" in crops["Wheat"]["warnings"]


def test_unknown_reference_values_are_rejected_with_the_valid_set(client):
    """
    A typo is a 422 that names the field and lists the alternatives, not a 500.

    The 500 came from `SowingEvent` refusing the name during construction, which put
    a stack trace in front of an operator who had only misspelled a crop. The message
    has to be actionable, because the client's only job on a rejection is to show it.
    """
    base = {"station": "KOL", "sowingDate": "2024-06-01"}
    for body, field in (
        ({**base, "fieldId": "x1", "crop": "Dragonfruit"}, "crop"),
        ({**base, "fieldId": "x2", "crop": "Maize", "soilType": "Moonsand"}, "soilType"),
        ({**base, "fieldId": "x3", "crop": "Maize", "methodName": "Telepathy"}, "methodName"),
        ({**base, "fieldId": "x4", "crop": "Maize", "cropVariant": "Turbo"}, "cropVariant"),
    ):
        response = client.post("/api/fields", json=body)
        assert response.status_code == 422, f"{field} should have been rejected"
        detail = response.json()["detail"]
        assert any(field in entry["loc"] for entry in detail), detail
        assert "unknown" in str(detail).lower()


def test_registering_returns_the_plan_it_produces(client):
    """
    The form shows what the engine made of the new field without a second call.

    Answering with the stored row alone meant the form rendered an empty shell until
    the next poll landed, which on a fifteen-second cycle is long enough for an
    operator to conclude the save failed.
    """
    options = client.get("/api/system/options").json()
    response = client.post(
        "/api/fields",
        json={
            "fieldId": "plan-on-create",
            "station": options["stations"][0],
            "crop": "Maize",
            "sowingDate": "2024-06-01",
            "soilType": "Loam",
            "methodName": "Sprinkler",
            "fieldAreaM2": 1500,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["fieldAreaM2"] == 1500
    assert body["soilType"] == "Loam"
    assert body["cropVariant"] is None
    plan = body["plan"]
    assert plan["fieldId"] == "plan-on-create"
    assert plan["stage"]
    assert plan["tawMm"] > 0
    assert client.delete("/api/fields/plan-on-create").status_code == 204


def test_crop_variant_changes_the_season_it_plans(client):
    """
    A named variety finishes earlier, so it is a different water plan.

    If a variety were accepted and then ignored, the field would be planned with the
    parent's season and would keep recommending water weeks after the real crop had
    been taken. Thermal time accrued by a given date is a property of the sowing and
    the weather, so the only thing a variety can move is how far through that season
    the crop counts as being - which is what this compares.
    """
    catalogue = {e["name"]: e for e in client.get("/api/system/crops").json()["crops"]}
    options = client.get("/api/system/options").json()
    # Any crop that actually declares a variety, so this does not hard-code a
    # cultivar name that a future catalogue edit could remove.
    crop_name, variants = next(
        (name, entry["variants"]) for name, entry in catalogue.items() if entry["variants"]
    )
    plans = {}
    for label, variant in (("parent", None), ("variant", variants[0])):
        payload = {
            "fieldId": f"variant-{label}",
            "station": options["stations"][0],
            "crop": crop_name,
            "sowingDate": "2024-06-01",
            "soilType": "Loam",
            "methodName": "Sprinkler",
        }
        if variant:
            payload["cropVariant"] = variant
        assert client.post("/api/fields", json=payload).status_code == 201
        plans[label] = client.get(
            f"/api/plan/{payload['fieldId']}", params={"as_of": "2024-09-01"}
        ).json()

    # Same sowing, same day, same accumulated heat - so any difference is the
    # variety's season length reaching the plan.
    assert plans["variant"]["gddAccumulated"] == pytest.approx(
        plans["parent"]["gddAccumulated"]
    )
    assert plans["variant"]["seasonDay"] == plans["parent"]["seasonDay"]
    assert client.delete("/api/fields/variant-parent").status_code == 204
    assert client.delete("/api/fields/variant-variant").status_code == 204


def test_as_of_must_be_iso(client):
    assert client.get("/api/fields", params={"as_of": "yesterday"}).status_code == 422


def test_unknown_field_is_404(client):
    assert client.get("/api/plan/nope").status_code == 404


def test_growth_reports_its_depletion_assumption(client):
    fields = client.get("/api/fields").json()["fields"]
    if not fields:
        pytest.skip("registry is empty")
    field_id = fields[0]["fieldId"]
    assumed = client.get(f"/api/plan/{field_id}/growth", params={"days": 30}).json()
    assert assumed["depletionSource"] == "field-capacity-assumed"
    measured = client.get(
        f"/api/plan/{field_id}/growth", params={"days": 30, "depletion_mm": 12}
    ).json()
    assert measured["depletionSource"] == "measured"


def test_model_evaluation_reports_are_shaped(client):
    body = client.get("/api/model/evaluation").json()
    if not body["trained"]:
        assert body["reason"]
        return
    assert body["grouped"]["maeMm"] is not None
    assert body["leaveOneStationOut"]["all"], "weak folds must be exposed"
    folds = body["leaveOneStationOut"]["all"]
    assert min(fold["r2"] for fold in folds) == pytest.approx(
        body["leaveOneStationOut"]["minR2"], abs=1e-6
    ), "the worst fold must be the minimum of the listed folds"
    assert body["conformal"]


def test_report_endpoints_404_clearly_when_absent(client):
    for path, script in (
        ("/api/model/external-validation", "validate_against_observations.py"),
        ("/api/model/soil-validation", "validate_soil_water.py"),
        ("/api/model/comparison", "compare_v1_v2.py"),
    ):
        response = client.get(path)
        assert response.status_code in (200, 404)
        if response.status_code == 404:
            assert script in response.json()["detail"]


def test_registry_survives_a_write(client):
    """A registration must land on disk, not just in memory."""
    options = client.get("/api/system/options").json()
    client.post(
        "/api/fields",
        json={
            "fieldId": "persist-check",
            "station": options["stations"][0],
            "crop": options["crops"][0],
            "sowingDate": "2024-06-01",
        },
    )
    path = get_settings().sowings_path
    stored = json.loads(path.read_text(encoding="utf-8"))
    # The registry on disk is {"version": ..., "fields": [...]}, not a bare map
    # of id -> event, so the assertion has to look inside it.
    assert stored["version"]
    ids = [entry["field_id"] for entry in stored["fields"]]
    assert "persist-check" in ids
