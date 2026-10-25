"""
Approved water plans: the farmer taps approve, the node's servo acts.

The loop has four states, and this file runs them against the real app on a
temp database: approve (the endpoint rebuilds the advisory itself, so the
node receives the engine's numbers, not the browser's), collect (the node's
next 1 Hz POST response carries the command), claim (the node reports its
valve open, which stops the re-issue), and done (the valve closes). The
fifth state — expired — is the sweep: a command nobody collected dies when
the approved window does, so nothing waters without a farmer watching.

The advisory behind the HTTP tests is a patched pipeline advisory: the real
model answers honestly for a synthetic probe, and honestly for a probe is
"no water owed", which is a refusal, not a loop. The probe field keeps the
404/refusal paths real while the repository-level tests own the loop.

The database is redirected to a temp file because a real node posts at 1 Hz
to a live server; these tests must never share its rows or race them.
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

PROBE_FIELD = "probe-approvals"
ZONE = "Probe-Approvals"


def _probe_advisory(field_id: str) -> dict:
    """A small advisory the two engine-shaped fields must carry."""
    if field_id != PROBE_FIELD:
        raise KeyError(field_id)
    return {
        "fieldId": PROBE_FIELD,
        "crop": "Wheat",
        "areaM2": 4000.0,
        "pumpRateLpm": 12.0,
        "recommendation": {
            "shouldIrrigate": True,
            "depthMm": 18.0,
            "finalDepthMm": 18.0,
            "volumeLitres": 72.0,
            "durationSeconds": 12120,
            "action": "MODERATE_IRRIGATION",
            "confidence": 0.9,
            "probability": 0.91,
            "threshold": 0.5,
        },
        "sensors": {"soilMoisture": 31.5},
        "status": "schedule",
    }


def _approve_probe(monkeypatch) -> None:
    """Point the advisor dependency at the probe advisory for this test."""
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is not None:
        monkeypatch.setattr(pipeline, "advisory", _probe_advisory, raising=False)
    else:
        pytest.skip("advisor pipeline is not started")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """One lifespan, with sowings and the database both redirected to temp.

    `get_database` caches its singleton process-wide, so the fixture swaps it
    for a temp-backed Database and restores the previous one afterwards — a
    module whose tests read the project database must still see it.
    """
    import app.core.database as database_module

    tmp_dir = tmp_path_factory.mktemp("approvals")
    fresh = database_module.Database(f"sqlite:///{(tmp_dir / 'approvals.db').as_posix()}")
    fresh.initialise()

    registry = tmp_dir / "sowings.json"
    source = ROOT / "data" / "sowings.json"
    if source.exists():
        shutil.copy(source, registry)
    else:
        registry.write_text(json.dumps({"version": 1, "fields": []}), encoding="utf-8")

    previous_db = database_module._database
    database_module._database = fresh
    import app.main as main_module

    original_sowings = main_module.settings.sowings_path
    main_module.settings.sowings_path = registry
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        database_module._database = previous_db
        main_module.settings.sowings_path = original_sowings


_REGISTRY_FILE = ROOT / "data" / "sowings.json"
_REGISTRY_TEXT = (
    _REGISTRY_FILE.read_text(encoding="utf-8") if _REGISTRY_FILE.exists() else None
)


def teardown_module() -> None:
    """The compat surface must never leave the project registry modified."""
    if _REGISTRY_FILE is not None:
        assert _REGISTRY_FILE.read_text(encoding="utf-8") == _REGISTRY_TEXT


def _telemetry_payload(zone: str, valve_open: bool | None = None) -> dict:
    """One bench-node-shaped reading, with the valve state selectable."""
    payload = {
        "device_id": "node-test-approvals",
        "zone": zone,
        "temperature": 27.4,
        "humidity": 60.1,
        "soil_moisture": 31.5,
        "soil_ph": 6.7,
        "ec": 1.2,
        "nitrogen": 45.0,
        "phosphorus": 30.0,
        "potassium": 55.0,
        "light_intensity": 12000.0,
        "rainfall": 0.0,
        "wind_speed": 1.5,
        "battery_volts": 3.95,
        "rssi_dbm": -60,
        "firmware": "test",
        "source": "test",
    }
    if valve_open is not None:
        payload["valve_open"] = valve_open
    return payload


def _repository():
    from app.core.database import get_database
    from app.db.repository import ApprovalRepository

    return ApprovalRepository(get_database())


def test_the_endpoint_approves_the_engine_s_own_numbers(client, monkeypatch):
    """
    Approving a wet plan answers with the advisory's depth, seconds and
    volume — rebuilt server-side, so the node can never be handed numbers a
    client invented — and the row opens with status `open`.
    """
    _approve_probe(monkeypatch)
    approval = client.post(f"/api/advisor/{PROBE_FIELD}/approve")
    assert approval.status_code == 200, approval.text
    record = approval.json()
    assert record["status"] == "open"
    assert record["fieldId"] == PROBE_FIELD
    assert record["depthMm"] == 18.0
    assert record["durationSeconds"] == 12120
    assert record["volumeLitres"] == 72.0
    assert record["pumpRateLpm"] == 12.0

    # And the single-field advisory carries the live approval on the next read.
    mirrored = client.get(f"/api/advisor/{PROBE_FIELD}").json()
    assert mirrored["approval"]["approvalId"] == record["approvalId"]
    assert mirrored["approval"]["status"] == "open"


def test_approving_a_dry_plan_is_refused(client, monkeypatch):
    """A plan that calls for no water has nothing to run; the answer is 409."""
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        pytest.skip("advisor pipeline is not started")
    dry = dict(_probe_advisory(PROBE_FIELD))
    dry["recommendation"] = {
        "shouldIrrigate": False,
        "depthMm": 0.0,
        "volumeLitres": 0.0,
        "durationSeconds": 0,
        "action": None,
        "confidence": 0.9,
        "probability": 0.1,
        "threshold": 0.5,
    }

    def advisory_for(field_id: str) -> dict:
        return dry if field_id == PROBE_FIELD else _probe_advisory(field_id)

    monkeypatch.setattr(pipeline, "advisory", advisory_for, raising=False)
    response = client.post(f"/api/advisor/{PROBE_FIELD}/approve")
    assert response.status_code == 409
    assert "no water today" in response.json()["detail"]


def test_unknown_field_cannot_be_approved(client, monkeypatch):
    """The 404 is the service's own — no advisory, no approval, ever."""
    _approve_probe(monkeypatch)
    response = client.post("/api/advisor/no-such-field/approve")
    assert response.status_code == 404


def test_the_node_collects_claims_and_completes_the_approved_run(client):
    """
    The full servo loop at the node's own 1 Hz rhythm: the open approval
    rides the POST response as a command; the node's own valve report claims
    it (re-issuing stops); its closing report finishes the run — and the
    audit trail keeps every step in the event log.
    """
    from app.core.database import get_database
    from app.db.repository import EventRepository

    # An approval for this test's zone, exactly as the endpoint would make
    # it: engine numbers, not client numbers.
    record = _repository().create(
        PROBE_FIELD,
        depth_mm=18.0,
        duration_seconds=120,
        volume_litres=72.0,
        pump_rate_lpm=None,
        zone=ZONE,
    )

    # 1. A fresh-and-ordinary POST carries the command.
    first = client.post(
        "/api/telemetry", json=_telemetry_payload(ZONE)
    ).json()
    assert first["command"]["action"] == "irrigate"
    assert first["command"]["approvalId"] == record["approvalId"]
    assert first["command"]["durationSeconds"] == 120
    assert first["command"]["minutes"] == 2

    # 2. The node opened its valve inside the window: claimed, order stops.
    claims = client.post(
        "/api/telemetry", json=_telemetry_payload(ZONE, valve_open=True)
    ).json()
    assert "command" not in claims
    stored = _repository().by_approval_id(record["approvalId"])
    assert stored is not None and stored["status"] == "running"
    assert stored["claimedAt"] is not None

    # 3. Valve closes: the approved run finished.
    client.post(
        "/api/telemetry", json=_telemetry_payload(ZONE, valve_open=False)
    )
    stored = _repository().by_approval_id(record["approvalId"])
    assert stored is not None and stored["status"] == "done"
    assert stored["closedAt"] is not None

    kinds = [
        event["kind"]
        for event in EventRepository(get_database()).list(limit=50)
        if event["zone"] == ZONE
    ]
    # The claim and completion are the node's doings; both belong in the log.
    assert "plan-running" in kinds
    assert "plan-done" in kinds


def test_a_command_only_speaks_for_its_own_zone(client):
    """
    The command goes out the same way the reading comes in: zone-scoped. A
    node posting for a different field never inherits another field's
    approved run, so no order can travel to soil it was not made for.
    """
    _repository().create(
        PROBE_FIELD,
        depth_mm=18.0,
        duration_seconds=120,
        volume_litres=72.0,
        pump_rate_lpm=None,
        zone="some-other-zone",
    )
    response = client.post(
        "/api/telemetry", json=_telemetry_payload(ZONE)
    ).json()
    assert "command" not in response, (
        "an order approved for another field must not ride this node's response"
    )


def test_a_newer_approval_supersedes_an_open_one(client):
    """
    Tapping approve twice never double-commands the servo: the open row is
    superseded (kept for audit) and the node collects only the newest order.
    """
    first = _repository().create(
        PROBE_FIELD,
        depth_mm=18.0,
        duration_seconds=120,
        volume_litres=72.0,
        pump_rate_lpm=None,
        zone=ZONE,
    )
    second = _repository().create(
        PROBE_FIELD,
        depth_mm=18.0,
        duration_seconds=240,
        volume_litres=72.0,
        pump_rate_lpm=None,
        zone=ZONE,
    )
    assert _repository().by_approval_id(first["approvalId"])["status"] == "superseded"
    assert _repository().by_approval_id(second["approvalId"])["status"] == "open"

    response = client.post(
        "/api/telemetry", json=_telemetry_payload(ZONE)
    ).json()
    assert response["command"]["approvalId"] == second["approvalId"]
    assert response["command"]["durationSeconds"] == 240


def test_an_unclaimed_command_expires_with_its_window(client):
    """
    A command nobody collected is not a standing order: past the approved
    duration plus the claim grace the sweep expires it and no response
    carries it any further. The farmer approves again to run again.
    """
    record = _repository().create(
        PROBE_FIELD,
        depth_mm=18.0,
        duration_seconds=120,
        volume_litres=72.0,
        pump_rate_lpm=None,
        zone=ZONE,
    )
    # The row was created now; rewind it past the whole approved window plus
    # grace, which is exactly the sweep's own arithmetic.
    with _repository().db.transaction() as cursor:
        cursor.execute(
            "UPDATE plan_approvals SET created_at = ? WHERE approval_id = ?",
            ("2020-01-01T00:00:00+00:00", record["approvalId"]),
        )
    response = client.post(
        "/api/telemetry", json=_telemetry_payload(ZONE)
    ).json()
    assert "command" not in response, "an expired approval must not command the node"
    stored = _repository().by_approval_id(record["approvalId"])
    assert stored is not None and stored["status"] == "expired"
