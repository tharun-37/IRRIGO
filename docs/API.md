# API reference

Base URL in development: `http://127.0.0.1:8000/api`
Interactive documentation: `http://127.0.0.1:8000/docs`

All ingest is under `/api/telemetry` and all control under `/api/devices`, so a firmware
change touches one file and the control surface stays auditable.

Field names in responses are camelCase. The device-facing and dashboard contracts are
JSON objects; the same values are stored in snake_case columns, and the conversion happens
in one named place on each side.

---

## System

### `GET /api/health`

Liveness, model bundle state, device counts and the full measured metric set. Used by the
dashboard header and the model page.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptimeSeconds": 575,
  "modelsLoaded": true,
  "deviceCount": 4,
  "onlineDevices": 4,
  "readingsToday": 64,
  "openAlerts": 2,
  "inferenceLatencyMs": 21.4,
  "calibration": { "expected_calibration_error": 0.0044, "recalibration_slope": 1.2015 },
  "depthUncertaintyMm": 2.2534,
  "irrigateThresholdGroups": 15,
  "modelMetrics": { "disease": {}, "irrigateDecision": {}, "waterDepth": {}, "yield": {}, "nutrient": {}, "latencyMs": {} },
  "waterSource": {
    "available": true,
    "records": 43836,
    "stations": 12,
    "span": { "from": "2015-01-01", "to": "2024-12-31" },
    "et0MeanMmDay": 4.541
  },
  "provenance": { "corpus_rows": 68149, "et0_method": "FAO-56 Penman-Monteith Eq. 6" }
}
```

`inferenceLatencyMs` is the measured cost of one whole decision, feature assembly included.
It is deliberately not the per-estimator figure in `modelMetrics.latencyMs`, which is a
training-time benchmark of one model in isolation and is an order of magnitude smaller;
reporting that as system latency would overstate the system by about 20x.

`calibration`, `depthUncertaintyMm` and `irrigateThresholdGroups` are surfaced at the top
level because they decide whether the served model can currently be trusted, and a client
should not have to know to look inside `provenance` for them.

`provenance` is included so a decision can be traced to the exact data and method that
produced it, rather than to a version string that could be stale. It also carries
`disease_label_caveat` and `disease_score_interpretation`, which state that the disease
model's labels are rule-derived and its scores measure memorisation rather than diagnosis.

### `GET /api/models/drift`

Population stability of live readings against the training distribution. `?hours=` (default
168, max 8760).

```json
{
  "available": true,
  "window_hours": 168,
  "readings_considered": 2609,
  "worst_feature": "temperature",
  "worst_psi": 0.0299,
  "frozen_sensors": [],
  "features": {
    "temperature": {
      "psi": 0.0209,
      "samples": 2609,
      "band": "negligible",
      "reference_mean": 26.9988,
      "reference_p50": 26.4,
      "live_mean": 26.5031
    }
  },
  "interpretation": "PSI below 0.1 is negligible, 0.1 to 0.25 warrants investigation, above 0.25 means the training distribution no longer describes the live data and the model should be retrained. A null PSI means too few readings to judge. A frozen sensor is a hardware fault, not drift."
}
```

A model keeps answering confidently after the field it was fitted on drifts, and nothing in
the decision path can notice, because the model never sees the training data again. This
endpoint is what makes that question answerable, by comparing recent readings against decile
edges recorded on the bundle at training time.

Two deliberate non-answers. A feature with fewer than 50 live readings returns `"psi": null`
rather than a number, because a score from a handful of points is noise that would read as
reassurance. And a probe reporting an identical value on every reading is reported as
`"band": "sensor_frozen"` and excluded from the drift ranking, since a stuck sensor is a
hardware fault that would otherwise dominate `worst_psi` and bury the features that really
moved.

### `GET /api/events`

Append-only audit log. `?limit=` (default 100). Kinds: `command`, `telemetry`, `alert`,
`system`.

---

## Ingest

### `POST /api/telemetry`

Submit one reading. Unknown devices are registered on first sight, so a node can be
commissioned without a separate step.

```json
{
  "device_id": "node-east-01",
  "zone": "East Terrace",
  "recorded_at": "2026-09-28T11:08:00Z",
  "soil_moisture": 13.1,
  "soil_temperature": 24.6,
  "soil_ph": 6.4,
  "ec": 1.12,
  "nitrogen": 42.0,
  "phosphorus": 18.5,
  "potassium": 145.0,
  "air_temperature": 25.8,
  "humidity": 58.0,
  "light_intensity": 12400.0,
  "rainfall": 0.0,
  "battery_volts": 4.05,
  "rssi_dbm": -58
}
```

Every field except `device_id` is optional; missing values are excluded before inference
rather than being imputed as zero.

```json
{
  "accepted": true,
  "readingId": 4711,
  "decisionId": 4702,
  "deviceId": "node-east-01",
  "zone": "East Terrace",
  "shouldIrrigate": false,
  "depthMm": 0.0,
  "durationSeconds": 0,
  "et0MmDay": 2.898,
  "alertsRaised": 0,
  "inferenceMs": 21.4,
  "irrigateProbability": 0.0205,
  "irrigateThreshold": 0.15,
  "irrigateThresholdSource": "per_group",
  "probabilityIsCalibrated": true,
  "depthUncertaintyMm": 2.2534,
  "explanation": ["..."]
}
```

Four of these exist so a client can tell a confident answer from a hedged one.

`irrigateProbability` is a calibrated probability, not a score to be compared against an
arbitrary cutoff, which is what makes `irrigateThreshold` interpretable as an event rate.

`irrigateThresholdSource` is `per_group` when the operating point was tuned on this field's
own crop and soil class, or `global_fallback` when the combination was too rare to tune on.
The fallback case is not hidden: the `explanation` array says so, and the recommendation
deserves more caution.

`depthUncertaintyMm` is the conformal half-width, so a client can show "12.4 ± 2.3 mm"
instead of implying the point estimate is exact. The 90% target is not perfectly achieved
(86% on held-out events), which is reported alongside it in `metrics.json`.

### `POST /api/telemetry/batch`

`{"readings": [ ... ]}`. Each reading is processed independently, so one malformed entry
does not discard the rest. The response reports per-reading status.

---

## Query

### `GET /api/telemetry/latest`

`?device_id=` optional. The current reading and decision for one device, or the most recent
across all devices. Used by the overview and zone pages.

### `GET /api/telemetry/history`

`?device_id=` required, `?hours=` (default 24), `?limit=` (default 500). Readings and
decisions over the window, oldest first.

### `GET /api/zones`

One row per zone: latest reading joined with the latest decision. The dashboard's zone
grid reads this and nothing else, so the whole overview is a single request rather than one
per zone.

| Field | Meaning |
| --- | --- |
| `zone`, `crop`, `deviceId`, `linkState` | Identity and link state |
| `soilMoisture`, `moistureBand` | Latest moisture and its band |
| `soilPh`, `ec`, `nitrogen`, `phosphorus`, `potassium` | Latest chemistry |
| `temperature`, `humidity`, `cropStressIndex` | Latest climate and stress |
| `valveOpen`, `valveRuntimeSeconds` | Current actuator state |
| `nextIrrigationAt` | Next planned cycle, or null |
| `lastSeen` | Null when the node has never reported |

### `GET /api/analytics/overview`

`?hours=` (1 to 2160, default 24). Water balance and efficiency, computed on demand from
the stored decisions.

| Field | Meaning |
| --- | --- |
| `waterAppliedMm` | Mean applied depth per irrigation event |
| `waterAppliedPctChange` | Change against the preceding window of equal length |
| `estimatedLitresSaved` | Against a fixed-ET0 baseline, no rain, no leaching. An estimate |
| `irrigationEvents`, `runtimeSeconds` | Actuation totals |
| `averageInferenceMs` | Mean model latency over the window |
| `decisionsCorrectPct` | Agreement with a transparent depletion rule. A sanity signal |
| `energyKwh` | At a nominal 750 W pump rating. An estimate |
| `series` | Bucketed `label`, `appliedMm`, `et0MmDay`, `depletionFraction` |

---

## Control

### `POST /api/devices/{device_id}/valve`

`{"open": true, "duration_seconds": 900}`. Manual override, independent of the model.
`duration_seconds` is optional; without it the valve stays in the commanded state until
changed.

```json
{ "accepted": true, "deviceId": "node-east-01", "valveOpen": true }
```

### `POST /api/devices/{device_id}/recommendation`

Re-runs inference against the device's last reading and returns a fresh decision, without
storing a new one. Used to inspect model behaviour after a configuration change.

### `POST /api/predict`

Out-of-band prediction for arbitrary sensor values, for what-if analysis. Takes the same
field names as `POST /api/telemetry` and returns a `Recommendation`. Stores nothing.

---

## Devices and alerts

### `GET /api/devices`, `GET /api/devices/{device_id}`

Node registry with crop, soil type, field area, firmware, link state, battery and RSSI.

### `GET /api/alerts`

`?open_only=` boolean, `?limit=` (default 100). Alerts are de-duplicated per device and
kind, so a condition that persists does not flood the log.

### `POST /api/alerts/{alert_id}/acknowledge`

Marks an alert seen. Acknowledging is deliberately separate from resolving: it records that
an operator has seen it, not that the condition has cleared.

---

## WebSocket

### `WS /ws/live`

Push channel for connected dashboards. The dashboard does not poll.

```json
{ "type": "telemetry",      "deviceId": "node-east-01", "zone": "East Terrace", "payload": { } }
{ "type": "recommendation", "deviceId": "node-east-01", "zone": "East Terrace", "payload": { } }
{ "type": "valve",          "deviceId": "node-east-01", "zone": "East Terrace", "payload": { } }
{ "type": "alert",          "deviceId": "node-east-01", "zone": "East Terrace", "payload": { } }
{ "type": "ping" }
```

The client reconnects with exponential backoff, from 1 s to a 15 s ceiling, and shows
connection state in the header rather than displaying stale numbers as if they were
current. Reconnection is on the client because a field dashboard will outlive any single
backend process.

---

## Errors

| Status | Meaning |
| --- | --- |
| `400` | Payload failed validation. `detail` names the offending field |
| `401` | Device key rejected, when `IIC_API_KEY` is set |
| `404` | Unknown device on a control or recompute call |
| `422` | Schema validation error, with the failing fields |
| `503` | Ingest accepted the reading but no decision could be produced |

A `503` is a deliberate distinction: the reading is stored and reported as accepted, but no
decision was made. Silent fallback to a default decision would hide a real fault, so the
condition is surfaced instead.

## Security

The API is unauthenticated, including the control endpoints, and is suitable for a
field-local network only. The `check_device_key` dependency and the `IIC_API_KEY` setting
are already in place for device ingest; operator authentication on control endpoints and
TLS are not implemented. See [ARCHITECTURE.md](ARCHITECTURE.md#security-posture).
