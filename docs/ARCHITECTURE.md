# Architecture

## Shape of the system

```
   Soil + air            Weather                Operator
   7-in-1 probe          NASA POWER             dashboard
        |                     |                     |
        v                     v                     ^
   ESP32 node  ---------->  FastAPI  <----------  WebSocket
        |                     |
        | POST /telemetry     | enrich with ET0
        v                     v
   reading row  ------>  model suite  ------>  decision row
                                  |                    |
                                  +----> alerts         +----> event log
                                                       +----> valve command
```

The backend is the only place that decides anything. Field nodes sense, report and act on
commands; they do not make irrigation decisions. That keeps the logic in one place,
versioned with the model that produced it.

## Components

### Field node

An ESP32 with a 7-in-1 soil probe on RS485, a relay per valve, and a flow switch. It wakes,
reads the probe, posts a reading, applies any valve command the backend returned, and
sleeps. Details in [HARDWARE.md](HARDWARE.md).

### Backend

FastAPI, single process, SQLite. Chosen deliberately: the workload is a handful of nodes
posting every few minutes, which SQLite handles comfortably, and a single-file database
keeps deployment to copying one file.

| Module | Responsibility |
| --- | --- |
| `api/routes.py` | The entire REST surface |
| `services/ingest.py` | Reading to decision to alert, as one transaction |
| `services/weather.py` | Nearest station, ET0 enrichment, offline fallback |
| `services/alerts.py` | Rule engine with de-duplication |
| `services/analytics.py` | Water balance, on demand from stored decisions |
| `realtime/hub.py` | WebSocket fan-out to connected dashboards |
| `db/repository.py` | Typed data access, snake_case in SQL, camelCase out |
| `core/ml_bridge.py` | Imports the `ml` package as a sibling, no packaging needed |

### Model suite

Shared between training and serving. `ml/features.py` defines the feature contract and
`ml/inference.py` is the only inference path, so the training and serving code cannot drift
apart. See [MODEL.md](MODEL.md).

### Dashboard

React, TypeScript, Tailwind, Recharts. Presentational primitives in
`components/ui.tsx`; pages compose them and hold no styling. One WebSocket is opened at the
shell and shared through context, so navigation does not reconnect.

## Data flow

### Reading to decision

1. `POST /api/telemetry` validates the payload. Unknown devices are registered on first
   sight, so a node can be commissioned without a separate step.
2. Recent history is loaded for the device, because the model uses trends, not just the
   instantaneous reading.
3. ET0 is attached. If the weather corpus has no station within range, Hargreaves is
   computed from temperature and the response records which method was used.
4. `Controller.decide` runs the estimators and assembles a `Decision` with a
   plain-language explanation.
5. The reading, the decision, any alerts and the event record are written in one
   transaction, so a failure cannot leave a reading without a decision.
6. The result is broadcast to dashboards and any valve command is returned to the node.

The decision is converted twice on purpose: `to_dict` produces camelCase for the API
contract the TypeScript client declares, and `to_record` produces the snake_case the SQL
columns use. One named place for each, rather than a conversion at every call site.

### Degradation

The system is built to keep controlling when a dependency is missing:

| Missing | Behaviour |
| --- | --- |
| Weather station in range | Hargreaves ET0 from temperature, recorded in the response |
| Model bundle | Rule-based fallback decision, flagged in the payload |
| Dashboard | Node keeps controlling; commands are still returned in the response |
| Backend | Node holds its last command and keeps the valve safe |

## Decision logic

The gate is depletion of plant-available water, not raw moisture percentage, because the
same moisture value means different things in different soils.

```
depletion = 1 - (theta - theta_wp) / (theta_fc - theta_wp)

raw ET0   = FAO-56 Penman-Monteith          (or Hargreaves fallback)
ETc       = raw ET0 * Kc(crop, stage)
ETadj     = ETc * (1 + Ks * stress) * Ka
net       = rainfall - ETadj - deep percolation
D(t+1)    = max(0, D(t) + net - irrigation)
```

Irrigation is triggered when depletion exceeds the point at which the crop begins to
close stomata, with a hysteresis band so a value sitting on the threshold cannot
oscillate. The classifier learns the residual mapping from soil type, crop and recent
history that the physical balance does not capture.

`theta_fc`, `theta_wp` and the allowed depletion fraction come from the soil type and crop
configuration, which is why the same reading produces different decisions in different
fields.

## Analytics

The dashboard's water balance is computed on demand from the stored decisions rather than
maintained as a running total. A running total can drift away from the records it
summarises; recomputing cannot.

Two figures are estimates and are labelled as such in the interface: saved water is
measured against a fixed-ET0 baseline with no rainfall and no leaching, and pump energy
assumes a nominal 750 W rating. Both are relative indications, not metered volumes.

## Storage

SQLite, one file, with foreign keys on and WAL enabled.

| Table | Contents |
| --- | --- |
| `devices` | Node identity, crop, soil, field area, link state, valve state |
| `readings` | Raw sensor values, one row per report |
| `decisions` | Model output, the explanation, and the raw estimator output |
| `alerts` | Operator alerts, de-duplicated per device and kind |
| `events` | Append-only audit log of commands, alerts and system events |

Storing the model output alongside the reading means history and analytics survive a
restart and a model upgrade, and it makes any past decision auditable.

## Security posture

The API is currently unauthenticated, including the valve control endpoints. It is
suitable for a field-local network only.

Before exposing it, add device API keys on ingest (the `check_device_key` dependency and
the `IIC_API_KEY` setting are already in place), operator authentication on control
endpoints, and TLS. The device table already carries a `link_key`, so the schema does not
need to change to close this gap.
