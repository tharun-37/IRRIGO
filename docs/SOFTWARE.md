# Software

Developer-facing documentation: the stack, the layout, how to run and verify each part,
and how to change it safely.

For what the system does and why, see [ARCHITECTURE.md](ARCHITECTURE.md). For the models,
[MODEL.md](MODEL.md). For endpoints, [API.md](API.md).

---

## Stack

| Layer | Choice | Why this and not the obvious alternative |
| --- | --- | --- |
| Language | Python 3.14 | scikit-learn and the scientific stack |
| API | FastAPI + Uvicorn | Typed, async, and the same app serves the WebSocket |
| Storage | SQLite, WAL | A few nodes posting every few minutes. One file to deploy |
| Models | scikit-learn `HistGradientBoosting` | Nonlinear, no preprocessing, sub-millisecond on CPU |
| Weather | NASA POWER daily API | Free, no key, documented, globally covering |
| Dashboard | React 18 + TypeScript + Vite | Fast builds, types catch contract drift at compile time |
| Styling | Tailwind | Consistent without a component library to keep in step |
| Charts | Recharts | Composable, and the only heavy dependency |

No GPU is required or used. The estimators are CPU-bound; the entire training run takes
about 40 seconds on 20 threads. Training a HistGradientBoosting model on a GPU would need
RAPIDS cuML or XGBoost, and would not make the pipeline faster in any way that matters at
this scale.

Training and serving want opposite thread counts, so they are separated deliberately.
Training is a batch job that benefits from every core. Serving pushes a single row through
eight tree ensembles, and at that size OpenMP thread synchronisation costs more than the
arithmetic: measured on this machine, 20 threads took about 43 ms per decision against
about 20 ms on one thread. `backend/app/main.py` therefore sets `OMP_NUM_THREADS=1` before
sklearn is imported, using `setdefault` so an explicit setting in the environment wins.
Training is a separate process and is unaffected.

---

## Layout

```
backend/
  app/
    main.py            lifespan, router mounting, offline sweep
    schemas.py         Pydantic request and response contracts
    api/routes.py      every endpoint
    core/config.py     environment-driven settings
    core/database.py   schema, migrations, connection
    core/ml_bridge.py  imports the ml package as a sibling
    db/repository.py   typed data access
    realtime/hub.py    WebSocket connection registry
    services/          ingest, weather, alerts, analytics
  ml/
    real_data.py       NASA POWER client, FAO-56 ET0
    dataset_builder.py forward water-balance simulation
    features.py        shared feature contract, used by training and serving
    inference.py       Controller.decide, the only inference path
    scripts/train.py   training, evaluation, serialisation
    tests/             FAO-56 reference checks, disease label provenance
    artifacts/         bundle, metrics, figures, training log
    artifacts/registry/ versioned bundles and run-to-run deltas, gitignored
  tools/simulate_nodes.py
  tests/test_api.py
frontend/
  src/api/client.ts    typed client, one request() implementation
  src/api/client.test.ts
  src/lib/format.ts    every displayed number passes through here
  src/lib/format.test.ts
  src/hooks/           WebSocket and polling, with tests
  src/components/ui.tsx presentational primitives only
  src/hooks/           WebSocket live feed, throttled polling
  src/lib/format.ts    numbers, times, units
  src/pages/           Overview, Zones, Analytics, Model, Alerts
firmware/esp32/        node firmware, pending
```

The load-bearing structural decision is that `ml/features.py` and `ml/inference.py` are
shared by training and serving. There is no second implementation of the feature logic, so
the two cannot disagree about what a feature means.

---

## Running

```bat
run.cmd
```

Or by hand, in two terminals:

```bat
cd backend
..\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

cd ..\frontend
npm run dev
```

The Vite dev server proxies `/api` and `/ws` to port 8000, so no absolute URL is baked into
the bundle and the frontend needs no environment configuration in development.

### Production build

```bat
cd frontend
npm run build     ->  frontend/dist
npm run preview   ->  serves dist on 4173 with SPA fallback
```

Because the dashboard uses `BrowserRouter`, whatever serves `dist` must fall back to
`index.html` for unknown paths, or a deep link such as `/zones` will 404. `vite preview`
does this already. Any other static host needs an equivalent rewrite rule.

---

## Configuration

Settings are environment-driven with a `.env` file in `backend/`. Every value has a working
default, so nothing has to be set to run the project.

| Variable | Default | Notes |
| --- | --- | --- |
| `IIC_DATABASE_URL` | SQLite in `backend/data` | Point at Postgres if the deployment needs it |
| `IIC_MODEL_BUNDLE_PATH` | `backend/ml/artifacts/iic_models.joblib` | |
| `IIC_DEVICE_API_KEY` | empty | **Empty disables device authentication.** Set it for anything but a bench |
| `IIC_ENABLE_WEATHER_ENRICHMENT` | `true` | Off disables ET0 enrichment entirely |
| `IIC_HARD_MAX_DEPTH_MM` | `45.0` | Cap on a single application, independent of the model |
| `IIC_HARD_MAX_RUNTIME_SECONDS` | `21600` | Valve runtime ceiling |
| `IIC_MAX_CLOCK_SKEW_SECONDS` | `300` | Readings further ahead than this are rejected |
| `IIC_TELEMETRY_RETENTION_DAYS` | `180` | Raw readings only; decisions are kept |
| `IIC_CORS_ORIGINS` | local dev ports | Only needed if the dashboard is on another origin |

`hard_max_depth_mm` and `hard_max_runtime_seconds` are applied outside the model. A model
that is wrong, or a corrupted bundle, should not be able to flood a field; these limits are
the backstop for that.

---

## Working on it

### Retraining

```bat
cd backend
..\.venv\Scripts\python ml\scripts\train.py --skip-tuning
```

About 40 seconds. Writes the bundle, `metrics.json`, figures and a training log to
`ml/artifacts/`. Drop `--skip-tuning` to search the decision threshold.

After retraining, restart the backend. The bundle is loaded once at startup so that a
corrupt file fails immediately and visibly rather than at the first irrigation.

### Adding a feature

1. Add it to the feature list in `ml/features.py`, with its agronomic definition and units.
2. Add the derivation in `dataset_builder.py`.
3. Retrain. The exact column order each estimator saw is recorded in the bundle, so a
   reorder cannot silently break serving.
4. Add the field to `app/schemas.py` if it arrives from a node, and to `frontend/src/types.ts`
   if the dashboard shows it.

Step 3 exists because the training frame assembles the one-hot groups in dataframe column
order, which is not the declared order. That mismatch produced a serve-time failure that
scikit-learn caught loudly, but recording the trained order removes the class of problem
entirely.

### Adding an endpoint

1. Add it to `app/api/routes.py`. Ingest goes under `/api/telemetry`, control under
   `/api/devices`.
2. Add the request and response models to `app/schemas.py`.
3. If the dashboard uses it, add the method to `frontend/src/api/client.ts` and the types to
   `frontend/src/types.ts`.

`tsc --noEmit` runs as part of `npm run build` and will fail on a contract mismatch, which
is the point.

### Database changes

`core/database.py` owns the schema and applies it at startup. Foreign keys are on and WAL is
enabled. Do not add a migration for a new column without a migration path for existing
databases, or a field deployment will fail to start after an update.

---

## Verifying

Backend, in three independent suites. None of them needs a server running, and each is
checking something the others cannot.

```bat
cd backend

..\.venv\Scripts\python ml\tests\test_fao56.py
```

Nine ET0 checks against published FAO-56 reference examples. This is the check to run
whenever anything touching evapotranspiration changes, because an ET0 error is invisible in
every downstream metric and simply makes the whole system confidently wrong.

```bat
..\.venv\Scripts\python ml\tests\test_disease_labels.py
```

Thirteen checks on the provenance of the disease labels. The corpus labels are synthesised
by a rule engine, and this asserts that they are reproducible from six features at 100%,
which is why the disease model's scores mean "memorised the lookup table" and not
"diagnosed disease". It exists so that a future change cannot quietly make the disease
model look better than it is.

```bat
..\.venv\Scripts\python tests\test_api.py
```

End-to-end API suite over an in-process test client: ingest, decision, alerts, control,
WebSocket, and the model's serving contract. The last part matters most — it asserts that
the irrigate probability is still calibrated and the depth estimate still carries its
uncertainty, because a retrain that dropped the calibration wrapper would otherwise pass
every functional check while quietly making every threshold meaningless.

Frontend, one command for the whole gate:

```bat
cd frontend
npm run verify
```

That runs typecheck, lint, tests and build in sequence. The pieces separately:

```bat
npm run typecheck   # tsc --noEmit
npm run lint        # eslint, type-aware
npm run test        # vitest, 36 tests
npm run build       # tsc -b && vite build
```

Lint is type-aware on purpose. `no-floating-promises` is enabled because this dashboard
drives a live irrigation controller, and an unawaited `fetch` is a valve command that
silently never arrives. The hook tests cover the two ways a monitoring surface can lie to
an operator: reporting a stale value after unmount, and holding the last good payload after
the backend dies.

Coverage is configured against `src/lib`, `src/api` and `src/hooks` with a 60% floor, since
those hold the pure logic encoding the safety rules rather than presentation code.

```bat
cd backend
..\.venv\Scripts\python tools\simulate_nodes.py --once --minutes 240 --step 15
```

Back-fills 64 readings across four zones and exits. Add `--interval 5` without `--once` to
stream live telemetry, which exercises the WebSocket and the live charts.

---

## Operational notes

**Single writer.** SQLite handles this workload, but the design assumes one backend
process. Running several against one database file will work at low volume and will not
scale. Move to Postgres before adding zones, not after.

**Unauthenticated control.** The valve endpoints have no operator authentication. Field-local
network only, until that is added. See the security posture section of
[ARCHITECTURE.md](ARCHITECTURE.md).

**Model bundle provenance.** `/api/health` reports the corpus row count, the ET0 method, the
split strategy and the bundle build time. That is deliberate: a decision should be
traceable to the data and method behind it, and a version string alone can be stale.

**Disease output is a risk indicator.** Rule-derived stress conditions, not pathogen
identification. It should prompt someone to look at the crop, never be treated as a
diagnosis.
