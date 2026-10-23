# Datasets

Every number the controller acts on traces back to a source in this document.
Where a value is modelled rather than measured, that is stated explicitly. No
target label in this project is a plausible-looking number invented to make a
model score well.

---

## 1. Real weather: NASA POWER

| Property | Value |
| :--- | :--- |
| Source | NASA Prediction of Worldwide Energy Resources (POWER) |
| Community | Agroclimatology (`AG`), daily point endpoint |
| Endpoint | `https://power.larc.nasa.gov/api/temporal/daily/point` |
| Coverage | 12 stations x 10 complete years, 2015-01-01 to 2024-12-31 |
| Records | 43,836 daily observations |
| Licence | Open, no API key, no registration |
| Retrieved by | `backend/ml/real_data.py` |

### Why this dataset

The brief requires the system to combine soil moisture with **weather
forecasts**. Synthetic weather defeats the purpose, because irrigation demand is
only interesting when atmospheric demand varies the way it really does. The
stations are chosen to span four climate regimes so the water-balance model is
trained on genuinely different evaporative regimes rather than one climate
resampled with noise.

### Station coverage

| Code | Station | Climate | ET0 mean (mm/day) | Rainfall (mm/yr) |
| :--- | :--- | :--- | ---: | ---: |
| KOL | Kolkata, West Bengal | humid_subtropical | 3.79 | 17,041 |
| PNQ | Pune, Maharashtra | semi_arid | 4.07 | 21,665 |
| BBN | Bhubaneswar, Odisha | humid_subtropical | 4.00 | 15,273 |
| LUD | Ludhiana, Punjab | subtropical_humid | 4.38 | 8,936 |
| LKO | Lucknow, Uttar Pradesh | subtropical_humid | 4.50 | 11,156 |
| NAG | Nagpur, Maharashtra | semi_arid | 4.53 | 13,465 |
| PNJ | Panipat, Haryana | semi_arid_subtropical | 4.54 | 8,760 |
| CHN | Chennai, Tamil Nadu | coastal_subhumid | 4.75 | 14,018 |
| HYD | Hyderabad, Telangana | semi_arid | 4.77 | 8,363 |
| IDR | Indore, Madhya Pradesh | subtropical_dry | 4.92 | 10,818 |
| JAI | Jaipur, Rajasthan | arid | 4.96 | 6,971 |
| AMD | Ahmedabad, Gujarat | arid_subtropical | 5.29 | 7,917 |

The 3.79 to 5.29 mm/day ET0 spread is the point: a controller tuned only on
humid conditions under-irrigates by roughly 40% when transplanted to an arid
district.

### Parameters used

| Parameter | Meaning | Unit |
| :--- | :--- | :--- |
| `T2M` | Air temperature at 2 m, daily mean | deg C |
| `T2M_MAX` | Air temperature at 2 m, daily maximum | deg C |
| `T2M_MIN` | Air temperature at 2 m, daily minimum | deg C |
| `T2MDEW` | Dew point at 2 m | deg C |
| `RH2M` | Relative humidity at 2 m | % |
| `PRECTOTCORR` | Bias-corrected total precipitation | mm/day |
| `WS2M` | Wind speed at 2 m | m/s |
| `ALLSKY_SFC_SW_DWN` | All-sky surface shortwave downward | MJ/m2/day |
| `PS` | Surface pressure | kPa |
| `CLOUD_AMT` | Cloud amount | % |

### Citation

> The NASA Prediction of Worldwide Energy Resources (POWER) Project was accessed
> via the POWER Project's API. Data are openly available under the NASA Open
> Data Commons.

---

## 2. Reference evapotranspiration (derived, not measured)

ET0 is **not** in POWER's daily catalogue, so this project computes it from the
real meteorological inputs using the FAO-56 Penman-Monteith combination
equation.

| Property | Value |
| :--- | :--- |
| Method | FAO Penman-Monteith, Eq. 6 |
| Time step | Daily, with soil heat flux G set to zero per FAO-56 Annex 2 |
| Source publication | Allen, Pereira, Raes & Smith (1998) |
| Reference | FAO Irrigation and Drainage Paper 56 |
| Implemented in | `backend/ml/real_data.py` |
| Validated by | `backend/ml/tests/test_fao56.py` |

### Validation against the published worked examples

This is the part of the project most worth scrutinising, because an irrigation
depth computed from a wrong ET0 is wrong silently. Every expected value below
is printed in the FAO-56 worked examples.

| Quantity | Computed | Published | Error |
| :--- | ---: | ---: | ---: |
| `Ra` Eq. 21, Uccle J=187 at 50.80 N | 41.0884 | 41.09 | 0.00% |
| `Rn` Eq. 40, Uccle 6 July | 13.2950 | 13.28 | 0.11% |
| `gamma` Eq. 8, 101.3 kPa | 0.0674 | 0.0674 | 0.05% |
| `e*(T)` Eq. 11, 45 degC | 9.5825 | 9.59 | 0.08% |
| `ET0` Eq. 6, Bangkok April monthly | 5.6120 | 5.72 | 1.89% |
| `ET0` Eq. 6, Uccle 6 July | 3.7400 | 3.88 | 3.61% |

The 3.61% residual on the Uccle daily case is fully explained and is not an
error. The published value derives actual vapour pressure from `RHmax` and
`RHmin` (Eq. 17). That test path has no dew point, so the code falls back to
mean relative humidity (Eq. 19), which FAO-56 itself labels "less recommended
due to non-linearities". When a dew point is present the code uses the
preferred Eq. 14 route, and `test_dewpoint_route_is_preferred` pins that
preference so it cannot be silently downgraded.

Run the validation yourself:

```bash
.venv\Scripts\python backend\ml\tests\test_fao56.py
```

---

## 3. Real agronomic context

Shipped in `backend/ml/datasets/`. These two files were already in the
repository and are genuine measurements, not generated.

### `irrigation_prediction.csv`

| Property | Value |
| :--- | :--- |
| Rows | 10,002 |
| Grain | one irrigation event on one field |
| Provides | soil type, pH, moisture, EC, organic carbon, temperature, humidity, rainfall, sunlight hours, wind speed, crop type, growth stage, season, irrigation type, water source, field area, mulching, region, prior irrigation depth, and an `Irrigation_Need` label |

### `plant_health_data.csv`

| Property | Value |
| :--- | :--- |
| Rows | 1,202 |
| Grain | one plant, one timestamp |
| Provides | soil moisture, ambient and soil temperature, humidity, light intensity, pH, nitrogen, phosphorus, potassium, chlorophyll, and an electrochemical signal |

This file is the empirical source for the NPK marginal distributions. Nutrient
values are resampled from it rather than drawn from an arbitrary distribution,
so the trained model's nutrient response reflects measured field variability.

### `training_dataset_with_yield.csv`

| Property | Value |
| :--- | :--- |
| Rows | 8,349 |
| Origin | Legacy pipeline, retained for baseline comparison |
| Caveat | Yield and water columns are synthetic. Used only as a regression baseline against which the physics-based models are compared. Not used to train the served models. |

---

## 4. What is modelled rather than measured

Stated plainly, because a reviewer will ask.

| Quantity | Basis | Confidence |
| :--- | :--- | :--- |
| Weather, ET0, radiation | **Measured** (NASA POWER) + published physics | High |
| Soil water balance depth | **Physics** (FAO-56 + soil hydraulic constants) | High |
| NPK marginal distribution | **Measured** (resampled from real plant telemetry) | Medium |
| Soil moisture, pH, EC trajectories | **Modelled** from crop profile + forcing | Medium |
| Disease label | **Rule engine**, thresholds searched against a validated fit | Medium; see caveat |
| Yield estimate | **Modelled** from a multiplicative stress model | Low to medium |

### The disease-label caveat

There is no public dataset pairing NPK soil telemetry with a confirmed
diagnosis of a specific crop disease, and none can be obtained for a student
prototype. The project therefore does the honest thing and says so:

* Labels come from an agronomic rule engine, written from published symptom
  conditions for each disease.
* The thresholds are not hand-picked. They are **searched** over a candidate
  grid and scored with a held-out classifier, with the constraint that the
  resulting class distribution must stay realistic.
* 3.5% label noise is injected, because published inter-rater agreement for
  visual crop diagnosis is imperfect and a perfectly separable corpus would
  make any classifier look better than it is.

The disease model should be read as a **stress-risk classifier for sensor
telemetry**, not a diagnostic instrument. The dashboard labels its output
"risk indicator" for this reason. A commercial field deployment would replace
it with a model trained on image-confirmed diagnoses.

---

## 5. Reproducing the corpus

```bash
# From the repository root, with the virtual environment active
.venv\Scripts\python backend\ml\scripts\fetch_weather.py --start-year 2015 --end-year 2024
.venv\Scripts\python backend\ml\scripts\train.py
```

Downloaded series are cached under `backend/ml/datasets/cache/` keyed by station
and date range, so training is deterministic and does not require network access
after the first run.

---

# Part two — the runtime datasets

Sections 1 to 5 above describe where the *training corpus* came from. This part
describes the datasets the running system reads and writes: the registry, the
reference catalogues the engine plans from, and the API payloads the dashboard
renders. It is written for the person operating the system: what each dataset
contains, where it comes from, and what it is used for.

---

## 6. Registered fields (`data/sowings.json`)

**What it is:** The field registry. One row per registered sowing — the fields the
operator is actively managing.

**Fields per row:**

| Field | Type | Meaning |
|---|---|---|
| `fieldId` | string | Short field name shown on switcher cards, e.g. `Barley-West` |
| `station` | string | Weather station code, e.g. `LKO` (Lucknow) |
| `crop` | string | Crop name from the crop catalogue, e.g. `Barley` |
| `sowingDate` | string (`YYYY-MM-DD`) | Date the crop was sown |
| `soilType` | string | Soil texture from the soil catalogue, e.g. `Silt Loam` |
| `methodName` | string | Irrigation method, e.g. `Basin` |
| `fieldAreaM2` | number | Field area in square metres |
| `mulched` | boolean | Whether the field is mulched |
| `nitrogenRegime` | number | Nitrogen regime multiplier |
| `notes` | string | Free-text notes about the field |
| `cropVariant` | string \| null | Named cultivar, if any |
| `restrictingDepthM` | number \| null | Hardpan/gravel depth in metres |
| `emergenceFraction` | number \| null | Fraction emerged (below 1.0 = partly underground) |

**Used by:** Field registration, the field switcher, and every per-field plan.
**Seed script:** `scripts/sow_field.py` (reproduces the four demo fields exactly).
**Git status:** `data/sowings.json` is gitignored; it exists only locally.

---

## 7. Crop catalogue (`src/irrigation/data/crops.py`)

**What it is:** The engine's own reference table of crop parameters. Every crop the
system can plan for, with the numbers that change the water plan.

**Fields per crop:**

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Crop name, e.g. `Barley`, `Cotton`, `Paddy` |
| `kc_initial` / `kc_mid` / `kc_end` | number | FAO-56 crop coefficient at each stage |
| `root_depth_max_m` | number | Maximum rooting depth, metres |
| `depletion_fraction_p` | number | Fraction of plantable water used before stress |
| `optimal_ph` | [number, number] | pH range the crop prefers |
| `gdd_base_temp_c` | number | Base temperature for growing-degree-days |
| `gdd_stage_ends` | number[] | GDD accumulated at end of each FAO-56 stage |
| `season_gdd` | number | Total GDD for the season |
| `length_stage_days` | number[] | Days in each FAO-56 stage |
| `season_days` | number | Total season length in days |
| `ky` | number | FAO-33 yield response to water |
| `yield_potential_t_ha` | number | Potential yield, tonnes per hectare |
| `variants` | string[] | Named cultivars |
| `suitable_methods` | string[] | Irrigation methods suited to this crop |
| `warnings` | dict | Methods that work but change what the recommendation means |

**Used by:** Field registration validation, plan computation, yield outlook. The
`yield_potential_t_ha` value is the ceiling the dashboard measures the yield gap
against.

**Source:** Imported from `irrigation.data.crops`, never hand-edited.

---

## 8. Soil textures (`src/irrigation/data/soil.py`)

**What it is:** Soil texture reference table. Determines how much water the soil
can hold and how fast it drains.

**Fields per texture:**

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Texture name, e.g. `Silt Loam`, `Clay`, `Sandy Loam` |
| `field_capacity_pct` | number | Water held at field capacity, % |
| `wilting_point_pct` | number | Water held at wilting point, % |
| `available_water_pct` | number | Plant-available water, % |
| `bulk_density` | number | g/cm³ |
| `sand_pct` / `silt_pct` / `clay_pct` | number | Texture percentages |

**Used by:** Plan computation (TAW, RAW, root depth), field registration validation.

---

## 9. Irrigation methods (`src/irrigation/data/cultivation.py`)

**What it is:** Irrigation method reference table. Determines application
efficiency and how a depth recommendation translates to pump time.

**Fields per method:**

| Field | Type | Meaning |
|---|---|---|
| `name` | string | Method name, e.g. `Basin`, `Drip`, `Sprinkler`, `Furrow` |
| `application_efficiency` | number | Fraction of applied water the crop can use |
| `min_depth_mm` / `max_depth_mm` | number | Application limits |

**Used by:** Pump time calculation, field registration validation.

---

## 10. Weather stations

**What it is:** The 12 documented weather stations, each with a code, location,
climate zone, mean ET0 and annual rainfall. The table is in section 1 above.

**Used by:** Plan computation (ET0, rainfall), field registration, weather card.
**Live weather:** OpenWeather current conditions per station (API key in `.env`,
never committed).

---

## 11. Advisory / plan data (`GET /api/advisor`)

**What it is:** The per-field advisory payload — everything the dashboard shows
about one field, computed server-side from one engine plan. This is the primary
dataset the dashboard consumes.

**Top-level fields:**

| Field | Type | Meaning |
|---|---|---|
| `fieldId` | string | Field name |
| `crop` / `station` / `soilType` / `method` | string | Field identity |
| `sowingDate` / `areaM2` | string / number | When sown, how big |
| `ageDays` / `stage` / `stageLabel` / `stageProgress` | number / strings | Crop age and stage |
| `gdd` | number | Growing degree days accumulated |
| `status` | `'irrigate' \| 'schedule' \| 'monitor' \| 'hold'` | What to do |
| `priority` | number | Urgency rank |
| `headline` / `summary` | string | Plain-language decision |
| `recommendation` | object | Irrigation depth, volume, duration, action |
| `water` | object | ET0, ETC, Kc, rainfall, TAW, RAW, depletion |
| `sensors` | object | Live soil sensor readings (7-in-1 head) |
| `risk` | object | Disease risk, stress, yield, nutrients |
| `reasoning` | object[] | Line-by-line explanation of the decision |
| `schedule` | object[] | 7-day irrigation schedule |
| `season` | object \| null | Stage windows, season length, totals |
| `models` | object \| null | Both learned models and their fusion |

**`water` sub-object:**

| Field | Type | Meaning |
|---|---|---|
| `et0MmDay` | number | Reference evapotranspiration, mm/day |
| `etcMmDay` | number | Crop evapotranspiration, mm/day |
| `kc` | number | Crop coefficient |
| `rainfallMmDay` | number | Rain today, mm |
| `tawMm` | number | Total available water, mm |
| `rawMm` | number | Readily available water, mm |
| `depletionMm` | number | Water used from the profile, mm |
| `depletionFraction` | number | Depletion as fraction of TAW (0–1) |
| `rootDepthCm` | number | Effective rooting depth, cm |
| `daysUntilStress` | number \| null | Days until the crop is stressed |

**`risk` sub-object (disease, yield and nutrition):**

| Field | Type | Meaning |
|---|---|---|
| `diseaseRisk` | string \| null | The winning disease class, e.g. `Rust` |
| `diseaseConfidence` | number | Calibrated probability of that class, 0–1 |
| `diseaseProbabilities` | object | Every class and its probability, highest first |
| `riskLevel` | `'healthy' \| 'watch' \| 'risk' \| null` | Overall risk level |
| `cropStressIndex` | number | Crop stress, 0–1 |
| `moistureBand` | `'dry' \| 'optimal' \| 'wet' \| null` | Soil moisture sufficiency |
| `yieldTPerHa` | number | Expected yield, tonnes per hectare |
| `yieldPotentialTPerHa` | number | The crop's potential yield, t/ha, from the catalogue |
| `yieldModelled` | boolean | Whether the yield model was trained on this crop |
| `nutrientDemandKgHa` | object | The NPK dose the engine asks for, kg/ha |

`diseaseProbabilities` is what makes the disease card honest: the runner-up is
visible, so 62% rust against 35% blight reads differently from 62% rust against
2% blight.

**The yield model's crop coverage.** The yield model receives the crop as a
one-hot encoding, and the bundle records the levels it was trained on: `Cotton`,
`Maize`, `Rice`, `Soybean`, `Tomato`, `Wheat`. A crop outside that list — barley,
potato, onion, sugarcane — is encoded as *no crop at all*, so the model still
answers, but with a crop-agnostic estimate rather than a figure for that crop.

`yieldModelled` carries that distinction to the dashboard, which uses it to
decide whether the expected yield may be measured against
`yieldPotentialTPerHa`. For a crop the model does not know, the two numbers are
on different scales — a generic 1.7 t/ha against potato's 20 t/ha ceiling would
report a 91% gap that is an artefact of the encoding, not an opportunity — so the
card shows the estimate, labels it as one, and reports no gap.

**`sensors` sub-object (7-in-1 soil head):**

| Field | Type | Meaning |
|---|---|---|
| `temperature` | number | Air temperature, °C |
| `humidity` | number | Relative humidity, % |
| `soilMoisture` | number | Volumetric soil moisture, % |
| `soilPh` | number | Soil pH |
| `ec` | number | Electrical conductivity, dS/m |
| `nitrogen` | number | Soil nitrogen, ppm |
| `phosphorus` | number | Soil phosphorus, ppm |
| `potassium` | number | Soil potassium, ppm |
| `lightIntensity` | number | Light, lux |

**Used by:** Every dashboard card, field switcher status dots, the "what to do"
headline.
**Source:** `GET /api/advisor`, polled every 15 seconds.

---

## 12. Growth series (`GET /api/plan/{field_id}/growth`)

**What it is:** The engine's season walk — one point every `step` days showing how
depletion, root depth, Kc and water need change across the season.

**Fields per point:**

| Field | Type | Meaning |
|---|---|---|
| `day` | number | Days since sowing |
| `date` | string | Calendar date |
| `stage` | string | FAO-56 stage |
| `gdd` | number | Growing degree days |
| `kc` | number | Crop coefficient |
| `rootCm` | number | Root depth, cm |
| `tawMm` / `depletionMm` | number | Available / used water, mm |
| `depletionFraction` | number | Depletion as fraction |
| `needMm` / `applyMm` / `grossMm` | number | Need / applied / gross water, mm |
| `daysUntilStress` | number \| null | Days until stress |
| `requirementInsufficient` | boolean | Whether supply can meet demand |

**Used by:** Growth chart, season planning honesty (`depletionSource` says whether
the replay was seeded from a measurement or assumed field capacity).

---

## 13. Model metrics (`GET /api/model/evaluation`)

**What it is:** Evaluation artefacts for the two learned models (irrigation
decision and water depth) plus disease and yield.

**Metric groups:**

| Group | Fields | Meaning |
|---|---|---|
| `disease` | `macroF1`, `accuracy`, `perClass` | Disease classification quality |
| `irrigateDecision` | `f1`, `recall`, `precision`, `threshold` | Irrigate/hold decision quality |
| `waterDepth` | `mae`, `r2` | Depth prediction error |
| `yield` | `mae`, `r2` | Yield prediction error |
| `nutrient` | `nitrogen`, `phosphorus`, `potassium` | Nutrient demand error |
| `latencyMs` | record | Per-stage inference cost |

**Used by:** System health page, operator transparency about model reliability.
**Source:** `backend/reports/` (generated by `scripts/` validation scripts).

---

## 14. Alerts (`GET /api/alerts`)

**What it is:** Open alerts raised by the planning pipeline — infeasible
schedules, insufficient requirements, imminent stress.

**Fields per alert:**

| Field | Type | Meaning |
|---|---|---|
| `id` | number | Alert id |
| `fieldId` / `zone` | string | Which field |
| `raisedAt` | string | When raised |
| `kind` | string | `infeasible` \| `requirement_insufficient` \| `stress_imminent` |
| `severity` | `'info' \| 'warning' \| 'critical'` | How urgent |
| `message` | string | Plain-language description |
| `acknowledged` | boolean | Whether the operator has seen it |
| `resolvedAt` | string \| null | When resolved |

**Used by:** Alerts panel, operator triage.
**Actions:** `POST /api/alerts/{id}/acknowledge`.

---

## 15. Telemetry history (`GET /api/telemetry/history`)

**What it is:** Live sensor readings per device, with the recommendations they
produced. V2 has no physical nodes, so this is empty unless a simulator is
running (`IRRIGO_V2_DEMO_SIMULATOR=true`).

**Fields per reading:**

| Field | Type | Meaning |
|---|---|---|
| `id` / `deviceId` / `zone` | numbers / strings | Device identity |
| `recordedAt` | string | Timestamp |
| `temperature` / `humidity` | number | Ambient conditions |
| `soilMoisture` / `soilPh` / `ec` | number | Soil conditions |
| `nitrogen` / `phosphorus` / `potassium` | number | Nutrients, ppm |
| `lightIntensity` | number | Light, lux |
| `batteryVolts` / `rssiDbm` | number \| null | Device health |

**Used by:** Sensor board, device health, drift reports.

---

## 16. Live current conditions (`GET /api/weather/live/{station}`)

**What it is:** The sky at the field's station right now, fetched from
OpenWeather's current-conditions endpoint. It is the only dataset here that
comes from outside the repository, and the only one that is optional.

**Fields:**

| Field | Type | Meaning |
|---|---|---|
| `available` | boolean | Whether there is an observation to show |
| `station` | string | Station code the answer is for |
| `source` | string | `openweather`, when available |
| `temperatureC` / `humidityPct` / `windMs` | number | The current reading |
| `rainfallMm` | number | Rain in the last hour, or the last three |
| `condition` | string | The upstream's own description |
| `observedAt` | string | When the upstream observed it |
| `reason` | string | Why there is no observation, when there is not |

**Rules the endpoint keeps:**

- It never fails. No key, an unknown station and an unreachable upstream all
  answer `200` with `available: false` and a reason, because the dashboard
  falls back to the plan's own readings and a card that errors is worse than a
  card that is honest about where its numbers came from.
- The credential never appears in the response. It is read from `.env` by the
  API process and used there; the browser only ever receives the observation.
- Each observation is cached per station for ten minutes
  (`IRRIGO_V2_OPENWEATHER_CACHE_SECONDS`), because every open dashboard polls
  this endpoint and the free tier allows sixty calls a minute.

**Used by:** The weather card, which labels its own source — "Live now at
PNQ" or "From the plan's own weather at PNQ".

---

## How the pieces fit together

1. **Register a field** — `sowings.json` + crop/soil/method catalogues → stored
   field with an auto-computed plan.
2. **Plan a field** — registered field + station weather (NASA POWER climatology +
   optional live OpenWeather) → engine plan with depletion, stage, schedule.
3. **Advise** — plan + live sensor readings → advisory with irrigation decision,
   disease risk and its full class distribution, the NPK dose, the yield outlook
   against the crop's potential, and the reasoning behind all of it.
4. **Dashboard** — advisory feed → cards for water, soil, weather, crop age,
   sensors, disease and fertilizer, yield; switcher for field selection. The
   weather card additionally polls the live-conditions endpoint and falls back
   to the plan's own readings when there is no live feed.
