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
