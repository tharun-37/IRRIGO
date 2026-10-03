# Irrigation intelligence

Estimates how much water a registered field needs over a two-week horizon, works
out when to apply it, and states plainly how much to trust the answer.

The previous generation of this system scored **0.082 R² on a station it had
never seen**. This one scores **0.708 mean R² across twelve leave-one-station-out
folds**. The gap is not a better algorithm. The old model was fitted to a label
that took three distinct values across 72,082 rows, and no amount of tuning fixes
that. The reasoning is in [ARCHITECTURE.md](ARCHITECTURE.md); this file is how to
run it and how far to believe it.

---

## Quick start

```powershell
$py = "D:\Codings\MP3\.venv\Scripts\python.exe"

# a worked example, printed to the console
& $py scripts\sow_field.py demo

# register a field
& $py scripts\sow_field.py add --field-id LKO-wheat-09 --station LKO `
    --crop Wheat --sowing-date 2024-11-05 --soil Silt_Loam --method Basin --area 5000

# what does this field need, and when?
& $py scripts\sow_field.py plan --field LKO-wheat-09

# the same field as the season develops, with a real depletion reading
& $py scripts\sow_field.py plan --field LKO-wheat-09 --as-of 2024-12-20 --depletion 95

# the whole season calendar, stage by stage
& $py scripts\sow_field.py season --field LKO-wheat-09

# how one field's water demand changes as it grows
& $py scripts\sow_field.py compare --field LKO-wheat-09

# what is registered
& $py scripts\sow_field.py list
```

`plan` prints the requirement in millimetres, the dated schedule, the depletion
at each step, the stage the crop is in, and any reason to be suspicious of the
answer.

A plan at the default `as_of` is a fresh, full soil profile, because that is the
state on the day of sowing. Three weeks later it is a guess, so pass the reading
from a probe with `--depletion`.

---

## What it does

**Registers sowings.** A field is registered once, with its crop, soil, method
and sowing date. From that date the system counts elapsed days and thermal time,
derives the growth stage, and tracks root depth, crop coefficient, total available
water and depletion forward. Nothing needs re-entering as the season advances.

```powershell
& $py scripts\sow_field.py compare
```

The same wheat field appears twice, seeded and mature:

| | seedling | mature |
|---|---|---|
| Kc | 0.40 | 1.15 |
| root depth | 30 cm | 180 cm |
| TAW | 60 mm | 360 mm |
| 14-day requirement | small | large |

That contrast is the whole system in one table. A crop needs more water as it
grows, for the boring reason that there is more root and more leaf, and a tool
that cannot express that is not a tool.

**Estimates the requirement.** `models.py` fits a gradient-boosted regressor on 36
features that describe the *physical state* of the field: Kc, root depth, depletion
and its fraction of TAW, growing degree days, and the weather. It reaches
**+72.8% skill against the textbook `Kc × ET0 × 14` calculation** a practitioner
would do by hand (4.34 mm against 15.97 mm, on rows that need water).

**Schedules it.** `optimizer.py` searches dated schedules by dynamic programming,
respecting the method's per-event cap, daily infiltration, field capacity and the
delivery cap. It will not apply more than was computed; if the requirement is too
small to avoid crop stress it says so in those words instead of quietly
over-watering.

**Reports its own uncertainty.** Split-conformal intervals, calibrated and then
*measured*:

| nominal | interval | realised coverage |
|---|---|---|
| 80% | ±6.53 mm | 80.0% |
| 90% | ±9.11 mm | 90.0% |

Read the realised column, not the nominal one. It is there because this
calibration step caught a real bug — the quantile level was inverted, producing an
interval ten times too narrow while labelling itself 90% correct.

**Refuses to over-promise.** "Needs water at all" is reported separately from the
regression, because it is a different question: recall **0.996**, missed **0.42%**,
false alarms **27.61%**. It is deliberately biased toward watering. A false alarm
wastes an irrigation; a false negative costs yield that cannot be recovered.

---

## Validated against independent data

A simulation that checks itself proves nothing, so the inputs are cross-checked
against ERA5 via Open-Meteo — a different implementation of the same physics.

**Evapotranspiration.** Two independent FAO-56 implementations, 12 stations,
2015–2024, 43,836 paired days:

```
mean r 0.911    ratio 0.998    bias -0.014 mm/day
```

The project's FAO-56 Penman-Monteith arithmetic agrees with ERA5 to within 0.2%
on the mean.

**Simulated soil water against observed soil water** — the check the project most
needed and did not have:

```
100% of 72,082 corpus rows paired
within-series seasonal shape, 555 series: mean r +0.352, 90.5% positive
observed depth profile monotonic increasing: yes
```

**What this does and does not prove.** ERA5 is a reanalysis, not a probe, and its
0–7 cm layer sits above most of the root zone. This is corroboration, not
calibration. The simulated soil profile is deliberately *not* fitted to ERA5:
that would replace a transparent physical model with an opaque correction that
transfers badly to a new field.

---

## Honest limitations

Read these before trusting a number in the field.

**The FAO-56 coefficients are global averages, uncalibrated for these fields.**
Crop coefficients, stage lengths and depletion fractions are published defaults.
That is the largest single source of error in the system and no amount of
modelling removes it. A local Kc curve and one season of probe data would help
more than any further model work.

**Soil texture is assigned, not measured.** Twelve FAO-56 classes allocated per
field specification. The external validation localises the error precisely: the
seasonal *dynamics* are sound (within-series r +0.35) but the *level* is off per
station in both directions. SoilGrids is the fix; the API was unreachable from
the build environment.

**GDD stage thresholds are an assumption.** FAO-56 publishes stage lengths in
days, not accumulated thermal time. Phenology will be wrong for varieties whose
thermal requirement differs from the table.

**Two irrigation methods are absent from the corpus.** `Flood` and `Paddy`
produce constant columns in the feature matrix. Rice on a paddy is a significant
Indian case and its absence is a real gap.

**One station fails, and it is reported.** Kolkata, leave-one-station-out
R² = −0.244. 87.2% of its rows need no water against 39.8% in training, because
it receives 9.07 mm/day of rain against 3.89 elsewhere. That is genuine covariate
shift from a climate with no analogue in the corpus, not a rounding error, and the
operational consequence — 11.12 mm error on KOL rows that do need water, against
4.18 mm overall — is the number that matters. Twelve stations is a small sample
of the subcontinent's climate. More stations is the fix.

**There is no labelled field data anywhere in this project.** Every water number
traces to NASA POWER, the FAO-56 tables, or this simulator. Nothing here has been
checked against a measured irrigation decision, and the claim "this predicts what
a farmer should apply" is not one the evidence supports yet.

---

## Data provenance

| what | source | coverage |
|---|---|---|
| weather | NASA POWER daily agroclimatology | 12 stations, 2015–2024 |
| cross-check | Open-Meteo Historical Weather API (ERA5) | 12 stations, 2015–2024 |
| soil moisture | ERA5 layers 0–7, 7–28, 28–100 cm | 12 stations, 2015–2024 |
| corpus | this simulator | 72,082 rows, 1,200 series |

Open-Meteo terms permit this use. POWER requires no key. SoilGrids and CHIRPS
were unreachable from the build environment and are not used.

Station coordinates are read from the POWER cache, so the two sources are
guaranteed to refer to the same points.

---

## Layout

```
src/irrigation/
  data/        climate, soils, crops, cultivation, external (ERA5)
  physics/     FAO-56 ET0, water balance, NIR, Kc curves, GDD
  phenology/   stage transitions
  policy/      trigger thresholds, method constraints
  residuals/   bias corrections
  corpus.py    season simulation
  features.py  feature contract + leakage audit
  models.py    splits, metrics, regressor, conformal
  optimizer.py constrained schedule search
  sowing.py    registry, day count, field plans
scripts/       sow_field, train_model, fetch_external,
               validate_against_observations, validate_soil_water,
               compare_v1_v2, refresh_external_cache
tests/         38 pytest cases
data/          corpus, sowings, caches
models/        trained model + manifest
reports/       evaluation, external_validation, soil_validation, v1_vs_v2
```

---

## Reproducing

```powershell
# 38 tests plus the physics invariant script
& $py -m pytest tests -q
& $py tests\test_physics_core.py

# external data (cached; ~40 MB)
& $py scripts\fetch_external.py
& $py scripts\refresh_external_cache.py

# validation
& $py scripts\validate_against_observations.py
& $py scripts\validate_soil_water.py

# model: grouped, leave-one-station-out, leave-one-crop-out
& $py scripts\train_model.py

# head-to-head against the previous generation
& $py scripts\compare_v1_v2.py
```

`train_model.py` takes about 80 seconds on 20 threads. `--quick` evaluates three
folds instead of all of them.

---

## Evaluation results

| split | MAE | R² |
|---|---|---|
| grouped by field series | 2.97 mm | +0.9571 |
| leave-one-station-out (12) | — | mean +0.7083, min −0.2442 |
| leave-one-crop-out (11) | — | mean +0.8205, min +0.5891 |

against the previous generation's **+0.0816** on an unseen station.

Permutation importance confirms the model is using physics rather than
memorising categories. The features that matter are `depletion_fraction`
(+0.408), `days_since_sowing` (+0.308) and `etc_mm` (+0.271). No crop, soil or
station identifier appears in the top twelve.

---

## The single most important design decision

The target is the **uncapped requirement**, and the cap lives in the optimiser.

The old target was `min(TAW - Dr, infiltration)`. `infiltration` is constant for
a given soil, so whenever it bound, the label *was* that constant. Three distinct
values. Reproduce that definition on this corpus and the degeneracy returns
immediately — 29.54% of all rows take the modal value. Against 0.01% for the
uncapped requirement across 40,402 distinct values.

A cap is a fact about a pump and a soil. A requirement is a fact about a crop.
Conflating them is what cost the previous generation its out-of-distribution
performance, and `compare_v1_v2.py` reproduces the failure on demand so the
argument does not have to be taken on trust.
