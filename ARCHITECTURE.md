# Architecture

## The problem this shape solves

A previous generation of this system scored 0.082 R² on a station it had never
seen. The cause was not a weak model, a bad algorithm, or insufficient data. It
was a label:

```
target = min(TAW - Dr, infiltration)
```

`infiltration` is a property of the soil texture, and it is constant for a given
soil. So the label was the smaller of two numbers, one of which never changes.
Whenever infiltration bound, the label *was* the infiltration rate. Across the
corpus that happened on 66.36% of irrigation events, leaving three distinct label
values: 22 mm, 12 mm and 32 mm. A model fitted to three values learns the soil
table, and a soil table learned from eleven stations tells you nothing about the
twelfth. The R² of 0.082 was the arithmetic consequence of the label, not a
property of the data.

Everything below follows from refusing to make that mistake again.
---

## Data flow

```
                    NASA POWER daily agroclimatology
                    12 stations, 2015-2024, 43,836 records
                              |
                              v
              +-------------------------------+
              |  FAO-56 Penman-Monteith (Eq 6)|
              |  -> ET0, net Rn, Ra, VPD      |
              +-------------------------------+
                              |
   Open-Meteo / ERA5  <------|------>  independent cross-check
   (ET0, rain, soil           |           (see Validation)
    moisture)                 v
                    +--------------------+
                    |  corpus.py         |
                    |  daily water       |
                    |  balance per field |
                    +--------------------+
                              |
                              v
        crop_corpus.csv  72,082 rows, 1,200 series, 40,402 distinct targets
                              |
              +---------------+----------------+
              |                                |
              v                                v
     +------------------+            +---------------------+
     | features.py      |            | models.py           |
     | physical state   |  36 cols   | HistGradientBoosting|
     | + leakage audit  | ---------> | + conformal         |
     +------------------+            +---------------------+
              |                                |
              v                                v
     registered sowings  <-----------  per-day requirement
              |                                |
              v                                v
     +------------------+            +---------------------+
     | sowing.py        |            | optimizer.py        |
     | day count, stage,| <---------| dynamic programme    |
     | requirement      |  plan      | over feasible dates |
     +------------------+            +---------------------+
              |
              v
     dated schedule, in millimetres, with explanations
```

---

## The five decisions that matter

### 1. The target is the requirement, not the application

The model learns how much water the root zone needs over a 14-day horizon. That
number is **uncapped**. Whether it can be placed in one pass is a property of the
pump and the soil, and it is recorded in a separate column.

Measured on this corpus, the difference is not marginal:

| target definition | distinct values | modal share among rows needing water |
|---|---|---|
| `min(TAW - Dr, infiltration)` | 854 | **29.54%** |
| uncapped NIR | 40,402 | **0.01%** |

The old definition is reproduced on the *new* corpus by `compare_v1_v2.py` so the
degeneracy is visible rather than remembered.

### 2. The regressor learns the residual, not the physics

`features.py` hands the estimator the *physical state* — Kc, root depth, TAW,
depletion, the depletion fraction, GDD — not raw identity. Crop and soil names are
deliberately excluded; by the time features are built, crop identity has already
been collapsed into those state variables, so a one-hot would only invite the
model to spend depth memorising fourteen categories.

The permutation audit confirms the design works. The top features are:

| feature | R² lost when shuffled |
|---|---|
| `depletion_fraction` | +0.408 |
| `days_since_sowing` | +0.308 |
| `etc_mm` | +0.271 |
| `is_monsoon` | +0.103 |
| `vpd_kpa` | +0.101 |
| `p_depletion_fraction` | +0.098 |
| `gdd_accumulated` | +0.080 |

No crop, soil or station identifier appears. The model is using season, soil state
and climate, which is what it should be using.

`FORBIDDEN` in `features.py` names the columns that would leak the target, and
`frame()` raises if any reaches the estimator. V1's pipeline computed the
requirement and then passed it to the model as an input.

### 3. Forward weather is climatological, not observed

Days already elapsed use the station's real record. Days ahead use the station's
own long-run monthly means. Using the realised future would leak information a
farmer does not have at six in the morning, and would produce a model that looks
excellent in evaluation and behaves badly in service.

Rain is credited at the *probability* of an event, not at full depth spread
thinly. A version that credited the monthly mean on every day was equivalent to
assuming it rains every day for a fortnight, which over-credits a monsoon and
under-irrigates a dry season.

### 4. The optimiser delivers the requirement and may not exceed it

`optimizer.py` searches dated schedules by dynamic programming over depletion
buckets, respecting the method's per-event cap, daily infiltration, field
capacity and the equipment turn-around. Stress is a hard constraint.

It also carries a delivery cap. Without it, the stress penalty makes
over-watering look attractive: asked for 20 mm, an early version applied 45 mm,
because every millimetre that kept the crop below its stress point improved the
objective. That is a model setting a grower's irrigation rate. The optimiser's job
is to *deliver* the computed requirement; when the requirement is genuinely too
small to avoid stress, `requirement_insufficient` is raised and the grower is told
in those words.

Three further defects turned up while exercising it against real fields, all of
them the kind that only appear when the optimiser is run on a case nobody
imagined. They are recorded because the fixes are load-bearing, not incidental.

**A harvest was being irrigated.** Late-season tables go on producing a Kc and a
root depth indefinitely, so a field whose grain was off — wheat at Lucknow, day
144, Kc 0.25, root 5 cm — was recommended 29.1 mm over 14 days. Past harvest the
requirement is zero by definition, and `_requirement_for` now says so.

**Equipment turn-around was documented and never read.** `min_interval_days` is
in the method table (Basin 6, Sprinkler 5, Drip 1) and the optimiser ignored it,
so a basin field with 80 mm to place applied it on day 0 and day 1. Depletion
alone cannot express "how long ago did the equipment last run", so the state
gained a cooldown dimension.

**A near-empty profile swallowed every application.** This is the subtle one. The
DP's state is a depletion bucket at 2 mm resolution, and two paths that apply very
different amounts of water can reach the *same* bucket when the profile is nearly
empty — the water is absorbed and the bucket does not move. The dominance test
compared cost, and applying water always costs more, so the optimiser discarded
every watering and returned a schedule that applied **nothing at all** while
reporting 120 mm unmet. Delivering the requirement is a resource, not a cost, so
the rule is now: for equal bucket and equal cooldown, keep the path that has
already delivered more.

**Delivery has to dominate the water cost.** With unmet demand charged at 2.0 per
millimetre and gross water counted as cost, the optimiser declined a 150 mm
requirement as not worth 300 mm of gross water. That is a schedule that costs a
grower yield to save water, which is the wrong trade in every direction. The
penalty is now 100 per millimetre, comfortably outside the order of water any
schedule could save, which makes the objective lexicographic in effect: deliver
first, then minimise.

**Sub-resolution shortfalls are not infeasibility.** Depletion is bucketed to 2 mm
and depths to a 5 mm grid, so the search cannot resolve a shortfall smaller than
its own resolution. A 0.3 mm miss out of 80.3 was being reported as an impossible
schedule. The tolerance is now the larger of one depletion bucket and 1% of the
requirement.

### 5. Intervals are calibrated where they mean something

Split conformal on absolute residuals, with the finite-sample `(n+1)/n`
correction. Two errors were made and fixed here, and both are worth recording.

The quantile level was initially `1 - coverage`, which is the *tenth* percentile
for a nominal 90% interval. The interval was ten times too narrow while
reporting itself as 90%. The giveaway was in its own output: realised coverage of
10% next to a label saying 90%. Always read the realised number.

The target is a mixture: 42% of rows are exactly zero, because the soil already
holds a fortnight of demand, and the model is essentially exact on those. The
marginal residual distribution therefore has a point mass at zero, and a naive
conformal interval returns **±0.0 mm at 90% coverage**. Not wrong about the
marginal distribution; useless for the decision, because the rows where an error
has a consequence are exactly the rows it reports as perfect. Calibration is
therefore restricted to rows that need water, and both intervals are reported.

---

## Validation against independent data

A simulation that validates itself proves nothing. The project therefore takes
its inputs from a second, independent implementation and compares.

### Reference evapotranspiration

The project computes ET0 from POWER meteorology using its own FAO-56
Penman-Monteith implementation. Open-Meteo computes the same equation from ERA5
with a different implementation. Two independent routes to the same physical
quantity:

| | result |
|---|---|
| mean daily correlation, 12 stations | **r = 0.911** |
| mean ratio (ERA5 / project) | **0.998** |
| mean bias | **−0.014 mm/day** |
| worst station | HYD at r = 0.857 |

Agreement to 0.2% on the mean validates the arithmetic. The script also decomposes
the gap into humidity, wind, radiation and temperature, so a bias is traceable to
a meteorological variable rather than merely observable.

One trap worth recording, because it points a diagnosis at the wrong variable.
Open-Meteo reports daily mean wind in km/h and the project stores m/s. Compared
raw, the table shows a 4.3x "bias" at every station — pure units. Dividing by 3.6
brings it to agreement, and the ET0 total is unaffected either way, which is
itself a useful check that wind is not dominating the energy balance.

### Rainfall

| | result |
|---|---|
| mean daily correlation | r = 0.668 |
| mean ratio (ERA5 / project) | 0.980 |
| worst station | LUD at r = 0.496 |

Weaker than ET0, as expected: one product is satellite-merged and bias-corrected,
the other a reanalysis. The per-station ratios scatter on both sides of 1.0
(AMD 1.31, KOL 1.10, LUD 0.90) with no systematic multiplier, which is the
signature of two independent products rather than a bug in one of them.

### Simulated soil water against observed soil water

This is the check the project most needed and did not have. The soil water state
is purely simulated, and until now nothing examined it. ERA5 provides observed
volumetric water content in three layers.

| | result |
|---|---|
| corpus rows paired with observations | **72,082 of 72,082 (100%)** |
| simulated mean | 0.2911 m³/m³ |
| observed root-zone proxy mean | 0.2837 m³/m³ |
| pooled correlation | +0.359 |
| **within-series seasonal shape, 555 series** | **mean r = +0.352** |
| share of series positively correlated | **90.5%** |
| observed depth profile monotonic increasing | **yes** (0.286 → 0.293 → 0.299) |

The within-series figure is the one with diagnostic power. Pooling every station
into a single correlation would be dominated by differences *between* fields,
which is lookup-table skill already established. The question asked is narrower:
within one field over one season, does the simulation move the way the land
surface moves? Correlation in the 0.2–0.6 range is the expected result for a
correct implementation compared against a 31 km reanalysis top layer. A
correlation near zero would be a genuine failure signal; 0.9 would mean the
comparison is wrong, not that the model is right.

Two results are worth reading closely.

**The weak stage is early.** Correlation rises monotonically with the season:
initial +0.149, development +0.186, mid-season +0.420, late +0.452. The
simulator matches observation least well when the crop is small and the water
demand is smallest, and best once the canopy is closed. That is the expected
signature of a correct water balance driven by radiation, and the early-season
weakness is where a local probe or a measured Kc would help most.

**The per-station levels disagree in both directions** (NAG simulation 0.310
against ERA5 0.409; LKO 0.321 against 0.231), and CHN correlates at +0.0005. A
correct dynamic process with the wrong soil texture still tracks the right
signals, and the level error is a soil-assignment error. This localises the
weakest input precisely: it is the texture table, not the water balance.

---

## Evaluation protocol

Three splits, because one cannot detect the failure that actually matters.

| split | what it tests | result |
|---|---|---|
| grouped by field series | V1's protocol, for comparability | MAE 2.97 mm, **R² 0.9571** |
| leave-one-station-out, 12 folds | unseen climate | **mean R² 0.7083**, min −0.2442 |
| leave-one-crop-out, 11 folds | unseen crop | **mean R² 0.8205**, min 0.5891 |

Against the previous generation's 0.0816 on an unseen station, the mean is 0.708.
The minimum is negative, and the next section explains why rather than averaging
it away.

### A grouped split that was not grouped

The grouped split is reported first, labelled "comparable with the previous
generation's protocol", and it is the one number here that was quietly wrong for
a while. `series_key` identified a series by station, crop, soil and **observation
date**. That makes every day its own group: 58,758 groups for 72,082 rows. It was
not a grouped split at all, but a random row split wearing a grouped split's name,
and consecutive days of one field — the closest thing in this corpus to duplicate
records — were landing on both sides of the boundary.

The honest number is **R² 0.9571, MAE 2.97 mm**, against 0.9648 and 2.83 mm
before. The inflation was real and small, which is worth saying plainly: the
metric was not trustworthy, but the model was not hiding much behind it. The
station and crop folds never used `series_key`, so 0.7083 and 0.8205 were always
correct and are unchanged.

The fix keys on the reconstructed sowing date, giving **1,200 series** of 16 to
170 days. A figure to keep in mind when reading any grouped-split result: 1,170
groups over 72,082 rows means the average held-out series is 62 days long, and a
model that has seen 61 other days of the same field is not making a field-scale
prediction. That is precisely why the station and crop folds are the ones that
decide deployability.

### Against the textbook

A model is not evaluated against its own loss.

| | MAE on rows needing water |
|---|---|
| `Kc × ET0 × 14` (what a practitioner does by hand) | 15.97 mm |
| this model | **4.34 mm** |

**+72.8% skill**, n = 3,843.

### The "does it need water" decision

Reported separately from the regression because it is a different question with a
different failure mode.

| | value |
|---|---|
| recall | 0.996 |
| missed (needs water, told otherwise) | **0.42%** |
| false alarms | 27.61% |

Conservatively biased by design. A false positive wastes an irrigation; a false
negative costs yield that cannot be recovered. In a monsoon the 27.61% false
alarm rate is also correct behaviour — the field is at field capacity and rain is
likely, and the model will not know that with certainty.

### The weak folds

**KOL, R² = −0.2442, MAE 4.95 mm.** 87.2% of its rows need no water, against
39.8% in the training set. Kolkata averages 9.07 mm/day of rain against 3.89 mm
elsewhere. The model is asked to predict a nearly-constant target in a climate
with no analogue in training. This is covariate shift, and it is a real weakness
rather than a metric artefact — though R² on a fold that is 87% zeros is
unstable by construction, and the operational number is the 11.12 mm error on
KOL rows that *do* need water, against 4.34 mm overall.

**CHN, R² = 0.2921, MAE 12.12 mm.** Hotter and wetter than the training mean
(ET0 4.43 against 3.60 mm/day), with 15.68 mm error when water is needed. The
same failure, less extreme.

Both point at the same gap: the corpus has 12 stations, and a station wetter or
hotter than all of them is outside the training distribution. More stations, or
climatic stratification of the corpus, is the fix.

**Groundnut, R² = 0.5891.** Weakest crop fold, but the distributional metrics are
healthy: median absolute error 1.67 mm and 80.4% of rows within 10 mm. The R² is
depressed by the same zero-heavy structure.

---

## Model

`HistGradientBoostingRegressor`, squared-error loss, 400 iterations at learning
rate 0.08, minimum 40 samples per leaf, L2 1.0, early stopping on a 10%
validation slice.

Squared error is a choice. The target is a quantity of water in millimetres, and
4 mm of error is 4 mm whether the field needs 2 mm or 60 mm. The corpus is 42%
exact zeros, so MAE is reported alongside RMSE, and the error on rows that need
water is reported alongside the error overall.

`irrigation_method` is one-hot against a **fixed declared domain**
(`CATEGORICAL_VALUES`), not one inferred from the corpus, so a re-run that lost a
method cannot silently reshape the estimator. A method absent from the corpus
becomes an all-zero column, which the tree treats as "never seen" rather than
crashing on.

---

## Known limitations

- **Soil texture is assigned, not measured.** Twelve FAO-56 texture classes,
  allocated per field specification. This is the weakest input and the external
  validation localises the error: the simulated moisture *level* is off by up to
  0.1 m³/m³ per station in both directions, while the seasonal *dynamics* are
  sound. `SoilGrids` is the fix and is the recommended next integration; the API
  was unreachable from the build environment.
- **GDD stage thresholds are uncalibrated.** FAO-56 publishes stage lengths in
  days, not accumulated thermal time. The GDD thresholds are a stated
  assumption, and phenology will be wrong for varieties whose thermal time
  differs from the table.
- **Two irrigation methods are absent from the corpus.** `Flood` and `Paddy`
  produce constant one-hot columns. Rice on a paddy is a significant Indian case
  and its absence is a real gap.
- **There is no labelled field data.** Every water number traces to NASA POWER,
  the FAO-56 tables, or this simulator. The ERA5 cross-check corroborates the
  atmosphere and the soil dynamics; it cannot validate the FAO-56 coefficients,
  which remain global averages against a stated ±5%.
- **The optimiser's event cost is in millimetres, not currency.** A caller that
  knows the cost of mobilising a pump should pass it; the default is a stated
  figure, not a fitted one.
