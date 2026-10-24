# Model quick-start: get a deployed crop-health model today

This is the **easiest path** from "no model" to "the app returns real verdicts".
No GPU, no deep learning: the serving slot takes a small sklearn classifier
over six scalar image statistics that the service already computes for every
upload.

## The contract (small by design)

`backend/app/services/crop_health_service.py` loads a deployed model from
`backend/models/crop_health/` — exactly two files:

| File                    | Contents                                                            |
|-------------------------|---------------------------------------------------------------------|
| `crop_health_model.joblib` | a sklearn model exposing `predict(X)` (and ideally `predict_proba(X)`) |
| `model_meta.json`       | `modelVersion`, `task`, `classes`, `crops`, `calibrated`, `validationStatus`, `limitations` |

`X` is one row of six scalars, in this order (see `app/services/image_storage.analyse_pixels`):

1. `lumaMean` — mean brightness (0–255)
2. `lumaStd` — brightness spread
3. `sharpness` — Laplacian variance of the greyscale image
4. `greenFraction` — fraction of leaf-green pixels
5. `darkFraction` — fraction of pixels under luma 60
6. `yellowFraction` — fraction of yellow-reading pixels

How the service then behaves (this is why it never guesses):

- a crop outside `meta.crops` → status `unsupported_crop` (no label named)
- confidence below `ABSTENTION_CONFIDENCE = 0.55` → status `uncertain`
- otherwise → status `result`, with the honest "verify in the field" advice

## Three steps (one command each)

### 1. Get a dataset, folder-per-class

Any GitHub clone or Kaggle download organised as:

```
dataset/
├── Bacterial leaf blight/   ← folder name = class label
│   ├── DSC_0400.JPG
│   └── ...
├── Brown spot/
└── Leaf smut/
```

Examples that work out of the box:

- **Rice**: `git clone --depth 1 https://github.com/Tanwar-12/DETECTING-RICE-LEAF-DISEASES.git` (its `DATASET/` folder)
- **Cotton**: Kaggle `seroshkarim/cotton-leaf-disease-dataset`
- **Potato**: Kaggle `emmarex/plantdisease` (PlantVillage subset)
- More per-crop sources: `docs/datasets-ppt-outline.md`

### 2. Train (one command)

From `backend/`, with the project venv:

```
python ml/scripts/train_crop_health_baseline.py --data <dataset_root> --crop rice
```

The script:

- computes the **same six statistics** for every image (train = serve, no drift)
- keeps only images that would pass the service's four quality gates
  (blur/exposure/content/resolution) and prints how many were filtered
- trains a `RandomForestClassifier(300, class_weight=balanced)` on a stratified
  75/25 split and prints holdout accuracy, macro-F1 and per-class P/R/F1
- writes the two contract files plus `training_log.txt` into
  `backend/ml/crop_health/artifacts/` (the model-side home, NOT the serving dir)

Example output shape:

```
dataset      : ...\DATASET
class counts : {'Bacterial leaf blight': 17, 'Brown spot': 11, 'Leaf smut': 6}
usable       : 34 of 120 images (the rest fail the same quality gates ...)
holdout accuracy : 0.889   macro-F1: 0.852   (n_test=9)
```

### 3. Deploy (one flag) and verify

```
python ml/scripts/train_crop_health_baseline.py --data <root> --crop rice --deploy
```

`--deploy` copies the artifact into `backend/models/crop_health/`. Restart
uvicorn, then check:

```
GET /api/crop-health/status   →   "deployedModel": true, "modelVersion": "crop-baseline-rice-..."
```

Upload a photo in the app → the widget/page can now return a `result` verdict
for the trained crop, and `unsupported_crop` for crops the dataset never
covered (this is correct behaviour, not a bug).

## Honest guardrails

- **Six statistics is a coarse lens.** It cannot see lesion shape or spread;
  it is a baseline that makes the pipeline real end-to-end, and every verdict
  ships with "not a confirmed diagnosis" advice.
- **`calibrated` stays `false`.** Per the deployment rules, raw probabilities
  are never called calibrated probabilities; the 0.55 abstention threshold is
  the only confidence control. Calibration (temperature scaling on a held-out
  field set) is the next validation step, not a flag to flip by hand.
- **Random splits are baselines.** The manifest's split rules require
  session-grouped splits on field photos; public lab-style datasets can only
  be "under test", and the meta says so.
- **Deployment moves by copy.** Git carries the meta and training log, not
  the weights (see `backend/ml/crop_health/artifacts/README.md`).

## Later upgrades (same contract, stronger model)

- More/better classes: merge datasets per crop (e.g. the 5,932-image Kaggle
  rice set) and re-record — the script treats each subfolder as a class label,
  so class names are simply folder names.
- A real CNN: `ml/scripts/leaf_cnn_train.py` (torch) already exists for
  full-image CNN training; exporting it behind the same six-scalar interface
  (e.g. probabilities from image pixels reduced to the scalar summary) is the
  documented upgrade path — or extend the service slot with a CNN artifact
  contract version bump.
- Calibration + field validation then flips `validationStatus` honestly, not
  the `calibrated` boolean.
