# Crop-health artifact directory

No model artifact is deployed here yet — that is a fact, not a placeholder, and
the crop-health service reports it that way (`deployedModel: false`, status
`pending`).

## Contract (when a model exists)

Two files land in this directory together; a model without both is treated as
absent:

1. `crop_health_model.joblib` — a joblib-serialised object exposing
   `predict(X)` (and optionally `predict_proba(X)`) where `X` is one row of six
   scalars, in this order:

   | # | feature        | meaning                                            |
   |---|----------------|----------------------------------------------------|
   | 1 | lumaMean       | mean brightness, 0-255                             |
   | 2 | lumaStd        | brightness spread                                  |
   | 3 | sharpness      | variance of the Laplacian of the greyscale image   |
   | 4 | greenFraction  | fraction of pixels reading leaf-green              |
   | 5 | darkFraction   | fraction of pixels under luma 60                   |
   | 6 | yellowFraction | fraction of pixels reading yellow                  |

2. `model_meta.json`:

   ```json
   {
     "modelVersion": "crop-disease-baseline-1",
     "task": "disease",
     "classes": ["leaf_blight", "rust", "healthy"],
     "crops": ["rice"],
     "calibrated": false,
     "calibrationMethod": null,
     "validationStatus": "uncalibrated — under test on held-out field set",
     "limitations": ["..."]
   }
   ```

## Rules the deployment must honour

- `calibrated` stays `false` until a method (e.g. temperature scaling on a
  held-out set) is documented in a model card; raw softmax output is never
  called a probability.
- `validationStatus` is a status label (`implemented` / `under test` /
  `pending` / `proposed`), never an accuracy claim.
- `classes`/`crops` drive the service: a crop outside `crops` returns
  `unsupported_crop`; a confidence below the abstention threshold returns
  `uncertain`.
- Large artifacts excluded from git if policy requires — the manifest and meta
  travel, weights move by copy.
