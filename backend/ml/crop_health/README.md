# backend/ml/crop_health

Model-side home for the image-based crop health work. Status:

| Piece                        | State      |
|------------------------------|------------|
| Dataset manifest             | implemented (this directory) |
| Tabular dataset profiling    | implemented (build_manifest.py) |
| Image collections            | pending — none collected yet |
| Preprocessing module         | scalar-pixel statistics implemented in `app/services/image_storage.py` |
| Training script              | proposed |
| Evaluation script            | proposed |
| Deployed artifact            | none — the service abstains honestly |

Generate the manifest:

    python backend/ml/crop_health/build_manifest.py

The manifest hashes every CSV in `backend/ml/datasets/` and records the
collection gap for the *image* work, so nothing downstream can claim a trained
disease model exists before real, consented, expert-labelled field photos and a
held-out test set do. The serving contract the model must satisfy lives in
`artifacts/README.md`; honour it or the service will (correctly) refuse to
classify.
