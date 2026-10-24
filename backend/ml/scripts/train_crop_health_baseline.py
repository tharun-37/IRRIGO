"""One-command baseline trainer for the crop-health inference slot.

Turns any folder-per-class image dataset (e.g. a GitHub or Kaggle disease set)
into the two-file artifact `app/services/crop_health_service.py` loads:

    crop_health_model.joblib   a sklearn model exposing predict()/predict_proba()
    model_meta.json            the side-car the service reads (classes, crops, ...)

The features are *exactly* the six scalar statistics the service computes at
serve time (`app/services/image_storage.analyse_pixels`), and only images that
would pass the same quality gates are trained on, so train and serve see the
same distribution. No deep learning, no GPU: a RandomForest over six scalars.

Usage (from backend/, with the project venv):

    python ml/scripts/train_crop_health_baseline.py --data <dataset_root> --crop rice

    # when satisfied with the printed metrics, deploy into the serving dir:
    python ml/scripts/train_crop_health_baseline.py --data <root> --crop rice --deploy

Honesty rules honoured (see backend/ml/crop_health/artifacts/README.md):
- `calibrated` stays false (no calibration method has been run on a held-out set);
- `validationStatus` is a status label, never an accuracy claim;
- the random-split caveat is written into the meta, because the manifest's
  split rules (group by capture session) cannot be honoured with public,
  pre-collected datasets alone.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))

from PIL import Image, ImageOps  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import f1_score, precision_recall_fscore_support  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from app.services.image_storage import analyse_pixels  # noqa: E402

try:  # gate thresholds come from the service so the two sides cannot drift
    from app.services.crop_health_service import (  # noqa: E402
        BLUR_VARIANCE_MIN,
        LUMA_BRIGHT_MIN,
        LUMA_DARK_MAX,
        LUMA_STD_FLAT_MAX,
    )
except Exception:  # pragma: no cover - fallback only
    BLUR_VARIANCE_MIN = 200.0
    LUMA_BRIGHT_MIN = 245.0
    LUMA_DARK_MAX = 35.0
    LUMA_STD_FLAT_MAX = 12.0

FEATURE_ORDER = (
    "lumaMean",
    "lumaStd",
    "sharpness",
    "greenFraction",
    "darkFraction",
    "yellowFraction",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
LOG_WIDTH = 62


def hr(char: str = "-") -> str:
    return char * LOG_WIDTH


def gate_reasons(stats: dict) -> list[str]:
    """Mirror of the service's four quality gates (crop_health_service.py)."""
    reasons = []
    if stats["width"] < 240 or stats["height"] < 240:
        reasons.append("resolution")
    if stats["sharpness"] < BLUR_VARIANCE_MIN:
        reasons.append("blur")
    if not (LUMA_DARK_MAX <= stats["lumaMean"] <= LUMA_BRIGHT_MIN):
        reasons.append("exposure")
    if stats["lumaStd"] < LUMA_STD_FLAT_MAX:
        reasons.append("content")
    return reasons


def load_pixels(path: Path) -> Image.Image:
    """Same minimal preprocessing the service applies before `analyse_pixels`."""
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def collect(data_root: Path) -> tuple[list[Path], list[str], Counter, list[str]]:
    """Read every image under each subfolder of `data_root` (folder = label)."""
    class_dirs = sorted(d for d in data_root.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders under {data_root}. Expected <root>/<class_name>/<images>.")

    feature_rows: list[list[float]] = []
    labels: list[str] = []
    skipped = Counter()
    skip_examples: list[str] = []
    for class_dir in class_dirs:
        label = class_dir.name
        images = [p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
        if not images:
            raise SystemExit(f"Class folder '{label}' holds no images ({class_dir}).")
        for path in sorted(images):
            try:
                image = load_pixels(path)
                stats = analyse_pixels(image)
            except Exception as error:  # unreadable file - count and continue
                skipped["unreadable"] += 1
                skip_examples.append(f"{path.name}: {error}")
                continue
            failed = gate_reasons(stats)
            if failed:
                skipped["gates:" + ",".join(failed)] += 1
                continue
            feature_rows.append([float(stats[name]) for name in FEATURE_ORDER])
            labels.append(label)
    return feature_rows, labels, skipped, skip_examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="folder-per-class image dataset root")
    parser.add_argument("--crop", required=True, help="crop the labels belong to (as the app spells it, e.g. Rice)")
    parser.add_argument(
        "--out",
        type=Path,
        default=BACKEND / "ml" / "crop_health" / "artifacts",
        help="where the artifact lands (default: backend/ml/crop_health/artifacts)",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="modelVersion string (default: crop-baseline-<crop>-<date>)",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="ALSO copy joblib + meta into the serving dir backend/models/crop_health",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(hr("="))
    print("crop-health baseline trainer (six-scalar RandomForest)")
    print(hr("="))
    rows, labels, skipped, skip_examples = collect(args.data)
    n_total = len(rows) + sum(skipped.values())
    print(f"dataset      : {args.data}")
    print(f"class counts : {dict(Counter(labels))}")
    print(f"usable       : {len(rows)} of {n_total} images (the rest fail the same quality gates the service enforces)")
    if skipped:
        print(f"skipped      : {dict(skipped)}")
        for example in skip_examples[:5]:
            print(f"               {example}")
    if len(set(labels)) < 2:
        raise SystemExit("Need at least two usable classes to train.")

    x_train, x_test, y_train, y_test = train_test_split(
        rows,
        labels,
        test_size=0.25,
        random_state=args.seed,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = float((y_pred == y_test).mean())
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=sorted(set(labels)), zero_division=0
    )

    print(hr("="))
    print(f"holdout accuracy : {accuracy:.3f}   macro-F1: {macro_f1:.3f}   (n_test={len(y_test)})")
    for name, p, r, f in zip(sorted(set(labels)), precision, recall, f1_per_class):
        print(f"  {name:<28} P={p:.2f}  R={r:.2f}  F1={f:.2f}")
    print(hr("="))
    print("NOTE: a random split of a public dataset is only ever a baseline.")
    print("      Per the manifest's split rules, near-duplicates from one")
    print("      capture session must not straddle splits - field photos")
    print("      with session grouping are what flips this to 'validated'.")

    classes = sorted(set(labels))
    model_version = args.model_version or f"crop-baseline-{args.crop.lower()}-{datetime.now().strftime('%Y%m%d')}"
    meta = {
        "modelVersion": model_version,
        "task": "disease",
        "classes": classes,
        "crops": [args.crop.strip().lower()],
        "calibrated": False,
        "calibrationMethod": None,
        "validationStatus": "uncalibrated - under test on held-out field set",
        "trainedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasetInfo": {
            "source": str(args.data),
            "imagesUsable": len(rows),
            "imagesAttempted": n_total,
            "skipped": dict(skipped),
            "split": "stratified 75/25 (random, NOT session-grouped)",
            "holdout": {
                "n": len(y_test),
                "accuracy": round(accuracy, 4),
                "macroF1": round(macro_f1, 4),
                "perClass": {
                    name: {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
                    for name, p, r, f in zip(sorted(set(labels)), precision, recall, f1_per_class)
                },
            },
        },
        "features": list(FEATURE_ORDER),
        "estimator": "RandomForestClassifier(n_estimators=300, class_weight=balanced)",
        "limitations": [
            "Baseline over six scalar image statistics; it cannot see lesion"
            " shape, size or spread. Treat every verdict as a hint, not a"
            " diagnosis.",
            "Trained on a random split of a pre-collected dataset; field-photo"
            " validation with session-grouped splits has not happened yet.",
            "Probabilities are uncalibrated (calibrated: false); the service's"
            " abstention threshold is the only confidence control.",
        ],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    joblib_path = args.out / "crop_health_model.joblib"
    meta_path = args.out / "model_meta.json"
    log_path = args.out / "training_log.txt"

    import joblib

    joblib.dump(model, joblib_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log_path.write_text(
        "\n".join(
            [
                hr("="),
                f"modelVersion           : {model_version}",
                f"source                 : {args.data}",
                f"classes                : {classes}",
                f"usable / attempted     : {len(rows)} / {n_total}",
                f"skipped by gates       : {dict(skipped)}",
                f"holdout n              : {len(y_test)}  (random 75/25, seed {args.seed})",
                f"holdout accuracy       : {accuracy:.4f}",
                f"holdout macro-F1       : {macro_f1:.4f}",
                "per-class              :",
                *[
                    f"  {name:<28} P={p:.2f}  R={r:.2f}  F1={f:.2f}"
                    for name, p, r, f in zip(sorted(set(labels)), precision, recall, f1_per_class)
                ],
                hr("="),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"artifact       : {joblib_path}")
    print(f"meta + log     : {meta_path.parent}")

    if args.deploy:
        serving = BACKEND / "models" / "crop_health"
        serving.mkdir(parents=True, exist_ok=True)
        shutil.copy2(joblib_path, serving / joblib_path.name)
        shutil.copy2(meta_path, serving / meta_path.name)
        print()
        print(hr("!"))
        print("DEPLOYED to the serving dir. Restart uvicorn and check")
        print("GET /api/crop-health/status -> deployedModel should be true.")
        print("The app will report statuses honestly: below 0.55 confidence")
        print("returns 'uncertain', never a guessed label.")
        print(hr("!"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
