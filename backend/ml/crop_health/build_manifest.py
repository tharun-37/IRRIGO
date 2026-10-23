"""
Build the crop-health dataset manifest.

The blueprint requires a manifest describing every dataset used for image and
crop-health work before any training can be claimed. This script walks
`backend/ml/datasets/`, hashes and profiles each file, and writes
`manifest.json` next to this module. It reruns any time a dataset changes.

Crop-*image* datasets (photos with verified labels) do not exist in this
repository yet. They are listed in `IMAGE_COLLECTIONS` as status `pending` so
the gap is visible in the manifest rather than hidden by its absence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import datetime as _dt
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"

#: The image collections the blueprint's crop-health MVP needs. None of them
#: exist locally yet; their entries stay `pending` until real, consented,
#: verified-label data lands.
IMAGE_COLLECTIONS = [
    {
        "collection": "leaf-disease-first-crop",
        "purpose": "disease classification baseline, first pilot crop",
        "status": "pending",
        "needs": "field photos with expert-verified labels; group splits by "
        "plant/capture session to prevent near-duplicate leakage",
    },
    {
        "collection": "pest-presence-first-crop",
        "purpose": "pest detection, separate task from disease",
        "status": "pending",
    },
    {
        "collection": "nutrient-symptoms-first-crop",
        "purpose": "nutrient deficiency symptoms, crop-specific labels only",
        "status": "pending",
        "needs": "crop-specific expert review; no universal nutrient claim",
    },
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_csv(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        rows = sum(1 for row in reader)
    return {"columns": header, "rows": rows}


def main() -> None:
    datasets: list[dict] = []
    for path in sorted(DATASETS_DIR.glob("*.csv")):
        entry: dict = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
            "status": "present",
        }
        try:
            entry.update(profile_csv(path))
        except Exception as error:
            entry["readError"] = str(error)
        datasets.append(entry)

    manifest = {
        "generatedAt": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "generator": "backend/ml/crop_health/build_manifest.py",
        "splitRules": {
            "trainValidationTest": "split before any tuning, kept in three disjoint groups",
            "grouping": "by plant / farm / capture session / source, so near-duplicates never straddle splits",
            "independentTest": "a held-out field-photo test set never used for architecture or threshold selection",
        },
        "tabularDatasets": datasets,
        "imageCollections": IMAGE_COLLECTIONS,
        "imageModelStatus": {
            "deployed": False,
            "note": "no crop image dataset, no trained image artifact; the "
            "crop-health service runs quality gates and abstains on "
            "classification until a validated artifact is installed",
        },
    }

    out = HERE / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(datasets)} tabular datasets)")


if __name__ == "__main__":
    main()
