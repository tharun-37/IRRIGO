"""
Tests for the agronomic rule engine that generates the disease labels.

These exist to make one uncomfortable fact impossible to forget: the disease
labels in the training corpus are not observations. They are the output of a
deterministic rule engine over six climate and soil features, published in
``artifacts/disease_thresholds.json``. The classifier that later learns those
labels can only be approximating a lookup table that ships with the project.

That is not a bug, and the corpus is not useless. It is a documented limit on
what the disease model can mean, and the reason its scores must never be
presented as diagnostic performance. The test below pins the fact in place so a
future change cannot quietly make the disease model look better than it is.

Run:
    python backend/ml/tests/test_disease_labels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

import dataset_builder as db  # noqa: E402

ARTIFACT_DIR = ML_DIR / "artifacts"
DATASET_DIR = ML_DIR / "datasets"

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label:<52} {detail}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label:<52} {detail}")


def load_thresholds() -> tuple[db.RuleThresholds, dict]:
    path = ARTIFACT_DIR / "disease_thresholds.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run backend/ml/scripts/train.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return db.RuleThresholds.from_dict(payload["thresholds"]), payload


def main() -> int:
    print("=" * 78)
    print("Disease label provenance: are these labels observed or synthesised?")
    print("=" * 78)

    thresholds, payload = load_thresholds()

    print("\nlabel source")
    description = f"{payload.get('description', '')} {payload.get('method', '')}".lower()
    check(
        "thresholds artefact declares itself a rule engine",
        "rule" in description,
        payload.get("description", "")[:60],
    )
    check(
        "artefact carries a diagnostic caveat",
        "not a diagnostic" in payload.get("caveat", "").lower(),
    )
    check(
        "artefact publishes its rules in plain text",
        len(payload.get("rules", [])) >= 5,
        f"{len(payload.get('rules', []))} rules",
    )

    corpus_path = DATASET_DIR / "training_corpus.csv"
    if not corpus_path.exists():
        raise SystemExit(f"{corpus_path} not found. Run the trainer first.")

    corpus = pd.read_csv(corpus_path, low_memory=False)
    print(f"\ncorpus rows: {len(corpus):,}")

    # The six features the rule engine reads. If the label is reproducible from
    # exactly these, it carries no information beyond them.
    rule_inputs = [
        "temperature", "humidity", "soil_moisture", "soil_ph",
        "nitrogen", "light_intensity",
    ]
    missing = [c for c in rule_inputs if c not in corpus.columns]
    check("all rule inputs present in corpus", not missing, f"missing: {missing or 'none'}")

    # Re-derive every label from the rule engine and compare with the stored
    # column. Agreement near 1.0 proves the label is a deterministic function of
    # the features, which is the whole point of this file.
    sample = corpus.dropna(subset=rule_inputs + ["target_disease"])
    if len(sample) > 20_000:
        sample = sample.sample(20_000, random_state=7)
    derived = np.array(
        [db.classify_disease(row, thresholds) for row in sample.to_dict("records")]
    )
    stored = sample["target_disease"].to_numpy()
    agreement = float((derived == stored).mean())

    print("\nleakage check: can the rule engine reproduce the labels?")
    check(
        "labels are reproducible from the rule inputs",
        agreement > 0.999,
        f"agreement {agreement * 100:.2f}% over {len(sample):,} rows",
    )
    check(
        "therefore the disease model is a rule approximation",
        agreement > 0.999,
        "any high disease score measures the rule table, not agronomy",
    )

    # The unseen-station score was a perfect 1.0, which is what a lookup table
    # produces and what a real classifier essentially never produces. Assert the
    # condition that makes such a number meaningless, so nobody quotes it.
    print("\nconsequence: the unseen-station disease score cannot be a generalisation claim")
    check(
        "perfect disease scores are explained by determinism",
        agreement > 0.999,
        "a memorised table transfers perfectly; a learned one usually does not",
    )

    # Sanity check the engine itself, so this file is not only a caveat.
    print("\nrule engine behaves as documented")
    healthy = {
        "temperature": 22.0, "humidity": 45.0, "soil_moisture": 20.0,
        "soil_ph": 6.6, "nitrogen": 60.0, "light_intensity": 40_000.0,
        "field_capacity_pct": 42.0,
    }
    check(
        "mild conditions resolve to Healthy",
        db.classify_disease(healthy, thresholds) == "Healthy",
        db.classify_disease(healthy, thresholds),
    )

    root_rot = dict(healthy, soil_moisture=41.0, soil_ph=5.2)
    check(
        "waterlogged acidic soil resolves to Root_Rot",
        db.classify_disease(root_rot, thresholds) == "Root_Rot",
        db.classify_disease(root_rot, thresholds),
    )

    mildew = dict(healthy, temperature=23.0, humidity=80.0, light_intensity=20_000.0)
    check(
        "warm humid shaded canopy resolves to Powdery_Mildew",
        db.classify_disease(mildew, thresholds) == "Powdery_Mildew",
        db.classify_disease(mildew, thresholds),
    )

    blight = dict(healthy, temperature=31.0, humidity=75.0, nitrogen=20.0)
    check(
        "hot humid nitrogen-poor canopy resolves to Early_Blight",
        db.classify_disease(blight, thresholds) == "Early_Blight",
        db.classify_disease(blight, thresholds),
    )

    bacterial = dict(healthy, temperature=29.0, humidity=88.0, soil_ph=7.9)
    check(
        "hot humid alkaline soil resolves to Bacterial_Leaf_Spot",
        db.classify_disease(bacterial, thresholds) == "Bacterial_Leaf_Spot",
        db.classify_disease(bacterial, thresholds),
    )

    # Every class the engine can emit must be one the project declares.
    check(
        "engine output stays inside the declared class list",
        set(derived) <= set(db.DISEASE_CLASSES) if hasattr(db, "DISEASE_CLASSES") else True,
        f"observed: {sorted(set(derived))}",
    )

    print("\n" + "=" * 78)
    if FAILED:
        print(f"{FAILED} check(s) failed, {PASSED} passed")
        return 1
    print(f"All {PASSED} checks passed.")
    print()
    print("Interpretation: the disease model is a transparent restatement of")
    print("artifacts/disease_thresholds.json. It is a stress-risk indicator for")
    print("the dashboard and must never be presented as a crop diagnosis.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
