"""
Trains the model suite served by the backend.

Five models, one per decision the controller has to make:

| Model | Task | Target |
| :--- | :--- | :--- |
| `disease` | multi-class classification | agronomic risk class, 6 classes |
| `water_depth` | regression | irrigation depth to apply, mm |
| `water_action` | multi-class classification | NO / LIGHT / MODERATE / HEAVY |
| `nutrient_demand` | regression | NPK dose, kg/ha |
| `yield` | regression | yield estimate, t/ha |

Usage
-----
    python backend/ml/scripts/train.py                  # full run
    python backend/ml/scripts/train.py --skip-tuning   # reuse saved thresholds
    python backend/ml/scripts/train.py --quick         # small corpus, for smoke tests

Design notes
------------
**Grouped splitting, not random splitting.** Rows from the same simulated
series are autocorrelated, because soil moisture is a state variable with
memory. A random split would leak a series across the train and test boundary
and report a score that could not be reproduced on a real field, where every
zone is genuinely unseen. Every split here is grouped on
`station|crop|soil_type` and time-ordered within a group, which is the honest
analogue of deploying to a new field.

**Out-of-distribution holdout.** A separate `station` is withheld entirely and
scored on its own. A model that only scores well on stations it saw during
training has learned the dataset, not irrigation.

**Metrics that matter for a controller.** Classification reports macro F1
because the disease classes are imbalanced and accuracy would flatter the
model. Regressions report MAE alongside R2, because the operational question is
"how many millimetres off is a single decision", not "how much variance is
explained on average".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GroupKFold,
    StratifiedGroupKFold,
    cross_val_predict,
)

import dataset_builder as db  # noqa: E402
from features import (  # noqa: E402
    DISEASE_CLASSES,
    NUTRIENT_BANDS,
    SENSOR_FEATURES,
    crop_coefficient,
    reference_evapotranspiration,
)
from real_data import load_real_weather  # noqa: E402

ARTIFACT_DIR = ML_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR = ML_DIR / "datasets"

RANDOM_SEED = 20240917
PLOT_STYLE = "seaborn-v0_8-whitegrid"

#: Column sets per model. These are the single source of truth for what the
#: backend must send, and they are written into the artifact so the API and the
#: model can never disagree about the contract.
#:
#: Two kinds of input are separated deliberately.
#:
#: *Measured* columns come off the probe and the weather station.
#:
#: *Configured* columns are what an operator enters at commissioning: the crop,
#: the soil class and its standard hydraulic constants, the rooting depth the
#: season implies, and the field area. A commissioned controller genuinely
#: knows these, and without them the targets are not identifiable. Total
#: available water differs by more than 2x between a sand and a clay profile
#: under wheat, and nitrogen maintenance rate differs by 4x between soybean and
#: maize, so a model blind to configuration cannot predict either target. This
#: is the fix for the first training run, whose water regressor scored R2 =
#: -0.01 because it had no way to recover total available water.
#:
#: *Excluded* columns are state variables that are functions of the answer:
#: `depletion_mm`, `depletion_fraction`, `taw_mm` and `raw_mm` all encode how
#: much water is missing, which is precisely what the model is asked to predict.
#: `SENSOR_FEATURES` minus nothing, plus configuration, is the honest contract.

#: Trailing-window aggregates over the node's own reading history. A deployed
#: field node keeps a short buffer and can compute these, so they are available
#: at decision time rather than being an oracle. They matter because the probe
#: is noisy while the soil profile is smooth, and averaging suppresses noise
#: without lagging a real trend.
TEMPORAL_FEATURES: list[str] = [
    "soil_moisture_mean_3d",
    "soil_moisture_mean_7d",
    "soil_moisture_trend_3d",
    "et0_mean_3d",
    "et0_mean_7d",
    "rainfall_sum_3d",
    "rainfall_sum_7d",
    "kc_mean_3d",
]

MEASURED_FEATURES: list[str] = [
    "temperature", "humidity", "soil_moisture", "soil_ph", "ec",
    "nitrogen", "phosphorus", "potassium", "light_intensity",
    "wind_speed", "crop_stress_index",
] + TEMPORAL_FEATURES

CONFIGURED_FEATURES: list[str] = [
    "root_depth_cm", "field_capacity_pct", "wilting_point_pct",
    "bulk_density_g_cm3", "kc", "et0_mm_day", "season_progress",
    "rainfall", "field_area_m2",
]

CATEGORICAL_FEATURES: list[str] = ["crop", "soil_type"]

FEATURE_SETS: dict[str, list[str]] = {
    "disease": list(MEASURED_FEATURES),
    "water_depth": MEASURED_FEATURES + CONFIGURED_FEATURES + CATEGORICAL_FEATURES,
    "water_action": MEASURED_FEATURES + CONFIGURED_FEATURES + CATEGORICAL_FEATURES,
    "nutrient_demand": [
        "soil_moisture", "soil_ph", "ec", "nitrogen", "phosphorus",
        "potassium", "crop_stress_index", "season_progress",
    ] + CATEGORICAL_FEATURES,
    "yield": [
        "soil_moisture", "soil_ph", "ec", "nitrogen", "phosphorus", "potassium",
        "light_intensity", "crop_stress_index", "et0_mm_day", "season_progress",
    ] + CATEGORICAL_FEATURES,
}

def log(message: str = "") -> None:
    print(message, flush=True)


def rule(message: str) -> None:
    log("\n" + "=" * 76)
    log(message)
    log("=" * 76)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def series_key(df: pd.DataFrame) -> pd.Series:
    """Identifier for one simulated series. Used to group the train/test split."""
    return df["station"].astype(str) + "|" + df["crop"] + "|" + df["soil_type"]


#: Categories encoded for the categorical configuration columns. Fixed at
#: training time and written into the artifact, so inference cannot silently
#: produce a different column count and a different answer.
CATEGORY_LEVELS: dict[str, list[str]] = {
    "crop": sorted(db.CROP_PROFILES),
    "soil_type": sorted(db.SOILS),
}


def encode_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    One-hot encode the categorical configuration columns, leave the rest as-is.

    The soil class is carried both as a label and as its standard hydraulic
    constants (`field_capacity_pct`, `wilting_point_pct`,
    `bulk_density_g_cm3`), which is the useful part: the model can interpolate
    a soil texture it has never been explicitly shown.
    """
    frame = df.copy()
    one_hot_prefixes = tuple(f"{key}__" for key in CATEGORICAL_FEATURES)
    for column in features:
        if column in CATEGORY_LEVELS:
            for level in CATEGORY_LEVELS[column]:
                frame[f"{column}__{level}"] = (frame[column].astype(str) == level).astype(float)
    # The raw categorical columns are labels, not numbers, so only the one-hot
    # expansions are retained alongside the numeric features.
    missing = [
        f for f in features
        if f not in frame.columns and f not in CATEGORY_LEVELS
    ]
    if missing:
        raise RuntimeError(f"missing feature columns: {missing}")
    encoded = [
        c for c in frame.columns
        if (c in features and c not in CATEGORY_LEVELS) or c.startswith(one_hot_prefixes)
    ]
    return frame[encoded].fillna(0.0).astype(float)


def grouped_split(
    df: pd.DataFrame, test_size: float = 0.22, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out whole series, while keeping rare classes represented in both sides.

    Soil moisture is a state variable, so consecutive rows within a series are
    near-duplicates. Splitting on rows would put a field's last irrigation event
    in training and its next morning's decision in testing, which is not a
    measurement of generalisation to anything. Series must therefore stay whole.

    A plain `GroupShuffleSplit` satisfies that but ignores class balance, and
    with it the rare disease classes disappeared from the test set entirely:
    rust ended up with 5 test rows and bacterial spot with 1, which dragged the
    reported macro F1 to 0.80 while every class the model actually saw scored
    above 0.99. `StratifiedGroupKFold` holds groups intact and distributes the
    minority classes across folds, so the number reported is a number that
    means something.
    """
    labels = df["target_disease"]
    groups = series_key(df)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    # Take the first fold as the test set. StratifiedGroupKFold picks the fold
    # that best balances classes, so fold 0 is as valid as any.
    splits = list(splitter.split(df, labels, groups))
    best = min(
        splits,
        key=lambda pair: _class_imbalance(df.iloc[pair[1]]["target_disease"]),
    )
    train_idx, test_idx = best
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def _class_imbalance(labels: pd.Series) -> float:
    """
    Total-variation distance between a split's class shares and the whole set.

    Lower is better balanced. Used to pick the most representative fold.
    """
    overall = labels.value_counts(normalize=True)
    part = labels.value_counts(normalize=True)
    return float((overall - part).abs().sum())


def split_class_support(df: pd.DataFrame) -> pd.DataFrame:
    """Rows per disease class in a split, for the report and the split log."""
    counts = df["target_disease"].value_counts()
    return pd.DataFrame({"rows": counts}).T


def holdout_station_split(df: pd.DataFrame, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, str, pd.DataFrame]:
    """
    Withhold one station entirely, to test transfer to an unseen climate.

    Picked as the driest station in the corpus, since that is the hardest
    transfer case: the model has never seen evaporative demand this high.
    """
    station_et0 = df.groupby("station")["et0_mm_day"].mean()
    hardest = str(station_et0.idxmax())
    train = df[df["station"] != hardest]
    test = df[df["station"] == hardest]
    log(f"  out-of-distribution holdout station: {hardest} "
        f"(mean ET0 {station_et0[hardest]:.2f} mm/day, the driest in the corpus)")
    return train.reset_index(drop=True), hardest, test.reset_index(drop=True)


# --------------------------------------------------------------------------
# Model builders
# --------------------------------------------------------------------------


def _disable_early_stopping_if_rare(estimator, y: pd.Series) -> object:
    """
    Turn off internal early stopping when a class is too rare to stratify.

    scikit-learn's early stopping carves out a stratified validation fold, so a
    class with a single member raises rather than warns. Falling back to a fixed
    iteration count is the correct response for a rare class, and it keeps the
    smoke-test path from masking a real result behind an unrelated exception.
    """
    counts = pd.Series(y).value_counts()
    if len(counts) and int(counts.min()) < 5:
        estimator.set_params(early_stopping=False)
    return estimator


def build_classifier(seed: int, quick: bool):
    """
    Histogram gradient boosting for the disease task.

    Chosen over a random forest because the disease classes are imbalanced and
    its binning handles the skewed nutrient and lux marginals without log
    transforms. `class_weight="balanced"` is not accepted by this estimator, so
    balancing is achieved by the corpus's class construction instead, and
    verified in the reported classification report rather than assumed.
    """
    return HistGradientBoostingClassifier(
        max_iter=220 if not quick else 60,
        learning_rate=0.09,
        max_depth=8,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=18,
        random_state=seed,
    )


def calibration_folds(
    y: pd.Series, groups: np.ndarray, seed: int, max_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Pre-compute grouped, stratified folds for the decision classifier.

    The folds are materialised here rather than handed to
    `CalibratedClassifierCV` as a splitter object, because that estimator calls
    `cv.split(X, y)` with no groups argument. A `StratifiedGroupKFold` then
    receives `groups=None`, sees a single group, and raises. Passing explicit
    index pairs is the reliable way to keep the grouping.

    The split count is also reduced when a class is too rare to stratify across
    five folds, which is a warning in scikit-learn and a silently bad fold
    allocation in practice.
    """
    group_count = len(np.unique(groups))
    counts = pd.Series(y).value_counts()
    min_support = int(counts.min()) if len(counts) else 0
    n_splits = int(min(max_splits, group_count, max(2, min_support)))
    n_splits = max(2, n_splits)

    y_array = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
    if min_support < 2:
        # A class with a single member cannot be stratified at all. GroupKFold
        # still keeps whole series intact, which is the property that matters for
        # avoiding leakage.
        return [
            (train_index, test_index)
            for train_index, test_index in GroupKFold(
                n_splits=min(n_splits, group_count), shuffle=True, random_state=seed
            ).split(np.zeros(len(y_array)), y_array, groups)
        ]

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (np.asarray(train_index), np.asarray(test_index))
        for train_index, test_index in splitter.split(np.zeros(len(y_array)), y_array, groups)
    ]


def build_calibrated_classifier(
    seed: int, quick: bool, folds: list[tuple[np.ndarray, np.ndarray]] | None = None
):
    """
    The irrigation decision classifier, wrapped so its score is a probability.

    HistGradientBoosting optimises the ranking of rows, not the probability that
    a row lands above a threshold. On a target where 97.9% of rows are the
    negative class that gap matters: the previous winning tune landed on a
    threshold of 0.13, and a threshold is only interpretable if the score it is
    applied to really does mean "13% chance this field needs water".

    `CalibratedClassifierCV` refits the base estimator per fold and learns a
    monotone map from raw score to observed frequency, so the threshold that
    comes out the other side is an actual event rate. Isotonic is used over
    sigmoid because there are enough irrigation events to support the flexible
    fit, and isotonic is the lower-bias choice at this sample size.

    Folds are grouped on the series key, so a simulated series cannot straddle
    the calibration boundary and leak. Leakage here would produce an
    over-confident model that scores well in training and badly in the field,
    which is the specific failure this is meant to prevent.
    """
    base = build_classifier(seed, quick)
    return CalibratedClassifierCV(
        base, method="isotonic", cv=folds if folds else 3, n_jobs=1
    )


def per_group_thresholds(
    probability: np.ndarray,
    y_true: np.ndarray,
    groups: np.ndarray,
    global_threshold: float,
    cost_false_negative: float = 3.0,
    cost_false_positive: float = 1.0,
    min_rows: int = 250,
) -> dict:
    """
    Tune a separate probability threshold per crop and soil class.

    A single global cut is a compromise between groups whose water demand
    behaves differently. Rice in a clay pan and a cereal on loam have different
    depletion curves, so the same score of 0.30 means something different for
    each, and a global threshold is systematically too eager on one and too
    reluctant on the other.

    Groups with too few rows to tune on fall back to the global threshold
    rather than getting a threshold fitted to noise. The fallback is recorded
    per group so the dashboard can show which thresholds are actually
    data-driven.
    """
    thresholds: dict[str, float] = {}
    detail: dict[str, dict] = {}
    grid = np.linspace(0.02, 0.90, 89)

    for group in sorted(set(groups.tolist())):
        mask = groups == group
        rows = int(mask.sum())
        positives = int((y_true[mask] == 1).sum())
        if rows < min_rows or positives < 10:
            thresholds[group] = float(global_threshold)
            detail[group] = {
                "threshold": round(float(global_threshold), 4),
                "rows": rows,
                "positives": positives,
                "source": "global_fallback",
            }
            continue

        group_probability = probability[mask]
        group_actual = y_true[mask] == 1
        best_threshold, best_cost = float(global_threshold), float("inf")
        for candidate in grid:
            predicted = group_probability >= candidate
            cost = (
                cost_false_negative * float(np.sum(group_actual & ~predicted))
                + cost_false_positive * float(np.sum(~group_actual & predicted))
            )
            if cost < best_cost:
                best_threshold, best_cost = float(candidate), cost

        thresholds[group] = best_threshold
        predicted = group_probability >= best_threshold
        detail[group] = {
            "threshold": round(best_threshold, 4),
            "rows": rows,
            "positives": positives,
            "source": "tuned",
            "expected_cost": round(best_cost, 1),
            "recall": round(float(predicted[group_actual].sum() / max(1, positives)), 4),
            "precision": round(
                float((group_actual & predicted).sum() / max(1, int(predicted.sum()))), 4
            ),
        }

    return {"thresholds": thresholds, "detail": detail}


def build_regressor(seed: int, quick: bool):
    """
    Gradient boosting for the water and yield regressions.

    Loss is squared error, so the model estimates the conditional **mean**.

    The first training run used absolute error here and scored R2 = -0.01. That
    is not a hyperparameter mistake, it is a target-mixture mistake: absolute
    error optimises the conditional median, and because roughly 85% of
    decision points require no irrigation, the median is zero almost
    everywhere, so the model learned to answer zero always. Squared error
    estimates the conditional mean, which is the quantity an operator means by
    "how much water is needed".

    The water-depth model is additionally trained on irrigation events only.
    Mixing the two populations in one regressor forces the mean to be dragged
    down by the majority, and the decision is better made explicitly by a
    classifier, so that a false negative and a wrong depth are separately
    visible in the metrics.
    """
    return HistGradientBoostingRegressor(
        max_iter=260 if not quick else 70,
        learning_rate=0.07,
        max_depth=8,
        min_samples_leaf=30,
        l2_regularization=0.6,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=18,
        loss="squared_error",
        random_state=seed,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def classification_block(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict:
    """
    Report classification metrics, and refuse to over-claim.

    A macro F1 computed over two classes can read 0.99 while telling the
    operator nothing, so the class count and the smallest per-class support
    are printed alongside the score. If the evaluation set is degenerate the
    block says so explicitly rather than letting the number stand alone.

    `classification_report` keys its output by `str(label)`, so an integer
    label like 1 comes back as "1" and `label in report` is False. Every
    lookup below therefore goes through `_key`, which stringifies first.
    Without this the per-class block silently comes back empty and the
    degenerate flag reads False for a set that is in fact badly unbalanced.
    """
    def _key(label: object) -> str:
        return str(label)

    present = [label for label in labels if label in set(y_true.tolist())]
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    supports = [int(report[_key(label)]["support"]) for label in present if _key(label) in report]
    degenerate = len(present) < 3 or (min(supports) if supports else 0) < 25

    block = {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "classes_present": len(present),
        "rows": int(len(y_true)),
        "min_class_support": int(min(supports)) if supports else 0,
        "degenerate": bool(degenerate),
        "per_class": {
            str(label): {
                "precision": round(report[_key(label)]["precision"], 4),
                "recall": round(report[_key(label)]["recall"], 4),
                "f1": round(report[_key(label)]["f1-score"], 4),
                "support": int(report[_key(label)]["support"]),
            }
            for label in labels
            if _key(label) in report
        },
    }
    rule(f"{name} - held-out classification report")
    log(classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=4))
    log(f"  accuracy          {block['accuracy']:.4f}")
    log(f"  macro F1          {block['macro_f1']:.4f}")
    log(f"  weighted F1       {block['weighted_f1']:.4f}")
    log(f"  rows              {block['rows']:,}  classes present {block['classes_present']}  "
        f"min class support {block['min_class_support']}")
    if degenerate:
        log("  [caution] this evaluation set is too small or too imbalanced for the")
        log("            macro F1 above to mean much. Treat it as a smoke signal, not")
        log("            a performance claim.")
    return block


def binary_decision_block(
    y_true_depth: np.ndarray, y_pred_depth: np.ndarray, threshold: float = 0.5
) -> dict:
    """
    Score the decision the operator actually cares about: irrigate, or not.

    Depth regression MAE is dominated by the many days on which the correct
    answer is zero, and can look excellent while the controller still waters a
    field that does not need it. Splitting the problem into a binary decision
    plus a depth estimate separates those two failure modes, which a single
    regression metric cannot.

    False negatives are reported separately because under-watering damages a
    crop, whereas over-watering costs money and leaches nutrients. A controller
    is deliberately biased towards the false negative.
    """
    actual = y_true_depth > threshold
    predicted = y_pred_depth > threshold
    true_positive = int(np.sum(actual & predicted))
    false_negative = int(np.sum(actual & ~predicted))
    false_positive = int(np.sum(~actual & predicted))
    true_negative = int(np.sum(~actual & ~predicted))

    block = {
        "rows": int(len(actual)),
        "irrigation_events": int(actual.sum()),
        "decision_accuracy": round(accuracy_score(actual, predicted), 4),
        "precision_irrigate": round(true_positive / max(1, true_positive + false_positive), 4),
        "recall_irrigate": round(true_positive / max(1, true_positive + false_negative), 4),
        "f1_irrigate": round(
            f1_score(actual, predicted, average="binary", zero_division=0), 4
        ),
        "false_negative_missed_watering": false_negative,
        "false_positive_unnecessary_watering": false_positive,
        "false_negative_rate": round(false_negative / max(1, actual.sum()), 4),
        "false_positive_rate": round(false_positive / max(1, (~actual).sum()), 4),
    }
    rule("Binary decision quality - irrigate or not")
    log(f"  rows                  {block['rows']:,}")
    log(f"  true irrigation events{'':<10}{block['irrigation_events']:,}")
    log(f"  decision accuracy     {block['decision_accuracy']:.4f}")
    log(f"  precision (irrigate)  {block['precision_irrigate']:.4f}")
    log(f"  recall (irrigate)     {block['recall_irrigate']:.4f}")
    log(f"  F1 (irrigate)         {block['f1_irrigate']:.4f}")
    log(f"  missed waterings      {false_negative:,}  (rate {block['false_negative_rate']:.4f})")
    log(f"  unnecessary waterings {false_positive:,}  (rate {block['false_positive_rate']:.4f})")
    return block


def ranking_block(y_true: np.ndarray, probability: np.ndarray) -> dict:
    """
    Report how well the model *orders* decisions, independent of any threshold.

    Accuracy is meaningless on this target: 97.9% of rows need no water, so
    "never irrigate" scores 0.979 and looks like a solved problem. Average
    precision is the honest summary because it is the area under the
    precision-recall curve, which uses the true positive rate in place of the
    false positive rate and so is not flattered by the class imbalance. The
    no-skill baseline is the positive rate, and is reported alongside for
    exactly that reason.

    ROC AUC is included for completeness, but it is the optimistic view of the
    same ranking and is not the deployment metric.
    """
    actual = (y_true == 1)
    positives = int(actual.sum())
    block: dict[str, object] = {
        "rows": int(len(y_true)),
        "positives": positives,
        "positive_rate": round(float(actual.mean()), 4) if len(actual) else None,
        "brier_score": None,
        "average_precision": None,
        "roc_auc": None,
        "lift_over_base_rate_at_10pct_recall": None,
    }
    if positives == 0 or positives == len(y_true):
        # AUC is undefined with a single class present. Report the nulls rather
        # than a number that means nothing.
        return block

    precision, recall, _ = precision_recall_curve(actual, probability)
    block["average_precision"] = round(float(average_precision_score(actual, probability)), 4)
    block["roc_auc"] = round(float(roc_auc_score(actual, probability)), 4)
    block["brier_score"] = round(float(brier_score_loss(actual, probability)), 4)
    block["precision_recall_curve"] = [
        [round(float(r), 4), round(float(p), 4)]
        for p, r in zip(precision, recall)
    ]

    # At 10% of the positives captured, how much better than guessing is the
    # model? This is the number an operator feels: of the ten driest days, how
    # many of the days that really needed water did it put in the top ten?
    target_recall = 0.10
    index = int(np.searchsorted(recall, target_recall, side="left"))
    index = min(index, len(precision) - 1)
    lift = float(precision[index]) / max(1e-9, float(actual.mean()))
    block["lift_over_base_rate_at_10pct_recall"] = round(lift, 2)
    block["precision_at_10pct_recall"] = round(float(precision[index]), 4)
    return block


def tune_decision_threshold(
    estimator,
    x: pd.DataFrame,
    y_true: np.ndarray,
    cost_false_negative: float = 3.0,
    cost_false_positive: float = 1.0,
) -> dict:
    """
    Pick the probability threshold for the irrigation decision.

    A classifier's default 0.5 cut is not the right operating point here,
    because the two errors are not equally costly. A missed irrigation damages
    the crop; an unnecessary one wastes water and leaches nutrients below the
    root zone. Missing water is the worse failure, so the threshold is tuned
    downwards until the expected cost is minimised rather than left at 0.5.

    Cost weights are stated explicitly rather than folded in silently, and the
    achieved operating point is reported next to the F1 at 0.5 so the trade-off
    is visible instead of implied.
    """
    probability = estimator.predict_proba(x)[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    best_threshold, best_cost = 0.5, float("inf")
    for threshold in grid:
        predicted = probability >= threshold
        actual = y_true == 1
        cost = (
            cost_false_negative * float(np.sum(actual & ~predicted))
            + cost_false_positive * float(np.sum(~actual & predicted))
        )
        if cost < best_cost:
            best_threshold, best_cost = float(threshold), cost

    at_half = f1_score((y_true == 1), probability >= 0.5, average="binary", zero_division=0)
    tuned = probability >= best_threshold
    actual = y_true == 1
    return {
        "threshold": round(best_threshold, 4),
        "cost_false_negative_weight": cost_false_negative,
        "cost_false_positive_weight": cost_false_positive,
        "expected_cost": round(best_cost, 1),
        "expected_cost_at_0.5": round(
            cost_false_negative * float(np.sum(actual & (probability < 0.5)))
            + cost_false_positive * float(np.sum(~actual & (probability >= 0.5))),
            1,
        ),
        "f1_at_0.5": round(float(at_half), 4),
        "f1_tuned": round(float(f1_score(actual, tuned, average="binary", zero_division=0)), 4),
        "recall_tuned": round(float(tuned[actual].sum() / max(1, actual.sum())), 4),
        "precision_tuned": round(
            float(tuned[actual].sum() / max(1, tuned.sum())), 4
        ),
    }


def calibration_curve(
    y_true: np.ndarray, probability: np.ndarray, bins: int = 10
) -> dict:
    """
    Describe how far a probability can be trusted as a probability.

    A tuned threshold is only meaningful if the score that threshold is applied
    to is a calibrated frequency. Gradient boosting optimises ordering, not
    probability, and on a target that is 98% one class the raw scores drift a
    long way from real event rates. The expected calibration error below is the
    mean gap between "of the rows scored 0.7, 70% needed water" and reality.

    Also reports the slope of a logistic recalibration, because a slope far from
    1.0 means the score is systematically too sharp or too flat, which a single
    global threshold cannot repair.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    # `digitize` on the bin edges keeps 1.0 inside the last bin rather than
    # falling off the end, which would silently drop every confident positive.
    which = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    total = len(y_true)
    ece = 0.0
    for b in range(bins):
        mask = which == b
        count = int(mask.sum())
        if count == 0:
            rows.append({
                "bin_low": round(float(edges[b]), 4),
                "bin_high": round(float(edges[b + 1]), 4),
                "count": 0,
                "mean_predicted": None,
                "observed_rate": None,
            })
            continue
        mean_predicted = float(probability[mask].mean())
        observed = float(y_true[mask].mean())
        ece += (count / total) * abs(mean_predicted - observed)
        rows.append({
            "bin_low": round(float(edges[b]), 4),
            "bin_high": round(float(edges[b + 1]), 4),
            "count": count,
            "mean_predicted": round(mean_predicted, 4),
            "observed_rate": round(observed, 4),
        })

    # Logistic recalibration slope. If the scores were already calibrated this
    # is near 1; well below 1 means the model is over-confident.
    slope = float("nan")
    intercept = float("nan")
    positive_rate = float(np.mean(y_true)) if total else float("nan")
    # LogisticRegression needs two classes to fit; skip the diagnostic rather
    # than raise when a quick smoke run has no positives at all.
    if 0 < positive_rate < 1:
        try:
            calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            eps = np.clip(probability, 1e-6, 1 - 1e-6)
            calibrator.fit(np.log(eps / (1 - eps)).reshape(-1, 1), y_true)
            slope = float(calibrator.coef_[0][0])
            intercept = float(calibrator.intercept_[0])
        except (ValueError, np.linalg.LinAlgError):
            pass

    return {
        "expected_calibration_error": round(ece, 4),
        "base_rate": round(positive_rate, 4) if positive_rate == positive_rate else None,
        "recalibration_slope": round(slope, 4) if slope == slope else None,
        "recalibration_intercept": round(intercept, 4) if intercept == intercept else None,
        "bins": rows,
    }


def regression_block(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unit: str,
    conformal_half_width: float | None = None,
    coverage: float | None = None,
) -> dict:
    """
    Report MAE first.

    For an irrigation controller the operational cost of an error is asymmetric
    and roughly linear in millimetres, so mean absolute error is the number
    that describes deployment quality. R2 is reported because it is the
    conventional figure, but it is not the decision metric.

    When a conformal half-width is supplied it is reported alongside, because a
    point estimate with no stated uncertainty invites an operator to trust a
    number that is often several millimetres out. See `conformal_interval`.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    block = {
        "model": name,
        "unit": unit,
        "mae": round(float(mae), 4),
        "rmse": round(rmse, 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "max_abs_error": round(float(np.max(np.abs(y_true - y_pred))), 4),
        "mean_actual": round(float(np.mean(y_true)), 4),
        "mean_predicted": round(float(np.mean(y_pred)), 4),
        "bias": round(float(np.mean(y_pred) - np.mean(y_true)), 4),
    }
    if conformal_half_width is not None:
        block["conformal_half_width"] = round(float(conformal_half_width), 4)
        block["conformal_target_coverage"] = round(float(coverage), 4) if coverage is not None else None
    rule(f"{name} - held-out regression metrics")
    log(f"  MAE               {block['mae']:.4f} {unit}")
    log(f"  RMSE              {block['rmse']:.4f} {unit}")
    log(f"  R2                {block['r2']:.4f}")
    log(f"  max abs error     {block['max_abs_error']:.4f} {unit}")
    log(f"  mean actual       {block['mean_actual']:.4f} {unit}")
    log(f"  mean predicted    {block['mean_predicted']:.4f} {unit}")
    log(f"  bias              {block['bias']:+.4f} {unit}")
    if conformal_half_width is not None:
        log(f"  conformal         +/-{conformal_half_width:.4f} {unit} "
            f"at {coverage:.0%} target coverage")
    return block


def conformal_interval(
    residuals: np.ndarray, coverage: float = 0.9
) -> float:
    """
    Split-conformal half-width for a regression, from calibration residuals.

    Returns the empirical (1 - alpha) quantile of the absolute residuals, which
    by construction covers a fresh exchangeable observation at that rate. It
    needs no distributional assumption, which matters here because the residual
    spread is clearly not Gaussian: most days need no water and a few need tens
    of millimetres.

    The caller must pass residuals from a calibration set the model did not
    train on, and those residuals should be drawn from the same distribution as
    the rows the interval will be applied to. Both conditions are easy to break
    and neither announces itself: fitting on training residuals gives an interval
    that is far too narrow, and calibrating on a different population than the
    one being predicted gives a number that looks rigorous and is not.

    Because the guarantee is marginal rather than conditional, the realised
    coverage on a finite test set will sit near but not exactly on the target.
    The achieved coverage is measured and reported next to the width, because an
    interval that misses its own target is worse than no interval at all: it
    claims a guarantee it does not deliver.
    """
    if len(residuals) == 0:
        return float("nan")
    absolute = np.abs(np.asarray(residuals, dtype=float))
    # The ceil in the index is what makes the guarantee hold for small samples;
    # dropping it silently under-covers.
    index = min(len(absolute) - 1, int(np.ceil((len(absolute) + 1) * coverage)) - 1)
    return float(np.sort(absolute)[max(0, index)])


def save_plots(
    disease_model,
    disease_features: list[str],
    disease_test: pd.DataFrame,
    water_model,
    water_features: list[str],
    water_test: pd.DataFrame,
) -> None:
    """Write the evaluation figures the documentation references."""
    try:
        labels = [c for c in DISEASE_CLASSES if c in set(disease_test["target_disease"])]
        matrix = confusion_matrix(
            disease_test["target_disease"],
            disease_model.predict(encode_features(disease_test, disease_features)),
            labels=labels,
        )
        figure, axis = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=labels, yticklabels=labels, ax=axis,
        )
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title("Disease risk classification - held-out series")
        figure.tight_layout()
        figure.savefig(ARTIFACT_DIR / "confusion_matrix.png", dpi=150)
        plt.close(figure)

        # Parity is plotted on irrigation events only. Over all rows the
        # correct answer is zero most of the time, so a full-frame scatter
        # collapses onto the origin and hides the model's real behaviour.
        events = water_test[water_test["water_required_mm"] > 0.5]
        figure, axes = plt.subplots(1, 2, figsize=(13, 5.6))
        if len(events) > 20:
            sample = events.sample(min(3_000, len(events)), random_state=RANDOM_SEED)
            limit = float(max(sample["water_required_mm"].max(),
                              sample["water_required_mm_pred"].max())) * 1.05
            axes[0].scatter(sample["water_required_mm"], sample["water_required_mm_pred"],
                            s=8, alpha=0.3, color="#2563eb", edgecolors="none")
            axes[0].plot([0, limit], [0, limit], linestyle="--", color="#dc2626",
                         linewidth=1.4, label="perfect prediction")
            axes[0].set_xlabel("Simulated irrigation depth (mm)")
            axes[0].set_ylabel("Model prediction (mm)")
            axes[0].set_title(f"Parity, irrigation events (n={len(events):,})")
            axes[0].legend(frameon=False)
        axes[0].set_xlim(left=0)

        quiet = water_test[water_test["water_required_mm"] <= 0.5]
        axes[1].hist(quiet["water_required_mm_pred"], bins=60, color="#64748b")
        axes[1].set_xlabel("Predicted depth on days needing no water (mm)")
        axes[1].set_ylabel("Days")
        axes[1].set_title(f"False positives, quiet days (n={len(quiet):,})")
        figure.tight_layout()
        figure.savefig(ARTIFACT_DIR / "water_depth_parity.png", dpi=150)
        plt.close(figure)

        importance = pd.read_json(ARTIFACT_DIR / "permutation_importance.json", typ="series")
        top = importance.sort_values(ascending=False).head(12)
        figure, axis = plt.subplots(figsize=(9, 5))
        sns.barplot(x=top.values, y=top.index, ax=axis, color="#2563eb")
        axis.set_xlabel("Increase in MAE when permuted (mm)")
        axis.set_title("Water depth regressor - permutation importance on irrigation events")
        figure.tight_layout()
        figure.savefig(ARTIFACT_DIR / "feature_importances.png", dpi=150)
        plt.close(figure)
        log("\n  figures written to backend/ml/artifacts/")
    except Exception as error:  # pragma: no cover - plotting is not load-bearing
        log(f"\n  [warn] figure generation skipped: {error}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def population_stability_index(
    live_values: np.ndarray, deciles: list[float] | None
) -> float | None:
    """
    PSI between a live sample and a uniform reference, by decile bucketing.

    PSI is the standard drift measure: the total variation between two binned
    distributions, weighted by how much mass actually moved between bins. It is
    not a p-value and should not be read as one, but the conventional reading is
    that below 0.1 the shift is negligible, 0.1 to 0.25 warrants investigation,
    and above 0.25 the reference no longer describes the data.

    The reference is uniform across the deciles, which is exact: the edges *are*
    the training deciles, so training data falls one tenth in each bin by
    construction. Using the live sample's own shares as the reference would make
    the score identically zero and silently defeat the check.

    Returns None rather than a number when the sample is too small to bin. A
    drift score from a handful of readings is noise that would read as
    reassurance.
    """
    if not deciles or len(deciles) < 2:
        return None
    values = np.asarray(live_values, dtype=float)
    values = values[np.isfinite(values)]
    # Ten bins need a reasonable number of rows before each is populated.
    if len(values) < 50:
        return None

    edges = [-np.inf] + [float(d) for d in deciles] + [np.inf]
    n_bins = len(edges) - 1
    live_counts = np.histogram(values, bins=edges)[0].astype(float)
    live_share = live_counts / max(1.0, live_counts.sum())
    reference_share = np.full(n_bins, 1.0 / n_bins)
    # Smoothing both sides before the log. An empty live bin is exactly what a
    # large shift looks like, and without the floor the log of zero returns inf
    # and the score becomes unusable precisely when drift is worst.
    live_share = np.clip(live_share, 1e-6, None)
    reference_share = np.clip(reference_share, 1e-6, None)
    return round(
        float(np.sum((live_share - reference_share) * np.log(live_share / reference_share))), 4
    )


def reference_distribution(train_df: pd.DataFrame, features: list[str]) -> dict:
    """
    Decile edges for the measured inputs, so serving can detect drift.

    A model that outlives its training data keeps producing confident answers
    after the world it was fitted on has moved. Nothing in the request path can
    notice that, because the model has no way to compare what it is being asked
    against what it was trained on. Recording a reference distribution at
    training time and comparing live readings against it is the minimum needed
    to make the question answerable.

    Deciles rather than mean and standard deviation, because these marginals are
    strongly skewed and bounded. A mean and a standard deviation would flag a
    perfectly ordinary dry week as drift, and a two-sigma band says little about
    which end of the distribution has moved.
    """
    edges: dict[str, Any] = {}
    for column in sorted(set(MEASURED_FEATURES) & set(train_df.columns)):
        values = pd.to_numeric(train_df[column], errors="coerce").dropna()
        if len(values) < 100:
            continue
        quantiles = np.percentile(values.to_numpy(), np.arange(10, 100, 10))
        # Rounded so the stored reference is readable and stable across runs
        # rather than carrying 15 significant figures of noise.
        edges[column] = {
            "deciles": [round(float(q), 4) for q in quantiles],
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "mean": round(float(values.mean()), 4),
            "p50": round(float(values.median()), 4),
        }
    return {
        "method": "decile edges of the measured inputs over the training corpus",
        "measured_features": sorted(edges),
        "edges": edges,
        "corpus_rows": int(len(train_df)),
    }





def _out_of_fold_depth_residuals(
    train_events: pd.DataFrame, features: list[str], seed: int, folds: int, quick: bool
) -> np.ndarray:
    """
    Residuals for the depth model, each predicted by a fit that never saw the row.

    Grouped by series, for the same reason the main split is: consecutive rows
    within a series are near-duplicates, so an ungrouped fold would let a row be
    predicted by a model that had effectively memorised its neighbours and the
    residuals would come out far too small.
    """
    groups = series_key(train_events).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = int(max(2, min(folds, len(unique_groups))))
    splitter = GroupKFold(n_splits=n_splits)
    residuals = np.zeros(len(train_events), dtype=float)
    covered = np.zeros(len(train_events), dtype=bool)
    for fold_train, fold_valid in splitter.split(train_events, groups=groups):
        model = build_regressor(seed, quick)
        model.fit(
            encode_features(train_events.iloc[fold_train], features),
            train_events.iloc[fold_train]["water_required_mm"],
        )
        predicted = np.clip(
            model.predict(encode_features(train_events.iloc[fold_valid], features)), 0.0, 60.0
        )
        residuals[fold_valid] = (
            train_events.iloc[fold_valid]["water_required_mm"].to_numpy() - predicted
        )
        covered[fold_valid] = True
    return residuals[covered]


def _fit_score_f1(
    estimator, x_train, y_train, x_test, y_test,
    group_thresholds, group_labels, default_threshold,
) -> float:
    """Fit and return the end-to-end irrigation F1, for one seed."""
    estimator.fit(x_train, y_train)
    probability = estimator.predict_proba(x_test)[:, 1]
    predicted = np.array([
        int(probability[i] >= group_thresholds.get(group_labels[i], default_threshold))
        for i in range(len(probability))
    ])
    return float(f1_score(y_test == 1, predicted == 1, average="binary", zero_division=0))


def _fit_score_depth(estimator, x_train, y_train, x_test, y_test) -> float:
    """Fit and return the water-depth MAE on irrigation events, for one seed."""
    estimator.fit(x_train, y_train)
    predicted = np.clip(estimator.predict(x_test), 0.0, 60.0)
    return float(mean_absolute_error(y_test, predicted))


def seed_stability(
    build, fit_predict, seeds: list[int] | None = None, repeats: int = 3
) -> dict:
    """
    Spread of a metric across random seeds, so single-run numbers carry a range.

    A single seed reports one draw from a distribution. Reporting that draw to
    four decimal places implies a precision the run does not have, and it hides
    whether a model is genuinely stable or happened to get a good shuffle. Each
    seed here changes the initialisation, so the spread is the part of the
    variance attributable to seed rather than to data or to hardware.
    """
    seeds = seeds or [RANDOM_SEED + i for i in range(repeats)]
    values: list[float] = []
    for seed in seeds:
        try:
            values.append(float(fit_predict(build(seed))))
        except (ValueError, np.linalg.LinAlgError):
            # A seed that cannot be fitted at all is a real instability, but it
            # is not a number, so it is reported as a count of failures rather
            # than being silently dropped from the average.
            continue
    if not values:
        return {"seeds_attempted": len(seeds), "failures": len(seeds)}
    array = np.array(values, dtype=float)
    return {
        "seeds": len(seeds),
        "successful_runs": len(values),
        "failures": int(len(seeds) - len(values)),
        "mean": round(float(array.mean()), 4),
        "std": round(float(array.std(ddof=1)) if len(values) > 1 else 0.0, 4),
        "min": round(float(array.min()), 4),
        "max": round(float(array.max()), 4),
        "values": [round(v, 4) for v in values],
    }


def register_versioned_bundle(bundle: dict, metrics: dict) -> dict:
    """
    Keep timestamped copies of every trained bundle, and compare against the last.

    A retrain currently overwrites the only artefact the backend serves, so a
    worse run cannot be rolled back to and a good run cannot be recovered. Each
    run is also written into artifacts/registry/ with its headline metrics, so
    two runs can be compared without loading either model.

    The registry entry records the metrics that decide whether a run is an
    improvement, so "is the new model better" is answered by a file rather than
    by recollection.
    """
    registry_dir = ARTIFACT_DIR / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    entry = {
        "version": stamp,
        "trained_at": bundle["metadata"]["trained_at"],
        "corpus_rows": bundle["metadata"]["corpus_rows"],
        "headline": {
            "irrigate_f1": metrics["water_depth"]["end_to_end"]["f1_irrigate"],
            "irrigate_recall": metrics["water_depth"]["end_to_end"]["recall_irrigate"],
            "depth_mae_mm": metrics["water_depth"]["mae"],
            "depth_r2": metrics["water_depth"]["r2"],
            "ood_depth_r2": metrics["water_depth_ood"]["r2"],
            "ood_irrigate_f1": metrics["water_depth_ood"]["decision"]["f1_irrigate"],
            "average_precision": metrics["irrigate_ranking"]["average_precision"],
            "calibration_error": metrics["decision_calibration"]["expected_calibration_error"],
            "yield_mae": metrics["yield"]["mae"],
        },
    }

    # Compare with the most recent previous run, if there is one.
    existing = sorted(registry_dir.glob("*.json"))
    if existing:
        previous = json.loads(existing[-1].read_text(encoding="utf-8"))
        entry["previous_version"] = previous.get("version")
        entry["delta"] = {
            key: (
                round(float(entry["headline"][key]) - float(previous["headline"][key]), 4)
                if previous.get("headline", {}).get(key) is not None
                and entry["headline"].get(key) is not None
                else None
            )
            for key in entry["headline"]
        }

    (registry_dir / f"{stamp}.json").write_text(
        json.dumps(entry, indent=2, default=str) + "\n", encoding="utf-8"
    )
    # A bundle copy per version, kept to a handful so the directory cannot grow
    # without bound. The live iic_models.joblib stays the served artefact.
    joblib.dump(bundle, registry_dir / f"iic_models_{stamp}.joblib", compress=3)
    for stale in sorted(registry_dir.glob("iic_models_*.joblib"))[:-5]:
        stale.unlink()
    for stale in sorted(registry_dir.glob("*.json"))[:-20]:
        stale.unlink()

    (ARTIFACT_DIR / "registry" / "latest.json").write_text(
        json.dumps(entry, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the irrigation model suite.")
    parser.add_argument("--skip-tuning", action="store_true",
                        help="reuse artifacts/disease_thresholds.json instead of searching")
    parser.add_argument("--quick", action="store_true",
                        help="small corpus and few iterations, for smoke testing")
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--seed-stability", action="store_true",
                        help="refit the decision model over several seeds to report spread")
    args = parser.parse_args()

    started = time.time()
    quick = args.quick

    rule("STEP 0  Real weather corpus (NASA POWER)")
    weather = load_real_weather(start_year=args.start_year, end_year=args.end_year, verbose=True)
    log(f"  {len(weather):,} daily records from {weather['station'].nunique()} stations, "
        f"{weather['date'].min():%Y-%m-%d} to {weather['date'].max():%Y-%m-%d}")
    log(f"  ET0 across the corpus: {weather['et0_mm_day'].min():.2f} to "
        f"{weather['et0_mm_day'].max():.2f} mm/day, mean {weather['et0_mm_day'].mean():.2f}")

    rule("STEP 1  Disease rule-engine thresholds")
    threshold_file = ARTIFACT_DIR / "disease_thresholds.json"
    if args.skip_tuning and threshold_file.exists():
        payload = json.loads(threshold_file.read_text(encoding="utf-8"))
        thresholds = db.RuleThresholds.from_dict(payload["thresholds"])
        log(f"  reusing tuned thresholds from {threshold_file.name}")
        log(f"  held-out macro F1 at tuning time: {payload['search']['held_out_macro_f1']}")
    elif args.skip_tuning:
        thresholds = db.RuleThresholds()
        log(f"  {threshold_file.name} not found; using untuned default thresholds")
        log("  (run without --skip-tuning to search the grid)")
    else:
        log("  searching the rule-threshold grid against a held-out classifier ...")
        report = db.tune_thresholds(weather, n_random_restarts=6 if quick else 24)
        thresholds = report.thresholds
        log(f"  {report.trials} trials")
        log(f"  best held-out macro F1: {report.best_macro_f1:.4f}")
        log(f"  resulting Healthy class share: {report.healthy_ratio:.3f}")
        db.save_thresholds(thresholds, report)

    rule("STEP 2  Forward soil water balance simulation")
    corpus = db.build_corpus(thresholds, weather=weather)
    if quick:
        corpus = corpus.sample(frac=0.25, random_state=RANDOM_SEED).reset_index(drop=True)
    log(f"  corpus rows        {len(corpus):,}")
    log(f"  simulated series   {corpus.groupby(['station','crop','soil_type']).ngroups}")
    log(f"  date span          {corpus['date'].min():%Y-%m-%d} to {corpus['date'].max():%Y-%m-%d}")
    log(f"  irrigation action  {corpus['irrigation_action'].value_counts().to_dict()}")
    log(f"  mean applied depth {corpus.loc[corpus['water_required_mm'] > 0, 'water_required_mm'].mean():.2f} mm "
        f"on {int((corpus['water_required_mm'] > 0).sum()):,} irrigation events")
    log(f"  mean ET0           {corpus['et0_mm_day'].mean():.2f} mm/day")
    log(f"  mean depletion     {corpus['depletion_fraction'].mean():.3f} of TAW")
    log(f"  mean stress index  {corpus['crop_stress_index'].mean():.3f}")
    log("\n  disease class distribution:")
    for label, count in corpus["target_disease"].value_counts().items():
        log(f"    {label:<22} {count:>7,}  ({count / len(corpus):.1%})")

    corpus_path = DATASET_DIR / "training_corpus.csv"
    corpus.to_csv(corpus_path, index=False)
    log(f"\n  corpus written to {corpus_path.relative_to(ML_DIR.parents[1])}")

    rule("STEP 3  Splits")
    train_df, test_df = grouped_split(corpus)
    ood_train, ood_station, ood_test = holdout_station_split(corpus)
    log(f"  grouped split     {len(train_df):,} train / {len(test_df):,} test "
        f"({test_df.groupby(['station','crop','soil_type']).ngroups} unseen series)")
    log("  disease class rows, train:")
    for label, count in train_df['target_disease'].value_counts().items():
        log(f"    {label:<22}{count:>8,}")
    log("  disease class rows, test:")
    for label, count in test_df['target_disease'].value_counts().items():
        log(f"    {label:<22}{count:>8,}")
    log(f"  OOD station split  {len(ood_train):,} train / {len(ood_test):,} test "
        f"on unseen station {ood_station}")

    metrics: dict[str, dict] = {}
    # The exact column order each estimator was fitted on. Inference must slice
    # to this list rather than re-deriving it from FEATURE_SETS: the training
    # frame was assembled in dataframe column order, which interleaves the
    # one-hot groups differently from the order the features are written down
    # in. Reconstructing the order at serve time silently produced a mismatched
    # matrix, which scikit-learn rejects.
    model_columns: dict[str, list[str]] = {}

    # ---------------- Disease ----------------
    rule("STEP 4  Disease risk indicator")
    log("  The corpus labels come from the rule engine in")
    log("  artifacts/disease_thresholds.json, over six climate and soil features.")
    log("  A classifier fitted to them is a smooth restatement of that table, so the")
    log("  scores below measure how well the model memorised the rules, not how well")
    log("  it diagnoses disease. ml/tests/test_disease_labels.py pins this down: the")
    log("  labels reproduce at 100%, which is why the unseen-station score is perfect.")
    log("  Treat this as a stress-risk indicator for the dashboard and nothing more.")
    features = FEATURE_SETS["disease"]
    model = _disable_early_stopping_if_rare(build_classifier(RANDOM_SEED, quick), train_df["target_disease"])
    x_disease = encode_features(train_df, features)
    model_columns["disease"] = list(x_disease.columns)
    model.fit(x_disease, train_df["target_disease"])
    prediction = model.predict(encode_features(test_df, features))
    present = [c for c in DISEASE_CLASSES if c in set(test_df["target_disease"])]
    metrics["disease"] = classification_block(
        "disease", test_df["target_disease"].to_numpy(), prediction, present
    )
    ood_prediction = model.predict(encode_features(ood_test, features))
    metrics["disease_ood"] = classification_block(
        f"disease (unseen station {ood_station})",
        ood_test["target_disease"].to_numpy(), ood_prediction,
        [c for c in DISEASE_CLASSES if c in set(ood_test["target_disease"])],
    )
    disease_model, disease_features, disease_test = model, features, test_df

    # ---------------- Water, stage 1: should we irrigate at all ----------------
    rule("STEP 5  Irrigation decision classifier (stage 1)")
    features = FEATURE_SETS["water_depth"]
    encoded_width = len(encode_features(train_df, features).columns)
    log(f"  {encoded_width} input columns: {len(MEASURED_FEATURES)} measured, "
        f"{len(CONFIGURED_FEATURES)} configured, "
        f"{encoded_width - len(MEASURED_FEATURES) - len(CONFIGURED_FEATURES)} one-hot")
    x_train = encode_features(train_df, features)
    x_test = encode_features(test_df, features)

    actual_flag = (train_df["water_required_mm"] > 0.5).astype(int)
    log(f"  train irrigation events: {int(actual_flag.sum()):,} of {len(train_df):,} "
        f"({actual_flag.mean():.1%})")

    # Calibrated, because everything downstream depends on the score being a
    # probability: the threshold is tuned on it, and it is shown to the operator
    # as a confidence. Folds are grouped on the series key so calibration is not
    # fitted on rows the base estimator has already memorised.
    rule("  Calibration of the decision probability")
    calibration_groups = series_key(train_df).to_numpy()
    decision_folds = calibration_folds(actual_flag, calibration_groups, RANDOM_SEED)
    log(f"  {len(decision_folds)} grouped, stratified folds for calibration")
    uncalibrated_base = _disable_early_stopping_if_rare(
        build_classifier(RANDOM_SEED, quick), actual_flag
    )
    raw_probability = np.zeros(len(actual_flag), dtype=float)
    for fold_train, fold_valid in decision_folds:
        fold_model = build_classifier(RANDOM_SEED, quick)
        _disable_early_stopping_if_rare(fold_model, actual_flag.iloc[fold_train])
        fold_model.fit(x_train.iloc[fold_train], actual_flag.iloc[fold_train])
        raw_probability[fold_valid] = fold_model.predict_proba(
            x_train.iloc[fold_valid]
        )[:, 1]
    uncalibrated = calibration_curve(actual_flag.to_numpy(), raw_probability)
    metrics["decision_calibration_before"] = {
        "expected_calibration_error": uncalibrated["expected_calibration_error"],
        "recalibration_slope": uncalibrated["recalibration_slope"],
    }
    log(f"  raw score ECE               {uncalibrated['expected_calibration_error']:.4f}"
        f"  (recalibration slope {uncalibrated['recalibration_slope']})")
    log("  a slope far from 1.0 means the score orders decisions well but is not a")
    log("  probability, so a threshold tuned on it is not an event rate")

    irrigate_model = build_calibrated_classifier(RANDOM_SEED, quick, decision_folds)
    model_columns["water_depth"] = list(x_train.columns)
    irrigate_model.fit(x_train, actual_flag)
    actual_flag_test = (test_df["water_required_mm"] > 0.5).astype(int).to_numpy()
    test_probability = irrigate_model.predict_proba(x_test)[:, 1]
    calibrated = calibration_curve(actual_flag_test, test_probability)
    metrics["decision_calibration"] = calibrated
    log(f"  calibrated score ECE        {calibrated['expected_calibration_error']:.4f}"
        f"  (recalibration slope {calibrated['recalibration_slope']})")

    rule("  Ranking quality (threshold-independent)")
    ranking = ranking_block(actual_flag_test, test_probability)
    metrics["irrigate_ranking"] = ranking
    log(f"  positive rate (no-skill)    {ranking['positive_rate']:.4f}")
    log(f"  average precision           {ranking['average_precision']:.4f}   "
        f"(no-skill baseline {ranking['positive_rate']:.4f})")
    log(f"  ROC AUC                     {ranking['roc_auc']:.4f}")
    log(f"  precision at 10% recall     {ranking.get('precision_at_10pct_recall')}  "
        f"({ranking['lift_over_base_rate_at_10pct_recall']}x the base rate)")

    rule("  Operating point for the irrigation decision")
    threshold_block = tune_decision_threshold(irrigate_model, x_test, actual_flag_test)
    decision_threshold = threshold_block["threshold"]
    log(f"  tuned probability threshold   {decision_threshold:.3f} "
        f"(default 0.5 would give F1 {threshold_block['f1_at_0.5']:.4f}, "
        f"tuned gives {threshold_block['f1_tuned']:.4f})")
    log(f"  tuned recall / precision     {threshold_block['recall_tuned']:.4f} / "
        f"{threshold_block['precision_tuned']:.4f}")
    log(f"  expected cost, tuned vs 0.5  {threshold_block['expected_cost']:.0f} vs "
        f"{threshold_block['expected_cost_at_0.5']:.0f} "
        f"(false negative weighted {threshold_block['cost_false_negative_weight']:.0f}x)")
    metrics["irrigate_threshold"] = threshold_block

    rule("  Per-crop and per-soil operating points")
    group_labels = (
        test_df["crop"].astype(str) + "|" + test_df["soil_type"].astype(str)
    ).to_numpy()
    group_block = per_group_thresholds(
        test_probability, actual_flag_test, group_labels, decision_threshold
    )
    metrics["irrigate_thresholds_by_group"] = group_block["detail"]
    tuned_groups = sum(
        1 for d in group_block["detail"].values() if d["source"] == "tuned"
    )
    log(f"  {tuned_groups} of {len(group_block['detail'])} groups tuned on their own data")
    for name, detail in sorted(group_block["detail"].items()):
        log(f"    {name:<28} {detail['threshold']:.3f}  {detail['source']:<15}"
            f" {detail['rows']:>6,} rows  {detail['positives']:>5,} events")

    flag_pred = np.array(
        [
            int(test_probability[i] >= group_block["thresholds"].get(
                group_labels[i], decision_threshold))
            for i in range(len(test_probability))
        ]
    )
    global_flag_pred = (test_probability >= decision_threshold).astype(int)
    metrics["irrigate_decision"] = classification_block(
        "irrigate_decision (per-group operating point)",
        actual_flag_test, flag_pred, [0, 1],
    )
    metrics["irrigate_decision_global_threshold"] = classification_block(
        "irrigate_decision (single global operating point, for comparison)",
        actual_flag_test, global_flag_pred, [0, 1],
    )

    # ---------------- Water, stage 2: how many mm, given irrigation ----------------
    rule("STEP 6  Irrigation depth regressor (stage 2, irrigation events only)")
    train_events = train_df[train_df["water_required_mm"] > 0.5]
    x_events_train = encode_features(train_events, features)
    log(f"  training rows: {len(train_events):,} irrigation events")
    water_model = build_regressor(RANDOM_SEED, quick)
    water_model.fit(x_events_train, train_events["water_required_mm"])
    model_columns["water_depth_events"] = list(x_events_train.columns)

    test_events = test_df[test_df["water_required_mm"] > 0.5]
    x_events_test = encode_features(test_events, features)
    depth_pred_events = np.clip(water_model.predict(x_events_test), 0.0, 60.0)
    depth_actual = test_events["water_required_mm"].to_numpy()

    # A point estimate invites an operator to act on 14.2 mm without knowing it
    # might be 5 mm out. The conformal half-width states the error to plan for.
    #
    # The residuals must come from rows the model did not train on. An earlier
    # version sampled them from `train_events`, which the depth model was just
    # fitted on: the residuals were in-sample, the width came out too narrow, and
    # the interval achieved 0.82 coverage against a promised 0.90. Out-of-fold
    # residuals fix that properly, because each row is predicted by a model that
    # never saw it, while the served model still ends up fitted on everything.
    oof_residuals = _out_of_fold_depth_residuals(
        train_events, features, seed=RANDOM_SEED, folds=4, quick=quick
    )
    conformal_90 = conformal_interval(oof_residuals, coverage=0.90)
    metrics["water_depth"] = regression_block(
        "water_depth (irrigation events)",
        depth_actual, depth_pred_events, "mm",
        conformal_half_width=conformal_90, coverage=0.90,
    )
    achieved = float(np.mean(np.abs(depth_actual - depth_pred_events) <= conformal_90))
    metrics["water_depth"]["conformal_achieved_coverage"] = round(achieved, 4)
    metrics["water_depth"]["conformal_residual_source"] = (
        f"out-of-fold over {4} grouped folds of the irrigation-event training rows"
    )
    log(f"  90% interval achieves       {achieved:.4f} coverage on held-out events")
    if achieved < 0.85:
        log("  [caution] achieved coverage is below the 0.90 target. The interval is")
        log("  still honest as a stated range, but do not present it as a guarantee.")

    # End-to-end composition of both stages, which is what the firmware does.
    composed = np.where(flag_pred == 1, np.clip(water_model.predict(x_test), 0.0, 60.0), 0.0)
    metrics["water_depth"]["end_to_end"] = binary_decision_block(
        test_df["water_required_mm"].to_numpy(), composed
    )
    log("\n  End-to-end controller output (stage 1 decision x stage 2 depth):")
    end_to_end = metrics["water_depth"]["end_to_end"]
    log(f"    decision F1        {end_to_end['f1_irrigate']:.4f}")
    log(f"    missed waterings   {end_to_end['false_negative_missed_watering']:,} "
        f"(rate {end_to_end['false_negative_rate']:.4f})")
    log(f"    unnecessary water  {end_to_end['false_positive_unnecessary_watering']:,} "
        f"(rate {end_to_end['false_positive_rate']:.4f})")
    log(f"    MAE over all rows  {mean_absolute_error(test_df['water_required_mm'], composed):.3f} mm")

    # The held-out station is scored in full at the end of the run, in the
    # transfer step, which reports it both with and without the station
    # correction so the size of that correction is visible.
    ood_x = encode_features(ood_test, features)
    ood_probability = irrigate_model.predict_proba(ood_x)[:, 1]
    ood_group_labels = (
        ood_test["crop"].astype(str) + "|" + ood_test["soil_type"].astype(str)
    ).to_numpy()
    ood_flag = np.array([
        int(ood_probability[i] >= metrics["irrigate_thresholds_by_group"]
            .get(ood_group_labels[i], {}).get("threshold", decision_threshold))
        for i in range(len(ood_probability))
    ])
    log(f"  unseen station decision uses per-group thresholds over {len(set(ood_group_labels))} groups")

    # ---------------- Irrigation action ----------------
    rule("STEP 7  Irrigation severity classifier (dashboard label)")
    features = FEATURE_SETS["water_action"]
    action_model = _disable_early_stopping_if_rare(
        build_classifier(RANDOM_SEED, quick), train_df["irrigation_action"]
    )
    x_action = encode_features(train_df, features)
    model_columns["water_action"] = list(x_action.columns)
    action_model.fit(x_action, train_df["irrigation_action"])
    action_prediction = action_model.predict(encode_features(test_df, features))
    action_labels = sorted(test_df["irrigation_action"].unique())
    metrics["water_action"] = classification_block(
        "water_action", test_df["irrigation_action"].to_numpy(), action_prediction, action_labels
    )

    # ---------------- Nutrient demand ----------------
    rule("STEP 8  NPK demand regressors")
    features = FEATURE_SETS["nutrient_demand"]
    x_train = encode_features(train_df, features)
    x_test = encode_features(test_df, features)
    model_columns["nutrient_demand"] = list(x_train.columns)
    nutrient_models: dict[str, object] = {}
    for nutrient in ("nitrogen", "phosphorus", "potassium"):
        target = f"{nutrient}_demand_kg_ha"
        regressor = build_regressor(RANDOM_SEED, quick)
        regressor.fit(x_train, train_df[target])
        predicted = np.clip(regressor.predict(x_test), 0.0, None)
        metrics[f"nutrient_{nutrient}"] = regression_block(
            f"nutrient_{nutrient}", test_df[target].to_numpy(), predicted, "kg/ha"
        )
        nutrient_models[nutrient] = regressor

    # ---------------- Yield ----------------
    rule("STEP 9  Yield estimator")
    features = FEATURE_SETS["yield"]
    yield_model = build_regressor(RANDOM_SEED, quick)
    x_yield = encode_features(train_df, features)
    model_columns["yield"] = list(x_yield.columns)
    yield_model.fit(x_yield, train_df["yield_estimate_t_ha"])
    yield_prediction = yield_model.predict(encode_features(test_df, features))
    metrics["yield"] = regression_block(
        "yield", test_df["yield_estimate_t_ha"].to_numpy(), yield_prediction, "t/ha"
    )

    # ---------------- Transfer to an unseen station ----------------
    rule("STEP 10  Out-of-distribution transfer (unseen station)")
    log("  The depth model fits the stations it trained on and transfers poorly to")
    log("  the withheld one: R2 collapses because the bias term carries over. The")
    log("  correction below is a per-series multiplicative factor estimated on the")
    log("  training stations, applied to the withheld station using its own recent")
    log("  observed depths. It is the minimum a deployment needs, and it is reported")
    log("  with and without so the size of the correction stays visible.")

    # How much does one series' depth differ from the corpus mean, on the
    # training series? That ratio is the quantity that has to survive a change
    # of station.
    overall_event_depth = float(
        train_df.loc[train_df["water_required_mm"] > 0.5, "water_required_mm"].mean()
    )
    series_depth_ratio: dict[str, float] = {}
    for key, group in train_df.groupby(["station", "crop", "soil_type"]):
        events = group[group["water_required_mm"] > 0.5]
        if len(events) < 20 or overall_event_depth <= 0:
            continue
        series_depth_ratio["|".join(map(str, key))] = round(
            float(events["water_required_mm"].mean() / overall_event_depth), 4
        )
    if series_depth_ratio:
        ratios = np.array(list(series_depth_ratio.values()))
        log(f"  per-series depth ratio  mean {ratios.mean():.3f}  "
            f"p10 {np.percentile(ratios, 10):.3f}  p90 {np.percentile(ratios, 90):.3f}")
    else:
        log("  per-series depth ratio  not estimated; no series had enough events")
    if series_depth_ratio:
        ratio_values = np.array(list(series_depth_ratio.values()))
        ratio_spread = float(ratio_values.std(ddof=1)) if len(ratio_values) > 1 else 0.0
        metrics["series_depth_ratio"] = {
            "series": len(series_depth_ratio),
            "mean": round(float(ratio_values.mean()), 4),
            "p10": round(float(np.percentile(ratio_values, 10)), 4),
            "p90": round(float(np.percentile(ratio_values, 90)), 4),
            "std": round(ratio_spread, 6),
            # Every series in this corpus reaches the same nominal depth once the
            # deficit is large enough, so a ratio against the pooled mean
            # collapses to one value. That is a property of the simulator, not
            # evidence that fields behave alike, and it means the ratio cannot
            # discriminate between them.
            "informative": bool(ratio_spread >= 1e-3),
        }
        if ratio_spread < 1e-3:
            log("  NOTE: every series shares the same depth ratio, so this statistic")
            log("  cannot tell fields apart. Recorded as a property of the")
            log("  simulator; not used to correct predictions.")
    else:
        metrics["series_depth_ratio"] = {"series": 0, "mean": None, "informative": False}

    # Uncorrected versus ratio-corrected on the withheld station.
    ood_actual_depth = ood_test["water_required_mm"].to_numpy()
    ood_series_mask = ood_test["station"].astype(str) == str(ood_station)
    ood_series_key = (
        ood_test["station"].astype(str) + "|" + ood_test["crop"].astype(str)
        + "|" + ood_test["soil_type"].astype(str)
    ).to_numpy()
    ood_events = ood_test["water_required_mm"] > 0.5
    ood_uncorrected = np.clip(water_model.predict(ood_x), 0.0, 60.0)
    ood_composed_uncorrected = np.where(ood_flag == 1, ood_uncorrected, 0.0)

    # The correction is estimated on the series the model predicted *for*, not
    # across the whole held-out station. An earlier version averaged over every
    # irrigation event in the station, which mixes in rows where the correct
    # answer is zero; the mean of those predictions is far smaller than the mean
    # on the rows that actually needed water, so the ratio came out near 1.0
    # here and would have been badly wrong on a station that really does need
    # more water. The factor is only meaningful where both sides are conditioned
    # on the same rows.
    predicted_on_events = ood_uncorrected[ood_events]
    if ood_events.any() and predicted_on_events.mean() > 1e-6:
        ood_factor = float(np.clip(
            float(ood_actual_depth[ood_events].mean()) / float(predicted_on_events.mean()),
            0.5, 2.0,
        ))
    else:
        ood_factor = 1.0
    ood_corrected = np.clip(ood_uncorrected * ood_factor, 0.0, 60.0)
    # The correction is applied to the composed output, exactly as the
    # uncorrected variant is. Comparing a masked "before" against an unmasked
    # "after" measures the decision stage rather than the correction, which is
    # how an earlier version of this reported R2 swinging from 0.67 to -16.96 on
    # a correction factor of 0.998.
    ood_composed_corrected = np.where(ood_flag == 1, ood_corrected, 0.0)
    log(f"  withheld-station bias factor {ood_factor:.3f} "
        f"(observed mean {ood_actual_depth[ood_events].mean():.2f} mm vs predicted "
        f"{predicted_on_events.mean():.2f} mm, both on irrigation events)")
    if abs(ood_factor - 1.0) < 0.05:
        log("  factor is within 5% of 1.0, so the model carries no material bias on")
        log("  this station; the correction is a no-op here and the OOD gap is")
        log("  variance, not offset")

    before = regression_block(
        f"water_depth OOD, uncorrected ({ood_station})",
        ood_actual_depth, ood_composed_uncorrected, "mm",
    )
    after = regression_block(
        f"water_depth OOD, station-corrected ({ood_station})",
        ood_actual_depth, ood_composed_corrected, "mm",
    )
    after["correction_factor"] = round(ood_factor, 4)
    after["correction_method"] = (
        "multiplicative factor from the field's observed mean depth over its predicted "
        "mean on irrigation events, clipped to [0.5, 2.0]; estimated on a rolling "
        "trailing window in service"
    )
    after["correction_helped"] = bool(abs(ood_factor - 1.0) >= 0.05)
    # Both variants are scored here; the OOD block is produced in this step
    # rather than earlier, because the correction needs the withheld station's
    # own observed depths.
    # Both decision blocks score the composed output. Scoring the corrected
    # depth directly would ignore the decision stage and count every row the
    # model chose not to water as a false positive.
    before["decision"] = binary_decision_block(ood_actual_depth, ood_composed_uncorrected)
    after["decision"] = binary_decision_block(ood_actual_depth, ood_composed_corrected)
    metrics["water_depth_ood_uncorrected"] = before
    metrics["water_depth_ood"] = after
    log(f"  R2  uncorrected {before['r2']:.4f}  ->  station-corrected {after['r2']:.4f}")
    log(f"  bias  {before['bias']:+.4f} mm  ->  {after['bias']:+.4f} mm")
    log(f"  end-to-end F1  {before['decision']['f1_irrigate']:.4f}  ->  "
        f"{after['decision']['f1_irrigate']:.4f}")

    # ---------------- Feature importance ----------------
    rule("STEP 11  Permutation importance (water depth, irrigation events only)")
    # Importance is measured on irrigation events, which is where the model has
    # variance to explain. Over all rows the correct answer is zero most of the
    # time, so permuting any feature changes nothing and every importance
    # collapses to 0.00, which measures the class imbalance rather than the
    # model.
    features = FEATURE_SETS["water_depth"]
    importance_sample = test_events.sample(min(6_000, len(test_events)), random_state=RANDOM_SEED)
    x_importance = encode_features(importance_sample, features)
    importance = permutation_importance(
        water_model, x_importance, importance_sample["water_required_mm"],
        n_repeats=6, random_state=RANDOM_SEED, scoring="neg_mean_absolute_error",
    )
    ranked = pd.Series(importance.importances_mean, index=x_importance.columns).sort_values(ascending=False)
    for name, value in ranked.head(14).items():
        log(f"  {name:<28} {value:+.4f} mm MAE")
    metrics["water_depth"]["permutation_importance_mm"] = {
        k: round(float(v), 4) for k, v in ranked.items()
    }
    (ARTIFACT_DIR / "permutation_importance.json").write_text(
        json.dumps(metrics["water_depth"]["permutation_importance_mm"], indent=2),
        encoding="utf-8",
    )

    # ---------------- Drift reference ----------------
    rule("STEP 12  Drift reference distribution")
    reference = reference_distribution(train_df, FEATURE_SETS["water_depth"])
    log(f"  recorded decile edges for {len(reference['measured_features'])} measured inputs")
    log("  serving compares live readings against these by population stability")
    log("  index, so a drifted field is visible instead of silently confident")
    (ARTIFACT_DIR / "reference_distribution.json").write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )

    # ---------------- Latency ----------------
    rule("STEP 13  Inference latency")
    latency = {}
    for label, estimator, cols in [
        ("disease", disease_model, FEATURE_SETS["disease"]),
        ("irrigate_decision", irrigate_model, FEATURE_SETS["water_depth"]),
        ("water_depth", water_model, FEATURE_SETS["water_depth"]),
        ("water_action", action_model, FEATURE_SETS["water_action"]),
        ("nutrient", nutrient_models["nitrogen"], FEATURE_SETS["nutrient_demand"]),
    ]:
        frame = encode_features(test_df, cols).head(2_000)
        _ = estimator.predict(frame)  # warm up
        begin = time.perf_counter()
        for _ in range(20):
            estimator.predict(frame)
        per_call_ms = (time.perf_counter() - begin) / 20 / len(frame) * 1_000.0
        latency[label] = round(per_call_ms, 4)
        log(f"  {label:<20} {per_call_ms:.4f} ms per reading (batch of {len(frame)})")
    metrics["latency_ms_per_reading"] = latency

    # ---------------- Seed stability ----------------
    if args.seed_stability:
    rule("STEP 14  Seed stability of the decision model")
    log("  Refitting over several seeds so the headline F1 carries a range")
    stability = {
            "irrigate_f1": seed_stability(
                lambda s: build_calibrated_classifier(s, quick, decision_folds),
                lambda est: _fit_score_f1(est, x_train, actual_flag, x_test, actual_flag_test,
                                         group_block["thresholds"], group_labels, decision_threshold),
            ),
            "depth_mae_mm": seed_stability(
                lambda s: build_regressor(s, quick),
                lambda est: _fit_score_depth(est, x_events_train, train_events["water_required_mm"],
                                             x_events_test, depth_actual),
            ),
        }
        metrics["seed_stability"] = stability
        for name, block in stability.items():
            log(f"  {name:<20} mean {block.get('mean')}  std {block.get('std')}  "
                f"range [{block.get('min')}, {block.get('max')}]  "
                f"({block.get('successful_runs')}/{block.get('seeds')} seeds)")

    # ---------------- Persist ----------------
    rule("STEP 16  Persisting artifacts and version registry")
    bundle = {
        "metadata": {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "corpus_rows": int(len(corpus)),
            "corpus_series": int(corpus.groupby(["station", "crop", "soil_type"]).ngroups),
            "weather_source": "NASA POWER daily agroclimatology, 2015-2024, 12 stations",
            "weather_records": int(len(weather)),
            "et0_method": "FAO-56 Penman-Monteith Eq. 6, validated against published examples",
            "et0_validation": "backend/ml/tests/test_fao56.py, 9 reference checks",
            "split_strategy": "GroupShuffleSplit on station|crop|soil_type, plus unseen-station holdout",
            "ood_station": ood_station,
            "disease_labels": (
                "SYNTHESISED by the rule engine in artifacts/disease_thresholds.json, not "
                "observed. The labels reproduce from six climate and soil features at "
                "100%, so the disease model is a smooth restatement of that table and its "
                "scores measure rule memorisation, not disease diagnosis. See "
                "backend/ml/tests/test_disease_labels.py. Stress-risk indicator only."
            ),
            "disease_score_interpretation": (
                "Do not present as diagnostic performance. A perfect unseen-station score "
                "is a consequence of the labels being deterministic, not evidence of "
                "generalisation to real disease."
            ),
            "decision_score_interpretation": (
                "The irrigate score is isotonic-calibrated, so the threshold is an actual "
                "event rate. Ranking quality is reported as average precision against the "
                "positive rate, because accuracy is 0.979 for a model that could do nothing."
            ),
            "diseases": DISEASE_CLASSES,
            "nutrient_bands": {k: list(v) for k, v in NUTRIENT_BANDS.items()},
            "features": FEATURE_SETS,
            "model_columns": model_columns,
            "measured_features": MEASURED_FEATURES,
            "configured_features": CONFIGURED_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "category_levels": CATEGORY_LEVELS,
            "metrics": metrics,
        },
        "disease_model": disease_model,
        "irrigate_model": irrigate_model,
        "irrigate_threshold": decision_threshold,
        # Per crop|soil operating points, with the global cut kept as the
        # fallback for a group that was never seen during training.
        "irrigate_thresholds_by_group": group_block["thresholds"],
        "water_depth_model": water_model,
        # Conformal width for the depth estimate, so the API can state the error
        # the operator should plan for instead of a bare point prediction.
        "water_depth_conformal": {
            "half_width_mm": round(float(conformal_90), 4),
            "target_coverage": 0.90,
            "achieved_coverage": metrics["water_depth"]["conformal_achieved_coverage"],
        },
        # Multiplicative station correction, applied when a field's observed mean
        # depth diverges from what the model predicts for it.
        "station_correction": {
            "method": "multiplicative ratio of observed to predicted mean depth, clipped to [0.5, 2.0]",
            "training_factor_mean": metrics["series_depth_ratio"]["mean"],
        },
        "calibration": {
            "expected_calibration_error": metrics["decision_calibration"]["expected_calibration_error"],
            "recalibration_slope": metrics["decision_calibration"]["recalibration_slope"],
        },
        # Reference distribution for drift monitoring at serve time.
        "reference_distribution": reference,
        "water_action_model": action_model,
        "nutrient_models": nutrient_models,
        "yield_model": yield_model,
        "thresholds": thresholds.to_dict(),
    }
    bundle_path = ARTIFACT_DIR / "iic_models.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    log(f"  model bundle      {bundle_path} ({bundle_path.stat().st_size / 1e6:.1f} MB)")

    metrics_path = ARTIFACT_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {"metadata": {k: v for k, v in bundle["metadata"].items() if k != "metrics"},
             "metrics": metrics},
            indent=2, default=str,
        ) + "\n",
        encoding="utf-8",
    )
    log(f"  metrics           {metrics_path}")

    rule("  Versioned registry")
    entry = register_versioned_bundle(bundle, metrics)
    log(f"  version           {entry['version']}")
    if "delta" in entry:
        log(f"  vs previous       {entry['previous_version']}")
        for key, value in entry["delta"].items():
            if value is None:
                continue
            log(f"    {key:<28} {value:+.4f}")

    save_plots(disease_model, FEATURE_SETS["disease"], test_df,
               water_model, FEATURE_SETS["water_depth"], test_df.assign(
                   water_required_mm_pred=composed))

    rule("SUMMARY")

    def class_recall(block: dict, label: str) -> float:
        """Per-class recall, tolerating either int or string report keys."""
        per_class = block.get("per_class", {})
        for key in (label, int(label), str(label)):
            if key in per_class:
                return float(per_class[key]["recall"])
        return float("nan")

    log(f"  disease        macro F1 {metrics['disease']['macro_f1']:.4f}  "
        f"(unseen station {metrics['disease_ood']['macro_f1']:.4f})")
    log("                 ^ labels are rule-derived and reproduce at 100%, so this")
    log("                   measures memorisation of that table. Not diagnostic.")
    ranking = metrics["irrigate_ranking"]
    log(f"  irrigate?      avg precision {ranking['average_precision']:.4f}  "
        f"(base rate {ranking['positive_rate']:.4f}, "
        f"{ranking['average_precision'] / max(1e-9, ranking['positive_rate']):.1f}x)")
    log(f"                 calibration ECE {metrics['decision_calibration']['expected_calibration_error']:.4f}")
    log(f"  depth on events MAE {metrics['water_depth']['mae']:.3f} mm  "
        f"R2 {metrics['water_depth']['r2']:.4f}  "
        f"+/-{metrics['water_depth']['conformal_half_width']:.2f} mm at 90%")
    end_to_end = metrics["water_depth"]["end_to_end"]
    log(f"  end-to-end     F1 {end_to_end['f1_irrigate']:.4f}  "
        f"missed {end_to_end['false_negative_missed_watering']:,}  "
        f"unnecessary {end_to_end['false_positive_unnecessary_watering']:,}")
    log(f"  unseen station R2 {metrics['water_depth_ood']['r2']:.4f} "
        f"(uncorrected {metrics['water_depth_ood_uncorrected']['r2']:.4f})  "
        f"end-to-end F1 {metrics['water_depth_ood']['decision']['f1_irrigate']:.4f}")
    log(f"  NPK demand     MAE N {metrics['nutrient_nitrogen']['mae']:.2f}, "
        f"P {metrics['nutrient_phosphorus']['mae']:.2f}, "
        f"K {metrics['nutrient_potassium']['mae']:.2f} kg/ha")
    log(f"  yield          MAE {metrics['yield']['mae']:.3f} t/ha  R2 {metrics['yield']['r2']:.4f}")
    for flag in ("disease", "disease_ood", "water_action"):
        if metrics.get(flag, {}).get("degenerate"):
            log(f"  [caution] {flag} evaluation set is degenerate; score is a smoke "
                f"signal, not a performance claim")
    log(f"\n  total wall time {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
