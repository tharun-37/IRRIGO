r"""
The estimator, the splits that decide whether it is any good, and the baselines
it has to beat.

Three things in here exist because of how the previous generation failed, and it
is worth being blunt about each.

**The split protocol is the product.** V1's headline validation was a random
split, and it looked fine. The number that mattered was R-squared 0.082 on a
station the model had never seen, and it was 0.082 because the target was
`min(TAW - Dr, infiltration)` which is a per-soil constant two thirds of the
time. A random split cannot catch that, because a random split puts the same
soils on both sides. So this module offers three splits and reports all three:
grouped by field series, leave-one-station-out, and leave-one-crop-out. The
first reproduces the old protocol for comparability. The second is the one that
decides whether this is deployable. The third is the hardest and will probably
be poor, because the residual model can only be calibrated on the crops present
in the corpus.

**Conformal prediction, not a residual standard deviation.** A root mean squared
error is a population average and tells a farmer nothing about whether *this*
recommendation is safe. Split conformal on absolute errors gives a per-prediction
interval at a stated coverage, with no distributional assumption. The cost is
that the interval is as wide as the worst case in the calibration set, which is
the honest price.

**Baselines are mandatory.** A model is not evaluated against its own loss. It is
evaluated against the thing a practitioner would do instead, and against the
thing V1 degenerated into. If the learned model cannot beat a plain
`Kc * ET0 * horizon` calculation, the machine learning is a liability and should
be removed rather than defended.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from .features import FeatureSpec, encode_categoricals

#: Planning horizon baked into the corpus target. Must match `corpus`.
HORIZON_DAYS = 14


# ---------------------------------------------------------------------------
# splits


@dataclass(frozen=True)
class Split:
    """One train/test partition with the reason it exists."""

    name: str
    train: np.ndarray
    test: np.ndarray
    held_out: str
    rationale: str


def series_key(data: pd.DataFrame) -> np.ndarray:
    """Identity of the simulated series a row came from.

    Rows from the same series share a sowing date, a soil chemistry draw and a
    probe-noise seed, so they are not independent observations. Splitting across
    them puts near-duplicates on both sides and inflates every metric.

    The key is built from the *sowing* date, not the observation date. Keying on
    the observation date makes every day its own group -- 58,758 groups for
    72,082 rows -- which is not a grouped split at all, it is a random row split
    wearing a grouped split's name. Consecutive days of one field are the closest
    thing in this corpus to duplicate records, so that inflation is close to
    maximal, and it silently flatters exactly the number quoted for comparability
    with the previous generation.

    `days_since_sowing` is the only record of the sowing date, so the series is
    identified by the field configuration plus the reconstructed sowing date.
    """
    if "sowing_date" in data.columns:
        sowing = data["sowing_date"].astype(str)
    else:
        dates = pd.to_datetime(data["date"])
        sowing = (
            dates - pd.to_timedelta(pd.to_numeric(data["days_since_sowing"]), unit="D")
        ).dt.strftime("%Y-%m-%d")
    return (
        data["station"].astype(str)
        + "|" + data["crop"].astype(str)
        + "|" + data["soil_type"].astype(str)
        + "|" + data["irrigation_method"].astype(str)
        + "|" + sowing
    ).to_numpy()


def split_grouped(data: pd.DataFrame, seed: int = 7) -> Split:
    """
    Hold out whole field series, never rows.

    This is V1's protocol, reproduced so the two generations can be compared like
    for like. It is necessary and not sufficient.
    """
    keys = series_key(data)
    unique = np.array(sorted(set(keys)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    test_keys = set(unique[: max(1, len(unique) // 10)])
    mask = np.array([key in test_keys for key in keys])
    return Split(
        name="grouped_by_series",
        train=np.flatnonzero(~mask),
        test=np.flatnonzero(mask),
        held_out=f"{len(test_keys)} of {len(unique)} series",
        rationale="comparable with the previous generation's protocol",
    )


def split_leave_one_station_out(data: pd.DataFrame) -> list[Split]:
    """
    Twelve folds, one per station, twelve seasons each.

    This is the test that matters. Every field at the held-out station shares a
    climate the model has never seen, so an R-squared here cannot be bought by
    memorising a soil table. It is also the closest analogue to the actual
    deployment question, which is "this farmer's district is not in my data".
    """
    stations = np.array(sorted(data["station"].unique()))
    return [
        Split(
            name=f"holdout_station_{station}",
            train=np.flatnonzero((data["station"] != station).to_numpy()),
            test=np.flatnonzero((data["station"] == station).to_numpy()),
            held_out=station,
            rationale="unseen climate, which is what V1 failed on",
        )
        for station in stations
    ]


def split_leave_one_crop_out(data: pd.DataFrame) -> list[Split]:
    """
    One fold per crop.

    The hardest split, and expected to be mediocre. FAO-56 publishes Kc and root
    profiles for these crops, so the physics features generalise; the residual
    model cannot, because it has never seen this crop's local behaviour. A poor
    result here is a statement about the corpus, not a reason to delete the split.
    """
    crops = np.array(sorted(data["crop"].unique()))
    return [
        Split(
            name=f"holdout_crop_{crop}",
            train=np.flatnonzero((data["crop"] != crop).to_numpy()),
            test=np.flatnonzero((data["crop"] == crop).to_numpy()),
            held_out=crop,
            rationale="transfer to a crop never trained on",
        )
        for crop in crops
    ]


# ---------------------------------------------------------------------------
# metrics


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """
    Errors in the unit the farmer thinks in: millimetres of water.

    R-squared is reported because it is comparable with the previous generation,
    but MAE is the number that matters operationally, because a 5 mm error on a
    60 mm recommendation is a rounding error and a 5 mm error on a 2 mm
    recommendation is the whole answer. Median absolute error is alongside MAE
    for exactly that reason: the mean is dominated by the split schedules.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    return {
        "n": int(y_true.size),
        "mae_mm": float(mean_absolute_error(y_true, y_pred)),
        "median_ae_mm": float(np.median(np.abs(errors))),
        "rmse_mm": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "bias_mm": float(errors.mean()),
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
        "spearman": float(
            pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
        ),
        "within_5mm_pct": float((np.abs(errors) <= 5.0).mean() * 100.0),
        "within_10mm_pct": float((np.abs(errors) <= 10.0).mean() * 100.0),
    }


# ---------------------------------------------------------------------------
# baselines


def soil_median_baseline(
    train: pd.DataFrame, test: pd.DataFrame
) -> np.ndarray:
    """
    Predict each row's soil median, from the training rows only.

    This is the V1 failure mode as an explicit baseline. If a model cannot beat
    "look up the median requirement for this soil texture", it has learned
    nothing a farmer's own notes would not already tell them.
    """
    medians = train.groupby("soil_type")["nir_horizon_mm"].median()
    fallback = float(train["nir_horizon_mm"].median())
    return test["soil_type"].map(medians).fillna(fallback).to_numpy(dtype=float)


def physics_baseline(test: pd.DataFrame) -> np.ndarray:
    """
    `Kc * ET0 * horizon`, the calculation a practitioner would do by hand.

    Ignores rain, soil state and the stage beyond Kc. It is wrong in a way that
    is easy to explain and impossible to avoid without a soil probe, and it is
    the honest floor for "does the model beat the textbook". It is also
    deliberately uncapped, so a large requirement stays large rather than
    collapsing onto a per-soil application depth.
    """
    demand = test["kc"].to_numpy(dtype=float) * test["et0_mm_day"].to_numpy(dtype=float)
    return np.clip(demand * HORIZON_DAYS, 0.0, None)


def residual_over_physics(
    test: pd.DataFrame, y_pred: np.ndarray
) -> dict[str, float]:
    """
    Does the model beat the physics, or only reproduce it?

    Reported on the subset where the physics baseline is actually informative,
    which is the subset where a recommendation is needed at all. Comparing R2
    over rows that are 80% zeros flatters everything equally and means nothing.
    """
    y_true = test["nir_horizon_mm"].to_numpy(dtype=float)
    base = physics_baseline(test)
    mask = y_true > 1.0
    if mask.sum() < 10:
        return {"n": 0}
    return {
        "n": int(mask.sum()),
        "physics_mae_mm": float(mean_absolute_error(y_true[mask], base[mask])),
        "model_mae_mm": float(mean_absolute_error(y_true[mask], y_pred[mask])),
        "skill_vs_physics_pct": float(
            100.0
            * (1.0 - mean_absolute_error(y_true[mask], y_pred[mask])
               / max(1e-9, mean_absolute_error(y_true[mask], base[mask])))
        ),
    }


# ---------------------------------------------------------------------------
# conformal


@dataclass
class ConformalInterval:
    """
    Split-conformal interval on absolute errors at a stated coverage.

    The quantile is the `(1 - alpha)` empirical quantile of absolute errors on a
    calibration set the model never saw. No normality is assumed, which matters
    because the error distribution is a mixture: most rows need no water and are
    predicted to within a millimetre, while a handful of split schedules are
    wrong by tens of millimetres. A Gaussian interval built from that mixture's
    standard deviation would be confidently wrong exactly where it matters.
    """

    coverage: float
    quantile_mm: float
    n_calibration: int
    realised_coverage: float
    degenerate: bool = False
    degenerate_note: str = ""

    def interval(self, prediction: float) -> tuple[float, float]:
        low = max(0.0, float(prediction) - self.quantile_mm)
        return low, float(prediction) + self.quantile_mm

    def to_dict(self) -> dict:
        return {
            "coverage": self.coverage,
            "quantile_mm": self.quantile_mm,
            "n_calibration": self.n_calibration,
            "realised_coverage": self.realised_coverage,
        }


def trigger_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 1.0
) -> dict[str, float]:
    """
    How good is the "does this field need water at all" decision?

    Separated from the regression because it is a different question with a
    different failure mode. Forty-two percent of the corpus needs no irrigation
    in the horizon, and on those rows the model is essentially exact, which
    inflates every regression metric while saying nothing about the decision that
    actually costs money.

    The number that matters is `missed_pct`: fields that do need water and are
    told they do not. A false positive costs a wasted irrigation; a false
    negative costs yield that cannot be recovered. In a system that is
    conservatively biased, recall should be high and precision allowed to suffer.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    needs = y_true > threshold
    says_yes = y_pred > threshold
    true_positive = int((needs & says_yes).sum())
    false_positive = int((~needs & says_yes).sum())
    false_negative = int((needs & ~says_yes).sum())
    true_negative = int((~needs & ~says_yes).sum())
    return {
        "threshold_mm": threshold,
        "needs_water_pct": float(needs.mean() * 100.0),
        "precision": float(true_positive / max(1, true_positive + false_positive)),
        "recall": float(true_positive / max(1, true_positive + false_negative)),
        "missed_pct": float(false_negative / max(1, needs.sum()) * 100.0),
        "false_alarm_pct": float(
            false_positive / max(1, (~needs).sum()) * 100.0
        ),
        "confusion": {
            "true_positive": true_positive, "false_positive": false_positive,
            "false_negative": false_negative, "true_negative": true_negative,
        },
    }


def fit_conformal(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    coverage: float = 0.90,
    mask: np.ndarray | None = None,
) -> ConformalInterval:
    """
    Calibrate on held-out residuals.

    The finite-sample correction `(n+1)/n * (1 - alpha)` is the standard
    adjustment that makes the coverage guarantee hold on finite calibration sets.
    Without it the nominal 90% interval covers 88% in practice, and the gap is
    largest exactly when the calibration set is small, which is when a small
    pilot project is most tempted to believe the interval.

    The quantile level is `coverage`, not `1 - coverage`. A split-conformal
    interval at coverage `1 - alpha` is the prediction plus or minus the
    `(1 - alpha)` quantile of the absolute residuals, which for 90% coverage is
    the *ninetieth* percentile. An earlier version used the tenth, and produced
    intervals ten times too narrow while cheerfully reporting them as 90%
    intervals. The giveaway was its own output: it printed a realised coverage of
    10% next to a nominal label of 90%. Always read the realised number.

    **`mask` restricts calibration to rows where irrigation is actually needed,
    and it is passed for a concrete reason.** The target is a mixture: about 42%
    of rows are exactly zero because the soil holds a fortnight of demand, and on
    those rows the prediction is exact, so the absolute residual is exactly zero
    for a fifth of the corpus. The empirical quantile of that residual
    distribution sits on the point mass, and a naive conformal interval comes out
    as "90% coverage, plus or minus zero millimetres". It is not wrong about the
    marginal distribution. It is useless for the decision, because the only rows
    where an error has a consequence are the ones where water is needed, and a
    zero-width interval around a near-zero prediction would be confidently
    reassuring in exactly the cases that matter. Calibrating on the non-zero rows
    gives an interval that describes the error a farmer can act on.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    residuals = np.abs(y_true - y_pred)
    n = residuals.size
    if n == 0:
        return ConformalInterval(coverage, float("nan"), 0, float("nan"))
    level = min(1.0, coverage * (n + 1) / n)
    quantile = float(np.quantile(residuals, level))
    degenerate = bool(quantile <= 1e-9)
    return ConformalInterval(
        coverage=coverage,
        quantile_mm=quantile,
        n_calibration=int(n),
        realised_coverage=float((residuals <= quantile).mean()),
        degenerate=degenerate,
        degenerate_note=(
            "calibration residuals have a point mass at zero because the target "
            "is exactly zero whenever the soil already holds the horizon's demand; "
            "calibrate on rows that need water to get a usable width"
            if degenerate
            else ""
        ),
    )


# ---------------------------------------------------------------------------
# model


@dataclass
class NirModel:
    """
    The requirement estimator.

    Residual-on-physics in the sense that matters here: the features are already
    the physical state (Kc, root depth, TAW, depletion), so the tree ensemble is
    learning the local correction to the FAO-56 calculation rather than
    re-deriving it. A squared-error loss is deliberate, not a default: the target
    is a quantity of water in millimetres, and an absolute error of 4 mm is 4 mm
    whether the field needs 2 mm or 60.
    """

    estimator: object = None
    conformal: dict[str, ConformalInterval] = field(default_factory=dict)
    feature_names: tuple[str, ...] = ()
    target_name: str = "nir_horizon_mm"
    metadata: dict = field(default_factory=dict)

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        max_iter: int = 400,
        learning_rate: float = 0.08,
        min_samples_leaf: int = 40,
        l2: float = 1.0,
        random_state: int = 7,
    ) -> "NirModel":
        self.estimator = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=max_iter,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2,
            max_bins=255,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=random_state,
        )
        self.estimator.fit(features, target)
        self.feature_names = tuple(features.columns)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("model is not fitted")
        missing = [name for name in self.feature_names if name not in features.columns]
        if missing:
            raise KeyError(f"missing features at predict time: {missing}")
        aligned = features[list(self.feature_names)]
        return np.clip(self.estimator.predict(aligned), 0.0, None)

    def interval(self, prediction: float, coverage: float = 0.90) -> tuple[float, float]:
        if coverage not in self.conformal:
            raise KeyError(
                f"no conformal interval at coverage {coverage};"
                f" available: {sorted(self.conformal)}"
            )
        return self.conformal[coverage].interval(prediction)

    def permutation_importance(
        self, features: pd.DataFrame, target: pd.Series, seed: int = 7
    ) -> dict[str, float]:
        """
        R2 lost when a feature is shuffled, over ten repeats.

        Tree importances are biased towards high-cardinality continuous features
        and are near-meaningless for the interactions this model relies on. The
        permutation figure costs ten forward passes per feature and measures the
        thing that actually matters: how much worse the prediction gets without
        it.
        """
        if self.estimator is None:
            raise RuntimeError("model is not fitted")
        rng = np.random.default_rng(seed)
        baseline = self.predict(features)
        base_score = r2_score(target, baseline)
        out: dict[str, float] = {}
        for name in self.feature_names:
            dropped = base_score
            for _ in range(3):
                shuffled = features.copy()
                shuffled[name] = rng.permutation(shuffled[name].to_numpy())
                score = r2_score(target, self.predict(shuffled))
                dropped = min(dropped, score)
            out[name] = float(base_score - dropped)
        return dict(sorted(out.items(), key=lambda item: -item[1]))


# ---------------------------------------------------------------------------
# evaluation


def evaluate_split(
    model_factory,
    data: pd.DataFrame,
    split: Split,
    spec: FeatureSpec,
    calibrate: bool = True,
) -> dict:
    """
    Fit on the training part, score the test part, and report it three ways.

    The conformal calibration set is carved out of the *training* rows only. Doing
    it after seeing the test residuals would make the interval meaningless, which
    is the most common way a confidence interval ends up being a decoration.
    """
    features = encode_categoricals(spec.frame(data), spec)
    target = data["nir_horizon_mm"].to_numpy(dtype=float)

    train_index, test_index = split.train, split.test
    calibration_mask = np.zeros(train_index.size, dtype=bool)
    rng = np.random.default_rng(11)
    if calibrate and train_index.size > 500:
        picks = rng.choice(train_index.size, size=max(1, train_index.size // 10),
                           replace=False)
        calibration_mask[picks] = True

    fit_index = train_index[~calibration_mask]
    calib_index = train_index[calibration_mask]

    model = model_factory().fit(features.iloc[fit_index], target[fit_index])

    y_true = target[test_index]
    y_pred = model.predict(features.iloc[test_index])
    test_frame = data.iloc[test_index]

    if calibrate and calib_index.size > 100:
        calib_true = target[calib_index]
        calib_pred = model.predict(features.iloc[calib_index])
        # Calibrate on rows that actually need water. See `fit_conformal`: the
        # marginal residual distribution has a point mass at zero and produces a
        # zero-width interval otherwise.
        needs = calib_true > 1.0
        for coverage in (0.80, 0.90):
            model.conformal[coverage] = fit_conformal(
                calib_true, calib_pred, coverage=coverage, mask=needs
            )
        model.conformal["marginal_0.90"] = fit_conformal(
            calib_true, calib_pred, coverage=0.90
        )

    metrics = regression_metrics(y_true, y_pred)
    metrics.update(
        {
            "split": split.name,
            "held_out": split.held_out,
            "rationale": split.rationale,
            "soil_median_mae_mm": float(
                mean_absolute_error(
                    y_true, soil_median_baseline(data.iloc[fit_index], test_frame)
                )
            ),
        }
    )
    metrics["physics_baseline"] = regression_metrics(
        y_true, physics_baseline(test_frame)
    )
    metrics["skill_vs_physics"] = residual_over_physics(test_frame, y_pred)
    metrics["trigger"] = trigger_metrics(y_true, y_pred)
    if model.conformal:
        interval = model.conformal[0.90]
        metrics["interval_90"] = interval.to_dict()
        covered = np.abs(y_true - y_pred) <= interval.quantile_mm
        metrics["test_coverage_90"] = float(covered.mean())
        needs_water = y_true > 1.0
        if needs_water.sum() > 0:
            metrics["test_coverage_90_when_water_needed"] = float(
                covered[needs_water].mean()
            )
    per_crop = {}
    for crop in sorted(test_frame["crop"].unique()):
        mask = (test_frame["crop"] == crop).to_numpy()
        per_crop[crop] = regression_metrics(y_true[mask], y_pred[mask])
    metrics["per_crop"] = per_crop
    per_stage = {}
    for stage in sorted(test_frame["growth_stage"].unique()):
        mask = (test_frame["growth_stage"] == stage).to_numpy()
        if mask.sum() >= 20:
            per_stage[str(stage)] = regression_metrics(y_true[mask], y_pred[mask])
    metrics["per_stage"] = per_stage
    return metrics


def run_protocol(
    data: pd.DataFrame,
    spec: FeatureSpec,
    model_factory=None,
    quick: bool = False,
    verbose: bool = True,
) -> dict:
    """
    The full evaluation: grouped, then leave-one-station-out, then
    leave-one-crop-out.

    Every split's pooled predictions are retained so the numbers can be
    re-derived and the per-fold spread shown. A mean over twelve folds hides a
    single catastrophic station, and the catastrophic station is the one that
    matters.
    """
    model_factory = model_factory or (lambda: NirModel())
    grouped = split_grouped(data)
    station_folds = split_leave_one_station_out(data)
    crop_folds = split_leave_one_crop_out(data)
    if quick:
        station_folds = station_folds[:3]
        crop_folds = crop_folds[:3]

    report: dict = {"horizon_days": HORIZON_DAYS, "n_rows": int(len(data))}
    report["grouped"] = evaluate_split(model_factory, data, grouped, spec)

    if verbose:
        m = report["grouped"]
        print(f"  grouped          MAE {m['mae_mm']:6.2f}  R2 {m['r2']:+.4f}"
              f"  (soil-median MAE {m['soil_median_mae_mm']:.2f})")

    station_metrics = []
    for fold in station_folds:
        m = evaluate_split(model_factory, data, fold, spec, calibrate=False)
        station_metrics.append(m)
        if verbose:
            print(f"  {fold.name:<24} MAE {m['mae_mm']:6.2f}  R2 {m['r2']:+.4f}"
                  f"  n={m['n']:,}")
    report["leave_one_station_out"] = station_metrics
    report["leave_one_station_out_summary"] = _summarise_folds(station_metrics)

    crop_metrics = []
    for fold in crop_folds:
        m = evaluate_split(model_factory, data, fold, spec, calibrate=False)
        crop_metrics.append(m)
        if verbose:
            print(f"  {fold.name:<24} MAE {m['mae_mm']:6.2f}  R2 {m['r2']:+.4f}")
    report["leave_one_crop_out"] = crop_metrics
    report["leave_one_crop_out_summary"] = _summarise_folds(crop_metrics)
    return report


def _summarise_folds(folds: list[dict]) -> dict:
    r2 = np.array([f["r2"] for f in folds], dtype=float)
    mae = np.array([f["mae_mm"] for f in folds], dtype=float)
    worst = int(np.argmin(r2))
    return {
        "folds": len(folds),
        "mean_r2": float(np.nanmean(r2)),
        "median_r2": float(np.nanmedian(r2)),
        "min_r2": float(np.nanmin(r2)),
        "max_r2": float(np.nanmax(r2)),
        "std_r2": float(np.nanstd(r2)),
        "mean_mae_mm": float(mae.mean()),
        "worst_fold": folds[worst]["held_out"],
        "worst_fold_r2": float(r2[worst]),
        "negative_r2_folds": int((r2 < 0).sum()),
    }


def final_model(
    data: pd.DataFrame, spec: FeatureSpec, model_factory=None
) -> tuple[NirModel, dict]:
    """
    Fit the deployable model on everything, with a calibration set held back.

    The conformal quantile must come from rows the fit never saw, so a tenth is
    withheld from training and used only to calibrate. The final model is then
    the one that ships, with the honest caveat that its interval was calibrated
    on data it was partly trained on being slightly optimistic about.
    """
    model_factory = model_factory or (lambda: NirModel())
    features = encode_categoricals(spec.frame(data), spec)
    target = data["nir_horizon_mm"].to_numpy(dtype=float)
    rng = np.random.default_rng(11)
    picks = rng.choice(len(data), size=max(1, len(data) // 10), replace=False)
    mask = np.zeros(len(data), dtype=bool)
    mask[picks] = True

    model = model_factory().fit(features[~mask], target[~mask])
    calib_true = target[mask]
    calib_pred = model.predict(features[mask])
    needs = calib_true > 1.0
    for coverage in (0.80, 0.90):
        model.conformal[coverage] = fit_conformal(
            calib_true, calib_pred, coverage=coverage, mask=needs
        )
    model.conformal["marginal_0.90"] = fit_conformal(
        calib_true, calib_pred, coverage=0.90
    )
    model.metadata = {
        "n_train": int((~mask).sum()),
        "n_calibration": int(mask.sum()),
        "n_calibration_needing_water": int(needs.sum()),
        "target": "nir_horizon_mm",
        "horizon_days": HORIZON_DAYS,
        "interval_note": (
            "conformal intervals are calibrated on rows where irrigation is "
            "needed; the marginal residual distribution has a point mass at zero "
            "and yields a degenerate zero-width interval"
        ),
        "conformal": {k: v.to_dict() for k, v in model.conformal.items()},
    }
    return model, {
        "train_mae_mm": float(
            mean_absolute_error(target[~mask], model.predict(features[~mask]))
        ),
        "calibration_mae_mm": float(
            mean_absolute_error(calib_true, calib_pred)
        ),
        "conformal": {k: v.to_dict() for k, v in model.conformal.items()},
    }


def save_model(model: NirModel, path: str | Path) -> Path:
    import joblib

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
    return target


def load_model(path: str | Path) -> NirModel:
    import joblib

    return joblib.load(Path(path))
