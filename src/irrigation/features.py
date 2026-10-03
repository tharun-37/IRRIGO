r"""
The feature contract: what the model is given, and what it must never be given.

This module exists to enforce one rule. The model learns the *residual* on top of
the physics, not the physics itself. Everything the FAO-56 tables already state
has been folded into a physical state variable before these features are built,
so the estimator spends its capacity on the part nobody can look up: the local
deviation of a real field from its reference crop and soil.

The evidence that this matters is in the previous generation's results. It fitted
a model to a target that was `min(TAW - Dr, infiltration)`, which is a per-soil
constant two thirds of the time, and the model duly learned the soil texture. Its
out-of-distribution R-squared on an unseen station was 0.082. A model given raw
inputs will always find the nearest lookup table in the data, and that table is
the one thing guaranteed not to transfer.

Two features are therefore banned outright, and the ban is a runtime assertion
rather than a comment:

`water_required_mm` and its aliases. That was V1's target. A feature that equals
the target is a leak, and the earlier version of this pipeline computed the
requirement and then handed it to the model as an input, which would have
produced a near-perfect R-squared and a useless model.

Crop and soil identity as bare one-hots. Crop identity has already been collapsed
into Kc, depletion fraction, root depth and GDD thresholds by the time features
are built. What remains that genuinely varies is a handful of coefficients, and
those are exposed as continuous interaction features instead of letting the model
spend depth memorising fourteen categories.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Names that would leak the target if they reached the estimator.
FORBIDDEN: frozenset[str] = frozenset({
    "nir_horizon_mm",
    "water_required_mm",
    "single_application_mm",
    "horizon_demand_mm",
    "horizon_rain_credit_mm",
    "horizon_soil_supply_mm",
    "projected_stress_days",
    "irrigation_applied_mm",
    "date",
    "year",
    "doy",
    "station",
})

#: Numeric features, in the order they are fed to the estimator.
NUMERIC_FEATURES: tuple[str, ...] = (
    # -- measured atmosphere ------------------------------------------------
    "et0_mm_day",
    "vpd_kpa",
    "humidity_pct",
    "air_temperature_c",
    "tmax_c",
    "tmin_c",
    "wind_speed_m_s",
    "solar_radiation_mj_m2_day",
    "rainfall_mm",
    "is_monsoon",
    # -- the crop, as the physics already reduced it to ----------------------
    "kc",
    "etc_mm",
    "root_depth_cm",
    "gdd_accumulated",
    "gdd_to_next_stage",
    "season_progress",
    "days_since_sowing",
    "p_depletion_fraction",
    # -- the soil, likewise -------------------------------------------------
    "taw_mm",
    "raw_mm",
    "depletion_mm",
    "depletion_fraction",
    # `soil_moisture_pct` was in this list and is gone: it is `soil_moisture_fraction`
    # multiplied by a hundred, so a tree can split on it at identical points and
    # the permutation audit reported it as contributing nothing while occupying a
    # column. Volumetric fraction is the unit the physics uses and the unit the
    # probe reports, so there is no reason to carry both.
    "soil_moisture_fraction",
    "soil_temperature_c",
    "soil_ph",
    "ec_ds_m",
    # -- the management, which the previous generation ignored entirely -------
    "field_area_m2",
    "mulched",
    "nitrogen_regime",
)

#: Categorical features, low-cardinality on purpose. `irrigation_method` decides
#: the application limit and the turn-around, so it genuinely changes what can be
#: recommended and cannot be reduced to a number. Nothing else survives: crop and
#: soil identity are excluded by the reasoning in the module docstring.
CATEGORICAL_FEATURES: tuple[str, ...] = ("irrigation_method",)

#: The full domain of each categorical, declared rather than inferred.
#:
#: Inferred from the corpus would make the model's input depend on which rows
#: happened to be built, so a re-run that lost a method would silently produce a
#: differently shaped estimator and a saved model would reject valid input. A
#: fixed domain means the encoded columns are stable across rebuilds, and a method
#: absent from the corpus simply gets an all-zero column, which the tree splits
#: on as "never seen this" rather than crashing on.
CATEGORICAL_VALUES: dict[str, tuple[str, ...]] = {
    "irrigation_method": (
        "Basin", "Drip", "Flood", "Furrow", "Sprinkler", "Paddy",
    ),
}

#: Growth stage, ordinal rather than nominal. The stages are an ordered sequence,
#: and a nominal encoding would let the model treat initial and late-season as
#: unrelated categories when in fact they are the two ends of the same curve.
STAGE_ORDER: tuple[str, ...] = ("initial", "development", "mid_season", "late_season")


@dataclass(frozen=True)
class FeatureSpec:
    """The full contract, including the ordinal encoding of growth stage."""

    numeric: tuple[str, ...] = NUMERIC_FEATURES
    categorical: tuple[str, ...] = CATEGORICAL_FEATURES

    @property
    def all_names(self) -> tuple[str, ...]:
        return self.numeric + self.categorical

    @property
    def derived_names(self) -> tuple[str, ...]:
        """Columns this module computes rather than reading, for the manifest."""
        return ("stage_ordinal",)

    def frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build the estimator's input, asserting the contract holds.

        The assertion is on the columns that must be *read*, not on the ones this
        module derives. A missing feature is a bug that should surface here rather
        than as a silently degraded model at three in the morning, and a forbidden
        column reaching the estimator is a leak that should never train at all.
        """
        missing = [name for name in self.all_names if name not in data.columns]
        if missing:
            raise KeyError(f"corpus is missing required features: {missing}")
        if "growth_stage" not in data.columns:
            raise KeyError(
                "corpus is missing 'growth_stage', which the stage ordinal is "
                "derived from"
            )

        present_forbidden = FORBIDDEN.intersection(data.columns)
        # `date`, `year` and `doy` are retained in the corpus for reporting and
        # for the grouped split. They are simply never selected here, so their
        # presence in the frame is not itself a leak.
        leaking = present_forbidden.intersection(self.all_names)
        if leaking:
            raise ValueError(
                f"refusing to build features: {sorted(leaking)} would leak the target"
            )

        out = data[list(self.numeric)].astype(float)

        for column in self.categorical:
            out[column] = data[column].astype(str)

        out["stage_ordinal"] = (
            data["growth_stage"].astype(str).map(
                {stage: float(i) for i, stage in enumerate(STAGE_ORDER)}
            ).fillna(-1.0)
        )
        return out


def encode_categoricals(
    features: pd.DataFrame, spec: "FeatureSpec | None" = None
) -> pd.DataFrame:
    """
    Turn the categorical columns into fixed one-hot columns.

    One-hot rather than ordinal because the methods are nominal: a `Furrow`
    column being 3 and a `Basin` column being 0 says nothing, and an integer code
    would let a tree learn a false ordering and interpolate between them. The
    estimator is a tree ensemble, so the one-hot is also what lets it find the
    interaction between method and soil without being told to.

    The output column order is fixed by `CATEGORICAL_VALUES`, not by the data, so
    a model saved today still accepts a frame built tomorrow.
    """
    spec = spec or FeatureSpec()
    out = features.select_dtypes(include=[np.number]).copy()
    for column in spec.categorical:
        if column not in features.columns:
            raise KeyError(f"missing categorical column {column!r}")
        for value in CATEGORICAL_VALUES.get(column, tuple()):
            out[f"{column}__{value}"] = (
                features[column].astype(str) == value
            ).astype(float)
    return out


def assert_no_leakage(features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    """
    Correlate every feature against the target and report the worst.

    Not a pass/fail gate, because a genuinely informative feature *will* correlate.
    It is a review aid: a single feature correlating above 0.95 with the target is
    almost always a leak, and a feature correlating at 0.00 is almost always
    something the corpus failed to compute. Both are bugs, and they look the same
    in a training log unless they are checked explicitly.
    """
    aligned = features.select_dtypes(include=[np.number])
    correlations = {}
    for column in aligned.columns:
        series = aligned[column]
        if series.std(skipna=True) in (0.0, None) or pd.isna(series.std(skipna=True)):
            correlations[column] = float("nan")
            continue
        correlations[column] = float(series.corr(target))
    return correlations


def leakage_report(features: pd.DataFrame, target: pd.Series) -> list[str]:
    """Human-readable leakage and dead-feature findings."""
    findings: list[str] = []
    correlations = assert_no_leakage(features, target)

    for column, value in sorted(
        correlations.items(), key=lambda item: abs(item[1]), reverse=True
    ):
        if np.isnan(value):
            findings.append(f"{column}: constant, carries no information")
        elif abs(value) > 0.95:
            findings.append(
                f"{column}: correlation {value:+.4f} with the target, almost "
                f"certainly a leak"
            )
        elif abs(value) < 1e-4:
            findings.append(f"{column}: correlation ~0, likely not computed correctly")

    return findings
