"""Head-to-head against the previous generation, and diagnosis of the weak folds.

The comparison is the point of the exercise. A new system that is merely better
on the metric it was tuned for has demonstrated nothing, so this reports the
previous generation's headline numbers next to this one's, and then asks the two
questions that actually decide whether the difference is real:

**Why does the previous generation fail out of distribution?** Because its target
was `min(TAW - Dr, infiltration)`, and the second term is a per-soil constant, so
two thirds of all its labels were one of three numbers. Reproduce that in one line
and the failure is not mysterious: the model learned the texture, and a texture
table learned from eleven other stations says nothing about the twelfth.

**Why does one station still fail here?** Leave-one-station-out over twelve
stations is reported fold by fold rather than as a mean, because a mean hides the
one station that would embarrass the system in the field. KOL is that station and
it is diagnosed rather than averaged away.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irrigation.features import FeatureSpec, encode_categoricals  # noqa: E402
from irrigation.models import (  # noqa: E402
    split_grouped,
    split_leave_one_station_out,
)

#: Headline figures from the previous generation's own evaluation, recorded here
#: rather than recomputed because that generation is frozen and its artefacts are
#: in a separate repository. Sourced from its README.
V1_HEADLINE = {
    "target": "min(TAW - Dr, infiltration)",
    "distinct_target_values": 3,
    "modal_target_share_pct": 39.13,
    "modal_value_mm": 22.0,
    "events_at_pure_application_cap_pct": 66.36,
    "soil_lookup_within_1mm_pct": 66.92,
    "ood_r2_unseen_station": 0.0816,
}


def diagnose_station(
    data: pd.DataFrame, features: pd.DataFrame, target: np.ndarray, station: str
) -> dict:
    """Why does this station fold behave as it does?"""
    test_mask = (data["station"] == station).to_numpy()
    test_frame = data[test_mask]
    y_true = target[test_mask]

    split = next(
        s for s in split_leave_one_station_out(data) if s.held_out == station
    )
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=400, learning_rate=0.08,
        min_samples_leaf=40, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=30, random_state=7,
    )
    model.fit(features.iloc[split.train], target[split.train])
    y_pred = np.clip(model.predict(features.iloc[split.test]), 0.0, None)

    train_frame = data.iloc[split.train]
    zero_test = float((y_true <= 1.0).mean())
    zero_train = float((train_frame["nir_horizon_mm"] <= 1.0).mean())
    return {
        "station": station,
        "n_test": int(test_mask.sum()),
        "n_train": int(len(split.train)),
        "zero_target_pct_test": zero_test * 100.0,
        "zero_target_pct_train": zero_train * 100.0,
        "test_et0_mean": float(test_frame["et0_mm_day"].mean()),
        "train_et0_mean": float(train_frame["et0_mm_day"].mean()),
        "test_rain_mean": float(test_frame["rainfall_mm"].mean()),
        "train_rain_mean": float(train_frame["rainfall_mm"].mean()),
        "test_annual_rain_mm": float(test_frame["rainfall_mm"].sum() / 5.0),
        "r2": float(1 - ((y_true - y_pred) ** 2).sum()
                    / ((y_true - y_true.mean()) ** 2).sum()),
        "mae_mm": float(np.abs(y_true - y_pred).mean()),
        "mae_when_water_needed_mm": float(
            np.abs(y_true - y_pred)[y_true > 1.0].mean()
        ) if (y_true > 1.0).any() else float("nan"),
        "mae_when_no_water_mm": float(
            np.abs(y_true - y_pred)[y_true <= 1.0].mean()
        ) if (y_true <= 1.0).any() else float("nan"),
        "predicted_rain_correlation": float(
            pd.Series(y_pred).corr(
                pd.Series(test_frame["rainfall_mm"].to_numpy()), method="spearman"
            )
        ),
        "test_crops": sorted(test_frame["crop"].unique().tolist()),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=str(ROOT / "data" / "crop_corpus.csv"))
    parser.add_argument("--evaluation", default=str(ROOT / "reports" / "evaluation.json"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "v1_vs_v2.json"))
    args = parser.parse_args(argv)

    print("loading corpus and evaluation ...")
    data = pd.read_csv(args.corpus, low_memory=False)
    evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    spec = FeatureSpec()
    features = encode_categoricals(spec.frame(data), spec)
    target = data["nir_horizon_mm"].to_numpy(dtype=float)

    protocol = evaluation["protocol"]
    grouped = protocol["grouped"]
    loso = protocol["leave_one_station_out_summary"]
    loco = protocol["leave_one_crop_out_summary"]

    print("\n=== the label, which is where the previous generation failed ===")
    v1 = V1_HEADLINE
    distinct = int(pd.Series(target).round(4).nunique())
    modal = target[target > 1.0]
    modal_share = float(
        (pd.Series(modal).value_counts().iloc[0] / max(1, modal.size)) * 100.0
    ) if modal.size else 0.0
    # Reproduce the old defect directly, on this corpus, to show what it would cost.
    taw = data["taw_mm"].to_numpy()
    infiltration_cap = data["max_application_mm"].to_numpy()
    old_target = np.minimum(taw, infiltration_cap)
    old_distinct = int(pd.Series(old_target).round(4).nunique())
    old_modal = old_target[old_target > 1.0]
    old_modal_share = float(
        (pd.Series(old_modal).value_counts().iloc[0] / max(1, old_modal.size)) * 100.0
    ) if old_modal.size else 0.0

    print(f"{'property':<44}{'V1':>14}{'V2':>14}")
    rows = [
        ("target definition", "min(TAW-Dr, infil)", "uncapped NIR"),
        ("distinct target values", f"{v1['distinct_target_values']:,}",
         f"{distinct:,}"),
        ("modal share among rows needing water",
         f"{v1['modal_target_share_pct']:.2f}%", f"{modal_share:.2f}%"),
        ("mean R2, unseen station", f"{v1['ood_r2_unseen_station']:+.4f}",
         f"{loso['mean_r2']:+.4f}"),
        ("worst unseen station", "n/a", f"{loso['worst_fold_r2']:+.4f} "
                                     f"({loso['worst_fold']})"),
        ("folds with negative R2", "n/a",
         f"{loso['negative_r2_folds']}/{loso['folds']}"),
    ]
    for label, old, new in rows:
        print(f"{label:<44}{old:>14}{new:>14}")

    print("\n  if this corpus had used V1's target definition:")
    print(f"    distinct values {old_distinct:,}  modal share {old_modal_share:.2f}%")
    print(f"    -> the same degeneracy, reproduced")

    print("\n=== the current evaluation ===")
    print(f"  grouped by series      MAE {grouped['mae_mm']:6.2f} mm  "
          f"R2 {grouped['r2']:+.4f}  n={grouped['n']:,}")
    skill = grouped.get("skill_vs_physics", {})
    if skill.get("n"):
        print(f"  vs the textbook Kc*ET0 calculation on rows that need water:")
        print(f"    model {skill['model_mae_mm']:.2f} mm vs "
              f"physics {skill['physics_mae_mm']:.2f} mm  "
              f"({skill['skill_vs_physics_pct']:+.1f}% skill, n={skill['n']:,})")
    trigger = grouped.get("trigger", {})
    if trigger:
        print(f"  'needs water at all' decision: recall {trigger['recall']:.3f}, "
              f"missed {trigger['missed_pct']:.2f}%, "
              f"false alarms {trigger['false_alarm_pct']:.2f}%")
    interval = grouped.get("interval_90")
    if interval:
        print(f"  conformal 90% interval +/- {interval['quantile_mm']:.1f} mm "
              f"(test coverage {grouped.get('test_coverage_90', float('nan')):.1%})")

    print("\n=== diagnosing the weak folds ===")
    weak = [f for f in protocol["leave_one_station_out"] if f["r2"] < 0.3]
    weak += [f for f in protocol["leave_one_crop_out"] if f["r2"] < 0.6]
    if not weak:
        print("  no fold below threshold")
    for fold in weak:
        kind = "station" if fold["held_out"] in set(data["station"]) else "crop"
        if kind != "station":
            print(f"\n  crop {fold['held_out']}: R2 {fold['r2']:+.4f}, "
                  f"MAE {fold['mae_mm']:.2f} mm")
            per_crop = fold.get("per_crop", {})
            if fold["held_out"] in per_crop:
                entry = per_crop[fold["held_out"]]
                print(f"    on its own rows: MAE {entry['mae_mm']:.2f} mm, "
                      f"median AE {entry['median_ae_mm']:.2f} mm, "
                      f"within 10 mm {entry['within_10mm_pct']:.1f}%")
            continue
        entry = diagnose_station(data, features, target, fold["held_out"])
        print(f"\n  station {entry['station']}: R2 {entry['r2']:+.4f}, "
              f"MAE {entry['mae_mm']:.2f} mm")
        print(f"    rows needing no water: {entry['zero_target_pct_test']:.1f}% "
              f"of test vs {entry['zero_target_pct_train']:.1f}% of train")
        print(f"    mean ET0 {entry['test_et0_mean']:.2f} mm/d (train "
              f"{entry['train_et0_mean']:.2f}); mean rain "
              f"{entry['test_rain_mean']:.2f} mm/d (train "
              f"{entry['train_rain_mean']:.2f})")
        print(f"    error when water IS needed: {entry['mae_when_water_needed_mm']:.2f} mm")
        print(f"    error when it is not:        {entry['mae_when_no_water_mm']:.2f} mm")

    report = {
        "v1_headline": V1_HEADLINE,
        "v2": {
            "grouped": {k: v for k, v in grouped.items()
                        if k not in ("per_crop", "per_stage")},
            "leave_one_station_out_summary": loso,
            "leave_one_crop_out_summary": loco,
            "leave_one_station_out": [
                {k: v for k, v in f.items() if k not in ("per_crop", "per_stage")}
                for f in protocol["leave_one_station_out"]
            ],
            "leave_one_crop_out": [
                {k: v for k, v in f.items() if k not in ("per_crop", "per_stage")}
                for f in protocol["leave_one_crop_out"]
            ],
            "target_distinct_values": distinct,
            "target_modal_share_pct": modal_share,
        },
        "old_target_reproduced_on_this_corpus": {
            "distinct_values": old_distinct,
            "modal_share_pct": old_modal_share,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
