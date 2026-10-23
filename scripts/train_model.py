"""Train the requirement estimator and run the full evaluation protocol.

Usage
  python scripts/train_model.py            # full protocol
  python scripts/train_model.py --quick    # 3 folds per leave-one-out, for a smoke run
  python scripts/train_model.py --skip-protocol
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irrigation.features import (  # noqa: E402
    FeatureSpec,
    encode_categoricals,
    leakage_report,
)
from irrigation.models import (  # noqa: E402
    NirModel,
    final_model,
    run_protocol,
    save_model,
)

CORPUS = ROOT / "data" / "crop_corpus.csv"
MODEL_DIR = ROOT / "models"
REPORT = ROOT / "reports"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float):
        return None if obj != obj else round(obj, 6)
    return obj


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-protocol", action="store_true")
    parser.add_argument("--no-importance", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    print("loading corpus ...", flush=True)
    data = pd.read_csv(CORPUS, low_memory=False)
    spec = FeatureSpec()
    features = encode_categoricals(spec.frame(data), spec)
    print(f"  {len(data):,} rows, {features.shape[1]} features, "
          f"{data['station'].nunique()} stations, {data['crop'].nunique()} crops")

    findings = leakage_report(features, data["nir_horizon_mm"])
    if findings:
        print("\nLEAKAGE / DEAD FEATURES")
        for item in findings:
            print(f"  {item}")
    else:
        print("  leakage audit: clean")

    report: dict = {"corpus": str(CORPUS), "n_rows": int(len(data))}

    if not args.skip_protocol:
        print("\n=== evaluation protocol ===", flush=True)
        report["protocol"] = run_protocol(data, spec, quick=args.quick)
        summary = report["protocol"]["leave_one_station_out_summary"]
        print(
            f"\n  leave-one-station-out: mean R2 {summary['mean_r2']:+.4f}"
            f"  min {summary['min_r2']:+.4f} ({summary['worst_fold']})"
            f"  negative folds {summary['negative_r2_folds']}/{summary['folds']}"
        )
        crop = report["protocol"]["leave_one_crop_out_summary"]
        print(
            f"  leave-one-crop-out:   mean R2 {crop['mean_r2']:+.4f}"
            f"  min {crop['min_r2']:+.4f} ({crop['worst_fold']})"
        )

    print("\n=== final deployable model ===", flush=True)
    model, fit_info = final_model(data, spec)
    print(f"  train MAE {fit_info['train_mae_mm']:.2f} mm"
          f" | held-out calibration MAE {fit_info['calibration_mae_mm']:.2f} mm")
    for coverage, interval in model.conformal.items():
        flag = "  [degenerate - see note]" if interval.degenerate else ""
        print(f"  conformal {coverage:<14} +/- {interval.quantile_mm:6.2f} mm"
              f"  n={interval.n_calibration:,}"
              f"  coverage {interval.realised_coverage:.1%}{flag}")
        if interval.degenerate_note:
            print(f"      note: {interval.degenerate_note}")
    report["final_model"] = fit_info

    if not args.no_importance:
        print("\n=== permutation importance (top 12) ===", flush=True)
        sample = data.sample(n=min(20_000, len(data)), random_state=3)
        importance = model.permutation_importance(
            encode_categoricals(spec.frame(sample), spec), sample["nir_horizon_mm"]
        )
        for name, score in list(importance.items())[:12]:
            print(f"  {name:<32} {score:+.4f}")
        dead = [name for name, score in importance.items() if abs(score) < 1e-4]
        if dead:
            print(f"  features with no measurable effect: {dead}")
        report["importance"] = importance

    model_path = save_model(model, MODEL_DIR / "nir_model.joblib")
    print(f"\n  saved model to {model_path}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "feature_names": list(model.feature_names),
        "target": model.target_name,
        "n_rows": int(len(data)),
        "created": pd.Timestamp.now().isoformat(),
        "corpus_sha_note": "regenerate with scripts/build_corpus.py",
        **{k: v for k, v in fit_info.items() if k != "conformal"},
    }
    (MODEL_DIR / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2), encoding="utf-8"
    )

    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "evaluation.json").write_text(
        json.dumps(_jsonable(report), indent=2), encoding="utf-8"
    )
    print(f"  wrote {REPORT / 'evaluation.json'}")
    print(f"  total {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
