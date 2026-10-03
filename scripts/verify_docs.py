"""Check that every figure quoted in the documentation matches the reports.

Documentation drifts silently. A README that says 90% coverage when the report
says 93% is worse than no README, because it is trusted. This asserts the
numbers against the artefacts that produced them.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from irrigation.models import series_key  # noqa: E402

TOL = 0.005

failures: list[str] = []
checks = 0


def check(label: str, actual: float, quoted: float, tol: float = TOL) -> None:
    global checks
    checks += 1
    if abs(float(actual) - float(quoted)) > tol:
        failures.append(
            f"{label}: report says {actual:.4f}, docs quote {quoted:.4f}"
        )


ev = json.loads((ROOT / "reports" / "evaluation.json").read_text(encoding="utf-8"))
ext = json.loads((ROOT / "reports" / "external_validation.json").read_text(encoding="utf-8"))
soil = json.loads((ROOT / "reports" / "soil_validation.json").read_text(encoding="utf-8"))
cmp_ = json.loads((ROOT / "reports" / "v1_vs_v2.json").read_text(encoding="utf-8"))

docs = (ROOT / "README.md").read_text(encoding="utf-8") + \
       (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")

proto = ev["protocol"]
grouped = proto["grouped"]
loso = proto["leave_one_station_out_summary"]
loco = proto["leave_one_crop_out_summary"]

# --- numbers quoted in README and ARCHITECTURE ---------------------------
check("grouped MAE", grouped["mae_mm"], 2.97)
check("grouped R2", grouped["r2"], 0.9571)
check("LOSO mean R2", loso["mean_r2"], 0.7083)
check("LOSO min R2", loso["worst_fold_r2"], -0.2442)
check("LOCO mean R2", loco["mean_r2"], 0.8205)
check("LOCO min R2", loco["worst_fold_r2"], 0.5891)
check("V1 OOD R2", cmp_["v1_headline"]["ood_r2_unseen_station"], 0.0816)

skill = grouped.get("skill_vs_physics", {})
check("skill vs physics pct", skill["skill_vs_physics_pct"], 72.8, tol=0.5)
check("model MAE on wet rows", skill["model_mae_mm"], 4.34, tol=0.5)
check("physics MAE on wet rows", skill["physics_mae_mm"], 15.97, tol=0.5)

trig = grouped.get("trigger", {})
check("trigger recall", trig["recall"], 0.996, tol=0.001)
check("trigger missed pct", trig["missed_pct"], 0.42, tol=0.02)
check("trigger false alarm pct", trig["false_alarm_pct"], 27.61, tol=0.02)

# The docs quote the deployable model's conformal widths, not the grouped fold's
# re-derived ones; those are two different intervals and conflating them would
# hide a real discrepancy.
conformal = ev["final_model"]["conformal"]
check("conformal 90 width", conformal["0.9"]["quantile_mm"], 9.11, tol=0.02)
check("conformal 80 width", conformal["0.8"]["quantile_mm"], 6.53, tol=0.02)
check("conformal 80 realised", conformal["0.8"]["realised_coverage"], 0.800, tol=0.001)
check("conformal 90 realised", conformal["0.9"]["realised_coverage"], 0.900, tol=0.001)

check("distinct targets", cmp_["v2"]["target_distinct_values"], 40402, tol=0)
check("modal share pct", cmp_["v2"]["target_modal_share_pct"], 0.01, tol=0.005)
check("old target modal share", cmp_["old_target_reproduced_on_this_corpus"]["modal_share_pct"],
      29.54, tol=0.02)

check("ET0 mean r", ext["summary"]["et0"]["mean_pearson_r"], 0.911, tol=0.001)
check("ET0 mean ratio", ext["summary"]["et0"]["mean_ratio"], 0.998, tol=0.001)
check("ET0 mean bias", ext["summary"]["et0"]["mean_bias"], -0.014, tol=0.002)
check("rain mean r", ext["summary"]["rain"]["mean_pearson_r"], 0.668, tol=0.001)
check("rain mean ratio", ext["summary"]["rain"]["mean_ratio"], 0.980, tol=0.001)

# --- soil validation ---------------------------------------------------
sv = soil["simulated_vs_observed"]
check("soil pooled r", sv["pooled_r"], 0.359, tol=0.001)
check("soil sim mean", sv["simulated_mean"], 0.2911, tol=0.0001)
check("soil obs mean", sv["observed_root_zone_mean"], 0.2837, tol=0.0001)
shape = sv["within_series"]
check("soil within-series mean r", shape["mean_within_series_r"], 0.352, tol=0.001)
check("soil positive share pct", shape["share_positive"] * 100.0, 90.5, tol=0.05)
check("soil depth monotonic", 1.0 if soil["depth_ordering"]["monotonic_with_depth"] else 0.0,
      1.0, tol=0)

# --- corpus size and pairing quoted in prose --------------------------
corpus_rows = ev["n_rows"]
paired = soil["n_paired_rows"]
for quoted in re.findall(r"([\d,]{5,}) rows", docs):
    checks += 1
    value = int(quoted.replace(",", ""))
    if value not in (corpus_rows, paired):
        failures.append(f"docs quote '{quoted} rows', corpus has {corpus_rows}, "
                        f"soil report paired {paired}")

# --- the two different "series" counts, quoted in prose ----------------
# The corpus has one series per field-season; the soil report scores one series
# per field-season-soil-depth combination, so the two numbers differ and both
# appear in the prose. Conflating them would make the check meaningless.
soil_series = shape["series_scored"]
_corpus = pd.read_csv(ROOT / "data" / "crop_corpus.csv",
                      usecols=["station", "crop", "soil_type", "irrigation_method",
                               "date", "days_since_sowing"])
corpus_series = int(pd.Series(series_key(_corpus)).nunique())
for quoted in re.findall(r"(\d{1,3}(?:,\d{3})*) series", docs):
    checks += 1
    value = int(quoted.replace(",", ""))
    if value not in (soil_series, corpus_series):
        failures.append(
            f"docs quote '{quoted} series'; corpus has {corpus_series}, "
            f"soil report scored {soil_series}"
        )

# --- test count quoted in README --------------------------------------
# Read from pytest itself rather than hard-coded, so the check cannot rot.
n_tests = int(
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--collect-only"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout.strip().split("tests collected")[0].strip().split()[-1]
)
for quoted in re.findall(r"(\d+) pytest", docs) + re.findall(r"# (\d+) tests", docs):
    checks += 1
    if int(quoted) != n_tests:
        failures.append(f"docs quote '{quoted} pytest cases', suite has {n_tests}")

print(f"checked {checks} figures quoted in README.md and ARCHITECTURE.md")
if failures:
    print("\nMISMATCHES:")
    for f in failures:
        print(f"  {f}")
    raise SystemExit(1)
print("all documentation figures match the reports")
