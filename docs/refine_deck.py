"""
Text-only refinement of the presentation deck.

Every edit here changes words, never formatting. Each paragraph in the source
deck holds exactly one run, so replacing the run text preserves the typeface,
size, weight and colour that was set in PowerPoint.

Run with:
    .venv\\Scripts\\python docs\\refine_deck.py

A timestamped backup of the original is written before anything is saved, and
the script reports every shape it touched.
"""

from __future__ import annotations

import copy
import shutil
from datetime import datetime
from pathlib import Path

from pptx import Presentation
from pptx.util import Pt

DECK = Path(__file__).resolve().parent / "Irrigation_Controller_Deck.pptx"

# --------------------------------------------------------------------------
# Edits
#
# Keyed by slide number (1-based) and shape name. Values are the new paragraph
# texts, top to bottom.
# --------------------------------------------------------------------------

EDITS: dict[int, dict[str, list[str]]] = {
    2: {
        "Text 3": [
            "Irrigation is still decided by eye, causing wasted water (over-irrigation) or "
            "crop stress (under-irrigation). Soil moisture alone is reactive: it ignores "
            "incoming rain and how fast the crop loses water. The proposed Intelligent "
            "Irrigation Controller:",
        ],
        "Text 4": [
            "Predicts irrigation need from soil moisture, real weather data and "
            "evapotranspiration (ET) computed with the FAO-56 Penman-Monteith equation",
            "Automates the valve from an ESP32 edge node reading a 7-in-1 RS485 NPK probe, "
            "closing the loop locally with a fail-safe rule",
            "Gives the operator a dashboard with schedules, water-stress alerts and "
            "per-zone nutrient guidance",
            "Uses a two-stage gradient-boosted model: decide whether to irrigate, then how "
            "deep, with a cost-weighted operating point",
            "Secondary outputs: crop disease-risk, NPK fertiliser dose and yield estimates "
            "from the same reading",
        ],
    },
    3: {
        "Text 5": [
            "Most published systems are reactive: no forecast, no evapotranspiration term",
            "Noisy, intermittent soil readings amplify into large irrigation errors on deep "
            "root zones",
            "Fuzzy or rule-only schedulers need site-specific tuning and cannot transfer "
            "between districts",
            "No public dataset pairs NPK soil telemetry with confirmed crop disease, so "
            "disease models are hard to validate honestly",
            "Edge-only or cloud-only designs trade accuracy against latency and uptime",
        ],
        "Text 8": [
            "Soil moisture plus real satellite meteorology plus a validated FAO-56 ET0",
            "Two-stage gradient boosting: irrigation decision, then irrigation depth in mm",
            "Forward soil water balance simulation over real weather, plus probe-noise "
            "injection and causal temporal smoothing",
            "Automated valve control with a live operator dashboard",
            "Disease risk, NPK dose and yield as add-on insights from the same reading",
        ],
    },
    4: {
        "Text 4": [
            "Forecast irrigation requirement from real soil moisture, NASA POWER "
            "meteorology and FAO-56 evapotranspiration",
            "Automate valve control from an ESP32 edge node with a 7-in-1 RS485 NPK probe "
            "and a latching solenoid, fail-safe on loss of link",
            "Deliver an operator dashboard with zone schedules, water-stress alerts and "
            "nutrient guidance",
            "Report disease risk, fertiliser dose and yield as additional insights, with "
            "an explicit statement of what each figure does and does not mean",
        ],
    },
    5: {
        "Text 5": [
            "10 years of daily meteorology from NASA POWER (12 stations, 43,836 days) plus "
            "7-in-1 NPK soil telemetry",
        ],
        "Text 8": [
            "FAO-56 ET0, soil water deficit, wetness ratio, crop stress index, causal 3- "
            "and 7-day trend features",
        ],
        "Text 11": [
            "Forward soil water balance integrated over the real weather record; hysteresis "
            "irrigation rule; physically derived targets",
        ],
        "Text 14": [
            "Applied depth in mm, pump runtime, deficit and excess risk levels per zone",
        ],
        "Text 17": [
            "Probe measurement-noise injection, per-zone nutrient drawdown, rainfall gap "
            "handling",
        ],
        "Text 20": [
            "8 models: gradient-boosted classifiers and regressors; two-stage "
            "irrigate/depth design; cost-tuned decision threshold",
        ],
        "Text 23": [
            "Stratified grouped split on unseen field series, plus a fully held-out station; "
            "macro F1, MAE, R², permutation importance, latency",
        ],
        "Text 26": [
            "Depth and volume, valve runtime, soil-water trend, stress and nutrient alerts, "
            "disease risk, confidence",
        ],
    },
    6: {
        # 'Text 5' is a one-line bold 16 pt heading, so it stays a heading.
        "Text 4": [
            "Air temperature, relative humidity, wind speed",
            "Soil moisture, plus 3- and 7-day mean and trend",
            "Soil pH and electrical conductivity",
            "Nitrogen, phosphorus, potassium (mg/kg)",
            "Light intensity (lux) and rainfall",
            "Crop stress index, derived from the above",
            "Configured: crop, soil class, root depth, field capacity, field area; "
            "from weather: ET0 and Kc",
        ],
        "Text 5": [
            "Models: HistGradientBoosting",
        ],
        "Text 6": [
            "Stage 1 classifier: irrigate or hold",
            "Stage 2 regressor: depth in mm, on irrigation events",
            "max_iter 220/260, max_depth 8, lr 0.07-0.09",
            "min_samples_leaf 30-40, L2 0.6-1.0",
            "Threshold tuned 3:1 toward missed irrigation",
            "Disease F1 0.908, depth MAE 1.05 mm (R2 0.963), <0.01 ms",
        ],
    },
    7: {
        "Text 4": [
            "7-in-1 RS485 NPK probe + air temperature, humidity and lux",
        ],
        "Text 7": [
            "NASA POWER daily meteorology, 12 stations, 2015-2024",
        ],
        "Text 10": [
            "FAO-56 ET0, soil water deficit, wetness ratio, stress index, 3- and 7-day "
            "trend features",
        ],
        "Text 13": [
            "Forward soil water balance simulation over real weather",
        ],
        "Text 15": [
            "Two-stage gradient boosting, grouped and unseen-station evaluation",
        ],
        "Text 18": [
            "Irrigation decision + depth in mm + valve runtime",
        ],
        "Text 21": [
            "ESP32 -> relay -> latching solenoid valve, fail-safe on link loss",
        ],
        "Text 24": [
            "FastAPI + WebSocket -> React dashboard: schedules, alerts, what-if",
        ],
    },
    8: {
        "Text 3": [
            "[1] R. G. Allen, L. S. Pereira, D. Raes and M. Smith, \"Crop evapotranspiration: "
            "guidelines for computing crop water requirements,\" FAO Irrigation and Drainage "
            "Paper 56, Rome, 1998.  (Source of the ET0, soil water balance and Kc equations.)",
            "[2] NASA Prediction of Worldwide Energy Resources (POWER) Project, daily "
            "agroclimatology point API, 12 stations, 2015-2024. "
            "https://power.larc.nasa.gov/docs/services/api/temporal/daily",
            "[3] G. Saha et al., \"Smart IoT-driven precision agriculture: Land mapping, crop "
            "prediction, and irrigation system,\" PLOS ONE, vol. 20, no. 3, e0319268, Mar. 2025.",
            "[4] F. Z. Bassine et al., \"Recent applications of machine learning, remote sensing, "
            "and IoT approaches in yield prediction: a critical review,\" arXiv:2306.04566, Jun. 2023.",
            "[5] V. H. U. Eze et al., \"Integrating IoT sensors and machine learning for "
            "sustainable precision agroecology,\" Discover Agriculture, vol. 3, Art. 83, May 2025.",
        ],
    },
}

#: The exact text each edited shape is expected to hold before the edit.
#:
#: This exists because shape names in this deck are not sequential with visual
#: order: the workflow diagram interleaves arrow glyph shapes with its content
#: boxes, so 'Text 5', 'Text 11' and 'Text 16' are arrows rather than labels.
#: A first pass addressed the wrong names and overwrote three arrows with body
#: text. Asserting the original text makes that class of mistake impossible to
#: make silently, and aborts the whole run rather than saving a corrupted deck.
EXPECTED_ORIGINAL: dict[int, dict[str, str]] = {
    2: {
        "Text 3": "Irrigation is still decided by eye",
        "Text 4": "Predicts irrigation need from soil moisture",
    },
    3: {
        "Text 5": "Most systems are reactive",
        "Text 8": "Soil moisture + weather forecast + ET features",
    },
    4: {
        "Text 4": "Forecast precise irrigation requirements",
    },
    5: {
        "Text 5": "IoT sensors, weather API",
        "Text 8": "ET index, soil water deficit, GDD",
        "Text 11": "Moisture trends, crop stress factors",
        "Text 14": "Optimal schedule",
        "Text 17": "Intermittent readings",
        "Text 20": "StandardScaler",
        "Text 23": "Cross-validation",
        "Text 26": "Daily volume, schedule",
    },
    6: {
        "Text 4": "Temperature",
        "Text 5": "Random Forest Models",
        "Text 6": "Regressor: irrigation water volume",
    },
    7: {
        "Text 4": "Soil + climate sensors",
        "Text 7": "Weather forecast API",
        "Text 10": "Feature engineering (ET, deficit, GDD)",
        "Text 13": "Cleaning + augmentation",
        "Text 15": "Random Forest training + evaluation",
        "Text 18": "Irrigation decision (irrigate/hold + volume)",
        "Text 21": "ESP32 → Relay → Water Pump",
        "Text 24": "Cloud dashboard: schedule + alerts",
    },
    8: {
        "Text 3": "[1] G. Saha et al.",
    },
}

#: Captions and stray text boxes, matched by their current text.
TEXT_MATCH_EDITS: dict[int, dict[str, str]] = {
    2: {
        "Uses Random Forest: robust to noisy data and explainable":
            "Two-stage gradient-boosted models: decide whether to irrigate, then how deep",
    },
    7: {
        "Feedback loop: sensors re-measure after irrigation. Secondary branch off the model: "
        "crop health + yield insights.":
            "Feedback loop: the probe re-measures after the valve closes, and the next cycle "
            "starts from the new state. Secondary branch off the model: disease risk, NPK "
            "dose and yield.",
    },
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def set_paragraph_text(paragraph, text: str) -> None:
    """Replace a paragraph's text, keeping the formatting of its first run."""
    if paragraph.runs:
        paragraph.runs[0].text = text
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        run = paragraph.add_run()
        run.text = text
        run.font.size = Pt(13)


def apply_lines(text_frame, lines: list[str], budget: int) -> int:
    """
    Rewrite a text frame with `lines`, never exceeding its paragraph budget.

    A text frame in this deck is a fixed-position box sized for the text it
    originally held. Adding a paragraph past that budget does not move anything
    else: the text simply overflows the box and renders outside the slide, over
    the title and the neighbouring column. An earlier version of this script did
    exactly that by putting eight lines into a one-line bold header, so frames
    are now never grown and the budget is enforced by the caller.

    Surplus paragraphs are blanked rather than deleted, which keeps the box
    geometry stable.
    """
    if len(lines) > budget:
        raise ValueError(
            f"text needs {len(lines)} lines but the shape holds {budget}. "
            f"Shorten the content: {lines[0][:60]!r}"
        )

    for index, line in enumerate(lines):
        set_paragraph_text(text_frame.paragraphs[index], line)

    for index in range(len(lines), len(text_frame.paragraphs)):
        set_paragraph_text(text_frame.paragraphs[index], "")

    # Wrap rather than let a long line run off the right edge, and never let
    # PowerPoint resize the box, which would shift the layout.
    text_frame.word_wrap = True
    return len(lines)


def main() -> int:
    if not DECK.exists():
        print(f"deck not found: {DECK}")
        return 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = DECK.with_name(f"{DECK.stem}.backup-{stamp}{DECK.suffix}")
    shutil.copy2(DECK, backup)
    print(f"backup written: {backup.name}\n")

    # Always edit from the pristine original, so re-running is idempotent rather
    # than compounding earlier edits.
    pristine = backup

    # Paragraph counts of every shape in the untouched original. These are the
    # budgets the replacement text must fit inside.
    reference = Presentation(pristine)
    budgets: dict[int, dict[str, int]] = {}
    for slide_number in EDITS:
        budgets[slide_number] = {
            shape.name: len(shape.text_frame.paragraphs)
            for shape in reference.slides[slide_number - 1].shapes
            if shape.has_text_frame
        }

    presentation = Presentation(pristine)
    touched = 0
    problems: list[str] = []

    # Validate every target before touching anything, so a bad name aborts the
    # run with the file untouched rather than half-editing a saved deck.
    for slide_number, expectations in EXPECTED_ORIGINAL.items():
        slide = presentation.slides[slide_number - 1]
        by_name = {shape.name: shape for shape in slide.shapes if shape.has_text_frame}
        for name, expected in expectations.items():
            shape = by_name.get(name)
            if shape is None:
                problems.append(f"slide {slide_number}: no shape named {name!r}")
                continue
            actual = shape.text_frame.text.strip()
            if not actual.startswith(expected):
                problems.append(
                    f"slide {slide_number} {name}: expected text starting "
                    f"{expected!r}, found {actual[:60]!r}"
                )

    # Validate every line budget before touching anything, for the same reason.
    for slide_number, shape_edits in EDITS.items():
        for name, lines in shape_edits.items():
            budget = budgets.get(slide_number, {}).get(name)
            if budget is None:
                problems.append(f"slide {slide_number} {name}: shape not found")
            elif len(lines) > budget:
                problems.append(
                    f"slide {slide_number} {name}: needs {len(lines)} lines, "
                    f"shape holds {budget}"
                )

    if problems:
        print("ABORTED: the deck does not match the expected original.")
        print("Restore a clean copy and re-run.\n")
        for problem in problems:
            print(f"  - {problem}")
        return 2

    for slide_number, shape_edits in EDITS.items():
        slide = presentation.slides[slide_number - 1]
        by_name = {shape.name: shape for shape in slide.shapes if shape.has_text_frame}
        for name, lines in shape_edits.items():
            shape = by_name[name]
            budget = budgets[slide_number][name]
            apply_lines(shape.text_frame, lines, budget)
            touched += 1
            print(f"  [ok]   slide {slide_number} {name}: {len(lines)}/{budget} lines")

    for slide_number, replacements in TEXT_MATCH_EDITS.items():
        slide = presentation.slides[slide_number - 1]
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for paragraph in shape.text_frame.paragraphs:
                if paragraph.text in replacements:
                    set_paragraph_text(paragraph, replacements[paragraph.text])
                    touched += 1
                    print(f"  [ok]   slide {slide_number} caption updated")

    presentation.save(DECK)
    print(f"\n{touched} text blocks updated")
    print(f"saved: {DECK.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
