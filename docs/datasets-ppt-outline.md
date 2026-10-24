# IRRIGO — Crop-Health Model: Datasets & Columns
*Slide-ready outline. Slide breaks are marked with `---`; each block maps to one slide.*

---

## Slide 1 — Why this model exists

**Title: Crop-Health Photo Check — closing the inference gap**

- The farmer app's photo-check tab is a finished, quality-gated pipeline with an **empty model slot** (`deployedModel: false`, "none-deployed")
- Goal: a deployed image model that returns a disease verdict for the crops IRRIGO manages
- Statuses: `result` (model verdict) · `uncertain` (confidence < 0.55) · `unsupported_crop` (crop not covered)
- Honest-by-design: every verdict says *"verify in the field — not a diagnosis"*

---

## Slide 2 — Image datasets per crop

**Title: Open datasets for the five field crops**

| Crop | Dataset | Size | Classes | Source |
|---|---|---|---|---|
| Rice `Paddy-North` | RiceDiseases-DataSet | 3 folders | Bacterial leaf blight, blast, brown spot | github.com/aldrin233/RiceDiseases-DataSet |
| Rice (alt.) | Rice Leaf Disease Images | 5,932 img | Blight, blast, brown spot, tungro | kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image |
| Wheat | Wheat disease images (JIC) | 999 img (19k on request) | Yellow rust, brown rust, septoria, mildew, healthy | zenodo.org/records/7573133 |
| Cotton | Cotton Leaf Disease | 1,710 real-world img | Bacterial blight, curl virus, fusarium wilt, healthy | kaggle.com/datasets/seroshkarim/cotton-leaf-disease-dataset |
| Potato | PlantVillage subset | ~2,150 img | Early blight, late blight, healthy | kaggle.com/datasets/emmarex/plantdisease |
| Barley | Net blotch / leaf scald / healthy | 3,012 img | Net blotch, leaf scald, healthy | data.mendeley.com/datasets/8nzpnwkncp/1 |

*Speaker note: barley has no large GitHub set; expect fine-tuning + own field collection.*

---

## Slide 3 — The model's input columns (six scalars)

**Title: Servings features — `analyse_pixels`, one row per upload**

| # | Column | Meaning |
|---|---|---|
| 1 | `lumaMean` | mean brightness (0–255) |
| 2 | `lumaStd` | brightness spread |
| 3 | `sharpness` | Laplacian variance (blur gate) |
| 4 | `greenFraction` | fraction of leaf-green pixels |
| 5 | `darkFraction` | fraction of pixels under luma 60 |
| 6 | `yellowFraction` | fraction of yellow-reading pixels |

- Computed for **every** upload — the same functions gate quality and feed the model (train = serve, no drift)
- Baseline model: `RandomForestClassifier(300, class_weight=balanced)` over these six columns
- Artifact = `crop_health_model.joblib` + `model_meta.json` in `backend/models/crop_health/`

---

## Slide 4 — Image dataset column schema

**Title: How the image datasets are structured**

**Folder-per-class layout** (GitHub/Kaggle convention):

```
dataset/
├── <class label>/        ← folder name IS the class column value
│   ├── img_0001.jpg
│   └── ...
└── <class label>/...
```

- Model-side equivalent table: `filename · sha256 · width · height · class(crop, disease) · split`
- Roboflow-style variant already supported by `ml/scripts/leaf_cnn_train.py`: `_classes.csv` = `filename, <class_1>…<class_n>` one-hot columns
- Label rule: only images passing the four quality gates (resolution ≥ 240px, blur ≥ 200, exposure 35–245, content std ≥ 12) enter training

---

## Slide 5 — Tabular datasets already in the repo

**Title: `backend/ml/datasets/` — columns that already work today**

| CSV | Purpose | Key columns |
|---|---|---|
| `training_corpus.csv` | irrigation/disease/yield model corpus | `station`, `crop`, `soil_moisture`, `et0_mm_day`, `kc`, `rainfall`, `water_required_mm`, `target_disease`, `yield_estimate_t_ha`, `npk_demand_kg_ha`, `irrigation_action` (50 cols; full list in `manifest.json`) |
| `irrigation_prediction.csv` | irrigation-need classifier | `Soil_Type`, `Soil_Moisture`, `Crop_Type`, `Crop_Growth_Stage`, `Irrigation_Type`, `Water_Source`, `Previous_Irrigation_mm`, `Irrigation_Need` |
| `training_dataset_with_yield.csv` | yield + disease targets | `Temperature`, `Humidity`, `Moisture`, `Nitrogen`, `Phosphorus`, `Potassium`, `PH`, `Light_Integrity`, `Yield_Rate`, `Target_Disease`, `Water_Required_Liter_m2` |
| `plant_health_data.csv` | sensor-based plant health | `Plant_ID`, `Soil_Moisture`, `Ambient_Temperature`, `Soil_pH`, `N/P/K_Level`, `Chlorophyll_Content`, `Plant_Health_Status` |

*Every file is sha256-hashed in `backend/ml/crop_health/manifest.json` so training runs are reproducible.*

---

## Slide 6 — Split & validation rules (the honesty rules)

**Title: What makes a model deployable here**

- Split before tuning; three disjoint groups; group by **plant/farm/capture session** so near-duplicates never straddle splits
- A **held-out field-photo test set** that never touches architecture or threshold selection
- `calibrated: false` until temperature scaling (or similar) is documented on a held-out set — raw softmax is never called a probability
- Abstention floor **0.55** — below it the farmer sees `uncertain`, never a guessed label
- Farmers' verdicts always carry: *model reading, not a diagnosis → verify in field → expert referral*

---

## Slide 7 — Roadmap on one line

**Title: From gap to verdict**

1. Baseline six-scalar model per crop → pipeline returns real `result`/`uncertain` verdicts **(one command: `train_crop_health_baseline.py`)**
2. Merge larger per-crop datasets + collect own field photos
3. CNN upgrade (torch pipeline ready: `leaf_cnn_train.py`) behind the same artifact contract
4. Calibration + session-grouped field validation → honest `validationStatus` upgrade
5. Connect ESP32 soil sensors (`POST /api/telemetry`) so the same dashboards show measured, not derived, values
