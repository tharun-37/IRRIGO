# Column Selection Rationale

This document explains why specific columns/fields were chosen for Crop Age, Seasonal Watering, Yield Estimation, and Disease Prediction in the IRRIGO system.

## 1. Crop Age & Growth Stage

**Source:** `backend/ml/datasets/training_corpus.csv`, advisory logic, frontend types in `frontend/src/types.ts` (Advisory.season/stage)

### Columns Used
| Column | Purpose | Rationale |
| --- | --- | --- |
| `date` | Time reference | Base for computing age from sowing date. |
| `station` / `climate` | Agro-climatic context | Growing degree days (GDD) and phenology vary by climate zone. |
| `crop` | Crop type | Different crops have different thermal time requirements and stage durations. |
| `soil_type` | Soil context | Affects emergence and root development rates. |
| `season_progress` | Normalized progress | Computed fraction through season; used to position on stage track. |
| `kc` (crop coefficient) | Stage-dependent factor | Changes by growth stage (initial/development/mid/late) per FAO-56. |
| `et0_mm_day`, `etc_mm_day` | Evaporative demand/supply need | Tied to stage (peak mid-season). |
| `stage`, `stageLabel`, `stageProgress`, `ageDays`, `gdd` | Engine outputs | Derived from sowing date + thermal accumulation and crop calendar. |
| `stageWindows[]` (in SeasonInfo) | Season plan | Per-stage duration, mean Kc/ETc/rain, gross requirement - shows where crop is in lifecycle. |

**Why:** Crop water need is not constant. Using sowing date + GDD/phenology + crop-specific stage windows matches FAO-56 and explains "age" to farmers. The UI shows stage track with marker computed from day counts in windows.

## 2. Seasonal Watering (Schedule & Water Balance)

**Source:** `training_corpus.csv` (water balance + schedule)

### Core Water Balance Columns
| Column | Purpose | Rationale |
| --- | --- | --- |
| `taw_mm` (Total Available Water) | Field capacity - wilting point in root zone | Defines plant-available water capacity. |
| `raw_mm` (Readily Available Water) | Fraction of TAW before stress | Trigger point for irrigation (conservative). |
| `depletion_mm`, `depletion_fraction` | Current deficit | Drives "dry/good/wet" band and refill timing. |
| `root_depth_cm` | Effective rooting depth | Scales TAW over time as crop develops. |
| `field_capacity_pct`, `wilting_point_pct`, `bulk_density_g_cm3` | Soil hydraulic props | Compute TAW from soil type. |
| `water_stress` | Stress indicator | Flags when depletion exceeds RAW. |

### Climate/Supply
| Column | Purpose | Rationale |
| --- | --- | --- |
| `et0_mm_day` | Reference evapotranspiration | FAO-56 Penman-Monteith style baseline. |
| `kc` | Crop coefficient | Adjusts ET0 to crop ETc by stage. |
| `rainfall`, `rainfall_sum_3d/7d` | Effective rainfall | Reduces irrigation need. |
| `et0_mean_3d/7d` | Recent demand | Smoothed to avoid reacting to single hot/cold day. |
| `soil_moisture`, `soil_moisture_mean_3d/7d`, `soil_moisture_trend_3d` | Observed wetness | Calibrates balance with sensor reality. |

### Schedule/Decision
| Column | Purpose | Rationale |
| --- | --- | --- |
| `schedule[]` (date, dayOffset, applyMm, grossMm, stage) | Next 7 days plan | Shows when water is planned across week. |
| `water_required_mm`, `water_required_liters` | Depth/volume needed | Converts to pump minutes/liters for farmer. |
| `irrigation_action` | Action label | Drives "Water today/Watch/Hold" messaging. |

**Why:** Combines physics (FAO-56 water balance) with observations (sensors) and history (rolling windows).

## 3. Yield Estimation

**Sources:** `training_dataset_with_yield.csv`, `training_corpus.csv` (`yield_estimate_t_ha`)

### Features Used
| Column | Purpose | Rationale |
| --- | --- | --- |
| `Temperature` / `temperature` | Growth rate | Affects biomass accumulation and heat stress. |
| `Humidity` / `humidity` | Disease/VPD effects | High humidity favors disease; VPD affects transpiration. |
| `Moisture` / `soil_moisture` | Water availability | Water stress caps yield. |
| `Nitrogen`, `Phosphorus`, `Potassium` | Nutrition | NPK drive yield potential; deficiencies limit. |
| `PH` / `soil_ph` | Nutrient availability | Impacts uptake efficiency. |
| `Light_Intensity` / `light_intensity` | Photosynthetically active radiation | Drives biomass production. |
| `Yield_Rate` / `yield_estimate_t_ha` | Target | Model learns environment->yield. |
| `crop`, `crop_stress_index`, `season_progress` | Crop/phenology context | Different crops have different yield potentials. |
| `yieldPotentialTPerHa`, `yieldModelled` | Metadata | Console shows potential vs estimate; flags if model seen crop. |

**Farmer display logic:** For rough estimates, we compute `acres × 5 quintal/acre`. For modelled estimates, we convert `t/ha` to `quintal/acre` (1 t/ha ≈ 4.047 quintal/acre) and multiply by actual field acres to show total yield. The UI shows both total and per-acre.

**Why:** Yield depends on water, nutrients, climate, and crop. Combining these features captures main yield-limiting factors.

## 4. Disease Prediction

**Sources:** `plant_health_data.csv`, `training_dataset_with_yield.csv` (`Target_Disease`), `training_corpus.csv` (`target_disease`)

### Features Used
| Column | Purpose | Rationale |
| --- | --- | --- |
| `Ambient_Temperature` / `temperature` | Pathogen development | Many fungal diseases prefer specific temp ranges. |
| `Humidity` | Leaf wetness proxy | Critical for spore germination/infection. |
| `Soil_Moisture` / `soil_moisture` | Root stress | Stressed plants more susceptible. |
| `Soil_Temperature` | Soil-borne pathogen activity | Affects soil pathogens. |
| `Soil_pH` | Nutrient availability/stress | Can influence disease susceptibility. |
| `Nitrogen_Level`, `Phosphorus_Level`, `Potassium_Level` / NPK | Nutrition | Imbalanced nutrition affects resistance. |
| `Chlorophyll_Content` | Plant vigor/stress | Early stress indicator. |
| `Electrochemical_Signal` | Physiological stress | Direct plant response to stress. |
| `Light_Intensity` | Canopy microclimate | Affects drying/wetness duration. |
| `crop_stress_index` | Composite stress | Aggregates multiple stressors. |
| `diseaseRisk`, `diseaseConfidence`, `diseaseProbabilities` | Outputs | Model gives likely disease + confidence + class probabilities. |

**Why:** Disease is multi-factorial (weather + nutrition + stress). Using environmental and plant-level signals improves detection. UI shows likely diseases with confidence and actionable advice.

## Design Principles
- **FAO-56 alignment:** Water balance uses TAW/RAW/depletion/Kc/ET0
- **Sensor-aligned:** Features match live telemetry/schema
- **Leakage-resistant:** Prefer current state + history, not future/post-decision data
- **Interpretable:** Farmers see reasons, not just numbers
- **Robust:** Rolling windows smooth noise; fallbacks handle missing data
- **Multi-task:** Single corpus supports irrigation, water depth/volume, disease, yield, nutrients
