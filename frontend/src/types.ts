/**
 * Domain types shared by the API client and the dashboard.
 *
 * These mirror the Pydantic schemas in `backend/app/schemas`. If a field here
 * is wrong the dashboard will render `undefined` rather than fail loudly, so
 * this file is kept deliberately explicit rather than inferred.
 */

export type LinkState = 'online' | 'offline' | 'degraded'
export type MoistureBand = 'dry' | 'optimal' | 'wet'
export type Severity = 'info' | 'warning' | 'critical'
export type RiskLevel = 'healthy' | 'watch' | 'risk'

export interface Device {
  id: string
  name: string
  zone: string
  crop: string
  soilType: string
  fieldAreaM2: number
  linkState: LinkState
  firmware: string
  lastSeen: string
  batteryVolts: number | null
  signalDbm: number | null
  uptimeSeconds: number
}

export interface Telemetry {
  id: number
  deviceId: string
  zone: string
  recordedAt: string
  temperature: number
  humidity: number
  soilMoisture: number
  soilPh: number
  ec: number
  nitrogen: number
  phosphorus: number
  potassium: number
  lightIntensity: number
  batteryVolts: number | null
  rssiDbm: number | null
  /**
   * The node's own valve state, when its firmware reports one. Null for rows
   * that never said — absent is meaningful, not false.
   */
  valveOpen?: boolean | null
}

export interface Recommendation {
  deviceId: string
  zone: string
  recordedAt: string
  shouldIrrigate: boolean
  depthMm: number
  volumeLitres: number
  durationSeconds: number
  action: 'NO_IRRIGATION' | 'LIGHT_IRRIGATION' | 'MODERATE_IRRIGATION' | 'HEAVY_IRRIGATION'
  diseaseRisk: string
  diseaseConfidence: number
  diseaseProbabilities: Record<string, number>
  yieldTPerHa: number
  nutrientDemandKgHa: { nitrogen: number; phosphorus: number; potassium: number }
  et0MmDay: number
  kc: number
  depletionFraction: number
  cropStressIndex: number
  moistureBand: MoistureBand
  riskLevel: RiskLevel
  confidence: number
  explanation: string[]
}

export interface ZoneSummary {
  zone: string
  crop: string
  deviceId: string
  linkState: LinkState
  soilMoisture: number | null
  moistureBand: MoistureBand
  soilPh: number | null
  ec: number | null
  nitrogen: number | null
  phosphorus: number | null
  potassium: number | null
  temperature: number | null
  humidity: number | null
  cropStressIndex: number | null
  lastSeen: string | null
  valveOpen: boolean
  valveRuntimeSeconds: number | null
  nextIrrigationAt: string | null
}

export interface Alert {
  id: number
  deviceId: string
  zone: string
  raisedAt: string
  kind: string
  severity: Severity
  message: string
  acknowledged: boolean
  resolvedAt: string | null
}

export interface EventRecord {
  id: number
  deviceId: string
  zone: string
  kind: 'command' | 'telemetry' | 'alert' | 'system'
  detail: string
  createdAt: string
  source: string
}

export interface WaterSeriesPoint {
  label: string
  appliedMm: number
  et0MmDay: number
  depletionFraction: number
}

export interface AnalyticsOverview {
  window: string
  waterAppliedMm: number
  waterAppliedPctChange: number
  estimatedLitresSaved: number
  irrigationEvents: number
  runtimeSeconds: number
  averageInferenceMs: number
  decisionsCorrectPct: number
  energyKwh: number
  series: WaterSeriesPoint[]
}

export interface ModelMetrics {
  disease: { macroF1: number; accuracy: number; perClass: Record<string, { f1: number; support: number }> }
  irrigateDecision: { f1: number; recall: number; precision: number; threshold: number }
  waterDepth: { mae: number; r2: number }
  yield: { mae: number; r2: number }
  nutrient: { nitrogen: number; phosphorus: number; potassium: number }
  latencyMs: Record<string, number>
}

/** Ack returned by POST /api/telemetry. */
export interface TelemetryAck {
  accepted: boolean
  readingId: number
  decisionId: number
  deviceId: string
  zone: string
  shouldIrrigate: boolean
  depthMm: number
  durationSeconds: number
  et0MmDay: number
  alertsRaised: number
  inferenceMs: number
  /**
   * A calibrated probability, not an arbitrary score. This is what makes
   * `irrigateThreshold` interpretable as an event rate.
   */
  irrigateProbability: number
  /** The operating point actually used for this field. */
  irrigateThreshold: number
  /** 'per_group' when tuned on this crop and soil, else the global fallback. */
  irrigateThresholdSource: 'per_group' | 'global_fallback' | 'unknown'
  probabilityIsCalibrated: boolean
  /** Conformal half-width on the depth estimate, in mm. */
  depthUncertaintyMm: number
  explanation: string[]
}

/** One feature's drift result. `psi` is null when there are too few readings. */
export interface DriftFeature {
  psi: number | null
  samples: number
  band?: 'negligible' | 'investigate' | 'significant' | 'sensor_frozen'
  note?: string
  reference_mean?: number
  reference_p50?: number
  live_mean?: number | null
}

export interface DriftReport {
  available: boolean
  reason?: string
  window_hours?: number
  readings_considered: number
  features: Record<string, DriftFeature>
  worst_feature?: string | null
  worst_psi?: number | null
  frozen_sensors: string[]
  interpretation?: string
}

export type AdvisoryStatus = 'irrigate' | 'schedule' | 'monitor' | 'hold'

export interface ReasoningItem {
  label: string
  value: string
  detail: string
  tone: 'healthy' | 'warn' | 'neutral'
}

export interface ScheduleDay {
  date: string
  dayOffset: number
  applyMm: number
  grossMm: number
  /**
   * The crop's requirement for this day, net of the rain the plan's horizon
   * expected — the input the optimiser scheduled against. Present even when
   * no pump is scheduled, so a correctly-wet week is a week of requirements,
   * not an empty chart.
   */
  needMm?: number
  stage: string
}

/**
 * The two learned models behind an advisory, and how they line up.
 *
 * V1's controller makes the operator-facing decision; V2's requirement estimator
 * predicts the coming fortnight's `nir_horizon_mm` with a conformal interval.
 * The fusion states whether they agree, and escalates when the decision model
 * misses a deficit the requirement model sees.
 */
export type ModelAgreement =
  | 'aligned'
  | 'agree_hold'
  | 'v1_conservative'
  | 'v2_only'
  | 'not_applicable'
  | 'model_unavailable'

export interface ModelV1Block {
  shouldIrrigate: boolean
  depthMm: number | null
  probability: number | null
  threshold: number | null
  confidence: number | null
  action: string | null
}

export interface ModelV2Block {
  target: string
  horizonDays: number
  physicsRequirementMm: number
  physicsSingleApplicationMm: number
  modelPresent: boolean
  /** False when the field is past harvest or otherwise outside the training contract. */
  applicable: boolean
  reason: string | null
  predictionMm: number | null
  intervalMm: [number, number] | null
  coverage: number | null
  residualVsPhysicsMm?: number
  error?: string
}

export interface ModelFusion {
  agreement: ModelAgreement
  finalStatus: AdvisoryStatus
  finalDepthMm: number
  v1DepthMm: number
  v2RequirementMm: number | null
  note: string
}

/** One growth stage the engine expects to run, and what it costs in water. */
export interface StageWindow {
  stage: string
  startDate: string
  endDate: string
  days: number
  meanKc: number
  meanEtcMmDay: number
  meanRainMmDay: number
  grossRequirementMm: number
  /**
   * This stage's share of the season in thermal time, from the crop
   * catalogue's gdd_stage_ends — a function of the crop alone, so every
   * field of one crop renders the same growth track. Null for fields whose
   * crop is outside the catalogue; the client falls back to the field's
   * own replay then.
   */
  profileShare?: number | null
  /** The catalogue's own mean Kc over this stage, companion to profileShare. */
  profileKc?: number | null
}

export interface SeasonInfo {
  stageWindows: StageWindow[]
  /**
   * The crop's peak-water stage span, resolved on the crop catalogue's own
   * Kc curve rather than on this field's windows — every field of one crop
   * gets the same span, whatever its station or sowing date. Absent when the
   * crop is outside the catalogue.
   */
  peakStretch?: { fromStage: string; toStage: string }
  daysInSeason: number
  seasonGrossMm: number
  seasonRainMm: number
}

export interface AdvisoryModels {
  v1: ModelV1Block
  v2: ModelV2Block
  fusion: ModelFusion
}

/**
 * The water advisor's per-field answer: the decision plus the reasoning and the
 * live sensor state it was drawn from. Everything is computed server-side from
 * one engine plan, so the headline and the numbers cannot disagree.
 */
export interface Advisory {
  fieldId: string
  crop: string
  station: string
  /** The station's human name and coordinates, for the farmer view's map. */
  stationName?: string | null
  stationLat?: number | null
  stationLon?: number | null
  soilType: string
  method: string | null
  sowingDate: string
  areaM2: number
  /** The field's own GPS pin, when the operator placed one at registration. */
  latitude?: number | null
  longitude?: number | null
  ageDays: number
  stage: string
  stageLabel: string
  stageProgress: number
  gdd: number
  status: AdvisoryStatus
  priority: number
  headline: string
  summary: string
  recommendation: {
    shouldIrrigate: boolean
    depthMm: number
    /** Depth after fusing the two models; present when the pipeline served it. */
    finalDepthMm?: number
    volumeLitres: number
    durationSeconds: number
    action: string | null
    confidence: number
    probability: number
    threshold: number
  }
  /**
   * The pump rate (litres/min) every runtime on this card was priced at:
   * seconds per millimetre = areaM2 / pumpRateLpm, the same volume-over-rate
   * arithmetic the decision block uses. Set so a held-today field can still
   * price its week's pump bars with the decision's own maths.
   */
  pumpRateLpm?: number
  /**
   * The field's freshest named leaf-photo read, when one is fresh enough to
   * still mean anything. The fertiliser chip acts on `nutrition` — hold a
   * blight's nitrogen, check the potash a leaf spot thrives on — and quotes
   * the photo that earned it. Absent (or null) means no photo is constraining
   * the plan today.
   */
  photoDisease?: {
    classLabel: string
    rawScore: number | null
    /** When the photo was analysed (ISO timestamp), the chip's provenance. */
    photoOn: string
    nutrition: {
      nutrient: 'nitrogen' | 'phosphorus' | 'potassium'
      action: 'withhold' | 'reduce' | 'check' | 'balance' | 'steady'
      note: string
    } | null
  } | null
  water: {
    et0MmDay: number
    etcMmDay: number
    kc: number
    rainfallMmDay: number
    tawMm: number
    rawMm: number
    depletionMm: number
    depletionFraction: number
    rootDepthCm: number
    daysUntilStress: number | null
  }
  sensors: Telemetry
  /**
   * The field's attached hardware node, when one posts real telemetry for
   * this zone. `isLive` is true only while the node's newest reading is
   * recent; the advisory's `sensors` then carry the node's own measurements
   * rather than the plan derivation. Present-but-stale records the last
   * seen time without claiming freshness.
   */
  live?: {
    deviceId: string
    source: string
    isLive: boolean
    lastSeenAt?: string
    ageSeconds?: number
    /** The node's own valve state, when its firmware reports one. */
    valveOpen?: boolean
  }
  risk: {
    diseaseRisk: string | null
    diseaseConfidence: number
    /**
     * The full disease class distribution, highest probability first.
     *
     * A single label hides the decision: 62% rust against 35% blight is a
     * spray, the same 62% against 2% blight is not. Empty when the bundle
     * carries no distribution, which is a normal state, not an error.
     */
    diseaseProbabilities: Record<string, number>
    riskLevel: RiskLevel | null
    cropStressIndex: number
    moistureBand: MoistureBand | null
    yieldTPerHa: number
    /** The crop's potential yield from the engine's catalogue, t/ha. */
    yieldPotentialTPerHa: number
    /** The crop's FAO-33 yield response factor, from the same catalogue. */
    yieldKy?: number
    /**
     * Whether the yield model was trained on this crop.
     *
     * The crop reaches the yield model as a one-hot encoding, so a crop the
     * model never saw still gets an answer — but a crop-agnostic one. The
     * dashboard uses this to avoid measuring a generic estimate against that
     * crop's ceiling, which would report an opportunity that does not exist.
     */
    yieldModelled: boolean
    /** The NPK dose the engine asks for, in kg per hectare. */
    nutrientDemandKgHa: {
      nitrogen: number
      phosphorus: number
      potassium: number
    }
  }
  reasoning: ReasoningItem[]
  schedule: ScheduleDay[]
  /** Stage-by-stage season plan; absent on the plain compat service. */
  season?: SeasonInfo
  /** Both learned models and their fusion; absent on the plain compat service. */
  models?: AdvisoryModels
}

/**
 * Live current conditions for a station, when a live feed is configured.
 *
 * The endpoint never fails: it answers `available: false` with a reason when
 * there is nothing to show, so the sky card can fall back to the plan's own
 * readings instead of reporting an error.
 */
export interface LiveWeather {
  available: boolean
  station: string
  source?: string
  reason?: string
  temperatureC?: number
  humidityPct?: number
  windMs?: number
  rainfallMm?: number
  condition?: string
  observedAt?: string
}

export interface AdvisorFeed {
  generatedAt: string
  fleet: {
    fields: number
    irrigate: number
    monitor: number
    hold: number
    recommendedMm: number
    recommendedLitres: number
  }
  /** Health of the fusion pipeline (V1 bundle + V2 requirement model). */
  pipeline?: {
    modelPresent: boolean
    target: string | null
    coverage: number | null
    features: number
    intervalWidthMm: number | null
    error: string | null
  }
  fields: Advisory[]
}

/* -------------------------------------------------------------------------- */
/* Field registration                                                           */
/* -------------------------------------------------------------------------- */

/**
 * One crop from the engine's catalogue.
 *
 * These are the numbers that make a cotton field need three times the water of a
 * rice field on the same day. They are on the wire rather than buried in the
 * engine so the interface can show *why* two fields with identical sensors get
 * different recommendations - a selector that only collects a name gives an
 * operator nothing to decide on.
 */
export interface CropOption {
  name: string
  kcInitial: number
  kcMid: number
  kcEnd: number
  /** Maximum rooting depth, metres, reached at mid-season. */
  rootDepthMaxM: number
  /** Fraction of available water the crop will draw down before it is stressed. */
  depletionFractionP: number
  optimalPh: [number, number]
  /** Species base temperature for thermal-time accumulation, Celsius. */
  gddBaseTempC: number
  /** GDD at the end of each FAO-56 stage; the last is the season total. */
  gddStageEnds: number[]
  seasonGdd: number
  /** Calendar length of each FAO-56 stage in days; the last is the season. */
  lengthStageDays: number[]
  seasonDays: number
  /** FAO-33 yield response to water, and the season's potential yield. */
  ky: number
  yieldPotentialTHa: number
  /** Named varieties: a short-season hybrid is a different water plan. */
  variants: string[]
  /** Every method that suits this crop, most efficient first. */
  suitableMethods: string[]
  /** Methods that work but change what the recommendation means, with the reason. */
  warnings: Record<string, string>
}

export interface CropCatalogue {
  crops: CropOption[]
}

/** Legal values for every reference field on a sowing. */
export interface RegistryOptions {
  stations: string[]
  crops: string[]
  soils: string[]
  methods: string[]
}

/**
 * A field registration request.
 *
 * Mirrors `FieldCreate` in `backend/app/schemas.py`. Only `fieldId`, `station`,
 * `crop` and `sowingDate` are required; the rest default server-side, which is
 * why the form can be submitted with four fields filled in.
 */
export interface FieldCreate {
  fieldId: string
  station: string
  crop: string
  sowingDate: string
  soilType?: string
  methodName?: string
  fieldAreaM2?: number
  mulched?: boolean
  nitrogenRegime?: number
  notes?: string
  /** Named cultivar. A short-duration hybrid is a different water plan. */
  cropVariant?: string
  /** Depth of a hardpan or gravel, metres. Caps what the roots can reach. */
  restrictingDepthM?: number
  /** Fraction of the crop considered emerged; below 1 it transpires less. */
  emergenceFraction?: number
  /**
   * Optional GPS pin of the field's own ground, in degrees. Given together or
   * not at all; when present the map draws the field here instead of near its
   * weather station.
   */
  latitude?: number
  longitude?: number
}

/** A stored sowing, as `POST /api/fields` and `GET /api/fields/{id}` return it. */
export interface FieldRecord {
  fieldId: string
  station: string
  crop: string
  cropVariant: string | null
  sowingDate: string
  soilType: string
  methodName: string
  fieldAreaM2: number
  mulched: boolean
  nitrogenRegime: number
  restrictingDepthM: number | null
  emergenceFraction: number | null
  daysSinceSowing: number
  seasonDay: number
  notes: string
  /**
   * The plan the engine produced for the new field, attached to the registration
   * response so the form can show a result without a second round trip.
   */
  plan?: FieldPlan
  planError?: string
}

export interface ScheduleDayPlan {
  offset: number
  applyMm: number
  grossApplyMm: number
  depletionBeforeMm: number
  depletionAfterMm: number
  stressBefore: boolean
  stressAfter: boolean
  reason: string
}

export interface FieldPlan {
  fieldId: string
  station: string
  crop: string
  asOf: string
  sowingDate: string
  daysSinceSowing: number
  stage: string
  stageProgress: number
  rootDepthCm: number
  tawMm: number
  rawMm: number
  depletionMm: number
  depletionFraction: number
  daysUntilStress: number | null
  kc: number
  et0MmDay: number
  etcMmDay: number
  /** Engine dataclass, serialised verbatim; snake_case by design. */
  requirement: Record<string, number | string | boolean>
  schedule: {
    events: ScheduleDayPlan[]
    totalNetMm: number
    totalGrossMm: number
    totalEvents: number
    firstEventDay: number | null
    unmetRequirementMm: number
    stressDays: number
    percolationMm: number
    infeasible: boolean
    requirementInsufficient: boolean
  } | null
  dataSource: string
}

export interface SystemHealth {
  status: 'ok' | 'degraded' | 'down'
  version: string
  uptimeSeconds: number
  modelsLoaded: boolean
  deviceCount: number
  onlineDevices: number
  readingsToday: number
  openAlerts: number
  /** Measured cost of one whole decision, feature assembly included. */
  inferenceLatencyMs: number
  calibration: { expected_calibration_error: number; recalibration_slope: number } | null
  /** Conformal half-width on the served depth estimate, in mm. */
  depthUncertaintyMm: number | null
  /** How many crop-and-soil groups have their own tuned operating point. */
  irrigateThresholdGroups: number | null
  modelMetrics: ModelMetrics | null
  waterSource: {
    available: boolean
    records: number
    stations: number
    span: { from: string; to: string }
    et0MeanMmDay: number
    et0MaxMmDay: number
    lastSync: string | null
    maxStationDistanceKm: number
  } | string
  lastWeatherSync: string | null
  provenance: Record<string, string | number | boolean>
}

// ---------------------------------------------------------------------------
// Crop health assistant
// ---------------------------------------------------------------------------

/** The blueprint's status enum; a detail string always travels with it. */
export type CropHealthResultStatus =
  | 'result'
  | 'uncertain'
  | 'unsupported_crop'
  | 'poor_image'
  | 'invalid_input'
  | 'processing_error'

export interface CropHealthGate {
  check: string
  passed: boolean
  detail: string
}

export interface CropHealthPrediction {
  task: string | null
  classLabel: string | null
  rawScore: number | null
  calibrated: boolean
  supportedClasses: string[]
}

export interface CropHealthAdviceOut {
  adviceCode: string
  adviceText: string
  textVersion: string
  followUpQuestion: string | null
  expertReferral: boolean
}

/**
 * General-practice spray/nutrition guidance, attached only to a named
 * `result` verdict. Never a dose, never a brand — the officer referral is
 * still the rule.
 */
export interface CropHealthTreatmentOut {
  heading: string
  text: string
  disclaimer: string
  textVersion: string
}

export interface CropHealthQuality {
  width?: number
  height?: number
  lumaMean?: number
  lumaStd?: number
  sharpness?: number
  greenFraction?: number
  darkFraction?: number
  yellowFraction?: number
}

export interface CropHealthAnalysis {
  analysisId: string
  createdAt: string
  crop: string
  plantPart: string | null
  growthStage: string | null
  fieldId: string | null
  zoneId: string | null
  resultStatus: CropHealthResultStatus
  statusReason: string
  prediction: CropHealthPrediction
  advice: CropHealthAdviceOut
  treatment?: CropHealthTreatmentOut | null
  quality?: CropHealthQuality
  gates: CropHealthGate[]
  modelVersion: string
  preprocessingVersion: string
  validationStatus: string
  imageReference: string | null
  consentGiven: boolean
  farmerFeedback?: string | null
}

export interface CropHealthList {
  analyses: CropHealthAnalysis[]
  count: number
}

export interface CropHealthStatus {
  serviceAvailable: boolean
  deployedModel: boolean
  modelVersion: string
  preprocessingVersion: string
  validationStatus: string
  supportedClasses: string[]
  supportedCrops: string[]
  limitations: string[]
  statusLabels: Record<string, string>
}

export interface CropHealthManifestTabular {
  file: string
  bytes: number
  sha256: string
  status: string
  rows?: number
  columns?: string[]
}

export interface CropHealthManifest {
  generatedAt?: string
  generator?: string
  tabularDatasets?: CropHealthManifestTabular[]
  imageCollections?: Array<{
    collection: string
    purpose: string
    status: string
    needs?: string
  }>
  imageModelStatus?: { deployed: boolean; note?: string }
}

// ---------------------------------------------------------------------------
// Fertilizer recommendation layer
// ---------------------------------------------------------------------------

export type FertilizerAreaUnit = 'hectare' | 'acre' | 'square_metre'

export interface FertilizerRequest {
  crop: string
  growthStage?: string
  areaValue: number
  areaUnit: FertilizerAreaUnit
  soilTest?: {
    units: 'kg_per_ha' | 'ppm'
    nitrogen?: number
    phosphorus?: number
    potassium?: number
    ph?: number
    source?: string
    sampledOn?: string
  } | null
  product?: {
    name: string
    nitrogenPct?: number
    p2o5Pct?: number
    k2oPct?: number
  } | null
  method?: string
}

export interface FertilizerNutrientLine {
  nutrient: string
  targetKgPerHa: number
  supplyKgPerHa: number | null
  deficitKgPerHa: number | null
  totalForFieldKg: number | null
}

export interface FertilizerRecommendation {
  status: 'computed' | 'needs_soil_test' | 'missing_product'
  crop: string
  growthStage: string | null
  areaValue: number
  areaUnit: string
  areaHectares: number
  productName: string | null
  nutrients: FertilizerNutrientLine[]
  totalProductForFieldKg: number | null
  applicationPerHaKg: number | null
  splits: Array<{
    sequence: number
    kgForField: number
  }>
  derivation: string[]
  missingInputs: string[]
  assumptions: string[]
  reminders: string[]
}

export interface FertilizerTargets {
  source: string
  units: string
  status: string
  targets: Array<{
    crop: string
    nitrogenKgHa: number
    p2o5KgHa: number
    k2oKgHa: number
  }>
}
