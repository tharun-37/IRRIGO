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
}

export interface SeasonInfo {
  stageWindows: StageWindow[]
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
  soilType: string
  method: string | null
  sowingDate: string
  areaM2: number
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
  risk: {
    diseaseRisk: string | null
    diseaseConfidence: number
    riskLevel: RiskLevel | null
    cropStressIndex: number
    moistureBand: MoistureBand | null
    yieldTPerHa: number
  }
  reasoning: ReasoningItem[]
  schedule: ScheduleDay[]
  /** Stage-by-stage season plan; absent on the plain compat service. */
  season?: SeasonInfo
  /** Both learned models and their fusion; absent on the plain compat service. */
  models?: AdvisoryModels
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
