/**
 * Crop Health — upload, history and honest model status.
 *
 * The page is built around what the backend actually is today: an image
 * quality service with an inference slot waiting for a validated model. The
 * status banner states that in plain words; the upload shows every quality
 * gate; the history records every failed upload the same as a successful one,
 * because a farmer's poor photo is part of the record too.
 *
 * The fertilizer card sits here because it is the crop-guidance sibling the
 * blueprint pairs it with: a prediction the models can never make alone, since
 * it needs the farmer's soil test, the product formulation and the real field
 * area. Its rule is visible in the result: no soil test means no product
 * quantity, ever.
 */

import { useCallback, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import { useLive } from '../App'
import { usePollingFetch } from '../hooks/usePollingFetch'
import { Card, SectionCard } from '../components/ui'
import { clockTime, num, titleCase } from '../lib/format'
import type {
  CropHealthAnalysis,
  CropHealthResultStatus,
  FertilizerRecommendation,
} from '../types'

const STATUS_TONE: Record<CropHealthResultStatus, { dot: string; text: string; bg: string; label: string }> = {
  result: { dot: 'bg-status-healthy', text: 'text-status-healthy', bg: 'bg-status-healthySoft', label: 'Result' },
  uncertain: { dot: 'bg-status-warn', text: 'text-status-warn', bg: 'bg-status-warnSoft', label: 'Uncertain' },
  unsupported_crop: { dot: 'bg-status-warn', text: 'text-status-warn', bg: 'bg-status-warnSoft', label: 'Crop not covered' },
  poor_image: { dot: 'bg-status-alert', text: 'text-status-alert', bg: 'bg-status-alertSoft', label: 'Poor photo' },
  invalid_input: { dot: 'bg-status-alert', text: 'text-status-alert', bg: 'bg-status-alertSoft', label: 'Invalid upload' },
  processing_error: { dot: 'bg-status-alert', text: 'text-status-alert', bg: 'bg-status-alertSoft', label: 'Service error' },
}

const CROP_STAGE_OPTIONS = ['early', 'vegetative', 'reproductive', 'maturity']

const PRODUCT_PRESETS: Record<string, { n?: number; p?: number; k?: number }> = {
  Urea: { n: 46 },
  DAP: { p: 46, n: 18 },
  MOP: { k: 60 },
  NPK: { n: 17, p: 17, k: 17 },
}

export default function CropHealth() {
  const { bump, refresh } = useLive()
  const status = usePollingFetch(useCallback(() => api.status(), []), bump, 30_000)
  const analyses = usePollingFetch(useCallback(() => api.analyses(30), []), bump, 10_000)
  const cropsCatalogue = usePollingFetch(useCallback(() => api.options(), []), 0, 60_000)

  // --- upload form state -------------------------------------------------
  const [file, setFile] = useState<File | null>(null)
  const [crop, setCrop] = useState('Rice')
  const [plantPart, setPlantPart] = useState('leaf')
  const [growthStage, setGrowthStage] = useState('')
  const [fieldId, setFieldId] = useState('')
  const [consent, setConsent] = useState(true)
  const [analyzing, setAnalyzing] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [latest, setLatest] = useState<CropHealthAnalysis | null>(null)
  const [removing, setRemoving] = useState<string | null>(null)

  // --- fertilizer form state ----------------------------------------------
  const [fertCrop, setFertCrop] = useState('Wheat')
  const [fertStage, setFertStage] = useState('vegetative')
  const [areaValue, setAreaValue] = useState(1)
  const [soilN, setSoilN] = useState('')
  const [soilP, setSoilP] = useState('')
  const [soilK, setSoilK] = useState('')
  const [soilPh, setSoilPh] = useState('')
  const [soilUnits, setSoilUnits] = useState<'kg_per_ha' | 'ppm'>('kg_per_ha')
  const [productName, setProductName] = useState('Urea')
  const [productN, setProductN] = useState('46')
  const [productP, setProductP] = useState('')
  const [productK, setProductK] = useState('')
  const [fertBusy, setFertBusy] = useState(false)
  const [fertResult, setFertResult] = useState<FertilizerRecommendation | null>(null)
  const [fertError, setFertError] = useState<string | null>(null)

  const cropOptions = cropsCatalogue.data?.crops ?? []
  const history = analyses.data?.analyses ?? []

  const submit = useCallback(async () => {
    if (!file) {
      setUploadError('Choose a photo first.')
      return
    }
    setAnalyzing(true)
    setUploadError(null)
    try {
      const result = await api.analyzeCropHealth(file, {
        crop,
        plantPart: plantPart || undefined,
        growthStage: growthStage || undefined,
        fieldId: fieldId || undefined,
        consent,
      })
      setLatest(result)
      refresh()
    } catch (cause) {
      setUploadError(cause instanceof ApiError ? cause.message : 'The upload failed.')
    } finally {
      setAnalyzing(false)
    }
  }, [file, crop, plantPart, growthStage, fieldId, consent, refresh])

  const remove = useCallback(
    async (id: string) => {
      setRemoving(id)
      try {
        await api.deleteAnalysis(id)
        if (latest?.analysisId === id) setLatest(null)
        refresh()
      } catch (cause) {
        setUploadError(cause instanceof ApiError ? cause.message : 'Could not delete the record.')
      } finally {
        setRemoving(null)
      }
    },
    [latest, refresh],
  )

  const applyProductPreset = useCallback((name: string) => {
    setProductName(name)
    const preset = PRODUCT_PRESETS[name]
    setProductN(preset?.n != null ? String(preset.n) : '')
    setProductP(preset?.p != null ? String(preset.p) : '')
    setProductK(preset?.k != null ? String(preset.k) : '')
  }, [])

  const recommend = useCallback(async () => {
    setFertBusy(true)
    setFertError(null)
    try {
      const soilEmpty = soilN === '' && soilP === '' && soilK === ''
      const result = await api.fertilizerRecommend({
        crop: fertCrop,
        growthStage: fertStage || undefined,
        areaValue,
        areaUnit: 'acre',
        soilTest: soilEmpty
          ? null
          : {
              units: soilUnits,
              nitrogen: soilN === '' ? undefined : Number(soilN),
              phosphorus: soilP === '' ? undefined : Number(soilP),
              potassium: soilK === '' ? undefined : Number(soilK),
              ph: soilPh === '' ? undefined : Number(soilPh),
              source: 'farmer-entered',
            },
        product:
          productName === ''
            ? null
            : {
                name: productName,
                nitrogenPct: productN === '' ? undefined : Number(productN),
                p2o5Pct: productP === '' ? undefined : Number(productP),
                k2oPct: productK === '' ? undefined : Number(productK),
              },
      })
      setFertResult(result)
      refresh()
    } catch (cause) {
      setFertError(cause instanceof ApiError ? cause.message : 'The calculation failed.')
    } finally {
      setFertBusy(false)
    }
  }, [fertCrop, fertStage, areaValue, soilN, soilP, soilK, soilPh, soilUnits, productName, productN, productP, productK, refresh])

  const deployed = status.data?.deployedModel ?? false

  return (
    <>
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-[20px] font-semibold tracking-tight text-ink-900">Crop Health</h1>
          <p className="text-[12px] text-ink-500">
            Photo checks, recorded analyses and the fertilizer layer — with the model's true status shown
          </p>
        </div>
        <Link to="/advisor" className="text-[12px] font-medium text-brand-700 hover:underline">
          Back to water advisor
        </Link>
      </div>

      <Card bodyClassName="card-body space-y-3">
        <div className="flex items-center gap-2.5">
          <span
            className={`h-2 w-2 shrink-0 rounded-full ${
              deployed ? 'bg-status-healthy' : 'bg-status-warn'
            }`}
          />
          <p className="text-[13px] font-semibold text-ink-800">
            {deployed
              ? `An image model is deployed (${status.data?.validationStatus})`
              : 'No validated image model is deployed — the service checks photo quality only, and never names a disease'}
          </p>
        </div>
        <ul className="list-disc space-y-1 pl-5 text-[12px] leading-snug text-slate-700">
          {(status.data?.limitations ?? []).map((limit) => (
            <li key={limit}>{limit}</li>
          ))}
        </ul>
      </Card>

      <div className="grid gap-3 lg:grid-cols-2">
        <Card title="Analyse a photo">
          <div className="space-y-2.5">
            <div className="grid grid-cols-2 gap-2">
              <label className="block text-[12px] font-medium text-ink-600">
                Crop
                <select
                  value={crop}
                  onChange={(event) => setCrop(event.target.value)}
                  className="mt-1 w-full rounded border border-ink-200 bg-white px-2 py-1.5 text-[12px] text-ink-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-600"
                >
                  {cropOptions.map((cropOption) => (
                    <option key={cropOption} value={cropOption}>
                      {cropOption}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block text-[12px] font-medium text-ink-600">
                Plant part
                <select
                  value={plantPart}
                  onChange={(event) => setPlantPart(event.target.value)}
                  className="mt-1 w-full rounded border border-ink-200 bg-white px-2 py-1.5 text-[12px] text-ink-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-600"
                >
                  {['leaf', 'whole_plant', 'fruit', 'stem', 'root'].map((part) => (
                    <option key={part} value={part}>
                      {titleCase(part)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block text-[12px] font-medium text-ink-600">
                Growth stage (optional)
                <select
                  value={growthStage}
                  onChange={(event) => setGrowthStage(event.target.value)}
                  className="mt-1 w-full rounded border border-ink-200 bg-white px-2 py-1.5 text-[12px] text-ink-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-600"
                >
                  <option value="">Not sure</option>
                  {CROP_STAGE_OPTIONS.map((stage) => (
                    <option key={stage} value={stage}>
                      {titleCase(stage)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="block text-[12px] font-medium text-ink-600">
                Field (optional)
                <input
                  type="text"
                  value={fieldId}
                  onChange={(event) => setFieldId(event.target.value)}
                  placeholder="e.g. Barley-West"
                  className="mt-1 w-full rounded border border-ink-200 bg-white px-2 py-1.5 text-[12px] text-ink-900 placeholder:text-ink-300 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-600"
                />
              </label>
            </div>

            <label className="block text-[12px] font-medium text-ink-600">
              Photo (JPEG, PNG or WebP, max 10 MB)
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                className="mt-1 block w-full rounded border border-ink-200 bg-white px-2 py-1.5 text-[12px] text-ink-800 file:mr-2 file:rounded file:border-0 file:bg-brand-50 file:px-2 file:py-0.5 file:text-[11px] file:font-semibold file:text-brand-700"
              />
            </label>

            <label className="flex items-start gap-2 text-[12px] leading-snug text-slate-700">
              <input
                type="checkbox"
                checked={consent}
                onChange={(event) => setConsent(event.target.checked)}
                className="mt-0.5 h-3.5 w-3.5"
              />
              <span>
                <strong className="font-semibold text-ink-800">Store the photo</strong> so this
                analysis stays in history. Untick the box for analyse-only — the photo is checked and
                deleted immediately, nothing is kept.
              </span>
            </label>

            <button
              type="button"
              onClick={submit}
              disabled={analyzing || !file}
              className="rounded bg-brand-600 px-3 py-1.5 text-[12px] font-semibold text-white shadow-sm transition-colors hover:bg-brand-700 disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-800"
            >
              {analyzing ? 'Analysing…' : 'Analyse photo'}
            </button>
            {uploadError && <p className="text-[12px] font-medium text-status-alert">{uploadError}</p>}

            {latest && <AnalysisResultCard analysis={latest} onDelete={remove} removing={removing === latest.analysisId} />}
          </div>
        </Card>

        <Card title="Recorded analyses">
          {analyses.error && <p className="text-[12px] text-status-alert">{analyses.error}</p>}
          {history.length === 0 && !analyses.error && (
            <p className="text-[12px] text-slate-600">
              No analyses recorded yet. Every analyse — including poor or failed photos — is recorded
              immediately after it runs.
            </p>
          )}
          <ul className="space-y-2">
            {history.map((entry) => {
              const tone = STATUS_TONE[entry.resultStatus] ?? STATUS_TONE.processing_error
              return (
                <li
                  key={entry.analysisId}
                  className="flex items-start gap-2.5 rounded border border-ink-200 px-2.5 py-2"
                >
                  <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${tone.dot}`} />
                  <div className="min-w-0 flex-1">
                    <p className="text-[12px] font-semibold text-ink-800">
                      {tone.label} · {entry.crop} · {clockTime(entry.createdAt)}
                    </p>
                    <p className="mt-0.5 line-clamp-2 text-[11px] leading-snug text-slate-600">
                      {entry.statusReason}
                    </p>
                    <p className="mt-0.5 text-[11px] text-ink-500">
                      model {entry.modelVersion} · advice {entry.advice.adviceCode}
                      {entry.imageReference ? '' : ' · photo not stored'}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => remove(entry.analysisId)}
                    disabled={removing === entry.analysisId}
                    className="shrink-0 rounded border border-ink-200 px-2 py-1 text-[11px] font-semibold text-ink-600 hover:border-status-alert hover:text-status-alert disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-800"
                    title="Delete this record and its stored photo"
                  >
                    {removing === entry.analysisId ? '…' : 'Delete'}
                  </button>
                </li>
              )
            })}
          </ul>
        </Card>
      </div>

      <FertilizerCard
        crop={fertCrop}
        onCrop={setFertCrop}
        stage={fertStage}
        onStage={setFertStage}
        area={areaValue}
        onArea={setAreaValue}
        soils={{ soilN, soilP, soilK, soilPh }}
        onSoil={{ soilN: setSoilN, soilP: setSoilP, soilK: setSoilK, soilPh: setSoilPh }}
        units={soilUnits}
        onUnits={setSoilUnits}
        product={{ name: productName, n: productN, p: productP, k: productK }}
        onProduct={{ name: applyProductPreset, n: setProductN, p: setProductP, k: setProductK }}
        busy={fertBusy}
        result={fertResult}
        error={fertError}
        onSubmit={recommend}
        cropOptions={cropOptions}
      />
    </>
  )
}

function AnalysisResultCard({
  analysis,
  onDelete,
  removing,
}: {
  analysis: CropHealthAnalysis
  onDelete: (id: string) => void
  removing: boolean
}) {
  const tone = STATUS_TONE[analysis.resultStatus] ?? STATUS_TONE.processing_error
  return (
    <div className="rounded border border-ink-200 bg-ink-50/60 p-3">
      <div className="flex items-center justify-between gap-2">
        <span className={`inline-flex items-center gap-1.5 rounded px-2 py-0.5 text-[11px] font-bold ${tone.bg} ${tone.text}`}>
          <span className={`h-1.5 w-1.5 rounded-full ${tone.dot}`} />
          {tone.label}
        </span>
        <button
          type="button"
          onClick={() => onDelete(analysis.analysisId)}
          disabled={removing}
          className="text-[11px] font-semibold text-ink-500 hover:text-status-alert disabled:opacity-50"
        >
          {removing ? 'Deleting…' : 'Delete record'}
        </button>
      </div>

      <p className="mt-2 text-[13px] font-semibold text-ink-900">
        {analysis.prediction.classLabel
          ? `${titleCase(analysis.prediction.classLabel)} (model reading, not a diagnosis)`
          : `${analysis.crop} · no reliable class returned`}
      </p>
      {analysis.prediction.rawScore != null && (
        <p className="text-[11px] text-slate-600">
          raw score {analysis.prediction.rawScore}
          {analysis.prediction.calibrated ? '' : ' — not a calibrated probability'}
        </p>
      )}
      <p className="mt-1 text-[12px] leading-snug text-slate-700">{analysis.statusReason}</p>

      <p className="mt-2 text-[12px] font-semibold text-ink-700">What to do next</p>
      <p className="text-[12px] leading-snug text-slate-700">{analysis.advice.adviceText}</p>
      {analysis.advice.followUpQuestion && (
        <p className="mt-1 text-[11px] text-ink-500">Check: {analysis.advice.followUpQuestion}</p>
      )}
      {analysis.advice.expertReferral && (
        <p className="mt-1 text-[11px] font-semibold text-status-warn">
          Contact your local agricultural officer for a confirmed identification.
        </p>
      )}

      {analysis.gates.length > 0 && (
        <ul className="mt-2 grid gap-1 border-t border-ink-200 pt-2 text-[11px] sm:grid-cols-2">
          {analysis.gates.map((gate) => (
            <li key={gate.check} className="flex items-start gap-1.5">
              <span className={`mt-1 h-1.5 w-1.5 shrink-0 rounded-full ${gate.passed ? 'bg-status-healthy' : 'bg-status-alert'}`} />
              <span className="text-slate-700">
                <span className="font-semibold text-ink-700">{titleCase(gate.check)}</span> — {gate.detail}
              </span>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-2 border-t border-ink-200 pt-2 text-[11px] text-ink-500">
        model {analysis.modelVersion} · preprocessing {analysis.preprocessingVersion} ·{' '}
        {analysis.validationStatus}
      </p>
    </div>
  )
}

interface FertilizerCardProps {
  crop: string
  onCrop: (value: string) => void
  stage: string
  onStage: (value: string) => void
  area: number
  onArea: (value: number) => void
  soils: { soilN: string; soilP: string; soilK: string; soilPh: string }
  onSoil: { soilN: (v: string) => void; soilP: (v: string) => void; soilK: (v: string) => void; soilPh: (v: string) => void }
  units: 'kg_per_ha' | 'ppm'
  onUnits: (value: 'kg_per_ha' | 'ppm') => void
  product: { name: string; n: string; p: string; k: string }
  onProduct: { name: (v: string) => void; n: (v: string) => void; p: (v: string) => void; k: (v: string) => void }
  busy: boolean
  result: FertilizerRecommendation | null
  error: string | null
  onSubmit: () => void
  cropOptions: string[]
}

const FERT_STATUS: Record<string, { dot: string; text: string; bg: string; label: string }> = {
  computed: { dot: 'bg-status-healthy', text: 'text-status-healthy', bg: 'bg-status-healthySoft', label: 'Computed' },
  needs_soil_test: { dot: 'bg-status-warn', text: 'text-status-warn', bg: 'bg-status-warnSoft', label: 'Needs a soil test' },
  missing_product: { dot: 'bg-status-warn', text: 'text-status-warn', bg: 'bg-status-warnSoft', label: 'No product given' },
}

function FertilizerCard(props: FertilizerCardProps) {
  const { result } = props
  const input =
    'w-full rounded border border-ink-200 bg-white px-2 py-1.5 text-[12px] text-ink-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-600'
  const label = 'block text-[12px] font-medium text-ink-600'
  return (
    <Card title="Fertilizer guidance (separate from the prediction model)">
      <div className="grid gap-3 lg:grid-cols-2">
        <div className="space-y-2.5">
          <p className="text-[12px] leading-snug text-slate-700">
            The calculation needs three things only you can provide: a soil test, the product's
            nutrient percentages, and the real area. Missing values are never read as zero — the
            result below says what is missing instead of inventing a quantity.
          </p>
          <div className="grid grid-cols-3 gap-2">
            <label className={label}>
              Crop
              <select value={props.crop} onChange={(event) => props.onCrop(event.target.value)} className={`mt-1 ${input}`}>
                {props.cropOptions.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </label>
            <label className={label}>
              Stage
              <select value={props.stage} onChange={(event) => props.onStage(event.target.value)} className={`mt-1 ${input}`}>
                {CROP_STAGE_OPTIONS.map((stage) => (
                  <option key={stage} value={stage}>{titleCase(stage)}</option>
                ))}
              </select>
            </label>
            <label className={label}>
              Area (acres)
              <input type="number" min={0.01} step={0.01} value={props.area}
                onChange={(event) => props.onArea(Number(event.target.value))}
                className={`mt-1 ${input}`} />
            </label>
          </div>

          <div className="grid grid-cols-4 gap-2">
            <label className={label}>
              Soil test units
              <select value={props.units} onChange={(event) => props.onUnits(event.target.value as 'kg_per_ha' | 'ppm')} className={`mt-1 ${input}`}>
                <option value="kg_per_ha">kg/ha</option>
                <option value="ppm">ppm</option>
              </select>
            </label>
            <label className={label}>
              N
              <input type="number" step={0.1} value={props.soils.soilN} onChange={(event) => props.onSoil.soilN(event.target.value)} placeholder="—" className={`mt-1 ${input}`} />
            </label>
            <label className={label}>
              P
              <input type="number" step={0.1} value={props.soils.soilP} onChange={(event) => props.onSoil.soilP(event.target.value)} placeholder="—" className={`mt-1 ${input}`} />
            </label>
            <label className={label}>
              K
              <input type="number" step={0.1} value={props.soils.soilK} onChange={(event) => props.onSoil.soilK(event.target.value)} placeholder="—" className={`mt-1 ${input}`} />
            </label>
          </div>

          <div className="grid grid-cols-4 gap-2">
            <label className={label}>
              Soil pH (optional)
              <input type="number" step={0.1} value={props.soils.soilPh} onChange={(event) => props.onSoil.soilPh(event.target.value)} placeholder="—" className={`mt-1 ${input}`} />
            </label>
            <label className={label}>
              Product
              <select value={props.product.name} onChange={(event) => props.onProduct.name(event.target.value)} className={`mt-1 ${input}`}>
                {['Urea', 'DAP', 'MOP', 'NPK'].map((preset) => (
                  <option key={preset} value={preset}>{preset}</option>
                ))}
                <option value="">None</option>
              </select>
            </label>
            <label className={label}>
              N %
              <input type="number" step={0.1} value={props.product.n} onChange={(event) => props.onProduct.n(event.target.value)} className={`mt-1 ${input}`} />
            </label>
            <div className="grid grid-cols-2 gap-1">
              <label className={label}>
                P2O5 %
                <input type="number" step={0.1} value={props.product.p} onChange={(event) => props.onProduct.p(event.target.value)} className={`mt-1 ${input}`} />
              </label>
              <label className={label}>
                K2O %
                <input type="number" step={0.1} value={props.product.k} onChange={(event) => props.onProduct.k(event.target.value)} className={`mt-1 ${input}`} />
              </label>
            </div>
          </div>

          <button
            type="button"
            onClick={props.onSubmit}
            disabled={props.busy}
            className="rounded bg-brand-600 px-3 py-1.5 text-[12px] font-semibold text-white shadow-sm transition-colors hover:bg-brand-700 disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-800"
          >
            {props.busy ? 'Calculating…' : 'Calculate recommendation'}
          </button>
          {props.error && <p className="text-[12px] font-medium text-status-alert">{props.error}</p>}
        </div>

        <SectionCard title="Result">
          {!result && <p className="text-[12px] text-slate-600">Fill the left panel and calculate.</p>}
          {result && <FertilizerResult result={result} />}
        </SectionCard>
      </div>
    </Card>
  )
}

function FertilizerResult({ result }: { result: FertilizerRecommendation }) {
  const tone = FERT_STATUS[result.status] ?? FERT_STATUS.missing_product
  return (
    <div className="space-y-2">
      <span className={`inline-flex items-center gap-1.5 rounded px-2 py-0.5 text-[11px] font-bold ${tone.bg} ${tone.text}`}>
        <span className={`h-1.5 w-1.5 rounded-full ${tone.dot}`} />
        {tone.label}
      </span>

      <div className="grid grid-cols-3 gap-1.5">
        {result.nutrients.map((line) => (
          <div key={line.nutrient} className="rounded border border-ink-200 px-2 py-1.5">
            <p className="text-[11px] font-bold uppercase text-ink-500">{line.nutrient}</p>
            <p className="text-[14px] font-semibold text-ink-900">{line.targetKgPerHa}</p>
            <p className="text-[11px] text-slate-600">kg/ha target</p>
            <p className="mt-0.5 text-[11px] text-slate-600">
              {line.supplyKgPerHa != null ? `soil ${num(line.supplyKgPerHa, 1)} kg/ha` : 'soil test missing'}
            </p>
          </div>
        ))}
      </div>

      {result.totalProductForFieldKg != null && (
        <p className="text-[13px] font-semibold text-ink-900">
          {num(result.totalProductForFieldKg, 2)} kg of {result.productName} for{' '}
          {num(result.areaValue, 2)} acres ({num(result.applicationPerHaKg ?? 0, 1)} kg/ha)
        </p>
      )}
      {result.splits.length > 0 && (
        <p className="text-[12px] text-slate-700">
          Split into {result.splits.length} doses of about{' '}
          {num(result.splits[0].kgForField, 2)} kg each.
        </p>
      )}

      {result.missingInputs.length > 0 && (
        <div>
          <p className="text-[12px] font-semibold text-status-warn">Missing inputs</p>
          <ul className="list-disc pl-5 text-[12px] text-slate-700">
            {result.missingInputs.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      <details className="text-[12px] text-slate-700">
        <summary className="cursor-pointer font-semibold text-ink-700">How this was derived</summary>
        <ul className="mt-1 list-disc pl-5 space-y-0.5">
          {result.derivation.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
        {result.assumptions.length > 0 && (
          <>
            <p className="mt-1 font-semibold text-ink-700">Assumptions</p>
            <ul className="mt-1 list-disc pl-5 space-y-0.5">
              {result.assumptions.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          </>
        )}
      </details>

      <details className="text-[12px] text-slate-700">
        <summary className="cursor-pointer font-semibold text-ink-700">Reminders</summary>
        <ul className="mt-1 list-disc pl-5 space-y-0.5">
          {result.reminders.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </details>
    </div>
  )
}
