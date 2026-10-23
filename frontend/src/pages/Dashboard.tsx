/**
 * Field view: is the soil dry, how old is the crop, and what should I do.
 *
 * Those three questions are the whole job, and they are answered in that order of
 * importance. Everything on this screen exists to support one of them; anything
 * that did not was cut, including — in V3 — the console view this page used to
 * carry behind a Farmer/Console switch. The app is for farmers, not developers,
 * so the farmer view is the only view and the header carries nothing a farmer
 * would not press.
 *
 * The screen itself lives in `FarmerView`; this module is the shell around it:
 * the field switcher, the field-registration handover, and the reductions the
 * farmer view is fed (per-field rows, the three advice chips). The advice chips
 * are built from the same helpers the old console cards used, so removing the
 * console could not make the farmer view disagree with the numbers it replaced.
 */

import { useCallback, useMemo, useState } from 'react'
import { api } from '../api/client'
import { useLive } from '../App'
import { usePollingFetch } from '../hooks/usePollingFetch'
import { GrainFilter, LeafGlyph } from '../components/ui'
import { AddField } from '../components/AddField'
import { FarmerView, type FarmerChip, type FarmerField } from './FarmerView'
import { int, num, relativeTime, titleCase } from '../lib/format'
import type { Advisory } from '../types'

const REFRESH = 15_000

type Band = 'dry' | 'good' | 'wet' | 'rest' | 'none'

function depthOf(field: Advisory): number {
  return field.recommendation.finalDepthMm ?? field.recommendation.depthMm
}

/**
 * Sufficiency bands, from V1's own feature module.
 *
 * These are the thresholds the inherited models were trained against, not
 * agronomic guesses: a reading inside them is one the decision model treats as
 * unconstrained. Sourced here rather than fetched so the farmer view renders the
 * same numbers the inference used.
 */
const NUTRIENT_BANDS = {
  nitrogen: [20, 40, 90, 120] as const,
  phosphorus: [15, 25, 60, 90] as const,
  potassium: [20, 35, 80, 110] as const,
}

/** True once the crop has been taken, when there is nothing left to water. */
function isFinished(field: Advisory): boolean {
  return stageKey(field) === 'post_harvest' || stageKey(field) === 'harvest'
}

/**
 * Soil wetness as a share of plantable water still held, and its verdict.
 */
function wetness(field: Advisory): { remaining: number; band: Band } {
  const depletion = field.water.depletionFraction
  let remaining = Math.max(0, Math.min(1, 1 - depletion))
  if (remaining >= 0.98) {
    remaining = 0.92
  }
  if (isFinished(field)) {
    return { remaining, band: 'rest' }
  }
  const band: Band = depletion >= 0.55 ? 'dry' : depletion <= 0.15 ? 'wet' : 'good'
  return { remaining, band }
}

function stageKey(field: Advisory): string {
  return String(field.stage ?? '').toLowerCase()
}

/** Square metres in an acre, for the farmer view's area conversion. */
const ACRE_M2 = 4046.86

/**
 * The fertilizer a shortfall turns into, in the product an operator buys.
 *
 * `fraction` is the share of the nutrient in the product, so the dose in kg/ha
 * of the product is the nutrient dose divided by it: urea is 46% nitrogen,
 * DAP 46% phosphate, muriate of potash 60% potash.
 */
const FERTILIZER_PRODUCTS = {
  nitrogen: { name: 'urea', fraction: 0.46 },
  phosphorus: { name: 'DAP', fraction: 0.46 },
  potassium: { name: 'MOP', fraction: 0.6 },
} as const

/** The element symbols a farmer reads on a fertilizer bag. */
const NUTRIENT_SYMBOLS = {
  nitrogen: 'N',
  phosphorus: 'P',
  potassium: 'K',
} as const

/**
 * The nutrients the soil is short of, by the same band test the fertilizer
 * card used.
 */
function nutrientShortages(s: Advisory['sensors']): (keyof typeof NUTRIENT_BANDS)[] {
  return (['nitrogen', 'phosphorus', 'potassium'] as const).filter(
    (nutrient) => s[nutrient] < NUTRIENT_BANDS[nutrient][0],
  )
}

/**
 * Fertilizer advice from live nutrient readings against V1 sufficiency bands.
 *
 * A shortfall is reported as the nutrient that is short; the farmer view turns
 * the engine's dose into bags of the product the farmer would actually buy.
 */
function fertilizerAdvice(
  s: Advisory['sensors'],
): { tone: 'healthy' | 'warn' | 'alert'; shortages: string[] } {
  const shortages = nutrientShortages(s).map((nutrient) => titleCase(nutrient))
  return { tone: shortages.length > 0 ? 'warn' : 'healthy', shortages }
}

/**
 * Pesticide advice from the disease risk level.
 *
 * A `risk` verdict always asks for a fungicide, whatever the model's
 * confidence. A farmer told "Disease risk" and then "no spray needed" on the
 * same day has been handed a contradiction to resolve alone — and "no spray
 * needed" on a risky crop is the expensive way to be wrong.
 */
function pesticideAdvice(
  risk: Advisory['risk'],
): { label: string; tone: 'healthy' | 'warn' | 'alert' } {
  const level = risk.riskLevel ?? 'healthy'

  if (level === 'risk') {
    return { label: 'Apply fungicide', tone: 'alert' }
  }
  if (level === 'watch') {
    return { label: 'Monitor closely', tone: 'warn' }
  }
  return { label: 'No pesticide needed', tone: 'healthy' }
}

/**
 * The farmer view's three chips: fertiliser, spray, yield.
 *
 * Built from the same helpers the old console cards used, so removing the
 * console could not make the farmer view disagree with the numbers it
 * replaced. What stays is the unit: the farmer buys bags, not kg/ha, and the
 * harvest is quoted per acre because that is the land measure the work is
 * done in.
 */
function farmerChips(field: Advisory): FarmerChip[] {
  const acres = field.areaM2 / ACRE_M2
  const fertilizer = fertilizerAdvice(field.sensors)
  const shortageKeys = nutrientShortages(field.sensors)
  const pesticide = pesticideAdvice(field.risk)

  let fertilizerChip: FarmerChip
  const shortage = shortageKeys[0]
  if (fertilizer.tone === 'warn' && shortage) {
    const product = FERTILIZER_PRODUCTS[shortage]
    const nutrientKgHa = field.risk.nutrientDemandKgHa?.[shortage] ?? 0
    const productKgAcre = nutrientKgHa / product.fraction / 2.471
    const bags = (productKgAcre * acres) / 50
    fertilizerChip = {
      key: 'fertilizer',
      title:
        bags < 0.25
          ? `Add a little ${product.name}`
          : `Add ${
              bags < 0.75 ? 'half a bag' : bags < 1.25 ? '1 bag' : `${Math.round(bags)} bags`
            } of ${product.name}`,
      detail: `${num(productKgAcre, 0)} kg/acre · ${shortageKeys
        .map((nutrient) => NUTRIENT_SYMBOLS[nutrient])
        .join(' and ')} short`,
      tone: 'warn',
    }
  } else {
    fertilizerChip = {
      key: 'fertilizer',
      title: 'No fertilizer today',
      detail: 'No fertilizer needed at this stage.',
      tone: 'healthy',
    }
  }

  const sprayChip: FarmerChip =
    pesticide.tone === 'alert'
      ? {
          key: 'spray',
          title: 'Spray fungicide',
          detail: 'Disease risk is high. Treat today.',
          tone: 'alert',
        }
      : pesticide.tone === 'warn'
        ? {
            key: 'spray',
            title: 'Check for disease',
            detail: 'Look for spots or rust before spraying.',
            tone: 'warn',
          }
        : {
            key: 'spray',
            title: 'No spray today',
            detail: 'No disease risk in this weather.',
            tone: 'healthy',
          }

  // Rough yield estimate: acres × 5 quintal/acre
  const fieldAcres = field.areaM2 / ACRE_M2
  const modelled = field.risk.yieldModelled ?? false
  const quintalAcreModelled = (field.risk.yieldTPerHa ?? 0) * 4.047
  const quintalAcre = modelled ? quintalAcreModelled : 5
  const totalQuintal = modelled ? quintalAcreModelled * fieldAcres : fieldAcres * 5
  const isBarleyGeneric = !modelled && field.crop?.toLowerCase().includes('barley')
  const yieldChip: FarmerChip = {
    key: 'yield',
    title: modelled ? 'Expected yield' : 'Rough yield estimate',
    detail: `${int(totalQuintal)} quintal total · ${num(quintalAcre, 0)} quintal/acre${isBarleyGeneric ? ' (rough)' : ''}`,
    tone: isBarleyGeneric ? 'warn' : 'neutral',
  }

  return [fertilizerChip, sprayChip, yieldChip]
}

/**
 * Reduce the advisory feed to the rows the farmer view lists and maps.
 *
 * The action is the same rule the old console hero used — a depth on the
 * recommendation means water today — so the row's colour and the headline
 * cannot disagree. Rows are ordered by urgency rather than by the feed's own
 * order, because the point of the list is to show the farmer which field to
 * walk to first.
 */
function farmerFields(fields: Advisory[]): FarmerField[] {
  const rows = fields.map((item): FarmerField => {
    const soil = wetness(item)
    const finished = soil.band === 'rest'
    const seconds = item.recommendation.durationSeconds
    const action: FarmerField['action'] = finished
      ? 'rest'
      : depthOf(item) > 0 || seconds > 0
        ? 'water'
        : item.status === 'monitor'
          ? 'watch'
          : 'ok'
    const minutes = Math.max(0, Math.round(seconds / 60))

    return {
      fieldId: item.fieldId,
      crop: item.crop,
      stage: item.stageLabel,
      areaM2: item.areaM2,
      remaining: soil.remaining,
      band: soil.band,
      minutes: action === 'water' ? Math.max(1, minutes) : 0,
      action,
      station: item.station,
      stationLat: item.stationLat ?? null,
      stationLon: item.stationLon ?? null,
    }
  })

  const urgency: Record<FarmerField['action'], number> = {
    water: 0,
    watch: 1,
    ok: 2,
    rest: 3,
  }
  return rows.sort((a, b) => urgency[a.action] - urgency[b.action])
}

export default function Dashboard() {
  const { bump, refresh } = useLive()
  const [fieldId, setFieldId] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)

  const advisor = usePollingFetch(useCallback(() => api.advisor(), []), bump, REFRESH)

  const feed = advisor.data
  const fields = useMemo(() => feed?.fields ?? [], [feed])
  const field = useMemo(
    () => fields.find((item) => item.fieldId === fieldId) ?? fields[0],
    [fields, fieldId],
  )

  // The live sky reading, when a live feed is configured. Polled far less often
  // than the advisory: the upstream observation is cached for ten minutes
  // server-side, so polling faster would only repeat the same number.
  const station = field?.station ?? ''
  const live = usePollingFetch(
    useCallback(
      () => (station ? api.liveWeather(station) : Promise.resolve(null)),
      [station],
    ),
    0,
    300_000,
  )

  const depth = field ? depthOf(field) : 0
  const soil = field ? wetness(field) : { remaining: 0, band: 'none' as Band }
  // The farmer view's list and map are a reduction of the same feed, so they
  // are derived once per payload rather than per render.
  const farmRows = useMemo(() => farmerFields(fields), [fields])

  return (
    <div className="grain min-h-screen lg:flex lg:flex-col lg:overflow-hidden lg:h-screen">
      <GrainFilter />
      <div className="grain-layer" aria-hidden="true" />
      {/* ---- Field switcher --------------------------------------------- */}
      {/* The bar spans the window, but its contents sit in the same centred
          column as the page below, so the logo and the cards share an edge.
          `1728` is the page column plus its two gutters (1680 + 2×24 at 1920). */}
      <header className="shrink-0 border-b border-white/70 bg-white/70 backdrop-blur">
        <div className="pad-gutter flex w-full flex-wrap items-center gap-2 py-2.5">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-card bg-gradient-to-br from-ink-900 to-ink-700 text-white shadow-lift">
            <LeafGlyph size={16} />
          </span>
          <span className="hidden shrink-0 text-[14px] font-semibold tracking-tight text-ink-900 sm:inline">
            IRRIGO
          </span>
          {/* The switcher drops to its own row on a phone. Sharing one 390px row
              with four controls left each field pill about forty pixels wide, and
              choosing the field is the first thing a farmer does here. */}
          <div className="order-last ml-2 flex w-full min-w-0 gap-1.5 overflow-x-auto sm:order-none sm:w-auto sm:flex-1">
            {fields.map((item) => {
              const active = item.fieldId === field?.fieldId
              // The dot carries the water decision, not sensor health: the
              // question a farmer asks is "does this field need me today".
              const action = farmRows.find((row) => row.fieldId === item.fieldId)?.action ?? 'ok'
              const dotTone =
                action === 'water'
                  ? 'bg-status-warn'
                  : action === 'watch'
                    ? 'bg-brand-500'
                    : action === 'rest'
                      ? 'bg-ink-300'
                      : 'bg-status-healthy'
              return (
                <button
                  key={item.fieldId}
                  type="button"
                  onClick={() => setFieldId(item.fieldId)}
                  title={`${item.crop} · ${item.ageDays} days · ${item.status}`}
                  className={`flex shrink-0 items-center gap-2 rounded-pill border px-3 py-1.5 text-[12px] font-semibold transition-colors ${
                    active
                      ? 'border-ink-900 bg-ink-900 text-white shadow-lift'
                      : 'border-white/80 bg-white/70 text-ink-600 hover:border-ink-300 hover:bg-white'
                  }`}
                >
                  {/* Amber: water today. Blue: soon. Green: fine. Grey: resting. */}
                  <span
                    className={`h-2 w-2 shrink-0 rounded-full ${dotTone} ${
                      dotTone === 'bg-status-healthy' && !active
                        ? 'ring-2 ring-status-healthy/25'
                        : ''
                    }`}
                  />
                  {item.fieldId}
                </button>
              )
            })}
          </div>
          {/* Add a field, in the same row as the switcher it will add to. The
              switcher is the one place on the screen that already answers "which
              fields exist", so putting the button beside it means the action is
              reachable from where the list it changes is read, rather than from a
              menu somewhere else.

              Labelled "Add field" rather than a bare "+": the pills immediately to
              its left are themselves fields, and a lone "+" next to them reads as
              another field until it is clicked. */}
          <button
            type="button"
            onClick={() => setAdding((open) => !open)}
            aria-expanded={adding}
            className="flex shrink-0 items-center gap-1 rounded-pill border border-dashed border-ink-400 px-3 py-1.5 text-[12px] font-semibold text-ink-500 transition-colors hover:border-ink-900 hover:bg-white hover:text-ink-900"
          >
            <span aria-hidden="true" className="text-[14px] leading-none">
              {adding ? '×' : '+'}
            </span>
            {adding ? 'Close' : 'Add field'}
          </button>
          <button
            type="button"
            onClick={refresh}
            className="shrink-0 text-[11px] text-ink-400 hover:text-ink-700"
          >
            {feed ? relativeTime(feed.generatedAt) : '…'}
          </button>
          {/* Whole-farm status, in one line beside the switcher: how many of the
              fields need the farmer today. */}
          {farmRows.length > 0 && (
            <span className="shrink-0 rounded-pill bg-white/70 px-2.5 py-1 text-[12px] font-semibold text-ink-700">
              {farmRows.filter((row) => row.action === 'water').length} of {farmRows.length}{' '}
              fields need water
            </span>
          )}
        </div>
      </header>

      <main className="scroll-smooth min-h-0 flex-1 overflow-y-auto">
        {advisor.error ? (
          <p className="p-6 text-[13px] text-status-alert">{advisor.error}</p>
        ) : adding ? (
          /* Registration takes over the page rather than being added to it. A form
             stacked above the field view leaves the operator reading two things at
             once, and the one underneath is about a field they are not registering
             - the question in front of them is "which crop, which station, sown
             when", and every card below it is a distraction from that. */
          <div className="pad-page mx-auto max-w-[900px]">
            <AddField
              onRegistered={(created) => {
                setFieldId(created)
                setAdding(false)
                refresh()
              }}
            />
          </div>
        ) : !field ? (
          <div className="pad-page mx-auto max-w-[1180px] space-y-4">
            <p className="text-[13px] text-ink-400">
              No fields registered yet. Add one to start planning.
            </p>
            <AddField onRegistered={setFieldId} />
          </div>
        ) : (
          <div className="scroll-smooth pad-page flex w-full flex-col gap-[clamp(0.7rem,1.5vw,1.1rem)] lg:h-full lg:overflow-y-auto">
            <FarmerView
              field={field}
              fields={farmRows}
              depth={depth}
              band={soil.band}
              chips={farmerChips(field)}
              live={live.data?.station === field.station ? live.data : null}
              onSelect={setFieldId}
            />
          </div>
        )}
      </main>
    </div>
  )
}
