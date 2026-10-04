/**
 * Field view: is the soil dry, how old is the crop, and what should I do.
 *
 * Those three questions are the whole job, and they are answered in that order of
 * importance. Everything on this screen exists to support one of them; anything
 * that did not was cut, including a schematic field map that looked like a map
 * without being one.
 *
 * **Dryness first.** A donut shows how much of the plantable water is gone,
 * against the refill point where a real operator would turn the pump on. The
 * verbal verdict — dry, good, or wet — is stated outright, because a percentage
 * alone is not something a farmer decides with.
 *
 * **Crop age second.** Water need is not a property of soil alone; a seedling
 * and a plant at peak demand in the same field need different amounts. The season
 * track shows where in its life the crop is, and the stage sets the daily demand.
 *
 * **The recommendation joins the two.** It is stated in millimetres and litres,
 * and the sentence under it explains it in terms of dryness and stage, so the
 * number is never a black box. Where the two learned models disagree the page
 * says so rather than presenting a blended figure as if both had produced it.
 */

import { useCallback, useMemo, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api/client'
import { useLive } from '../App'
import { usePollingFetch } from '../hooks/usePollingFetch'
import {
  GrainFilter,
  LeafGlyph,
  NutrientBar,
  RainGlyph,
  Reading,
  SkyGlyph,
  ToneBadge,
} from '../components/ui'
import { AddField } from '../components/AddField'
import { int, num, relativeTime, titleCase } from '../lib/format'
import type { Advisory } from '../types'

const REFRESH = 15_000

type Band = 'dry' | 'good' | 'wet' | 'rest' | 'none'

const BAND_STYLE: Record<Band, { ring: string; text: string; label: string; word: string }> = {
  dry: { ring: 'stroke-status-warn', text: 'text-status-warn', label: 'text-status-warn', word: 'Dry' },
  good: {
    ring: 'stroke-status-healthy',
    text: 'text-status-healthy',
    label: 'text-status-healthy',
    word: 'Good',
  },
  wet: { ring: 'stroke-brand-600', text: 'text-brand-600', label: 'text-brand-600', word: 'Wet' },
  rest: { ring: 'stroke-ink-300', text: 'text-ink-500', label: 'text-ink-500', word: 'Resting' },
  none: { ring: 'stroke-ink-200', text: 'text-ink-400', label: 'text-ink-400', word: '—' },
}

/** Two-stop gradient per verdict, so the ring reads as lit rather than flat. */
const RING_GRADIENT: Record<Band, [string, string]> = {
  dry: ['#fbbf24', '#ea580c'],
  good: ['#4ade80', '#15803d'],
  wet: ['#60a5fa', '#1d4ed8'],
  rest: ['#cbd5e1', '#94a3b8'],
  none: ['#e2e8f0', '#cbd5e1'],
}

const STAGES = [
  { key: 'initial', label: 'Establish', from: 0, to: 0.2 },
  { key: 'development', label: 'Vegetative', from: 0.2, to: 0.5 },
  { key: 'mid_season', label: 'Peak demand', from: 0.5, to: 0.8 },
  { key: 'late_season', label: 'Ripening', from: 0.8, to: 1 },
]

function depthOf(field: Advisory): number {
  return field.recommendation.finalDepthMm ?? field.recommendation.depthMm
}

/**
 * Sufficiency bands, from V1's own feature module.
 *
 * These are the thresholds the inherited models were trained against, not
 * agronomic guesses: a reading inside them is one the decision model treats as
 * unconstrained. Sourced here rather than fetched so the board renders the same
 * numbers the inference used.
 */
const NUTRIENT_BANDS = {
  nitrogen: [20, 40, 90, 120] as const,
  phosphorus: [15, 25, 60, 90] as const,
  potassium: [20, 35, 80, 110] as const,
}
const PH_BAND: [number, number] = [6.0, 7.5]
const MOISTURE_BAND: [number, number] = [38, 62]

/** The fields' sensor-reading health, for the switcher's status dot. */
function sensorTone(field: Advisory): 'good' | 'warn' | 'offline' {
  const sensors = field.sensors
  const finite = [sensors.soilMoisture, sensors.temperature, sensors.soilPh, sensors.ec].every(
    (value) => typeof value === 'number' && Number.isFinite(value),
  )
  if (!finite) return 'offline'
  const outOfBand =
    sensors.soilMoisture < 12 ||
    sensors.soilPh < 4.5 ||
    sensors.soilPh > 8.5 ||
    sensors.ec > 6.0
  return outOfBand ? 'warn' : 'good'
}

/** True once the crop has been taken, when there is nothing left to water. */
function isFinished(field: Advisory): boolean {
  const stage = stageKey(field)
  return stage === 'post_harvest' || stage === 'harvest'
}

/**
 * Soil wetness as a share of plantable water still held, and its verdict.
 *
 * The verdict is qualified by the crop, not reported on its own. A dry soil is
 * urgent only when something is growing in it: showing "Dry" beside "no water
 * needed" on a field whose crop was harvested three weeks ago would be two true
 * statements that read as a contradiction, and the operator has no way to tell
 * which one to act on.
 */
function wetness(field: Advisory): { remaining: number; band: Band } {
  const depletion = field.water.depletionFraction
  const remaining = Math.max(0, Math.min(1, 1 - depletion))
  if (isFinished(field)) {
    return { remaining, band: 'rest' }
  }
  const band: Band = depletion >= 0.55 ? 'dry' : depletion <= 0.15 ? 'wet' : 'good'
  return { remaining, band }
}

function stageKey(field: Advisory): string {
  return String(field.stage ?? '').toLowerCase()
}

/** The recommendation, in one sentence, in terms of dryness and age. */
function reason(field: Advisory, band: Band): string {
  const crop = field.crop.toLowerCase()
  if (isFinished(field)) {
    return `The ${crop} was harvested. Nothing needs watering until the next sowing, so the soil being dry does not matter.`
  }
  if (band === 'wet') {
    return `The soil is wet. Watering now would waste water and leave the ${crop} without air around its roots.`
  }
  if (field.status === 'schedule') {
    // The soil is still readable; the forward-looking requirement is what asks
    // for water. Saying the soil "still holds what it needs" here would
    // contradict the number underneath it.
    return `The soil is fine today, but the next fortnight needs more than it has: the ${crop} is at ${field.stageLabel.toLowerCase()} and using about ${num(field.water.etcMmDay, 1)} mm a day. Water in before that runs out.`
  }
  if (band === 'dry') {
    const stage = stageKey(field)
    if (stage === 'mid_season' || stage === 'late_season') {
      return `The soil is dry and the ${crop} is at its thirstiest stage, using about ${num(field.water.etcMmDay, 1)} mm a day. Water now.`
    }
    return `The soil is drying out and the ${crop} is still young, so keep its roots damp while it settles in.`
  }
  return `The soil still holds the water the ${crop} needs at this stage. No watering today.`
}

/**
 * The one instruction.
 *
 * Driven by the depth actually being served rather than by the status alone.
 * The two can disagree — the fusion may hold the decision's own depth while
 * leaving its "schedule" status — and a headline that says "no watering today"
 * above a 10.8 mm figure is worse than no headline at all.
 */
function instruction(field: Advisory, depth: number): string {
  if (depth > 0) {
    return field.status === 'irrigate' ? 'Water today' : 'Water within 24 hours'
  }
  switch (field.status) {
    case 'monitor': {
      const days = field.water.daysUntilStress
      return days === null || days === undefined
        ? 'Hold for now'
        : `Hold · water in ${days} day${days === 1 ? '' : 's'}`
    }
    default:
      return 'No water needed'
  }
}

function agreementNote(field: Advisory): string {
  const models = field.models
  if (!models) return ''
  const notes: Record<string, string> = {
    aligned: 'Both models agree on this watering.',
    agree_hold: 'Both models agree: no water needed yet.',
    v1_conservative: 'The models differ; the cautious figure is used.',
    v2_only: 'The models differ; the larger figure is used to be safe.',
    not_applicable: 'Past harvest, so the requirement model is not used.',
    model_unavailable: 'Only the water balance is available right now.',
  }
  return notes[models.fusion.agreement] ?? ''
}

type Banner = 'hold' | 'irrigate' | 'monitor'

function bannerFor(field: Advisory, depth: number): Banner {
  if (depth > 0 || field.status === 'irrigate') return 'irrigate'
  if (field.status === 'monitor') return 'monitor'
  return 'hold'
}

const BANNER_STYLE: Record<
  Banner,
  { badge: string; icon: string; label: string; edge: string }
> = {
  hold: {
    badge: 'bg-status-warn text-white',
    icon: 'text-status-warn',
    label: 'Pump held',
    edge: 'from-status-warnSoft/90',
  },
  irrigate: {
    badge: 'bg-brand-600 text-white',
    icon: 'text-brand-600',
    label: 'Irrigate today',
    edge: 'from-brand-50',
  },
  monitor: {
    badge: 'bg-brand-100 text-brand-700',
    icon: 'text-brand-600',
    label: 'Hold · watch today',
    edge: 'from-brand-50/70',
  },
}

export default function Dashboard() {
  const { bump, refresh } = useLive()
  const [fieldId, setFieldId] = useState<string | null>(null)
  const [adding, setAdding] = useState(false)

  const advisor = usePollingFetch(useCallback(() => api.advisor(), []), bump, REFRESH)
  const analytics = usePollingFetch(useCallback(() => api.analytics(168), []), bump, REFRESH)

  const feed = advisor.data
  const fields = useMemo(() => feed?.fields ?? [], [feed])
  const field = useMemo(
    () => fields.find((item) => item.fieldId === fieldId) ?? fields[0],
    [fields, fieldId],
  )

  const depth = field ? depthOf(field) : 0
  const soil = field ? wetness(field) : { remaining: 0, band: 'none' as Band }
  const totalMm = fields.reduce((sum, item) => sum + depthOf(item), 0)

  // Available water over the recent record, which is the dryness trend a farmer
  // reads: rising means the soil is giving up water faster than it is replaced.
  const trend = useMemo(
    () =>
      (analytics.data?.series ?? []).map((point) => ({
        label: point.label,
        available: Math.max(0, Math.min(100, Math.round((1 - point.depletionFraction) * 100))),
      })),
    [analytics.data],
  )

  return (
    <div className="grain min-h-screen lg:flex lg:flex-col lg:overflow-hidden lg:h-screen">
      <GrainFilter />
      <div className="grain-layer" aria-hidden="true" />
      {/* ---- Field switcher --------------------------------------------- */}
      <header className="flex shrink-0 items-center gap-2 border-b border-white/70 bg-white/70 px-4 py-2.5 backdrop-blur">
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-card bg-gradient-to-br from-ink-900 to-ink-700 text-white shadow-lift">
          <LeafGlyph size={16} />
        </span>
        <span className="shrink-0 text-[14px] font-semibold tracking-tight text-ink-900">
          IRRIGO
        </span>
        <div className="ml-2 flex min-w-0 flex-1 gap-1.5 overflow-x-auto">
          {fields.map((item) => {
            const active = item.fieldId === field?.fieldId
            const tone = sensorTone(item)
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
                {/* Status is readable without selecting the field: green normal,
                    amber needs attention, grey when the head is not reporting. */}
                <span
                  className={`h-2 w-2 shrink-0 rounded-full ${
                    tone === 'good'
                      ? 'bg-status-healthy'
                      : tone === 'warn'
                        ? 'bg-status-warn'
                        : 'bg-ink-300'
                  } ${tone === 'good' && !active ? 'ring-2 ring-status-healthy/25' : ''}`}
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
      </header>

      <main className="min-h-0 flex-1 overflow-y-auto lg:overflow-hidden">
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
<div className="pad-page mx-auto flex max-w-[1180px] flex-col gap-[clamp(0.7rem,1.5vw,1.1rem)] lg:h-full lg:max-w-none lg:overflow-y-auto">
            {/* ---- The answer, on its own full-width row --------------- */}
            {/* This is what the operator opened the page for, so it is not made
                to share a row with three supporting readings and shrink to fit
                beside them. */}
            {/* `shrink-0` on the row wrappers matters. A flex child that clips its own
                overflow (`overflow-hidden` on the hero and the sky card) loses
                its automatic minimum size, so the column was free to squash it
                below its content and the decision's detail lines were cut off.
                The column scrolls; the cards keep their natural height. */}
            <div className="shrink-0">
              <DecisionHero field={field} depth={depth} band={soil.band} />
            </div>

            {/* ---- The evidence for it: three compact readings ---------- */}
            {/* Cards stretch to a common height so the row ends on one line.
                The slack is absorbed by the card's *content* — the wetness row
                centres itself, the sky card already pins its figure to the base —
                so equalising costs no empty space. On md the third card would
                otherwise sit alone in a half-width row, so it spans the row and
                the three read as two even bands. */}
            <div className="grid shrink-0 gap-[clamp(0.7rem,1.5vw,1.1rem)] md:grid-cols-2 xl:grid-cols-3">
              <WetnessCard field={field} remaining={soil.remaining} band={soil.band} />
              <SkyCard field={field} />
              {/* The span wrapper has to stretch the card itself. As a plain
                  block the card kept its own height inside a taller row and sat
                  short of its neighbours' edge; as a `flex` item without
                  `flex-1` it shrink-wrapped its content instead and collapsed
                  into a narrow column. `flex-1` is what fills the cell. */}
              <div className="flex md:col-span-2 xl:col-span-1">
                <AgeCard field={field} />
              </div>
            </div>

            {/* ---- The field record, as a strip rather than a card ------ */}
            <div className="shrink-0">
              <FieldStrip field={field} />
            </div>

            {/* The sensor head, given its own row: seven readings is the most
                looked-at data on the page and it was buried in small text. */}
            <div className="shrink-0">
              <SensorBoard field={field} />
            </div>

            {/* ---- The week ahead ---------------------------------------- */}
            {/* Equal heights again, with the slack handed to the two charts:
                both are `flex-1` inside the card body, so the bars and the area
                plot get taller rather than the cards ending three lines apart. */}
            <div className="grid shrink-0 gap-[clamp(0.7rem,1.5vw,1.1rem)] lg:grid-cols-3">
              <PlanCard field={field} />
              <TrendCard
                trend={trend}
                band={soil.band}
                refill={
                  field.water.tawMm > 0
                    ? Math.round(
                        Math.max(
                          0,
                          Math.min(100, (field.water.rawMm / field.water.tawMm) * 100),
                        ),
                      )
                    : 50
                }
              />
              <BudgetCard fields={fields} totalMm={totalMm} />
            </div>

            <p className="pb-2 pt-1 text-center">
              <Link to="/model" className="text-[11px] text-ink-400 hover:text-ink-700">
                How this recommendation is calculated
              </Link>
            </p>
          </div>
        )}
      </main>
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Cards                                                                       */
/* -------------------------------------------------------------------------- */

/**
 * A titled card.
 *
 * Content is left-aligned and the body keeps its natural height. The previous
 * version took `centred` and `fill` props, which centred sparse contents inside
 * a height imposed by a taller neighbour; the result was a card that was two
 * thirds empty with its text floating in the middle. A card that only holds as
 * much as it needs is easier to read than one aligned to an arbitrary grid.
 */
function Card({
  title,
  hint,
  children,
  className = '',
}: {
  title: string
  hint?: string
  children: ReactNode
  className?: string
}) {
  return (
    <section className={`card flex flex-col ${className}`}>
      <div className="pad-card flex items-baseline justify-between gap-2 pt-[clamp(0.75rem,2cqw,1.1rem)]">
        {/* Title and hint are both `ink-700`. They were `ink-500` and `ink-400`
            when the section fills were a white-to-tint gradient fading out well
            before this corner; against a solid 100-level fill `ink-500` measures
            about 4.1:1, which is under the 4.5:1 small text needs, and `ink-400`
            is far worse. The hierarchy between a title and its hint is already
            carried by size, weight and case, so the colour does not have to do it. */}
        <h2 className="type-label font-semibold uppercase tracking-[0.07em] text-ink-700">
          {title}
        </h2>
        {hint && <span className="type-label shrink-0 text-ink-700/80">{hint}</span>}
      </div>
      {/* The body is a column so that when the card is stretched by a taller
          neighbour, the child marked `flex-1` (a chart, a reading row) grows into
          the difference instead of the card ending with a blank lower third. */}
      <div className="pad-card pad-card-y flex flex-1 flex-col pt-[clamp(0.5rem,1.4cqw,0.85rem)]">
        {children}
      </div>
    </section>
  )
}


/**
 * The answer, on its own full-width row.
 *
 * This is what the operator opened the page for, so it gets the whole width and
 * reads left to right as one sentence: what to do, how deep, how much water, how
 * long, and why. It was previously one card among four equals, which forced the
 * depth figure down to the size of the supporting readings beside it and left the
 * reasoning orphaned in another card.
 */
function DecisionHero({
  field,
  depth,
  band,
}: {
  field: Advisory
  depth: number
  band: Band
}) {
  const banner = bannerFor(field, depth)
  const style = BANNER_STYLE[banner]
  const minutes = Math.max(1, Math.round(field.recommendation.durationSeconds / 60))
  const depletion = field.water.depletionFraction
  const refillAt =
    field.water.tawMm > 0 ? Math.round((field.water.rawMm / field.water.tawMm) * 100) : 55
  const rain = field.water.rainfallMmDay

  return (
    <section
      className={`card relative overflow-hidden bg-gradient-to-br ${style.edge} via-white/70 to-white/40`}
    >
      <div className="pad-card pad-card-y flex flex-wrap items-start justify-between gap-x-8 gap-y-4">
        {/* What to do. The badge and the sentence are one unit. */}
        <div className="min-w-0">
          <span
            className={`inline-block rounded-pill px-2.5 py-1 text-[10px] font-bold uppercase tracking-[0.08em] ${style.badge} shadow-panel`}
          >
            {style.label}
          </span>
          <h1 className="type-headline mt-2 font-semibold leading-tight tracking-tight text-ink-900">
            {instruction(field, depth)}
          </h1>
          {depth > 0 && (
            <p className="tnum mt-1 text-[13px] text-ink-500">
              {int(field.recommendation.volumeLitres)} L · {minutes} min at the pump
            </p>
          )}
        </div>

        {/* How deep. The one figure large enough to read across a field. */}
        <div className="shrink-0 text-right">
          <div className="micro">Recommended depth</div>
          <div className="tnum type-display mt-0.5 font-semibold leading-none tracking-tight text-ink-900">
            {depth > 0 ? num(depth, 1) : '0'}
            <span className="type-value ml-1.5 font-medium text-ink-400">mm</span>
          </div>
        </div>
      </div>

      {/* The three inputs the number came from. Recessed into a panel so the row
          reads as evidence sitting under the answer rather than competing with
          it, then the sentence that ties them together. */}
      <div className="pad-card border-t border-ink-100/80 py-[clamp(0.6rem,1.6cqw,1rem)]">
        <div className="panel grid gap-x-5 gap-y-3 px-[clamp(0.7rem,2cqw,1.1rem)] py-[clamp(0.6rem,1.6cqw,0.9rem)] sm:grid-cols-3">
          <Input
            label="Soil"
            value={`${num(field.sensors.soilMoisture, 0)}%`}
            detail={`${Math.round(depletion * 100)}% used · refill at ${refillAt}%`}
            tone={band === 'dry' ? 'warn' : band === 'rest' ? 'idle' : 'healthy'}
          />
          <Input
            label="Crop need"
            value={`${num(field.water.etcMmDay, 1)} mm/d`}
            detail={`day ${field.ageDays} · ${field.stageLabel}`}
            tone="neutral"
          />
          <Input
            label="Rain today"
            value={`${num(rain, 1)} mm`}
            detail={rain > 0.05 ? 'counted against need' : 'none falling'}
            tone={rain > 0.05 ? 'healthy' : 'neutral'}
          />
        </div>
      </div>

      <p className="pad-card border-t border-ink-100/80 py-[clamp(0.6rem,1.6cqw,0.95rem)] text-[12px] leading-relaxed text-ink-600">
        {reason(field, band)}
        {agreementNote(field) && (
          <span className="text-ink-400"> {agreementNote(field)}</span>
        )}
      </p>
    </section>
  )
}

function Input({
  label,
  value,
  detail,
  tone,
}: {
  label: string
  value: string
  detail: string
  tone: 'neutral' | 'healthy' | 'warn' | 'idle'
}) {
  const color =
    tone === 'warn'
      ? 'text-status-warn'
      : tone === 'healthy'
        ? 'text-status-healthy'
        : tone === 'idle'
          ? 'text-ink-400'
          : 'text-ink-900'
  return (
    <div className="min-w-0">
      <dt className="micro truncate">{label}</dt>
      <dd className={`type-figure tnum mt-0.5 font-semibold ${color}`}>{value}</dd>
      <dd className="type-label mt-0.5 truncate text-ink-400">{detail}</dd>
    </div>
  )
}

/**
 * The seven readings from the soil sensor head.
 *
 * All seven the head reports are shown, each against the range that decides
 * something: moisture and pH against V1's sufficiency bands, EC against the
 * salinity threshold, and the three nutrients against their bands. Battery and
 * signal are reported too, as absent, because V2 has no physical node to report
 * them and an empty bar would imply a reading of zero.
 */
function SensorBoard({ field }: { field: Advisory }) {
  const s = field.sensors
  const moisture = s.soilMoisture
  const moistureGood = moisture >= MOISTURE_BAND[0] && moisture <= MOISTURE_BAND[1]
  const ph = s.soilPh
  const phGood = ph >= PH_BAND[0] && ph <= PH_BAND[1]
  const ec = s.ec
  const ecGood = ec <= 2.0

  return (
    <Card title="7-in-1 soil sensor" hint={relativeTime(s.recordedAt)} className="tint-mint">
      {/* Soil readings only. Air temperature and humidity are ambient and belong
          to the sky card; listing them here as well printed the same number in
          two places on one screen. */}
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Reading
          label="Moisture"
          value={num(moisture, 0)}
          unit="%"
          tone={moistureGood ? 'healthy' : moisture < MOISTURE_BAND[0] ? 'warn' : 'idle'}
          badge={moistureGood ? 'Ideal' : moisture < MOISTURE_BAND[0] ? 'Low' : 'Wet'}
        />
        <Reading
          label="EC"
          value={num(ec, 2)}
          unit="dS/m"
          tone={ecGood ? 'healthy' : 'warn'}
          badge={ecGood ? 'Safe' : 'Salty'}
        />
        <Reading
          label="pH"
          value={num(ph, 2)}
          tone={phGood ? 'healthy' : 'warn'}
          badge={phGood ? 'Optimal' : ph < PH_BAND[0] ? 'Acidic' : 'Alkaline'}
        />
        <Reading label="Light" value={int(s.lightIntensity)} unit="lx" tone="neutral" />
      </div>

      <div className="panel mt-3 grid gap-3 px-[clamp(0.6rem,1.8cqw,0.95rem)] py-[clamp(0.6rem,1.8cqw,0.95rem)] sm:grid-cols-3">
        <NutrientBar
          label="Nitrogen"
          value={s.nitrogen}
          low={NUTRIENT_BANDS.nitrogen[0]}
          mid={NUTRIENT_BANDS.nitrogen[1]}
          high={NUTRIENT_BANDS.nitrogen[2]}
          scaleMax={NUTRIENT_BANDS.nitrogen[3]}
        />
        <NutrientBar
          label="Phosphorus"
          value={s.phosphorus}
          low={NUTRIENT_BANDS.phosphorus[0]}
          mid={NUTRIENT_BANDS.phosphorus[1]}
          high={NUTRIENT_BANDS.phosphorus[2]}
          scaleMax={NUTRIENT_BANDS.phosphorus[3]}
        />
        <NutrientBar
          label="Potassium"
          value={s.potassium}
          low={NUTRIENT_BANDS.potassium[0]}
          mid={NUTRIENT_BANDS.potassium[1]}
          high={NUTRIENT_BANDS.potassium[2]}
          scaleMax={NUTRIENT_BANDS.potassium[3]}
        />
      </div>

      {/* Battery and signal are absent because there is no physical node. The
          earlier version gave them a ninth tile showing an em dash, which read as
          a broken reading rather than a field that was never instrumented. */}
      <p className="mt-3 text-[10px] leading-relaxed text-ink-400">
        Bands are V1&rsquo;s sufficiency thresholds. No battery or signal is reported: this
        deployment has no physical sensor head.
      </p>
    </Card>
  )
}

/**
 * How dry the soil is.
 *
 * A ring rather than a bar because the whole question is "how much of the
 * plantable water is left", and the refill point has to be visible on the same
 * scale as the level. The word dry/good/wet is the answer; the ring is the
 * evidence.
 */
function WetnessCard({
  field,
  remaining,
  band,
}: {
  field: Advisory
  remaining: number
  band: Band
}) {
  const style = BAND_STYLE[band]
  const radius = 58
  const circumference = 2 * Math.PI * radius
  const refill = field.water.tawMm > 0 ? Math.min(1, field.water.rawMm / field.water.tawMm) : 0.5
  const fill = Math.max(0, Math.min(1, remaining))
  const tint =
    band === 'dry'
      ? 'tint-dry'
      : band === 'wet'
        ? 'tint-wet'
        : band === 'rest'
          ? 'tint-rest'
          : 'tint-good'

  return (
    <Card title="Soil water" hint={`${num(field.sensors.soilMoisture, 0)}% moisture`} className={tint}>
      {/* Horizontal: the ring carries the proportion on the left and the verdict
          and numbers sit beside it. `flex-1` so a stretched card centres this row
          in the extra height rather than dropping its footnote into it. */}
      <div className="flex flex-1 items-center gap-3.5">
        {/* Sized from the card's own width so it cannot overflow a narrow column. */}
        <div className="relative aspect-square w-[38cqw] max-w-[116px] shrink-0">
          <svg viewBox="0 0 140 140" className="h-full w-full -rotate-90">
            <defs>
              <linearGradient id="ringFill" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor={RING_GRADIENT[band][0]} />
                <stop offset="100%" stopColor={RING_GRADIENT[band][1]} />
              </linearGradient>
            </defs>
            <circle cx="70" cy="70" r={radius} fill="none" stroke="#e2e8f0" strokeWidth="13" />
            <circle
              cx="70"
              cy="70"
              r={radius}
              fill="none"
              stroke="url(#ringFill)"
              strokeWidth="13"
              strokeLinecap="round"
              strokeDasharray={`${circumference * fill} ${circumference}`}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={`tnum text-[clamp(1rem,5cqw,1.35rem)] font-semibold leading-none ${style.text}`}>
              {Math.round(fill * 100)}%
            </span>
            <span className="mt-1 text-[9px] font-semibold uppercase tracking-[0.08em] text-ink-400">
              left
            </span>
          </div>
        </div>

        <div className="panel min-w-0 flex-1 px-[clamp(0.6rem,1.8cqw,0.95rem)] py-[clamp(0.55rem,1.6cqw,0.85rem)]">
          <div className={`text-[clamp(1.05rem,3.6cqw,1.35rem)] font-semibold leading-none ${style.text}`}>
            {style.word}
          </div>
          <dl className="mt-2 space-y-1 text-[11px] text-ink-700">
            <Line label="Used" value={`${num(field.water.depletionMm)} mm`} />
            <Line label="Available" value={`${num(field.water.tawMm)} mm`} />
            <Line label="Refill at" value={`${num(field.water.rawMm)} mm`} />
            <Line label="Roots" value={`${num(field.water.rootDepthCm, 0)} cm`} />
          </dl>
        </div>
      </div>

      <p className="mt-3 border-t border-ink-100 pt-2.5 text-[11px] leading-snug text-ink-400">
        {band === 'rest'
          ? 'No crop in the ground, so soil wetness does not drive a decision here.'
          : `Pump on at the refill point, ${Math.round(refill * 100)}% of available water.`}
      </p>
    </Card>
  )
}

/**
 * How old the crop is, and what it means for water.
 *
 * Water need rises and falls across a season, so the same soil serves a seedling
 * and a peak-demand plant differently. The track makes the position visible
 * rather than leaving "mid_season" to be interpreted.
 */
function AgeCard({ field }: { field: Advisory }) {
  const progress = Math.max(0, Math.min(1, field.stageProgress))
  const current = stageKey(field)
  const finished = current === 'post_harvest' || current === 'harvest'

  return (
    <Card title="Crop age" hint={field.crop} className="tint-violet flex-1 justify-between">
      <div className="flex items-end justify-between gap-4">
        <div>
          <span className="tnum text-[clamp(1.75rem,6cqw,2.4rem)] font-semibold leading-none tracking-tight text-ink-900">
            {field.ageDays}
          </span>
          <span className="ml-1.5 text-[12px] text-ink-500">days old</span>
        </div>
        <div className="text-right">
          <div className="type-value font-semibold text-ink-900">
            {finished ? 'Finished' : field.stageLabel}
          </div>
          <div className="tnum text-[10px] text-ink-400">
            {/* `stageProgress` is progress through the *current stage*, so it is
                labelled as such; season position comes from the day counts. */}
            {finished
              ? 'season complete'
              : `${Math.round(progress * 100)}% through this stage`}
          </div>
        </div>
      </div>

      <StageTrack field={field} progress={progress} current={current} finished={finished} />

      <div className="panel mt-2.5 grid grid-cols-3 gap-2 px-[clamp(0.6rem,1.8cqw,0.95rem)] py-[clamp(0.55rem,1.6cqw,0.85rem)]">
        {field.season && field.season.daysInSeason > 0 ? (
          <>
            <MiniStat label="Sun today" value={`${num(field.water.et0MmDay, 1)} mm`} />
            <MiniStat
              label="Season"
              value={`${field.ageDays}/${field.season.daysInSeason} d`}
            />
            <MiniStat
              label="Season water"
              value={`${num(field.season.seasonGrossMm, 0)} mm`}
            />
          </>
        ) : (
          <>
            <MiniStat label="Sun today" value={`${num(field.water.et0MmDay, 1)} mm`} />
            <MiniStat label="Roots" value={`${num(field.water.rootDepthCm, 0)} cm`} />
            <MiniStat label="Rain" value={`${num(field.water.rainfallMmDay, 1)} mm`} />
          </>
        )}
      </div>
    </Card>
  )
}

/**
 * Where the crop sits in its season, and what each stage still costs.
 *
 * The engine's own stage windows are used where available, because they are
 * derived from accumulated thermal time and carry the water cost of each stage.
 * The fallback track is the fixed FAO-56 progression, used when a sowing is too
 * new for windows to exist yet.
 */
function StageTrack({
  field,
  progress,
  current,
  finished,
}: {
  field: Advisory
  progress: number
  current: string
  finished: boolean
}) {
  const windows = field.season?.stageWindows ?? []

  /**
   * Where the crop actually sits on the track, in season-wide percent.
   *
   * Computed from day counts, never from `stageProgress`. That field is progress
   * *within the current stage* — 0.63 for a barley 55 days into an 83-day
   * mid-season — so comparing it against season-wide segment offsets pushed the
   * marker to the segment's left edge every time.
   */
  const seasonDays =
    field.season?.daysInSeason && field.season.daysInSeason > 0
      ? field.season.daysInSeason
      : null

  const marker = (() => {
    if (windows.length === 0 || !seasonDays) return null
    let cursor = 0
    for (const window of windows) {
      const days = Math.max(1, window.days)
      if (window.stage === current) {
        const within = (field.ageDays - cursor) / days
        return {
          percent:
            ((cursor + Math.max(0, Math.min(1, within)) * days) / seasonDays) * 100,
          within: Math.max(0, Math.min(1, within)),
        }
      }
      cursor += days
    }
    return null
  })()

  if (windows.length > 0) {
    const total = windows.reduce((sum, window) => sum + Math.max(1, window.days), 0)

    const segments: { stage: string; start: number; share: number }[] = []
    let cursor = 0
    for (const window of windows) {
      const share = (Math.max(1, window.days) / total) * 100
      segments.push({ stage: window.stage, start: cursor, share })
      cursor += share
    }

    const at = Math.min(99.2, Math.max(0.8, marker?.percent ?? 0))

    return (
      <div className="mt-4">
        {/* `relative` so the marker can sit on the track, `overflow-visible` so it
            is not clipped by the track's own rounded ends. */}
        <div className="relative">
          <div className="flex h-2 gap-0.5 overflow-hidden rounded-pill bg-ink-100">
            {segments.map((segment) => {
              const isCurrent = segment.stage === current
              const past = segment.start + segment.share <= progress
              return (
                <div
                  key={segment.stage}
                  title={`${titleCase(segment.stage)}`}
                  style={{ width: `${segment.share}%` }}
                  className={
                    isCurrent ? 'stage-live' : past ? 'bg-brand-100' : 'bg-ink-200/70'
                  }
                />
              )
            })}
          </div>

          {/* The marker. Drawn as a triangle rather than a glyph so it can be sized in
              pixels against the track instead of inheriting a font size, and
              given a white halo so it stays legible over the blue fill.

              `-top-[11px]` lifts it clear of the track. At `top-0` the 10px
              triangle overlapped the 8px bar almost entirely and its point sank
              into the blue, which read as a misalignment rather than a pointer.
              The hairline still crosses the bar and carries the position down to
              the stage labels. */}
          {marker && (
            <>
              <span
                aria-hidden="true"
                className="pointer-events-none absolute -top-[11px] z-20 -translate-x-1/2"
                style={{ left: `${at}%` }}
              >
                <svg width="15" height="11" viewBox="0 0 14 10" aria-hidden="true">
                  <path
                    d="M7 9.5 1 1.2h12L7 9.5Z"
                    fill="#0f172a"
                    stroke="#ffffff"
                    strokeWidth="1.4"
                    strokeLinejoin="round"
                  />
                </svg>
              </span>
              <span
                aria-hidden="true"
                className="pointer-events-none absolute top-0 z-10 h-5 w-0.5 -translate-x-1/2 rounded-full bg-ink-900/45"
                style={{ left: `${at}%` }}
              />
            </>
          )}
        </div>

        <div className="mt-2 flex justify-between gap-1">
          {windows.map((window) => (
            <span
              key={window.stage}
              className={`min-w-0 flex-1 truncate text-[10px] ${
                window.stage === current ? 'font-semibold text-brand-700' : 'text-ink-400'
              }`}
            >
              {titleCase(window.stage)}
            </span>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="mt-4">
      <div className="relative">
        <div className="flex h-2 gap-0.5 overflow-hidden rounded-pill bg-ink-100">
          {STAGES.map((stage) => (
            <div
              key={stage.key}
              style={{ width: `${(stage.to - stage.from) * 100}%` }}
              className={stage.key === current ? 'stage-live' : 'bg-ink-200/70'}
            />
          ))}
        </div>
        {!finished && (
          <>
            <span
              aria-hidden="true"
              className="pointer-events-none absolute -top-[11px] z-20 -translate-x-1/2"
              style={{ left: `${Math.min(99.2, Math.max(0.8, progress * 100))}%` }}
            >
              <svg width="15" height="11" viewBox="0 0 14 10" aria-hidden="true">
                <path
                  d="M7 9.5 1 1.2h12L7 9.5Z"
                  fill="#0f172a"
                  stroke="#ffffff"
                  strokeWidth="1.4"
                  strokeLinejoin="round"
                />
              </svg>
            </span>
            <span
              aria-hidden="true"
              className="pointer-events-none absolute top-0 z-10 h-5 w-0.5 -translate-x-1/2 rounded-full bg-ink-900/45"
              style={{ left: `${Math.min(99.2, Math.max(0.8, progress * 100))}%` }}
            />
          </>
        )}
      </div>
      <div className="relative mt-1.5 h-4">
        {STAGES.map((stage) => (
          <span
            key={stage.key}
            style={{ left: `${stage.from * 100}%` }}
            className={`absolute text-[10px] ${
              stage.key === current ? 'font-semibold text-brand-700' : 'text-ink-400'
            }`}
          >
            {stage.label}
          </span>
        ))}
      </div>
    </div>
  )
}

/**
 * Seven days of planned water, as a grower reads it off a diary.
 *
 * Drawn the way a bar-chart card reads: the range across the top so the week is
 * legible as a quantity, an average line through the bars so a single tall day
 * can be judged against the rest, and the days themselves listed underneath
 * rather than only drawn. A bar chart with no numbers is a picture, not a plan.
 */
function PlanCard({ field }: { field: Advisory }) {
  const scheduled = field.schedule.filter((day) => day.grossMm > 0)
  const peak = Math.max(...field.schedule.map((day) => day.grossMm), 0)
  const mean =
    scheduled.length > 0
      ? scheduled.reduce((sum, day) => sum + day.grossMm, 0) / scheduled.length
      : 0
  const total = field.schedule.reduce((sum, day) => sum + day.grossMm, 0)
  const depth = depthOf(field)

  // The engine schedules nothing when the root zone is already full, while the
  // models may still call for water inside the fortnight. Saying "0 planned"
  // beside a 10.8 mm recommendation reads as a broken card; naming the
  // disagreement reads as the thing it is.
  const unscheduled = total <= 0 && depth > 0

  return (
    <Card title="Next seven days" hint={`${scheduled.length} planned`} className="tint-sky">
      <div className="flex items-baseline gap-2">
        <span className="tnum text-[30px] font-semibold leading-none tracking-tight text-ink-900">
          {peak > 0 ? num(peak, 0) : '0'}
        </span>
        <span className="text-[19px] font-light leading-none text-ink-300">mm</span>
        <span className="ml-1 text-[11px] text-ink-400">largest single day</span>
      </div>

      {unscheduled ? (
        <div className="mt-4 rounded-card border border-brand-100 bg-brand-50/70 px-3 py-2.5">
          <p className="text-[11px] leading-relaxed text-brand-900">
            The root zone is still full, so the engine has nothing scheduled this week. The
            models still advise{' '}
            <span className="tnum font-semibold">{num(depth, 1)} mm</span> inside the
            fortnight — see the decision above.
          </p>
        </div>
      ) : (
        <>
          <div className="relative mt-4 h-[74px] min-h-[74px] flex-1">
            {mean > 0 && (
              <>
                <div
                  className="absolute inset-x-0 border-t border-dashed border-ink-300"
                  style={{ bottom: `${(mean / peak) * 100}%` }}
                />
                <span
                  className="absolute right-0 -translate-y-full rounded bg-ink-900 px-1.5 py-0.5 text-[9px] font-semibold text-white"
                  style={{ bottom: `calc(${(mean / peak) * 100}% + 2px)` }}
                >
                  avg {num(mean, 0)}
                </span>
              </>
            )}

            <div className="flex h-full items-end justify-between gap-1.5">
              {field.schedule.map((day) => {
                const height = peak > 0 ? (day.grossMm / peak) * 100 : 0
                const date = new Date(day.date)
                return (
                  <div key={day.date} className="flex flex-1 flex-col items-center gap-1">
                    <div className="flex h-full w-full items-end">
                      <div
                        className={`w-full rounded-t-[3px] ${
                          day.grossMm > 0
                            ? 'bg-gradient-to-t from-brand-700 to-brand-500'
                            : 'bg-ink-200/70'
                        }`}
                        style={{ height: `${Math.max(height, 3)}%` }}
                        title={`${date.toDateString()}: ${num(day.grossMm, 1)} mm gross`}
                      />
                    </div>
                    <span className="text-[9px] font-medium text-ink-400">
                      {date.toLocaleDateString([], { weekday: 'narrow' })}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>

          <dl className="mt-3 border-t border-ink-100 pt-1">
            {field.schedule.slice(0, 3).map((day) => {
              const date = new Date(day.date)
              return (
                <div
                  key={day.date}
                  className="flex items-baseline justify-between border-b border-ink-50 py-1.5 last:border-b-0"
                >
                  <dt className="text-[10px] text-ink-400">
                    {date.toLocaleDateString([], { weekday: 'short', day: 'numeric' })}
                  </dt>
                  <dd
                    className={`tnum text-[12px] font-semibold ${
                      day.grossMm > 0 ? 'text-ink-900' : 'text-ink-300'
                    }`}
                  >
                    {day.grossMm > 0 ? `${num(day.grossMm, 1)} mm` : '—'}
                  </dd>
                </div>
              )
            })}
          </dl>
        </>
      )}

      {/* `mt-auto`: when the chart is present the chart is `flex-1` and this
          already lands on the base; when the engine has scheduled nothing there
          is no chart, and without this the note sat under the message with the
          card's spare height left as a hole at the bottom. */}
      <p className="mt-auto pt-2.5 text-[10px] text-ink-400">
        Gross water at the pump, including application losses. Week total{' '}
        {num(total, 0)} mm.
      </p>
    </Card>
  )
}

/**
 * The whole farm's week.
 *
 * The plan and the soil trend are both single-field and read thin alone, so the
 * third slot answers the question neither can: what does this week cost the
 * farm, in water and in money-equivalent volume, and how do the fields compare.
 * Fleet figures come from the same advisory payload the decision above was
 * computed from, so the totals cannot disagree with the cards above.
 */
function BudgetCard({
  fields,
  totalMm,
}: {
  fields: Advisory[]
  totalMm: number
}) {
  const scheduled = fields.filter((item) => depthOf(item) > 0)
  const peak = fields.reduce((best, item) => Math.max(best, depthOf(item)), 0)
  const litres = fields.reduce((sum, item) => sum + item.recommendation.volumeLitres, 0)
  const minutes = Math.round(
    fields.reduce((sum, item) => sum + item.recommendation.durationSeconds, 0) / 60,
  )
  const totalArea = fields.reduce((sum, item) => sum + item.areaM2, 0)

  return (
    <Card title="Farm this week" hint={`${fields.length} fields`} className="tint-amber">
      <div className="panel flex items-end justify-between gap-3 px-[clamp(0.6rem,1.8cqw,0.95rem)] py-[clamp(0.55rem,1.6cqw,0.85rem)]">
        <div>
          <div className="micro">Recommended</div>
          <div className="tnum mt-1 text-[clamp(1.5rem,5cqw,1.95rem)] font-semibold leading-none tracking-tight text-ink-900">
            {num(totalMm, 1)}
            <span className="ml-1 text-[14px] font-medium text-ink-400">mm</span>
          </div>
        </div>
        <div className="text-right">
          <div className="micro">Pump time</div>
          <div className="tnum mt-1 text-[17px] font-semibold text-ink-900">
            {minutes}
            <span className="ml-1 text-[11px] font-medium text-ink-400">min</span>
          </div>
        </div>
      </div>

      <dl className="mt-2.5 grid grid-cols-2 gap-x-4 gap-y-1.5 text-[11px]">
        <Line label="Fields to water" value={`${scheduled.length} of ${fields.length}`} />
        <Line label="Total volume" value={`${int(litres)} L`} />
        <Line label="Largest single" value={peak > 0 ? `${num(peak, 1)} mm` : '—'} />
        <Line label="Area covered" value={`${num(totalArea / 10_000, 2)} ha`} />
      </dl>

      {/* Fields ranked by depth: the comparison the operator is actually after. */}
      <div className="mt-3 border-t border-ink-100 pt-2.5">
        <div className="micro mb-1.5">By depth</div>
        <div className="space-y-1.5">
          {[...fields]
            .sort((a, b) => depthOf(b) - depthOf(a))
            .map((item) => {
              const depth = depthOf(item)
              return (
                <div key={item.fieldId} className="flex items-center gap-2">
                  <span className="w-24 shrink-0 truncate text-[11px] text-ink-600">
                    {item.fieldId}
                  </span>
                  <div className="h-2 min-w-0 flex-1 overflow-hidden rounded-pill bg-ink-100 shadow-[inset_0_1px_2px_rgba(15,23,42,0.12)]">
                    <div
                      className={`h-full rounded-pill ${
                        depth === 0 ? 'bg-ink-200' : 'bg-gradient-to-r from-brand-600 to-brand-400'
                      }`}
                      style={{ width: `${peak > 0 ? (depth / peak) * 100 : 0}%` }}
                    />
                  </div>
                  <span
                    className={`tnum w-12 shrink-0 text-right text-[11px] font-semibold ${
                      depth > 0 ? 'text-ink-900' : 'text-ink-300'
                    }`}
                  >
                    {depth > 0 ? `${num(depth, 1)}` : '—'}
                  </span>
                </div>
              )
            })}
        </div>
      </div>
    </Card>
  )
}

function TrendCard({
  trend,
  band,
  refill,
}: {
  trend: { label: string; available: number }[]
  band: Band
  refill: number
}) {
  const color =
    band === 'dry' ? '#f59e0b' : band === 'wet' ? '#2563eb' : band === 'rest' ? '#94a3b8' : '#16a34a'

  /**
   * A y-domain fitted to the data instead of a fixed 0-100.
   *
   * The series sits wherever the soil happens to be — a field at 85% available
   * barely moves in a week — so a 0-100 axis compresses it into a few pixels at
   * the top and the card reads as an empty plot with a stray dashed line. The
   * domain still has to contain the refill point, because comparing the two is
   * the entire purpose of the chart, and it is padded so the line never lands on
   * the frame.
   */
  const domain = (() => {
    if (trend.length === 0) return [0, 100] as [number, number]
    const values = trend.map((point) => point.available)
    const low = Math.min(...values, refill)
    const high = Math.max(...values, refill)
    const pad = Math.max(6, (high - low) * 0.25)
    const min = Math.max(0, Math.floor((low - pad) / 5) * 5)
    const max = Math.min(100, Math.ceil((high + pad) / 5) * 5)
    return [min, max === min ? min + 10 : max] as [number, number]
  })()

  return (
    <Card title="Water in the soil" hint="last 7 days" className="tint-cyan">
      {trend.length === 0 ? (
        <p className="py-10 text-center text-[12px] text-ink-400">No readings yet.</p>
      ) : (
        <div className="h-[130px] min-h-[130px] flex-1">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={trend} margin={{ top: 6, right: 8, left: -26, bottom: 0 }}>
              <defs>
                {/* Opaque enough at the curve to actually read as an area; the
                    earlier 0.28-to-0.02 ramp assumed a full-height series and
                    left a nearly flat line looking like no line at all. */}
                <linearGradient id="soilFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.34} />
                  <stop offset="100%" stopColor={color} stopOpacity={0.06} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="2 4" stroke="#e2e8f0" vertical={false} />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 9, fill: '#94a3b8' }}
                axisLine={false}
                tickLine={false}
                interval="preserveStartEnd"
              />
              <YAxis
                domain={domain}
                tick={{ fontSize: 9, fill: '#94a3b8' }}
                axisLine={false}
                tickLine={false}
                width={44}
              />
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 10, border: '1px solid #e2e8f0' }}
              />
              <ReferenceLine
                y={refill}
                stroke="#f59e0b"
                strokeDasharray="4 4"
                label={{
                  value: 'refill',
                  position: 'insideBottomRight',
                  fill: '#b45309',
                  fontSize: 9,
                }}
              />
              <Area
                type="monotone"
                dataKey="available"
                name="Available"
                stroke={color}
                strokeWidth={2}
                fill="url(#soilFill)"
                dot={{ r: 2, fill: color, strokeWidth: 0 }}
                activeDot={{ r: 4 }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
      <p className="mt-2.5 text-[11px] text-ink-400">
        Share of plantable water still in the root zone. Below the refill line the crop
        needs water.
      </p>
    </Card>
  )
}

/**
 * Weather, as the reading an operator glances at before walking to a field.
 *
 * There is no forecast in this system, so the card says `now` on its face rather
 * than implying a prediction, and it carries nothing the page does not already
 * state: the previous version repeated rain and crop water use, which the
 * decision directly above already reports, and squeezed them into a narrow column
 * where the edge glyph ran through the text.
 *
 * One glyph, in the corner where nothing else is drawn.
 */
function SkyCard({ field }: { field: Advisory }) {
  const s = field.sensors
  const rain = field.water.rainfallMmDay
  const humid = s.humidity
  const wet = rain > 0.05 || humid > 78
  const temp = s.temperature

  const condition = wet
    ? 'Wet · hold off'
    : rain > 0.02
      ? 'Drizzle'
      : humid > 60
        ? 'Cloudy'
        : 'Clear'
  const Glyph = wet ? RainGlyph : SkyGlyph

  return (
    <section className="sky-wash card relative flex flex-col overflow-hidden text-white">
      {/* In the corner, behind nothing. */}
      <Glyph size={104} className="pointer-events-none absolute right-2 top-2 text-white/15" />

      <div className="pad-card relative flex flex-1 flex-col pad-card-y">
        <div className="flex items-baseline justify-between gap-3">
          <h2 className="type-label font-semibold uppercase tracking-[0.07em] text-white/60">
            {field.station} · now
          </h2>
          <span className="shrink-0 text-[10px] font-medium uppercase tracking-[0.07em] text-white/70">
            {condition}
          </span>
        </div>

        {/* Stacked, not side by side: a row of label/value pairs beside a large
            figure left each pair about forty pixels wide. */}
        <div className="mt-auto pt-3">
          <span className="tnum block text-[clamp(2rem,9cqw,2.9rem)] font-semibold leading-none tracking-tight">
            {num(temp, 0)}°
          </span>

          {/* Recessed with a translucent white so the group sits into the blue
              rather than floating on it. */}
          <dl className="mt-3 space-y-1 rounded-card border border-white/15 bg-white/10 px-[clamp(0.6rem,1.8cqw,0.95rem)] py-[clamp(0.5rem,1.5cqw,0.8rem)] text-[11px] shadow-[inset_0_1px_2px_rgba(15,23,42,0.18)]">
            {/* Humidity only. Air temperature is the headline above it and rain
                today is one of the three decision inputs further up the page, so
                listing either here just printed the same figure twice. */}
            <div className="flex items-baseline justify-between gap-3">
              <dt className="text-white/65">Humidity</dt>
              <dd className="tnum font-semibold">{num(humid, 0)}%</dd>
            </div>
          </dl>

          <p className="mt-2.5 text-[10px] leading-snug text-white/50">
            Observed at {field.station}. No forecast is stored.
          </p>
        </div>
      </div>
    </section>
  )
}

/**
 * The field record, as a horizontal strip.
 *
 * This was a fourth card in the top row, which meant seven short label/value
 * pills stacked inside a card sized by its neighbour and a large empty lower
 * half. Laid out as one strip it is a single line of reference facts, which is
 * what it always was.
 */
function FieldStrip({ field }: { field: Advisory }) {
  const riskTone =
    field.risk.riskLevel === 'risk' ? 'alert' : field.risk.riskLevel === 'watch' ? 'warn' : 'healthy'

  const facts: { label: string; value: string }[] = [
    { label: 'Area', value: `${num(field.areaM2 / 10_000, 2)} ha` },
    { label: 'Soil', value: titleCase(field.soilType) },
    { label: 'Station', value: field.station },
    { label: 'Irrigation', value: titleCase(field.method) },
    { label: 'Roots', value: `${num(field.water.rootDepthCm, 0)} cm` },
    { label: 'Sown', value: new Date(field.sowingDate).toLocaleDateString([], { day: 'numeric', month: 'short', year: 'numeric' }) },
  ]

  return (
    <section className="card pad-card tint-slate py-[clamp(0.7rem,1.8cqw,1rem)]">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2.5">
        <span className="type-label shrink-0 font-semibold uppercase tracking-[0.07em] text-ink-400">
          Field
        </span>
        {facts.map((fact) => (
          <span
            key={fact.label}
            className="panel min-w-0 px-[clamp(0.5rem,1.4cqw,0.75rem)] py-1"
          >
            <span className="block text-[9px] uppercase tracking-[0.06em] text-ink-400">
              {fact.label}
            </span>
            <span className="block truncate text-[12px] font-semibold text-ink-900">
              {fact.value}
            </span>
          </span>
        ))}
        <span className="panel ml-auto flex items-center gap-2 px-[clamp(0.5rem,1.4cqw,0.75rem)] py-1">
          <span className="text-[9px] uppercase tracking-[0.06em] text-ink-400">Disease</span>
          <ToneBadge tone={riskTone}>{titleCase(field.risk.diseaseRisk)}</ToneBadge>
          <span className="tnum text-[11px] text-ink-500">
            {Math.round((field.risk.diseaseConfidence ?? 0) * 100)}% · stress{' '}
            {num(field.risk.cropStressIndex, 2)} · {num(field.risk.yieldTPerHa, 1)} t/ha
          </span>
        </span>
      </div>
    </section>
  )
}

function Line({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <dt className="shrink-0 text-ink-500">{label}</dt>
      <dd className="tnum truncate text-right font-medium text-ink-900">{value}</dd>
    </div>
  )
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="tnum text-[15px] font-semibold text-ink-900">{value}</div>
      <div className="mt-0.5 text-[10px] uppercase tracking-[0.06em] text-ink-400">{label}</div>
    </div>
  )
}
