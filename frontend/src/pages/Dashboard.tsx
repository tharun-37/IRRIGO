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
import { api } from '../api/client'
import { useLive } from '../App'
import { usePollingFetch } from '../hooks/usePollingFetch'
import {
  GrainFilter,
  LeafGlyph,
  NutrientBar,
  PillSegment,
  RainGlyph,
  Reading,
  SkyGlyph,
  ToneBadge,
} from '../components/ui'
import { AddField } from '../components/AddField'
import { FarmerView, type FarmerChip, type FarmerField } from './FarmerView'
import { int, num, relativeTime, titleCase } from '../lib/format'
import type { Advisory, LiveWeather } from '../types'

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

/** True once the crop has been taken, when there is nothing left to water. */
function isFinished(field: Advisory): boolean {
  const stage = stageKey(field)
  return stage === 'post_harvest' || stage === 'harvest'
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
 * The instruction.
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
  // Farmer-first by default. The console is the same page with every reading
  // put back, so the switch is a depth control rather than a second product.
  const [view, setView] = useState<'farmer' | 'full'>('farmer')

  const advisor = usePollingFetch(useCallback(() => api.advisor(), []), bump, REFRESH)

  const feed = advisor.data
  const fields = useMemo(() => feed?.fields ?? [], [feed])
  const field = useMemo(
    () => fields.find((item) => item.fieldId === fieldId) ?? fields[0],
    [fields, fieldId],
  )

  // The sky card's live readings, when a live feed is configured. Polled far
  // less often than the advisory: the upstream observation is cached for ten
  // minutes server-side, so polling faster would only repeat the same number.
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
  const totalMm = fields.reduce((sum, item) => sum + depthOf(item), 0)
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
        <div className="pad-gutter mx-auto flex w-full max-w-[1728px] flex-wrap items-center gap-2 py-2.5">
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
          {/* The depth control. Two views over one payload: the farmer's answer
              and the full evidence. It sits with the other chrome so it reads as
              a way of looking at the field, not as a choice to make about it. */}
          <PillSegment
            options={[
              { label: 'Farmer', value: 'farmer' as const },
              { label: 'Console', value: 'full' as const },
            ]}
            value={view}
            onChange={setView}
            ariaLabel="View"
          />
          <button
            type="button"
            onClick={refresh}
            className="shrink-0 text-[11px] text-ink-400 hover:text-ink-700"
          >
            {feed ? relativeTime(feed.generatedAt) : '…'}
          </button>
          {/* Whole-farm status, in one line beside the switcher: how many of the
              fields need the farmer today. */}
          {farmRows.length > 0 && view === 'farmer' && (
            <span className="shrink-0 rounded-pill bg-white/70 px-2.5 py-1 text-[12px] font-semibold text-ink-700">
              {farmRows.filter((row) => row.action === 'water').length} of {farmRows.length}{' '}
              fields need water
            </span>
          )}
          {/* The crop-health desk lives on its own route; the farmer reaches it
              from the field view without hunting the analytical pages. */}
          {view === 'farmer' && (
            <Link
              to="/crop-health"
              className="shrink-0 rounded-pill bg-white/70 px-2.5 py-1 text-[12px] font-semibold text-ink-700 transition-colors hover:bg-white hover:text-brand-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-800"
            >
              Crop health
            </Link>
          )}
        </div>
      </header>

      <main className="scroll-smooth min-h-0 flex-1 overflow-y-auto lg:overflow-hidden">
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
            <div className="scroll-smooth pad-page mx-auto flex w-full max-w-[1728px] flex-col gap-[clamp(0.7rem,1.5vw,1.1rem)] lg:h-full lg:overflow-y-auto">
            {/* Farmer view and console are alternatives, not layers: one is
                unmounted while the other is read. Keeping both mounted and
                hiding one leaves two copies of the same sentence in the page,
                which is both a reading problem and a screen-reader one. */}
            {view === 'farmer' ? (
              <FarmerView
                field={field}
                fields={farmRows}
                depth={depth}
                band={soil.band}
                chips={farmerChips(field)}
                live={live.data?.station === field.station ? live.data : null}
                onSelect={setFieldId}
                onMore={() => setView('full')}
              />
            ) : (
              <>
            {/* ---- The answer, and the week's bill, side by side ---------- */}
            {/* ---- Who this advice is about, before the advice -------- */}
            <div className="shrink-0">
              <FieldDetails field={field} />
            </div>

            {/* What this field needs and what the whole farm needs are the same
                question at two scales, so they sit together on the row directly
                under the field details rather than the farm total being pushed
                to the bottom of the page under three charts. Half and half: the
                hero is the widest thing on the screen but it does not need full
                width to say "water 10.8 mm", and splitting the row means a 1080p
                screen shows the operator's whole decision without scrolling.

                `shrink-0` on the wrapper matters. A flex child that clips its own
                overflow (`overflow-hidden` on the hero and the sky card) loses
                its automatic minimum size, so the column was free to squash it
                below its content and the decision's detail lines were cut off. */}
            {/* Equal heights: the grid's default stretch, so the budget card ends on the
                hero's line instead of stopping short of it. The slack goes to
                the "by depth" bars, which spread into it rather than leaving a
                void under the card's last row. */}
            <div className="grid shrink-0 gap-[clamp(0.7rem,1.5vw,1.1rem)] lg:grid-cols-2">
              <DecisionHero field={field} depth={depth} band={soil.band} />
              <BudgetCard fields={fields} totalMm={totalMm} />
            </div>

            {/* ---- The evidence for it: three compact readings ---------- */}
            {/* Equal heights again, for the same reason: these three carry very
                different amounts of content, so whichever is tallest sets the
                row and the others have to absorb the difference. Each hands its
                slack to one bordered surface — the numbers panel, the humidity
                box, the season panel — so the extra height is filled instead of
                showing as blank card. On md the third card would otherwise sit
                alone in a half-width row, so it spans the row and the three
                read as two even bands. */}
            <div className="grid shrink-0 gap-[clamp(0.7rem,1.5vw,1.1rem)] md:grid-cols-2 xl:grid-cols-3">
              <WetnessCard field={field} remaining={soil.remaining} band={soil.band} />
              <SkyCard
            field={field}
            // Guarded on the station, because switching fields leaves the
            // previous station's observation in the hook until the new fetch
            // lands, and one field's sky must never be shown against another's.
            live={live.data?.station === field.station ? live.data : null}
          />
              <div className="md:col-span-2 xl:col-span-1">
                <AgeCard field={field} />
              </div>
            </div>

            {/* The sensor head, given its own row: seven readings is the most
                looked-at data on the page and it was buried in small text. */}
            <div className="shrink-0">
              <SensorBoard field={field} />
            </div>

            {/* Disease and yield: what the crop is fighting, and what it will give.
                Farmer-first: these sit below the water decision so the operator
                reads "what to do" before "why the crop looks like it does". */}
            <div className="shrink-0">
              <div className="grid shrink-0 gap-[clamp(0.7rem,1.5vw,1.1rem)] lg:grid-cols-2">
                <DiseaseCard field={field} />
                <YieldCard field={field} />
              </div>
            </div>
              </>
            )}
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
  // `h-full` matters for the card that is not a direct grid item: crop age sits
  // in a wrapper so it can span two columns on md, and the grid stretches that
  // wrapper without stretching the card inside it, which left it the shortest
  // card in the row. Against an auto-height parent the percentage resolves to
  // auto, so this is a no-op everywhere else.
  return (
    <section className={`card flex h-full flex-col ${className}`}>
      <div className="pad-card flex items-baseline justify-between gap-2 pt-[clamp(0.75rem,2cqw,1.1rem)]">
        {/* Title and hint are `ink-700` rather than `ink-500`: both are 10-11px
            uppercase, and at this size the lighter ink sits close to the 4.5:1
            threshold on a white plate. The hierarchy between a title and its hint
            is already carried by size, weight and case. */}
        <h2 className="type-label font-semibold uppercase tracking-[0.07em] text-ink-700">
          {title}
        </h2>
        {hint && <span className="type-label shrink-0 text-ink-500">{hint}</span>}
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
  const extra = alsoToday(field)

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
            it, then the sentence that ties them together. No divider above
            either: a `border-t` with margin above and padding below is what put
            a band of empty space through the middle of this card. The panel and
            the sentence are separated by their own spacing. */}
        <div className="pad-card py-[clamp(0.6rem,1.6cqw,1rem)]">
        {/* Three across, but only once the hero is actually in a wide column.
            `sm:` is a viewport breakpoint, and the hero now shares a row with the
            farm total - so on a desktop at `sm` the three inputs were being laid
            across half the screen, roughly 150px each, which is where the cramped
            half-row text came from. Keyed to `lg`, the same breakpoint that splits
            the row in the first place, so the two cannot disagree. */}
        <div className="panel grid gap-x-5 gap-y-3 px-[clamp(0.7rem,2cqw,1.1rem)] py-[clamp(0.6rem,1.6cqw,0.9rem)] lg:grid-cols-3">
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

      <p className="pad-card py-[clamp(0.6rem,1.6cqw,0.95rem)] text-[12px] leading-relaxed text-ink-600">
        {reason(field, band)}
        {agreementNote(field) && (
          <span className="text-ink-400"> {agreementNote(field)}</span>
        )}
        {/* The rest of today's work, in the same sentence. The water decision
            is the headline, but fertilizer and disease are also today's work,
            and an operator who has to scroll to the bottom of the page to find
            out whether to spray has already been asked the wrong question. */}
        {extra && <span className="text-ink-700"> {extra}</span>}
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
    <Card title="7-in-1 soil sensor" hint={relativeTime(s.recordedAt)}>
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

  return (
    <Card title="Soil water" hint={`${num(field.sensors.soilMoisture, 0)}% moisture`}>
      {/* `flex-1` so this row, and not the card's own background, absorbs the height
          the tallest neighbour in the row imposes. `items-stretch` hands that
          height to the numbers panel; the ring stays its own size and centres. */}
      <div className="flex flex-1 items-stretch gap-3.5">
        {/* Sized from the card's own width so it cannot overflow a narrow column. */}
        <div className="relative aspect-square w-[38cqw] max-w-[116px] shrink-0 self-center">
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

        <div className="panel flex min-w-0 flex-1 flex-col px-[clamp(0.6rem,1.8cqw,0.95rem)] py-[clamp(0.55rem,1.6cqw,0.85rem)]">
          <div className={`text-[clamp(1.05rem,3.6cqw,1.35rem)] font-semibold leading-none ${style.text}`}>
            {style.word}
          </div>
          {/* `justify-between` so the four readings spread down the panel when it
              is stretched, instead of clustering at the top under empty panel. */}
          <dl className="mt-2 flex flex-1 flex-col justify-between gap-1 text-[11px] text-ink-700">
            <Line label="Used" value={`${num(field.water.depletionMm)} mm`} />
            <Line label="Available" value={`${num(field.water.tawMm)} mm`} />
            <Line label="Refill at" value={`${num(field.water.rawMm)} mm`} />
            <Line label="Roots" value={`${num(field.water.rootDepthCm, 0)} cm`} />
          </dl>
        </div>
      </div>

      {/* No rule above: the border and the padding under it were the empty band this
            footnote used to sit below. */}
        <p className="mt-2 text-[11px] leading-snug text-ink-400">
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
    <Card title="Crop age" hint={field.crop}>
      {/* `min-w-0` on the stage side: a long label like "Mid-season (peak demand)"
          then wraps inside the column instead of running into the age figure,
          which is the one thing here that must not be broken up. */}
      <div className="flex items-end justify-between gap-4">
        <div className="shrink-0">
          <span className="tnum text-[clamp(1.75rem,6cqw,2.4rem)] font-semibold leading-none tracking-tight text-ink-900">
            {field.ageDays}
          </span>
          <span className="ml-1.5 text-[12px] text-ink-500">days old</span>
        </div>
        <div className="min-w-0 text-right">
          <div className="type-value break-words font-semibold text-ink-900">
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

      {/* The season panel takes the height a taller neighbour imposes, so this card
          ends on the row's line with its own surface filling the difference. */}
      <div className="panel mt-2.5 grid flex-1 grid-cols-3 gap-2 px-[clamp(0.6rem,1.8cqw,0.95rem)] py-[clamp(0.55rem,1.6cqw,0.85rem)]">
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
 * The whole farm's week, beside the decision above it.
 *
 * This answers the question the per-field cards structurally cannot: what does
 * this week cost the farm in water, in pump time, and how do the fields compare
 * with one another. It shares the top row with the decision because those are the
 * same question at two scales - what this field needs, and what the farm needs -
 * and answering them on opposite ends of the page made the operator scroll between
 * them to work out whether the recommendation was affordable.
 *
 * Fleet figures come from the same advisory payload the decision above was computed
 * from, so the total cannot disagree with the card beside it.
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
    <Card title="Farm this week" hint={`${fields.length} fields`}>
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

{/* Fields ranked by depth: the comparison the operator is actually after.
            This block takes whatever height the hero beside it imposes, so the two
            cards end on one line and the bars spread into the difference rather
            than the card ending with a void under the last one. No rule above it:
            the border, and the padding under it, were the blank band above
            "by depth". */}
        <div className="mt-2 flex flex-1 flex-col">
          <div className="micro mb-1.5 shrink-0">By depth</div>
        <div className="flex flex-1 flex-col justify-between gap-1.5">
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

/**
 * Weather, as the reading an operator glances at before walking to a field.
 *
 * There is no forecast in this system, so the card says `now` on its face rather
 * than implying a prediction, and it carries nothing the page does not already
 * state: the previous version repeated rain and crop water use, which the
 * decision directly above already reports, and squeezed them into a narrow column
 * where the edge glyph ran through the text.
 *
 * Two sources, and the card says which one it is showing. The live feed is the
 * sky this minute; the plan's own readings are the weather the crop was planned
 * against, replayed from the NASA POWER corpus. When the live feed is
 * unavailable the card falls back to the plan's readings and says so, rather
 * than showing an empty box or an error the operator cannot act on.
 *
 * One glyph, in the corner where nothing else is drawn.
 */
function SkyCard({ field, live }: { field: Advisory; live: LiveWeather | null }) {
  const s = field.sensors
  const isLive = Boolean(live?.available)
  const rain = isLive ? (live?.rainfallMm ?? 0) : field.water.rainfallMmDay
  const humid = isLive ? (live?.humidityPct ?? 0) : s.humidity
  const wet = rain > 0.05 || humid > 78
  const temp = isLive ? (live?.temperatureC ?? s.temperature) : s.temperature

  const condition = isLive
    ? (live?.condition ? titleCase(live.condition) : wet ? 'Wet' : 'Clear')
    : wet
      ? 'Wet · hold off'
      : rain > 0.02
        ? 'Drizzle'
        : humid > 60
          ? 'Cloudy'
          : 'Clear'
  const Glyph = wet ? RainGlyph : SkyGlyph

  return (
    <section className="sky-wash card relative flex h-full flex-col overflow-hidden text-white">
      {/* In the corner, behind nothing. */}
      <Glyph size={104} className="pointer-events-none absolute right-2 top-2 text-white/15" />

        {/* Top-aligned, deliberately. `justify-center` here was tried and is wrong: it
            pushed the header away from the top edge, and because the card clips
            its overflow, a card whose content outgrows it would lose the header
            off the top. Whatever height this card inherits from its taller
            neighbours goes to the humidity box instead, so the surface grows
            rather than the card opening a band above its own first line. */}
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
              figure left each pair about forty pixels wide. The temperature is the
              card's headline and is sized from the card's own width. */}
          <div className="flex flex-1 flex-col pt-3">
            <span className="tnum block shrink-0 text-[clamp(2.25rem,15cqw,4.25rem)] font-semibold leading-none tracking-tight">
              {num(temp, 0)}°
            </span>

            {/* Two readings, because one could not fill this card. The neighbours
                either side carry a four-figure panel and a season track, so a
                lone humidity row left this one short and the spare height had to
                go somewhere visible. Rain today is a condition of the weather in
                its own right rather than a decision input, which is why it reads
                here as well as in the inputs above. */}
            <dl className="mt-3 flex flex-1 flex-col justify-center gap-1.5 rounded-card border border-white/15 bg-white/10 px-[clamp(0.6rem,1.8cqw,0.95rem)] py-[clamp(0.5rem,1.5cqw,0.8rem)] text-[11px] shadow-[inset_0_1px_2px_rgba(15,23,42,0.18)]">
              <div className="flex items-baseline justify-between gap-3">
                <dt className="text-white/65">Humidity</dt>
                <dd className="tnum font-semibold">{num(humid, 0)}%</dd>
              </div>
              <div className="flex items-baseline justify-between gap-3">
                <dt className="text-white/65">Rain today</dt>
                <dd className="tnum font-semibold">{num(rain, 1)} mm</dd>
              </div>
            </dl>
          </div>

          <p className="mt-2.5 shrink-0 text-[10px] leading-snug text-white/50">
            {isLive
              ? `Live now at ${field.station}. No forecast is stored.`
              : `From the plan's own weather at ${field.station}. No forecast is stored.`}
          </p>
        </div>
    </section>
  )
}

/**
 * The field record, as the first thing on the page.
 *
 * This was a strip of six pills sitting below the readings, which read as a
 * footnote to the advice instead of the identity of the field the advice is
 * about. Moved to the top, and cut back to the four facts nothing else states.
 *
 * Each fact is a recessed panel rather than a label and a value on one line.
 * Four pairs spread across a wide screen put every value four hundred pixels
 * from its own label, which left the strip too thin to read as anything; a cell
 * carries its label above its value and gives the band some body. It is the same
 * shape as the four sensor panels further down the page.
 */
function FieldDetails({ field }: { field: Advisory }) {
  // Only the physical basics, and only the ones nothing else on this screen
  // prints. The field name and crop are on the switcher and the crop-age card,
  // the station is the weather card's title, day and stage are on the hero's
  // crop-need line, roots are in the soil water card, the season figures are in
  // the crop-age panel, and the risk and yield numbers were a list nobody acted
  // on. What is left is four facts this is the only place to read.
  const rows: { label: string; value: string }[] = [
    { label: 'Soil', value: titleCase(field.soilType) },
    { label: 'Area', value: `${num(field.areaM2 / 10_000, 2)} ha` },
    { label: 'Irrigation', value: titleCase(field.method) },
    {
      label: 'Sown',
      value: new Date(field.sowingDate).toLocaleDateString([], {
        day: 'numeric',
        month: 'short',
        year: 'numeric',
      }),
    },
  ]

  return (
    <section className="card pad-card">
      <h2 className="type-label font-semibold uppercase tracking-[0.07em] text-ink-500">
        Field details
      </h2>

      {/* A recessed panel per fact, label above value. As one line per fact across a
          wide screen the value ended up hundreds of pixels from its own label,
          which is what made the strip look like a thin rule rather than a
          record. Same shape as the four sensor panels lower down the page. */}
      <dl className="mt-2.5 grid gap-[clamp(0.4rem,0.9vw,0.7rem)] sm:grid-cols-2 lg:grid-cols-4">
        {rows.map((row) => (
          <div
            key={row.label}
            className="panel min-w-0 px-[clamp(0.6rem,1.8cqw,0.95rem)] py-[clamp(0.5rem,1.4cqw,0.75rem)]"
          >
            <dt className="micro truncate">{row.label}</dt>
            <dd className="mt-0.5 truncate text-[clamp(0.8125rem,1.5cqw,0.9375rem)] font-semibold text-ink-900">
              {row.value}
            </dd>
          </div>
        ))}
      </dl>
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
    // `h-full justify-center`: the panel this sits in grows to match a taller
    // neighbour, and each stat stays centred in its column rather than hanging
    // from the top of it.
    <div className="flex h-full flex-col justify-center">
      <div className="tnum text-[15px] font-semibold text-ink-900">{value}</div>
      <div className="mt-0.5 text-[10px] uppercase tracking-[0.06em] text-ink-400">{label}</div>
    </div>
  )
}

/**
 * Disease & crop health: what the crop is fighting and what to do about it.
 *
 * The disease block is the engine's own output, so the operator sees the same
 * verdict the plan was computed from. The class distribution comes with it,
 * because a single label hides the decision: 62% rust against 35% blight is a
 * spray, the same 62% against 2% blight is not.
 *
 * Fertilizer advice is the engine's NPK dose read against the live soil
 * readings, converted to the product the operator would actually buy. The card
 * ends in an action, not just a number.
 */
function DiseaseCard({ field }: { field: Advisory }) {
  const risk = field.risk
  const level = risk.riskLevel ?? 'healthy'
  const confidence = Math.round((risk.diseaseConfidence ?? 0) * 100)
  const stress = risk.cropStressIndex ?? 0
  const s = field.sensors

  const fertilizer = fertilizerAdvice(s, risk.nutrientDemandKgHa)
  const pesticide = pesticideAdvice(risk)
  const action = diseaseAction(level, fertilizer, pesticide)

  const tone = level === 'risk' ? 'alert' : level === 'watch' ? 'warn' : 'healthy'
  const label =
    level === 'risk' ? 'Disease risk' : level === 'watch' ? 'Disease watch' : 'Healthy'

  // The winning class and the one behind it. With no distribution the winning
  // label still stands; the runner-up simply has nothing to compare against.
  const ranked = rankedDiseases(risk.diseaseProbabilities)
  const likely = risk.diseaseRisk ?? ranked[0]?.label ?? null
  const runnerUp = ranked[1] ?? null
  const close = runnerUp !== null && ranked[0].probability - runnerUp.probability < 0.15

  return (
    // The hint carries the caveat, because the card names a disease and a
    // farmer who reads "Rust" as a confirmed diagnosis will act on it. The
    // classifier is a stress-risk model over sensor telemetry; the crop name is
    // already in the decision card above and in the yield card beside this one.
    <Card title="Disease & crop health" hint="Risk indicator">
      <div className="flex items-baseline justify-between gap-3">
        <ToneBadge tone={tone}>{label}</ToneBadge>
        <span className="tnum text-[12px] font-semibold text-ink-900">
          {confidence}% confidence
        </span>
      </div>

      <div className="mt-3 panel grid gap-[clamp(0.4rem,0.9vw,0.7rem)] sm:grid-cols-2 lg:grid-cols-4">
        <div className="min-w-0">
          <dt className="micro truncate">Most likely</dt>
          <dd className="mt-0.5 truncate text-[clamp(0.8125rem,1.5cqw,0.9375rem)] font-semibold text-ink-900">
            {likely ? titleCase(likely) : '—'}
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="micro truncate">Next possibility</dt>
          <dd className="mt-0.5 truncate text-[clamp(0.8125rem,1.5cqw,0.9375rem)] font-semibold text-ink-900">
            {runnerUp
              ? `${titleCase(runnerUp.label)} ${Math.round(runnerUp.probability * 100)}%`
              : '—'}
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="micro truncate">Crop stress index</dt>
          <dd className="mt-0.5 truncate text-[clamp(0.8125rem,1.5cqw,0.9375rem)] font-semibold text-ink-900">
            {num(stress, 2)}
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="micro truncate">Pesticide</dt>
          <dd className="mt-0.5 truncate text-[clamp(0.8125rem,1.5cqw,0.9375rem)] font-semibold text-ink-900">
            {pesticide.label}
          </dd>
        </div>
      </div>

      {/* The dose the engine asks for, so "fertilizer" is a quantity to buy
          rather than a direction. */}
      <div className="mt-2.5 panel grid gap-[clamp(0.4rem,0.9vw,0.7rem)] sm:grid-cols-3">
        <div className="min-w-0">
          <dt className="micro truncate">Nitrogen</dt>
          <dd className="mt-0.5 truncate text-[clamp(0.8125rem,1.5cqw,0.9375rem)] font-semibold text-ink-900">
            {num(risk.nutrientDemandKgHa?.nitrogen ?? 0, 0)} kg/ha
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="micro truncate">Phosphorus</dt>
          <dd className="mt-0.5 truncate text-[clamp(0.8125rem,1.5cqw,0.9375rem)] font-semibold text-ink-900">
            {num(risk.nutrientDemandKgHa?.phosphorus ?? 0, 0)} kg/ha
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="micro truncate">Potassium</dt>
          <dd className="mt-0.5 truncate text-[clamp(0.8125rem,1.5cqw,0.9375rem)] font-semibold text-ink-900">
            {num(risk.nutrientDemandKgHa?.potassium ?? 0, 0)} kg/ha
          </dd>
        </div>
      </div>

      <p className="mt-2.5 text-[11px] leading-snug text-ink-600">
        {action}
        {close && runnerUp ? ` ${titleCase(runnerUp.label)} is close behind — check the field before spraying.` : ''}
      </p>
    </Card>
  )
}

/**
 * The rest of today's work, in one clause.
 *
 * Water is the headline, but fertilizer and disease are also today's work, and
 * an operator who has to scroll to the bottom of the page to find out whether
 * to spray has already been asked the wrong question. Returns nothing when
 * there is nothing extra to do, so a healthy field is not padded with advice
 * the operator does not need.
 */
function alsoToday(field: Advisory): string {
  const fertilizer = fertilizerAdvice(field.sensors, field.risk.nutrientDemandKgHa)
  const level = field.risk.riskLevel ?? 'healthy'

  const parts: string[] = []
  if (fertilizer.tone === 'warn') {
    parts.push(`apply fertilizer for ${fertilizer.shortages.join(' and ').toLowerCase()}`)
  }
  if (level === 'risk') parts.push('spray a fungicide')
  else if (level === 'watch') parts.push('scout the field for disease')

  if (parts.length === 0) return ''
  return `Also today: ${parts.join(' and ')}.`
}

/** Square metres in an acre, for the farmer view's area conversion. */
const ACRE_M2 = 4046.86

/**
 * The farmer view's three chips: fertiliser, spray, yield.
 *
 * Built from the same helpers the console cards use, so the two views cannot
 * disagree about whether a field needs urea or a fungicide. What changes is the
 * unit: the console reports the nutrient dose, the farmer buys bags, and the
 * yield is quoted per acre because that is the land measure the work is done in.
 */
function farmerChips(field: Advisory): FarmerChip[] {
  const acres = field.areaM2 / ACRE_M2
  const shortages = nutrientShortages(field.sensors)
  const fertilizer = fertilizerAdvice(field.sensors, field.risk.nutrientDemandKgHa)
  const pesticide = pesticideAdvice(field.risk)

  let fertilizerChip: FarmerChip
  const shortage = shortages[0]
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
      detail: `${num(productKgAcre, 0)} kg/acre · ${shortages
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
 * The action is the same rule the hero uses — a depth on the recommendation
 * means water today — so the row's colour and the headline cannot disagree.
 * Rows are ordered by urgency rather than by the feed's own order, because the
 * point of the list is to show the farmer which field to walk to first.
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

/**
 * Yield outlook: what the crop is expected to produce and what is holding it back.
 *
 * Expected yield is shown against the crop's own potential from the engine's
 * catalogue, so the gap is the opportunity the operator is leaving on the
 * table. The limiting factor — disease, water, nutrients, soil or nothing —
 * says where to act first.
 *
 * The gap is only measured when the yield model was actually trained on the
 * crop. A crop the model does not know still gets a number, but a generic one,
 * and setting a generic estimate against a crop-specific ceiling would report
 * a gap that is an artefact rather than an opportunity.
 */
function YieldCard({ field }: { field: Advisory }) {
  const yieldTPerHa = field.risk.yieldTPerHa ?? 0
  const potentialTPerHa = field.risk.yieldPotentialTPerHa ?? 0
  const stress = field.risk.cropStressIndex ?? 0
  const depletion = field.water.depletionFraction
  const riskLevel = field.risk.riskLevel ?? 'healthy'
  const modelled = field.risk.yieldModelled ?? false

  // The gap is only meaningful against a known ceiling, and only when the
  // prediction is on the same scale as that ceiling.
  const comparable = modelled && potentialTPerHa > 0
  const gap = comparable ? Math.max(0, potentialTPerHa - yieldTPerHa) : null
  const gapPct = gap !== null ? Math.round((gap / potentialTPerHa) * 100) : null

  const limiting = yieldLimitingFactor({
    riskLevel,
    shouldIrrigate: field.recommendation.shouldIrrigate,
    depletion,
    nutrientsLow: nutrientShortages(field.sensors).length > 0,
    stress,
    gapPct,
    modelled,
  })

  return (
    <Card title="Yield outlook" hint={field.crop}>
      <div className="flex items-baseline justify-between gap-3">
        <div className="shrink-0">
          <span className="tnum text-[clamp(1.75rem,6cqw,2.4rem)] font-semibold leading-none tracking-tight text-ink-900">
            {num(yieldTPerHa, 1)}
          </span>
          <span className="ml-1.5 text-[12px] text-ink-500">
            {modelled ? 't/ha expected' : 't/ha estimate'}
          </span>
        </div>
        <div className="text-right">
          <div className="type-value font-semibold text-ink-900">
            {potentialTPerHa > 0 ? num(potentialTPerHa, 1) : '—'}
          </div>
          <div className="tnum text-[10px] text-ink-400">potential t/ha</div>
        </div>
      </div>

      <div className="mt-3 panel grid gap-[clamp(0.4rem,0.9vw,0.7rem)] sm:grid-cols-2 lg:grid-cols-3">
        <div className="min-w-0">
          <dt className="micro truncate">Limiting factor</dt>
          <dd className="mt-0.5 truncate text-[clamp(0.8125rem,1.5cqw,0.9375rem)] font-semibold text-ink-900">
            {limiting.label}
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="micro truncate">Yield gap</dt>
          <dd className="mt-0.5 truncate text-[clamp(0.8125rem,1.5cqw,0.9375rem)] font-semibold text-ink-900">
            {gapPct === null ? '—' : `${gapPct}%`}
          </dd>
        </div>
        <div className="min-w-0">
          <dt className="micro truncate">Stress index</dt>
          <dd className="mt-0.5 truncate text-[clamp(0.8125rem,1.5cqw,0.9375rem)] font-semibold text-ink-900">
            {num(stress, 2)}
          </dd>
        </div>
      </div>

      <p className="mt-2.5 text-[11px] leading-snug text-ink-600">
        {yieldAdvice({
          yieldTPerHa,
          limiting: limiting.label,
          gap,
          gapPct,
          modelled,
          crop: field.crop,
          potentialTPerHa,
        })}
      </p>
    </Card>
  )
}

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
 * card uses.
 *
 * Shared rather than duplicated so the yield card and the fertilizer card can
 * never disagree: a field cannot be short of nutrients on one card and
 * adequately fed on the other.
 */
function nutrientShortages(s: Advisory['sensors']): (keyof typeof NUTRIENT_BANDS)[] {
  return (['nitrogen', 'phosphorus', 'potassium'] as const).filter(
    (nutrient) => s[nutrient] < NUTRIENT_BANDS[nutrient][0],
  )
}

/**
 * Fertilizer advice from live nutrient readings against V1 sufficiency bands.
 *
 * A shortfall is reported twice: the nutrient that is short, and the dose of
 * the product that covers it. The dose comes from the engine's own nutrient
 * plan, so it is the same number the agronomy was computed from.
 */
function fertilizerAdvice(
  s: Advisory['sensors'],
  demand: Advisory['risk']['nutrientDemandKgHa'],
): { tone: 'healthy' | 'warn' | 'alert'; shortages: string[]; dose: string } {
  const shortages: string[] = []
  const doses: string[] = []

  for (const nutrient of nutrientShortages(s)) {
    const kgHa = demand?.[nutrient] ?? 0
    const product = FERTILIZER_PRODUCTS[nutrient]
    const symbol = NUTRIENT_SYMBOLS[nutrient]
    shortages.push(titleCase(nutrient))
    // The product dose only makes sense when there is a dose to apply; a
    // shortfall with no plan behind it is still worth naming.
    doses.push(
      kgHa > 0
        ? `${num(kgHa, 0)} kg/ha ${symbol} (about ${num(kgHa / product.fraction, 0)} kg/ha ${product.name})`
        : `${num(kgHa, 0)} kg/ha ${symbol}`,
    )
  }

  if (shortages.length === 0) {
    return { tone: 'healthy', shortages, dose: '' }
  }
  return { tone: 'warn', shortages, dose: doses.join(' · ') }
}

/**
 * The disease classes in likelihood order.
 *
 * Sorted here rather than trusted from the wire, because the order is the
 * point: the runner-up is only a runner-up if the list is sorted.
 */
function rankedDiseases(
  probabilities: Record<string, number> | undefined,
): { label: string; probability: number }[] {
  if (!probabilities) return []
  return Object.entries(probabilities)
    .map(([label, probability]) => ({ label, probability: Number(probability) || 0 }))
    .sort((a, b) => b.probability - a.probability)
}

/**
 * Pesticide advice from the disease risk level.
 *
 * A `risk` verdict always asks for a fungicide, whatever the model's
 * confidence. The confidence is shown beside the advice so the operator can
 * weigh it, but a card that says "Disease risk" in one line and "No pesticide
 * needed" in the next is a contradiction the operator has to resolve alone —
 * and "no pesticide needed" on a risky crop is the expensive way to be wrong.
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
 * Single plain-language action for the disease card footer.
 *
 * The order is the order of urgency: the disease decides whether to spray, the
 * nutrients decide what to apply, and the sentence names both in the words the
 * operator would use with a shopkeeper.
 */
function diseaseAction(
  level: string,
  fertilizer: { tone: 'healthy' | 'warn' | 'alert'; shortages: string[]; dose: string },
  pesticide: { label: string; tone: 'healthy' | 'warn' | 'alert' },
): string {
  const parts: string[] = []

  // At `risk` the fungicide *is* the treatment, so naming both would repeat
  // itself: "treat the disease now, spray a fungicide".
  if (level === 'risk' && pesticide.tone !== 'alert') parts.push('treat the disease now')
  else if (level === 'watch') parts.push('scout the field this week')

  if (pesticide.tone === 'alert') parts.push('spray a fungicide')
  else if (pesticide.tone === 'warn') parts.push('keep a fungicide ready')

  const fertilizerSentence =
    fertilizer.tone === 'warn'
      ? `Apply fertilizer for ${fertilizer.shortages.join(' and ').toLowerCase()}: ${fertilizer.dose}.`
      : 'Nutrients adequate — no fertilizer needed.'

  if (parts.length === 0) {
    return fertilizer.tone === 'warn'
      ? `No disease action needed. ${fertilizerSentence}`
      : `Crop is healthy — no disease or fertilizer action needed. ${fertilizerSentence}`
  }
  return `Act now: ${parts.join(', ')}. ${fertilizerSentence}`
}

/**
 * What is holding yield back: disease, water, nutrients, soil or nothing.
 *
 * The order is the order of what a farmer can act on today, and each test
 * reads the same signal the matching card headlines. Water is the engine's
 * own irrigation decision rather than the moisture band on its own, because
 * "the soil is on the dry side" and "irrigate today" are different statements
 * and the dashboard must not make both about the same field.
 *
 * With no actionable driver, the label depends on the gap. Calling a field
 * "near potential" while it sits 45% below its ceiling is the kind of
 * contradiction that makes a farmer stop trusting the card, so a large gap
 * without a driver is named as exactly that.
 */
function yieldLimitingFactor({
  riskLevel,
  shouldIrrigate,
  depletion,
  nutrientsLow,
  stress,
  gapPct,
  modelled,
}: {
  riskLevel: string
  shouldIrrigate: boolean
  depletion: number
  nutrientsLow: boolean
  stress: number
  gapPct: number | null
  modelled: boolean
}): { label: string } {
  if (riskLevel === 'risk') return { label: 'Disease' }
  if (shouldIrrigate || depletion > 0.55) return { label: 'Water' }
  if (nutrientsLow) return { label: 'Nutrients' }
  if (stress > 0.5) return { label: 'Soil conditions' }
  if (!modelled) return { label: 'None' }
  if (gapPct !== null && gapPct >= 15) return { label: 'No single cause' }
  return { label: 'None — near potential' }
}

/**
 * Plain-language yield advice for the card footer.
 *
 * The gap is quoted in tonnes, not only in percent: a farmer thinks in what
 * the field will give, and "0.8 t/ha" is a number that can be weighed against
 * the cost of the water or fertilizer that would close it.
 */
function yieldAdvice({
  yieldTPerHa,
  limiting,
  gap,
  gapPct,
  modelled,
  crop,
  potentialTPerHa,
}: {
  yieldTPerHa: number
  limiting: string
  gap: number | null
  gapPct: number | null
  modelled: boolean
  crop: string
  potentialTPerHa: number
}): string {
  const shortfall =
    gap !== null && gap > 0 ? ` That is ${num(gap, 1)} t/ha (${gapPct}%) short of potential.` : ''

  // The part that says what to do, kept apart from the caveat about the
  // figure itself so the actionable sentence comes first.
  const subject = modelled ? `expected yield of ${num(yieldTPerHa, 1)} t/ha` : 'the crop'
  let action = ''
  if (limiting === 'Disease') {
    action = `Treat disease first — ${subject} will recover once pressure eases.`
  } else if (limiting === 'Water') {
    action = `Irrigate now — ${subject} is being held back by dry soil.`
  } else if (limiting === 'Nutrients') {
    action = `Feed the crop — ${subject} is being held back by low nutrients.`
  } else if (limiting === 'Soil conditions') {
    action = `Check the soil — ${subject} is being held back by readings outside the crop's band.`
  } else if (limiting === 'No single cause') {
    action = `Expected yield of ${num(yieldTPerHa, 1)} t/ha. Nothing to act on today — no disease to treat, no water to add, no fertilizer to apply.`
  } else if (limiting === 'None — near potential') {
    action = `Expected yield of ${num(yieldTPerHa, 1)} t/ha — nothing is holding the crop back.`
  }

  if (!modelled) {
    // The model has no basis for this crop, so the figure is a generic
    // estimate. Saying so is better than a number that reads as a barley or
    // potato yield when it is neither.
    const caveat = `The yield model has no ${crop} basis, so ${num(yieldTPerHa, 1)} t/ha is a generic estimate rather than a ${crop} figure.`
    const ceiling =
      potentialTPerHa > 0 ? ` The catalogue puts ${crop} at ${num(potentialTPerHa, 1)} t/ha.` : ''
    return [action, caveat, ceiling].filter(Boolean).join(' ')
  }

  if (limiting === 'No single cause') {
    // The gap is real even though nothing on the field explains it: the
    // field's own conditions account for it, and that is worth saying plainly
    // rather than leaving the farmer with two numbers that disagree.
    const unexplained =
      gap !== null && gap > 0
        ? ` The ${num(gap, 1)} t/ha (${gapPct}%) gap is what this field's conditions are costing.`
        : ''
    return `${action}${unexplained}`
  }

  return `${action}${shortfall}`
}
