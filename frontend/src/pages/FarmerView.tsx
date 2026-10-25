/**
 * Farmer view: the farm on a map, one action, one number, two buttons.
 *
 * The console answers "is the model right?" at a level of detail an agronomist
 * needs. A farmer standing in the field asks a smaller question: what do I do
 * today, on which field, how much — and what else is on the list. This view
 * answers only that, in the order the work happens (water, then fertiliser,
 * then spray), and it keeps every figure in a unit the work is done in: minutes
 * at the pump, bags of product, quintals per acre.
 *
 * The satellite map is the "where" that the console never had. It is a real
 * map of real imagery (see `FieldMap`), but the polygons are illustrative: the
 * registry knows a station and an area, not a surveyed boundary. The card says
 * so rather than letting a farmer mistake it for their own fence line.
 *
 * Nothing here is a new model. The action, the depth, the fertiliser dose and
 * the disease level all come from the same advisory payload the console reads;
 * only the presentation differs, and "More details" hands the field back to the
 * console rather than growing a second copy of it.
 *
 * Layout mirrors the reference board: a left rail with crop age, today's job
 * and the next seven days; a right side whose top two thirds is the satellite
 * map, with soil water and expected yield (plus the no-spray / no-fertiliser
 * alert rows) underneath.
 */

import { useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { AlertGlyph, LeafGlyph } from '../components/ui'
import CropHealthCheck from '../components/CropHealthCheck'
import { FieldMap, type MapField, type MapTone } from '../components/FieldMap'
import { int, num, relativeTime, titleCase } from '../lib/format'
import type { Advisory, LiveWeather, Telemetry } from '../types'

type Band = 'dry' | 'good' | 'wet' | 'rest' | 'none'

/**
 * The decision model's dry floor — backend `MOISTURE_BAND = (38, 62)`
 * (`backend/ml/inference.py`). A reading below 38% is the band the model
 * treats as dry. Restated here so the visible hold rule and the models
 * cannot drift apart silently.
 */
const MOISTURE_DRY_PCT = 38

export interface FarmerChip {
  key: 'fertilizer' | 'spray' | 'yield'
  title: string
  detail: string
  tone: 'healthy' | 'warn' | 'alert' | 'neutral'
}

/** One field as the farm overview and the map need it, already reduced. */
export interface FarmerField {
  fieldId: string
  crop: string
  stage: string
  areaM2: number
  /** Share of plantable water still in the soil, 0–1. */
  remaining: number
  band: Band
  /** Pump minutes the plan asks for today; 0 when nothing is scheduled. */
  minutes: number
  action: 'water' | 'watch' | 'ok' | 'rest'
  station: string
  stationLat: number | null
  stationLon: number | null
  /** The field's own GPS pin, when the operator placed one. */
  latitude?: number | null
  longitude?: number | null
}

/**
 * Where the demo farm sits, per station.
 *
 * The station coordinates are district references, and several are city
 * centres: a field drawn there would sit on a roof. These anchors are a few
 * kilometres out into farmland, chosen once and labelled "approximate" on the
 * map. A station that is not here falls back to its own coordinates, which is
 * still honest — just less photogenic.
 */
const FARM_ANCHORS: Record<string, { lat: number; lon: number }> = {
  LKO: { lat: 26.79, lon: 80.99 }, // farmland south-east of Lucknow
  HYD: { lat: 17.35, lon: 78.2 }, // farmland west of Hyderabad
  PNQ: { lat: 18.64, lon: 73.99 }, // farmland north-east of Pune
}

/**
 * Per-field overrides, for the demo fields whose station anchor lands on
 * built-up ground. The Barley-West station anchor sits inside Lucknow's
 * cantonment, so it is drawn instead on verified open farmland at Bakshi Ka
 * Talab, north-west of the city — checked against the satellite imagery the
 * map itself shows, so the plot never sits on a roof.
 */
const FIELD_ANCHORS: Record<string, { lat: number; lon: number }> = {
  'Barley-West': { lat: 26.85, lon: 80.83 },
}

/** Radius of the demo ring, in degrees of latitude (~1.2 km). */
const RING_DEG = 0.011

const ACCENT = {
  water: {
    ring: 'border-brand-100',
    icon: 'bg-brand-50 text-brand-700',
    kicker: 'text-brand-700',
  },
  watch: {
    ring: 'border-status-warnSoft',
    icon: 'bg-status-warnSoft text-[#b45309]',
    kicker: 'text-[#b45309]',
  },
  idle: {
    ring: 'border-status-healthySoft',
    icon: 'bg-status-healthySoft text-status-healthy',
    kicker: 'text-[#15803d]',
  },
  rest: {
    ring: 'border-ink-200',
    icon: 'bg-ink-100 text-ink-500',
    kicker: 'text-ink-500',
  },
} as const

const CHIP_TONE = {
  healthy: 'text-status-healthy',
  warn: 'text-[#b45309]',
  alert: 'text-status-alert',
  neutral: 'text-ink-500',
} as const

/** The season track's fallback shape, when the engine has no stage windows. */
const FALLBACK_STAGES = [
  { key: 'initial', label: 'Initial', from: 0, to: 0.2 },
  { key: 'development', label: 'Development', from: 0.2, to: 0.5 },
  { key: 'mid_season', label: 'Mid Season', from: 0.5, to: 0.8 },
  { key: 'late_season', label: 'Late Season', from: 0.8, to: 1 },
] as const

/**
 * Relative water demand per FAO-56 stage, for the track's colour fallback.
 *
 * Used only when a crop has no engine stage windows. It is the shape the
 * Kc curve implies — the crop drinks little while establishing, most through
 * development and mid-season, easing off as it ripens — not a computed rate.
 */
const FALLBACK_DEMAND: Record<string, number> = {
  initial: 0.5,
  development: 0.85,
  mid_season: 1,
  late_season: 0.7,
} as const

/** One sentence, the reason the action is the action. Not a model explanation. */
function whyLine(field: Advisory, band: Band, watered: boolean): string {
  const crop = field.crop.toLowerCase()
  const stage = String(field.stage ?? '').toLowerCase()

  if (band === 'rest') {
    return `The ${crop} is harvested, so dry soil does not matter now.`
  }
  if (band === 'dry') {
    return stage === 'mid_season' || stage === 'late_season'
      ? `The soil is dry and the ${crop} is using the most water it will all season.`
      : `The soil is drying out while the ${crop} is still getting established.`
  }
  if (band === 'wet') {
    return watered
      ? `The soil is wet now, but the ${crop} will need this water over the next few days.`
      : 'The soil is wet. Watering now would waste water.'
  }
  if (watered) {
    return `The soil is fine today, but the ${crop} will need more water before the week is out.`
  }
  return `The soil still holds enough water for the ${crop} at this stage.`
}

function stageWords(field: Advisory): string {
  return titleCase(String(field.stageLabel ?? '').replace(/\(.*?\)/g, '').trim())
}

function todayKey(): string {
  const now = new Date()
  return `${now.getFullYear()}-${now.getMonth() + 1}-${now.getDate()}`
}

function readDone(key: string): boolean {
  try {
    return window.localStorage.getItem(key) === '1'
  } catch {
    return false
  }
}

function writeDone(key: string, value: boolean): void {
  try {
    if (value) window.localStorage.setItem(key, '1')
    else window.localStorage.removeItem(key)
  } catch {
    // A private-mode browser throws on write. The mark is a convenience, not
    // the record, so losing it is better than breaking the screen over it.
  }
}

/**
 * Position the fields around their station.
 *
 * Fields sharing a station (two of the four demo fields do) are placed on a
 * ring so they do not sit on top of each other. A field the operator pinned
 * with its own GPS coordinates is drawn exactly there — it was placed on the
 * map already, at registration — and grouped apart so it never drags its
 * station-neighbours with it. Positions without a pin are illustrative
 * and say so on the map.
 */
function plotsFor(fields: FarmerField[]): MapField[] {
  // A field with its own verified anchor — a survey override or the
  // operator's own GPS pin — is grouped apart from its station, so moving
  // one plot never drags its neighbours with it.
  const ownPin = (field: FarmerField): { lat: number; lon: number } | null =>
    typeof field.latitude === 'number' && typeof field.longitude === 'number'
      ? { lat: field.latitude, lon: field.longitude }
      : null
  const groupKey = (field: FarmerField): string =>
    FIELD_ANCHORS[field.fieldId] || ownPin(field) ? `field:${field.fieldId}` : field.station
  const byStation = new Map<string, FarmerField[]>()
  for (const field of fields) {
    const key = groupKey(field)
    const group = byStation.get(key) ?? []
    group.push(field)
    byStation.set(key, group)
  }

  const plots: MapField[] = []
  for (const [, group] of byStation) {
    const first = group[0]
    const pin = ownPin(first)
    const anchor =
      pin ??
      FIELD_ANCHORS[first.fieldId] ??
      FARM_ANCHORS[first.station] ??
      (typeof first.stationLat === 'number' && typeof first.stationLon === 'number'
        ? { lat: first.stationLat, lon: first.stationLon }
        : null)
    if (!anchor) continue

    const lonScale = Math.max(0.2, Math.cos((anchor.lat * Math.PI) / 180))
    group.forEach((field, index) => {
      // A single-plot group sits dead on its anchor — there is nothing to fan
      // out from, and for a GPS-pinned field "exactly here" is the point.
      const offset = group.length === 1 ? 0 : 1
      const angle = (2 * Math.PI * index) / group.length - Math.PI / 2
      const acres = field.areaM2 / 4046.86
      plots.push({
        fieldId: field.fieldId,
        label: `${acres.toFixed(1)} ac`,
        lat: anchor.lat + Math.cos(angle) * RING_DEG * offset,
        lon: anchor.lon + (Math.sin(angle) * RING_DEG * offset) / lonScale,
        tone: plotTone(field),
        gps: Boolean(ownPin(field)),
      })
    })
  }
  return plots
}

/** The map's colour carries the action, not just the soil band. */
function plotTone(field: FarmerField): MapTone {
  if (field.action === 'water') return 'dry'
  if (field.action === 'rest') return 'rest'
  return field.band
}

/**
 * Real-calendar weekday labels for the horizon.
 *
 * Offset 0 is the operator's own today, so the chart reads on the calendar
 * its reader uses — the plan's *data* is the engine's weather replay and is
 * anchored at the archive's last day (2024-12-31, a Tuesday), which must
 * never rename the operator's week. The bar heights stay the plan's; only
 * the axis labels follow the real calendar.
 */
function weekday(offset: number, today = new Date()): string {
  const date = new Date(today)
  date.setDate(today.getDate() + offset)
  return date.toLocaleDateString(undefined, { weekday: 'short' })
}

function SpeakerGlyph({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="M4 10v4h3l4 3.5v-11L7 10H4Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinejoin="round"
      />
      <path
        d="M15 9.5a3.5 3.5 0 0 1 0 5M17.5 7a7 7 0 0 1 0 10"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.7"
        strokeLinecap="round"
      />
    </svg>
  )
}

function CheckGlyph({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path
        d="m5 12.5 4.5 4.5L19 7"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

function ChipIcon({ chip }: { chip: FarmerChip }) {
  if (chip.key === 'fertilizer') {
    // A sack, because that is the object the farmer buys.
    return (
      <svg width={20} height={20} viewBox="0 0 24 24" aria-hidden="true">
        <path
          d="M8 4h8l2 3v13H6V7l2-3Z"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.7"
          strokeLinejoin="round"
        />
        <path d="M9.5 4c0 2 1 3 2.5 3s2.5-1 2.5-3" fill="none" stroke="currentColor" strokeWidth="1.5" />
        <path d="M9 11h6M9 14.5h4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    )
  }
  if (chip.key === 'spray') {
    return <AlertGlyph size={20} />
  }
  return <LeafGlyph size={20} />
}

/* -------------------------------------------------------------------------- */
/* Cards                                                                       */
/* -------------------------------------------------------------------------- */

/** How old the crop is, and where that puts it in the season. */
function CropAgeCard({ field }: { field: Advisory }) {
  const current = String(field.stage ?? '').toLowerCase()
  const finished = current === 'post_harvest' || current === 'harvest'
  const progress = Math.max(0, Math.min(1, field.stageProgress))
  const windows = useMemo(() => field.season?.stageWindows ?? [], [field.season])

  /** The current stage's water rate, straight from the engine's stage plan. */
  const currentWindow = useMemo(
    () => windows.find((window) => window.stage === current) ?? null,
    [current, windows],
  )
  const stageKc = currentWindow ? currentWindow.meanKc : field.water.kc
  const stageEtc = currentWindow ? currentWindow.meanEtcMmDay : field.water.etcMmDay

  /**
   * The track's stage shares — the axis the whole card is drawn on.
   *
   * Primary: the crop catalogue's thermal shares (`profileShare`, each stage's
   * fraction of the season's growing degrees). Those are a function of the
   * crop's parameters alone, so every field of one crop — Cotton-South,
   * Live-Field, or a field registered tomorrow — lays out identical blocks and
   * puts the peak bar and the arrow in identical places. The engine's
   * stageProgress is linear in GDD within a stage, so the arrow is exact on
   * this axis.
   *
   * Fallback: the field's own calendar replay (day shares), for advisories
   * without the profile figures.
   */
  const shares = useMemo(() => {
    if (windows.length === 0) return null
    if (
      windows.every(
        (window) => typeof window.profileShare === 'number' && window.profileShare > 0,
      )
    ) {
      return windows.map((window) => ({
        key: window.stage,
        share: (window.profileShare as number) * 100,
      }))
    }
    const total = windows.reduce((sum, window) => sum + Math.max(1, window.days), 0)
    return windows.map((window) => ({
      key: window.stage,
      share: (Math.max(1, window.days) / total) * 100,
    }))
  }, [windows])

  const segments = useMemo(() => {
    if (shares) {
      const maxEtc = Math.max(...windows.map((window) => window.meanEtcMmDay), 0.001)
      let cursor = 0
      return shares.map((item) => {
        const window = windows.find((entry) => entry.stage === item.key)
        const segment = {
          key: item.key,
          label: titleCase(item.key),
          start: cursor,
          share: item.share,
          // Relative water demand of this stage, 0–1, by its own ETc against
          // the thirstiest stage of the same season. It does not colour the
          // track — it only locates the peak-water bar below when the advisory
          // has no crop-level peakStretch to use.
          demand: window ? window.meanEtcMmDay / maxEtc : 0.6,
          etc: window ? window.meanEtcMmDay : null,
          kc: window ? window.meanKc : null,
        }
        cursor += item.share
        return segment
      })
    }
    // No engine windows: the track falls back to the FAO-56 stage shape, and
    // the colours fall back to the shape that shape implies — early and late
    // stages drink less than development and mid-season.
    return FALLBACK_STAGES.map((stage) => ({
      key: stage.key,
      label: stage.label,
      start: stage.from * 100,
      share: (stage.to - stage.from) * 100,
      demand: FALLBACK_DEMAND[stage.key] ?? 0.6,
      etc: null,
      kc: null,
    }))
  }, [shares, windows])

  const marker = useMemo(() => {
    if (finished) return null
    if (shares) {
      let cursor = 0
      for (const item of shares) {
        if (item.key === current) {
          return Math.min(100, Math.max(0, cursor + progress * item.share))
        }
        cursor += item.share
      }
      return null
    }
    const stage = FALLBACK_STAGES.find((item) => item.key === current)
    if (!stage) return null
    return (stage.from + progress * (stage.to - stage.from)) * 100
  }, [current, finished, progress, shares])

  /**
   * The peak-water interval the blue bar sits over.
   *
   * Primary source is the crop profile's own `peakStretch` (backend resolves it
   * on the catalogue's Kc curve over thermal time), so the bar covers the same
   * stages for every field of a crop — Cotton-South and Live-Field agree, and
   * so does a field registered tomorrow. The per-field windows only supply the
   * on-track geometry for that span.
   *
   * The ETc-based fallback below is for advisories without the span (older
   * fixtures, crops outside the catalogue): the thirstiest window's demand,
   * extended over neighbours within 90%.
   */
  const peakDemand = useMemo(() => {
    if (segments.length === 0) return null
    const stretch = field.season?.peakStretch
    if (stretch) {
      const from = segments.findIndex((segment) => segment.key === stretch.fromStage)
      const to = segments.findIndex((segment) => segment.key === stretch.toStage)
      if (from >= 0 && to >= from) {
        const etcValues = segments
          .slice(from, to + 1)
          .map((segment) => segment.etc)
          .filter((v) => v != null)
        return {
          start: segments[from].start,
          span: segments[to].start + segments[to].share - segments[from].start,
          fromStage: segments[from].label,
          toStage: segments[to].label,
          etc: etcValues.length ? Math.max(...(etcValues as number[])) : null,
        }
      }
    }
    let peakIndex = 0
    segments.forEach((segment, i) => {
      if (segment.demand > segments[peakIndex].demand) peakIndex = i
    })
    let from = peakIndex
    while (from > 0 && segments[from - 1].demand >= 0.9 * segments[peakIndex].demand) from--
    let to = peakIndex
    while (to < segments.length - 1 && segments[to + 1].demand >= 0.9 * segments[peakIndex].demand) to++
    const etcValues = segments.slice(from, to + 1).map((segment) => segment.etc).filter((v) => v != null)
    return {
      start: segments[from].start,
      span: segments[to].start + segments[to].share - segments[from].start,
      fromStage: segments[from].label,
      toStage: segments[to].label,
      etc: etcValues.length ? Math.max(...(etcValues as number[])) : null,
    }
  }, [field.season, segments])

  return (
    <section className="card flex h-full min-h-0 flex-col overflow-hidden" aria-label="Crop age">
      <div className="pad-card flex h-full min-h-0 flex-col px-4 pt-2 pb-2.5">
        <div className="flex shrink-0 items-baseline justify-between gap-3">
          <p className="text-[10px] font-bold uppercase tracking-[0.1em] text-ink-500">
            Crop age
          </p>
          <span className="text-[11px] font-semibold text-slate-600">{field.crop}</span>
        </div>

        {/* The figure and its stage breath in whatever room the tile has, so a
            taller window spends it here instead of a hole above the track. */}
        <div className="flex min-h-0 flex-1 flex-col justify-center">
          <div className="flex items-baseline gap-1.5">
            <span className="tnum text-[40px] font-semibold leading-none tracking-tight text-ink-900">
              {field.ageDays}
            </span>
            <span className="text-[14px] font-medium text-slate-600">days old</span>
          </div>

          <p className="mt-1 text-[14px] font-semibold text-brand-700">
            {finished ? 'Season complete' : stageWords(field)}
          </p>
        </div>

        <div className="shrink-0">
          <div className="relative">
            <div className="flex h-3 gap-0.5 overflow-hidden rounded-pill bg-ink-100">
              {segments.map((segment) => (
                <div
                  key={segment.key}
                  title={
                    segment.etc != null
                      ? `${segment.label} · ~${num(segment.etc, 1)} mm/day${segment.kc != null ? `, Kc ${num(segment.kc, 2)}` : ''}`
                      : segment.label
                  }
                  style={{ width: `${segment.share}%` }}
                  className="bg-ink-200/70"
                />
              ))}
            </div>
            {peakDemand !== null && (
              <span
                className="pointer-events-auto absolute bottom-0.5 top-0.5 z-[5] rounded-pill bg-brand-500 shadow-[0_0_0_1px_rgba(255,255,255,0.6)]"
                style={{
                  left: `${peakDemand.start + peakDemand.span * 0.07}%`,
                  width: `${peakDemand.span * 0.86}%`,
                }}
                title={`Peak water stretch · ${
                  peakDemand.fromStage === peakDemand.toStage
                    ? peakDemand.fromStage
                    : `${peakDemand.fromStage} → ${peakDemand.toStage}`
                }${peakDemand.etc != null ? `, up to ~${num(peakDemand.etc, 1)} mm/day` : ''}`}
              />
            )}
            {marker !== null && (
              <>
                <span
                  aria-hidden="true"
                  className="pointer-events-none absolute -top-[10px] z-20 -translate-x-1/2"
                  style={{ left: `${Math.min(99, Math.max(1, marker))}%` }}
                >
                  <svg width="14" height="10" viewBox="0 0 14 10" aria-hidden="true">
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
                  className="pointer-events-none absolute top-0 z-10 h-4 w-0.5 -translate-x-1/2 rounded-full bg-ink-900/45"
                  style={{ left: `${Math.min(99, Math.max(1, marker))}%` }}
                />
              </>
            )}
          </div>
          <div className="mt-2 flex justify-between gap-1">
            {segments.map((segment) => (
              <span
                key={segment.key}
                className={`min-w-0 flex-1 truncate text-[12px] ${
                  segment.key === current
                    ? 'font-semibold text-brand-700'
                    : 'font-medium text-ink-600'
                }`}
              >
                {segment.label}
              </span>
            ))}
          </div>
          <p className="mt-1 text-[10px] leading-none text-ink-400">
            Blue bar = the crop's peak-water stretch · arrow = today ({field.ageDays} days)
          </p>
        </div>

        <dl className="panel mt-2 grid grid-cols-3 gap-2 px-3 py-1.5">
          <MiniStat label="Stage Kc" value={finished ? '—' : num(stageKc, 2)} />
          <MiniStat
            label="Stage water"
            value={finished ? '—' : `${num(stageEtc, 1)} mm/d`}
          />
          <MiniStat
            label="Season water"
            value={
              field.season && field.season.seasonGrossMm > 0
                ? `${num(field.season.seasonGrossMm, 0)} mm`
                : `${num(field.water.rootDepthCm, 0)} cm roots`
            }
          />
        </dl>

        {/* The connection this card exists to make visible: the age is not a
            fact to admire, it is the water plan's driver. Each stage carries
            its own crop coefficient, so the same field drinks a different
            amount as it grows — and two crops of the same age drink different
            amounts. This is why one cotton field and another, sown weeks
            apart, get different pump minutes. */}
        <p className="mt-1.5 text-[11px] leading-snug text-slate-700">
          {finished
            ? 'Season over — the plan stops watering a harvested crop.'
            : field.ageDays < 10
              ? `Young crop: stage Kc ${num(stageKc, 2)} keeps early watering small until roots reach deeper.`
              : `Growth drives water: this stage's Kc ${num(stageKc, 2)} is why the plan asks for ${num(stageEtc, 1)} mm/day.`}
        </p>
      </div>
    </section>
  )
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="truncate text-[10px] font-semibold uppercase tracking-[0.06em] text-ink-500">
        {label}
      </dt>
      <dd className="tnum mt-0.5 truncate text-[14px] font-semibold text-ink-900">{value}</dd>
    </div>
  )
}

/**
 * The next seven days, as need and pump days.
 *
 * Two layers, both from the engine:
 * - the pale bar is the day's crop requirement — the horizon's own demand
 *   (Kc x ET0 off the crop's curve) net of the expected rain credited — the
 *   exact input the optimiser scheduled against. It is present even when no
 *   pump is scheduled, so a correctly-wet week is a week of requirements,
 *   never an empty chart;
 * - the solid blue bar is the pump day the optimiser chose, at the depth it
 *   chose, on top of that requirement.
 *
 * With a live node attached, the week bends to what the probe reads, in one
 * place only: today. The plan's replay opens the soil at a historical
 * depletion; the node reports what is really in the ground now. If the probe
 * reads the band the plan would water from, today's pump is marked held —
 * water the plan wanted today is not needed on the moisture the node
 * measured — and the day the plan expects it moves is not invented either;
 * the node's next decision at the next POST settles it.
 */
function WeekCard({ field }: { field: Advisory }) {
  const days = (field.schedule ?? []).slice(0, 7)
  const finished = field.stage === 'post_harvest' || field.stage === 'harvest'
  const weekTotalNeed = days.reduce((sum, day) => sum + (day.needMm ?? 0), 0)

  // Today's hold decision, from the node's own reading. Same band rule the
  // decision model uses (dry floor MOISTURE_DRY_PCT — backed by the model's
  // MOISTURE_BAND): hold only when the node is fresh AND the soil is not in
  // the dry band. When the season is over, there is nothing to hold.
  const liveHoldsToday =
    field.live?.isLive &&
    !finished &&
    field.sensors.soilMoisture != null &&
    Number.isFinite(field.sensors.soilMoisture) &&
    field.sensors.soilMoisture >= MOISTURE_DRY_PCT

  // Pump-day minutes from the decision's own runtime-to-depth ratio, so any
  // bar's label and today's action card cannot disagree.
  const depthToday = field.recommendation.finalDepthMm ?? field.recommendation.depthMm
  const secondsPerMm =
    depthToday > 0 ? field.recommendation.durationSeconds / depthToday : 0

  const peak = Math.max(
    5,
    ...days.map((day) => Math.max(day.applyMm, day.needMm ?? 0)),
  )
  const anyPump = days.some((day) => day.applyMm > 0)

  return (
    <section
      className="card h-full min-h-0 flex-col overflow-hidden" aria-label="Next seven days">
      <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <header className="flex shrink-0 items-baseline justify-between gap-3 px-4 pt-2.5">
        <h2 className="text-[13px] font-semibold tracking-tight text-ink-900">Next 7 days</h2>
        <span className="text-[11px] text-slate-700">plan · {field.crop.toLowerCase()}</span>
      </header>
      <div className="flex min-h-0 flex-1 gap-1.5 px-4 pb-1.5 pt-2">
        {days.map((day) => {
            const isToday = day.dayOffset === 0
            const held = isToday && liveHoldsToday
            const need = Math.max(0, day.needMm ?? 0)
            const needHeight =
              need > 0.05 ? `${Math.max(6, (need / peak) * 100)}%` : '3px'
            const pumpHeight =
              day.applyMm > 0 ? `${Math.max(6, (day.applyMm / peak) * 100)}%` : undefined
            const minutes = Math.round(day.applyMm * (secondsPerMm > 0 ? secondsPerMm : 8.33) / 60)
            return (
              <div key={day.dayOffset} className="flex min-w-0 flex-1 flex-col items-center">
                <span className={`tnum flex h-4 shrink-0 items-end text-[10px] font-semibold ${
                  held ? 'text-[#b45309]' : 'text-slate-700'
                }`}>
                  {day.applyMm > 0 ? (held ? 'held' : `${minutes}m`) : ''}
                </span>
                <div
                  className="relative w-full flex-1"
                  aria-label={`${isToday ? 'Today' : weekday(day.dayOffset)}: needs ~${need} mm${
                    day.applyMm > 0 ? `, pump ${day.applyMm} mm` : ''
                  }`}
                >
                  {/* The requirement this day pays, from the horizon's demand
                      and expected rain. Pale, and always present while the
                      crop is in the ground. */}
                  <span
                    className="absolute bottom-0 w-full rounded-t-[3px] bg-brand-200/80"
                    style={{ height: needHeight }}
                  />
                  {/* The pump day the optimiser chose, drawn over the need it
                      was scheduled to satisfy. */}
                  {pumpHeight !== undefined && (
                    <span
                      className={`absolute bottom-0 z-10 w-full rounded-t-[3px] ring-1 ring-inset ring-white/70 ${
                        held
                          ? 'bg-gradient-to-t from-[#ea580c] to-[#fbbf24] opacity-60'
                          : 'bg-brand-500'
                      }`}
                      style={{ height: pumpHeight }}
                    />
                  )}
                </div>
                <span className={`mt-1 shrink-0 text-[11px] font-medium uppercase tracking-[0.04em] ${
                  isToday ? 'text-brand-700 font-semibold' : 'text-slate-700'
                }`}>
                  {isToday ? 'Today' : weekday(day.dayOffset)}
                </span>
              </div>
            )
          })}
      </div>
      <p className="shrink-0 px-4 pb-2 text-[11px] leading-snug text-slate-700">
        {finished
          ? 'Season complete — the plan stops watering a harvested crop; the pale bars are the bare-soil remainder.'
          : liveHoldsToday
            ? 'The node reads wetter than the plan assumed — today’s pump is held, and the week of requirements still stands.'
            : anyPump
              ? `Pale = the crop's daily need (${num(weekTotalNeed, 0)} mm this week) · blue = pump days · grey mm = projected soil depletion (pump at RAW when it nears the stress line).`
              : `No pump this week: the soil holds ${num((field.water.tawMm ?? 0) - (field.water.depletionMm ?? 0), 0)} mm of available water and the week takes ${num(weekTotalNeed, 0)} mm — the pump returns when depletion (grey) nears the ${num(field.water.rawMm ?? 0, 0)} mm stress line.`}
      </p>
      </div>
    </section>
  )
}

/**
 * The soil, as one number, and the watering plan it drives.
 *
 * The ring is the soil-moisture percentage — the same reading that decides
 * irrigation on the node — not a volume: a farmer reading their probe thinks
 * "thirty-one percent", not "twelve millimetres". When a live node is fresh
 * the ring shows the node's own measurement; otherwise the plan-derived
 * value. The right column connects it to the plan: when the crop reaches
 * stress, where the field stands now, and today's pump minutes.
 */
function SoilWaterCard({
  field,
  band,
}: {
  field: Advisory
  band: Band
}) {
  const style = BAND_STYLE[band]
  const radius = 58
  const circumference = 2 * Math.PI * radius
  // The ring's fill is the moisture itself, on the probe's 0–100% scale.
  const moisture = Math.max(0, Math.min(100, Number(field.sensors.soilMoisture) || 0))
  const fill = moisture / 100
  const finishedSoil = field.stage === 'post_harvest' || field.stage === 'harvest'
  const daysToStress = field.water.daysUntilStress
  const minutes = Math.max(1, Math.round(field.recommendation.durationSeconds / 60))
  const watering = field.recommendation.durationSeconds > 0

  return (
    <section className="card h-full min-h-0 flex-col overflow-hidden" aria-label="Soil water">
      <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <header className="flex shrink-0 items-baseline justify-between gap-2 px-4 pt-2">
        <h2 className="text-[11px] font-bold uppercase tracking-[0.1em] text-ink-600">
          Soil moisture
        </h2>
        <span className="flex items-baseline gap-1.5 text-[11px] text-slate-700">
          {field.live?.isLive && (
            <span
              className="rounded-full bg-emerald-100 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-[0.08em] text-emerald-800"
              title={`Live ESP32 node ${field.live.deviceId}, newest reading ${
                field.live.ageSeconds != null ? `${Math.round(field.live.ageSeconds / 60) < 1 ? "<1 min" : `${Math.round(field.live.ageSeconds / 60)} min`} old` : "just now"
              }`}
            >
              Live node
            </span>
          )}
        </span>
      </header>

      <div className="flex min-h-0 flex-1 items-center gap-4 px-4 pb-2 pt-1.5">
        <div className="relative aspect-square w-[100px] shrink-0 self-center">
          <svg viewBox="0 0 140 140" className="h-full w-full -rotate-90">
            <defs>
              <linearGradient id="farmerRing" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor={style.from} />
                <stop offset="100%" stopColor={style.to} />
              </linearGradient>
            </defs>
            <circle cx="70" cy="70" r={radius} fill="none" stroke="#e2e8f0" strokeWidth="13" />
            <circle
              cx="70"
              cy="70"
              r={radius}
              fill="none"
              stroke="url(#farmerRing)"
              strokeWidth="13"
              strokeLinecap="round"
              strokeDasharray={`${circumference * fill} ${circumference}`}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={`tnum text-[clamp(1rem,5cqw,1.35rem)] font-semibold leading-none ${style.text}`}>
              {Math.round(moisture)}%
            </span>
            <span className="mt-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-ink-400">
              moisture
            </span>
          </div>
        </div>

        <dl className="panel flex min-w-0 flex-1 flex-col justify-between gap-1 px-3 py-2 text-[11px] text-ink-700">
          <Line
            label="Stress"
            value={finishedSoil
              ? 'season over'
              : daysToStress != null && daysToStress > 0
                ? `in ${Math.round(daysToStress)} d`
                : 'now'}
          />
          <Line
            label="Stage"
            value={titleCase(String(field.stageLabel ?? '').replace(/\(.*?\)/g, '').trim())}
          />
          <Line
            label="Plan today"
            value={finishedSoil ? 'none' : watering ? `${minutes} min pump` : 'rest'}
          />
          {field.live?.valveOpen != null && (
            <Line label="Valve" value={field.live.valveOpen ? 'OPEN' : 'CLOSED'} />
          )}
        </dl>
      </div>

      <p className="shrink-0 px-4 pb-2 text-[11px] leading-snug text-slate-700">
        {band === 'rest'
          ? 'No crop in the ground, so soil wetness does not drive a decision here.'
          : `${num(field.sensors.soilMoisture, 0)}% soil moisture — ${
              field.live?.isLive ? 'the node’s own probe' : 'the plan’s estimate'
            }. The plan waters before it reaches stress.`}
      </p>
      </div>
    </section>
  )
}

function Line({
  label,
  value,
}: {
  label: string
  value: ReactNode
}) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <dt className="shrink-0 text-ink-500">{label}</dt>
      <dd className="tnum truncate font-semibold text-ink-900">{value}</dd>
    </div>
  )
}

/**
 * The bench node's own readings, in the freed third of the row.
 *
 * Rendered only while a node's real telemetry is fresh for this field — the
 * one field the node is attached to. The values are the node's measurements,
 * not the plan derivation, which is the whole point of the card existing.
 */
function LiveNodeCard({ field }: { field: Advisory }) {
  const live = field.live
  const ageSeconds = live?.ageSeconds ?? 0
  const ageText =
    ageSeconds < 90
      ? `${Math.max(1, Math.round(ageSeconds))}s ago`
      : `${Math.round(ageSeconds / 60)}m ago`
  return (
    <section className="card h-full min-h-0 flex-col overflow-hidden" aria-label="Live node">
      <header className="flex shrink-0 items-baseline justify-between gap-2 px-4 pt-2">
        <h2 className="text-[11px] font-bold uppercase tracking-[0.1em] text-ink-600">
          Node · live
        </h2>
        <span className="rounded-full bg-emerald-100 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-[0.08em] text-emerald-800">
          {ageText}
        </span>
      </header>
      <div className="flex min-h-0 flex-1 items-start px-4 pb-2 pt-1.5">
        <dl className="panel w-full px-3 py-2 text-[11px] text-ink-700">
          <Line label="Soil" value={`${num(field.sensors.soilMoisture, 1)}% (vol.)`} />
          <Line label="Air" value={`${num(field.sensors.temperature, 1)}°C`} />
          <Line label="Humidity" value={`${num(field.sensors.humidity, 0)}%`} />
          <Line
            label="Light"
            value={
              field.sensors.lightIntensity >= 1000
                ? `${(field.sensors.lightIntensity / 1000).toFixed(1)} klx`
                : `${num(field.sensors.lightIntensity, 0)} lx`
            }
          />
          {field.sensors.rssiDbm != null && (
            <Line label="Wi-Fi" value={`${field.sensors.rssiDbm} dBm`} />
          )}
          {live?.valveOpen != null && (
            <Line
              label="Valve"
              value={
                live.valveOpen ? (
                  <span
                    aria-hidden="true"
                    className="inline-flex items-center gap-1 font-bold text-brand-700"
                  >
                    <span className="relative flex h-1.5 w-1.5">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-brand-500 opacity-75" />
                      <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-brand-600" />
                    </span>
                    OPEN
                  </span>
                ) : (
                  <span className="font-semibold text-ink-500">CLOSED</span>
                )
              }
            />
          )}
        </dl>
      </div>
      <p className="shrink-0 px-4 pb-2 text-[11px] leading-snug text-slate-700">
        ESP32 {live?.deviceId} posts every ~10 s{live?.valveOpen != null ? ' and reports its valve' : ''}. Soil
        water, air and light above are the node's own measurements, not the plan's estimates.
      </p>
    </section>
  )
}

/** What the crop should give, and the things that are not on today's list. */
function YieldCard({ chips }: { chips: FarmerChip[] }) {
  const yieldChip = chips.find((chip) => chip.key === 'yield')
  const notes = chips.filter((chip) => chip.key !== 'yield')

  const detail = yieldChip?.detail ?? '—'
  // "X quintal total · Y quintal/acre" renders as two lines: the whole field's
  // harvest is the headline, the per-acre rate is the figure beneath it.
  const parts = detail.split(' · ')
  const totalLabel = parts[0] ?? detail
  const perAcreLabel = parts.slice(1).join(' · ')

  return (
    <section className="card h-full min-h-0 flex-col overflow-hidden" aria-label="Yield and today's notes">
      <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <div className="flex min-h-0 flex-1 flex-col px-4 pb-2 pt-2">
        <p className="text-[10px] font-bold uppercase tracking-[0.1em] text-ink-500">
          {yieldChip?.title ?? 'Expected yield'}
        </p>
        <p className="tnum mt-0.5 text-[24px] font-semibold leading-tight tracking-tight text-ink-900">
          {totalLabel}
        </p>
        <p className="tnum mt-0.5 text-[12px] font-semibold leading-snug text-slate-700">
          {perAcreLabel}
        </p>
        <p className="mt-0.5 text-[11px] leading-snug text-slate-700">
          What this field should give at harvest.
        </p>

        <ul className="mt-2 flex flex-1 flex-col justify-center divide-y divide-ink-200/70">
          {notes.map((chip) => (
            <li key={chip.key} className="flex items-start gap-3 py-1.5">
              <span className={`mt-0.5 shrink-0 ${CHIP_TONE[chip.tone]}`}>
                <ChipIcon chip={chip} />
              </span>
              <div className="min-w-0">
                <p className="text-[13px] font-semibold leading-snug text-ink-900">
                  {chip.title}
                </p>
                <p className="mt-0.5 text-[11px] leading-snug text-slate-600">{chip.detail}</p>
              </div>
            </li>
          ))}
        </ul>
      </div>
      </div>
    </section>
  )
}

/* -------------------------------------------------------------------------- */
/* The view                                                                    */
/* -------------------------------------------------------------------------- */

export function FarmerView({
  field,
  fields,
  depth,
  band,
  chips,
  live,
  node,
  onSelect,
}: {
  field: Advisory
  fields: FarmerField[]
  depth: number
  band: Band
  chips: FarmerChip[]
  live: LiveWeather | null
  /** The node's freshest own POST for this field, from the 10 s telemetry poll. */
  node?: Telemetry | null
  onSelect: (fieldId: string) => void
}) {
  const watered = depth > 0
  const rest = band === 'rest'
  const watch = !watered && !rest && field.status === 'monitor'

  const state = watered ? 'water' : rest ? 'rest' : watch ? 'watch' : 'idle'
  const accent = ACCENT[state]

  const minutes = Math.max(1, Math.round(field.recommendation.durationSeconds / 60))
  const acres = field.areaM2 / 4046.86
  const acreWord = Math.abs(acres - 1) < 0.05 ? 'acre' : 'acres'
  const daysToStress = field.water.daysUntilStress

  const action = watered
    ? 'Water today'
    : rest
      ? 'Season finished'
      : watch
        ? 'Watch today'
        : 'Nothing to do today'

  const kicker =
    state === 'water'
      ? 'Today’s job'
      : state === 'watch'
        ? 'Today’s job'
        : state === 'idle'
          ? 'All clear'
          : 'Season over'

  const why = whyLine(field, band, watered)
  const rain = field.water.rainfallMmDay > 0.05 ? ' Rain today is already counted.' : ''

  const doneKey = `irrigo.done.${field.fieldId}.${todayKey()}`
  const [done, setDone] = useState(() => readDone(doneKey))

  useEffect(() => {
    setDone(readDone(doneKey))
  }, [doneKey])

  useEffect(
    () => () => {
      if ('speechSynthesis' in window) window.speechSynthesis.cancel()
    },
    [],
  )

  const spokeText = [
    action + '.',
    watered ? `Run the pump for ${minutes} minutes.` : '',
    why,
    rain,
  ]
    .filter(Boolean)
    .join(' ')

  const listen = () => {
    if (!('speechSynthesis' in window)) return
    if (typeof SpeechSynthesisUtterance === 'undefined') return
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(spokeText)
    utterance.lang = navigator.language || 'en'
    utterance.rate = 0.9
    window.speechSynthesis.speak(utterance)
  }

  const toggleDone = () => {
    const next = !done
    setDone(next)
    writeDone(doneKey, next)
  }

  const plots = useMemo(() => plotsFor(fields), [fields])
  const centre = plots.find((plot) => plot.fieldId === field.fieldId) ?? null

  const sky = live?.available
    ? `${num(live.temperatureC, 0)}° now · ${num(live.rainfallMm, 1)} mm rain today`
    : `${num(field.sensors.temperature, 0)}° · ${num(field.water.rainfallMmDay, 1)} mm rain today`

  return (
    <div className="flex w-full flex-col gap-2.5 pb-3 xl:h-full xl:min-h-0 xl:pb-4">
      <div className="grid min-h-0 gap-2.5 xl:flex-1 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.4fr)]">
        {/* ---- Left rail: the season, the decision, the week, the photo check
            Four cards of one width. Rows share the height evenly, but each one
            keeps a floor at its own content (`minmax(min-content,1fr)`), so a
            shorter screen grows the card that needs it instead of clipping,
            and a taller screen keeps the tiles level. */}
        <div className="grid min-h-0 grid-rows-[repeat(4,minmax(min-content,1fr))] gap-2.5">
          <CropAgeCard field={field} />

          {/* The whole decision: what to do, how much, and the two things to press. */}
          <section
            className={`card h-full min-h-0 overflow-hidden border-2 ${accent.ring}`}
            aria-label="Today's action"
          >
            <div className="pad-card flex h-full flex-col overflow-hidden py-2">
              <div className="flex items-center gap-2.5">
                <div className="min-w-0">
                  <p className={`text-[10px] font-bold uppercase tracking-[0.09em] ${accent.kicker}`}>
                    {kicker}
                  </p>
                  <h1 className="text-[24px] font-semibold leading-tight tracking-tight text-ink-900 sm:text-[28px]">
                    {action}
                  </h1>
                </div>
              </div>

              {watered && (
                <div className="mt-1.5">
                  <div className="flex items-baseline gap-2">
                    <span className="tnum text-[42px] font-semibold leading-none tracking-tight text-ink-900 sm:text-[50px]">
                      {minutes}
                    </span>
                    <span className="text-[18px] font-medium text-ink-700 sm:text-[20px]">minutes</span>
                  </div>
                  <p className="mt-1 text-[13px] text-slate-700">
                    at the pump · {num(depth, 1)} mm over {num(acres, 1)} {acreWord}
                    <span className="text-ink-600"> · {int(field.recommendation.volumeLitres)} L</span>
                  </p>
                </div>
              )}

              {!watered && watch && typeof daysToStress === 'number' && daysToStress > 0 && (
                <div className="mt-1.5">
                  <div className="flex items-baseline gap-2">
                    <span className="tnum text-[42px] font-semibold leading-none tracking-tight text-ink-900">
                      {Math.round(daysToStress)}
                    </span>
                    <span className="text-[18px] font-medium text-ink-700">
                      day{daysToStress === 1 ? '' : 's'} to go
                    </span>
                  </div>
                  <p className="mt-1 text-[12px] text-slate-600">
                    until this field is likely to need water
                  </p>
                </div>
              )}

              <p className="mt-1.5 text-[13px] leading-relaxed text-ink-700">
                <span className="font-semibold text-ink-900">Why: </span>
                {why}
                {rain}
              </p>

              <p className="mt-1 flex items-center gap-1.5 text-[13px] text-slate-700">
                <span
                  aria-hidden="true"
                  className="h-1.5 w-1.5 shrink-0 rounded-full bg-brand-500"
                />
                {sky}
              </p>

              {/* The node's own state, the moment it last told the backend — the
                  closed loop, as the node reports it, on the card that issued
                  the plan it is running. Shown only while the node is fresh. */}
              {node && (
                <p className="mt-1 flex flex-wrap items-center gap-x-1.5 gap-y-0.5 rounded-panel bg-white/70 px-2.5 py-1.5 text-[12px] text-ink-700">
                  <span className="rounded-full bg-emerald-100 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-[0.08em] text-emerald-800">
                    ESP32 node
                  </span>
                  <span className="tnum text-ink-700">
                    posted{' '}
                    {relativeTime(node.recordedAt)}
                  </span>
                  {node.valveOpen != null && (
                    <span className="tnum">
                      · valve{' '}
                      <span
                        className={`font-bold ${
                          node.valveOpen ? 'text-brand-700' : 'text-ink-500'
                        }`}
                      >
                        {node.valveOpen ? 'OPEN' : 'CLOSED'}
                      </span>
                      {node.valveOpen &&
                        watered &&
                        field.live?.isLive &&
                        ' · running this plan'}
                    </span>
                  )}
                </p>
              )}

              <div className="mt-auto grid grid-cols-2 gap-2 pt-1.5">
                <button
                  type="button"
                  onClick={listen}
                  className="flex h-9 items-center justify-center gap-2 rounded-card border border-ink-200 bg-white text-[13px] font-semibold text-ink-900 transition-colors hover:border-ink-300 hover:bg-ink-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
                >
                  <SpeakerGlyph size={18} />
                  Listen
                </button>
                <button
                  type="button"
                  onClick={toggleDone}
                  aria-pressed={done}
                  className={`flex h-9 items-center justify-center gap-2 rounded-card border text-[13px] font-semibold transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600 ${
                    done
                      ? 'border-status-healthy bg-status-healthySoft text-[#15803d]'
                      : 'border-ink-200 bg-white text-ink-900 hover:border-ink-300 hover:bg-ink-50'
                  }`}
                >
                  <CheckGlyph size={18} />
                  {done ? (
                    <span className="flex flex-col items-start leading-tight">
                      <span>Done</span>
                      <span aria-hidden="true" className="text-[10px] font-medium">
                        Saved for today.
                      </span>
                    </span>
                  ) : (
                    'Mark done'
                  )}
                </button>
              </div>
            </div>
          </section>

          <WeekCard field={field} />

          {/* ---- The crop-photo check, embedded like the map cards ---------- */}
          {/* The same capture flow as the /crop-health page, compact. It follows
              the field being read — picking a field here moves the whole view,
              exactly like the map — and the answer renders in this card, so a
              farmer never leaves the field to have a plant checked. */}
          <section
            className="card flex h-full min-h-0 flex-col overflow-hidden"
            aria-label="Crop photo check"
          >
            <div className="flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto px-3 pb-2.5 pt-2.5">
              <CropHealthCheck
                compact
                fields={fields.map((item) => ({ id: item.fieldId, crop: item.crop }))}
                activeFieldId={field.fieldId}
                onFieldSelect={onSelect}
              />
            </div>
          </section>
        </div>

        {/* ---- Right side: the field, then water and yield ----------------- */}
        {/* The bottom row gets exactly one left-rail row's height: the rail is
            four 1fr rows with three 10px gaps, so one row is
            (100% - 30px)/4 — the same formula here keeps the soil-water and
            yield cards level with the four rail cards at any window size, and
            the map takes whatever is left. */}
        <div className="flex min-h-0 flex-col gap-2.5">
          <div className="min-h-0 flex-[3] xl:min-h-[12rem] xl:[flex:1_1_0%]">
            <FieldMap
              fields={plots}
              selectedId={field.fieldId}
              centre={centre ? { lat: centre.lat, lon: centre.lon } : null}
              onSelect={onSelect}
              className="h-full"
            />
          </div>

          {/* One rail-row's height when it fits, never smaller than its own
              content — so the soil-water and yield cards stay level with the
              left tiles on big screens and stay whole on small ones. With a
              live node attached, both slim down and the third slot carries
              the node's own sensor readings. */}
          <div
            className={
              field.live?.isLive
                ? 'grid min-h-0 flex-1 gap-2.5 sm:grid-cols-[minmax(0,0.95fr)_minmax(0,1fr)_minmax(0,0.95fr)] xl:[flex:0_1_calc((100%_-_30px)/4)] xl:min-h-[auto]'
                : 'grid min-h-0 flex-1 gap-2.5 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.08fr)] xl:[flex:0_1_calc((100%_-_30px)/4)] xl:min-h-[auto]'
            }
          >
            <SoilWaterCard field={field} band={band} />
            <YieldCard chips={chips} />
            {field.live?.isLive && <LiveNodeCard field={field} />}
          </div>
        </div>
      </div>
    </div>
  )
}

/* Band styles used by the soil-water ring. */
const BAND_STYLE: Record<Band, { text: string; from: string; to: string }> = {
  dry: { text: 'text-[#b45309]', from: '#fbbf24', to: '#ea580c' },
  good: { text: 'text-[#15803d]', from: '#4ade80', to: '#15803d' },
  wet: { text: 'text-brand-600', from: '#60a5fa', to: '#1d4ed8' },
  rest: { text: 'text-ink-500', from: '#cbd5e1', to: '#94a3b8' },
  none: { text: 'text-ink-500', from: '#e2e8f0', to: '#cbd5e1' },
}
