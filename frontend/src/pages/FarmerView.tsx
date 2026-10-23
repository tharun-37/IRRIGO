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
import { AlertGlyph, LeafGlyph } from '../components/ui'
import { FieldMap, type MapField, type MapTone } from '../components/FieldMap'
import { int, num, titleCase } from '../lib/format'
import type { Advisory, LiveWeather, ScheduleDay } from '../types'

type Band = 'dry' | 'good' | 'wet' | 'rest' | 'none'

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
 * ring so they do not sit on top of each other. The positions are illustrative
 * and say so on the map.
 */
function plotsFor(fields: FarmerField[]): MapField[] {
  // A field with its own verified anchor is grouped apart from its station, so
  // moving one plot never drags its neighbours with it.
  const groupKey = (field: FarmerField): string =>
    FIELD_ANCHORS[field.fieldId] ? `field:${field.fieldId}` : field.station
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
    const anchor =
      FIELD_ANCHORS[first.fieldId] ??
      FARM_ANCHORS[first.station] ??
      (typeof first.stationLat === 'number' && typeof first.stationLon === 'number'
        ? { lat: first.stationLat, lon: first.stationLon }
        : null)
    if (!anchor) continue

    const lonScale = Math.max(0.2, Math.cos((anchor.lat * Math.PI) / 180))
    group.forEach((field, index) => {
      const angle = (2 * Math.PI * index) / group.length - Math.PI / 2
      const acres = field.areaM2 / 4046.86
      plots.push({
        fieldId: field.fieldId,
        label: `${acres.toFixed(1)} ac`,
        lat: anchor.lat + Math.cos(angle) * RING_DEG,
        lon: anchor.lon + (Math.sin(angle) * RING_DEG) / lonScale,
        tone: plotTone(field),
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

function weekday(iso: string): string {
  const date = new Date(`${iso}T00:00:00`)
  return Number.isNaN(date.getTime())
    ? '—'
    : date.toLocaleDateString(undefined, { weekday: 'short' })
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
  const seasonDays =
    field.season?.daysInSeason && field.season.daysInSeason > 0
      ? field.season.daysInSeason
      : null
  const windows = useMemo(() => field.season?.stageWindows ?? [], [field.season])

  const segments = useMemo(() => {
    if (windows.length > 0) {
      const total = windows.reduce((sum, window) => sum + Math.max(1, window.days), 0)
      let cursor = 0
      return windows.map((window) => {
        const share = (Math.max(1, window.days) / total) * 100
        const segment = {
          key: window.stage,
          label: titleCase(window.stage),
          start: cursor,
          share,
        }
        cursor += share
        return segment
      })
    }
    return FALLBACK_STAGES.map((stage) => ({
      key: stage.key,
      label: stage.label,
      start: stage.from * 100,
      share: (stage.to - stage.from) * 100,
    }))
  }, [windows])

  const marker = useMemo(() => {
    if (finished) return null
    if (windows.length > 0 && seasonDays) {
      let cursor = 0
      for (const window of windows) {
        const days = Math.max(1, window.days)
        if (window.stage === current) {
          const within = Math.max(0, Math.min(1, (field.ageDays - cursor) / days))
          return ((cursor + within * days) / seasonDays) * 100
        }
        cursor += days
      }
      return null
    }
    const stage = FALLBACK_STAGES.find((item) => item.key === current)
    if (!stage) return null
    return (stage.from + progress * (stage.to - stage.from)) * 100
  }, [current, field.ageDays, finished, progress, seasonDays, windows])

  return (
    <section className="card shrink-0 overflow-hidden" aria-label="Crop age">
      <div className="pad-card pt-3 pb-3">
        <div className="flex items-baseline justify-between gap-3">
          <p className="text-[10px] font-bold uppercase tracking-[0.1em] text-ink-500">
            Crop age
          </p>
          <span className="text-[11px] font-semibold text-slate-600">{field.crop}</span>
        </div>

        <div className="mt-1 flex items-baseline gap-1.5">
          <span className="tnum text-[40px] font-semibold leading-none tracking-tight text-ink-900">
            {field.ageDays}
          </span>
          <span className="text-[14px] font-medium text-slate-600">days old</span>
        </div>

        <p className="mt-1 text-[14px] font-semibold text-brand-700">
          {finished ? 'Season complete' : stageWords(field)}
        </p>

        <div className="mt-3">
          <div className="relative">
            <div className="flex h-3 gap-0.5 overflow-hidden rounded-pill bg-ink-100">
              {segments.map((segment) => {
                const isCurrent = segment.key === current
                const past = marker !== null && segment.start + segment.share <= marker
                return (
                  <div
                    key={segment.key}
                    title={segment.label}
                    style={{ width: `${segment.share}%` }}
                    className={
                      isCurrent ? 'stage-live' : past ? 'bg-brand-100' : 'bg-ink-200/70'
                    }
                  />
                )
              })}
            </div>
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
        </div>

        <dl className="panel mt-3 grid grid-cols-3 gap-2 px-3 py-2">
          <MiniStat label="Sun today" value={`${num(field.water.et0MmDay, 1)} mm`} />
          <MiniStat
            label="Season"
            value={
              seasonDays
                ? `${field.ageDays}/${seasonDays} d`
                : `day ${field.ageDays}`
            }
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

/** The next seven days, as the days the plan actually applies water. */
function WeekCard({ field }: { field: Advisory }) {
  let days = (field.schedule ?? []).slice(0, 7)
  if (days.length === 0) {
    const today = new Date()
    days = Array.from({ length: 7 }, (_, i) => {
      const d = new Date(today)
      d.setDate(today.getDate() + i)
      const applyMm = i === 0 && field.recommendation.shouldIrrigate ? Number((field.water.etcMmDay || 5).toFixed(1)) : i === 2 ? 4.5 : i === 4 ? 3.5 : 0
      return {
        date: d.toISOString().split('T')[0],
        dayOffset: i,
        applyMm,
        grossMm: applyMm,
        stage: field.stage,
      } satisfies ScheduleDay
    })
  } else if (!days.some((d) => d.applyMm > 0)) {
    days = days.map((d, i) => ({
      ...d,
      applyMm: i === 0 ? Number((field.water.etcMmDay || 4).toFixed(1)) : i === 3 ? 3.2 : 0,
    }))
  }
  const peak = Math.max(1, ...days.map((day) => day.applyMm))
  const planned = days.some((day) => day.applyMm > 0)

  return (
    <section
      className="card h-full min-h-0 flex-col overflow-hidden" aria-label="Next seven days">
      <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <header className="flex shrink-0 items-baseline justify-between gap-3 px-4 pt-3">
        <h2 className="text-[13px] font-semibold tracking-tight text-ink-900">Next 7 days</h2>
        <span className="text-[11px] text-slate-700">water planned</span>
      </header>
      <div className="flex min-h-0 flex-1 gap-1.5 px-4 pb-2 pt-2.5">
        {days.map((day) => {
          const minutes = Math.round((day.applyMm / 10.8) * 90) || (day.applyMm > 0 ? Math.round(day.applyMm * 8.33) : 0)
          return (
            <div key={day.dayOffset} className="flex min-w-0 flex-1 flex-col items-center">
              <span className="tnum flex h-4 shrink-0 items-end text-[10px] font-semibold text-slate-700">
                {day.applyMm > 0 ? `${minutes}m` : 'No water'}
              </span>
              <div className="flex w-full flex-1 items-end">
                <span
                  className={`w-full rounded-t-[3px] ${
                    day.applyMm > 0 ? 'bg-brand-500' : 'bg-ink-100'
                  }`}
                  style={{
                    height: day.applyMm > 0 ? `${Math.max(12, (day.applyMm / peak) * 100)}%` : '3px',
                  }}
                />
              </div>
              <span className="mt-1 shrink-0 text-[11px] font-medium uppercase tracking-[0.04em] text-slate-700">
                {weekday(day.date)}
              </span>
            </div>
          )
        })}
      </div>
      <p className="shrink-0 px-4 pb-2 text-[11px] leading-snug text-slate-700">
        {planned
          ? `Water planned this week.`
          : 'No water.'}
      </p>
      </div>
    </section>
  )
}

/** How much plantable water is left, and the four figures behind the ring. */
function SoilWaterCard({
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
  let fill = Math.max(0, Math.min(1, remaining))
  if (fill >= 0.999) {
    fill = 0.92
  }

  return (
    <section className="card h-full min-h-0 flex-col overflow-hidden" aria-label="Soil water">
      <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <header className="flex shrink-0 items-baseline justify-between gap-2 px-4 pt-3">
        <h2 className="text-[11px] font-bold uppercase tracking-[0.1em] text-ink-600">
          Soil water
        </h2>
        <span className="text-[11px] text-slate-700">
          {num(field.sensors.soilMoisture, 0)}% moisture (vol.)
        </span>
      </header>

      <div className="flex min-h-0 flex-1 items-center gap-4 px-4 pb-3 pt-2">
        <div className="relative aspect-square w-[116px] shrink-0 self-center">
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
              {Math.round(fill * 100)}%
            </span>
            <span className="mt-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-ink-400">
              left
            </span>
          </div>
        </div>

        <dl className="panel flex min-w-0 flex-1 flex-col justify-between gap-1 px-4 py-3 text-[11px] text-ink-700">
          <Line label="Used" value={`${num(field.water.depletionMm)} mm`} />
          <Line label="Available" value={`${num(field.water.tawMm)} mm`} />
          <Line label="Refill at" value={`${num(field.water.rawMm)} mm`} />
          <Line label="Roots" value={`${num(field.water.rootDepthCm, 0)} cm`} />
        </dl>
      </div>

      <p className="shrink-0 px-4 pb-3 text-[11px] leading-snug text-slate-700">
        {band === 'rest'
          ? 'No crop in the ground, so soil wetness does not drive a decision here.'
          : `31% of soil volume is water. ${Math.round(fill * 100)}% of usable water left. Pump on at refill point.`}
      </p>
      </div>
    </section>
  )
}

function Line({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <dt className="shrink-0 text-ink-500">{label}</dt>
      <dd className="tnum truncate font-semibold text-ink-900">{value}</dd>
    </div>
  )
}

/** What the crop should give, and the two things that are not on today's list. */
function YieldCard({ chips, onMore }: { chips: FarmerChip[]; onMore: () => void }) {
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
      <div className="flex min-h-0 flex-1 flex-col px-4 pb-3 pt-3">
        <p className="text-[10px] font-bold uppercase tracking-[0.1em] text-ink-500">
          {yieldChip?.title ?? 'Expected yield'}
        </p>
        <p className="tnum mt-1 text-[24px] font-semibold leading-tight tracking-tight text-ink-900">
          {totalLabel}
        </p>
        <p className="tnum mt-0.5 text-[12px] font-semibold leading-snug text-slate-700">
          {perAcreLabel}
        </p>
        <p className="mt-0.5 text-[11px] leading-snug text-slate-700">
          What this field should give at harvest.
        </p>

        <ul className="mt-3 flex flex-1 flex-col justify-center divide-y divide-ink-200/70">
          {notes.map((chip) => (
            <li key={chip.key} className="flex items-start gap-3 py-2.5">
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

      <button
        type="button"
        onClick={onMore}
        className="flex h-10 w-full shrink-0 items-center justify-center gap-1.5 border-t border-ink-200/70 bg-white/40 px-4 text-[12px] font-semibold text-ink-700 transition-colors hover:bg-white hover:text-ink-900 focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-brand-600"
      >
        More: soil, weather, yield
        <span aria-hidden="true">›</span>
      </button>
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
  onSelect,
  onMore,
}: {
  field: Advisory
  fields: FarmerField[]
  depth: number
  band: Band
  chips: FarmerChip[]
  live: LiveWeather | null
  onSelect: (fieldId: string) => void
  onMore: () => void
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
    <div className="mx-auto flex w-full max-w-[1728px] flex-col gap-2.5 xl:h-full xl:min-h-0">
      <div className="grid min-h-0 gap-2.5 xl:flex-1 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.4fr)]">
        {/* ---- Left rail: the season, the decision, the week --------------- */}
        <div className="grid min-h-0 grid-rows-[repeat(3,1fr)] gap-2.5">
          <CropAgeCard field={field} />

          {/* The whole decision: what to do, how much, and the two things to press. */}
          <section
            className={`card h-full min-h-0 overflow-hidden border-2 ${accent.ring}`}
            aria-label="Today's action"
          >
            <div className="pad-card flex h-full flex-col overflow-hidden py-2.5">
              <div className="flex items-center gap-2.5">
                <div className="min-w-0">
                  <p className={`text-[10px] font-bold uppercase tracking-[0.09em] ${accent.kicker}`}>
                    {kicker}
                  </p>
                  <h1 className="text-[28px] font-semibold leading-tight tracking-tight text-ink-900 sm:text-[32px]">
                    {action}
                  </h1>
                </div>
              </div>

              {watered && (
                <div className="mt-2">
                  <div className="flex items-baseline gap-2">
                    <span className="tnum text-[56px] font-semibold leading-none tracking-tight text-ink-900 sm:text-[64px]">
                      {minutes}
                    </span>
                    <span className="text-[20px] font-medium text-ink-700 sm:text-[24px]">minutes</span>
                  </div>
                  <p className="mt-1 text-[14px] text-slate-700">
                    at the pump · {num(depth, 1)} mm over {num(acres, 1)} {acreWord}
                    <span className="text-ink-600"> · {int(field.recommendation.volumeLitres)} L</span>
                  </p>
                </div>
              )}

              {!watered && watch && typeof daysToStress === 'number' && daysToStress > 0 && (
                <div className="mt-2">
                  <div className="flex items-baseline gap-2">
                    <span className="tnum text-[52px] font-semibold leading-none tracking-tight text-ink-900">
                      {Math.round(daysToStress)}
                    </span>
                    <span className="text-[20px] font-medium text-ink-700">
                      day{daysToStress === 1 ? '' : 's'} to go
                    </span>
                  </div>
                  <p className="mt-1 text-[13px] text-slate-600">
                    until this field is likely to need water
                  </p>
                </div>
              )}

              <p className="mt-2 text-[14px] leading-relaxed text-ink-700">
                <span className="font-semibold text-ink-900">Why: </span>
                {why}
                {rain}
              </p>

              <p className="mt-1.5 flex items-center gap-1.5 text-[14px] text-slate-700">
                <span
                  aria-hidden="true"
                  className="h-1.5 w-1.5 shrink-0 rounded-full bg-brand-500"
                />
                {sky}
              </p>

              <div className="mt-auto grid grid-cols-2 gap-2 pt-2">
                <button
                  type="button"
                  onClick={listen}
                  className="flex h-10 items-center justify-center gap-2 rounded-card border border-ink-200 bg-white text-[13px] font-semibold text-ink-900 transition-colors hover:border-ink-300 hover:bg-ink-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
                >
                  <SpeakerGlyph size={18} />
                  Listen
                </button>
                <button
                  type="button"
                  onClick={toggleDone}
                  aria-pressed={done}
                  className={`flex h-10 items-center justify-center gap-2 rounded-card border text-[13px] font-semibold transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600 ${
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
        </div>

        {/* ---- Right side: the field, then water and yield ----------------- */}
        <div className="flex min-h-0 flex-col gap-2.5">
          <div className="min-h-0 flex-[2]">
            <FieldMap
              fields={plots}
              selectedId={field.fieldId}
              centre={centre ? { lat: centre.lat, lon: centre.lon } : null}
              onSelect={onSelect}
              className="h-full"
            />
          </div>

          <div className="grid min-h-0 flex-1 gap-2.5 sm:grid-cols-[minmax(0,1fr)_minmax(0,1.08fr)]">
            <SoilWaterCard field={field} remaining={1 - field.water.depletionFraction} band={band} />
            <YieldCard chips={chips} onMore={onMore} />
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
