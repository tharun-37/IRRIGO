/**
 * The Water Advisor: the screen the product is really about.
 *
 * Each field gets one plain-language answer — irrigate, schedule, watch or hold
 * — with the reasoning that produced it. The numbers are not recomputed here:
 * the backend derives the whole advisory from a single FAO-56 plan, so the
 * headline, the stage, the root-zone balance and the schedule are guaranteed to
 * agree with one another. This page's job is to make that plan legible at a
 * glance, most urgent field first.
 */

import { useCallback } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api/client'
import { useLive } from '../App'
import { usePollingFetch } from '../hooks/usePollingFetch'
import { Card, DropGlyph, GaugeGlyph, LeafGlyph, Metric } from '../components/ui'
import { duration, int, num, pct, relativeTime, titleCase } from '../lib/format'
import type { Advisory, AdvisoryModels, AdvisoryStatus, ScheduleDay, Telemetry } from '../types'

const STATUS_STYLE: Record<
  AdvisoryStatus,
  { label: string; badge: string; ring: string; accent: string }
> = {
  irrigate: {
    label: 'Irrigate now',
    badge: 'bg-status-alert text-white',
    ring: 'ring-status-alertSoft',
    accent: 'text-status-alert',
  },
  schedule: {
    label: 'Schedule',
    badge: 'bg-status-warn text-white',
    ring: 'ring-status-warnSoft',
    accent: 'text-status-warn',
  },
  monitor: {
    label: 'Watch',
    badge: 'bg-brand-600 text-white',
    ring: 'ring-brand-100',
    accent: 'text-brand-700',
  },
  hold: {
    label: 'Hold',
    badge: 'bg-status-healthy text-white',
    ring: 'ring-status-healthySoft',
    accent: 'text-status-healthy',
  },
}

const TONE_TEXT: Record<string, string> = {
  healthy: 'text-status-healthy',
  warn: 'text-status-warn',
  neutral: 'text-ink-900',
}

function StatusBadge({ status }: { status: AdvisoryStatus }) {
  const style = STATUS_STYLE[status]
  return (
    <span className={`rounded px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em] ${style.badge}`}>
      {style.label}
    </span>
  )
}

function SensorTile({ label, value, unit }: { label: string; value: string; unit?: string }) {
  return (
    <div className="rounded border border-ink-200 px-3 py-2">
      <div className="tnum text-[15px] font-semibold text-ink-900">
        {value}
        {unit && <span className="ml-0.5 text-[11px] font-medium text-ink-400">{unit}</span>}
      </div>
      <div className="mt-0.5 text-[10px] font-medium uppercase tracking-[0.06em] text-ink-400">
        {label}
      </div>
    </div>
  )
}

function RootZoneBar({ advisory }: { advisory: Advisory }) {
  const fraction = Math.max(0, Math.min(1, advisory.water.depletionFraction))
  const stressAt =
    advisory.water.tawMm > 0
      ? Math.max(0, Math.min(1, advisory.water.rawMm / advisory.water.tawMm))
      : 0.55
  const stressed = fraction >= stressAt
  return (
    <div>
      <div className="flex items-center justify-between text-[11px] text-ink-500">
        <span>Root-zone depletion</span>
        <span className="tnum font-medium text-ink-700">
          {num(advisory.water.depletionMm)} / {num(advisory.water.tawMm)} mm
        </span>
      </div>
      <div className="relative mt-1.5 h-2 w-full overflow-hidden rounded-full bg-ink-100">
        <div
          className={`h-full rounded-full transition-[width] duration-500 ${
            stressed ? 'bg-status-warn' : 'bg-status-healthy'
          }`}
          style={{ width: `${fraction * 100}%` }}
        />
        <span
          className="absolute top-[-2px] h-3 w-0.5 bg-ink-900/60"
          style={{ left: `${stressAt * 100}%` }}
          title="Stress threshold (readily available water)"
        />
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-ink-400">
        <span>Field capacity</span>
        <span>Refill point ({pct(stressAt)})</span>
        <span>Wilted</span>
      </div>
    </div>
  )
}

const AGREEMENT_STYLE: Record<string, { label: string; tone: string }> = {
  aligned: { label: 'Aligned', tone: 'text-status-healthy' },
  agree_hold: { label: 'Aligned on hold', tone: 'text-status-healthy' },
  v1_conservative: { label: 'Decision more cautious', tone: 'text-status-warn' },
  v2_only: { label: 'Requirement flags demand', tone: 'text-status-warn' },
  not_applicable: { label: 'No active crop', tone: 'text-ink-400' },
  model_unavailable: { label: 'Requirement model off', tone: 'text-ink-400' },
}

/**
 * The two learned models side by side.
 *
 * V1's controller sets the event decision; V2's estimator sets the fortnight's
 * requirement and its uncertainty. Showing them together is the point: when they
 * disagree, the operator can see it rather than trusting a silently blended
 * number.
 */
function ModelFusionCard({ models }: { models: AdvisoryModels }) {
  const agreeing =
    models.fusion.agreement === 'aligned' || models.fusion.agreement === 'agree_hold'
  const style = AGREEMENT_STYLE[models.fusion.agreement] ?? {
    label: models.fusion.agreement,
    tone: 'text-ink-900',
  }
  const interval = models.v2.intervalMm
  return (
    <div className="rounded border border-ink-200 bg-ink-50 p-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-medium uppercase tracking-[0.08em] text-ink-400">
          Both models
        </span>
        <span className={`text-[11px] font-semibold ${style.tone}`}>
          {agreeing ? '✓ ' : ''}
          {style.label}
        </span>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-3">
        <div>
          <div className="text-[10px] font-medium uppercase tracking-[0.06em] text-ink-400">
            V2 · 14-day need
          </div>
          <div className="tnum text-[15px] font-semibold text-ink-900">
            {models.v2.predictionMm === null ? '—' : `${num(models.v2.predictionMm, 1)} mm`}
          </div>
          <div className="text-[10px] text-ink-400">
            {!models.v2.applicable
              ? models.v2.reason ?? 'not applicable'
              : interval
                ? `${num(interval[0], 0)}–${num(interval[1], 0)} mm @ ${pct(models.v2.coverage ?? 0)}`
                : 'no interval'}
          </div>
        </div>
        <div>
          <div className="text-[10px] font-medium uppercase tracking-[0.06em] text-ink-400">
            V1 · this event
          </div>
          <div className="tnum text-[15px] font-semibold text-ink-900">
            {num(models.fusion.v1DepthMm, 1)} mm
          </div>
          <div className="text-[10px] text-ink-400">
            {models.v1.shouldIrrigate ? 'irrigate' : 'hold'}
            {models.v1.probability !== null && models.v1.threshold !== null
              ? ` · ${pct(models.v1.probability)} vs ${pct(models.v1.threshold)}`
              : ''}
          </div>
        </div>
      </div>
      <p className="mt-2 text-[10px] leading-snug text-ink-400">{models.fusion.note}</p>
    </div>
  )
}

function ScheduleBars({ schedule }: { schedule: ScheduleDay[] }) {
  const peak = Math.max(1, ...schedule.map((day) => day.grossMm))
  return (
    <div className="flex items-end gap-1.5">
      {schedule.map((day) => {
        const height = day.grossMm > 0 ? Math.max(8, (day.grossMm / peak) * 46) : 3
        return (
          <div key={day.date} className="flex flex-1 flex-col items-center gap-1">
            <div className="flex h-[48px] w-full items-end">
              <div
                className={`w-full rounded-sm ${
                  day.grossMm > 0 ? 'bg-brand-500' : 'bg-ink-200'
                }`}
                style={{ height }}
                title={day.grossMm > 0 ? `${num(day.grossMm, 1)} mm gross` : 'No application'}
              />
            </div>
            <span className="text-[9px] text-ink-400">+{day.dayOffset}d</span>
          </div>
        )
      })}
    </div>
  )
}

function AdvisoryCard({ advisory }: { advisory: Advisory }) {
  const style = STATUS_STYLE[advisory.status]
  const sensors: Telemetry = advisory.sensors
  const rec = advisory.recommendation
  // The fused depth, when the pipeline served it: an escalation can recommend
  // water even though the V1 controller alone would have held.
  const depthMm = rec.finalDepthMm ?? (rec.shouldIrrigate ? rec.depthMm : 0)
  return (
    <section className={`card ring-1 ${style.ring}`}>
      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.55fr)_minmax(0,1fr)]">
        <div className="card-body space-y-4">
          <div className="flex flex-wrap items-start gap-3">
            <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-brand-600 text-white">
              <LeafGlyph size={18} />
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-[15px] font-semibold text-ink-900">{advisory.fieldId}</h2>
                <StatusBadge status={advisory.status} />
                <span className="text-[11px] text-ink-400">
                  {advisory.crop} · {advisory.station} · {advisory.soilType}
                </span>
              </div>
              <p className={`mt-1 text-[17px] font-semibold tracking-tight ${style.accent}`}>
                {advisory.headline}
              </p>
            </div>
            <div className="text-right">
              <div className="tnum text-[22px] font-semibold leading-none text-ink-900">
                {advisory.priority}
                <span className="ml-0.5 text-[12px] font-medium text-ink-400">/100</span>
              </div>
              <div className="mt-1 text-[10px] font-medium uppercase tracking-[0.08em] text-ink-400">
                Urgency
              </div>
            </div>
          </div>

          <p className="text-[12px] leading-relaxed text-ink-600">{advisory.summary}</p>

          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <div className="rounded border border-ink-200 bg-ink-50 px-3 py-2">
              <div className="text-[10px] font-medium uppercase tracking-[0.06em] text-ink-400">
                Day
              </div>
              <div className="tnum text-[15px] font-semibold text-ink-900">
                {advisory.ageDays}
                <span className="ml-0.5 text-[11px] font-medium text-ink-400">d</span>
              </div>
            </div>
            <div className="rounded border border-ink-200 bg-ink-50 px-3 py-2">
              <div className="text-[10px] font-medium uppercase tracking-[0.06em] text-ink-400">
                Stage
              </div>
              <div className="truncate text-[12px] font-semibold text-ink-900" title={advisory.stageLabel}>
                {advisory.stageLabel}
              </div>
            </div>
            <div className="rounded border border-ink-200 bg-ink-50 px-3 py-2">
              <div className="text-[10px] font-medium uppercase tracking-[0.06em] text-ink-400">
                Season
              </div>
              <div className="tnum text-[15px] font-semibold text-ink-900">
                {pct(advisory.stageProgress)}
              </div>
            </div>
            <div className="rounded border border-ink-200 bg-ink-50 px-3 py-2">
              <div className="text-[10px] font-medium uppercase tracking-[0.06em] text-ink-400">
                GDD
              </div>
              <div className="tnum text-[15px] font-semibold text-ink-900">{int(advisory.gdd)}</div>
            </div>
          </div>

          <RootZoneBar advisory={advisory} />

          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
            <SensorTile label="Soil moisture" value={num(sensors.soilMoisture)} unit="%" />
            <SensorTile label="Soil temp" value={num(sensors.temperature)} unit="°C" />
            <SensorTile label="Humidity" value={num(sensors.humidity)} unit="%" />
            <SensorTile label="EC" value={num(sensors.ec, 2)} unit="mS" />
            <SensorTile label="pH" value={num(sensors.soilPh)} />
            <SensorTile
              label="N-P-K"
              value={`${num(sensors.nitrogen, 0)}/${num(sensors.phosphorus, 0)}/${num(sensors.potassium, 0)}`}
            />
          </div>

          <div className="flex flex-wrap gap-x-6 gap-y-2">
            {advisory.reasoning.map((item) => (
              <div key={item.label} className="min-w-[140px]">
                <div className="text-[10px] font-medium uppercase tracking-[0.06em] text-ink-400">
                  {item.label}
                </div>
                <div className={`text-[13px] font-semibold ${TONE_TEXT[item.tone] ?? 'text-ink-900'}`}>
                  {item.value}
                </div>
                <div className="text-[10px] leading-snug text-ink-400">{item.detail}</div>
              </div>
            ))}
          </div>

          {advisory.models && <ModelFusionCard models={advisory.models} />}
        </div>

        <div className="border-t border-ink-200 p-5 lg:border-l lg:border-t-0">
          <div className="flex items-start justify-between">
            <div>
              <div className="text-[10px] font-medium uppercase tracking-[0.08em] text-ink-400">
                Recommendation
              </div>
              <div className="tnum mt-1 text-[26px] font-semibold leading-none text-ink-900">
                {depthMm > 0 ? num(depthMm, 1) : '0'}
                <span className="ml-1 text-[13px] font-medium text-ink-400">mm</span>
              </div>
            </div>
            <span className="flex h-9 w-9 items-center justify-center rounded-full bg-brand-50 text-brand-700">
              <DropGlyph size={17} />
            </span>
          </div>

          <dl className="mt-3 space-y-1.5 text-[11px]">
            <div className="flex justify-between">
              <dt className="text-ink-500">Volume</dt>
              <dd className="tnum font-medium text-ink-700">{int(rec.volumeLitres)} L</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-ink-500">Pump runtime</dt>
              <dd className="tnum font-medium text-ink-700">{duration(rec.durationSeconds)}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-ink-500">Model confidence</dt>
              <dd className="tnum font-medium text-ink-700">{pct(rec.confidence)}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-ink-500">ET0 / crop use</dt>
              <dd className="tnum font-medium text-ink-700">
                {num(advisory.water.et0MmDay)} / {num(advisory.water.etcMmDay)} mm
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-ink-500">Root depth</dt>
              <dd className="tnum font-medium text-ink-700">{num(advisory.water.rootDepthCm, 0)} cm</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-ink-500">Disease risk</dt>
              <dd className="font-medium text-ink-700">
                {titleCase(advisory.risk.diseaseRisk ?? '—')}
              </dd>
            </div>
          </dl>

          <div className="mt-4">
            <div className="mb-1.5 flex items-center justify-between">
              <span className="text-[10px] font-medium uppercase tracking-[0.08em] text-ink-400">
                Next 7 days
              </span>
              <span className="text-[10px] text-ink-400">net → gross</span>
            </div>
            <ScheduleBars schedule={advisory.schedule} />
          </div>

          <div className="mt-4 flex gap-2">
            <Link to="/zones" className="btn-secondary flex-1 text-center">
              Zone detail
            </Link>
            <Link to="/model" className="btn-secondary flex-1 text-center">
              Why?
            </Link>
          </div>
        </div>
      </div>
    </section>
  )
}

export default function Advisor() {
  const { bump } = useLive()
  const feed = usePollingFetch(useCallback(() => api.advisor(), []), bump, 15_000)

  const fields = feed.data?.fields ?? []
  const fleet = feed.data?.fleet

  if (feed.error) {
    return (
      <Card title="Advisor unavailable">
        <p className="text-[13px] text-ink-500">{feed.error}</p>
        <p className="mt-2 text-[12px] text-ink-400">
          The advisor needs the weather corpus and the sowing registry. Check the
          API log, then reload.
        </p>
      </Card>
    )
  }

  return (
    <>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-[20px] font-semibold tracking-tight text-ink-900">Water Advisor</h1>
          <p className="text-[12px] text-ink-500">
            Automated irrigation guidance from live soil sensors, crop stage and the
            FAO-56 water balance
          </p>
        </div>
        <div className="text-right text-[11px] text-ink-400">
          <div>{feed.data ? `Updated ${relativeTime(feed.data.generatedAt)}` : 'Loading…'}</div>
          <div>Crop-stage aware · ET0 from NASA POWER</div>
        </div>
      </div>

      <Card bodyClassName="">
        <div className="grid grid-cols-1 divide-y divide-ink-200 sm:grid-cols-3 sm:divide-x sm:divide-y-0">
          <div className="p-5">
            <Metric
              value={fleet ? fleet.irrigate : '—'}
              caption="Fields needing water now"
            />
          </div>
          <div className="p-5">
            <Metric
              value={fleet ? fleet.monitor : '—'}
              caption="Fields to watch this cycle"
            />
          </div>
          <div className="p-5">
            <Metric
              value={fleet ? num(fleet.recommendedMm, 1) : '—'}
              unit="mm"
              caption="Total recommended depth"
            />
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_280px]">
        <div className="space-y-4">
          {feed.loading && fields.length === 0 && (
            <Card>
              <p className="py-10 text-center text-[12px] text-ink-400">
                Building the first advisory from the weather corpus…
              </p>
            </Card>
          )}
          {!feed.loading && fields.length === 0 && (
            <Card>
              <p className="py-10 text-center text-[12px] text-ink-400">
                No fields registered. Register a sowing to receive advice.
              </p>
            </Card>
          )}
          {fields.map((advisory) => (
            <AdvisoryCard key={advisory.fieldId} advisory={advisory} />
          ))}
        </div>

        <aside className="space-y-4">
          <Card title="Fleet">
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <span className="flex h-9 w-9 items-center justify-center rounded-full bg-brand-600 text-white">
                  <GaugeGlyph size={17} />
                </span>
                <div>
                  <div className="tnum text-[22px] font-semibold leading-none text-ink-900">
                    {fleet ? fleet.fields : '—'}
                  </div>
                  <div className="mt-1 text-[10px] font-medium uppercase tracking-[0.08em] text-ink-400">
                    Fields monitored
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2 text-center">
                <div className="rounded border border-ink-200 py-2">
                  <div className="tnum text-[16px] font-semibold text-status-alert">
                    {fleet?.irrigate ?? '—'}
                  </div>
                  <div className="text-[10px] text-ink-400">Irrigate</div>
                </div>
                <div className="rounded border border-ink-200 py-2">
                  <div className="tnum text-[16px] font-semibold text-status-warn">
                    {fleet?.monitor ?? '—'}
                  </div>
                  <div className="text-[10px] text-ink-400">Watch</div>
                </div>
                <div className="rounded border border-ink-200 py-2">
                  <div className="tnum text-[16px] font-semibold text-status-healthy">
                    {fleet?.hold ?? '—'}
                  </div>
                  <div className="text-[10px] text-ink-400">Hold</div>
                </div>
              </div>
              <div className="rounded border border-ink-200 bg-ink-50 px-3 py-2">
                <div className="text-[10px] font-medium uppercase tracking-[0.06em] text-ink-400">
                  Recommended volume
                </div>
                <div className="tnum text-[16px] font-semibold text-ink-900">
                  {fleet ? int(fleet.recommendedLitres) : '—'}
                  <span className="ml-1 text-[11px] font-medium text-ink-400">L</span>
                </div>
              </div>
            </div>
          </Card>

          <Card title="How to read this">
            <ul className="space-y-2 text-[11px] leading-relaxed text-ink-500">
              <li>
                <span className="font-semibold text-status-alert">Irrigate now</span> — the root
                zone has reached the refill point for its current crop stage.
              </li>
              <li>
                <span className="font-semibold text-status-warn">Schedule</span> — water is
                required within the plan horizon; act before stress begins.
              </li>
              <li>
                <span className="font-semibold text-brand-700">Watch</span> — dry enough to
                monitor; the stress horizon is close.
              </li>
              <li>
                <span className="font-semibold text-status-healthy">Hold</span> — the profile has
                enough plant-available water for now.
              </li>
              <li className="pt-1 text-ink-400">
                Depth is net application; the schedule shows gross depth after the method's
                efficiency.
              </li>
            </ul>
          </Card>
        </aside>
      </div>
    </>
  )
}
