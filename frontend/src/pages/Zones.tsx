/**
 * Zones: per-zone control.
 *
 * Valve control writes to the device, so every control here is deliberately
 * explicit: the button states what it will do, the result is reported from the
 * server's response, and a manual override is visually separated from the
 * model's own recommendation so an operator is never confused about which one
 * is currently in charge.
 */

import { useCallback, useState } from 'react'
import { api, ApiError } from '../api/client'
import { useLive } from '../App'
import { usePollingFetch } from '../hooks/usePollingFetch'
import { Card, LinkBadge, MoistureChip, RiskChip, SeverityChip } from '../components/ui'
import { clockTime, duration, num, relativeTime, titleCase } from '../lib/format'
import type { ZoneSummary } from '../types'

export default function Zones() {
  const { bump, refresh } = useLive()
  const [busy, setBusy] = useState<string | null>(null)
  const [message, setMessage] = useState<{ tone: 'ok' | 'error'; text: string } | null>(null)

  const zones = usePollingFetch(useCallback(() => api.zones(), []), bump)
  const latest = usePollingFetch(useCallback(() => api.latest(), []), bump)

  const zoneList = zones.data?.zones ?? []
  const recommendations = latest.data?.recommendations ?? []

  const act = useCallback(
    async (zone: ZoneSummary, action: 'open' | 'close' | 'recompute') => {
      setBusy(`${zone.zone}:${action}`)
      setMessage(null)
      try {
        if (action === 'recompute') {
          const result = await api.recompute(zone.deviceId)
          setMessage({
            tone: 'ok',
            text: `${zone.zone}: ${titleCase(result.action)}, ${num(result.depthMm, 1)} mm.`,
          })
        } else {
          const result = await api.valve(zone.deviceId, action === 'open')
          setMessage({
            tone: 'ok',
            text: `${zone.zone}: valve command ${result.valveOpen ? 'OPEN' : 'CLOSED'}.`,
          })
        }
        refresh()
      } catch (cause) {
        setMessage({
          tone: 'error',
          text: cause instanceof ApiError ? cause.message : 'Command failed.',
        })
      } finally {
        setBusy(null)
      }
    },
    [refresh],
  )

  return (
    <>
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-[20px] font-semibold tracking-tight text-ink-900">Zones</h1>
          <p className="text-[12px] text-ink-500">
            Sensor readings, model decisions and manual valve control
          </p>
        </div>
        <button type="button" className="btn-secondary" onClick={zones.reload} disabled={zones.loading}>
          Refresh
        </button>
      </div>

      {message && (
        <div
          className={`rounded border px-4 py-2.5 text-[12px] ${
            message.tone === 'ok'
              ? 'border-status-healthy/30 bg-status-healthySoft text-status-healthy'
              : 'border-status-alert/30 bg-status-alertSoft text-status-alert'
          }`}
        >
          {message.text}
        </div>
      )}

      {zones.error && <Card title="Could not load zones"><p className="text-[13px] text-ink-500">{zones.error}</p></Card>}

      {zoneList.length === 0 && !zones.error ? (
        <Card title="No zones">
          <p className="text-[13px] text-ink-500">
            No zones have reported. Post telemetry to{' '}
            <code className="rounded bg-ink-100 px-1 py-0.5 font-mono text-[11px]">/api/telemetry</code>{' '}
            or run the node simulator.
          </p>
        </Card>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {zoneList.map((zone) => {
            const recommendation = recommendations.find((item) => item.zone === zone.zone)
            const isBusy = busy?.startsWith(`${zone.zone}:`) ?? false
            return (
              <Card key={zone.zone} title={zone.zone} action={<LinkBadge state={zone.linkState} />}>
                <div className="space-y-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[12px] text-ink-500">{zone.crop}</span>
                    <MoistureChip band={zone.moistureBand} />
                    {recommendation && <RiskChip level={recommendation.riskLevel} />}
                    {zone.valveOpen && (
                      <span className="chip bg-brand-100 text-brand-700">Valve open</span>
                    )}
                  </div>

                  <dl className="grid grid-cols-3 gap-3 border-y border-ink-100 py-3">
                    <Reading label="Moisture" value={`${num(zone.soilMoisture)}%`} />
                    <Reading label="Temperature" value={`${num(zone.temperature)} C`} />
                    <Reading label="Humidity" value={`${num(zone.humidity)}%`} />
                    <Reading label="pH" value={num(zone.soilPh)} />
                    <Reading label="EC" value={`${num(zone.ec, 2)} mS/cm`} />
                    <Reading label="Stress" value={num(zone.cropStressIndex, 2)} />
                    <Reading label="N" value={`${num(zone.nitrogen, 0)} kg/ha`} />
                    <Reading label="P" value={`${num(zone.phosphorus, 0)} kg/ha`} />
                    <Reading label="K" value={`${num(zone.potassium, 0)} kg/ha`} />
                  </dl>

                  {recommendation && (
                    <div className="rounded border border-ink-200 bg-ink-50 p-3">
                      <div className="flex items-center justify-between text-[12px]">
                        <span className="font-medium text-ink-900">
                          {recommendation.shouldIrrigate ? 'Irrigate' : 'Hold'} &middot;{' '}
                          {num(recommendation.depthMm, 1)} mm
                        </span>
                        <span className="tnum text-ink-500">
                          ET0 {num(recommendation.et0MmDay, 2)} mm/day
                        </span>
                      </div>
                      <p className="mt-1.5 text-[11px] leading-relaxed text-ink-500">
                        {recommendation.explanation[0] ?? 'No explanation recorded.'}
                      </p>
                      <p className="mt-1 text-[11px] text-ink-400">
                        {titleCase(recommendation.action)} &middot; confidence{' '}
                        {num(recommendation.confidence, 2)} &middot;{' '}
                        {clockTime(recommendation.recordedAt)}
                      </p>
                    </div>
                  )}

                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      className="btn-primary"
                      disabled={isBusy}
                      onClick={() => void act(zone, 'open')}
                    >
                      Open valve
                    </button>
                    <button
                      type="button"
                      className="btn-secondary"
                      disabled={isBusy}
                      onClick={() => void act(zone, 'close')}
                    >
                      Close valve
                    </button>
                    <button
                      type="button"
                      className="btn-ghost"
                      disabled={isBusy}
                      onClick={() => void act(zone, 'recompute')}
                    >
                      Re-run model
                    </button>
                    <span className="ml-auto text-[11px] text-ink-400">
                      {zone.valveOpen
                        ? `Running ${duration(zone.valveRuntimeSeconds)}`
                        : `Seen ${relativeTime(zone.lastSeen)}`}
                    </span>
                  </div>
                </div>
              </Card>
            )
          })}
        </div>
      )}

      <Card title="Recent events">
        <EventFeed />
      </Card>
    </>
  )
}

function Reading({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-[10px] font-medium uppercase tracking-[0.08em] text-ink-400">{label}</dt>
      <dd className="tnum mt-0.5 text-[13px] text-ink-900">{value}</dd>
    </div>
  )
}

function EventFeed() {
  const events = usePollingFetch(useCallback(() => api.events(25), []), 0, 20_000)
  const list = events.data?.events ?? []
  if (list.length === 0) {
    return <p className="py-6 text-center text-[12px] text-ink-400">No events recorded.</p>
  }
  return (
    <ul className="divide-y divide-ink-100">
      {list.map((event) => (
        <li key={event.id} className="flex items-center gap-3 py-2 text-[12px]">
          <span className="w-20 shrink-0 text-ink-400">{clockTime(event.createdAt)}</span>
          <span className="w-16 shrink-0 font-medium text-ink-700">{event.zone}</span>
          <span className="w-20 shrink-0 text-ink-500">{titleCase(event.kind)}</span>
          <span className="min-w-0 flex-1 truncate text-ink-700">{event.detail}</span>
          <span className="shrink-0 text-[11px] text-ink-400">{event.source}</span>
        </li>
      ))}
    </ul>
  )
}

export { SeverityChip }
