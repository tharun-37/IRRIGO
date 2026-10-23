/**
 * Overview: the screen an operator looks at first.
 *
 * Answers three questions in order: is the system healthy, what is each zone
 * doing right now, and is anything wrong. Detail lives on the other pages, so
 * this one favours a summary that can be read in a few seconds.
 */

import { useCallback, useMemo } from 'react'
import { Link } from 'react-router-dom'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api/client'
import { useLive } from '../App'
import { usePollingFetch } from '../hooks/usePollingFetch'
import {
  AlertGlyph,
  Card,
  DropGlyph,
  GaugeGlyph,
  IconStat,
  LinkBadge,
  Metric,
  MoistureChip,
  RiskChip,
  StatRow,
} from '../components/ui'
import { clockTime, duration, int, num, relativeTime, titleCase } from '../lib/format'
import type { ZoneSummary } from '../types'

const chartAxis = { stroke: '#cbd5e1', fontSize: 11 }

export default function Overview() {
  const { bump, telemetry } = useLive()

  const health = usePollingFetch(useCallback(() => api.health(), []), bump, 15_000)
  const zones = usePollingFetch(useCallback(() => api.zones(), []), bump)
  const alerts = usePollingFetch(useCallback(() => api.alerts(true, 20), []), bump)

  const zoneList = useMemo<ZoneSummary[]>(() => zones.data?.zones ?? [], [zones.data])

  // Live frames arrive newest first; the chart reads left to right, so reverse.
  const moistureSeries = useMemo(
    () =>
      [...telemetry]
        .slice(0, 60)
        .reverse()
        .map((reading) => ({
          label: clockTime(reading.recordedAt),
          moisture: reading.soilMoisture,
          temperature: reading.temperature,
        })),
    [telemetry],
  )

  const bandCounts = useMemo(() => {
    const counts = { dry: 0, optimal: 0, wet: 0 }
    for (const zone of zoneList) counts[zone.moistureBand] += 1
    return counts
  }, [zoneList])

  const bandChart = useMemo(
    () => [
      { name: 'Dry', fill: '#f59e0b', value: bandCounts.dry },
      { name: 'Optimal', fill: '#16a34a', value: bandCounts.optimal },
      { name: 'Wet', fill: '#3b82f6', value: bandCounts.wet },
    ],
    [bandCounts],
  )

  const openAlerts = alerts.data?.alerts ?? []
  const h = health.data

  if (health.error) {
    return (
      <Card title="Backend unreachable">
        <p className="text-[13px] text-ink-500">{health.error}</p>
        <p className="mt-2 text-[12px] text-ink-400">
          Start the API with{' '}
          <code className="rounded bg-ink-100 px-1 py-0.5 font-mono text-[11px]">
            python -m uvicorn app.main:app --port 8000
          </code>{' '}
          from the <code className="font-mono">backend</code> directory.
        </p>
      </Card>
    )
  }

  return (
    <>
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-[20px] font-semibold tracking-tight text-ink-900">Overview</h1>
          <p className="text-[12px] text-ink-500">
            Live soil, water and model state across all zones
          </p>
        </div>
        <Link to="/zones" className="btn-secondary">
          Manage valves
        </Link>
      </div>

      <Card bodyClassName="">
        <StatRow
          items={[
            { value: h ? `${h.onlineDevices}/${h.deviceCount}` : '—', label: 'Devices online' },
            { value: int(h?.readingsToday), label: 'Readings today' },
            { value: int(h?.openAlerts), label: 'Open alerts' },
            { value: h ? duration(h.uptimeSeconds) : '—', label: 'Uptime' },
          ]}
        />
        <div className="grid grid-cols-1 divide-y divide-ink-200 sm:grid-cols-3 sm:divide-x sm:divide-y-0">
          <div className="p-5">
            <Metric value={num(h?.inferenceLatencyMs, 2)} unit="ms" caption="Inference latency" />
          </div>
          <div className="p-5">
            <Metric
              value={h?.modelMetrics ? num(h.modelMetrics.waterDepth.mae, 2) : '—'}
              unit="mm"
              caption="Water depth MAE (holdout)"
            />
          </div>
          <div className="p-5">
            <Metric
              value={h?.modelMetrics ? num(h.modelMetrics.disease.macroF1, 3) : '—'}
              caption="Disease risk macro F1"
            />
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card title="System" className="lg:col-span-1">
          <div className="space-y-5">
            <IconStat
              tone={h?.status === 'ok' ? 'healthy' : 'warn'}
              glyph={<GaugeGlyph size={17} />}
              value={h ? titleCase(h.status) : '—'}
              label="Backend status"
              sublabel={h ? `v${h.version}` : undefined}
            />
            <IconStat
              tone={openAlerts.length > 0 ? 'warn' : 'healthy'}
              glyph={<AlertGlyph size={17} />}
              value={int(openAlerts.length)}
              label="Open alerts"
              action={{ label: 'Review alerts', to: '/alerts' }}
            />
            <IconStat
              tone="idle"
              glyph={<DropGlyph size={17} />}
              value={int(zoneList.length)}
              label="Zones reporting"
              sublabel={h?.lastWeatherSync ? `ET0 sync ${relativeTime(h.lastWeatherSync)}` : undefined}
            />
          </div>
        </Card>

        <Card title="Soil moisture trend" className="lg:col-span-2">
          {moistureSeries.length < 2 ? (
            <p className="py-10 text-center text-[12px] text-ink-400">
              Waiting for live telemetry frames.
            </p>
          ) : (
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart data={moistureSeries} margin={{ top: 6, right: 6, left: -18, bottom: 0 }}>
                <defs>
                  <linearGradient id="moistureFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.28} />
                    <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="label" {...chartAxis} tickLine={false} minTickGap={28} />
                <YAxis {...chartAxis} tickLine={false} width={44} unit="%" />
                <Tooltip
                  contentStyle={{ fontSize: 12, borderRadius: 4, border: '1px solid #e2e8f0' }}
                  formatter={(value: number) => [`${value.toFixed(1)}%`, 'Moisture']}
                />
                <Area
                  type="monotone"
                  dataKey="moisture"
                  stroke="#2563eb"
                  strokeWidth={1.8}
                  fill="url(#moistureFill)"
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Card title="Moisture distribution" className="lg:col-span-1">
          {zoneList.length === 0 ? (
            <p className="py-8 text-center text-[12px] text-ink-400">No zones configured.</p>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={bandChart} margin={{ top: 6, right: 6, left: -24, bottom: 0 }}>
                <CartesianGrid stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="name" {...chartAxis} tickLine={false} />
                <YAxis {...chartAxis} tickLine={false} allowDecimals={false} width={36} />
                <Tooltip
                  cursor={{ fill: '#f8fafc' }}
                  contentStyle={{ fontSize: 12, borderRadius: 4, border: '1px solid #e2e8f0' }}
                />
                <Bar dataKey="value" name="Zones" radius={[3, 3, 0, 0]} isAnimationActive={false} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card
          title="Zones"
          className="lg:col-span-2"
          action={
            <Link to="/zones" className="text-[12px] font-medium text-brand-700 hover:underline">
              View all
            </Link>
          }
        >
          {zoneList.length === 0 ? (
            <p className="py-8 text-center text-[12px] text-ink-400">
              No zones yet. Run the node simulator or post telemetry to populate this table.
            </p>
          ) : (
            <div className="-mx-5 -my-1 overflow-x-auto">
              <table className="w-full min-w-[640px] text-left">
                <thead>
                  <tr className="border-b border-ink-200 text-[10px] uppercase tracking-[0.08em] text-ink-400">
                    <th className="px-5 py-2 font-medium">Zone</th>
                    <th className="px-3 py-2 font-medium">Link</th>
                    <th className="px-3 py-2 font-medium">Moisture</th>
                    <th className="px-3 py-2 font-medium">Band</th>
                    <th className="px-3 py-2 font-medium">pH</th>
                    <th className="px-3 py-2 font-medium">EC</th>
                    <th className="px-3 py-2 font-medium">N-P-K</th>
                    <th className="px-3 py-2 font-medium">Stress</th>
                    <th className="px-5 py-2 font-medium">Last seen</th>
                  </tr>
                </thead>
                <tbody>
                  {zoneList.map((zone) => (
                    <tr key={zone.zone} className="border-b border-ink-100 text-[12px] last:border-0">
                      <td className="px-5 py-2.5">
                        <div className="font-medium text-ink-900">{zone.zone}</div>
                        <div className="text-[11px] text-ink-400">{zone.crop}</div>
                      </td>
                      <td className="px-3 py-2.5">
                        <LinkBadge state={zone.linkState} />
                      </td>
                      <td className="tnum px-3 py-2.5 text-ink-900">{num(zone.soilMoisture)}%</td>
                      <td className="px-3 py-2.5">
                        <MoistureChip band={zone.moistureBand} />
                      </td>
                      <td className="tnum px-3 py-2.5 text-ink-700">{num(zone.soilPh)}</td>
                      <td className="tnum px-3 py-2.5 text-ink-700">{num(zone.ec, 2)}</td>
                      <td className="tnum px-3 py-2.5 text-ink-700">
                        {num(zone.nitrogen, 0)}/{num(zone.phosphorus, 0)}/{num(zone.potassium, 0)}
                      </td>
                      <td className="tnum px-3 py-2.5 text-ink-700">
                        {zone.cropStressIndex === null ? '—' : num(zone.cropStressIndex, 2)}
                      </td>
                      <td className="px-5 py-2.5 text-ink-500">{relativeTime(zone.lastSeen)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      <Card
        title="Latest recommendations"
        action={
          <Link to="/zones" className="text-[12px] font-medium text-brand-700 hover:underline">
            Zone detail
          </Link>
        }
      >
        {openAlerts.length === 0 && zoneList.length === 0 ? (
          <p className="py-6 text-center text-[12px] text-ink-400">No decisions recorded yet.</p>
        ) : (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
            {zoneList.slice(0, 6).map((zone) => (
              <div key={zone.zone} className="rounded border border-ink-200 p-3">
                <div className="flex items-center justify-between">
                  <span className="text-[12px] font-medium text-ink-900">{zone.zone}</span>
                  <RiskChip level={zone.valveOpen ? 'watch' : 'healthy'} />
                </div>
                <p className="mt-1.5 text-[11px] leading-relaxed text-ink-500">
                  {zone.valveOpen
                    ? `Valve open, ${duration(zone.valveRuntimeSeconds)} elapsed.`
                    : 'Valve closed, no irrigation scheduled.'}
                </p>
                <p className="mt-1 text-[11px] text-ink-400">
                  {zone.nextIrrigationAt
                    ? `Next planned: ${clockTime(zone.nextIrrigationAt)}`
                    : 'No next cycle scheduled'}
                </p>
              </div>
            ))}
          </div>
        )}
      </Card>
    </>
  )
}
