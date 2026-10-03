/**
 * Analytics: water balance and efficiency over the selected window.
 *
 * The figures come from the backend's analytics endpoint rather than being
 * recomputed in the browser, so the dashboard and the stored records cannot
 * disagree. "Water saved" is an estimate against a fixed-ET0 baseline and is
 * labelled as such in the caption.
 */

import { useCallback, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api/client'
import { useLive } from '../App'
import { usePollingFetch } from '../hooks/usePollingFetch'
import { Card, Metric, StatRow } from '../components/ui'
import { int, num } from '../lib/format'

const WINDOWS = [
  { label: '6 hours', hours: 6 },
  { label: '24 hours', hours: 24 },
  { label: '7 days', hours: 168 },
  { label: '30 days', hours: 720 },
]

const axis = { stroke: '#cbd5e1', fontSize: 11 }
const tooltipStyle = { fontSize: 12, borderRadius: 4, border: '1px solid #e2e8f0' }

export default function Analytics() {
  const { bump } = useLive()
  const [hours, setHours] = useState(24)

  const overview = usePollingFetch(
    useCallback(() => api.analytics(hours), [hours]),
    bump,
    20_000,
  )
  const data = overview.data
  const series = data?.series ?? []

  return (
    <>
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-[20px] font-semibold tracking-tight text-ink-900">Analytics</h1>
          <p className="text-[12px] text-ink-500">
            Water applied against reference evapotranspiration
          </p>
        </div>
        <div className="flex items-center gap-1 rounded border border-ink-200 bg-white p-0.5">
          {WINDOWS.map((option) => (
            <button
              key={option.hours}
              type="button"
              onClick={() => setHours(option.hours)}
              className={`rounded-[3px] px-2.5 py-1 text-[12px] font-medium transition-colors ${
                hours === option.hours
                  ? 'bg-brand-600 text-white'
                  : 'text-ink-500 hover:bg-ink-50'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      {overview.error && (
        <Card title="Could not load analytics">
          <p className="text-[13px] text-ink-500">{overview.error}</p>
        </Card>
      )}

      <Card bodyClassName="">
        <StatRow
          items={[
            { value: num(data?.waterAppliedMm, 1), unit: 'mm', label: 'Mean depth per event' },
            { value: int(data?.irrigationEvents), label: 'Irrigation events' },
            { value: int(data?.estimatedLitresSaved), unit: 'L', label: 'Estimated saved' },
            {
              value: int(data?.runtimeSeconds === undefined ? undefined : data.runtimeSeconds / 60),
              unit: 'min',
              label: 'Valve runtime',
            },
          ]}
        />
        <div className="grid grid-cols-2 divide-x divide-ink-200 border-t border-ink-200 sm:grid-cols-4">
          <div className="p-5">
            <Metric
              value={data ? `${data.waterAppliedPctChange > 0 ? '+' : ''}${num(data.waterAppliedPctChange, 1)}` : '—'}
              unit="%"
              caption="Change vs previous period"
            />
          </div>
          <div className="p-5">
            <Metric
              value={data ? num(data.averageInferenceMs, 2) : '—'}
              unit="ms"
              caption="Mean decision time over the window"
            />
          </div>
          <div className="p-5">
            <Metric
              value={data ? num(data.decisionsCorrectPct, 1) : '—'}
              unit="%"
              caption="Decisions matching agronomic rule"
            />
          </div>
          <div className="p-5">
            <Metric
              value={data ? num(data.energyKwh, 3) : '—'}
              unit="kWh"
              caption="Estimated pump energy"
            />
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card title="Applied water vs ET0">
          {series.length === 0 ? (
            <p className="py-12 text-center text-[12px] text-ink-400">No irrigation in this window.</p>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={series} margin={{ top: 6, right: 6, left: -18, bottom: 0 }}>
                <CartesianGrid stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="label" {...axis} tickLine={false} minTickGap={24} />
                <YAxis {...axis} tickLine={false} width={44} unit="mm" />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: '#f8fafc' }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="appliedMm" name="Applied (mm)" fill="#2563eb" radius={[3, 3, 0, 0]} isAnimationActive={false} />
                <Bar dataKey="et0MmDay" name="ET0 (mm/day)" fill="#cbd5e1" radius={[3, 3, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </Card>

        <Card title="Root-zone depletion">
          {series.length === 0 ? (
            <p className="py-12 text-center text-[12px] text-ink-400">No data in this window.</p>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={series} margin={{ top: 6, right: 6, left: -18, bottom: 0 }}>
                <CartesianGrid stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="label" {...axis} tickLine={false} minTickGap={24} />
                <YAxis {...axis} tickLine={false} width={44} domain={[0, 1]} />
                <Tooltip contentStyle={tooltipStyle} formatter={(value: number) => value.toFixed(3)} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line
                  type="monotone"
                  dataKey="depletionFraction"
                  name="Depletion fraction"
                  stroke="#f59e0b"
                  strokeWidth={1.8}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </Card>
      </div>

      <p className="text-[11px] leading-relaxed text-ink-400">
        Window: {data?.window ?? hours + 'h'}. Saved water is estimated against a fixed-ET0 baseline
        with no rain or leaching, so treat it as a relative figure rather than a metered volume.
        ET0 comes from NASA POWER daily records for the nearest reference station.
      </p>
    </>
  )
}
