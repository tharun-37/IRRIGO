/**
 * Model health: what the deployed estimators actually score.
 *
 * This page deliberately shows measured numbers with their evaluation caveats
 * attached, including the disease score that the training run flagged as
 * degenerate. A dashboard that hides a weak metric is worse than no dashboard.
 */

import { useCallback } from 'react'
import { Link } from 'react-router-dom'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api/client'
import { useLive } from '../App'
import { usePollingFetch } from '../hooks/usePollingFetch'
import { Card, Metric, StatRow } from '../components/ui'
import { num, titleCase } from '../lib/format'

const axis = { stroke: '#cbd5e1', fontSize: 11 }
const tooltipStyle = { fontSize: 12, borderRadius: 4, border: '1px solid #e2e8f0' }

export default function ModelHealth() {
  const { bump } = useLive()
  const health = usePollingFetch(useCallback(() => api.health(), []), bump, 20_000)
  const drift = usePollingFetch(useCallback(() => api.drift(168), []), bump, 60_000)
  const metrics = health.data?.modelMetrics ?? null

  const diseaseBars = metrics
    ? Object.entries(metrics.disease.perClass).map(([label, value]) => ({
        name: titleCase(label),
        f1: value.f1,
        support: value.support,
      }))
    : []

  const latencyBars = metrics
    ? Object.entries(metrics.latencyMs).map(([key, value]) => ({ name: key, ms: value }))
    : []

  const driftRows = drift.data?.available
    ? Object.entries(drift.data.features)
        .filter(([, feature]) => feature.psi !== null)
        .map(([name, feature]) => ({ name: titleCase(name), ...feature }))
        .sort((a, b) => (b.psi ?? 0) - (a.psi ?? 0))
    : []

  if (!metrics) {
    return (
      <Card title="Model metrics">
        <p className="text-[13px] text-ink-500">
          {health.error ?? 'Metrics are unavailable. The backend has not loaded a model bundle yet.'}
        </p>
      </Card>
    )
  }

  return (
    <>
      <div className="flex items-end justify-between">
        <div>
          <h1 className="text-[20px] font-semibold tracking-tight text-ink-900">Model</h1>
          <p className="text-[12px] text-ink-500">
            Holdout performance, calibration and drift for the estimators in the deployed bundle
          </p>
        </div>
        <span className="text-[11px] text-ink-400">
          Bundle v{health.data?.version ?? '—'} &middot; loaded{' '}
          {health.data?.modelsLoaded ? 'yes' : 'no'} &middot;{' '}
          {health.data?.irrigateThresholdGroups ?? 0} tuned operating points
        </span>
      </div>

      <Card bodyClassName="">
        <StatRow
          items={[
            { value: num(metrics.irrigateDecision.f1, 3), label: 'Irrigation decision F1' },
            { value: num(metrics.irrigateDecision.precision, 3), label: 'Decision precision' },
            { value: num(metrics.waterDepth.mae, 2), unit: 'mm', label: 'Depth MAE' },
            { value: num(metrics.yield.r2, 3), label: 'Yield R2' },
          ]}
        />
        <div className="grid grid-cols-2 divide-x divide-ink-200 border-t border-ink-200 sm:grid-cols-4">
          <div className="p-5">
            <Metric
              value={num(metrics.irrigateDecision.threshold, 2)}
              caption="Decision threshold (tuned)"
            />
          </div>
          <div className="p-5">
            <Metric
              value={num(metrics.irrigateDecision.recall, 3)}
              caption="Decision recall"
            />
          </div>
          <div className="p-5">
            <Metric
              value={num(metrics.waterDepth.r2, 3)}
              caption="Depth R2"
            />
          </div>
          <div className="p-5">
            <Metric
              value={num(health.data?.calibration?.expected_calibration_error, 4)}
              caption="Calibration error"
            />
          </div>
          <div className="p-5">
            <Metric
              value={health.data?.depthUncertaintyMm ? `±${num(health.data.depthUncertaintyMm, 2)}` : '—'}
              unit="mm"
              caption="Depth 90% interval"
            />
          </div>
          <div className="p-5">
            <Metric
              value={num(health.data?.inferenceLatencyMs, 2)}
              unit="ms"
              caption="End-to-end latency"
            />
          </div>
        </div>
      </Card>

      <Card title="Drift against the training distribution">
        {!drift.data?.available ? (
          <p className="py-6 text-center text-[12px] text-ink-400">
            {drift.data?.reason ?? drift.error ?? 'Drift report unavailable.'}
          </p>
        ) : (
          <>
            {drift.data.frozen_sensors.length > 0 && (
              <p className="mb-3 rounded border border-status-critical/30 bg-status-critical/5 px-3 py-2 text-[11px] leading-relaxed text-ink-700">
                Probes reporting a single unchanged value:{' '}
                <strong>{drift.data.frozen_sensors.join(', ')}</strong>. That is a hardware
                fault, not drift. The model is reading a stuck sensor.
              </p>
            )}
            <ul className="divide-y divide-ink-100">
              {driftRows.slice(0, 8).map((row) => (
                <li key={row.name} className="flex items-center justify-between py-2 text-[12px]">
                  <span className="text-ink-700">{row.name}</span>
                  <span className="flex items-center gap-3">
                    <span className="text-ink-500">
                      live {num(row.live_mean, 1)} / train {num(row.reference_mean, 1)}
                    </span>
                    <span
                      className={`w-16 text-right font-medium ${
                        row.band === 'significant'
                          ? 'text-status-critical'
                          : row.band === 'investigate'
                            ? 'text-status-warning'
                            : 'text-status-healthy'
                      }`}
                    >
                      {num(row.psi, 3)}
                    </span>
                  </span>
                </li>
              ))}
              {driftRows.length === 0 && (
                <li className="py-6 text-center text-[12px] text-ink-400">
                  Not enough readings in the window to judge.
                </li>
              )}
            </ul>
            <p className="mt-3 border-t border-ink-100 pt-3 text-[11px] leading-relaxed text-ink-500">
              {drift.data.interpretation} Window: {drift.data.readings_considered} readings
              over {drift.data.window_hours}h.
            </p>
          </>
        )}
      </Card>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card title="Disease risk, per class F1">
          {diseaseBars.length === 0 ? (
            <p className="py-10 text-center text-[12px] text-ink-400">No per-class data.</p>
          ) : (
            <ResponsiveContainer width="100%" height={230}>
              <BarChart data={diseaseBars} margin={{ top: 6, right: 6, left: -20, bottom: 0 }}>
                <CartesianGrid stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="name" {...axis} tickLine={false} />
                <YAxis {...axis} tickLine={false} width={40} domain={[0, 1]} />
                <Tooltip
                  contentStyle={tooltipStyle}
                  formatter={(value: number, _name, item) => [
                    `${value.toFixed(3)} (n=${(item.payload as { support: number }).support})`,
                    'F1',
                  ]}
                />
                <Bar dataKey="f1" radius={[3, 3, 0, 0]} isAnimationActive={false}>
                  {diseaseBars.map((entry) => (
                    <Cell
                      key={entry.name}
                      fill={entry.f1 < 0.6 ? '#f59e0b' : entry.f1 >= 0.9 ? '#16a34a' : '#3b82f6'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
          <p className="mt-3 border-t border-ink-100 pt-3 text-[11px] leading-relaxed text-ink-500">
            <strong>These numbers are not diagnostic performance.</strong> The training labels
            are synthesised by the rule engine in <code>disease_thresholds.json</code> and
            reproduce from six climate and soil features at 100%, so this classifier is a smooth
            restatement of that lookup table. The score above measures how well it memorised the
            rules, and a perfect unseen-station score would follow from the labels being
            deterministic rather than from anything the model learned. The held-out window also
            contains very few positive cases, one class with a support of 2. Use it as a
            stress-risk indicator for the dashboard, never as a crop diagnosis.
          </p>
        </Card>

        <Card title="Per-estimator training benchmark">
          {latencyBars.length === 0 ? (
            <p className="py-10 text-center text-[12px] text-ink-400">No latency data.</p>
          ) : (
            <>
              <p className="mb-2 text-[11px] leading-relaxed text-ink-500">
                One estimator in isolation on a warm batch, as measured during training. A
                live decision runs all of these plus feature assembly, and costs far more
                than their sum — see the end-to-end figure above.
              </p>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart
                  data={latencyBars}
                  layout="vertical"
                  margin={{ top: 6, right: 24, left: 18, bottom: 0 }}
                >
                  <CartesianGrid stroke="#f1f5f9" horizontal={false} />
                  <XAxis type="number" {...axis} tickLine={false} unit=" ms" />
                  <YAxis type="category" dataKey="name" {...axis} tickLine={false} width={92} />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    formatter={(value: number) => [`${value.toFixed(2)} ms`, 'Latency']}
                  />
                  <Bar dataKey="ms" fill="#94a3b8" radius={[0, 3, 3, 0]} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </>
          )}
        </Card>
      </div>

      <Card title="Nutrient demand regressors">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {(['nitrogen', 'phosphorus', 'potassium'] as const).map((nutrient) => (
            <div key={nutrient} className="rounded border border-ink-200 p-3">
              <div className="text-[10px] font-medium uppercase tracking-[0.08em] text-ink-400">
                {titleCase(nutrient)}
              </div>
              <div className="tnum mt-1 text-[22px] font-semibold tracking-tight text-ink-900">
                {num(metrics.nutrient[nutrient], 2)}
                <span className="ml-1 text-[12px] font-medium text-ink-400">kg/ha MAE</span>
              </div>
            </div>
          ))}
        </div>
        <p className="mt-4 text-[11px] leading-relaxed text-ink-400">
          Trained on a physically simulated corpus of {health.data?.deviceCount ?? 6} zone
          configurations over 12 NASA POWER reference stations, 2015 to 2024. See{' '}
          <Link to="/model" className="text-brand-700 hover:underline">
            docs/DATASETS.md
          </Link>{' '}
          in the repository for provenance and limitations.
        </p>
      </Card>
    </>
  )
}
