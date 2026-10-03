/**
 * Alerts and the system event log.
 *
 * Alerts are actionable, so each row offers the one operation that resolves it.
 * Acknowledging is deliberately separate from resolving: acknowledging says an
 * operator has seen it, resolving says the condition has actually cleared.
 */

import { useCallback, useState } from 'react'
import { api, ApiError } from '../api/client'
import { useLive } from '../App'
import { usePollingFetch } from '../hooks/usePollingFetch'
import { Card, SeverityChip, StatRow } from '../components/ui'
import { clockTime, int, relativeTime, titleCase } from '../lib/format'

export default function Alerts() {
  const { bump, refresh } = useLive()
  const [openOnly, setOpenOnly] = useState(false)
  const [busy, setBusy] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)

  const alerts = usePollingFetch(
    useCallback(() => api.alerts(openOnly, 200), [openOnly]),
    bump,
    15_000,
  )
  const events = usePollingFetch(useCallback(() => api.events(40), []), 0, 20_000)

  const list = alerts.data?.alerts ?? []
  const openCount = list.filter((alert) => !alert.acknowledged).length

  const acknowledge = useCallback(
    async (id: number) => {
      setBusy(id)
      setError(null)
      try {
        await api.acknowledgeAlert(id)
        refresh()
      } catch (cause) {
        setError(cause instanceof ApiError ? cause.message : 'Could not acknowledge the alert.')
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
          <h1 className="text-[20px] font-semibold tracking-tight text-ink-900">Alerts</h1>
          <p className="text-[12px] text-ink-500">Conditions raised by the rule engine and the model</p>
        </div>
        <div className="flex items-center gap-1 rounded border border-ink-200 bg-white p-0.5">
          <button
            type="button"
            onClick={() => setOpenOnly(false)}
            className={`rounded-[3px] px-2.5 py-1 text-[12px] font-medium ${
              openOnly ? 'text-ink-500 hover:bg-ink-50' : 'bg-brand-600 text-white'
            }`}
          >
            All
          </button>
          <button
            type="button"
            onClick={() => setOpenOnly(true)}
            className={`rounded-[3px] px-2.5 py-1 text-[12px] font-medium ${
              openOnly ? 'bg-brand-600 text-white' : 'text-ink-500 hover:bg-ink-50'
            }`}
          >
            Open only
          </button>
        </div>
      </div>

      <Card bodyClassName="">
        <StatRow
          items={[
            { value: int(list.length), label: 'Shown' },
            { value: int(openCount), label: 'Unacknowledged' },
            {
              value: int(list.filter((alert) => alert.severity === 'critical').length),
              label: 'Critical',
            },
            { value: int(list.filter((alert) => alert.acknowledged).length), label: 'Acknowledged' },
          ]}
        />
      </Card>

      {error && (
        <div className="rounded border border-status-alert/30 bg-status-alertSoft px-4 py-2.5 text-[12px] text-status-alert">
          {error}
        </div>
      )}

      <Card title="Alerts">
        {list.length === 0 ? (
          <p className="py-8 text-center text-[12px] text-ink-400">
            {openOnly ? 'No open alerts.' : 'No alerts recorded.'}
          </p>
        ) : (
          <ul className="divide-y divide-ink-100">
            {list.map((alert) => (
              <li key={alert.id} className="flex flex-wrap items-center gap-3 py-3">
                <SeverityChip severity={alert.severity} />
                <div className="min-w-0 flex-1">
                  <p className="text-[13px] text-ink-900">{alert.message}</p>
                  <p className="mt-0.5 text-[11px] text-ink-400">
                    {titleCase(alert.kind)} &middot; {alert.zone} &middot; {alert.deviceId} &middot;{' '}
                    {clockTime(alert.raisedAt)} ({relativeTime(alert.raisedAt)})
                  </p>
                </div>
                {alert.acknowledged ? (
                  <span className="text-[11px] text-ink-400">
                    Acknowledged{alert.resolvedAt ? `, resolved ${relativeTime(alert.resolvedAt)}` : ''}
                  </span>
                ) : (
                  <button
                    type="button"
                    className="btn-secondary"
                    disabled={busy === alert.id}
                    onClick={() => void acknowledge(alert.id)}
                  >
                    Acknowledge
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card title="Event log">
        {(events.data?.events ?? []).length === 0 ? (
          <p className="py-6 text-center text-[12px] text-ink-400">No events recorded.</p>
        ) : (
          <ul className="divide-y divide-ink-100">
            {(events.data?.events ?? []).map((event) => (
              <li key={event.id} className="flex items-center gap-3 py-2 text-[12px]">
                <span className="w-20 shrink-0 text-ink-400">{clockTime(event.createdAt)}</span>
                <span className="w-16 shrink-0 font-medium text-ink-700">{event.zone}</span>
                <span className="w-20 shrink-0 text-ink-500">{titleCase(event.kind)}</span>
                <span className="min-w-0 flex-1 truncate text-ink-700">{event.detail}</span>
                <span className="shrink-0 text-[11px] text-ink-400">{event.source}</span>
              </li>
            ))}
          </ul>
        )}
      </Card>
    </>
  )
}
