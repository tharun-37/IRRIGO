/**
 * Crop Health — photo check, in the words a farmer acts on.
 *
 * The flow is three beats and nothing else: pick the field, take a photo of the
 * plant, read the answer. Choosing the field chooses the crop — the crop grows
 * in the field, so the farmer is never asked to name a variety or a plant part.
 * The photo is captured with the camera intent, analysed the moment it is
 * taken, and the result replaces the form instead of arriving as a panel below
 * it.
 *
 * The flow itself lives in `components/CropHealthCheck`, shared with the
 * farmer view's compact widget; this page adds the header, the honest status
 * line and the recorded history. The fertilizer form that used to sit here is
 * gone from the farmer's view: it is a soil-test calculation with its own
 * vocabulary, and it does not belong in a capture flow. The backend keeps the
 * endpoint.
 */

import { useCallback, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import { useLive } from '../App'
import { usePollingFetch } from '../hooks/usePollingFetch'
import CropHealthCheck from '../components/CropHealthCheck'
import { VERDICT } from '../components/CropHealthCheck'
import { clockTime } from '../lib/format'

export default function CropHealth() {
  const { refresh } = useLive()
  const analyses = usePollingFetch(useCallback(() => api.analyses(20), []), 0, 10_000)
  const [removing, setRemoving] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const history = analyses.data?.analyses ?? []

  const remove = useCallback(
    async (id: string) => {
      setRemoving(id)
      try {
        await api.deleteAnalysis(id)
        refresh()
        return true
      } catch (cause) {
        setError(cause instanceof ApiError ? cause.message : 'Could not delete this record.')
        return false
      } finally {
        setRemoving(null)
      }
    },
    [refresh],
  )

  return (
    <>
      <div className="flex items-end justify-between gap-3">
        <div>
          <h1 className="text-[22px] font-semibold tracking-tight text-ink-900">Crop health</h1>
          <p className="text-[13px] text-ink-500">
            Take a photo of the plant. We check it and tell you what to do.
          </p>
        </div>
        <Link to="/" className="shrink-0 text-[13px] font-semibold text-brand-700 hover:underline">
          Back to the field
        </Link>
      </div>

      <CropHealthCheck />

      {error && (
        <p className="rounded border border-status-alert/40 bg-status-alertSoft px-3 py-2 text-[13px] font-semibold text-status-alert">
          {error}
        </p>
      )}

      {/* ---- Past checks, in the same plain words ------------------------ */}
      <section className="rounded-card border border-ink-200 bg-white/70 px-4 py-3.5">
        <h2 className="text-[14px] font-semibold text-ink-900">Past checks</h2>
        {history.length === 0 ? (
          <p className="mt-1.5 text-[12px] text-slate-600">
            No checks yet. Every photo you check — even a poor one — is recorded here.
          </p>
        ) : (
          <ul className="mt-2 space-y-2">
            {history.map((entry) => {
              const verdict = VERDICT[entry.resultStatus] ?? VERDICT.processing_error
              return (
                <li
                  key={entry.analysisId}
                  className="flex items-start gap-2.5 rounded border border-ink-200 px-2.5 py-2"
                >
                  <span
                    className={`mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-[11px] font-bold ${verdict.badge}`}
                  >
                    {verdict.word}
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="text-[12px] font-semibold text-ink-800">
                      {entry.crop} · {clockTime(entry.createdAt)}
                      {entry.fieldId ? ` · ${entry.fieldId}` : ''}
                    </p>
                    <p className="mt-0.5 line-clamp-2 text-[12px] leading-snug text-slate-600">
                      {entry.statusReason}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => remove(entry.analysisId)}
                    disabled={removing === entry.analysisId}
                    className="shrink-0 rounded border border-ink-200 px-2 py-1 text-[12px] font-semibold text-ink-600 hover:border-status-alert hover:text-status-alert disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-800"
                    title="Delete this record and its stored photo"
                  >
                    {removing === entry.analysisId ? '…' : 'Delete'}
                  </button>
                </li>
              )
            })}
          </ul>
        )}
      </section>
    </>
  )
}
