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
 * The page is still honest about what the service is: a photo-quality gate with
 * an inference slot waiting for a validated model. When nothing can be named
 * the answer says so in plain words — "we can't name a disease yet" — rather
 * than inventing a verdict. Failed photos are part of the record too, and every
 * stored photo is one tap away from deletion.
 *
 * The fertilizer form that used to sit on this page is gone from the farmer's
 * view: it is a soil-test calculation with its own vocabulary, and it does not
 * belong in a capture flow. The backend keeps the endpoint.
 */

import { useCallback, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, ApiError } from '../api/client'
import { useLive } from '../App'
import { usePollingFetch } from '../hooks/usePollingFetch'
import { clockTime, titleCase } from '../lib/format'
import type { Advisory, CropHealthAnalysis, CropHealthResultStatus } from '../types'

const REFRESH = 15_000

/** The verdict, in words the farmer reads over the field. */
const VERDICT: Record<CropHealthResultStatus, { word: string; badge: string }> = {
  result: { word: 'Checked', badge: 'bg-status-healthySoft text-[#15803d]' },
  uncertain: { word: 'Not sure yet', badge: 'bg-status-warnSoft text-[#b45309]' },
  unsupported_crop: { word: 'Crop not covered', badge: 'bg-status-warnSoft text-[#b45309]' },
  poor_image: { word: 'Photo not clear', badge: 'bg-status-alertSoft text-status-alert' },
  invalid_input: { word: 'Photo not usable', badge: 'bg-status-alertSoft text-status-alert' },
  processing_error: { word: 'Something went wrong', badge: 'bg-status-alertSoft text-status-alert' },
}

/** Crop per demo field, used when the advisory feed has not answered yet. */
const FIELD_CROPS: Record<string, string> = {
  'Barley-West': 'Barley',
  'Cotton-South': 'Cotton',
  'Potato-East': 'Potato',
  'Paddy-North': 'Rice',
}

export default function CropHealth() {
  const { bump, refresh } = useLive()
  const status = usePollingFetch(useCallback(() => api.status(), []), bump, 30_000)
  const advisor = usePollingFetch(useCallback(() => api.advisor(), []), bump, REFRESH)
  const historyFeed = usePollingFetch(useCallback(() => api.analyses(20), []), bump, 10_000)

  const [fieldId, setFieldId] = useState<string | null>(null)
  const [consent, setConsent] = useState(true)
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState<CropHealthAnalysis | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [removing, setRemoving] = useState<string | null>(null)
  const cameraRef = useRef<HTMLInputElement | null>(null)

  // The field buttons come from the farm's own fields — the four demo fields
  // answered from constants until the feed lands, so the farmer is never shown
  // an empty step.
  const fieldList = useMemo(() => {
    const fromFeed = (advisor.data?.fields ?? []).map((item: Advisory) => ({
      id: item.fieldId,
      crop: item.crop,
    }))
    const known = new Set(fromFeed.map((item) => item.id))
    const fallback = Object.entries(FIELD_CROPS)
      .filter(([id]) => !known.has(id))
      .map(([id, crop]) => ({ id, crop }))
    return [...fromFeed, ...fallback]
  }, [advisor.data])

  const field = fieldList.find((item) => item.id === fieldId) ?? null
  const history = historyFeed.data?.analyses ?? []
  const deployed = status.data?.deployedModel ?? false

  const takePhoto = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0] ?? null
      // Reset the input so re-taking a photo of the same plant fires the
      // change event again.
      event.target.value = ''
      if (!file) return
      if (!fieldId) {
        setError('Tap your field first, then take the photo.')
        return
      }
      const crop = field?.crop ?? FIELD_CROPS[fieldId] ?? 'Rice'
      setAnalyzing(true)
      setError(null)
      setResult(null)
      try {
        const analysis = await api.analyzeCropHealth(file, {
          crop,
          plantPart: 'leaf',
          fieldId,
          consent,
        })
        setResult(analysis)
        refresh()
      } catch (cause) {
        setError(cause instanceof ApiError ? cause.message : 'The photo check failed. Try again.')
      } finally {
        setAnalyzing(false)
      }
    },
    [fieldId, field?.crop, consent, refresh],
  )

  const remove = useCallback(
    async (id: string) => {
      setRemoving(id)
      try {
        await api.deleteAnalysis(id)
        if (result?.analysisId === id) setResult(null)
        refresh()
      } catch (cause) {
        setError(cause instanceof ApiError ? cause.message : 'Could not delete this record.')
      } finally {
        setRemoving(null)
      }
    },
    [result, refresh],
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

      {/* One honest line about what the check can do today. It stays quiet: the
          farmer reads it once, not at every capture. */}
      <p className="rounded border border-ink-200 bg-white/70 px-3 py-2 text-[12px] leading-snug text-slate-700">
        {deployed
          ? 'This check is running with a validated photo model. Its answer is a model reading, not a diagnosis.'
          : 'This check is still being tested. It can tell if a photo is usable, but it cannot name a disease yet.'}
      </p>

      {error && (
        <p className="rounded border border-status-alert/40 bg-status-alertSoft px-3 py-2 text-[13px] font-semibold text-status-alert">
          {error}
        </p>
      )}

      {/* ---- Step 1: the field, so the crop is never asked --------------- */}
      <section className="rounded-card border border-ink-200 bg-white/70 px-4 py-3.5">
        <h2 className="text-[14px] font-semibold text-ink-900">1. Tap your field</h2>
        <div className="mt-2.5 flex flex-wrap gap-2">
          {fieldList.map((item) => {
            const active = item.id === fieldId
            return (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  setFieldId(item.id)
                  setError(null)
                }}
                aria-pressed={active}
                className={`rounded-card border px-4 py-2.5 text-[14px] font-semibold transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-600 ${
                  active
                    ? 'border-ink-900 bg-ink-900 text-white shadow-lift'
                    : 'border-ink-200 bg-white text-ink-700 hover:border-ink-400'
                }`}
              >
                {item.id}
                <span className={`ml-2 text-[12px] font-medium ${active ? 'text-white/70' : 'text-ink-400'}`}>
                  {item.crop}
                </span>
              </button>
            )
          })}
        </div>
      </section>

      {/* ---- Step 2: the photo, captured with the camera ------------------ */}
      <section className="rounded-card border border-ink-200 bg-white/70 px-4 py-3.5">
        <h2 className="text-[14px] font-semibold text-ink-900">
          2. {fieldId ? 'Take a photo of the plant' : 'Take a photo (choose your field above first)'}
        </h2>
        <input
          ref={cameraRef}
          type="file"
          accept="image/*"
          capture="environment"
          onChange={takePhoto}
          className="sr-only"
          aria-label="Take a photo of the plant"
        />
        <button
          type="button"
          onClick={() => cameraRef.current?.click()}
          disabled={analyzing}
          className="mt-2.5 h-12 w-full rounded-card bg-brand-600 px-4 text-[15px] font-semibold text-white shadow-sm transition-colors hover:bg-brand-700 disabled:opacity-60 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-800"
        >
          {analyzing ? 'Checking your photo…' : 'Open camera or choose photo'}
        </button>
        {analyzing && (
          <p className="mt-2 text-[13px] text-ink-600" role="status">
            Checking your photo. This takes a few seconds.
          </p>
        )}
        <label className="mt-2.5 flex items-start gap-2 text-[12px] leading-snug text-slate-700">
          <input
            type="checkbox"
            checked={consent}
            onChange={(event) => setConsent(event.target.checked)}
            className="mt-0.5 h-4 w-4"
          />
          <span>
            Keep this photo in history. Untick for one-time checks — the photo is checked and
            deleted straight away, nothing is kept.
          </span>
        </label>
      </section>

      {/* ---- The answer, directly ---------------------------------------- */}
      {result && <ResultCard analysis={result} onDelete={remove} removing={removing === result.analysisId} />}

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

/**
 * The answer card.
 *
 * No model versions, no gate list, no raw scores: those were the console's
 * vocabulary and the farmer never asked for them. The farmer gets the verdict,
 * the reason in plain words, and what to do next — the three things the advice
 * text was written for.
 */
function ResultCard({
  analysis,
  onDelete,
  removing,
}: {
  analysis: CropHealthAnalysis
  onDelete: (id: string) => void
  removing: boolean
}) {
  const verdict = VERDICT[analysis.resultStatus] ?? VERDICT.processing_error
  const className = analysis.prediction.classLabel
  // A "healthy" class is not a disease worth naming; reading "Possible: Healthy"
  // back to the farmer would be the model's vocabulary, not theirs.
  const headline = className
    ? /health/i.test(className)
      ? 'No disease spotted in this photo'
      : `Possible: ${titleCase(className)}`
    : `${analysis.crop} · no disease could be named`
  return (
    <section className="rounded-card border border-ink-200 bg-ink-50/70 px-4 py-4" aria-label="Photo check result">
      <div className="flex items-center justify-between gap-2">
        <span className={`inline-flex items-center gap-1.5 rounded px-2.5 py-1 text-[13px] font-bold ${verdict.badge}`}>
          {verdict.word}
        </span>
        <button
          type="button"
          onClick={() => onDelete(analysis.analysisId)}
          disabled={removing}
          className="text-[12px] font-semibold text-ink-500 hover:text-status-alert disabled:opacity-50"
        >
          {removing ? 'Deleting…' : 'Delete record'}
        </button>
      </div>

      <p className="mt-2.5 text-[18px] font-semibold leading-snug text-ink-900">{headline}</p>
      <p className="mt-1 text-[13px] leading-relaxed text-slate-700">{analysis.statusReason}</p>

      <p className="mt-3 text-[14px] font-bold text-ink-800">What to do next</p>
      <p className="text-[14px] leading-relaxed text-ink-800">{analysis.advice.adviceText}</p>
      {analysis.advice.followUpQuestion && (
        <p className="mt-1.5 text-[12px] text-ink-600">Check: {analysis.advice.followUpQuestion}</p>
      )}
      {analysis.advice.expertReferral && (
        <p className="mt-1.5 text-[13px] font-semibold text-status-warn">
          Contact your local agricultural officer for a confirmed identification.
        </p>
      )}
      <p className="mt-2.5 text-[11px] text-ink-400">Checked at {clockTime(analysis.createdAt)}</p>
    </section>
  )
}
