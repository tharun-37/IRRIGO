/**
 * Crop Health capture flow — pick the field, take a photo, read the answer.
 *
 * One flow, two chrome: the `/crop-health` page renders it full-size with the
 * history below it, and the farmer view embeds it as a compact widget card so
 * the check sits beside Crop age and the water decision without leaving the
 * field. Choosing the field chooses the crop — the crop grows in the field, so
 * the farmer is never asked to name a variety or a plant part.
 *
 * The widget follows the field the farmer is already reading (`activeFieldId`)
 * and pushes a choice back up through `onFieldSelect`, so picking Barley-West
 * on the widget switches the whole view, exactly like the map.
 *
 * The page is honest about what the service is: a photo-quality gate with an
 * inference slot waiting for a validated model. When nothing can be named the
 * answer says so in plain words — "status pending" — rather than inventing a
 * verdict. Failed photos are part of the record too, and every stored photo is
 * one tap away from deletion.
 */

import { useCallback, useMemo, useRef, useState } from 'react'
import { api, ApiError } from '../api/client'
import { useLive } from '../App'
import { usePollingFetch } from '../hooks/usePollingFetch'
import { clockTime } from '../lib/format'
import type { Advisory, CropHealthAnalysis, CropHealthResultStatus } from '../types'

const REFRESH = 15_000

/** The verdict, in words the farmer reads over the field. */
export const VERDICT: Record<CropHealthResultStatus, { word: string; badge: string }> = {
  result: { word: 'Checked', badge: 'bg-status-healthySoft text-[#15803d]' },
  uncertain: { word: 'Not sure yet', badge: 'bg-status-warnSoft text-[#b45309]' },
  unsupported_crop: { word: 'Crop not covered', badge: 'bg-status-warnSoft text-[#b45309]' },
  poor_image: { word: 'Photo not clear', badge: 'bg-status-alertSoft text-status-alert' },
  invalid_input: { word: 'Photo not usable', badge: 'bg-status-alertSoft text-status-alert' },
  processing_error: { word: 'Something went wrong', badge: 'bg-status-alertSoft text-status-alert' },
}

/** Crop per demo field, used when the advisory feed has not answered yet. */
export const FIELD_CROPS: Record<string, string> = {
  'Barley-West': 'Barley',
  'Cotton-South': 'Cotton',
  'Potato-East': 'Potato',
  'Paddy-North': 'Rice',
}

export interface CropHealthCheckProps {
  /**
   * The fields the farmer picks between, as `{ id, crop }`. Omitted on the
   * page, which reduces its own advisory feed — and always leaves the four
   * demo fields offered.
   */
  fields?: { id: string; crop: string }[]
  /** The field the farmer is already reading (the widget follows this). */
  activeFieldId?: string | null
  /** Push a widget choice back to the farmer view (it switches the whole view). */
  onFieldSelect?: (fieldId: string) => void
  /** Widget layout: tight paddings, smaller titles, no step headings. */
  compact?: boolean
}

export default function CropHealthCheck({
  fields: providedFields,
  activeFieldId = null,
  onFieldSelect,
  compact = false,
}: CropHealthCheckProps) {
  const { bump, refresh } = useLive()
  const status = usePollingFetch(useCallback(() => api.status(), []), bump, 30_000)
  // The page reduces its own advisor feed; the widget is handed the farmer
  // view's list, so it must not poll the feed a second time.
  const advisor = usePollingFetch(
    useCallback(
      () => (providedFields ? Promise.resolve(null) : api.advisor()),
      [providedFields],
    ),
    bump,
    REFRESH,
  )

  const [picked, setPicked] = useState<string | null>(null)
  const [consent, setConsent] = useState(true)
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState<CropHealthAnalysis | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [removing, setRemoving] = useState<string | null>(null)
  const cameraRef = useRef<HTMLInputElement | null>(null)

  const fieldList = useMemo(() => {
    if (providedFields) return providedFields
    const fromFeed = (advisor.data?.fields ?? []).map((item: Advisory) => ({
      id: item.fieldId,
      crop: item.crop,
    }))
    const known = new Set(fromFeed.map((item) => item.id))
    const fallback = Object.entries(FIELD_CROPS)
      .filter(([id]) => !known.has(id))
      .map(([id, crop]) => ({ id, crop }))
    return [...fromFeed, ...fallback]
  }, [providedFields, advisor.data])

  // The widget follows the field the farmer is reading; on the page the farmer
  // taps the field explicitly, so a photo is never analysed against a field
  // nobody chose. A fresh pick takes over from there.
  const fieldId = picked ?? activeFieldId ?? null
  const field = fieldList.find((item) => item.id === fieldId) ?? null
  const deployed = status.data?.deployedModel ?? false

  const pickField = useCallback(
    (id: string) => {
      setPicked(id)
      setError(null)
      onFieldSelect?.(id)
    },
    [onFieldSelect],
  )

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

  const removeResult = useCallback(
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

  const cameraButtonLabel = compact ? 'Open camera' : 'Open camera or choose photo'

  return (
    <>
      {/* ---- Step 1: the field, so the crop is never asked --------------- */}
      {/* The page asks explicitly, because nothing else tells it which plant is
          being photographed. The widget has no picker of its own: the switcher
          at the top of the field view already chooses the field, and the map
          moves it — so the widget silently follows the field being read and
          just names it under the title. */}
      {compact ? (
        <p className="px-1 pt-1 text-[10px] font-bold uppercase tracking-[0.1em] text-ink-500">
          Crop photo check{field ? ` · ${field.id}` : ''}
        </p>
      ) : (
        <section className="rounded-card border border-ink-200 bg-white/70 px-4 py-3.5">
          <h2 className="text-[14px] font-semibold text-ink-900">1. Tap your field</h2>
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            {fieldList.map((item) => {
              const active = item.id === fieldId
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => pickField(item.id)}
                  aria-pressed={active}
                  className={`rounded-card border px-4 py-2.5 text-[14px] font-semibold transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-600 ${
                    active
                      ? 'border-ink-900 bg-ink-900 text-white shadow-lift'
                      : 'border-ink-200 bg-white text-ink-700 hover:border-ink-400'
                  }`}
                >
                  {item.id}
                  <span className={`ml-1.5 font-medium ${active ? 'text-white/70' : 'text-ink-400'}`}>
                    {item.crop}
                  </span>
                </button>
              )
            })}
          </div>
        </section>
      )}

      {/* ---- Step 2: the photo, captured with the camera ------------------ */}
      <section
        className={
          compact
            ? 'rounded border border-ink-200/70 bg-white/60 px-3 pb-2.5 pt-2'
            : 'rounded-card border border-ink-200 bg-white/70 px-4 py-3.5'
        }
      >
        {!compact && (
          <h2 className="text-[14px] font-semibold text-ink-900">
            2. {fieldId ? 'Take a photo of the plant' : 'Take a photo (choose your field above first)'}
          </h2>
        )}
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
          className={`w-full rounded-card bg-brand-600 font-semibold text-white shadow-sm transition-colors hover:bg-brand-700 disabled:opacity-60 focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand-800 ${
            compact ? 'mt-1.5 h-10 px-3 text-[14px]' : 'mt-2.5 h-12 px-4 text-[15px]'
          }`}
        >
          {analyzing ? (compact ? 'Checking…' : 'Checking your photo…') : cameraButtonLabel}
        </button>
        {analyzing && !compact && (
          <p className="mt-2 text-[13px] text-ink-600" role="status">
            Checking your photo. This takes a few seconds.
          </p>
        )}
        <label
          className={`flex items-start gap-2 leading-snug text-slate-700 ${
            compact ? 'mt-1.5 text-[11px]' : 'mt-2.5 text-[12px]'
          }`}
        >
          <input
            type="checkbox"
            checked={consent}
            onChange={(event) => setConsent(event.target.checked)}
            className={`mt-0.5 ${compact ? 'h-3.5 w-3.5' : 'h-4 w-4'}`}
          />
          <span>
            Keep this photo in history. Untick for one-time checks — the photo is checked and
            deleted straight away, nothing is kept.
          </span>
        </label>
      </section>

      {/* One honest line about what the check can do today. */}
      {compact ? (
        <p className="px-1 text-[11px] leading-snug text-ink-500">
          {deployed
            ? 'Model reading, not a diagnosis.'
            : 'Still being tested — it cannot name a disease yet.'}
        </p>
      ) : (
        <p className="rounded border border-ink-200 bg-white/70 px-3 py-2 text-[12px] leading-snug text-slate-700">
          {deployed
            ? 'This check is running with a validated photo model. Its answer is a model reading, not a diagnosis.'
            : 'This check is still being tested. It can tell if a photo is usable, but it cannot name a disease yet.'}
        </p>
      )}

      {error && (
        <p className="rounded border border-status-alert/40 bg-status-alertSoft px-3 py-2 text-[13px] font-semibold text-status-alert">
          {error}
        </p>
      )}

      {/* ---- The answer, directly ---------------------------------------- */}
      {result && (
        <ResultCard
          analysis={result}
          compact={compact}
          onDelete={removeResult}
          removing={removing === result.analysisId}
        />
      )}
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
export function ResultCard({
  analysis,
  compact = false,
  onDelete,
  removing = false,
}: {
  analysis: CropHealthAnalysis
  compact?: boolean
  onDelete?: (id: string) => void
  removing?: boolean
}) {
  const verdict = VERDICT[analysis.resultStatus] ?? VERDICT.processing_error
  const className = analysis.prediction.classLabel
  // A "healthy" class is not a disease worth naming; reading "Possible: Healthy"
  // back to the farmer would be the model's vocabulary, not theirs.
  // The shared model names across every crop's class list, so a barley photo
  // can carry a cotton-flavoured label; the leading crop word is presentation
  // noise and is stripped — the farmer already knows their crop. The full
  // label still travels in the record.
  const diseaseName = className?.replace(
    /^(rice|barley|wheat|cotton|potato|maize|tomato|sugarcane)\s+/i,
    '',
  )
  const headline = className
    ? /health/i.test(className)
      ? 'No disease spotted in this photo'
      : `Possible: ${titleCase(diseaseName ?? '')}`
    : analysis.resultStatus === 'result'
      ? 'Reads normal — no disease found'
      : `${analysis.crop} · no disease could be named`
  return (
    <section
      className={
        compact
          ? 'rounded border border-ink-200 bg-ink-50/70 px-3 py-2.5'
          : 'rounded-card border border-ink-200 bg-ink-50/70 px-4 py-4'
      }
      aria-label="Photo check result"
    >
      <div className="flex items-center justify-between gap-2">
        <span
          className={`inline-flex items-center rounded font-bold ${verdict.badge} ${
            compact ? 'px-2 py-0.5 text-[12px]' : 'px-2.5 py-1 text-[13px]'
          }`}
        >
          {verdict.word}
        </span>
        {onDelete && (
          <button
            type="button"
            onClick={() => onDelete(analysis.analysisId)}
            disabled={removing}
            className="text-[12px] font-semibold text-ink-500 hover:text-status-alert disabled:opacity-50"
          >
            {removing ? 'Deleting…' : 'Delete record'}
          </button>
        )}
      </div>

      <p
        className={`font-semibold leading-snug text-ink-900 ${compact ? 'mt-1.5 text-[14px]' : 'mt-2.5 text-[18px]'}`}
      >
        {headline}
      </p>
      <p className={`leading-relaxed text-slate-700 ${compact ? 'mt-1 text-[12px]' : 'mt-1 text-[13px]'}`}>
        {analysis.statusReason}
      </p>

      <p className={`font-bold text-ink-800 ${compact ? 'mt-1.5 text-[12px]' : 'mt-3 text-[14px]'}`}>
        What to do next
      </p>
      <p className={`leading-relaxed text-ink-800 ${compact ? 'text-[12px]' : 'text-[14px]'}`}>
        {analysis.advice.adviceText}
      </p>
      {analysis.advice.followUpQuestion && (
        <p className={`text-ink-600 ${compact ? 'mt-1 text-[11px]' : 'mt-1.5 text-[12px]'}`}>
          Check: {analysis.advice.followUpQuestion}
        </p>
      )}
      {analysis.advice.expertReferral && (
        <p className={`font-semibold text-status-warn ${compact ? 'mt-1 text-[12px]' : 'mt-1.5 text-[13px]'}`}>
          Contact your local agricultural officer for a confirmed identification.
        </p>
      )}

      {analysis.treatment && (
        <div
          className={`mt-2 rounded border border-status-warn/40 bg-status-warnSoft ${
            compact ? 'px-2.5 py-1.5' : 'px-3 py-2.5'
          }`}
        >
          <p className={`font-bold text-ink-800 ${compact ? 'text-[12px]' : 'text-[14px]'}`}>
            {analysis.treatment.heading}
          </p>
          <p className={`leading-relaxed text-ink-800 ${compact ? 'text-[12px]' : 'text-[14px]'}`}>
            {analysis.treatment.text}
          </p>
          <p className={`text-ink-500 ${compact ? 'mt-1 text-[11px]' : 'mt-1.5 text-[12px]'}`}>
            {analysis.treatment.disclaimer}
          </p>
        </div>
      )}
      <p className={`text-ink-400 ${compact ? 'mt-1.5 text-[10px]' : 'mt-2.5 text-[11px]'}`}>
        Checked at {clockTime(analysis.createdAt)}
      </p>
    </section>
  )
}

function titleCase(value: string | null | undefined): string {
  if (!value) return '—'
  return value
    .toLowerCase()
    .split(/[\s_]+/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}
