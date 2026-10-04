/**
 * Registering a new field.
 *
 * A sowing is the one thing an operator enters by hand, and everything the
 * planner says afterwards is derived from it. Three decisions therefore have to be
 * made deliberately rather than typed hopefully: which crop, where it is planted,
 * and when. This form treats them that way.
 *
 * **The crop is chosen from cards, not a dropdown.** A dropdown of fourteen names
 * is a list, and a list is not a decision. Each card carries the three numbers
 * that make the crop what it is - the crop coefficient it starts at, how deep it
 * roots, and how much of its available water it will draw down before it is
 * stressed - so choosing cotton over rice is a choice between 65% and 10%, not
 * between two words. Those numbers are also why two fields with identical
 * sensors get different recommendations, and an operator who cannot see that will
 * read the difference as an error.
 *
 * **The consequence of a pairing is shown before it is committed.** Rice on
 * sprinkler grows, and wastes most of the water; the engine knows that and says
 * so, and the warning is rendered here rather than discovered on a water bill a
 * season later.
 *
 * **The result comes back with the field.** The registration response carries the
 * plan the engine produced, so the summary below the form is the actual first
 * answer for the new field and not a placeholder that fills in on the next poll.
 */

import { useCallback, useMemo, useState } from 'react'
import { api, ApiError } from '../api/client'
import { usePollingFetch } from '../hooks/usePollingFetch'
import { num, pct, titleCase } from '../lib/format'
import type { CropOption, FieldCreate, FieldRecord } from '../types'

/**
 * The crops offered as cards, in the order a field is usually cropped.
 *
 * The catalogue holds fourteen; these four are the ones the fleet is built around
 * and the ones with tuned data behind them. The rest stay reachable through the
 * catalogue endpoint rather than being deleted - a grower with groundnuts should
 * not be told the system does not do groundnuts, only that it is not offering them
 * on the quick path.
 */
const PRIMARY_CROPS = ['Wheat', 'Rice', 'Maize', 'Cotton'] as const

type FormState = {
  fieldId: string
  station: string
  crop: string
  cropVariant: string
  sowingDate: string
  soilType: string
  methodName: string
  fieldAreaM2: string
  notes: string
}

const EMPTY: FormState = {
  fieldId: '',
  station: '',
  crop: '',
  cropVariant: '',
  sowingDate: '',
  soilType: '',
  methodName: '',
  fieldAreaM2: '',
  notes: '',
}

/** Today, as YYYY-MM-DD in local time rather than UTC. */
function today(): string {
  const now = new Date()
  const pad = (value: number) => String(value).padStart(2, '0')
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`
}

/**
 * A field id derived from the station and crop, so the common case needs no typing.
 *
 * Ids are what appears on every card and in every plan, so making them short and
 * predictable is worth more than making them clever.
 */
function suggestId(station: string, crop: string): string {
  const slug = (value: string) =>
    value
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
  const parts = [slug(station), slug(crop), '01'].filter(Boolean)
  return parts.join('-')
}

/**
 * Client-side validation, mirroring `FieldCreate` in the backend.
 *
 * The server validates everything regardless; this exists so a mistake is caught
 * before a round trip and reported against the input that caused it, which is the
 * difference between a form that corrects you and one that shouts.
 */
function validate(form: FormState, cropOption: CropOption | undefined): string[] {
  const problems: string[] = []
  if (!form.fieldId.trim()) problems.push('Give the field a name or id.')
  else if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(form.fieldId.trim())) {
    problems.push('The field id may use letters, numbers, dots, dashes and underscores.')
  }
  if (!form.station) problems.push('Choose the weather station nearest the field.')
  if (!form.crop) problems.push('Choose a crop.')
  if (!form.sowingDate) problems.push('Enter the sowing date.')
  else if (Number.isNaN(Date.parse(form.sowingDate))) {
    problems.push('The sowing date is not a date.')
  } else if (Date.parse(form.sowingDate) > Date.now()) {
    problems.push('The sowing date is in the future. Nothing is growing yet.')
  }
  if (form.fieldAreaM2) {
    const area = Number(form.fieldAreaM2)
    if (!Number.isFinite(area) || area <= 0) problems.push('The area must be a positive number.')
  }
  if (cropOption && !cropOption.variants.includes(form.cropVariant) && form.cropVariant) {
    problems.push(`"${form.cropVariant}" is not a known variety of ${cropOption.name}.`)
  }
  return problems
}

/** One crop, as a selectable card. */
function CropCard({
  crop,
  selected,
  onSelect,
}: {
  crop: CropOption
  selected: boolean
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={`group flex flex-col rounded-card border p-3 text-left transition-colors ${
        selected
          ? 'border-ink-900 bg-ink-900 text-white'
          : 'border-ink-200 bg-white/70 hover:border-ink-400 hover:bg-white'
      }`}
    >
      <span className="flex items-baseline justify-between gap-2">
        <span className={`text-[14px] font-semibold ${selected ? 'text-white' : 'text-ink-900'}`}>
          {titleCase(crop.name)}
        </span>
        <span className={`micro ${selected ? 'text-white/60' : ''}`}>
          {num(crop.seasonGdd, 0)} GDD
        </span>
      </span>
      <span className={`mt-1.5 text-[11px] leading-snug ${selected ? 'text-white/75' : 'text-ink-500'}`}>
        Roots {num(crop.rootDepthMaxM, 1)} m · draws down {pct(crop.depletionFractionP)} of its
        water before it is stressed · Kc {num(crop.kcInitial, 2)} to {num(crop.kcMid, 2)}
      </span>
    </button>
  )
}

/**
 * What the engine makes of the field as entered, before it is saved.
 *
 * Only the figures that follow directly from the crop and the sowing date are
 * shown, and each is labelled with where it comes from. Quoting a season length
 * in GDD alone would mean nothing to most people, so it is paired with the stage
 * boundaries it produces.
 */
function CropDetail({ crop }: { crop: CropOption }) {
  const rows: { label: string; value: string }[] = [
    { label: 'Peak water use', value: `Kc ${num(crop.kcMid, 2)}` },
    { label: 'Roots reach', value: `${num(crop.rootDepthMaxM, 1)} m` },
    { label: 'Stress threshold', value: `${pct(crop.depletionFractionP)} of available water` },
    { label: 'Season length', value: `${crop.seasonDays} days` },
    { label: 'Soil pH', value: `${num(crop.optimalPh[0], 1)} to ${num(crop.optimalPh[1], 1)}` },
    {
      label: 'Yield potential',
      value: `${num(crop.yieldPotentialTHa, 1)} t/ha (Ky ${num(crop.ky, 2)})`,
    },
  ]
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-3">
      {rows.map((row) => (
        <div key={row.label} className="min-w-0">
          <dt className="micro truncate">{row.label}</dt>
          <dd className="type-value mt-0.5 font-semibold text-ink-900">{row.value}</dd>
        </div>
      ))}
    </dl>
  )
}

/** The engine's verdict on the crop and method pairing, when there is one. */
function MethodWarning({ crop, method }: { crop: CropOption; method: string }) {
  const reason = crop.warnings[method]
  if (!reason) return null
  return (
    <p
      role="status"
      className="rounded-card border border-status-warn/30 bg-status-warnSoft px-3 py-2 text-[11px] leading-relaxed text-status-warn"
    >
      <span className="font-semibold">{method} on {titleCase(crop.name)}: </span>
      {reason}
    </p>
  )
}

/** The saved field's first plan, so the result is visible without a page reload. */
function RegisteredPlan({ field }: { field: FieldRecord }) {
  const plan = field.plan
  if (!plan) {
    return (
      <p className="text-[11px] text-status-warn">
        Registered. The engine could not plan it yet{field.planError ? `: ${field.planError}` : '.'}
      </p>
    )
  }
  const need = Number(plan.requirement.net_requirement_mm ?? 0)
  return (
    <div className="rounded-card border border-status-healthy/30 bg-status-healthySoft p-3">
      <p className="text-[12px] font-semibold text-status-healthy">
        {field.fieldId} registered and planned
      </p>
      <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1.5 sm:grid-cols-4">
        <Fact label="Stage" value={titleCase(plan.stage)} />
        <Fact label="Crop age" value={`${plan.daysSinceSowing} d`} />
        <Fact label="Root zone" value={`${num(plan.rootDepthCm, 0)} cm`} />
        <Fact label="14-day need" value={`${num(need, 1)} mm`} />
      </dl>
      <p className="mt-2 text-[10px] text-ink-500">
        Planned for {plan.asOf}, the latest day the weather record covers.
      </p>
    </div>
  )
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="micro truncate">{label}</dt>
      <dd className="tnum type-value mt-0.5 font-semibold text-ink-900">{value}</dd>
    </div>
  )
}

/** A labelled form control. */
function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: React.ReactNode
}) {
  return (
    <label className="block min-w-0">
      <span className="micro">{label}</span>
      {children}
      {hint && <span className="mt-0.5 block text-[10px] leading-snug text-ink-400">{hint}</span>}
    </label>
  )
}

const CONTROL =
  'mt-1 w-full rounded-card border border-ink-200 bg-white px-2.5 py-1.5 text-[12px] ' +
  'text-ink-900 outline-none transition-colors focus:border-brand-500 focus:ring-2 ' +
  'focus:ring-brand-100 disabled:cursor-not-allowed disabled:bg-ink-50'

export function AddField({ onRegistered }: { onRegistered: (fieldId: string) => void }) {
  const catalogue = usePollingFetch(useCallback(() => api.crops(), []), 0, 300_000)
  const options = usePollingFetch(useCallback(() => api.options(), []), 0, 300_000)

  const [form, setForm] = useState<FormState>(EMPTY)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState<FieldRecord | null>(null)

// The `?? []` fallbacks are wrapped rather than left inline: an empty array
// literal built during render is a new object every render, which makes every
// `useMemo` below recompute on every keystroke and defeats the point of them.
const crops = useMemo(() => catalogue.data?.crops ?? [], [catalogue.data])
const stations = useMemo(() => options.data?.stations ?? [], [options.data])
const soils = useMemo(() => options.data?.soils ?? [], [options.data])
const allMethods = useMemo(() => options.data?.methods ?? [], [options.data])

const primary = useMemo(
  () => PRIMARY_CROPS.map((name) => crops.find((c) => c.name === name)).filter(Boolean) as CropOption[],
  [crops],
)
// Everything the catalogue holds that is not one of the four. Offered as chips
// rather than omitted: the fleet may already be cropping them.
const secondary = useMemo(
  () => crops.filter((c) => !PRIMARY_CROPS.includes(c.name as (typeof PRIMARY_CROPS)[number])),
  [crops],
)
const cropOption = useMemo(
  () => crops.find((c) => c.name === form.crop),
  [crops, form.crop],
)

  // The method list leads with what suits this crop, ordered by application
  // efficiency, then everything else. The unsuitable ones stay reachable and carry
  // their warning, because a grower with only a sprinkler should not be stopped by
  // a recommendation.
  //
  // Deduplicated because a concatenation of overlapping lists renders every shared
  // method twice, which React reports as a duplicate key and which would show the
  // operator the same method in the dropdown twice.
  const methods = useMemo(() => {
    const ordered = [...(cropOption?.suitableMethods ?? []), ...allMethods]
    return ordered.filter((name, index) => ordered.indexOf(name) === index)
  }, [cropOption, allMethods])

  const set = useCallback(<K extends keyof FormState>(key: K, value: FormState[K]) => {
    setForm((previous) => ({ ...previous, [key]: value }))
    setSaved(null)
  }, [])

  const problems = validate(form, cropOption)

  const submit = useCallback(
    async (event: React.FormEvent) => {
      event.preventDefault()
      if (problems.length > 0 || busy) return
      setBusy(true)
      setError(null)
      const body: FieldCreate = {
        fieldId: form.fieldId.trim(),
        station: form.station,
        crop: form.crop,
        sowingDate: form.sowingDate,
      }
      if (form.soilType) body.soilType = form.soilType
      if (form.methodName) body.methodName = form.methodName
      if (form.cropVariant) body.cropVariant = form.cropVariant
      if (form.fieldAreaM2) body.fieldAreaM2 = Number(form.fieldAreaM2)
      if (form.notes.trim()) body.notes = form.notes.trim()
      try {
        const created = await api.createField(body)
        setSaved(created)
        setForm(EMPTY)
        onRegistered(created.fieldId)
      } catch (cause) {
        setError(
          cause instanceof ApiError ? cause.message : 'The field could not be registered.',
        )
      } finally {
        setBusy(false)
      }
    },
    [busy, form, problems.length, onRegistered],
  )

  if (catalogue.error) {
    return (
      <div className="rounded-card border border-status-alert/30 bg-status-alertSoft p-4 text-[12px] text-status-alert">
        The crop catalogue could not be loaded: {catalogue.error}
      </div>
    )
  }

  return (
    <form onSubmit={submit} className="glass rounded-card p-4 sm:p-5">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-[15px] font-semibold tracking-tight text-ink-900">Register a field</h2>
        <span className="text-[10px] uppercase tracking-[0.08em] text-ink-400">
          Sowing · FAO-56 water balance
        </span>
      </div>

      {/* ---- Crop: the decision the rest of the plan follows from ---------- */}
      <fieldset className="mt-4">
        <legend className="micro">Crop type</legend>
        <div className="mt-1.5 grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-4">
          {primary.map((crop) => (
            <CropCard
              key={crop.name}
              crop={crop}
              selected={form.crop === crop.name}
              onSelect={() => {
                set('crop', crop.name)
                set('cropVariant', '')
                // Default to the crop's most efficient suitable method. Leaving
                // the method blank would silently pick Sprinkler, which is
                // arbitrary: for a rice field it is a pairing the engine warns
                // about, and for a wheat field it is merely an unstated choice.
                const preferred = crop.suitableMethods[0]
                if (preferred) set('methodName', preferred)
              }}
            />
          ))}
          {primary.length === 0 && (
            <p className="text-[12px] text-ink-400">Loading the crop catalogue…</p>
          )}
        </div>

        {/* The rest of the catalogue, as chips rather than cards. They are all
            selectable because a grower who farms barley cannot be told the system
            does not do barley - only that it is not one of the four headline
            crops. Same code path, less emphasis. */}
        {secondary.length > 0 && (
          <div className="mt-2 flex flex-wrap items-center gap-1.5">
            <span className="micro mr-0.5">Also</span>
            {secondary.map((crop) => {
              const selected = form.crop === crop.name
              return (
                <button
                  key={crop.name}
                  type="button"
                  aria-pressed={selected}
                  onClick={() => {
                    set('crop', crop.name)
                    set('cropVariant', '')
                    const preferred = crop.suitableMethods[0]
                    if (preferred) set('methodName', preferred)
                  }}
                  className={`rounded-pill border px-2.5 py-1 text-[11px] font-medium transition-colors ${
                    selected
                      ? 'border-ink-900 bg-ink-900 text-white'
                      : 'border-white/70 bg-white/60 text-ink-600 hover:border-ink-400 hover:bg-white'
                  }`}
                >
                  {titleCase(crop.name)}
                </button>
              )
            })}
          </div>
        )}
      </fieldset>

      {cropOption && (
        <div className="glass-text mt-3 p-3">
          <CropDetail crop={cropOption} />
        </div>
      )}

      {/* ---- Identity and placement -------------------------------------- */}
      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Field label="Field id" hint="Shown on every card and plan.">
          <input
            className={CONTROL}
            value={form.fieldId}
            placeholder={suggestId(form.station || 'HYD', cropOption?.name || 'Wheat')}
            onChange={(e) => set('fieldId', e.target.value)}
          />
        </Field>

        <Field label="Weather station" hint="Supplies ET0 and the rain record.">
          <select
            className={CONTROL}
            value={form.station}
            onChange={(e) => set('station', e.target.value)}
          >
            <option value="">Select…</option>
            {stations.map((station) => (
              <option key={station} value={station}>
                {station}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Sowing date" hint="Day 0 of the season.">
          <input
            type="date"
            className={CONTROL}
            value={form.sowingDate}
            max={today()}
            onChange={(e) => set('sowingDate', e.target.value)}
          />
        </Field>

        <Field label="Area" hint="Square metres. Sets volume, not depth.">
          <input
            className={CONTROL}
            inputMode="decimal"
            value={form.fieldAreaM2}
            placeholder="1000"
            onChange={(e) => set('fieldAreaM2', e.target.value)}
          />
        </Field>

        <Field label="Soil texture" hint="Sets how much water the root zone holds.">
          <select
            className={CONTROL}
            value={form.soilType}
            onChange={(e) => set('soilType', e.target.value)}
          >
            <option value="">Loam (default)</option>
            {soils.map((soil) => (
              <option key={soil} value={soil}>
                {titleCase(soil)}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Irrigation method" hint="Sets efficiency and event size.">
          <select
            className={CONTROL}
            value={form.methodName}
            onChange={(e) => set('methodName', e.target.value)}
          >
            <option value="">Sprinkler (default)</option>
            {methods.map((method) => (
              <option key={method} value={method}>
                {titleCase(method)}
                {cropOption?.warnings[method] ? ' — not advised' : ''}
              </option>
            ))}
          </select>
        </Field>

        {cropOption && cropOption.variants.length > 0 && (
          <Field label="Variety" hint="A short-duration hybrid finishes earlier.">
            <select
              className={CONTROL}
              value={form.cropVariant}
              onChange={(e) => set('cropVariant', e.target.value)}
            >
              <option value="">Standard</option>
              {cropOption.variants.map((variant) => (
                <option key={variant} value={variant}>
                  {titleCase(variant)}
                </option>
              ))}
            </select>
          </Field>
        )}

        <Field label="Notes" hint="Optional.">
          <input
            className={CONTROL}
            value={form.notes}
            placeholder="Block C, north terrace"
            onChange={(e) => set('notes', e.target.value)}
          />
        </Field>
      </div>

      {cropOption && form.methodName && (
        <div className="mt-3">
          <MethodWarning crop={cropOption} method={form.methodName} />
        </div>
      )}

      {problems.length > 0 && (
        <ul className="mt-3 space-y-0.5 text-[11px] text-status-warn">
          {problems.map((problem) => (
            <li key={problem}>{problem}</li>
          ))}
        </ul>
      )}

      {error && (
        <p
          role="alert"
          className="mt-3 rounded-card border border-status-alert/30 bg-status-alertSoft px-3 py-2 text-[11px] leading-relaxed text-status-alert"
        >
          {error}
        </p>
      )}

      {saved && (
        <div className="mt-3">
          <RegisteredPlan field={saved} />
        </div>
      )}

      <div className="mt-4 flex items-center gap-2">
        <button
          type="submit"
          className="btn-primary"
          disabled={busy || problems.length > 0 || catalogue.loading}
        >
          {busy ? 'Registering…' : 'Register field'}
        </button>
        {problems.length > 0 && (
          <span className="text-[11px] text-ink-400">
            {problems.length} field{problems.length === 1 ? '' : 's'} need attention
          </span>
        )}
      </div>
    </form>
  )
}