/**
 * Presentational primitives.
 *
 * These are the only components that decide how something looks. Pages compose
 * them; nothing here knows about irrigation. That separation is what keeps the
 * stat tiles, alert rows and charts visually consistent with one another.
 */

import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { num } from '../lib/format'
import type { LinkState, MoistureBand, RiskLevel, Severity } from '../types'

/* -------------------------------------------------------------------------- */
/* Layout                                                                      */
/* -------------------------------------------------------------------------- */

export function Card({
  title,
  action,
  children,
  className = '',
  bodyClassName = 'card-body',
}: {
  title?: string
  action?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
}) {
  return (
    <section className={`card ${className}`}>
      {title && (
        <header className="card-header">
          <h2 className="card-title">{title}</h2>
          {action}
        </header>
      )}
      <div className={bodyClassName}>{children}</div>
    </section>
  )
}

/**
 * A card with a title row and no border of its own.
 *
 * The shell is a bento grid: a page that stacks bordered boxes on a bordered
 * page reads as a pile, whereas a divided surface reads as one instrument. This
 * is the cell that divides; the grid supplies the hairlines.
 */
export function SectionCard({
  title,
  action,
  children,
  className = '',
  bodyClassName = '',
}: {
  title?: string
  action?: ReactNode
  children: ReactNode
  className?: string
  bodyClassName?: string
}) {
  return (
    <section className={className}>
      {title && (
        <div className="cell-head">
          <h2 className="cell-title">{title}</h2>
          {action}
        </div>
      )}
      <div className={bodyClassName}>{children}</div>
    </section>
  )
}

/** A bordered strip of equal-width statistic cells, as in the reference. */
export function StatRow({ items }: { items: { value: string | number; label: string; unit?: string }[] }) {
  return (
    <div className="grid grid-cols-2 divide-x divide-ink-200 border-b border-ink-200 sm:grid-cols-4 sm:divide-x">
      {items.map((item) => (
        <div key={item.label} className="stat-cell">
          <div className="stat-value">
            {item.value}
            {item.unit && <span className="ml-0.5 text-[13px] font-medium text-ink-400">{item.unit}</span>}
          </div>
          <div className="stat-label">{item.label}</div>
        </div>
      ))}
    </div>
  )
}

/* -------------------------------------------------------------------------- */
/* Status                                                                      */
/* -------------------------------------------------------------------------- */

const LINK_STYLE: Record<LinkState, { dot: string; text: string; ring: string }> = {
  online: { dot: 'bg-status-healthy', text: 'text-status-healthy', ring: 'ring-status-healthySoft' },
  degraded: { dot: 'bg-status-warn', text: 'text-status-warn', ring: 'ring-status-warnSoft' },
  offline: { dot: 'bg-status-alert', text: 'text-status-alert', ring: 'ring-status-alertSoft' },
}

const BAND_STYLE: Record<MoistureBand, { bg: string; text: string; label: string }> = {
  dry: { bg: 'bg-status-warnSoft', text: 'text-status-warn', label: 'Dry' },
  optimal: { bg: 'bg-status-healthySoft', text: 'text-status-healthy', label: 'Optimal' },
  wet: { bg: 'bg-brand-100', text: 'text-brand-700', label: 'Wet' },
}

/** Neutral chip, for states that are neither a moisture band nor a risk. */
const IDLE_CHIP = { bg: 'bg-ink-100', text: 'text-ink-500', label: 'Idle' }

const RISK_STYLE: Record<RiskLevel, { bg: string; text: string; label: string }> = {  healthy: { bg: 'bg-status-healthySoft', text: 'text-status-healthy', label: 'Healthy' },
  watch: { bg: 'bg-status-warnSoft', text: 'text-status-warn', label: 'Watch' },
  risk: { bg: 'bg-status-alertSoft', text: 'text-status-alert', label: 'Risk' },
}

const SEVERITY_STYLE: Record<Severity, { bg: string; text: string; label: string }> = {
  info: { bg: 'bg-brand-100', text: 'text-brand-700', label: 'Info' },
  warning: { bg: 'bg-status-warnSoft', text: 'text-status-warn', label: 'Warning' },
  critical: { bg: 'bg-status-alertSoft', text: 'text-status-alert', label: 'Critical' },
}

export function LinkBadge({ state, showLabel = true }: { state: LinkState; showLabel?: boolean }) {
  const style = LINK_STYLE[state]
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="relative flex h-2 w-2">
        {state === 'online' && (
          <span className={`absolute inline-flex h-full w-full animate-pulseRing rounded-full ${style.dot}`} />
        )}
        <span className={`relative inline-flex h-2 w-2 rounded-full ${style.dot}`} />
      </span>
      {showLabel && <span className={`text-[11px] font-medium ${style.text}`}>{state}</span>}
    </span>
  )
}

export function Chip({
  children,
  tone = 'idle',
}: {
  children: ReactNode
  tone?: keyof typeof BAND_STYLE | 'idle'
}) {
  const style = tone === 'idle' ? IDLE_CHIP : BAND_STYLE[tone]
  return <span className={`chip ${style.bg} ${style.text}`}>{children}</span>
}

export function MoistureChip({ band }: { band: MoistureBand }) {
  const style = BAND_STYLE[band]
  return <span className={`chip ${style.bg} ${style.text}`}>{style.label}</span>
}

export function RiskChip({ level }: { level: RiskLevel }) {
  const style = RISK_STYLE[level]
  return <span className={`chip ${style.bg} ${style.text}`}>{style.label}</span>
}

export function SeverityChip({ severity }: { severity: Severity }) {
  const style = SEVERITY_STYLE[severity]
  return <span className={`chip ${style.bg} ${style.text}`}>{style.label}</span>
}

/* -------------------------------------------------------------------------- */
/* Metrics                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * A headline figure with an optional unit suffix and caption. The reference
 * layout places these in a bare three-column strip, so this has no border of
 * its own and inherits the row from the page.
 */
export function Metric({
  value,
  unit,
  caption,
  align = 'left',
}: {
  value: string | number
  unit?: string
  caption: string
  align?: 'left' | 'center'
}) {
  return (
    <div className={align === 'center' ? 'text-center' : ''}>
      <div className="tnum text-[30px] font-semibold leading-none tracking-tight text-ink-900">
        {value}
        {unit && <span className="ml-1 text-[15px] font-medium text-ink-400">{unit}</span>}
      </div>
      <div className="mt-1.5 text-[11px] text-ink-500">{caption}</div>
    </div>
  )
}

/**
 * Icon plus a large count, with a secondary line beneath. Used for the system
 * and alert summaries.
 */
export function IconStat({
  tone,
  glyph,
  value,
  label,
  sublabel,
  action,
}: {
  tone: 'healthy' | 'warn' | 'alert' | 'idle'
  glyph: ReactNode
  value: string | number
  label: string
  sublabel?: string
  action?: { label: string; to: string; primary?: boolean }
}) {
  const toneClass = {
    healthy: 'bg-status-healthy',
    warn: 'bg-status-warn',
    alert: 'bg-status-alert',
    idle: 'bg-status-idle',
  }[tone]

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-3">
        <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full ${toneClass} text-white`}>
          {glyph}
        </span>
        <div className="min-w-0">
          <div className="tnum text-[24px] font-semibold leading-none tracking-tight text-ink-900">
            {value}
          </div>
          <div className="mt-1 text-[10px] font-medium uppercase tracking-[0.08em] text-ink-400">
            {label}
          </div>
        </div>
        {sublabel && <div className="ml-auto text-right text-[11px] text-ink-500">{sublabel}</div>}
      </div>
      {action && (
        <Link to={action.to} className={action.primary ? 'btn-primary self-start' : 'btn-secondary self-start'}>
          {action.label}
        </Link>
      )}
    </div>
  )
}

/**
 * Large percentage with a usage caption and progress track. Drawn to float
 * right of the summary row in the reference layout.
 */
export function UsageCard({
  percent,
  caption,
  action,
}: {
  percent: number
  caption: string
  action?: { label: string; to: string }
}) {
  const clamped = Math.max(0, Math.min(100, percent))
  return (
    <Card className="h-full">
      <div className="card-body flex h-full flex-col">
        <div className="flex items-start gap-3">
          <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-status-healthy text-white">
            <GaugeGlyph />
          </span>
          <div className="min-w-0">
            <div className="tnum text-[30px] font-semibold leading-none tracking-tight text-ink-900">
              {clamped.toFixed(2)}
              <span className="ml-0.5 text-[16px] font-medium text-ink-400">%</span>
            </div>
            <div className="mt-1.5 text-[11px] text-ink-500">{caption}</div>
          </div>
        </div>
        <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-ink-100">
          <div
            className="h-full rounded-full bg-status-healthy transition-[width] duration-500"
            style={{ width: `${clamped}%` }}
          />
        </div>
        {action && (
          <Link to={action.to} className="btn-secondary mt-auto self-end">
            {action.label}
          </Link>
        )}
      </div>
    </Card>
  )
}

/* -------------------------------------------------------------------------- */
/* Glyphs                                                                      */
/* -------------------------------------------------------------------------- */

const stroke = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const }

/**
 * A labelled metric with a status badge, for the sensor grid.
 *
 * The value is the largest thing in the cell and the unit sits with it, because
 * a sensor board read at arm's length outdoors is read as a number first and
 * interpreted second. The badge carries the interpretation.
 */
export function Reading({
  label,
  value,
  unit,
  tone = 'neutral',
  badge,
}: {
  label: string
  value: string
  unit?: string
  tone?: 'neutral' | 'healthy' | 'warn' | 'alert' | 'idle'
  badge?: string
}) {
  const valueTone = {
    neutral: 'text-ink-900',
    healthy: 'text-status-healthy',
    warn: 'text-status-warn',
    alert: 'text-status-alert',
    idle: 'text-ink-400',
  }[tone]

  return (
    <div className="panel min-w-0 px-[clamp(0.6rem,1.8cqw,0.85rem)] py-[clamp(0.5rem,1.5cqw,0.75rem)]">
      <div className="flex items-baseline justify-between gap-1.5">
        <span className="micro truncate">{label}</span>
        {badge && <ToneBadge tone={tone}>{badge}</ToneBadge>}
      </div>
      <div
        className={`tnum mt-1.5 text-[clamp(1.125rem,4cqw,1.5rem)] font-semibold leading-none ${valueTone}`}
      >
        {value}
        {unit && <span className="ml-0.5 text-[11px] font-semibold text-ink-400">{unit}</span>}
      </div>
    </div>
  )
}

const BADGE_TONE = {
  neutral: 'bg-ink-100 text-ink-500',
  healthy: 'bg-status-healthySoft text-status-healthy',
  warn: 'bg-status-warnSoft text-status-warn',
  alert: 'bg-status-alertSoft text-status-alert',
  idle: 'bg-ink-100 text-ink-400',
}

export function ToneBadge({
  tone = 'neutral',
  children,
}: {
  tone?: 'neutral' | 'healthy' | 'warn' | 'alert' | 'idle'
  children: ReactNode
}) {
  return (
    <span
      className={`shrink-0 rounded-pill px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-[0.05em] ${BADGE_TONE[tone]}`}
    >
      {children}
    </span>
  )
}

/**
 * A nutrient bar drawn against its sufficiency band.
 *
 * The band is the point: a bare number cannot say whether 40 ppm of nitrogen is
 * a shortage or a surplus. The track shows the full plausible range, a lighter
 * section marks the band V1's models treat as sufficient, and the fill marks
 * where this reading sits.
 */
export function NutrientBar({
  label,
  value,
  low,
  mid,
  high,
  scaleMax,
  unit = 'ppm',
}: {
  label: string
  value: number
  low: number
  mid: number
  high: number
  scaleMax: number
  unit?: string
}) {
  const position = Math.max(0, Math.min(100, (value / scaleMax) * 100))
  const inBand = value >= low && value <= high
  const below = value < low

  return (
    <div className="min-w-0">
      <div className="flex items-baseline justify-between gap-1.5">
        <span className="text-[11px] font-medium text-ink-600">{label}</span>
        <span className="tnum text-[11px] font-semibold text-ink-900">
          {num(value, 0)}
          <span className="ml-0.5 text-[9px] font-medium text-ink-400">{unit}</span>
        </span>
      </div>
      {/* The track is a groove, not a painted line: an inset shadow gives the bar a
          floor for the fill and the marker to sit in. */}
      <div className="relative mt-1.5 h-2 w-full overflow-hidden rounded-pill bg-ink-100 shadow-[inset_0_1px_2px_rgba(15,23,42,0.12)]">
        <div
          className="absolute inset-y-0 bg-brand-100"
          style={{ left: `${(low / scaleMax) * 100}%`, width: `${((high - low) / scaleMax) * 100}%` }}
        />
        <div
          className="absolute inset-y-0 w-0.5 bg-ink-300"
          style={{ left: `${(mid / scaleMax) * 100}%` }}
        />
        <div
          className={`absolute inset-y-0 w-1 rounded-pill shadow-[0_0_0_1px_rgba(255,255,255,0.7)] transition-[left] duration-500 ${
            inBand ? 'bg-status-healthy' : below ? 'bg-status-warn' : 'bg-brand-600'
          }`}
          style={{ left: `calc(${position}% - 2px)` }}
        />
      </div>
      <div className="mt-1 flex justify-between text-[9px] text-ink-400">
        <span>{num(low, 0)}</span>
        <span className={inBand ? 'text-status-healthy' : below ? 'text-status-warn' : 'text-ink-500'}>
          {inBand ? 'In range' : below ? 'Below' : 'Above'}
        </span>
        <span>{num(high, 0)}</span>
      </div>
    </div>
  )
}

/**
 * The turbulence that warps the page gradient.
 *
 * Rendered once at the root. `feDisplacementMap` needs the turbulence as a
 * result to read from, so both live in one filter; the `scale` is the amount of
 * displacement and is deliberately small, because at a larger value the wash
 * stops reading as a gradient and starts reading as a mistake.
 */
export function GrainFilter() {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      style={{ position: 'absolute', width: 0, height: 0, overflow: 'hidden' }}
    >
      <filter id="grain-warp">
        <feTurbulence
          type="fractalNoise"
          baseFrequency="0.008 0.014"
          numOctaves={3}
          seed={7}
          stitchTiles="stitch"
          result="noise"
        />
        <feDisplacementMap
          in="SourceGraphic"
          in2="noise"
          scale={54}
          xChannelSelector="R"
          yChannelSelector="G"
        />
      </filter>
    </svg>
  )
}

/** A large glyph that deliberately breaks its card's edge. */
export function SkyGlyph({ size = 96, className = '' }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      aria-hidden="true"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth={2.2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="24" cy="24" r="9" />
      <path d="M24 8v3M24 37v3M8 24h3M37 24h3M13 13l2 2M33 33l2 2M13 35l2-2M33 15l2-2" />
      <path d="M20 52h24a9 9 0 0 0 2-17.8A13 13 0 0 0 22 36a8 8 0 0 0-2 16Z" />
    </svg>
  )
}

export function RainGlyph({ size = 96, className = '' }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      aria-hidden="true"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth={2.2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M18 40h28a9 9 0 0 0 2-17.8A13 13 0 0 0 20 24a8 8 0 0 0-2 16Z" />
      <path d="M22 47l-3 7M32 47l-3 7M42 47l-3 7" />
    </svg>
  )
}

export function GaugeGlyph({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 20a8 8 0 1 1 8-8" {...stroke} />
      <path d="M12 12l4.5-3.5" {...stroke} />
      <circle cx="12" cy="12" r="1.4" fill="currentColor" />
    </svg>
  )
}

export function DropGlyph({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3.5c3 3.6 5.5 6.4 5.5 9.3A5.5 5.5 0 0 1 6.5 12.8C6.5 9.9 9 7.1 12 3.5Z" {...stroke} />
    </svg>
  )
}

export function AlertGlyph({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 8.5v4.2" {...stroke} />
      <circle cx="12" cy="16.2" r="1" fill="currentColor" />
      <path d="M10.3 4.2 2.9 17.4A2 2 0 0 0 4.6 20.4h14.8a2 2 0 0 0 1.7-3L13.7 4.2a2 2 0 0 0-3.4 0Z" {...stroke} />
    </svg>
  )
}

export function ChipGlyph({ size = 18 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <rect x="7" y="7" width="10" height="10" rx="2" {...stroke} />
      <path d="M10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4" {...stroke} />
    </svg>
  )
}

export function LeafGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 20c0-8 5-14 16-15 0 10-5 15-12 15H4Z" {...stroke} />
      <path d="M9 15c2-3 5-5 8-6" {...stroke} />
    </svg>
  )
}

export function SignalGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 18v-3M9 18v-6M14 18v-9M19 18V6" {...stroke} />
    </svg>
  )
}

export function BoltGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M13 3 5 14h6l-1 7 8-11h-6l1-7Z" {...stroke} />
    </svg>
  )
}

/* -------------------------------------------------------------------------- */
/* Shell controls                                                              */
/* -------------------------------------------------------------------------- */

/** A segmented pill group. The reference uses one for periods and one for modes. */
export function PillSegment<T extends string | number>({
  options,
  value,
  onChange,
  ariaLabel,
}: {
  options: { label: string; value: T }[]
  value: T
  onChange: (value: T) => void
  ariaLabel?: string
}) {
  return (
    <div
      role="group"
      aria-label={ariaLabel}
      className="inline-flex items-center gap-0.5 rounded-pill bg-ink-100 p-0.5"
    >
      {options.map((option) => (
        <button
          key={String(option.value)}
          type="button"
          onClick={() => onChange(option.value)}
          aria-pressed={option.value === value}
          className={`rounded-pill px-3 py-1 text-[11px] font-semibold transition-colors ${
            option.value === value
              ? 'bg-white text-ink-900 shadow-card'
              : 'text-ink-500 hover:text-ink-900'
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

/** A circular icon button, for the utility cluster in the top bar. */
export function GlyphButton({
  label,
  onClick,
  active = false,
  children,
}: {
  label: string
  onClick?: () => void
  active?: boolean
  children: ReactNode
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className={`flex h-9 w-9 items-center justify-center rounded-pill border transition-colors ${
        active
          ? 'border-ink-200 bg-ink-100 text-ink-900'
          : 'border-ink-200 bg-white text-ink-500 hover:bg-ink-50 hover:text-ink-900'
      }`}
    >
      {children}
    </button>
  )
}

/**
 * The overflow affordance the reference puts on every chart card.
 *
 * It is a button rather than decoration so the control is reachable by keyboard
 * and announced as a control, even where the menu behind it is not wired yet.
 */
export function Kebab({ label = 'More options' }: { label?: string }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className="flex h-7 w-7 items-center justify-center rounded-pill text-ink-400 transition-colors hover:bg-ink-100 hover:text-ink-700"
    >
      <svg width={16} height={16} viewBox="0 0 24 24" aria-hidden="true">
        <circle cx="12" cy="5.5" r="1.4" fill="currentColor" />
        <circle cx="12" cy="12" r="1.4" fill="currentColor" />
        <circle cx="12" cy="18.5" r="1.4" fill="currentColor" />
      </svg>
    </button>
  )
}

/**
 * A compact bar series for a trend with no axis.
 *
 * Deliberately not a chart: at this size the reader needs the shape and the
 * last value, not a scale, and an axis would cost more space than it returns.
 */
export function SparkBars({
  values,
  height = 44,
  tone = '#2563eb',
  highlightLast = true,
}: {
  values: number[]
  height?: number
  tone?: string
  highlightLast?: boolean
}) {
  if (values.length === 0) return <div style={{ height }} />
  const peak = Math.max(...values, 1e-9)
  const gap = values.length > 24 ? 1 : 2
  return (
    <div className="flex items-end" style={{ height, gap }}>
      {values.map((value, index) => {
        const last = index === values.length - 1
        return (
          <span
            key={index}
            className="flex-1 rounded-t-[2px]"
            style={{
              height: `${Math.max(2, (Math.max(0, value) / peak) * 100)}%`,
              backgroundColor: highlightLast && last ? tone : `${tone}59`,
            }}
          />
        )
      })}
      <span style={{ width: gap - 1 }} />
    </div>
  )
}

/** A proportion bar with an optional second, hatched segment. */
export function ProportionBar({
  value,
  tone = 'bg-brand-600',
  className = '',
}: {
  value: number
  tone?: string
  className?: string
}) {
  const clamped = Math.max(0, Math.min(100, value))
  return (
    <div className={`hatch ${className}`}>
      <div
        className={`h-full rounded-pill ${tone} transition-[width] duration-500`}
        style={{ width: `${clamped}%` }}
      />
    </div>
  )
}

export function GridGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <rect x="4" y="4" width="7" height="7" rx="2" {...stroke} />
      <rect x="13" y="4" width="7" height="7" rx="2" {...stroke} />
      <rect x="4" y="13" width="7" height="7" rx="2" {...stroke} />
      <rect x="13" y="13" width="7" height="7" rx="2" {...stroke} />
    </svg>
  )
}

export function BellGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M18 15V11a6 6 0 1 0-12 0v4l-1.6 2.2A1 1 0 0 0 5.2 19h13.6a1 1 0 0 0 .8-1.8L18 15Z" {...stroke} />
      <path d="M10 19.5a2 2 0 0 0 4 0" {...stroke} />
    </svg>
  )
}

export function SearchGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="11" cy="11" r="6.5" {...stroke} />
      <path d="m16 16 4 4" {...stroke} />
    </svg>
  )
}

export function TrendGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 16.5 9 11l3.5 3.5L20 7" {...stroke} />
      <path d="M15 7h5v5" {...stroke} />
    </svg>
  )
}

export function LayersGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="m12 3 9 5-9 5-9-5 9-5Z" {...stroke} />
      <path d="m3.5 12.5 8.5 4.7 8.5-4.7" {...stroke} />
    </svg>
  )
}

export function SlidersGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 8h10M18 8h2M4 16h4M12 16h8" {...stroke} />
      <circle cx="16" cy="8" r="2" {...stroke} />
      <circle cx="10" cy="16" r="2" {...stroke} />
    </svg>
  )
}

export function TargetGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="8.5" {...stroke} />
      <circle cx="12" cy="12" r="4" {...stroke} />
      <circle cx="12" cy="12" r="1" fill="currentColor" />
    </svg>
  )
}

export function ClockGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="8.5" {...stroke} />
      <path d="M12 7.5V12l3 1.8" {...stroke} />
    </svg>
  )
}

export function PinGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 21c4-4.5 6-7.4 6-10a6 6 0 1 0-12 0c0 2.6 2 5.5 6 10Z" {...stroke} />
      <circle cx="12" cy="11" r="2.2" {...stroke} />
    </svg>
  )
}

/** The small, single-weight weather glyph used inside chips and lists. */
export function WeatherGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="8.5" cy="9" r="3.5" {...stroke} />
      <path d="M8.5 3v1.5M8.5 13.5V15M3 9h1.5M12.5 9H14M4.7 5.2l1 1M11.3 11.8l1 1M4.7 12.8l1-1" {...stroke} />
      <path d="M10 19.5h8.2a2.8 2.8 0 0 0 .3-5.6 4.2 4.2 0 0 0-8-1 3.3 3.3 0 0 0-.5 6.6Z" {...stroke} />
    </svg>
  )
}

export function TaskGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <rect x="4" y="4" width="16" height="16" rx="3" {...stroke} />
      <path d="M8.5 12.2l2.4 2.4 4.6-5" {...stroke} />
    </svg>
  )
}

export function UserGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="8.5" r="3.5" {...stroke} />
      <path d="M5 19.5a7 7 0 0 1 14 0" {...stroke} />
    </svg>
  )
}

export function RefreshGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M20 11a8 8 0 0 0-13.7-5.2L4 8" {...stroke} />
      <path d="M4 4v4h4" {...stroke} />
      <path d="M4 13a8 8 0 0 0 13.7 5.2L20 16" {...stroke} />
      <path d="M20 20v-4h-4" {...stroke} />
    </svg>
  )
}

export function MapGlyph({ size = 16 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="m3.5 6.5 5-2 7 2 5-2v13l-5 2-7-2-5 2v-13Z" {...stroke} />
      <path d="M8.5 4.5v13M15.5 6.5v13" {...stroke} />
    </svg>
  )
}
