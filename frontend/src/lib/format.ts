/**
 * Small formatting helpers.
 *
 * Every number the operator reads passes through here, so units, precision and
 * the "no data yet" case are decided once instead of at each call site. A
 * missing value renders as an em dash rather than `NaN`, which matters because
 * a device that has not reported yet is a normal state, not an error.
 */

const DASH = '—'

export function num(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH
  return value.toFixed(digits)
}

export function pct(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH
  return `${(value * 100).toFixed(digits)}%`
}

export function int(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH
  return Math.round(value).toLocaleString()
}

/** Durations read better as "3d 4h" than as a raw second count. */
export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return DASH
  const total = Math.max(0, Math.floor(seconds))
  const days = Math.floor(total / 86_400)
  const hours = Math.floor((total % 86_400) / 3_600)
  const minutes = Math.floor((total % 3_600) / 60)
  if (days > 0) return `${days}d ${hours}h`
  if (hours > 0) return `${hours}h ${minutes}m`
  if (minutes > 0) return `${minutes}m`
  return `${total}s`
}

export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return DASH
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return DASH
  const seconds = Math.round((Date.now() - then) / 1000)
  if (seconds < 0) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3_600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86_400) return `${Math.floor(seconds / 3_600)}h ago`
  return `${Math.floor(seconds / 86_400)}d ago`
}

export function clockTime(iso: string | null | undefined): string {
  if (!iso) return DASH
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return DASH
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function titleCase(value: string | null | undefined): string {
  if (!value) return DASH
  return value
    .toLowerCase()
    .split(/[\s_]+/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}
