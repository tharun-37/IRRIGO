/**
 * Thin API client.
 *
 * Every call goes through `request`, so error handling, JSON parsing and the
 * base URL are decided in exactly one place. The Vite dev server proxies
 * `/api` to the backend, so no absolute URL is baked into the bundle.
 */

import type {
  Advisory,
  AdvisorFeed,
  Alert,
  AnalyticsOverview,
  CropCatalogue,
  Device,
  DriftReport,
  EventRecord,
  FieldCreate,
  FieldRecord,
  Recommendation,
  RegistryOptions,
  SystemHealth,
  Telemetry,
  ZoneSummary,
} from '../types'

const BASE = import.meta.env.VITE_API_BASE ?? '/api'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

/**
 * Pull the human-readable part out of a FastAPI error body.
 *
 * FastAPI has two error shapes and only one of them is a string. A raised
 * `HTTPException` gives `{"detail": "..."}`, which is the message an operator
 * should read. A request that fails *validation* gives
 * `{"detail": [{"loc": [...], "msg": "..."}]}`, and collapsing that to
 * "422 Unprocessable Entity" throws away the only part anyone can act on - which
 * field was wrong and what the accepted values are.
 */
function readableDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string' && detail.length > 0) return detail
  if (Array.isArray(detail)) {
    const messages = detail
      .map((entry) => {
        if (typeof entry !== 'object' || entry === null) return null
        const item = entry as { loc?: unknown[]; msg?: string }
        const field = Array.isArray(item.loc) ? item.loc.filter((p) => p !== 'body').join('.') : ''
        const msg = (item.msg ?? '').replace(/^Value error,\s*/, '')
        if (!msg) return null
        return field ? `${field}: ${msg}` : msg
      })
      .filter((line): line is string => Boolean(line))
    if (messages.length > 0) return messages.join('; ')
  }
  return fallback
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${BASE}${path}`, {
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
      ...init,
    })
  } catch (cause) {
    // A transport failure is the common case when the backend is not running,
    // and the operator needs to be told which it is.
    throw new ApiError(
      'Cannot reach the controller backend. Confirm it is running on port 8000.',
      0,
      cause,
    )
  }

  if (!response.ok) {
    let detail: unknown
    try {
      detail = await response.json()
    } catch {
      detail = await response.text()
    }
    const body = typeof detail === 'object' && detail !== null ? (detail as { detail?: unknown }).detail : detail
    throw new ApiError(
      readableDetail(body, `${response.status} ${response.statusText}`),
      response.status,
      detail,
    )
  }

  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

function query(params: Record<string, string | number | boolean | undefined>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== '') search.set(key, String(value))
  }
  const text = search.toString()
  return text ? `?${text}` : ''
}

export const api = {
  health: () => request<SystemHealth>('/health'),

  devices: () => request<{ devices: Device[] }>('/devices'),
  device: (id: string) => request<Device>(`/devices/${id}`),

  zones: () => request<{ zones: ZoneSummary[] }>('/zones'),

  /**
   * Legal values for every reference field on a sowing. The engine's own tables,
   * so the form can never offer a name the registry would refuse.
   */
  options: () => request<RegistryOptions>('/system/options'),

  /**
   * The crop catalogue with the parameters behind each crop.
   *
   * Deliberately not behind the planning dependency: the form needs the crop list
   * while the weather corpus is still loading, and a 503 on the screen that would
   * explain the problem is the wrong answer.
   */
  crops: () => request<CropCatalogue>('/system/crops'),

  /** Register a sowing. Answers with the stored field and the plan it produces. */
  createField: (body: FieldCreate, replace = false) =>
    request<FieldRecord>(`/fields${query({ replace })}`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  field: (id: string) => request<FieldRecord>(`/fields/${id}`),

  deleteField: (id: string) => request<void>(`/fields/${id}`, { method: 'DELETE' }),

  /**
   * The water advisor: every field's irrigation answer with its reasoning,
   * live sensor state and seven-day schedule, ranked by urgency.
   */
  advisor: () => request<AdvisorFeed>('/advisor'),
  advisorField: (id: string) => request<Advisory>(`/advisor/${id}`),

  latest: (deviceId?: string) =>
    request<{ readings: Telemetry[]; recommendations: Recommendation[] }>(
      `/telemetry/latest${query({ device_id: deviceId })}`,
    ),

  history: (deviceId: string, hours = 24, limit = 500) =>
    request<{ readings: Telemetry[]; recommendations: Recommendation[] }>(
      `/telemetry/history${query({ device_id: deviceId, hours, limit })}`,
    ),

  analytics: (hours = 24) => request<AnalyticsOverview>(`/analytics/overview${query({ hours })}`),

  /**
   * Population stability of live readings against the training distribution.
   * A model keeps answering confidently after the field drifts, and nothing in
   * the decision path can notice, so this is what makes that question
   * answerable.
   */
  drift: (hours = 168) => request<DriftReport>(`/models/drift${query({ hours })}`),

  alerts: (openOnly = false, limit = 100) =>
    request<{ alerts: Alert[] }>(`/alerts${query({ open_only: openOnly, limit })}`),

  acknowledgeAlert: (id: number) => request<Alert>(`/alerts/${id}/acknowledge`, { method: 'POST' }),

  events: (limit = 100) => request<{ events: EventRecord[] }>(`/events${query({ limit })}`),

  /** Manual override: open or close a valve regardless of the model. */
  valve: (deviceId: string, open: boolean, durationSeconds?: number) =>
    request<{ accepted: boolean; deviceId: string; valveOpen: boolean }>(
      `/devices/${deviceId}/valve`,
      {
        method: 'POST',
        body: JSON.stringify({ open, duration_seconds: durationSeconds }),
      },
    ),

  /** Ask the backend to re-run inference for a device, using its last reading. */
  recompute: (deviceId: string) =>
    request<Recommendation>(`/devices/${deviceId}/recommendation`, { method: 'POST' }),

  /** Out-of-band prediction for arbitrary sensor values, for what-if analysis. */
  predict: (payload: Record<string, unknown>) =>
    request<Recommendation>('/predict', { method: 'POST', body: JSON.stringify(payload) }),
}
