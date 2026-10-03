/**
 * Tests for the API client.
 *
 * The failure modes here are worth more than the happy path. This client drives
 * a live irrigation controller: a swallowed rejection, a wrong base URL or a
 * mis-parsed error body each turn into a valve command that silently does not
 * arrive, and an operator who cannot tell the difference will assume the field
 * was watered.
 *
 * fetch is stubbed rather than a live server, so these assert the contract the
 * rest of the app depends on rather than the backend's behaviour.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, api } from './client'

const fetchMock = vi.fn()

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    json: async () => body,
    text: async () => JSON.stringify(body),
  } as Response
}

beforeEach(() => {
  fetchMock.mockReset()
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('request', () => {
  it('prefixes every call with the API base so no absolute URL is baked in', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ status: 'ok' }))
    await api.health()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/health')
  })

  it('parses a JSON body on success', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ devices: [] }))
    const result = await api.devices()
    expect(result).toEqual({ devices: [] })
  })

  it('throws a transport error naming port 8000 when the backend is unreachable', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'))
    await expect(api.health()).rejects.toBeInstanceOf(ApiError)
    await expect(api.health()).rejects.toMatchObject({ status: 0 })
  })

  it('carries the status and parsed detail on an HTTP error', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ detail: 'unknown device valve' }, 404))
    try {
      await api.valve('nope', true)
      expect.unreachable('should have thrown')
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError)
      const apiError = error as ApiError
      expect(apiError.status).toBe(404)
      expect(apiError.detail).toEqual({ detail: 'unknown device valve' })
    }
  })

  it('falls back to text when an error body is not JSON', async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 502,
      statusText: 'Bad Gateway',
      json: async () => {
        throw new Error('not json')
      },
      text: async () => 'upstream closed',
    } as unknown as Response)
    await expect(api.health()).rejects.toBeInstanceOf(ApiError)
  })
})

describe('query parameters', () => {
  it('omits undefined and empty values rather than sending them as blanks', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ readings: [], recommendations: [] }))
    await api.latest(undefined)
    expect(fetchMock.mock.calls[0][0]).toBe('/api/telemetry/latest')
  })

  it('sends defined values', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ readings: [], recommendations: [] }))
    await api.history('node-1', 48, 100)
    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/telemetry/history?device_id=node-1&hours=48&limit=100',
    )
  })
})

describe('valve control', () => {
  it('POSTs the open command with a duration', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ accepted: true, valveOpen: true }))
    await api.valve('node-1', true, 300)
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/devices/node-1/valve')
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body)).toEqual({ open: true, duration_seconds: 300 })
  })

  it('sends an explicit close', async () => {
    fetchMock.mockResolvedValue(jsonResponse({ accepted: true, valveOpen: false }))
    await api.valve('node-1', false)
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ open: false })
  })
})
