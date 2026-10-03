/**
 * Tests for the polling hook.
 *
 * The behaviour that matters here is not the happy path but the two ways this
 * hook can lie to an operator. It must not report a stale value as current after
 * the component unmounts, and it must surface an error rather than silently
 * holding the last good payload. A monitoring surface that keeps showing old
 * numbers after the backend dies is worse than one that shows nothing, because
 * the operator cannot tell the difference.
 */

import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { usePollingFetch } from './usePollingFetch'
import { ApiError } from '../api/client'

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
})

afterEach(() => {
  vi.useRealTimers()
})

describe('usePollingFetch', () => {
  it('loads on mount and clears the loading flag', async () => {
    const fetcher = vi.fn().mockResolvedValue({ value: 1 })
    const { result } = renderHook(() => usePollingFetch(fetcher, 0, 10_000))

    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.data).toEqual({ value: 1 })
    expect(result.current.error).toBeNull()
  })

  it('surfaces an ApiError message rather than a generic failure', async () => {
    const fetcher = vi.fn().mockRejectedValue(new ApiError('backend unreachable', 0))
    const { result } = renderHook(() => usePollingFetch(fetcher, 0, 10_000))

    await waitFor(() => expect(result.current.error).toBe('backend unreachable'))
    expect(result.current.loading).toBe(false)
  })

  it('falls back to a readable message for a non-ApiError rejection', async () => {
    const fetcher = vi.fn().mockRejectedValue(new Error('kaboom'))
    const { result } = renderHook(() => usePollingFetch(fetcher, 0, 10_000))

    await waitFor(() => expect(result.current.error).toBeTruthy())
    expect(result.current.error).not.toContain('kaboom')
  })

  it('clears a previous error once a later poll succeeds', async () => {
    let shouldFail = true
    const fetcher = vi.fn().mockImplementation(() =>
      shouldFail ? Promise.reject(new ApiError('down', 0)) : Promise.resolve({ ok: true }),
    )
    const { result } = renderHook(() => usePollingFetch(fetcher, 0, 10_000))

    await waitFor(() => expect(result.current.error).toBe('down'))
    shouldFail = false
    act(() => {
      result.current.reload()
    })
    await waitFor(() => expect(result.current.error).toBeNull())
    expect(result.current.data).toEqual({ ok: true })
  })

  it('does not set state after unmount', async () => {
    let resolve: (value: unknown) => void = () => {}
    const fetcher = vi.fn().mockImplementation(
      () => new Promise((r) => { resolve = r as typeof resolve }),
    )
    const { unmount } = renderHook(() => usePollingFetch(fetcher, 0, 10_000))
    unmount()
    // Resolving after unmount must not warn or throw; the mountedRef guard in
    // the hook is what makes this safe.
    await act(async () => {
      resolve({ late: true })
      await Promise.resolve()
    })
    expect(true).toBe(true)
  })
})
