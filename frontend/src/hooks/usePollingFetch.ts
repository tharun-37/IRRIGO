/**
 * Shared data-fetching hook.
 *
 * V2 has no WebSocket and nothing that changes by itself, so `bump` is kept as a
 * compatibility seam V1 left behind: a shell refresh and a page reload both want
 * the same data, and this is the single path. Pages that need fresher data pass
 * a shorter interval; they do not re-render on a fake stream.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from '../api/client'

const MIN_REFETCH_MS = 3_000

export function usePollingFetch<T>(
  fetcher: () => Promise<T>,
  _bump = 0,
  intervalMs = 10_000,
): { data: T | null; error: string | null; loading: boolean; reload: () => void } {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const lastFetchRef = useRef(0)
  const mountedRef = useRef(true)

  const load = useCallback(async () => {
    lastFetchRef.current = Date.now()
    try {
      const result = await fetcher()
      if (!mountedRef.current) return
      setData(result)
      setError(null)
    } catch (cause) {
      if (!mountedRef.current) return
      setError(cause instanceof ApiError ? cause.message : 'Unexpected error while loading data.')
    } finally {
      if (mountedRef.current) setLoading(false)
    }
  }, [fetcher])

  useEffect(() => {
    mountedRef.current = true
    void load()
    return () => {
      mountedRef.current = false
    }
  }, [load])

  useEffect(() => {
    if (Date.now() - lastFetchRef.current < MIN_REFETCH_MS) return
    void load()
  }, [_bump, load])

  useEffect(() => {
    const timer = window.setInterval(() => void load(), intervalMs)
    return () => window.clearInterval(timer)
  }, [load, intervalMs])

  return { data, error, loading, reload: () => void load() }
}
