/**
 * Live-state feed for the shell.
 *
 * V1 pushed telemetry over a WebSocket; V2 is a planner rather than a
 * controller and has no live stream, so there is nothing to push. This hook
 * keeps the same interface (`onTelemetry`, `onRecommendation`, `state`) but
 * polls the engine's current plan-derived state on a timer instead of holding a
 * socket, and it feeds only *unseen* items upward so the shell's capped list is
 * a real sequence of states rather than the same window repeated every poll.
 *
 * Connection state is honest in the same way the socket was: `open` means "the
 * engine is answering", `closed` means "it just stopped", and the poll
 * retries so the header recovers without a reload.
 */

import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { Recommendation, Telemetry } from '../types'

export type SocketState = 'connecting' | 'open' | 'closed'

export interface LiveFrame {
  type: 'telemetry' | 'recommendation' | 'valve' | 'alert' | 'ping'
  deviceId?: string
  zone?: string
  payload?: unknown
}

interface Options {
  onTelemetry?: (reading: Telemetry) => void
  onRecommendation?: (recommendation: Recommendation) => void
  onFrame?: (frame: LiveFrame) => void
}

const POLL_MS = 10_000

export function useLiveTelemetry({ onTelemetry, onRecommendation }: Options) {
  const [state, setState] = useState<SocketState>('connecting')

  // Callbacks live in a ref so that changing them does not tear down the poll
  // loop, which would restart on every parent re-render.
  const handlers = useRef({ onTelemetry, onRecommendation })
  handlers.current = { onTelemetry, onRecommendation }

  const seenReadings = useRef(new Set<number>())
  const seenRecommendations = useRef(new Set<string>())

  useEffect(() => {
    let stopped = false
    let timer: number | null = null

    const poll = async () => {
      try {
        const latest = await api.latest()
        if (stopped) return
        for (const reading of latest.readings) {
          if (seenReadings.current.has(reading.id)) continue
          seenReadings.current.add(reading.id)
          handlers.current.onTelemetry?.(reading)
        }
        for (const recommendation of latest.recommendations) {
          const key = `${recommendation.deviceId}:${recommendation.recordedAt}`
          if (seenRecommendations.current.has(key)) continue
          seenRecommendations.current.add(key)
          handlers.current.onRecommendation?.(recommendation)
        }
        setState('open')
      } catch {
        if (!stopped) setState('closed')
      }
    }

    void poll()
    timer = window.setInterval(() => void poll(), POLL_MS)
    return () => {
      stopped = true
      if (timer !== null) window.clearInterval(timer)
    }
  }, [])

  return { state }
}