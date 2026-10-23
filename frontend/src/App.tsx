/**
 * Application shell.
 *
 * The field view is the product, so the shell is deliberately thin: a status
 * bar and two routes. V3 removed the console pages entirely — the app is for
 * farmers, not developers, so there is no Farmer/Console switch to hide the
 * analytical pages behind, and the only page beyond the field is the
 * crop-photo check.
 *
 * The live stream is opened once here and pushed down through context so the
 * map and the detail panel refresh together.
 */

import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import { Route, Routes } from 'react-router-dom'
import { useLiveTelemetry } from './hooks/useLiveTelemetry'
import type { Recommendation, Telemetry } from './types'
import Dashboard from './pages/Dashboard'
import CropHealth from './pages/CropHealth'

interface LiveContextValue {
  connected: boolean
  telemetry: Telemetry[]
  recommendations: Recommendation[]
  bump: number
  refresh: () => void
}

const LiveContext = createContext<LiveContextValue | null>(null)

export function useLive(): LiveContextValue {
  const value = useContext(LiveContext)
  if (!value) throw new Error('useLive must be used inside the application shell')
  return value
}

export default function App() {
  const [telemetry, setTelemetry] = useState<Telemetry[]>([])
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [bump, setBump] = useState(0)

  const onTelemetry = useCallback((reading: Telemetry) => {
    setTelemetry((previous) => [reading, ...previous].slice(0, 300))
    setBump((value) => value + 1)
  }, [])

  const onRecommendation = useCallback((recommendation: Recommendation) => {
    setRecommendations((previous) => [recommendation, ...previous].slice(0, 300))
    setBump((value) => value + 1)
  }, [])

  const { state } = useLiveTelemetry({ onTelemetry, onRecommendation })
  const refresh = useCallback(() => setBump((value) => value + 1), [])

  const live = useMemo<LiveContextValue>(
    () => ({ connected: state === 'open', telemetry, recommendations, bump, refresh }),
    [state, telemetry, recommendations, bump, refresh],
  )

  return (
    <LiveContext.Provider value={live}>
      <main className="min-h-screen lg:h-screen lg:min-h-0">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/crop-health" element={<CropHealth />} />
          <Route path="*" element={<Dashboard />} />
        </Routes>
      </main>
    </LiveContext.Provider>
  )
}
