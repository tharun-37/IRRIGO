/**
 * Application shell.
 *
 * The field view is the product, so the shell is deliberately thin: a status
 * bar and the route. There is no navigation rail and no list of destinations
 * competing with the map for attention. The analytical pages still exist behind
 * their routes for whoever is checking the model's numbers rather than acting on
 * them, and the field view links to that one page.
 *
 * The live stream is opened once here and pushed down through context so the
 * map and the detail panel refresh together.
 */

import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import { Route, Routes } from 'react-router-dom'
import { useLiveTelemetry } from './hooks/useLiveTelemetry'
import type { Recommendation, Telemetry } from './types'
import Dashboard from './pages/Dashboard'
import Overview from './pages/Overview'
import Zones from './pages/Zones'
import Analytics from './pages/Analytics'
import ModelHealth from './pages/ModelHealth'
import Alerts from './pages/Alerts'
import Advisor from './pages/Advisor'

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
          <Route path="/overview" element={<Overview />} />
          <Route path="/zones" element={<Zones />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/model" element={<ModelHealth />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/advisor" element={<Advisor />} />
          <Route path="*" element={<Dashboard />} />
        </Routes>
      </main>
    </LiveContext.Provider>
  )
}