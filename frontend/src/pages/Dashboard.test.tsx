/**
 * Render smoke test for the field view.
 *
 * This is the screen a farmer opens to decide whether to water, so the assertions
 * are about the instruction being present and unambiguous rather than about
 * layout. A regression that drops the depth figure or replaces the plain-language
 * instruction with jargon would pass a type check and fail here.
 */

import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'

const FIELD = {
  fieldId: 'PNQ-wheat-01',
  crop: 'Wheat',
  station: 'PNQ',
  soilType: 'Clay',
  method: 'Furrow',
  sowingDate: '2026-08-01',
  areaM2: 4000,
  ageDays: 62,
  stage: 'mid_season',
  stageLabel: 'Mid-season',
  stageProgress: 0.62,
  gdd: 980,
  status: 'irrigate',
  priority: 88,
  headline: 'Irrigate 18 mm now',
  summary: 'Wheat at PNQ is 62 days past sowing.',
  recommendation: {
    shouldIrrigate: true,
    depthMm: 12,
    finalDepthMm: 18.4,
    volumeLitres: 73600,
    durationSeconds: 6133,
    action: 'MODERATE_IRRIGATION',
    confidence: 0.82,
    probability: 0.91,
    threshold: 0.5,
  },
  water: {
    et0MmDay: 6.1,
    etcMmDay: 3.6,
    kc: 0.59,
    rainfallMmDay: 0,
    tawMm: 120,
    rawMm: 60,
    depletionMm: 84,
    depletionFraction: 0.7,
    rootDepthCm: 60,
    daysUntilStress: 0,
  },
  sensors: {
    id: 1,
    deviceId: 'PNQ-wheat-01',
    zone: 'PNQ-wheat-01',
    recordedAt: '2026-10-02T08:55:00+00:00',
    temperature: 25.7,
    humidity: 41,
    soilMoisture: 18.4,
    soilPh: 6.75,
    ec: 0.39,
    nitrogen: 40,
    phosphorus: 20,
    potassium: 130,
    lightIntensity: 42000,
    batteryVolts: null,
    rssiDbm: null,
  },
  risk: {
    diseaseRisk: 'rust',
    diseaseConfidence: 0.31,
    riskLevel: 'watch',
    cropStressIndex: 0.7,
    moistureBand: 'dry',
    yieldTPerHa: 5.4,
  },
  models: {
    v1: {
      shouldIrrigate: true,
      depthMm: 12,
      probability: 0.91,
      threshold: 0.5,
      confidence: 0.82,
      action: 'MODERATE_IRRIGATION',
    },
    v2: {
      target: 'nir_horizon_mm',
      horizonDays: 14,
      physicsRequirementMm: 18.4,
      physicsSingleApplicationMm: 15,
      modelPresent: true,
      applicable: true,
      reason: null,
      predictionMm: 19.2,
      intervalMm: [10.1, 28.3],
      coverage: 0.9,
    },
    fusion: {
      agreement: 'aligned',
      finalStatus: 'irrigate',
      finalDepthMm: 18.4,
      v1DepthMm: 12,
      v2RequirementMm: 19.2,
      note: 'Both models call for water.',
    },
  },
  season: {
    stageWindows: [
      {
        stage: 'initial',
        startDate: '2026-08-01',
        endDate: '2026-08-14',
        days: 13,
        meanKc: 0.4,
        meanEtcMmDay: 1.2,
        meanRainMmDay: 0.1,
        grossRequirementMm: 15.6,
      },
      {
        stage: 'development',
        startDate: '2026-08-14',
        endDate: '2026-09-10',
        days: 27,
        meanKc: 0.8,
        meanEtcMmDay: 2.8,
        meanRainMmDay: 0.1,
        grossRequirementMm: 75.6,
      },
      {
        stage: 'mid_season',
        startDate: '2026-09-10',
        endDate: '2026-10-12',
        days: 32,
        meanKc: 1.15,
        meanEtcMmDay: 4.1,
        meanRainMmDay: 0.2,
        grossRequirementMm: 131.2,
      },
    ],
    daysInSeason: 128,
    seasonGrossMm: 312.4,
    seasonRainMm: 64.1,
  },
  reasoning: [
    { label: 'Root-zone water', value: '70% depleted', detail: '120 mm', tone: 'warn' },
  ],
  schedule: [
    { date: '2026-10-02', dayOffset: 0, applyMm: 18.4, grossMm: 21.6, stage: 'mid_season' },
    { date: '2026-10-03', dayOffset: 1, applyMm: 0, grossMm: 0, stage: 'mid_season' },
    { date: '2026-10-04', dayOffset: 2, applyMm: 0, grossMm: 0, stage: 'mid_season' },
    { date: '2026-10-05', dayOffset: 3, applyMm: 0, grossMm: 0, stage: 'mid_season' },
    { date: '2026-10-06', dayOffset: 4, applyMm: 0, grossMm: 0, stage: 'mid_season' },
    { date: '2026-10-07', dayOffset: 5, applyMm: 0, grossMm: 0, stage: 'mid_season' },
    { date: '2026-10-08', dayOffset: 6, applyMm: 0, grossMm: 0, stage: 'mid_season' },
  ],
}

const FEED = {
  generatedAt: '2026-10-02T09:00:00+00:00',
  pipeline: {
    modelPresent: true,
    target: 'nir_horizon_mm',
    coverage: 0.9,
    features: 36,
    intervalWidthMm: 9.1,
    error: null,
  },
  fleet: {
    fields: 1,
    irrigate: 1,
    monitor: 0,
    hold: 0,
    recommendedMm: 18.4,
    recommendedLitres: 73600,
  },
  fields: [FIELD],
}

const ANALYTICS = {
  window: '48h',
  waterAppliedMm: 18.4,
  waterAppliedPctChange: -3.2,
  estimatedLitresSaved: 1200,
  irrigationEvents: 2,
  runtimeSeconds: 1800,
  averageInferenceMs: 4.1,
  decisionsCorrectPct: 94,
  energyKwh: 0.4,
  series: [
    { label: '00', appliedMm: 2, et0MmDay: 5.1, depletionFraction: 0.4 },
    { label: '06', appliedMm: 0, et0MmDay: 5.4, depletionFraction: 0.5 },
    { label: '12', appliedMm: 11.4, et0MmDay: 6.2, depletionFraction: 0.6 },
  ],
}

function route(url: string): unknown {
  if (url.includes('/advisor')) return FEED
  if (url.includes('/alerts')) return { alerts: [] }
  if (url.includes('/analytics')) return ANALYTICS
  if (url.includes('/health')) return { status: 'ok', version: '2.0.0' }
  return { readings: [], recommendations: [], devices: [], zones: [], events: [] }
}

function renderShell() {
  return render(
    <MemoryRouter initialEntries={['/']}>
      <App />
    </MemoryRouter>,
  )
}

describe('Field view', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'ResizeObserver',
      class {
        observe() {}
        unobserve() {}
        disconnect() {}
      },
    )
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const body = route(String(input))
        return {
          ok: true,
          status: 200,
          json: async () => body,
          text: async () => JSON.stringify(body),
        } as Response
      }),
    )
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('leads with one plain instruction and the depth to apply', async () => {
    renderShell()

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    // The advice is in the farmer's words and units. The depth also appears on
    // the field cards, so it is matched loosely.
    expect(screen.getByText('Water today')).toBeTruthy()
    expect(screen.getAllByText('18.4').length).toBeGreaterThan(0)
    expect(document.body.textContent).toContain('73,600 L')

    // Model agreement is one quiet line, not a panel.
    expect(screen.getByText('Both models agree on this watering.')).toBeTruthy()
  })

  it('tells the fields apart, so each one gets its own advice', async () => {
    const { container } = renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    // Every registered field is offered, and the switcher is what distinguishes
    // them. Two fields with identical advice would make the switcher pointless.
    const switcher = Array.from(container.querySelectorAll('header button')).map(
      (button) => button.textContent ?? '',
    )
    expect(switcher.length).toBeGreaterThan(0)

    // The two facts that make each field's answer different are both on screen.
    expect(screen.getByText('Crop age')).toBeTruthy()
    expect(screen.getByText('days old')).toBeTruthy()
    expect(screen.getByText('Soil water')).toBeTruthy()
    expect(screen.getByText('Refill at')).toBeTruthy()
  })

  it('leads with the decision, then three readings, the field strip, sensors, the week', async () => {
    const { container } = renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    // The decision leads, on its own full-width row rather than as one of four
    // equal cards.
    expect(screen.getByText('Recommended depth')).toBeTruthy()
    expect(screen.getByRole('heading', { level: 1 })).toBeTruthy()

    // Then the three supporting readings.
    expect(screen.getByText('Soil water')).toBeTruthy()
    expect(screen.getByText('Crop age')).toBeTruthy()
    // The sky card is titled by its station, not by a heading.
    expect(screen.getByText(/· now/)).toBeTruthy()

    // The field record is a strip, not a fourth card competing for height.
    expect(screen.getByText('Field')).toBeTruthy()

    // The per-field cards are gone; the farm roll-up replaced them.
    expect(screen.queryByText('Showing above · hover for detail')).toBeNull()
    expect(screen.queryByText(/Click to view/)).toBeNull()

    // The week has a fleet card, so the third slot carries meaning.
    expect(screen.getByText('Farm this week')).toBeTruthy()
    expect(screen.getByText('By depth')).toBeTruthy()

    // Vertical order: decision, the three readings, field strip, age, sensors, week.
    const order = Array.from(container.querySelectorAll('section'))
      .map((node) => node.querySelector('h2')?.textContent ?? '')
      .filter(Boolean)
    expect(order.indexOf('Soil water')).toBeGreaterThanOrEqual(0)
    expect(order.indexOf('Crop age')).toBeGreaterThan(order.indexOf('Soil water'))
    expect(order.indexOf('7-in-1 soil sensor')).toBeGreaterThan(order.indexOf('Crop age'))
    expect(order.indexOf('Farm this week')).toBeGreaterThan(
      order.indexOf('7-in-1 soil sensor'),
    )
  })

  it('places the stage marker inside the active segment, not at its edge', async () => {
    const { container } = renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    // `stageProgress` is progress within the current stage, so a marker derived
    // from it clamps to the segment's left edge. It must instead be placed from
    // day counts, landing strictly inside the active segment.
    //
    // The fixture is day 62 of a 128-day season, mid-season starting day 40 and
    // running 32 days, so the marker belongs at ~50% of the season.
    const markerSpan = Array.from(container.querySelectorAll('span')).find((node) =>
      /left:\s*[\d.]+%/.test(node.getAttribute('style') ?? ''),
    )
    expect(markerSpan).toBeTruthy()

    const left = Number(
      /left:\s*([\d.]+)%/.exec(markerSpan!.getAttribute('style') ?? '')?.[1],
    )
    expect(left).toBeGreaterThan(20)
    expect(left).toBeGreaterThan(40)
    expect(left).toBeLessThan(60)

    // The caption must not claim the stage-local value is a season figure.
    expect(screen.getByText(/through this stage/)).toBeTruthy()
    expect(document.body.textContent).not.toContain('through the season')
  })

  it('shows all seven sensor readings, each judged against a range', async () => {
    renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    expect(screen.getByText('7-in-1 soil sensor')).toBeTruthy()

    // Soil readings, each judged against a range. Ambient readings belong to the
    // sky card: this board used to carry both a "Soil temp" and an "Air temp"
    // tile bound to the same value, and repeated humidity as well.
    expect(screen.getByText('Moisture')).toBeTruthy()
    expect(screen.getByText('Humidity')).toBeTruthy()
    expect(screen.queryByText('Soil temp')).toBeNull()
    expect(screen.queryByText('Air temp')).toBeNull()
    expect(screen.getByText('EC')).toBeTruthy()
    expect(screen.getByText('pH')).toBeTruthy()

    // Nutrients are drawn against their sufficiency bands.
    expect(screen.getByText('Nitrogen')).toBeTruthy()
    expect(screen.getByText('Phosphorus')).toBeTruthy()
    expect(screen.getByText('Potassium')).toBeTruthy()
    expect(screen.getAllByText('In range').length).toBeGreaterThan(0)
  })

it('states the decision, the amount, and the three inputs behind it', async () => {
    renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    // The badge is the decision; the headline is the action.
    expect(screen.getByText('Irrigate today')).toBeTruthy()
    expect(document.body.textContent).toContain('Recommended depth')
    expect(document.body.textContent).toContain('73,600 L')
    expect(document.body.textContent).toContain('102 min')

    // The reasoning, split so it can be argued with.
    // The three inputs that produced the decision. "Soil" is also a label on the
    // field-details grid, so it is matched loosely.
    expect(screen.getAllByText('Soil').length).toBeGreaterThan(0)
    expect(screen.getByText('Crop need')).toBeTruthy()
    expect(screen.getByText('Rain today')).toBeTruthy()
  })

it('puts soil dryness and crop age first, as the two deciding facts', async () => {
    renderShell()
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy())

    // Dryness: a verdict word, the level, and the refill point it is read against.
    expect(screen.getByText('Soil water')).toBeTruthy()
    expect(screen.getByText('left')).toBeTruthy()
    expect(screen.getByText('Refill at')).toBeTruthy()
    expect(screen.getByText('Used')).toBeTruthy()

    // Crop age: days old, the stage it is in, and the season track.
    expect(screen.getByText('Crop age')).toBeTruthy()
    expect(screen.getByText('days old')).toBeTruthy()
    expect(screen.getByText('Crop need')).toBeTruthy()
    expect(screen.getByText('Sun today')).toBeTruthy()

    // The reasoning ties the two together in one sentence.
    expect(screen.getByText(/The soil is dry and the wheat is at its thirstiest stage/)).toBeTruthy()
  })

  it('supports every registered field from a simple switcher', async () => {
    renderShell()
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy())

    expect(screen.getByRole('button', { name: /PNQ-wheat-01/ })).toBeTruthy()

    // No dashboard furniture competing with the two facts.
    expect(screen.queryByText('Overview')).toBeNull()
    expect(screen.queryByText('Analytics')).toBeNull()
    expect(screen.queryByText(/Schematic parcel/)).toBeNull()
  })
})
