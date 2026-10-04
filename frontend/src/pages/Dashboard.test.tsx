/**
 * Render smoke test for the field view.
 *
 * This is the screen a farmer opens to decide whether to water, so the assertions
 * are about the instruction being present and unambiguous rather than about
 * layout. A regression that drops the depth figure or replaces the plain-language
 * instruction with jargon would pass a type check and fail here.
 */

import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
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
  if (url.includes('/system/crops')) return { crops: [CROP_OPTION] }
  if (url.includes('/system/options'))
    return {
      stations: ['LKO', 'PNQ'],
      crops: ['Barley', 'Cotton'],
      soils: ['Loam', 'Silt_Loam'],
      methods: ['Basin', 'Drip', 'Paddy', 'Sprinkler'],
    }
  if (url.includes('/advisor')) return FEED
  if (url.includes('/alerts')) return { alerts: [] }
  if (url.includes('/analytics')) return ANALYTICS
  if (url.includes('/health')) return { status: 'ok', version: '2.0.0' }
  return { readings: [], recommendations: [], devices: [], zones: [], events: [] }
}

/** One catalogue entry, enough for the form to render a crop card. */
const CROP_OPTION = {
  name: 'Barley',
  kcInitial: 0.35,
  kcMid: 1.15,
  kcEnd: 0.4,
  rootDepthMaxM: 1.2,
  depletionFractionP: 0.55,
  optimalPh: [6, 7.5],
  gddBaseTempC: 5,
  gddStageEnds: [350, 850, 1350, 1650],
  seasonGdd: 1650,
  lengthStageDays: [20, 30, 90, 25],
  seasonDays: 165,
  ky: 1.1,
  yieldPotentialTHa: 4.2,
  variants: [],
  suitableMethods: ['Sprinkler', 'Basin'],
  warnings: {},
}

function renderShell() {  return render(
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
    // By heading role: "Crop age" is also a fact in the field details block, so
    // the text alone no longer picks out this card.
    expect(screen.getByRole('heading', { name: 'Crop age' })).toBeTruthy()
    expect(screen.getByText('days old')).toBeTruthy()
    expect(screen.getByText('Soil water')).toBeTruthy()
    expect(screen.getByText('Refill at')).toBeTruthy()
  })

  it('leads with the field details, then the decision, three readings, sensors, the week', async () => {
    const { container } = renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    // The field's own identity comes first, so "which field is this, and what is
    // it" is answered before "what should I do". Scoped to that section, because
    // "Crop age" and "Stage" are also readings further down the page.
    const details = screen.getByRole('heading', { name: 'Field details' }).closest('section')
    expect(details).toBeTruthy()
    for (const fact of [
      'Field',
      'Crop',
      'Station',
      'Soil',
      'Irrigation',
      'Area',
      'Sown',
      'Stage',
      'Crop age',
      'Roots',
    ]) {
      expect(within(details as HTMLElement).getByText(fact)).toBeTruthy()
    }
    // And the numbers behind the recommendation, not just the field's own facts.
    for (const outcome of ['Disease risk', 'Crop stress index', 'Yield outlook']) {
      expect(within(details as HTMLElement).getByText(outcome)).toBeTruthy()
    }

    // The decision leads what follows. It shares its row with the farm total,
    // which is the same question at two scales.
    expect(screen.getByText('Recommended depth')).toBeTruthy()
    expect(screen.getByRole('heading', { level: 1 })).toBeTruthy()
    expect(screen.getByText('Farm this week')).toBeTruthy()
    expect(screen.getByText('By depth')).toBeTruthy()

    // The decision and the farm total are neighbours, not a screen apart.
    const topRow = screen.getByText('Recommended depth').closest('section')?.parentElement
    expect(topRow).toBe(screen.getByText('Farm this week').closest('section')?.parentElement)

    // Then the three supporting readings.
    expect(screen.getByText('Soil water')).toBeTruthy()
    // By heading role: "Crop age" is also a fact in the field details block, so
    // the text alone no longer picks out this card.
    expect(screen.getByRole('heading', { name: 'Crop age' })).toBeTruthy()
    // The sky card is titled by its station, not by a heading.
    expect(screen.getByText(/· now/)).toBeTruthy()

    // The per-field cards are gone; the farm roll-up replaced them.
    expect(screen.queryByText('Showing above · hover for detail')).toBeNull()
    expect(screen.queryByText(/Click to view/)).toBeNull()

    // The two charts are gone as well. They were the only reason this page pulled
    // the 168-hour analytics series, so that request went with them.
    expect(screen.queryByText('Next seven days')).toBeNull()
    expect(screen.queryByText('Water in the soil')).toBeNull()

    // Vertical order: field details, decision and farm total, the three
    // readings, then the sensors.
    const order = Array.from(container.querySelectorAll('section'))
      .map((node) => node.querySelector('h2')?.textContent ?? '')
      .filter(Boolean)
    expect(order.indexOf('Field details')).toBeGreaterThanOrEqual(0)
    expect(order.indexOf('Farm this week')).toBeGreaterThan(order.indexOf('Field details'))
    expect(order.indexOf('Soil water')).toBeGreaterThan(order.indexOf('Farm this week'))
    expect(order.indexOf('Crop age')).toBeGreaterThan(order.indexOf('Soil water'))
    expect(order.indexOf('7-in-1 soil sensor')).toBeGreaterThan(order.indexOf('Crop age'))
  })

  it('gives every card in a row the same height, filling the difference', async () => {
    renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    // Both rows stretch, so cards in a row end on one line instead of leaving a
    // gap under the shorter ones.
    const decisionRow = screen.getByText('Farm this week').closest('section')?.parentElement
    expect(decisionRow?.className).not.toContain('items-start')

    const readingsRow = screen.getByText('Soil water').closest('section')?.parentElement
    expect(readingsRow?.className).not.toContain('items-start')

    // And the height a taller neighbour imposes is handed to a bordered surface
    // rather than left as blank card. One growing surface per card in the row.
    // By heading role: "Crop age" is also a fact in the field details block.
    for (const title of ['Soil water', 'Crop age', 'Farm this week']) {
      const card = screen.getByRole('heading', { name: title }).closest('section')
      expect(card?.querySelector('.flex-1')).toBeTruthy()
    }
    // The sky card is not a `Card`, so its humidity box is the surface that
    // grows there.
    const sky = screen.getByText(/· now/).closest('section')
    expect(sky?.querySelector('dl.flex-1')).toBeTruthy()
  })

  it('fills every row of the field details, with no dead cells', async () => {
    renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    // A fixed column count always leaves its last row short when the fact count
    // is not a multiple of it. Growing cells in a wrapping row fill the width
    // whatever the count works out as.
    const details = screen.getByRole('heading', { name: 'Field details' }).closest('section')
    const lists = Array.from(details?.querySelectorAll('dl') ?? [])
    expect(lists).toHaveLength(2)
    for (const list of lists) {
      expect(list.className).toContain('flex-wrap')
      // Every cell grows, so no row ends with an unfilled slot.
      expect(list.querySelector(':scope > div:not([class*="col-span"])')).toBeTruthy()
      for (const cell of Array.from(list.children)) {
        expect(cell.className).toContain('flex-1')
      }
    }
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
    // By heading role: "Crop age" is also a fact in the field details block, so
    // the text alone no longer picks out this card.
    expect(screen.getByRole('heading', { name: 'Crop age' })).toBeTruthy()
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

  it('offers to add a field from the switcher it will add to', async () => {
    renderShell()
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy())

    // The button belongs in the switcher row, alongside the field pills: that row
    // is the one place on the screen that already answers "which fields exist".
    const add = screen.getByRole('button', { name: /Add field/i })
    const lastField = screen.getByRole('button', { name: /PNQ-wheat-01/ })
    expect(add.getAttribute('aria-expanded')).toBe('false')
    expect(add.closest('header')).toBe(lastField.closest('header'))

    // Opening it takes over the page rather than being added to it. The field view
    // underneath is about a field the operator is not registering, and every card
    // in it is a distraction from "which crop, which station, sown when".
    fireEvent.click(add)
    expect(await screen.findByText('Register a field')).toBeTruthy()
    expect(screen.getByRole('button', { name: /Close/i }).getAttribute('aria-expanded')).toBe(
      'true',
    )
    // The crop cards come from the engine's catalogue rather than a hard-coded list,
    // and every catalogue entry is selectable - including Barley, which is not one
    // of the four headline crops but is cropped on this farm.
    expect(screen.getByRole('button', { name: /^Barley\b/i })).toBeTruthy()
    // And the field view is not rendered underneath it.
    expect(screen.queryByText('Soil water')).toBeNull()
    expect(screen.queryByText('Crop age')).toBeNull()
    expect(screen.queryByRole('heading', { name: 'Water today' })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /Close/i }))
    await waitFor(() => expect(screen.queryByText('Register a field')).toBeNull())
    // Closing gives the field view back, unchanged.
    expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy()
  })
})
