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
    diseaseConfidence: 0.41,
    diseaseProbabilities: { rust: 0.41, blight: 0.34, healthy: 0.25 },
    riskLevel: 'watch',
    cropStressIndex: 0.7,
    moistureBand: 'dry',
    yieldTPerHa: 5.4,
    yieldPotentialTPerHa: 6.4,
    yieldModelled: true,
    nutrientDemandKgHa: { nitrogen: 42.5, phosphorus: 18.2, potassium: 31.4 },
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

function renderShell(view: 'farmer' | 'console' = 'console') {
  const result = render(
    <MemoryRouter initialEntries={['/']}>
      <App />
    </MemoryRouter>,
  )
  // The page opens on the farmer view. These assertions are about the console,
  // so switch before the advisory resolves; the switch is in the header, which
  // renders without waiting for data.
  if (view === 'console') {
    fireEvent.click(screen.getByRole('button', { name: 'Console' }))
  }
  return result
}

/**
 * Render the dashboard with one field swapped into the advisor feed.
 *
 * The advice a card gives depends on the field it is looking at, so most of
 * the interesting cases are the same field with one reading changed.
 */
function renderWithField(field: unknown, view: 'farmer' | 'console' = 'console') {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const body = route(String(input))
      const payload = body === FEED ? { ...FEED, fields: [field] } : body
      return {
        ok: true,
        status: 200,
        json: async () => payload,
        text: async () => JSON.stringify(payload),
      } as Response
    }),
  )
  return renderShell(view)
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
    // it" is answered before "what should I do".
    const details = screen.getByRole('heading', { name: 'Field details' }).closest('section')
    expect(details).toBeTruthy()

    // A short strip of the physical basics, and only the ones nothing else on the
    // screen prints: the name and crop are on the switcher and the crop-age
    // card, the station titles the weather card, day and stage are on the hero,
    // roots are in soil water and the season figures are in the crop-age panel.
    const detailText = (details as HTMLElement).textContent ?? ''
    for (const fact of ['Soil', 'Area', 'Irrigation', 'Sown']) {
      expect(within(details as HTMLElement).getByText(fact)).toBeTruthy()
    }
    for (const dropped of [
      'Name',
      'Crop',
      'Station',
      'Roots',
      'Stage',
      'Crop age',
      'Degree days',
      'Disease risk',
      'Confidence',
      'Crop stress index',
      'Yield outlook',
      'Stress in',
    ]) {
      expect(detailText).not.toContain(dropped)
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

    // Every card fills the grid cell it is given. Crop age is not a direct grid
    // item — it sits in a wrapper so it can span two columns on md — and a
    // stretched wrapper does not stretch the card inside it, which is what left
    // it the shortest card in the row.
    for (const title of ['Soil water', 'Crop age', 'Farm this week']) {
      const card = screen.getByRole('heading', { name: title }).closest('section')
      expect(card?.className).toContain('h-full')
    }
    const sky = screen.getByText(/· now/).closest('section')
    expect(sky?.className).toContain('h-full')
  })

  it('does not fill the sky card by growing its one-row humidity box', async () => {
    renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

// The sky card carries two readings, not one. Its neighbours either side hold a
    // four-figure panel and a season track, so a lone humidity row left this
    // card short and the spare height had to go somewhere visible.
    const sky = screen.getByText(/· now/).closest('section')
    expect(within(sky as HTMLElement).getByText('Humidity')).toBeTruthy()
    expect(within(sky as HTMLElement).getByText('Rain today')).toBeTruthy()

    // Top-aligned: `justify-center` on the body pushed the header off the top
    // edge, and would clip it outright on a card whose content outgrew it.
    const body = sky?.querySelector(':scope > div')
    expect(body?.className).not.toContain('justify-center')

    // And the temperature is sized from the card's own width.
    expect(sky?.querySelector('.text-\\[clamp\\(2\\.25rem\\,15cqw\\,4\\.25rem\\)\\]')).toBeTruthy()
  })

  it('shows the weather from the plan when there is no live feed', async () => {
    renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    const sky = screen.getByText(/· now/).closest('section')
    expect(sky).toBeTruthy()

    // The plan's own readings: 25.7 °C, 41 % humidity, no rain.
    expect(within(sky as HTMLElement).getByText('26°')).toBeTruthy()
    expect(within(sky as HTMLElement).getByText('41%')).toBeTruthy()
    expect(within(sky as HTMLElement).getByText('0.0 mm')).toBeTruthy()

    // And the card says where its numbers came from, so nobody reads them as
    // the sky right now.
    expect(
      within(sky as HTMLElement).getByText(/From the plan.s own weather at PNQ/),
    ).toBeTruthy()
  })

  it('shows the live observation when the feed answers', async () => {
    const live = {
      available: true,
      station: 'PNQ',
      source: 'openweather',
      temperatureC: 18.2,
      humidityPct: 88,
      windMs: 4.1,
      rainfallMm: 4.5,
      condition: 'light rain',
      observedAt: '2026-10-02T09:00:00+00:00',
    }
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input)
        const payload = url.includes('/weather/live/') ? live : route(url)
        return {
          ok: true,
          status: 200,
          json: async () => payload,
          text: async () => JSON.stringify(payload),
        } as Response
      }),
    )

    renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    const sky = screen.getByText(/· now/).closest('section')
    await waitFor(() => expect(within(sky as HTMLElement).getByText('18°')).toBeTruthy())

    // The live reading replaces the plan’s own, right down to the rain.
    expect(within(sky as HTMLElement).getByText('88%')).toBeTruthy()
    expect(within(sky as HTMLElement).getByText('4.5 mm')).toBeTruthy()
    expect(within(sky as HTMLElement).getByText('Light Rain')).toBeTruthy()

    // And the card says so: this is the sky right now, not the weather the
    // crop was planned against.
    expect(within(sky as HTMLElement).getByText(/Live now at PNQ/)).toBeTruthy()
  })

  it('keeps the field details to the four facts nothing else states', async () => {
    renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    const details = screen.getByRole('heading', { name: 'Field details' }).closest('section')

    // One row of four recessed panels, each label above its value. As four
    // label/value lines spread across the full width, every value sat hundreds
    // of pixels from its own label and the strip read as a thin rule.
    expect(details?.querySelectorAll('h3')).toHaveLength(0)
    const list = details?.querySelector('dl')
    expect(list?.children).toHaveLength(4)
    expect(list?.className).toContain('lg:grid-cols-4')
    for (const cell of Array.from(list?.children ?? [])) {
      expect(cell.className).toContain('panel')
      expect(cell.querySelector('dt')).toBeTruthy()
      expect(cell.querySelector('dd')).toBeTruthy()
    }

    // Soil type is the fact this block exists for. The fixture's is 'clay', so
    // the title-cased rendering is checked rather than the raw value.
    expect(within(details as HTMLElement).getByText('Clay')).toBeTruthy()
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

    // Nutrients are drawn against their sufficiency bands. Scoped to the sensor
    // board: the disease card names the same three nutrients as its fertilizer
    // plan, and a page-wide lookup now finds two of each.
    const board = screen.getByText('7-in-1 soil sensor').closest('section')
    expect(within(board as HTMLElement).getByText('Nitrogen')).toBeTruthy()
    expect(within(board as HTMLElement).getByText('Phosphorus')).toBeTruthy()
    expect(within(board as HTMLElement).getByText('Potassium')).toBeTruthy()
    expect(screen.getAllByText('In range').length).toBeGreaterThan(0)
  })

  it('tells the farmer what the crop is fighting and what to do about it', async () => {
    renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    const disease = screen
      .getByRole('heading', { name: 'Disease & crop health' })
      .closest('section')

    // The verdict, in words a farmer would use, not a probability.
    expect(within(disease as HTMLElement).getByText('Disease watch')).toBeTruthy()
    // The card names a disease, so it must say what it is: a risk indicator
    // over sensor telemetry, not a confirmed diagnosis.
    expect(within(disease as HTMLElement).getByText('Risk indicator')).toBeTruthy()
    // And how sure the model is, so the verdict can be weighed.
    expect(within(disease as HTMLElement).getByText('41% confidence')).toBeTruthy()
    expect(within(disease as HTMLElement).getByText('Crop stress index')).toBeTruthy()
    expect(within(disease as HTMLElement).getByText('0.70')).toBeTruthy()

    // The likely disease and the one behind it. A single label hides the
    // decision: 41% rust against 34% blight is not 41% rust against 2% blight.
    expect(within(disease as HTMLElement).getByText('Most likely')).toBeTruthy()
    expect(within(disease as HTMLElement).getByText('Rust')).toBeTruthy()
    expect(within(disease as HTMLElement).getByText('Next possibility')).toBeTruthy()
    expect(within(disease as HTMLElement).getByText('Blight 34%')).toBeTruthy()

    // The pesticide advice names the thing the operator would buy and apply.
    expect(within(disease as HTMLElement).getByText('Pesticide')).toBeTruthy()
    expect(within(disease as HTMLElement).getByText('Monitor closely')).toBeTruthy()

    // The fertilizer dose the engine asks for, so "fertilizer" is a quantity.
    expect(within(disease as HTMLElement).getByText('43 kg/ha')).toBeTruthy()
    expect(within(disease as HTMLElement).getByText('18 kg/ha')).toBeTruthy()
    expect(within(disease as HTMLElement).getByText('31 kg/ha')).toBeTruthy()

    // The fixture's nutrients (40/20/130 ppm) are all inside the sufficiency
    // bands, so no shortfall is invented.
    expect(within(disease as HTMLElement).getByText(/Nutrients adequate/)).toBeTruthy()

    // The close runner-up is called out, because it changes whether to spray.
    expect(
      within(disease as HTMLElement).getByText(/Blight is close behind/),
    ).toBeTruthy()
    expect(
      within(disease as HTMLElement).getByText(/Act now: scout the field this week/),
    ).toBeTruthy()

    // And the whole action list is repeated at the top of the page, where the
    // operator reads first: the water decision alone is not today's only work.
    expect(screen.getByText(/Also today: scout the field for disease/)).toBeTruthy()
  })

  it('turns a nutrient shortfall into a dose of a product the farmer can buy', async () => {
    // Same feed, but the soil nitrogen has fallen below the sufficiency band.
    const starved = {
      ...FIELD,
      sensors: { ...FIELD.sensors, nitrogen: 8 },
    }
    renderWithField(starved)

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    const disease = screen
      .getByRole('heading', { name: 'Disease & crop health' })
      .closest('section')

    // The dose covers the engine's 42.5 kg/ha of nitrogen, and it is expressed
    // in the product the operator buys: 42.5 / 0.46 ≈ 92 kg/ha of urea.
    expect(
      within(disease as HTMLElement).getByText(/43 kg\/ha N \(about 92 kg\/ha urea\)/),
    ).toBeTruthy()
    // And the action says to apply it.
    expect(
      within(disease as HTMLElement).getByText(/Apply fertilizer for nitrogen/),
    ).toBeTruthy()
  })

  it('still asks for a fungicide when the verdict is risky but the model is unsure', async () => {
    // A `risk` verdict with a low confidence. The confidence is shown next to
    // the advice so the operator can weigh it, but the advice itself must not
    // contradict the verdict: "Disease risk" and "No pesticide needed" on the
    // same card is a contradiction only the operator can resolve.
    const risky = {
      ...FIELD,
      risk: {
        ...FIELD.risk,
        riskLevel: 'risk' as const,
        diseaseConfidence: 0.35,
        diseaseProbabilities: { rust: 0.35, blight: 0.31, healthy: 0.34 },
      },
    }
    renderWithField(risky)

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    const disease = screen
      .getByRole('heading', { name: 'Disease & crop health' })
      .closest('section')

    expect(within(disease as HTMLElement).getByText('Disease risk')).toBeTruthy()
    expect(within(disease as HTMLElement).getByText('Apply fungicide')).toBeTruthy()
    // The uncertainty is visible, so the operator knows to scout before mixing.
    expect(within(disease as HTMLElement).getByText('35% confidence')).toBeTruthy()
    // And the action names the spray once, not twice.
    expect(within(disease as HTMLElement).getByText(/Act now: spray a fungicide\./)).toBeTruthy()
    expect(
      within(disease as HTMLElement).queryByText(/treat the disease now/),
    ).toBeNull()
    // The top of the page carries it too.
    expect(screen.getByText(/Also today: spray a fungicide/)).toBeTruthy()
  })

  it('shows the yield outlook and names what is holding it back', async () => {
    renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    const outlook = screen.getByRole('heading', { name: 'Yield outlook' }).closest('section')

    // The number the farmer is working towards, in the unit they sell in.
    expect(within(outlook as HTMLElement).getByText('5.4')).toBeTruthy()
    expect(within(outlook as HTMLElement).getByText('t/ha expected')).toBeTruthy()
    // The ceiling comes from the crop catalogue, not from a formula: 6.4 t/ha.
    expect(within(outlook as HTMLElement).getByText('6.4')).toBeTruthy()
    expect(within(outlook as HTMLElement).getByText('potential t/ha')).toBeTruthy()

    // And the reason it may not arrive. The fixture is 70% depleted, so water
    // is what is holding the crop back.
    expect(within(outlook as HTMLElement).getByText('Limiting factor')).toBeTruthy()
    expect(within(outlook as HTMLElement).getByText('Water')).toBeTruthy()

    // The gap is quoted in tonnes and percent, so it can be weighed against the
    // cost of the water that would close it: 6.4 - 5.4 = 1.0 t/ha, 16%.
    expect(within(outlook as HTMLElement).getByText('Yield gap')).toBeTruthy()
    expect(within(outlook as HTMLElement).getByText('16%')).toBeTruthy()
    expect(within(outlook as HTMLElement).getByText(/Irrigate now/)).toBeTruthy()
    expect(within(outlook as HTMLElement).getByText(/1\.0 t\/ha \(16%\) short of potential/)).toBeTruthy()
  })

  it('does not report a yield gap for a crop the yield model does not cover', async () => {
    // The model was trained on six crops and potato is not one of them. It
    // still answers, but with a crop-agnostic number, and setting that against
    // potato's 20 t/ha ceiling would report a 91% opportunity that is an
    // artefact of the encoding rather than anything the farmer can act on.
    const potato = {
      ...FIELD,
      crop: 'Potato',
      risk: {
        ...FIELD.risk,
        yieldTPerHa: 1.7,
        yieldPotentialTPerHa: 20,
        yieldModelled: false,
      },
    }
    renderWithField(potato)

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    const outlook = screen.getByRole('heading', { name: 'Yield outlook' }).closest('section')

    // The figure is still shown, but labelled as the estimate it is.
    expect(within(outlook as HTMLElement).getByText('1.7')).toBeTruthy()
    expect(within(outlook as HTMLElement).getByText('t/ha estimate')).toBeTruthy()
    // And no gap is claimed against a ceiling the figure was never measured on.
    expect(within(outlook as HTMLElement).getByText('Yield gap')).toBeTruthy()
    expect(within(outlook as HTMLElement).getByText('—')).toBeTruthy()
    expect(within(outlook as HTMLElement).queryByText('91%')).toBeNull()
    // The caveat says why, and what the catalogue does know.
    expect(within(outlook as HTMLElement).getByText(/no Potato basis/)).toBeTruthy()
    expect(within(outlook as HTMLElement).getByText(/catalogue puts Potato at 20/)).toBeTruthy()
  })

  it('does not call a field near potential while it is far below its ceiling', async () => {
    // Nothing is wrong: no disease, no irrigation due, nutrients adequate.
    // The model still puts the field 45% below its ceiling, so the card has to
    // say that rather than claim the field is near potential.
    const shortOfPotential = {
      ...FIELD,
      status: 'monitor',
      recommendation: { ...FIELD.recommendation, shouldIrrigate: false, depthMm: 0 },
      water: { ...FIELD.water, depletionFraction: 0.1 },
      risk: {
        ...FIELD.risk,
        riskLevel: 'healthy' as const,
        cropStressIndex: 0.05,
        yieldTPerHa: 3.5,
        yieldPotentialTPerHa: 6.4,
        yieldModelled: true,
      },
    }
    renderWithField(shortOfPotential)

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water within 24 hours' })).toBeTruthy(),
    )

    const outlook = screen.getByRole('heading', { name: 'Yield outlook' }).closest('section')

    expect(within(outlook as HTMLElement).getByText('No single cause')).toBeTruthy()
    expect(within(outlook as HTMLElement).queryByText('None — near potential')).toBeNull()
    expect(within(outlook as HTMLElement).getByText('45%')).toBeTruthy()
    expect(within(outlook as HTMLElement).getByText(/Nothing to act on today/)).toBeTruthy()
    expect(
      within(outlook as HTMLElement).getByText(/2\.9 t\/ha \(45%\) gap is what this field's conditions/),
    ).toBeTruthy()
  })

  it('puts disease and yield after the water decision, not before it', async () => {
    const { container } = renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    // The operator's order of questions: what do I do today, then what is wrong
    // with the crop, then what will I get. Disease before yield, because a
    // disease verdict changes the yield answer.
    const order = Array.from(container.querySelectorAll('section'))
      .map((node) => node.querySelector('h1, h2')?.textContent ?? '')
      .filter(Boolean)
    expect(order.indexOf('Water today')).toBeGreaterThanOrEqual(0)
    expect(order.indexOf('Disease & crop health')).toBeGreaterThan(
      order.indexOf('7-in-1 soil sensor'),
    )
    expect(order.indexOf('Yield outlook')).toBeGreaterThan(
      order.indexOf('Disease & crop health'),
    )
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
    // field-details grid and "Rain today" now also reads on the weather card, so
    // these are matched loosely or scoped.
    expect(screen.getAllByText('Soil').length).toBeGreaterThan(0)
    expect(screen.getByText('Crop need')).toBeTruthy()
    const hero = screen.getByRole('heading', { level: 1 }).closest('section')
    expect(within(hero as HTMLElement).getByText('Rain today')).toBeTruthy()
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

  it('opens on the farmer view: one action, in pump minutes rather than millimetres', async () => {
    renderWithField(FIELD, 'farmer')

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    // 6,133 seconds is 102 minutes at the pump: the unit the work is done in.
    expect(screen.getByText('102')).toBeTruthy()
    expect(screen.getByText('minutes')).toBeTruthy()
    expect(screen.getByText(/at the pump · 18.4 mm over 1.0 acre/)).toBeTruthy()

    // One sentence that says why, and the two controls a farmer has to press.
    expect(screen.getByText(/The soil is dry and the wheat is using the most water/)).toBeTruthy()
    expect(screen.getByRole('button', { name: /Listen/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Mark done/ })).toBeTruthy()

    // None of the console's instrument vocabulary leaks into this view.
    expect(screen.queryByText('Recommended depth')).toBeNull()
    expect(screen.queryByText('7-in-1 soil sensor')).toBeNull()
    expect(screen.queryByText('Crop stress index')).toBeNull()
  })

  it('farmer view puts the rest of today in farmer units', async () => {
    renderWithField(FIELD, 'farmer')

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    // The fertilizer band test is the same one the console card uses, so it
    // still says "adequate" here; the spray level is the same "watch".
    expect(screen.getByText('No fertilizer today')).toBeTruthy()
    expect(screen.getByText('Check for disease')).toBeTruthy()

    // 5.4 t/ha is about 22 quintal to the acre, which is how it is sold.
    expect(screen.getByText('Expected yield')).toBeTruthy()
    expect(screen.getByText('22 quintal/acre')).toBeTruthy()
  })

  it('farmer view records a done mark per field per day', async () => {
    renderWithField(FIELD, 'farmer')

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    const mark = screen.getByRole('button', { name: /Mark done/ })
    expect(mark.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(mark)

    const done = screen.getByRole('button', { name: /^Done$/ })
    expect(done.getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByText('Saved for today.')).toBeTruthy()
  })

  it('hands the field back to the console from the farmer view', async () => {
    renderWithField(FIELD, 'farmer')

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    fireEvent.click(screen.getByRole('button', { name: /More: soil, weather, yield/ }))

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Disease & crop health' })).toBeTruthy(),
    )
    expect(screen.queryByRole('button', { name: /Mark done/ })).toBeNull()
  })
})
