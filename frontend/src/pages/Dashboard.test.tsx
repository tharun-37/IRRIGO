/**
 * Render smoke test for the field view.
 *
 * This is the screen a farmer opens to decide whether to water, so the assertions
 * are about the instruction being present and unambiguous rather than about
 * layout. A regression that drops the depth figure or replaces the plain-language
 * instruction with jargon would pass a type check and fail here.
 *
 * V3 is farmer-only: there is no console view to switch into, so every test here
 * runs against the one view, and a couple of tests assert the console's absence —
 * including that "More" now hands over to the crop-photo check rather than to a
 * set of analytical cards.
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

const CROP_HEALTH_STATUS = {
  serviceAvailable: true,
  deployedModel: false,
  modelVersion: 'none',
  preprocessingVersion: 'none',
  validationStatus: 'pending',
  supportedClasses: [],
  supportedCrops: ['rice', 'wheat', 'maize', 'cotton', 'potato', 'tomato', 'barley', 'sugarcane', 'groundnut', 'onion'],
  limitations: ['no validated model deployed'],
  statusLabels: {},
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
  if (url.includes('/crop-health/status'))
    return CROP_HEALTH_STATUS
  if (url.includes('/crop-health/analyses')) return { analyses: [], count: 0 }
  if (url.includes('/health')) return { status: 'ok', version: '2.0.0' }
  return { readings: [], recommendations: [], devices: [], zones: [], events: [] }
}

/** The field view is the only view; the shell renders it directly. */
function renderShell(initial = '/') {
  return render(
    <MemoryRouter initialEntries={[initial]}>
      <App />
    </MemoryRouter>,
  )
}

/**
 * Render the dashboard with one field swapped into the advisor feed.
 *
 * The advice a card gives depends on the field it is looking at, so most of
 * the interesting cases are the same field with one reading changed.
 */
function renderWithField(field: unknown) {
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
  return renderShell()
}

describe('Field view', () => {
  beforeEach(() => {
    // This environment ships no storage at all; a small stub keeps the
    // "cleared read outlives the page" semantics testable, and unstubAllGlobals
    // drops it after each test so one test's ack cannot leak into the next.
    const backing: Record<string, string> = {}
    vi.stubGlobal('localStorage', {
      getItem: (key: string) => (key in backing ? backing[key] : null),
      setItem: (key: string, value: string) => {
        backing[key] = String(value)
      },
      removeItem: (key: string) => {
        delete backing[key]
      },
      clear: () => {
        for (const key of Object.keys(backing)) delete backing[key]
      },
    })
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

    // The advice is in the farmer's words and units. The depth appears inside
    // the pump sentence, so it is matched within that line.
    expect(screen.getByText('Water today')).toBeTruthy()
    expect(
      screen.getByText(/at the pump · 18\.4 mm over 1\.0 acre/),
    ).toBeTruthy()
    expect(document.body.textContent).toContain('73,600 L')

    // The reason is one sentence a farmer can argue with, not a model dump.
    expect(
      screen.getByText(/The soil is dry and the wheat is using the most water/),
    ).toBeTruthy()
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
    expect(screen.getByText('Soil moisture')).toBeTruthy()
    expect(screen.getByText('Plan today')).toBeTruthy()
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

    // The action card's sky line carries the live reading when there is one.
    await waitFor(() =>
      expect(screen.getByText(/18° now · 4\.5 mm rain today/)).toBeTruthy(),
    )
  })

  it('states the decision, in pump minutes rather than millimetres', async () => {
    renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    // 6,133 seconds is 102 minutes at the pump: the unit the work is done in.
    expect(screen.getByText('102')).toBeTruthy()
    expect(screen.getByText('minutes')).toBeTruthy()
    expect(screen.getByText(/at the pump · 18.4 mm over 1.0 acre/)).toBeTruthy()

    // The two controls a farmer has to press: Listen, and Approve & run
    // (since this field owes water and has no approval yet).
    expect(screen.getByRole('button', { name: /Listen/ })).toBeTruthy()
    expect(screen.getByRole('button', { name: /Approve & run/i })).toBeTruthy()

    // The console's instrument vocabulary stays gone with the console.
    expect(screen.queryByText('Recommended depth')).toBeNull()
    expect(screen.queryByText('7-in-1 soil sensor')).toBeNull()
    expect(screen.queryByText('Crop stress index')).toBeNull()
  })

  it('puts the rest of today in farmer units', async () => {
    renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    expect(screen.getByText('No fertilizer today')).toBeTruthy()
    // The watch chip names the likeliest disease above the healthy remainder.
    expect(screen.getByText('Watch for Rust')).toBeTruthy()

    // 5.4 t/ha is about 22 quintal to the acre, which is how it is sold.
    expect(screen.getByText('Expected yield')).toBeTruthy()
    expect(screen.getByText('22 quintal/acre')).toBeTruthy()
  })

  it('turns a nutrient shortfall into a bag of product the farmer can buy', async () => {
    // Same feed, but the soil nitrogen has fallen below the sufficiency band.
    // The engine asks for 42.5 kg/ha of nitrogen; 42.5 / 0.46 / 2.471 is about
    // 37 kg/acre of urea, and this field at 0.99 acres rounds to half a bag.
    const starved = {
      ...FIELD,
      sensors: { ...FIELD.sensors, nitrogen: 8 },
    }
    renderWithField(starved)

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    const notes = screen.getByLabelText(/Yield and today.s notes/)
    expect(within(notes as HTMLElement).getByText('Add half a bag of urea')).toBeTruthy()
    expect(within(notes as HTMLElement).getByText(/37 kg\/acre · N short/)).toBeTruthy()
  })

  it('holds the bag of urea when a fresh photo names a blight', async () => {
    // Same starved soil, but this week's leaf photo named bacterial leaf
    // blight. The backend attaches the disease's nutrition word to the
    // advisory, and the chip vetoes the very feed the sensors were about to
    // call for — quoting the photo that earned the hold.
    const held = {
      ...FIELD,
      sensors: { ...FIELD.sensors, nitrogen: 8 },
      photoDisease: {
        classLabel: 'Wheat Bacterial leaf blight',
        rawScore: 0.81,
        photoOn: '2026-10-02T07:41:00+00:00',
        nutrition: {
          nutrient: 'nitrogen',
          action: 'withhold',
          note: 'Extra nitrogen prolongs blight; stop the planned nitrogen until the crop recovers.',
        },
      },
    }
    renderWithField(held)

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    const notes = screen.getByLabelText(/Yield and today.s notes/)
    const hold = within(notes as HTMLElement).getByText('Hold the fertilizer today')
    expect(hold).toBeTruthy()
    expect(within(notes as HTMLElement).queryByText(/of urea/)).toBeNull()
    expect(document.body.textContent).toContain('was found in a photo on 2026-10-02')
  })

  it('passes the photo-sourced potash check before the nitrogen feed', async () => {
    // A short nitrogen band still owes urea, but this week's photo read
    // Alternaria, which thrives on potash-starved plants: the chip keeps the
    // dose and demands the potash check first, with the photo's date on it.
    const checked = {
      ...FIELD,
      sensors: { ...FIELD.sensors, nitrogen: 8 },
      photoDisease: {
        classLabel: 'Cotton Alternaria leaf spot',
        rawScore: 0.72,
        photoOn: '2026-10-02T07:41:00+00:00',
        nutrition: {
          nutrient: 'potassium',
          action: 'check',
          note: 'Alternaria thrives on potassium-starved plants.',
        },
      },
    }
    renderWithField(checked)

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    const notes = screen.getByLabelText(/Yield and today.s notes/)
    expect(within(notes as HTMLElement).getByText('Add half a bag of urea')).toBeTruthy()
    expect(within(notes as HTMLElement).getByText(/Check potash first/)).toBeTruthy()
    expect(within(notes as HTMLElement).getByText(/photo of 2026-10-02/)).toBeTruthy()
  })

  it('still carries the photo when nothing is owed, with the read named', async () => {
    // The screenshot case: a cotton leaf photo names bacterial blight, no
    // nutrient band is tripped and cotton owes no calendar pass — so the
    // dose is genuinely zero. The chip may not invent one, but it may not
    // stay silent either: it keeps the honest "no feed" title and quotes
    // the read the photo made.
    const named = {
      ...FIELD,
      photoDisease: {
        classLabel: 'Cotton Bacterial blight',
        rawScore: 0.66,
        photoOn: '2026-10-02T12:35:00+00:00',
        nutrition: null,
      },
    }
    renderWithField(named)

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    const notes = screen.getByLabelText(/Yield and today.s notes/)
    // Title stays honest; the read and its date arrive in the detail.
    expect(within(notes as HTMLElement).getByText('No fertilizer today')).toBeTruthy()
    expect(
      within(notes as HTMLElement).getByText(
        /a leaf photo named Cotton Bacterial blight on 2026-10-02/,
      ),
    ).toBeTruthy()
  })

  it('returns the chip to green when the farmer marks the leaves clear', async () => {
    // The demo beat: the pushed photo turned the chip warn; the farmer reads
    // it, taps "Mark leaves clear", and the chip reverts to the green
    // no-feed verdict it had before the photo — with the read no longer
    // quoted — because the ack is stored per field, outliving the page.
    const named = {
      ...FIELD,
      photoDisease: {
        classLabel: 'Cotton Bacterial blight',
        rawScore: 0.66,
        photoOn: '2026-10-25T09:00:00+00:00',
        nutrition: null,
      },
    }
    renderWithField(named)

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )
    const notes = screen.getByLabelText(/Yield and today.s notes/)
    expect(within(notes as HTMLElement).getByText('No fertilizer today')).toBeTruthy()

    fireEvent.click(
      within(notes as HTMLElement).getByRole('button', {
        name: /mark leaves clear/i,
      }),
    )

    expect(
      within(notes as HTMLElement).getByText('No fertilizer needed at this stage.'),
    ).toBeTruthy()
    expect(within(notes as HTMLElement).queryByText(/a leaf photo named/)).toBeNull()
    expect(localStorage.getItem('irrig.photoAck.PNQ-wheat-01')).toBe(
      '2026-10-25T09:00:00+00:00',
    )

    // Green after refresh too: the same feed is still served, the stored ack
    // keeps the read cleared on a remount.
    cleanup()
    renderWithField(named)
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )
    const remounted = screen.getByLabelText(/Yield and today.s notes/)
    expect(
      within(remounted as HTMLElement).getByText('No fertilizer needed at this stage.'),
    ).toBeTruthy()
    expect(
      within(remounted as HTMLElement).queryByText(/a leaf photo named/),
    ).toBeNull()
  })

  it('re-opens the note when a newer photo names a disease again', async () => {
    // A cleared read stays cleared, but the chip is not deaf: pushing a fresh
    // photo (a later timestamp) warns again, so each demo pass can re-run
    // green → warn → clear without touching the storage by hand.
    localStorage.setItem('irrig.photoAck.PNQ-wheat-01', '2026-10-25T09:00:00+00:00')
    const newer = {
      ...FIELD,
      photoDisease: {
        classLabel: 'Cotton Bacterial blight',
        rawScore: 0.58,
        photoOn: '2026-10-25T11:30:00+00:00',
        nutrition: null,
      },
    }
    renderWithField(newer)

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )
    const notes = screen.getByLabelText(/Yield and today.s notes/)
    expect(
      within(notes as HTMLElement).getByText(
        /a leaf photo named Cotton Bacterial blight on 2026-10-25/,
      ),
    ).toBeTruthy()
  })

  it('shows the approve button when water is owed and the plan is not yet approved', async () => {
    // A field the plan says needs water, with no approval yet: the button
    // reads "Approve & run" and calls the approve endpoint.
    const watered = {
      ...FIELD,
      recommendation: {
        ...FIELD.recommendation,
        shouldIrrigate: true,
        finalDepthMm: 18.0,
        depthMm: 18.0,
        volumeLitres: 72.0,
        durationSeconds: 120,
        action: 'MODERATE_IRRIGATION',
        confidence: 0.9,
        probability: 0.91,
        threshold: 0.5,
      },
      pumpRateLpm: 12.0,
      areaM2: 4000,
    }
    renderWithField(watered)

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )
    const btn = screen.getByRole('button', { name: /approve & run/i })
    expect(btn).toBeTruthy()
    // button is enabled by default in the test fixture
  })

  it('hides the approve button once the plan is approved', async () => {
    // An open approval on the advisory replaces the approve button with the
    // done toggle, because the node already has the command.
    const approved = {
      ...FIELD,
      recommendation: {
        ...FIELD.recommendation,
        shouldIrrigate: true,
        finalDepthMm: 18.0,
        depthMm: 18.0,
        volumeLitres: 72.0,
        durationSeconds: 120,
        action: 'MODERATE_IRRIGATION',
        confidence: 0.9,
        probability: 0.91,
        threshold: 0.5,
      },
      pumpRateLpm: 12.0,
      areaM2: 4000,
      approval: {
        approvalId: 'abc123',
        approvedAt: '2026-10-25T09:00:00+00:00',
        fieldId: 'PNQ-wheat-01',
        status: 'open',
        depthMm: 18.0,
        durationSeconds: 120,
        volumeLitres: 72.0,
        pumpRateLpm: 12.0,
      },
    }
    renderWithField(approved)

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )
    expect(screen.queryByRole('button', { name: /approve & run/i })).toBeNull()
    expect(screen.getByRole('button', { name: /mark done/i })).toBeTruthy()
  })

  it('carries the calendar top-dress when the sensors read adequate', async () => {
    // Same feed, but a barley field at mid-season: the sensors sit inside the
    // sufficiency bands, yet the crop calendar still owes the split N pass.
    // The dose comes from the decision model's demand (42.5 kg/ha N), so the
    // chip states the same product arithmetic as a shortage would.
    const barley = {
      ...FIELD,
      crop: 'Barley',
    }
    renderWithField(barley)

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    const notes = screen.getByLabelText(/Yield and today.s notes/)
    expect(within(notes as HTMLElement).getByText('Add half a bag of urea')).toBeTruthy()
    expect(within(notes as HTMLElement).getByText(/37 kg\/acre · N top-dress at mid-season/)).toBeTruthy()
  })

  it('breeds no top-dress chip for a crop with no calendar pass', async () => {
    // Wheat at mid-season has no top-dress on file here: the chips must not
    // invent fertilizer for crops the rule does not cover.
    renderWithField(FIELD)

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    const notes = screen.getByLabelText(/Yield and today.s notes/)
    expect(within(notes as HTMLElement).getByText('No fertilizer today')).toBeTruthy()
  })

  it('supports every registered field from a simple switcher, with no console left', async () => {
    renderShell()
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy())

    // Exact match: the photo-check widget's small-caps title also contains
    // "PNQ-wheat-01", so the switcher pill is matched exactly.
    expect(screen.getByRole('button', { name: /^PNQ-wheat-01$/ })).toBeTruthy()

    // No dashboard furniture competing with the two facts, and no console
    // switcher to reach any of it.
    expect(screen.queryByText('Overview')).toBeNull()
    expect(screen.queryByText('Analytics')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Console' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Farmer' })).toBeNull()
    expect(screen.queryByText(/Schematic parcel/)).toBeNull()
  })

  it('summarises the farm in one line beside the switcher', async () => {
    renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    expect(screen.getByText('1 of 1 fields need water')).toBeTruthy()
  })

  it('embeds the photo check as a fourth card that follows the field being read', async () => {
    renderShell()
    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    // The widget is one of the four equal-width cards under the season, the
    // decision and the week.
    const widget = screen.getByLabelText('Crop photo check')
    expect(widget).toBeTruthy()
    expect(within(widget as HTMLElement).getByRole('button', { name: 'Open camera' })).toBeTruthy()

    // It has no field picker of its own: the switcher at the top and the map
    // already choose the field, so the widget only names the one it is reading.
    expect(within(widget as HTMLElement).queryByText('1. Tap your field')).toBeNull()
    expect(within(widget as HTMLElement).queryByRole('button', { name: /Barley-West/ })).toBeNull()
    expect(within(widget as HTMLElement).queryByRole('button', { name: /Cotton-South/ })).toBeNull()
    expect(within(widget as HTMLElement).queryByRole('button', { name: /Potato-East/ })).toBeNull()
    expect(within(widget as HTMLElement).queryByRole('button', { name: /Paddy-North/ })).toBeNull()
    expect(
      within(widget as HTMLElement).getByText(/Crop photo check\s+·\s+PNQ-wheat-01/),
    ).toBeTruthy()

    // It is the compact chrome: no step heading and no history, which stay on
    // the /crop-health page.
    expect(within(widget as HTMLElement).queryByText('Past checks')).toBeNull()
  })

  it('offers to add a field from the switcher it will add to', async () => {
    renderShell()
    await waitFor(() => expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy())

    // The button belongs in the switcher row, alongside the field pills: that row
    // is the one place on the screen that already answers "which fields exist".
    const add = screen.getByRole('button', { name: /Add field/i })
    // Exact name: the photo-check widget's title also carries this field's
    // name, and `add.closest('header')` must be compared with the switcher's.
    const lastField = screen.getByRole('button', { name: /^PNQ-wheat-01$/ })
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
    expect(screen.queryByText('Soil moisture')).toBeNull()
    expect(screen.queryByText('Crop age')).toBeNull()
    expect(screen.queryByRole('heading', { name: 'Water today' })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /Close/i }))
    await waitFor(() => expect(screen.queryByText('Register a field')).toBeNull())
    // Closing gives the field view back, unchanged.
    expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy()
  })

  it('records a done mark per field per day', async () => {
    // Use a field with an active running approval so the "Stop watering"
    // button shows (watered fields without approval show "Approve & run" instead).
    const approvedField = {
      ...FIELD,
      recommendation: {
        ...FIELD.recommendation,
        shouldIrrigate: true,
        finalDepthMm: 18.0,
        depthMm: 18.0,
        volumeLitres: 72.0,
        durationSeconds: 120,
        action: 'MODERATE_IRRIGATION',
        confidence: 0.9,
        probability: 0.91,
        threshold: 0.5,
      },
      approval: {
        approvalId: 'test-approval',
        approvedAt: '2026-10-25T09:00:00+00:00',
        fieldId: 'PNQ-wheat-01',
        status: 'running',
        depthMm: 18.0,
        durationSeconds: 120,
        volumeLitres: 72.0,
        pumpRateLpm: 12.0,
      },
    }
    renderWithField(approvedField)

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Water today' })).toBeTruthy(),
    )

    const mark = screen.getByRole('button', { name: /Stop watering/ })
    expect(mark.getAttribute('aria-pressed')).toBe('false')
    fireEvent.click(mark)

    const done = screen.getByRole('button', { name: /Done for today/ })
    expect(done.getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByText('Done for today')).toBeTruthy()
  })

  it('keeps the crop photo check page a straight capture flow', async () => {
    // The widget covers the everyday check on the field view, and there is no
    // header button or More handover any more: the full page carries the guided
    // flow plus history and is reached by its own route.
    renderShell('/crop-health')

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Crop health' })).toBeTruthy(),
    )

    // The capture flow, in three beats: pick the field, take the photo, read
    // the result. Step 2 names the missing step until a field is tapped; the
    // farmer's own field is offered first, and the demo fields are always
    // available.
    expect(screen.getByText('1. Tap your field')).toBeTruthy()
    expect(screen.getByText('2. Take a photo (choose your field above first)')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /Barley-West Barley/ }))
    await waitFor(() =>
      expect(screen.getByText('2. Take a photo of the plant')).toBeTruthy(),
    )
    expect(
      screen.getByRole('button', { name: 'Open camera or choose photo' }),
    ).toBeTruthy()
    expect(screen.getByRole('button', { name: /Paddy-North Rice/ })).toBeTruthy()
    // The recorded history lives here too.
    expect(screen.getByText('Past checks')).toBeTruthy()
  })

  it('keeps the crop photo check honest about what the model can do', async () => {
    renderShell('/crop-health')

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Crop health' })).toBeTruthy(),
    )

    // The status banner says what the service actually is today: a photo-quality
    // check with no validated model behind it — never a diagnosis.
    expect(
      screen.getByText(/still being tested\. It can tell if a photo is usable/),
    ).toBeTruthy()
  })
})
