/**
 * Tests for the field-registration form.
 *
 * A sowing is the one input an operator types by hand and everything downstream is
 * derived from it, so the assertions here are about the two things that would
 * quietly produce a wrong plan: a crop that is not really selected, and a rejection
 * whose message nobody can act on.
 */

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError, api } from '../api/client'
import { AddField } from './AddField'

/** Shaped like an entry from `GET /api/system/crops`. */
function crop(over: Record<string, unknown> = {}) {
  return {
    name: 'Cotton',
    kcInitial: 0.35,
    kcMid: 1.15,
    kcEnd: 0.7,
    rootDepthMaxM: 2,
    depletionFractionP: 0.65,
    optimalPh: [6, 8],
    gddBaseTempC: 12,
    gddStageEnds: [400, 900, 1900, 2400],
    seasonGdd: 2400,
    lengthStageDays: [30, 50, 60, 25],
    seasonDays: 165,
    ky: 1.1,
    yieldPotentialTHa: 1.8,
    variants: ['Bt_Hybrid'],
    suitableMethods: ['Drip', 'Furrow', 'Sprinkler', 'Basin', 'Flood'],
    warnings: {},
    ...over,
  }
}

const CROP = crop()
const MAIZE = crop({ name: 'Maize', suitableMethods: ['Drip', 'Sprinkler', 'Furrow', 'Basin', 'Flood'] })
const WHEAT = crop({ name: 'Wheat', suitableMethods: ['Sprinkler', 'Basin', 'Furrow', 'Drip', 'Flood'] })
/** Barley is not one of the four headline crops but is on the farm, so it must be reachable. */
const BARLEY = crop({ name: 'Barley', suitableMethods: ['Sprinkler', 'Basin', 'Furrow'] })
const RICE = crop({
  name: 'Rice',
  kcInitial: 1.05,
  rootDepthMaxM: 0.5,
  depletionFractionP: 0.1,
  variants: [],
  // The engine flags every non-ponded method for rice, so paddy is the only one left.
  suitableMethods: ['Paddy'],
  warnings: {
    Sprinkler: 'Rice is managed as a maintained pond depth.',
  },
})

const OPTIONS = {
  stations: ['HYD', 'LKO', 'PNQ'],
  crops: ['Barley', 'Cotton', 'Maize', 'Potato', 'Rice', 'Wheat'],
  soils: ['Clay', 'Loam', 'Sandy_Loam', 'Silt_Loam'],
  methods: ['Basin', 'Drip', 'Flood', 'Furrow', 'Paddy', 'Sprinkler'],
}

const CREATED = {
  fieldId: 'PNQ-cotton-01',
  station: 'PNQ',
  crop: 'Cotton',
  cropVariant: null,
  sowingDate: '2024-09-01',
  soilType: 'Loam',
  methodName: 'Drip',
  fieldAreaM2: 2200,
  mulched: false,
  nitrogenRegime: 1,
  restrictingDepthM: null,
  emergenceFraction: 0.5,
  daysSinceSowing: 121,
  seasonDay: 122,
  notes: '',
  plan: {
    fieldId: 'PNQ-cotton-01',
    station: 'PNQ',
    crop: 'Cotton',
    asOf: '2024-12-31',
    sowingDate: '2024-09-01',
    daysSinceSowing: 121,
    stage: 'mid_season',
    stageProgress: 0.7,
    rootDepthCm: 200,
    tawMm: 280,
    rawMm: 182,
    depletionMm: 45,
    depletionFraction: 0.16,
    daysUntilStress: null,
    kc: 1.15,
    et0MmDay: 4.1,
    etcMmDay: 4.7,
    requirement: { net_requirement_mm: 12.9 },
    schedule: null,
    dataSource: 'nasa-power',
  },
}

const cropsMock = vi.fn()
const optionsMock = vi.fn()
const createMock = vi.fn()

beforeEach(() => {
  cropsMock
    .mockReset()
    .mockResolvedValue({ crops: [CROP, RICE, MAIZE, WHEAT, BARLEY] })
  optionsMock.mockReset().mockResolvedValue(OPTIONS)
  createMock.mockReset().mockResolvedValue(CREATED)
  vi.spyOn(api, 'crops').mockImplementation(cropsMock)
  vi.spyOn(api, 'options').mockImplementation(optionsMock)
  vi.spyOn(api, 'createField').mockImplementation(createMock)
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function fillRequired(): void {
  fireEvent.change(screen.getByLabelText(/field id/i), {
    target: { value: 'PNQ-cotton-01' },
  })
  fireEvent.change(screen.getByLabelText(/weather station/i), {
    target: { value: 'PNQ' },
  })
  fireEvent.change(screen.getByLabelText(/sowing date/i), { target: { value: '2024-09-01' } })
}

function pickCrop(name: string): void {
  fireEvent.click(screen.getByRole('button', { name: new RegExp(`^${name}\\b`, 'i') }))
}

describe('AddField', () => {
  it('offers the four primary crops as cards and every other crop as a chip', async () => {
    render(<AddField onRegistered={() => {}} />)
    await waitFor(() => expect(cropsMock).toHaveBeenCalled())
    for (const name of ['Wheat', 'Rice', 'Maize', 'Cotton']) {
      expect(screen.getByRole('button', { name: new RegExp(`^${name}\\b`, 'i') })).toBeTruthy()
    }
    // Barley is not a headline crop but is cropped on this farm, so it has to be
    // selectable. Telling a grower the system does not do barley is not the same
    // as not making barley the first choice.
    expect(screen.getByRole('button', { name: /^Barley\b/i })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: /^Maize\b/i }))
    expect(
      screen.getByRole('button', { name: /^Maize\b/i }).getAttribute('aria-pressed'),
    ).toBe('true')
  })

  it('shows the selected crop the numbers behind its water demand', async () => {
    render(<AddField onRegistered={() => {}} />)
    await waitFor(() => expect(cropsMock).toHaveBeenCalled())
    pickCrop('Cotton')
    // Rooting depth, the depletion threshold and the season length are what
    // explain why cotton needs more water than rice on identical sensors.
    expect(screen.getByText('2.0 m')).toBeTruthy()
    expect(screen.getByText('65% of available water')).toBeTruthy()
    expect(screen.getByText('165 days')).toBeTruthy()
  })

  it('warns before committing a crop and method pairing the engine dislikes', async () => {
    render(<AddField onRegistered={() => {}} />)
    await waitFor(() => expect(cropsMock).toHaveBeenCalled())
    pickCrop('Rice')
    // Selecting the crop defaults to the only method the engine considers suitable,
    // so there is nothing to warn about yet.
    expect(screen.getByLabelText(/irrigation method/i)).toHaveProperty('value', 'Paddy')
    expect(screen.queryByText(/maintained pond depth/)).toBeNull()

    fireEvent.change(screen.getByLabelText(/irrigation method/i), {
      target: { value: 'Sprinkler' },
    })
    expect(screen.getByText(/maintained pond depth/)).toBeTruthy()
  })

  it('lists every method exactly once', async () => {
    render(<AddField onRegistered={() => {}} />)
    await waitFor(() => expect(cropsMock).toHaveBeenCalled())
    pickCrop('Cotton')
    const select = screen.getByLabelText(/irrigation method/i) as HTMLSelectElement
    // The first option is the "no selection" placeholder, not a method.
    const names = Array.from(select.options)
      .map((option) => option.value)
      .filter(Boolean)
    expect(names).toEqual([...new Set(names)])
    expect(names).toHaveLength(OPTIONS.methods.length)
    // The crop's most efficient suitable method leads, so the default is defensible.
    expect(names[0]).toBe('Drip')
  })

  it('will not submit while required fields are missing', async () => {
    render(<AddField onRegistered={() => {}} />)
    await waitFor(() => expect(cropsMock).toHaveBeenCalled())
    expect(screen.getByText(/Choose a crop/)).toBeTruthy()
    expect(
      (screen.getByRole('button', { name: /register field/i }) as HTMLButtonElement).disabled,
    ).toBe(true)
  })

  it('refuses a sowing date in the future', async () => {
    render(<AddField onRegistered={() => {}} />)
    await waitFor(() => expect(cropsMock).toHaveBeenCalled())
    pickCrop('Cotton')
    fireEvent.change(screen.getByLabelText(/field id/i), {
      target: { value: 'PNQ-cotton-01' },
    })
    fireEvent.change(screen.getByLabelText(/weather station/i), {
      target: { value: 'PNQ' },
    })
    fireEvent.change(screen.getByLabelText(/sowing date/i), {
      target: { value: '2099-01-01' },
    })
    expect(screen.getByText(/in the future/)).toBeTruthy()
  })

  it('posts the registration and shows the plan that came back with it', async () => {
    const registered = vi.fn()
    render(<AddField onRegistered={registered} />)
    await waitFor(() => expect(cropsMock).toHaveBeenCalled())
    pickCrop('Cotton')
    fillRequired()
    fireEvent.click(screen.getByRole('button', { name: /register field/i }))

    await waitFor(() => expect(createMock).toHaveBeenCalled())
    expect(createMock).toHaveBeenCalledWith(
      expect.objectContaining({
        fieldId: 'PNQ-cotton-01',
        station: 'PNQ',
        crop: 'Cotton',
        sowingDate: '2024-09-01',
        // The method the crop selection defaulted to is carried through.
        methodName: 'Drip',
      }),
    )
    // The new field's own first answer, not a placeholder that fills in later.
    expect(await screen.findByText(/PNQ-cotton-01 registered and planned/)).toBeTruthy()
    expect(screen.getByText('12.9 mm')).toBeTruthy()
    expect(registered).toHaveBeenCalledWith('PNQ-cotton-01')
  })

  it('surfaces the server message when a registration is rejected', async () => {
    createMock.mockRejectedValue(
      new ApiError("crop: unknown crop 'Dragonfruit'; available: ['Barley', 'Cotton']", 422),
    )
    render(<AddField onRegistered={() => {}} />)
    await waitFor(() => expect(cropsMock).toHaveBeenCalled())
    pickCrop('Cotton')
    fillRequired()
    fireEvent.click(screen.getByRole('button', { name: /register field/i }))

    // The rejection has to reach the operator: a silent failure here looks
    // identical to a form that did not save.
    expect(await screen.findByRole('alert')).toHaveProperty(
      'textContent',
      expect.stringContaining('Dragonfruit'),
    )
  })

  it('reports a catalogue that will not load instead of rendering an empty form', async () => {
    cropsMock.mockRejectedValue(new ApiError('cannot reach the backend', 0))
    render(<AddField onRegistered={() => {}} />)
    expect(await screen.findByText(/crop catalogue could not be loaded/i)).toBeTruthy()
    // An empty crop list is worse than an error: it reads as "no crops exist".
    expect(screen.queryByRole('button', { name: /register field/i })).toBeNull()
  })
})