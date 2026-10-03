/**
 * Tests for the formatting helpers.
 *
 * These are the functions every displayed number passes through, and the
 * "no data yet" case is the one that matters most: a device that has not
 * reported is a normal state, and a `NaN` on screen reads as a fault in the
 * controller rather than an absence of data.
 */

import { describe, expect, it } from 'vitest'
import { clockTime, duration, int, num, pct, relativeTime, titleCase } from './format'

const DASH = '—'

describe('num', () => {
  it('renders a value to the requested precision', () => {
    expect(num(3.14159, 2)).toBe('3.14')
    expect(num(3.14159, 0)).toBe('3')
  })

  it('defaults to one decimal place', () => {
    expect(num(12.34)).toBe('12.3')
  })

  it('renders a missing value as a dash rather than NaN', () => {
    expect(num(null)).toBe(DASH)
    expect(num(undefined)).toBe(DASH)
    expect(num(Number.NaN)).toBe(DASH)
  })

  it('keeps zero, which is a real reading and not missing data', () => {
    expect(num(0)).toBe('0.0')
  })
})

describe('pct', () => {
  it('scales a fraction to a percentage', () => {
    expect(pct(0.42)).toBe('42%')
    // 0.4265 * 100 is 42.649999999999999 in binary floating point, so the
    // assertion states the arithmetic the operator actually sees rather than
    // an idealised one.
    expect(pct(0.4265, 1)).toBe('42.6%')
  })

  it('renders a missing value as a dash', () => {
    expect(pct(null)).toBe(DASH)
  })

  it('keeps zero percent, which means the model found no risk', () => {
    expect(pct(0)).toBe('0%')
  })
})

describe('int', () => {
  it('rounds and groups thousands', () => {
    expect(int(1_234.6)).toBe('1,235')
  })

  it('renders a missing value as a dash', () => {
    expect(int(undefined)).toBe(DASH)
  })
})

describe('duration', () => {
  it('reads as days and hours past a day', () => {
    expect(duration(86_400 + 3 * 3_600)).toBe('1d 3h')
  })

  it('reads as hours and minutes under a day', () => {
    expect(duration(3 * 3_600 + 25 * 60)).toBe('3h 25m')
  })

  it('reads as minutes under an hour', () => {
    expect(duration(125)).toBe('2m')
  })

  it('reads as seconds under a minute', () => {
    expect(duration(42)).toBe('42s')
  })

  it('renders a missing value as a dash', () => {
    expect(duration(null)).toBe(DASH)
  })

  it('clamps a negative duration to zero, since a negative irrigation is nonsense', () => {
    expect(duration(-500)).toBe('0s')
  })
})

describe('relativeTime', () => {
  it('reports seconds, minutes, hours and days', () => {
    const now = Date.now()
    const ago = (seconds: number) => new Date(now - seconds * 1000).toISOString()
    expect(relativeTime(ago(5))).toBe('5s ago')
    expect(relativeTime(ago(300))).toBe('5m ago')
    expect(relativeTime(ago(7_200))).toBe('2h ago')
    expect(relativeTime(ago(172_800))).toBe('2d ago')
  })

  it('treats a future timestamp as just now, since clock skew is common', () => {
    const ahead = new Date(Date.now() + 60_000).toISOString()
    expect(relativeTime(ahead)).toBe('just now')
  })

  it('renders missing and unparseable input as a dash', () => {
    expect(relativeTime(null)).toBe(DASH)
    expect(relativeTime('')).toBe(DASH)
    expect(relativeTime('not a date')).toBe(DASH)
  })
})

describe('clockTime', () => {
  it('renders a time of day', () => {
    const value = new Date('2026-01-01T14:35:00Z').toISOString()
    expect(clockTime(value)).not.toBe(DASH)
  })

  it('renders missing and unparseable input as a dash', () => {
    expect(clockTime(null)).toBe(DASH)
    expect(clockTime('nonsense')).toBe(DASH)
  })
})

describe('titleCase', () => {
  it('title-cases values that arrive in snake or upper case', () => {
    expect(titleCase('LIGHT_IRRIGATION')).toBe('Light Irrigation')
    expect(titleCase('root rot')).toBe('Root Rot')
  })

  it('renders a missing value as a dash', () => {
    expect(titleCase(null)).toBe(DASH)
  })
})
