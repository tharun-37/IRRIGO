/**
 * The farm on satellite imagery.
 *
 * The dashboard's other cards answer "what do I do"; this one answers "where".
 * It is the first thing a farmer looks for on a phone — the field they are
 * standing in — so the map is a real, pannable map rather than a decorative
 * rectangle.
 *
 * **Where the imagery comes from.** Two free, keyless tile services:
 *
 * - Esri World Imagery, the default: recent high-resolution true colour.
 * - Sentinel-2 cloudless (EOX), a real Sentinel-2 mosaic, for the wider view.
 *
 * Neither needs a key and neither is proxied, so no credential is exposed to
 * the browser — the trade is that the tiles are requested straight from the
 * provider and their attribution is shown on the map, as both require.
 *
 * **Where the polygons come from.** The sowing registry stores a station and an
 * area, not a boundary. The station is a real coordinate, but it is a district
 * reference and often its city; the field itself is somewhere nearby. The
 * polygons are therefore explicitly illustrative: they are laid out in a ring
 * around a farm area near the station, sized from the real area, and the map
 * says so. Presenting them as surveyed boundaries would be a lie the farmer
 * could check against the ground.
 *
 * Leaflet is imported dynamically so the map's ~150 kB is paid for by the
 * farmer view only, and not at all under jsdom, which has no layout engine to
 * give the map a size.
 */

import { useEffect, useRef, useState } from 'react'
import 'leaflet/dist/leaflet.css'

export type MapTone = 'dry' | 'good' | 'wet' | 'rest' | 'none'

/** One plot on the map, already positioned and labelled by the caller. */
export interface MapField {
  fieldId: string
  label: string
  lat: number
  lon: number
  tone: MapTone
}

const TONE_COLOR: Record<MapTone, string> = {
  dry: '#f59e0b',
  good: '#16a34a',
  wet: '#2563eb',
  rest: '#94a3b8',
  none: '#94a3b8',
}

type LayerKey = 'satellite' | 'sentinel'

const LAYERS: Record<
  LayerKey,
  { label: string; url: string; attribution: string; maxNativeZoom: number; maxZoom: number }
> = {
  satellite: {
    label: 'Satellite',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Imagery © Esri, Maxar, Earthstar Geographics',
    maxNativeZoom: 19,
    maxZoom: 19,
  },
  sentinel: {
    label: 'Sentinel-2',
    url: 'https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2023_3857/default/g/{z}/{y}/{x}.jpg',
    attribution: 'Sentinel-2 cloudless © EOX IT Services, CC BY 4.0',
    maxNativeZoom: 15,
    maxZoom: 18,
  },
}

const LEGEND: { tone: MapTone; word: string }[] = [
  { tone: 'dry', word: 'Water' },
  { tone: 'good', word: 'Fine' },
  { tone: 'wet', word: 'Wet' },
  { tone: 'rest', word: 'Resting' },
]

// jsdom has no layout engine, so a Leaflet map there is a container with no
// size and a pile of warnings. The field list below the map carries the same
// information, so the tests lose nothing by skipping the tiles.
const ENABLED = import.meta.env.MODE !== 'test'

/** Default view when a field has no station coordinates: central India. */
const FALLBACK_CENTRE = { lat: 21.15, lon: 78.09 }

export function FieldMap({
  fields,
  selectedId,
  centre,
  onSelect,
  className = '',
}: {
  fields: MapField[]
  selectedId: string | null
  centre: { lat: number; lon: number } | null
  onSelect: (fieldId: string) => void
  className?: string
}) {
  const hostRef = useRef<HTMLDivElement | null>(null)
  const mapRef = useRef<import('leaflet').Map | null>(null)
  const leafletRef = useRef<typeof import('leaflet') | null>(null)
  const tilesRef = useRef<import('leaflet').TileLayer | null>(null)
  const selectRef = useRef(onSelect)
  selectRef.current = onSelect

  const [layer, setLayer] = useState<LayerKey>('satellite')
  const [ready, setReady] = useState(false)
  const [failed, setFailed] = useState(false)

  const view = centre ?? FALLBACK_CENTRE
  // The initial view is read once, when the map is created. Keeping it in a ref
  // stops a field change from tearing the map down and building it again.
  const initialView = useRef(view)
  const viewRef = useRef(view)
  viewRef.current = view

  useEffect(() => {
    if (!ENABLED) return
    const host = hostRef.current
    if (!host) return
    let cancelled = false
    let map: import('leaflet').Map | null = null

    void (async () => {
      const L = (await import('leaflet')).default
      if (cancelled || !hostRef.current) return
      leafletRef.current = L
      map = L.map(hostRef.current, {
        zoomControl: false,
        // The page scrolls; a map that swallows the wheel makes the page feel
        // stuck. Zoom is on the control, and touch pinch still works.
        scrollWheelZoom: false,
        zoomSnap: 0.5,
        attributionControl: true,
      }).setView([initialView.current.lat, initialView.current.lon], 17)
      L.control.zoom({ position: 'bottomright' }).addTo(map)
      mapRef.current = map
      setReady(true)
    })().catch(() => {
      if (!cancelled) setFailed(true)
    })

    return () => {
      cancelled = true
      map?.remove()
      mapRef.current = null
      leafletRef.current = null
      tilesRef.current = null
      setReady(false)
      // Leaflet stores its identity on the DOM node. React reuses that node
      // across the StrictMode mount/unmount pair, and a stale id makes the
      // second mount throw "Map container is already initialized".
      const used = host as unknown as { _leaflet_id?: number }
      used._leaflet_id = undefined
    }
  }, [])

  // Recentre when the chosen field changes, keeping the zoom the farmer set.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    map.panTo([viewRef.current.lat, viewRef.current.lon], { animate: true })
  }, [centre?.lat, centre?.lon, ready])

  // Tile layer. Recreated on the layer toggle only.
  useEffect(() => {
    const map = mapRef.current
    const L = leafletRef.current
    if (!map || !L || !ready) return
    const spec = LAYERS[layer]
    const tiles = L.tileLayer(spec.url, {
      attribution: spec.attribution,
      maxNativeZoom: spec.maxNativeZoom,
      maxZoom: spec.maxZoom,
      crossOrigin: true,
    }).addTo(map)
    tilesRef.current = tiles
    return () => {
      tiles.remove()
      if (tilesRef.current === tiles) tilesRef.current = null
    }
  }, [layer, ready])

  // Field outlines. Rebuilt when the selected field moves, because selection is
  // drawn as weight and fill rather than as a second marker on top.
  useEffect(() => {
    const map = mapRef.current
    const L = leafletRef.current
    if (!map || !L || !ready) return
    const group = L.layerGroup().addTo(map)
    const metresPerDegree = 111_320

    for (const field of fields) {
      const selected = field.fieldId === selectedId
      const halfLat = 160 / metresPerDegree
      const halfLon = halfLat / Math.max(0.2, Math.cos((field.lat * Math.PI) / 180))
      const polygon = L.polygon(
        [
          [field.lat - halfLat, field.lon - halfLon],
          [field.lat - halfLat, field.lon + halfLon],
          [field.lat + halfLat, field.lon + halfLon],
          [field.lat + halfLat, field.lon - halfLon],
        ],
        {
          color: selected ? '#0f172a' : TONE_COLOR[field.tone],
          weight: selected ? 3 : 2,
          opacity: selected ? 1 : 0.85,
          fillColor: TONE_COLOR[field.tone],
          fillOpacity: selected ? 0.34 : 0.18,
          dashArray: selected ? undefined : '3 3',
        },
      )
      // A DOM node, not an HTML string: the field id is user input, and Leaflet
      // would otherwise set it with innerHTML.
      const tip = document.createElement('span')
      tip.textContent = field.label
      polygon.bindTooltip(tip, {
        permanent: true,
        direction: 'center',
        className: 'field-map__tip',
      })
      polygon.on('click', () => selectRef.current(field.fieldId))
      group.addLayer(polygon)
    }

    return () => {
      group.remove()
    }
  }, [fields, selectedId, ready])

  // The card animates in beside a resizing column; Leaflet only notices a size
  // change when told, and an uninformed map is the classic half-grey canvas.
  useEffect(() => {
    const map = mapRef.current
    const host = hostRef.current
    if (!map || !host || !ready || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(() => map.invalidateSize())
    observer.observe(host)
    return () => observer.disconnect()
  }, [ready])

  const selected = fields.find((field) => field.fieldId === selectedId) ?? null

  return (
    <section
      className={`field-map card relative overflow-hidden ${className}`}
      aria-label="Satellite map of your fields"
    >
      <div ref={hostRef} className="absolute inset-0 z-0" />

      {/* Vignette: the tiles are bright and the white chrome needs a dark edge
          to sit against. Purely optical, and it never intercepts a gesture. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 z-[450]"
        style={{
          background:
            'radial-gradient(130% 100% at 50% 45%, transparent 45%, rgba(15,23,42,0.38) 100%)',
        }}
      />

      <div className="pointer-events-none absolute inset-x-0 top-0 z-[700] flex items-start justify-between gap-2 p-3">
        <div className="pointer-events-auto max-w-[70%] rounded-card bg-ink-900/85 px-3 py-2 text-white shadow-lift backdrop-blur">
          <p className="text-[11px] font-bold uppercase tracking-[0.12em] text-white/90">
            Your field
          </p>
          <p className="truncate text-[14px] font-semibold leading-tight">
            {selected?.fieldId ?? 'No field selected'}
          </p>
          <p className="truncate text-[12px] text-white/90">
            Approximate location · boundary illustrative
          </p>
        </div>

        <div className="pointer-events-auto flex items-center gap-0.5 rounded-pill bg-white/90 p-0.5 shadow-lift backdrop-blur">
          {(Object.keys(LAYERS) as LayerKey[]).map((key) => (
            <button
              key={key}
              type="button"
              onClick={() => setLayer(key)}
              aria-pressed={layer === key}
              className={`min-h-[34px] rounded-pill px-3 py-1.5 text-[12px] font-semibold transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-brand-600 ${
                layer === key ? 'bg-ink-900 text-white' : 'text-slate-700 hover:text-ink-900'
              }`}
            >
              {LAYERS[key].label}
            </button>
          ))}
        </div>
      </div>

      <div className="pointer-events-none absolute inset-x-0 bottom-0 z-[700] flex items-end justify-between gap-2 p-3">
        <div className="pointer-events-auto flex items-center gap-2.5 rounded-pill bg-white/90 px-3 py-1.5 text-[12px] font-medium text-slate-700 shadow-lift backdrop-blur">
          {LEGEND.map((item) => (
            <span key={item.tone} className="flex items-center gap-1">
              <span
                className="h-2 w-2 rounded-full"
                style={{ backgroundColor: TONE_COLOR[item.tone] }}
              />
              {item.word}
            </span>
          ))}
        </div>
      </div>

      {!ENABLED && (
        <div className="absolute inset-0 z-[600] grid place-items-center bg-gradient-to-br from-[#cfe0c6] via-[#a8c9a0] to-[#6f9c7a]">
          <p className="rounded-pill bg-white/80 px-3 py-1.5 text-[11px] font-semibold text-ink-700">
            Satellite view
          </p>
        </div>
      )}

      {ENABLED && !ready && !failed && (
        <div className="absolute inset-0 z-[600] grid place-items-center bg-ink-100">
          <p className="text-[11px] font-medium text-ink-500">Loading satellite imagery…</p>
        </div>
      )}

      {failed && (
        <div className="absolute inset-0 z-[600] grid place-items-center bg-ink-100 px-4 text-center">
          <p className="text-[11px] font-medium text-ink-500">
            The map could not load. Your fields are listed below.
          </p>
        </div>
      )}
    </section>
  )
}
