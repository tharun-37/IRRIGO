/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  // Section hues are composed at runtime - `section-${section}` in the Card
  // primitive - so the scanner sees only the template literal and none of the real
  // names. Without this it strips every one of them from the build and the cards
  // silently lose their edges, which is the kind of failure that survives a type
  // check and a passing test run and only shows up on screen.
  safelist: [
    'section-edge',
    // The edge is drawn on a pseudo-element, and a safelist entry for that has to
    // be the variant form. Without this second entry `section-edge` survives and
    // `section-edge::after` does not, which is a card with a hue and no edge.
    'section-edge:after',
    'section-mint',
    'section-sky',
    'section-cyan',
    'section-violet',
    'section-amber',
    'section-rose',
    'section-slate',
  ],
  theme: {
    extend: {
      colors: {
        // A single restrained palette. Every status colour in the interface
        // comes from here so that "amber" means the same thing in a stat tile,
        // a chart legend and an alert row.
        brand: {
          50: '#eff6ff',
          100: '#dbeafe',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        status: {
          healthy: '#16a34a',
          healthySoft: '#dcfce7',
          warn: '#f59e0b',
          warnSoft: '#fef3c7',
          alert: '#dc2626',
          alertSoft: '#fee2e2',
          idle: '#64748b',
          idleSoft: '#e2e8f0',
        },
        ink: {
          900: '#0f172a',
          700: '#334155',
          500: '#64748b',
          400: '#94a3b8',
          300: '#cbd5e1',
          200: '#e2e8f0',
          100: '#f1f5f9',
          50: '#f8fafc',
        },
        page: {
          from: '#c7d7fe',
          to: '#eef2ff',
        },
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Consolas', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        // A card is a raised surface, and it was spending four drop-shadow layers
        // and two large blurs to say so. Over a page that already has a warped
        // gradient behind it, all of that weight did not read as elevation - it
        // read as dirt, and it darkened the tinted background immediately around
        // every card so a grid of them sat on a visible grey field.
        //
        // The depth is now carried by translucency instead: the card is a
        // translucent plate with a backdrop blur (see `.card` in index.css), so
        // what sits behind it shows through and softens. That is a real optical
        // depth cue, and it survives on a low-contrast background where a drop
        // shadow has nothing to land against and dies.
        //
        // What is left is the minimum a plate needs: a hairline contact shadow
        // where it meets the page, and the inner white rim that keeps the top edge
        // lit. Without the rim the card reads as a hole rather than a surface.
        card: '0 1px 2px rgba(15, 23, 42, 0.05), inset 0 1px 0 rgba(255, 255, 255, 0.9)',
        // Grouped content *inside* a card is recessed rather than raised, so the
        // eye reads card-then-panel as two depths. Halved from the previous inset:
        // at the old depth a nested surface looked like a separate card that had
        // been dropped into another one, which is not the relationship.
        panel: 'inset 0 1px 2px rgba(15, 23, 42, 0.06), 0 1px 0 rgba(255, 255, 255, 0.7)',
        // Interactive chrome, above cards: it must clear the card it sits on, so
        // this one keeps a genuine cast rather than a tint.
        lift: '0 1px 2px rgba(15, 23, 42, 0.07), 0 6px 14px -6px rgba(15, 23, 42, 0.14)',
        nav: '0 1px 2px rgba(15, 23, 42, 0.05)',
      },
      borderRadius: {
        card: '12px',
        pill: '999px',
      },
      keyframes: {
        pulseRing: {
          '0%': { transform: 'scale(0.9)', opacity: '0.7' },
          '100%': { transform: 'scale(1.6)', opacity: '0' },
        },
      },
      animation: {
        pulseRing: 'pulseRing 1.8s ease-out infinite',
      },
    },
  },
  plugins: [],
}
