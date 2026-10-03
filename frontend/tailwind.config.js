/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
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
        // A card is a raised surface, so it needs several layers to read as
        // one: a hairline contact shadow where it meets the page, a tight
        // ambient shadow for the edge, a soft bloom under the body, and a wide
        // one for the lift. Four rather than three, with the outer two carrying
        // more alpha, because at the previous weights the shadow died before it
        // reached the tinted page and adjacent cards separated by a hairline
        // border alone. The inner white rim is kept: without it the top edge
        // loses the light source and the card reads as a hole rather than a
        // plate sitting above the page.
        card:
          '0 1px 2px rgba(15, 23, 42, 0.06), 0 3px 8px -2px rgba(15, 23, 42, 0.08), ' +
          '0 16px 34px -10px rgba(15, 23, 42, 0.20), ' +
          '0 32px 64px -24px rgba(15, 23, 42, 0.24), ' +
          'inset 0 1px 0 rgba(255, 255, 255, 0.95)',
        // Grouped content *inside* a card is recessed rather than raised, so the
        // eye reads card-then-panel as two depths instead of everything floating
        // at the same level. Deepened alongside `card`: a raised card over a
        // barely-sunk panel gave one step of depth, where the nesting is the
        // whole point of the two-level hierarchy.
        panel:
          'inset 0 2px 4px rgba(15, 23, 42, 0.09), inset 0 1px 2px rgba(15, 23, 42, 0.05), ' +
          '0 1px 0 rgba(255, 255, 255, 0.85)',
        // Interactive chrome, above cards: it must clear the card it sits on.
        lift:
          '0 2px 4px rgba(15, 23, 42, 0.08), 0 10px 20px -6px rgba(15, 23, 42, 0.16), ' +
          '0 24px 48px -12px rgba(15, 23, 42, 0.26)',
        nav: '0 1px 2px rgba(15, 23, 42, 0.06), 0 8px 24px -12px rgba(15, 23, 42, 0.18)',
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
