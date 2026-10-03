import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['src/**/*.test.{ts,tsx}'],
    // Plans are pure functions, so the polling hook and the formatting helpers
    // carry the logic worth guarding. Coverage thresholds are set on them rather
    // than on the page components, which are cheap-but-brittle to assert on.
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      include: ['src/lib/**', 'src/api/**', 'src/hooks/**'],
      thresholds: { lines: 60, functions: 60, statements: 60 },
    },
  },
})
