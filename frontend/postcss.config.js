export default {
  plugins: {
    // Named form: Vite re-resolves this file when it changes, which re-initialises
    // the CSS pipeline and makes Tailwind re-read tailwind.config.js. Editing the
    // theme alone does not reach a dev server that is already running — the old
    // boxShadow values survive until the process restarts.
    tailwindcss: { config: './tailwind.config.js' },
    autoprefixer: {},
  },
}
