/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      screens: {
        // Used only by the auth pages (Login/Forgot/Reset password) in place
        // of `sm:` for their compact-vs-full-size split. `sm:` is width-only,
        // so a landscape tablet or a short browser window (wide, but under
        // ~780px tall) still qualified as "desktop" and got full-size
        // padding/fonts sized for viewports that are also tall enough to fit
        // them — overflowing the actual visible height and forcing a scroll
        // to reach the submit button. `roomy:` requires BOTH dimensions.
        roomy: { raw: '(min-width: 640px) and (min-height: 780px)' },
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
      },
      colors: {
        magik: {
          orange:        '#f97316',
          'orange-hover':'#ea6c0a',
          bg:            '#000000',
          sidebar:       '#0f0f0f',
          card:          '#141414',
          input:         '#1c1c1c',
          border:        '#242424',
          'border-light':'#333333',
          muted:         '#6b7280',
          subtle:        '#9ca3af',
        },
      },
    },
  },
  plugins: [],
}
