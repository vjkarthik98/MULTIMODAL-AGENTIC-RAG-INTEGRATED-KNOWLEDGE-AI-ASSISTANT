/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
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
