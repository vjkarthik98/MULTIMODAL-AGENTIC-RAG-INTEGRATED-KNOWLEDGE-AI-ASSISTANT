import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/auth': { target: 'http://localhost:8000', changeOrigin: true },
      '/rag':  { target: 'http://localhost:8000', changeOrigin: true },
      '/admin':{ target: 'http://localhost:8000', changeOrigin: true },
      '/api':  { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
  // sourcemap explicitly off (this is already Vite's default, but stated so a
  // future contributor enabling it for error-tracking has to make a conscious
  // choice rather than silently exposing full source in production).
  build: { outDir: 'dist', sourcemap: false },
})
