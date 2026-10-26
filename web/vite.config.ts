import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const backend = process.env.DROWNING_API_TARGET || 'http://127.0.0.1:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    proxy: {
      '/api': backend,
      '/health': backend,
      '/ready': backend,
    },
  },
})
