import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// During `npm run dev`, proxy API/upload requests to the FastAPI server
// so the frontend can be developed on a laptop while pointed at the real
// backend (or `uvicorn app.main:app --reload` running locally).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/uploads': 'http://localhost:8000',
    },
  },
})
