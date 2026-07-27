// Same-origin in production (FastAPI serves the built frontend); proxied
// to localhost:8000 in dev via vite.config.js.

export async function fetchScans() {
  const res = await fetch('/api/scans')
  if (!res.ok) throw new Error('Failed to load scans')
  return res.json()
}

export async function fetchScan(id) {
  const res = await fetch(`/api/scans/${id}`)
  if (!res.ok) throw new Error('Failed to load scan')
  return res.json()
}
