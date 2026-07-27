import { useEffect, useState } from 'react'
import { fetchScans } from './api'
import ScanCard from './components/ScanCard.jsx'

const POLL_INTERVAL_MS = 3000

export default function App() {
  const [scans, setScans] = useState([])
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const data = await fetchScans()
        if (!cancelled) {
          setScans(data)
          setError(null)
        }
      } catch (err) {
        if (!cancelled) setError(err.message)
      }
    }

    poll()
    const interval = setInterval(poll, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  return (
    <main className="app">
      <header className="app-header">
        <h1>Card scanner</h1>
        <p className="subtitle">Photos from the BeagleBone Black, identified here.</p>
      </header>

      {error && <p className="error">Couldn't reach the backend: {error}</p>}

      <div className="scan-grid">
        {scans.length === 0 && !error && (
          <p className="empty">No scans yet — waiting for the scanner.</p>
        )}
        {scans.map((scan) => (
          <ScanCard key={scan.id} scan={scan} />
        ))}
      </div>
    </main>
  )
}
