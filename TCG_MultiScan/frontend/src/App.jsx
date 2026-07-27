import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Fetch data from the FastAPI server endpoint
    fetch('http://localhost:8000/api/data')
      .then((res) => res.json())
      .then((data) => {
        setData(data.message)
        setLoading(false)
      })
      .catch((err) => {
        console.error("Error fetching data:", err)
        setLoading(false)
      })
  }, [])

  return (
    <div style={{ textAlign: 'center', marginTop: '50px' }}>
      <h1>React + FastAPI</h1>
      {loading ? <p>Loading backend data...</p> : <p>Backend says: {data}</p>}
    </div>
  )
}

export default App

// Start your React development environment:
// npm run dev

// Open http://localhost:5173 in your web browser to see your 
// React interface successfully displaying data parsed 
// straight from your backend
