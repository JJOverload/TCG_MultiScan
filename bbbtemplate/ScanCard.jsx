export default function ScanCard({ scan }) {
  return (
    <article className="scan-card" data-status={scan.status}>
      <img src={scan.image_url} alt="Captured card" className="scan-thumb" />

      <div className="scan-body">
        <span className={`scan-status scan-status--${scan.status}`}>{scan.status}</span>

        {scan.status === 'done' && scan.cards.length === 0 && (
          <p className="scan-empty">No cards recognized</p>
        )}

        <ul className="scan-cards">
          {scan.cards.map((card, i) => (
            <li key={i}>
              <span className="card-name">{card.name}</span>
              <span className="card-confidence">{Math.round(card.confidence * 100)}%</span>
            </li>
          ))}
        </ul>

        {scan.status === 'error' && <p className="scan-error">{scan.error}</p>}
      </div>
    </article>
  )
}
