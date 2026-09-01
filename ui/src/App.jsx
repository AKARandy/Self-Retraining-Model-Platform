import { useEffect, useState, useCallback } from 'react'

const API = '/api'

function usePolling(fn, intervalMs = 5000) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const refresh = useCallback(async () => {
    try {
      const res = await fetch(fn())
      if (!res.ok) throw new Error(`${res.status}`)
      setData(await res.json())
      setError(null)
    } catch (e) {
      setError(String(e))
    }
  }, [fn])
  useEffect(() => {
    refresh()
    const t = setInterval(refresh, intervalMs)
    return () => clearInterval(t)
  }, [refresh, intervalMs])
  return [data, error, refresh]
}

function Section({ title, subtitle, children }) {
  return (
    <section className="card">
      <h2>{title}</h2>
      {subtitle && <p className="muted">{subtitle}</p>}
      {children}
    </section>
  )
}

function StatusPill({ status }) {
  const cls = status === 'Succeeded' ? 'ok' : status === 'Failed' || status === 'Error' ? 'bad' : 'run'
  return <span className={`pill ${cls}`}>{status || '?'}</span>
}

export default function App() {
  const [apiKey, setApiKey] = useState(localStorage.getItem('apiKey') || '')

  const [datasets] = usePolling(() => `${API}/datasets`)
  const [versions, versionsErr] = usePolling(() => `${API}/datasets/1/versions`)
  const [runs] = usePolling(() => `${API}/train-runs`)
  const [models] = usePolling(() => `${API}/registry/models`)
  const [current, currentErr] = usePolling(() => `${API}/registry/models/house-price-sk/current`)
  const [drift] = usePolling(() => `${API}/monitoring/drift-checks`)

  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')

  async function post(path, body) {
    if (!apiKey) {
      setMsg('set an API key first (Authorization: Bearer …)')
      return null
    }
    setBusy(true)
    setMsg('')
    try {
      const res = await fetch(`${API}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      setMsg(res.ok ? 'ok' : `${res.status}: ${data.detail || 'error'}`)
      return data
    } finally {
      setBusy(false)
    }
  }

  async function submitRun() {
    await post('/train-runs', { dataset_id: 1, n_trials: 15 })
  }
  async function checkDrift() {
    await post('/monitoring/check-drift', {})
  }
  async function saveKey() {
    localStorage.setItem('apiKey', apiKey)
    setMsg('api key stored locally')
  }

  return (
    <div className="wrap">
      <header>
        <h1>MLOps Platform</h1>
        <p className="muted">
          house-prices · FastAPI monolith · Argo Workflows · MLflow · drift → retrain loop (polls every 5s)
        </p>
      </header>

      <Section title="API key" subtitle="Bearer token for write endpoints (stored in localStorage only)">
        <div className="row">
          <input
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="paste API_KEY value from .env"
          />
          <button onClick={saveKey} disabled={busy}>Save</button>
        </div>
        {msg && <p className="msg">{msg}</p>}
      </Section>

      <Section title="Production model" subtitle="current Production stage from the MLflow registry">
        {current ? (
          <p className="big">
            {current.name} <span className="pill ok">v{current.version}</span>
          </p>
        ) : (
          <p className="muted">{currentErr ? 'none promoted yet' : '…'}</p>
        )}
      </Section>

      <Section title="Dataset versions" subtitle="DVC-versioned data registered through the API">
        {versions ? (
          <table>
            <thead><tr><th>version</th><th>rows</th><th>cols</th><th>source</th><th>dvc md5</th></tr></thead>
            <tbody>
              {versions.map((v) => (
                <tr key={v.version}>
                  <td>v{v.version}</td>
                  <td>{v.n_rows}</td>
                  <td>{v.n_cols}</td>
                  <td>{v.original_filename}</td>
                  <td className="mono">{v.dvc_md5.slice(0, 12)}…</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">{versionsErr || '…'}</p>
        )}
      </Section>

      <Section title="Training runs" subtitle="Argo DAG status, synced through GET /train-runs">
        <div className="row">
          <button onClick={submitRun} disabled={busy}>Submit training run (15 trials)</button>
        </div>
        {runs ? (
          <table>
            <thead><tr><th>#</th><th>workflow</th><th>status</th><th>params</th><th>link</th></tr></thead>
            <tbody>
              {runs.slice(0, 8).map((r) => (
                <tr key={r.id}>
                  <td>{r.id}</td>
                  <td className="mono">{r.argo_name || '—'}</td>
                  <td><StatusPill status={r.status} /></td>
                  <td className="mono">{JSON.stringify(r.params)}</td>
                  <td>{r.argo_name && <a href={r.argo_ui} target="_blank" rel="noreferrer">Argo UI</a>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">…</p>
        )}
      </Section>

      <Section title="Model registry" subtitle="all registered models and stages">
        {models && models.length ? (
          <table>
            <thead><tr><th>model</th><th>versions</th><th>latest stages</th></tr></thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.name}>
                  <td>{m.name}</td>
                  <td>{m.n_versions}</td>
                  <td className="mono">{JSON.stringify(m.latest)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">…</p>
        )}
      </Section>

      <Section title="Drift monitoring" subtitle="recent checks; a breach auto-triggers retraining">
        <div className="row">
          <button onClick={checkDrift} disabled={busy}>Run drift check now</button>
        </div>
        {drift && drift.length ? (
          <table>
            <thead><tr><th>#</th><th>verdict</th><th>retrain triggered</th><th>training run</th><th>when</th></tr></thead>
            <tbody>
              {drift.map((d) => (
                <tr key={d.id}>
                  <td>{d.id}</td>
                  <td><span className={`pill ${d.verdict === 'ok' ? 'ok' : 'bad'}`}>{d.verdict}</span></td>
                  <td>{d.triggered_retrain ? 'yes' : 'no'}</td>
                  <td>{d.training_run_id || '—'}</td>
                  <td className="mono">{d.created_at.slice(0, 19)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">no checks yet</p>
        )}
      </Section>
    </div>
  )
}
