import type { ReactNode } from 'react'

export function Panel({ kicker, title, subtitle, right, children, id }: {
  kicker?: string; title: string; subtitle?: string; right?: ReactNode; children: ReactNode; id?: string
}) {
  return <section className="panel page-panel" id={id}>
    <div className="panel-header">
      <div>{kicker && <div className="panel-kicker">{kicker}</div>}<h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>
      {right}
    </div>
    <div className="panel-body">{children}</div>
  </section>
}

export function Banner({ error, notice }: { error?: string; notice?: string }) {
  if (!error && !notice) return null
  return <div className={`message flow ${error ? 'error' : 'notice'}`} role="status">{error || notice}</div>
}

export function Tag({ tone, children }: { tone: 'ok' | 'warn' | 'bad' | 'info'; children: ReactNode }) {
  return <span className={`tag tag-${tone}`}>{children}</span>
}

export function Check({ ok, label, hint }: { ok: boolean | null; label: string; hint?: string }) {
  return <li className={`check ${ok === null ? 'na' : ok ? 'ok' : 'bad'}`}>
    <span className="check-mark" aria-hidden>{ok === null ? '·' : ok ? '✓' : '✕'}</span>
    <span>{label}{!ok && hint ? <small>{hint}</small> : null}</span>
  </li>
}

export function Bars({ rows }: { rows: Array<{ label: string; value: number; note?: string }> }) {
  const max = Math.max(1, ...rows.map((row) => row.value))
  return <div className="bars">{rows.map((row) => <div className="bar-row" key={row.label}>
    <span className="bar-label">{row.label}</span>
    <span className="bar-track"><i style={{ width: `${(row.value / max) * 100}%` }} /></span>
    <span className="bar-value">{new Intl.NumberFormat('vi-VN').format(row.value)}{row.note ? ` · ${row.note}` : ''}</span>
  </div>)}</div>
}

export function Matrix({ labels, rows, rowTitle, colTitle }: {
  labels: string[]; rows: number[][]; rowTitle: string; colTitle: string
}) {
  return <div className="table-scroll"><table className="matrix">
    <thead><tr><th>{rowTitle} ↓ / {colTitle} →</th>{labels.map((label) => <th key={label}>{label}</th>)}</tr></thead>
    <tbody>{rows.map((row, index) => <tr key={labels[index]}><th>{labels[index]}</th>{row.map((value, column) =>
      <td key={labels[column]} className={index === column ? 'diag' : value > 0 ? 'off' : ''}>{value}</td>)}</tr>)}</tbody>
  </table></div>
}
