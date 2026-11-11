import { useState } from 'react'
import type { CSSProperties, ReactNode } from 'react'
import { CELL_GRID_MAX, CONFIDENCE_MAX, CONFIDENCE_MIN, CONFIDENCE_PRESETS, SCORE_LABELS, clampCellSide, scoreText } from './api'
import type { CellGridChoice } from './api'

/** Slider + preset buttons for the per-box confidence threshold (0–1, uncalibrated model score). */
export function ConfidenceControl({ value, onChange, disabled, title = 'Ngưỡng confidence', hint }: {
  value: number; onChange: (value: number) => void; disabled?: boolean; title?: string; hint?: ReactNode
}) {
  return <div className="confidence-control" role="group" aria-label={title}>
    <div className="confidence-head">
      <span>{title}</span>
      <b className="confidence-value">≥ {value.toFixed(2)}</b>
    </div>
    <input type="range" min={CONFIDENCE_MIN} max={CONFIDENCE_MAX} step={0.05} value={value} disabled={disabled}
      aria-label={`${title} (0–1)`} onChange={(event) => onChange(Number(event.target.value))}
      style={{ '--fill': `${((value - CONFIDENCE_MIN) / (CONFIDENCE_MAX - CONFIDENCE_MIN)) * 100}%` } as CSSProperties} />
    <div className="confidence-presets">
      {CONFIDENCE_PRESETS.map((preset) => <button type="button" key={preset} disabled={disabled}
        className={Math.abs(preset - value) < 0.001 ? 'on' : ''} onClick={() => onChange(preset)}>{preset.toFixed(2)}</button>)}
    </div>
    {hint && <small className="confidence-hint">{hint}</small>}
  </div>
}

/** One side of the grid. The text being typed is kept as a draft (so the field can be emptied while retyping);
 *  only in-range whole numbers are committed live, and the draft is clamped to 1–32 when the field loses focus. */
function CellSideInput({ label, value, disabled, onCommit }: { label: string; value: number; disabled?: boolean; onCommit: (value: number) => void }) {
  const [draft, setDraft] = useState<string | null>(null)
  return <label className="cell-grid-field"><span>{label}</span>
    <input type="number" min={1} max={CELL_GRID_MAX} step={1} value={draft ?? String(value)} disabled={disabled}
      aria-label={`Số ${label.toLowerCase()} của lưới`}
      onChange={(event) => {
        setDraft(event.target.value)
        const typed = event.target.valueAsNumber
        if (Number.isInteger(typed) && typed >= 1 && typed <= CELL_GRID_MAX) onCommit(typed)
      }}
      onBlur={() => {
        if (draft !== null && draft.trim() !== '' && Number.isFinite(Number(draft))) onCommit(clampCellSide(Number(draft)))
        setDraft(null)
      }} /></label>
}

/** Switch + columns x rows for the cell gate: an alarm is confirmed only where its grid cell holds sustained drowning votes. */
export function CellGridControl({ value, onChange, disabled, hint }: {
  value: CellGridChoice; onChange: (value: CellGridChoice) => void; disabled?: boolean; hint?: ReactNode
}) {
  const side = (label: string, key: 'cols' | 'rows') => <CellSideInput label={label} value={value[key]} disabled={disabled || !value.enabled}
    onCommit={(next) => onChange({ ...value, [key]: next })} />
  return <div className="confidence-control cell-grid-control" role="group" aria-label="Tự chia vùng">
    <div className="cell-grid-row">
      <label className="switch"><input type="checkbox" checked={value.enabled} disabled={disabled}
        onChange={(event) => onChange({ ...value, enabled: event.target.checked })} /><span />Tự chia vùng</label>
      {side('Cột', 'cols')}<span className="muted">×</span>{side('Hàng', 'rows')}
    </div>
    <small className="confidence-hint">{hint ?? (value.enabled
      ? `Khung hình chia thành lưới ${value.cols}×${value.rows}. Cảnh báo đuối nước chỉ được xác nhận khi ô người đó đứng có phiếu drowning liên tục; người bơi lướt qua nhiều ô không bị báo.`
      : 'Đã tắt: cảnh báo dựa trên từng người, không kiểm tra theo vùng. Người cứu hộ bơi nhanh dễ bị báo nhầm hơn.')}</small>
  </div>
}

export function ClassScores({ scores }: { scores?: number[] | null }) {
  return <span className="class-scores">{SCORE_LABELS.map((label, index) =>
    <span key={label} className="class-score"><span>{label}</span><b>{scoreText(scores?.[index])}</b></span>)}</span>
}

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
