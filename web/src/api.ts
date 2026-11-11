import { useCallback, useEffect, useRef, useState } from 'react'

export const api = (path: string) => `${import.meta.env.VITE_API_BASE_URL ?? ''}${path}`

export class ApiError extends Error {
  status: number
  detail: unknown
  constructor(message: string, status: number, detail: unknown) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

function describe(detail: unknown, fallback: string): string {
  if (typeof detail === 'string') return detail
  if (detail && typeof detail === 'object') {
    const record = detail as Record<string, unknown>
    if (Array.isArray(record.reasons)) return record.reasons.join('; ')
    if (typeof record.reason === 'string') return record.reason
    if (Array.isArray(detail)) return detail.map((item) => (item as { msg?: string }).msg ?? JSON.stringify(item)).join('; ')
    return JSON.stringify(detail)
  }
  return fallback
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  if (!(init?.body instanceof FormData) && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  const response = await fetch(api(path), { ...init, headers })
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new ApiError(describe(body?.detail, `HTTP ${response.status}`), response.status, body?.detail)
  }
  return response.json() as Promise<T>
}

export const post = <T,>(path: string, body?: unknown) =>
  request<T>(path, { method: 'POST', body: body === undefined ? undefined : JSON.stringify(body) })

export const fmt = (value: number | undefined | null) => new Intl.NumberFormat('vi-VN').format(value ?? 0)
export const pct = (value: number | undefined | null) => (value == null ? '—' : `${(value * 100).toFixed(1)}%`)
/** Raw model score on the 0–1 scale; separate from detector thresholds and metric percentages. */
export const scoreText = (value: number | undefined | null) => (value == null || !Number.isFinite(value) ? '—' : value.toFixed(3))
export const SCORE_LABELS = ['drowning', 'swimming', 'out_of_water'] as const
export const when = (value: string) => new Date(value).toLocaleString('vi-VN', { dateStyle: 'short', timeStyle: 'short' })
export const newKey = (prefix: string) => `${prefix}-${crypto.randomUUID()}`

/** Per-box confidence threshold accepted by the API (boxes scoring lower are not returned). */
export const CONFIDENCE_MIN = 0.05
export const CONFIDENCE_MAX = 0.95
export const CONFIDENCE_PRESETS = [0.25, 0.4, 0.5, 0.7] as const
const CONFIDENCE_KEY = 'confidence-threshold'
export const clampConfidence = (value: number) =>
  Math.round(Math.min(CONFIDENCE_MAX, Math.max(CONFIDENCE_MIN, value)) * 100) / 100

/** The threshold chosen by the user, shared by every page and kept across reloads. ``null`` = server default. */
export function useConfidence() {
  const [value, setValue] = useState<number | null>(() => {
    const stored = Number(localStorage.getItem(CONFIDENCE_KEY))
    return stored ? clampConfidence(stored) : null
  })
  const update = useCallback((next: number) => {
    const clamped = clampConfidence(next)
    localStorage.setItem(CONFIDENCE_KEY, String(clamped))
    setValue(clamped)
  }, [])
  return [value, update] as const
}

/** The cell-grid ("Tự chia vùng") choice for one run: switch plus columns x rows (each 1–32). */
export const CELL_GRID_MAX = 32
export type CellGridChoice = { enabled: boolean; cols: number; rows: number }
const CELL_GRID_KEY = 'cell-grid-choice'
export const clampCellSide = (value: number) => Math.min(CELL_GRID_MAX, Math.max(1, Math.round(Number.isFinite(value) ? value : 1)))

/** Server value ("6x4" / "off") -> control state. Unknown or missing text means the built-in 6×4, switched on. */
export function parseCellGrid(text?: string | null): CellGridChoice {
  const value = (text ?? '').trim().toLowerCase()
  const match = /^(\d+)x(\d+)$/.exec(value)
  if (match) return { enabled: true, cols: clampCellSide(Number(match[1])), rows: clampCellSide(Number(match[2])) }
  return { enabled: value !== 'off', cols: 6, rows: 4 }
}

/** Control state -> API value; ``fallback`` (no choice made yet) lets the server use its own default. */
export const cellGridParam = (choice: CellGridChoice | null, fallback?: string) =>
  choice === null ? fallback : choice.enabled ? `${choice.cols}x${choice.rows}` : 'off'

/** The grid chosen by the user, shared by every page and kept across reloads. ``null`` = server default. */
export function useCellGrid() {
  const [value, setValue] = useState<CellGridChoice | null>(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(CELL_GRID_KEY) ?? 'null')
      return stored && typeof stored === 'object'
        ? { enabled: Boolean(stored.enabled), cols: clampCellSide(Number(stored.cols)), rows: clampCellSide(Number(stored.rows)) } : null
    } catch { return null }
  })
  const update = useCallback((next: CellGridChoice) => {
    const clean = { enabled: next.enabled, cols: clampCellSide(next.cols), rows: clampCellSide(next.rows) }
    try { localStorage.setItem(CELL_GRID_KEY, JSON.stringify(clean)) } catch { /* private window: keep the choice in memory only */ }
    setValue(clean)
  }, [])
  return [value, update] as const
}

const MODEL_KEY = 'selected-model'

/** The model chosen by the user, shared by the Camera and Đánh giá pages and kept across reloads. ``null`` = none chosen yet. */
export function useChosenModel() {
  const [value, setValue] = useState<string | null>(() => { try { return localStorage.getItem(MODEL_KEY) } catch { return null } })
  const update = useCallback((id: string) => {
    try { localStorage.setItem(MODEL_KEY, id) } catch { /* private window: the choice is kept in memory only */ }
    setValue(id)
  }, [])
  return [value, update] as const
}

/** The chosen model if the list still has it, otherwise the active (default) one, otherwise the first. */
export const resolveModelId = (models: Array<{ id: string; status: string }>, chosen: string | null) =>
  models.find((m) => m.id === chosen)?.id ?? models.find((m) => m.status === 'active')?.id ?? models[0]?.id

/** Re-run `load` every `ms` while `active` is true (and once on mount). */
export function usePolling(load: () => Promise<void>, ms: number, active: boolean) {
  const saved = useRef(load)
  useEffect(() => { saved.current = load }, [load])
  useEffect(() => {
    void saved.current()
    if (!active) return
    const timer = window.setInterval(() => void saved.current(), ms)
    return () => window.clearInterval(timer)
  }, [ms, active])
}

export function useAction() {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const run = useCallback(async (work: () => Promise<string | void>) => {
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const message = await work()
      if (typeof message === 'string') setNotice(message)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally {
      setBusy(false)
    }
  }, [])
  return { busy, error, notice, run, setError, setNotice }
}

export type JobSummary = {
  id: string; job_type: string; state: string; phase: string; created_at: string; updated_at: string
  error_code?: string | null; error_summary?: string | null; remote_ref?: string | null; remote_version?: string | null
}
export type JobEvent = { sequence: number; from_state: string | null; to_state: string; event_type: string; created_at: string; details: Record<string, unknown> }
