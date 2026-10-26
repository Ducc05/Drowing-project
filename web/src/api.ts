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
