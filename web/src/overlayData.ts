import { useEffect, useMemo } from 'react'
import { SCORE_LABELS as LABELS } from './api'

export type Label = (typeof LABELS)[number]
export const labelText: Record<Label, string> = { drowning: 'drowning', swimming: 'swimming', out_of_water: 'out_of_water' }
/** Plain-language names shown next to the raw ontology label. */
export const labelVi: Record<Label, string> = { drowning: 'Dấu hiệu đuối nước', swimming: 'Đang bơi', out_of_water: 'Ngoài nước' }
export const labelColor: Record<string, string> = { drowning: '#ef4444', swimming: '#10b981', out_of_water: '#3b82f6' }
export const ALERT_COLOR = '#ff2d2d'

/** ``m:ss`` (or ``h:mm:ss``) for a millisecond offset. */
export function clock(ms: number) {
  const total = Math.max(0, Math.round(ms / 1000)), h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60
  return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${m}:${String(s).padStart(2, '0')}`
}

/** Total time covered by the intervals, counting overlaps once. */
export function coveredMs(intervals: Array<{ start: number; end: number }>) {
  let total = 0, cursor = -Infinity
  for (const { start, end } of [...intervals].sort((a, b) => a.start - b.start)) {
    const from = Math.max(start, cursor)
    if (end > from) total += end - from
    cursor = Math.max(cursor, end)
  }
  return total
}

export const trackName = (trackId: string) => trackId.replace('trk-', '#')

export type Detection = {
  label_mode?: "alarm" | "all"; track_id: string; timestamp_ms: number; frame_index: number; bbox_xyxy: number[]; label: Label | null; class_scores: number[] | null; prediction_branch: string; detector_confidence?: number; alert?: boolean }
export type VideoSummary = { detection_count?: number; frame_count?: number; processed_frames?: number; frame_stride?: number; width?: number; height?: number; fps_metadata?: number; max_gap_ms?: number; render?: 'overlay' | 'video' }
/** ``peak`` is the highest drowning score seen inside the interval (0–1, uncalibrated). */
export type AlertSegment = { track_id: string; start: number; end: number; peak: number | null }

/** Continuous alert intervals per tracker ID; a gap longer than ``maxGapMs`` starts a new interval. */
export function alertSegments(detections: Detection[], maxGapMs = 1500): AlertSegment[] {
  const open = new Map<string, AlertSegment>(), done: AlertSegment[] = []
  for (const row of [...detections].sort((a, b) => a.timestamp_ms - b.timestamp_ms)) {
    if (!row.alert) continue
    const score = row.class_scores?.[0] ?? null
    const current = open.get(row.track_id)
    if (current && row.timestamp_ms - current.end <= maxGapMs) {
      current.end = row.timestamp_ms
      if (score != null) current.peak = Math.max(current.peak ?? 0, score)
    } else {
      if (current) done.push(current)
      open.set(row.track_id, { track_id: row.track_id, start: row.timestamp_ms, end: row.timestamp_ms, peak: score })
    }
  }
  return [...done, ...open.values()].sort((a, b) => a.start - b.start)
}

export function useAlertSegments(detections: Detection[], intervals?: AlertSegment[]) {
  return useMemo(() => intervals ?? alertSegments(detections), [detections, intervals])
}

/** Repeating beep while ``active``; browsers only allow sound after the user has interacted with the page. */
export function useAlarmSound(active: boolean) {
  useEffect(() => {
    if (!active) return
    const AudioContextClass = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AudioContextClass) return
    const context = new AudioContextClass()
    const beep = () => {
      const oscillator = context.createOscillator(), gain = context.createGain()
      oscillator.frequency.value = 880; gain.gain.value = 0.15
      oscillator.connect(gain); gain.connect(context.destination)
      oscillator.start(); oscillator.stop(context.currentTime + 0.25)
    }
    beep()
    const timer = window.setInterval(beep, 1000)
    return () => { window.clearInterval(timer); void context.close() }
  }, [active])
}
