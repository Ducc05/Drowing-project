import { useEffect } from 'react'
import { SCORE_LABELS as LABELS } from './api'

export type Label = (typeof LABELS)[number]
export const labelText: Record<Label, string> = { drowning: 'drowning', swimming: 'swimming', out_of_water: 'out_of_water' }
export const labelColor: Record<string, string> = { drowning: '#d86a63', swimming: '#5aa6d8', out_of_water: '#8fb78f' }
export const ALERT_COLOR = '#e5262d'

export type Detection = { track_id: string; timestamp_ms: number; frame_index: number; bbox_xyxy: number[]; label: Label | null; class_scores: number[] | null; prediction_branch: string; detector_confidence?: number; alert?: boolean }
export type VideoSummary = { detection_count?: number; frame_count?: number; processed_frames?: number; frame_stride?: number; width?: number; height?: number; fps_metadata?: number; max_gap_ms?: number; render?: 'overlay' | 'video' }
export type AlertSegment = { track_id: string; start: number; end: number }

/** Continuous alert intervals per tracker ID; a gap longer than ``maxGapMs`` starts a new interval. */
export function alertSegments(detections: Detection[], maxGapMs = 1500): AlertSegment[] {
  const open = new Map<string, AlertSegment>(), done: AlertSegment[] = []
  for (const row of [...detections].sort((a, b) => a.timestamp_ms - b.timestamp_ms)) {
    if (!row.alert) continue
    const current = open.get(row.track_id)
    if (current && row.timestamp_ms - current.end <= maxGapMs) current.end = row.timestamp_ms
    else {
      if (current) done.push(current)
      open.set(row.track_id, { track_id: row.track_id, start: row.timestamp_ms, end: row.timestamp_ms })
    }
  }
  return [...done, ...open.values()].sort((a, b) => a.start - b.start)
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
