import { useEffect, useMemo, useRef, useState } from 'react'
import type { RefObject } from 'react'
import { api, request, SCORE_LABELS as LABELS, scoreText } from './api'
import { ALERT_COLOR, alertSegments, labelColor, labelText } from './overlayData'
import type { Detection, Label, VideoSummary } from './overlayData'

const trackName = (trackId: string) => trackId.replace('trk-', '#')

export function VideoOverlay({ video, detections, summary }: { video: RefObject<HTMLVideoElement | null>; detections: Detection[]; summary?: VideoSummary }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const element = video.current, layer = canvas.current
    if (!element || !layer || summary?.render !== 'overlay') return
    let raf = 0, frameHandle: number | undefined, alive = true
    const byTrack = new Map<string, Detection[]>()
    for (const record of detections) byTrack.set(record.track_id, [...(byTrack.get(record.track_id) ?? []), record])
    for (const list of byTrack.values()) list.sort((a, b) => a.timestamp_ms - b.timestamp_ms)
    const draw = () => {
      if (!alive) return
      const bounds = element.getBoundingClientRect(), ratio = window.devicePixelRatio || 1
      if (bounds.width && bounds.height) {
        if (layer.width !== Math.round(bounds.width * ratio) || layer.height !== Math.round(bounds.height * ratio)) {
          layer.width = Math.round(bounds.width * ratio); layer.height = Math.round(bounds.height * ratio)
        }
        const ctx = layer.getContext('2d')
        if (ctx && element.videoWidth && element.videoHeight) {
          ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, bounds.width, bounds.height)
          const sx = bounds.width / element.videoWidth, sy = bounds.height / element.videoHeight
          const at = element.currentTime * 1000
          const step = 1000 * (summary.frame_stride ?? 3) / (summary.fps_metadata || 30)
          for (const [trackId, list] of byTrack) {
            let left: Detection | undefined, right: Detection | undefined
            for (const row of list) { if (row.timestamp_ms <= at) left = row; if (row.timestamp_ms >= at) { right = row; break } }
            let box: number[] | undefined
            if (left && right && left !== right && right.timestamp_ms - left.timestamp_ms <= (summary.max_gap_ms ?? 1500)) {
              const amount = (at - left.timestamp_ms) / Math.max(1, right.timestamp_ms - left.timestamp_ms)
              box = left.bbox_xyxy.map((value, index) => value + (right!.bbox_xyxy[index] - value) * amount)
            } else if (left && right && left === right) box = left.bbox_xyxy
            else if (left && at - left.timestamp_ms <= step) box = left.bbox_xyxy
            if (!box || !left) continue
            const label = left.label ?? 'unknown', color = left.alert ? ALERT_COLOR : labelColor[label] ?? '#fff'
            const [x1, y1, x2, y2] = box
            ctx.strokeStyle = color; ctx.lineWidth = left.alert ? 4 : 2; ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)
            const classIndex = LABELS.indexOf(left.label as Label), score = left.class_scores && classIndex >= 0 ? left.class_scores[classIndex] : null
            const caption = left.alert ? `${trackName(trackId)}  CẢNH BÁO ĐUỐI NƯỚC`
              : `${trackName(trackId)}  ${left.label ? labelText[left.label] : label}${score == null ? '' : `  ${scoreText(score)}`}`
            ctx.font = left.alert ? 'bold 14px system-ui' : '14px system-ui'
            const textWidth = ctx.measureText(caption).width, tx = x1 * sx, ty = Math.max(17, y1 * sy - 5)
            ctx.fillStyle = color; ctx.fillRect(tx, ty - 15, textWidth + 8, 19); ctx.fillStyle = left.alert ? '#fff' : '#101820'; ctx.fillText(caption, tx + 4, ty - 1)
          }
        }
      }
      if ('requestVideoFrameCallback' in element) frameHandle = element.requestVideoFrameCallback(draw)
      else raf = requestAnimationFrame(draw)
    }
    const resize = new ResizeObserver(draw); resize.observe(element); draw()
    return () => { alive = false; resize.disconnect(); cancelAnimationFrame(raf); if (frameHandle !== undefined && 'cancelVideoFrameCallback' in element) element.cancelVideoFrameCallback(frameHandle) }
  }, [detections, summary, video])
  return summary?.render === 'overlay' ? <canvas ref={canvas} className="video-overlay" aria-hidden="true" /> : null
}

export function AlertBanner({ tracks }: { tracks: string[] }) {
  if (tracks.length === 0) return null
  const who = tracks.length <= 3 ? tracks.map(trackName).join(', ') : `${tracks.length} người`
  return <div className="alert-banner" role="alert"><span className="alert-dot" />CẢNH BÁO ĐUỐI NƯỚC · {who}</div>
}

/** Banner for the alert intervals that cover the current playback time, plus a clickable list of all of them. */
export function VideoAlerts({ detections, playbackMs, onSeek }: { detections: Detection[]; playbackMs: number; onSeek: (ms: number) => void }) {
  const segments = useMemo(() => alertSegments(detections), [detections])
  const active = segments.filter((s) => playbackMs >= s.start && playbackMs <= s.end + 500).map((s) => s.track_id)
  return <div className="video-alerts">
    <AlertBanner tracks={[...new Set(active)]} />
    {segments.length === 0
      ? <p className="muted">Không có cảnh báo đuối nước trong video này.</p>
      : <div className="alert-list"><b>{segments.length} lần cảnh báo:</b>{segments.map((s) =>
        <button key={`${s.track_id}-${s.start}`} className="alert-chip" onClick={() => onSeek(s.start)}>
          {trackName(s.track_id)} · {(s.start / 1000).toFixed(1)}–{(s.end / 1000).toFixed(1)}s</button>)}</div>}
  </div>
}

/** Result player for a finished video job: overlay boxes, alert banner and the list of alert intervals. */
export function ResultVideo({ jobId, src }: { jobId: string; src: string }) {
  const video = useRef<HTMLVideoElement>(null)
  const [detections, setDetections] = useState<Detection[]>([])
  const [summary, setSummary] = useState<VideoSummary>()
  const [playbackMs, setPlaybackMs] = useState(0)
  const [error, setError] = useState('')
  useEffect(() => {
    request<{ detections: Detection[] }>(`/api/v1/jobs/${jobId}/results/detections`).then((r) => setDetections(r.detections)).catch((e) => setError(String(e)))
    request<{ summary?: VideoSummary }>(`/api/v1/jobs/${jobId}/results`).then((r) => setSummary(r.summary)).catch((e) => setError(String(e)))
  }, [jobId])
  return <div className="result-video">
    <div className="video-shell"><video ref={video} controls src={api(src)}
      onTimeUpdate={(event) => setPlaybackMs(event.currentTarget.currentTime * 1000)}
      onSeeked={(event) => setPlaybackMs(event.currentTarget.currentTime * 1000)} />
      <VideoOverlay video={video} detections={detections} summary={summary} /></div>
    {error ? <p className="muted">Không tải được detections: {error}</p>
      : <VideoAlerts detections={detections} playbackMs={playbackMs}
        onSeek={(ms) => { if (video.current) video.current.currentTime = ms / 1000 }} />}
  </div>
}
