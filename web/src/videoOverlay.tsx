import { useEffect, useMemo, useRef, useState } from 'react'
import type { RefObject } from 'react'
import { api, fmt, request, SCORE_LABELS as LABELS, scoreText } from './api'
import { Icon } from './icons'
import { ConfidenceControl } from './ui'
import { ALERT_COLOR, alertSegments, boxScore, clock, coveredMs, labelColor, labelText, labelVi, trackName, trackStats } from './overlayData'
import { cellOverlayOf } from './overlayData'
import type { AlertSegment, CellOverlay, Detection, DetectionsResult, TrackStat, VideoSummary } from './overlayData'

/** Boxes drawn over the browser video. Every box carries its own score; ``minScore`` hides boxes below it (view only). */
export function VideoOverlay({ video, detections, summary, showBoxes = true, showAllLabels = true, minScore = 0, cells, showCells = true }: { video: RefObject<HTMLVideoElement | null>; detections: Detection[]; summary?: VideoSummary; showBoxes?: boolean; showAllLabels?: boolean; minScore?: number; cells?: CellOverlay; showCells?: boolean }) {
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
          // object-fit: contain — the picture is letterboxed inside the element.
          const scale = Math.min(bounds.width / element.videoWidth, bounds.height / element.videoHeight)
          const ox = (bounds.width - element.videoWidth * scale) / 2, oy = (bounds.height - element.videoHeight * scale) / 2
          const at = element.currentTime * 1000
          const step = 1000 * (summary.frame_stride ?? 3) / (summary.fps_metadata || 30)
          if (cells && showCells) {
            // Grid of the alert gate: red tint = this region currently holds sustained drowning votes.
            const gw = element.videoWidth * scale, gh = element.videoHeight * scale, cw = gw / cells.cols, ch = gh / cells.rows
            ctx.fillStyle = 'rgba(239, 68, 68, 0.28)'
            for (const { cell: [col, row], start, end } of cells.intervals) if (at >= start && at < end) ctx.fillRect(ox + col * cw, oy + row * ch, cw, ch)
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)'; ctx.lineWidth = 1; ctx.beginPath()
            for (let col = 1; col < cells.cols; col++) { ctx.moveTo(ox + col * cw, oy); ctx.lineTo(ox + col * cw, oy + gh) }
            for (let row = 1; row < cells.rows; row++) { ctx.moveTo(ox, oy + row * ch); ctx.lineTo(ox + gw, oy + row * ch) }
            ctx.stroke()
          }
          for (const [trackId, list] of showBoxes ? byTrack : []) {
            let left: Detection | undefined, right: Detection | undefined
            for (const row of list) { if (row.timestamp_ms <= at) left = row; if (row.timestamp_ms >= at) { right = row; break } }
            let box: number[] | undefined
            if (left && right && left !== right && right.timestamp_ms - left.timestamp_ms <= (summary.max_gap_ms ?? 1500)) {
              const amount = (at - left.timestamp_ms) / Math.max(1, right.timestamp_ms - left.timestamp_ms)
              box = left.bbox_xyxy.map((value, index) => value + (right!.bbox_xyxy[index] - value) * amount)
            } else if (left && right && left === right) box = left.bbox_xyxy
            else if (left && at - left.timestamp_ms <= step) box = left.bbox_xyxy
            if (!box || !left) continue
            const score = boxScore(left)
            if (score != null && score < minScore) continue
            const label = left.label ?? 'unknown', color = left.alert ? ALERT_COLOR : labelColor[label] ?? '#e4e9f0'
            const [x1, y1, x2, y2] = box.map((value, index) => value * scale + (index % 2 ? oy : ox))
            ctx.strokeStyle = color; ctx.lineWidth = left.alert ? 3.5 : 2
            ctx.beginPath(); ctx.roundRect(x1, y1, x2 - x1, y2 - y1, 3); ctx.stroke()
            const scoreTail = score == null ? '' : ` ${score.toFixed(2)}`
            const caption = left.alert ? `${trackName(trackId)} · ĐUỐI NƯỚC${scoreTail}`
              : showAllLabels ? `${trackName(trackId)} · ${left.label ? labelText[left.label] : label}${scoreTail}`
              : `${trackName(trackId)}${scoreTail}`
            ctx.font = `${left.alert ? 700 : 600} 12px 'Be Vietnam Pro', 'Segoe UI', sans-serif`
            const textWidth = ctx.measureText(caption).width, tx = x1, ty = Math.max(20, y1 - 4)
            ctx.fillStyle = color; ctx.beginPath(); ctx.roundRect(tx - 1, ty - 17, textWidth + 12, 19, 4); ctx.fill()
            ctx.fillStyle = '#fff'; ctx.fillText(caption, tx + 5, ty - 4)
          }
        }
      }
      if ('requestVideoFrameCallback' in element) frameHandle = element.requestVideoFrameCallback(draw)
      else raf = requestAnimationFrame(draw)
    }
    const resize = new ResizeObserver(draw); resize.observe(element); draw()
    return () => { alive = false; resize.disconnect(); cancelAnimationFrame(raf); if (frameHandle !== undefined && 'cancelVideoFrameCallback' in element) element.cancelVideoFrameCallback(frameHandle) }
  }, [detections, summary, video, showBoxes, showAllLabels, minScore, cells, showCells])
  return summary?.render === 'overlay' ? <canvas ref={canvas} className="video-overlay" aria-hidden="true" /> : null
}

export function AlertBanner({ tracks }: { tracks: string[] }) {
  if (tracks.length === 0) return null
  const who = tracks.length <= 3 ? tracks.map(trackName).join(', ') : `${tracks.length} người`
  return <div className="alert-banner" role="alert">
    <Icon name="alert" size={20} />
    <div><b>Cảnh báo đuối nước</b><span>Người {who} · hãy quan sát trực tiếp ngay</span></div>
  </div>
}

/** Video length: from the player once metadata is loaded, otherwise from the job summary. */
function summaryDurationMs(summary?: VideoSummary) {
  return summary?.frame_count && summary.fps_metadata ? (summary.frame_count / summary.fps_metadata) * 1000 : 0
}

export function VideoStats({ detections, segments, durationMs, summary }: { detections: Detection[]; segments: AlertSegment[]; durationMs: number; summary?: VideoSummary }) {
  const tracks = useMemo(() => new Set(detections.map((d) => d.track_id)).size, [detections])
  const alertMs = coveredMs(segments)
  return <div className="video-stats">
    <div className="vstat"><span>Thời lượng video</span><b>{durationMs ? clock(durationMs) : '—'}</b>
      <small>{summary?.processed_frames != null ? `${fmt(summary.processed_frames)} khung đã phân tích` : summary?.frame_count != null ? `${fmt(summary.frame_count)} khung` : ' '}</small></div>
    <div className="vstat"><span>ID tracker</span><b>{fmt(tracks)}</b><small>ID tạm, không phải số người</small></div>
    <div className={`vstat ${segments.length ? 'danger' : 'calm'}`}><span>Đoạn cảnh báo</span><b>{fmt(segments.length)}</b>
      <small>{segments.length ? 'cần người xem lại' : 'không có cảnh báo'}</small></div>
    <div className={`vstat ${segments.length ? 'danger' : 'calm'}`}><span>Tổng thời gian cảnh báo</span><b>{clock(alertMs)}</b>
      <small>gộp khoảng trùng, không cộng dồn</small></div>
    {summary?.processing_fps != null && <div className="vstat"><span>Tốc độ model</span><b>{summary.processing_fps.toFixed(1)} khung/giây</b>
      <small>video gốc {summary.fps_metadata != null ? `${+summary.fps_metadata.toFixed(2)} khung/giây` : '—'}, bước nhảy {summary.frame_stride ?? '—'} · đo trên máy chạy server</small></div>}
  </div>
}

/** Seek bar with every alert interval drawn in red and the current playhead. */
export function AlertTimeline({ segments, durationMs, playbackMs, onSeek }: { segments: AlertSegment[]; durationMs: number; playbackMs: number; onSeek: (ms: number) => void }) {
  const span = Math.max(1, durationMs, ...segments.map((s) => s.end))
  const at = (ms: number) => `${Math.min(100, (ms / span) * 100)}%`
  return <div className="alert-timeline">
    <div className="alert-track" onClick={(event) => {
      const box = event.currentTarget.getBoundingClientRect()
      onSeek(((event.clientX - box.left) / box.width) * span)
    }}>
      {segments.map((s, index) => <button key={`${s.track_id}-${s.start}`} className="alert-span" style={{ left: at(s.start), width: `max(6px, ${((s.end - s.start) / span) * 100}%)` }}
        title={`Đoạn ${index + 1}: ${clock(s.start)}–${clock(s.end)} · ${trackName(s.track_id)}`} aria-label={`Tua tới đoạn cảnh báo ${index + 1}`}
        onClick={(event) => { event.stopPropagation(); onSeek(s.start) }} />)}
      <span className="playhead" style={{ left: at(playbackMs) }} />
    </div>
    <div className="alert-timeline-scale"><span>0:00</span><span>{clock(playbackMs)}</span><span>{clock(span)}</span></div>
  </div>
}

/** One card per alert interval, with a seek button and an optional "mark this" action. */
export function AlertList({ segments, playbackMs, onSeek, onPick }: { segments: AlertSegment[]; playbackMs: number; onSeek: (ms: number) => void; onPick?: (segment: AlertSegment) => void }) {
  if (segments.length === 0) return <div className="alert-empty">
    <Icon name="check" size={22} />
    <b>Chưa ghi nhận dấu hiệu đuối nước</b>
    <span>Không có ID nào vượt ngưỡng cảnh báo trong video này.</span>
  </div>
  return <ol className="alert-cards">{segments.map((s, index) => {
    const live = playbackMs >= s.start && playbackMs <= s.end + 500
    return <li key={`${s.track_id}-${s.start}`} className={`alert-card ${live ? 'live' : ''}`}>
      <div className="alert-card-head">
        <b>Đoạn {index + 1} · {clock(s.start)} – {clock(s.end)}</b>
        <span className="pill pill-bad">drowning</span>
      </div>
      <div className="alert-card-meta">
        <span>Người <b>{trackName(s.track_id)}</b></span>
        <span>{((s.end - s.start) / 1000).toFixed(1)} giây</span>
        {s.peak != null && <span>điểm cao nhất <b>{scoreText(s.peak)}</b></span>}
      </div>
      {s.peak != null && <div className="score-meter" aria-hidden="true"><i style={{ width: `${Math.min(100, s.peak * 100)}%` }} /></div>}
      <div className="alert-card-actions">
        <button className="btn btn-small" onClick={() => onSeek(s.start)}><Icon name="play" size={14} /> Tua tới</button>
        {onPick && <button className="btn btn-small btn-ghost" onClick={() => onPick(s)}>Đánh dấu</button>}
      </div>
    </li>
  })}</ol>
}

/** View-only score filter for finished results: hides boxes below the value, never re-runs the model. */
export function ScoreFilter({ value, runThreshold, onChange }: { value: number; runThreshold?: number | null; onChange: (value: number) => void }) {
  return <ConfidenceControl title="Lọc hiển thị ô theo điểm" value={value} onChange={onChange}
    hint={<>Chỉ ẩn/hiện ô khi xem, không chạy lại model.{runThreshold != null && <> Lúc chạy đã dùng ngưỡng <b>{runThreshold.toFixed(2)}</b>: ô thấp hơn mức đó không tồn tại trong kết quả.</>} Cảnh báo không đổi theo bộ lọc này.</>} />
}

/** "Đánh giá theo từng ô" for video: one row per tracker ID with its own scores, label share and alert. */
export function TrackTable({ detections, minScore = 0, onSeek, onPick }: { detections: Detection[]; minScore?: number; onSeek?: (ms: number) => void; onPick?: (track: TrackStat) => void }) {
  const rows = useMemo(() => trackStats(detections), [detections])
  if (rows.length === 0) return <p className="muted">Không có ô nào được phát hiện trong video.</p>
  return <div className="table-scroll"><table className="data-table track-table">
    <thead><tr><th>ID</th><th>Nhãn chiếm đa số</th><th>Điểm TB</th><th>Điểm cao nhất</th><th>Điểm drowning cao nhất</th><th>Số khung</th><th>Thời gian</th><th>Cảnh báo</th><th /></tr></thead>
    <tbody>{rows.map((row) => <tr key={row.track_id} className={row.maxScore != null && row.maxScore < minScore ? 'dimmed' : row.alerted ? 'alert-row' : undefined}>
      <td className="mono">{trackName(row.track_id)}</td>
      <td>{row.label ? <><i className="dot" style={{ background: labelColor[row.label] }} />{labelText[row.label]} <small className="muted">{Math.round(row.labelShare * 100)}%</small></> : '—'}</td>
      <td><ScoreBar value={row.meanScore} /></td>
      <td>{scoreText(row.maxScore)}</td>
      <td>{row.maxDrowning == null ? '—' : <b className={row.maxDrowning >= 0.5 ? 'text-bad' : undefined}>{scoreText(row.maxDrowning)}</b>}</td>
      <td>{fmt(row.frames)}</td>
      <td className="mono">{clock(row.first)}–{clock(row.last)}</td>
      <td>{row.alerted ? <span className="pill pill-bad">có</span> : <span className="pill">không</span>}</td>
      <td className="row-actions">{onSeek && <button className="btn btn-small" onClick={() => onSeek(row.first)}>Tua tới</button>}
        {onPick && <button className="btn btn-small btn-ghost" onClick={() => onPick(row)}>Đánh dấu</button>}</td>
    </tr>)}</tbody></table></div>
}

/** Compact 0–1 score with a bar, coloured by level. */
export function ScoreBar({ value }: { value: number | null | undefined }) {
  if (value == null) return <span className="muted">—</span>
  const tone = value >= 0.7 ? 'high' : value >= 0.4 ? 'mid' : 'low'
  return <span className={`score-bar ${tone}`}><i><u style={{ width: `${Math.min(100, value * 100)}%` }} /></i><b>{value.toFixed(2)}</b></span>
}

export function LabelLegend() {
  return <div className="label-legend">{LABELS.map((label) => <span key={label}><i style={{ background: labelColor[label] }} />{labelVi[label]} <small>{label}</small></span>)}</div>
}

/** Result player for a finished video job: overlay boxes, alert banner, red timeline and the list of alert intervals. */
export function ResultVideo({ jobId, src }: { jobId: string; src: string }) {
  const video = useRef<HTMLVideoElement>(null)
  const [detections, setDetections] = useState<Detection[]>([])
  const [alertIntervals, setAlertIntervals] = useState<AlertSegment[]>()
  const [cells, setCells] = useState<CellOverlay>()
  const [showCells, setShowCells] = useState(true)
  const [summary, setSummary] = useState<VideoSummary>()
  const [playbackMs, setPlaybackMs] = useState(0)
  const [durationMs, setDurationMs] = useState(0)
  const [showBoxes, setShowBoxes] = useState(true)
  const [showAllLabels, setShowAllLabels] = useState(true)
  const [minScore, setMinScore] = useState<number | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    request<DetectionsResult>(`/api/v1/jobs/${jobId}/results/detections`).then((r) => { setDetections(r.detections); setAlertIntervals(r.alert_intervals); setCells(cellOverlayOf(r)) }).catch((e) => setError(String(e)))
    request<{ summary?: VideoSummary }>(`/api/v1/jobs/${jobId}/results`).then((r) => setSummary(r.summary)).catch((e) => setError(String(e)))
  }, [jobId])
  const segments = useMemo(() => alertIntervals ?? alertSegments(detections), [detections, alertIntervals])
  const active = [...new Set(segments.filter((s) => playbackMs >= s.start && playbackMs < s.end + (alertIntervals ? 0 : 500)).map((s) => s.track_id))]
  const seek = (ms: number) => { if (video.current) video.current.currentTime = ms / 1000 }
  const duration = durationMs || summaryDurationMs(summary)
  const viewThreshold = minScore ?? summary?.confidence_threshold ?? 0.05
  return <div className="result-video">
    <VideoStats detections={detections} segments={segments} durationMs={duration} summary={summary} />
    <div className="analysis-layout">
      <div className="analysis-player">
        <div className={`video-shell ${active.length ? 'alerting' : ''}`}>
          <video ref={video} controls src={api(src)}
            onLoadedMetadata={(event) => setDurationMs(event.currentTarget.duration * 1000 || 0)}
            onTimeUpdate={(event) => setPlaybackMs(event.currentTarget.currentTime * 1000)}
            onSeeked={(event) => setPlaybackMs(event.currentTarget.currentTime * 1000)} />
          <VideoOverlay video={video} detections={detections} summary={summary} showBoxes={showBoxes} showAllLabels={showAllLabels} minScore={viewThreshold} cells={cells} showCells={showCells} />
        </div>
        <AlertBanner tracks={active} />
        <AlertTimeline segments={segments} durationMs={duration} playbackMs={playbackMs} onSeek={seek} />
        <div className="player-foot">
          <LabelLegend />
          {summary?.render === "overlay" && <label className="switch"><input type="checkbox" checked={showAllLabels} onChange={(e) => setShowAllLabels(e.target.checked)} /><span />Hiện nhãn + điểm trên từng ô</label>}
          {summary?.render === 'overlay' && <label className="switch"><input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} /><span />Hiện khung</label>}
          {summary?.render === 'overlay' && cells && <label className="switch" title={`Lưới ${cells.cols}×${cells.rows}: chỉ vùng có phiếu drowning liên tục ${cells.window_s} giây mới được xác nhận cảnh báo`}><input type="checkbox" checked={showCells} onChange={(e) => setShowCells(e.target.checked)} /><span />Hiện lưới vùng (đỏ = vùng đang cảnh báo)</label>}
        </div>
        {summary?.render === 'overlay' && <ScoreFilter value={viewThreshold} runThreshold={summary?.confidence_threshold} onChange={setMinScore} />}
      </div>
      <aside className="analysis-side">
        <div className="side-head"><b>Các đoạn cảnh báo</b><span className={`pill ${segments.length ? 'pill-bad' : 'pill-ok'}`}>{segments.length}</span></div>
        {error ? <p className="muted">Không tải được detections: {error}</p>
          : <AlertList segments={segments} playbackMs={playbackMs} onSeek={seek} />}
        <p className="fine-print">Điểm 0–1 là đầu ra của model, chưa hiệu chuẩn, không phải xác suất đuối nước. Người trực vẫn cần tự quan sát trước khi can thiệp.</p>
      </aside>
    </div>
    <div className="track-section">
      <h4 className="block-title">Đánh giá theo từng ô (mỗi ID tracker)</h4>
      <p className="muted">Mỗi dòng là một ô được theo dõi qua thời gian, với điểm riêng của nó. Ô có điểm cao nhất dưới bộ lọc hiển thị được làm mờ.</p>
      <TrackTable detections={detections} minScore={viewThreshold} onSeek={seek} />
    </div>
  </div>
}
