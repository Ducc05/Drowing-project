import { LiveVideoPreview } from '../LiveVideoPreview'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import { api, fmt, newKey, pct, post, request, SCORE_LABELS as LABELS, scoreText, useAction, when } from '../api'
import type { JobSummary } from '../api'
import { Banner, ClassScores, Matrix, Panel, Tag } from '../ui'
import { labelColor, labelText, trackName, useAlertSegments } from '../overlayData'
import { AlertBanner, AlertList, AlertTimeline, LabelLegend, VideoOverlay, VideoStats } from '../videoOverlay'
import type { AlertSegment, Detection, Label, VideoSummary } from '../overlayData'

type Metrics = { accuracy?: number; macro_f1?: number; per_class_recall?: Record<string, number>; confusion_matrix?: number[][]; support?: unknown }
type ModelRow = {
  id: string; display_version: string; status: string; model_kind: 'temporal' | 'image_tracked' | 'h20saver' | null; temporal_trained: boolean; completion_kind: string
  metrics: { image_validation?: Metrics; temporal_validation?: Metrics; device?: string }; cached_locally: boolean; created_at: string
  checkpoint: { drive_url: string | null; archive_sha256: string; weights_sha256?: string | null; size_bytes: number; status: string } | null
  source?: 'imported_h20saver' | 'kaggle_registry'
}
type PresenceMark = {
  id: string; start_ms: number; end_ms: number; people: number; expected_label: Label | null; note: string
  detected_boxes: number; detected_people_max: number; frames_with_detection: number; expected_frames: number
  detection_rate: number; predicted_label: Label | null
}
type PresenceSummary = {
  intervals: number; with_people: number; empty: number; interval_recall_any_person: number | null
  interval_recall_full_count: number | null; mean_frame_detection_rate: number | null
  empty_intervals_with_detections: number; empty_false_alarm_rate: number | null
  label_accuracy_when_detected: number | null; label_checked_intervals: number
}
type Mark = {
  id: string; reviewer: string; track_id: string; start_ms: number; end_ms: number; verdict: 'correct' | 'wrong'
  expected_label: Label | null; predicted_label: Label; frames: number; label_agreement: number; note: string
}
type MarkSummary = {
  segments: number; correct: number; wrong: number; wrong_without_expected_label: number
  confusion_expected_by_predicted: Record<Label, Record<Label, number>>
  per_class: Record<Label, { reviewed_segments: number; recall: number | null }>; total_reviewed_ms: number; caveat: string
}
type Evaluation = {
  training_metrics_note: string; test_runs: { videos: number; images: number; videos_with_review_marks: number }
  human_review: MarkSummary; decisions: Array<{ decision: string; reviewer: string; comment: string; at: string }>
}
type ImageDetection = { detection_id?: string; bbox_xyxy: number[]; label: Label | null; class_scores: number[] | null; prediction_branch: string; detector_confidence?: number }
type MediaResult = { job_id: string; state: string; modality: 'image' | 'video'; model_version_id: string; files: Record<string, string>; summary?: VideoSummary }
type VideoJob = JobSummary & { mode?: 'image' | 'video' | string | null; model_version_id?: string | null; idempotency_key?: string; frame_stride?: number; progress?: { frame: number; total: number; processed: number; elapsed_s: number; eta_s: number | null } | null }
type Segment = { label: Label; start: number; end: number; branch: string }

function MetricsBlock({ title, metrics }: { title: string; metrics?: Metrics }) {
  if (!metrics) return <div className="metric-box"><h4>{title}</h4><p className="muted">Không có.</p></div>
  return <div className="metric-box"><h4>{title}</h4>
    <p>Độ chính xác <b>{pct(metrics.accuracy)}</b> · Macro-F1 <b>{pct(metrics.macro_f1)}</b></p>
    <p className="muted">Recall: {LABELS.map((l) => `${labelText[l]} ${pct(metrics.per_class_recall?.[l])}`).join(' · ')}</p>
    {metrics.confusion_matrix && <Matrix labels={LABELS.map((l) => l)} rows={metrics.confusion_matrix} rowTitle="thật" colTitle="dự đoán" />}</div>
}

function ImageReview({ jobId }: { jobId: string }) {
  const [result, setResult] = useState<MediaResult | null>(null)
  const [detections, setDetections] = useState<ImageDetection[]>([])
  const [error, setError] = useState('')
  useEffect(() => {
    let alive = true
    Promise.all([
      request<MediaResult>(`/api/v1/jobs/${jobId}/results`),
      request<{ detections: ImageDetection[] }>(`/api/v1/jobs/${jobId}/results/detections`),
    ]).then(([media, rows]) => {
      if (alive) { setResult(media); setDetections(rows.detections) }
    }).catch((reason) => { if (alive) setError(String(reason)) })
    return () => { alive = false }
  }, [jobId])
  return <div className="image-review">
    {error && <Banner error={error} />}
    {result?.files.annotated && <img className="image-review-preview" src={api(result.files.annotated)} alt="Ảnh kết quả với hộp và nhãn dự đoán" />}
    <div><b>{fmt(detections.length)} đối tượng được phát hiện</b>
      <p className="muted">Điểm ba nhãn nằm trên thang 0–1, chưa phải xác suất đã hiệu chuẩn. Điểm YOLO tìm người được ghi riêng.</p>
      {detections.length > 0 && <ul className="image-detection-list">{detections.map((item, index) => {
        const label = item.label
        const scoreIndex = label ? LABELS.indexOf(label) : -1
        const score = item.class_scores && scoreIndex >= 0 ? item.class_scores[scoreIndex] : null
        const box = item.bbox_xyxy.map((value) => Math.round(value)).join(', ')
        return <li key={item.detection_id ?? index}>
          <b>{label ? labelText[label] : 'Chưa phân loại'}</b>{score != null && ` · điểm ${scoreText(score)}`}
          {item.detector_confidence != null && ` · YOLO tìm người ${scoreText(item.detector_confidence)}`}
          <span> · bbox [{box}] · {item.prediction_branch}</span>
          <ClassScores scores={item.class_scores} />
        </li>
      })}</ul>}
    </div>
  </div>
}

function buildSegments(detections: Detection[]): Record<string, Segment[]> {
  const byTrack: Record<string, Detection[]> = {}
  for (const item of detections) (byTrack[item.track_id] ??= []).push(item)
  const result: Record<string, Segment[]> = {}
  for (const [track, items] of Object.entries(byTrack)) {
    items.sort((a, b) => a.timestamp_ms - b.timestamp_ms)
    const gaps = items.slice(1).map((item, i) => item.timestamp_ms - items[i].timestamp_ms).sort((a, b) => a - b)
    const step = gaps.length ? gaps[Math.floor(gaps.length / 2)] : 100
    const segments: Segment[] = []
    items.forEach((item, index) => {
      if (!item.label) return
      const end = index + 1 < items.length ? items[index + 1].timestamp_ms : item.timestamp_ms + step
      const last = segments[segments.length - 1]
      if (last && last.label === item.label && last.branch === item.prediction_branch && item.timestamp_ms - last.end <= step * 2) last.end = end
      else segments.push({ label: item.label, start: item.timestamp_ms, end, branch: item.prediction_branch })
    })
    result[track] = segments
  }
  return result
}

function VideoReview({ jobId, reviewer, onChanged }: { jobId: string; reviewer: string; onChanged: () => void }) {
  const video = useRef<HTMLVideoElement>(null)
  const markForm = useRef<HTMLDivElement>(null)
  const [playbackMs, setPlaybackMs] = useState(0)
  const [durationMs, setDurationMs] = useState(0)
  const [showBoxes, setShowBoxes] = useState(true)
  const [showAllLabels, setShowAllLabels] = useState(false)
  const [detections, setDetections] = useState<Detection[]>([])
  const [alertIntervals, setAlertIntervals] = useState<AlertSegment[]>()
  const [summary, setSummary] = useState<VideoSummary>()
  const [marks, setMarks] = useState<Mark[]>([])
  const [track, setTrack] = useState('')
  const [start, setStart] = useState<number | null>(null)
  const [end, setEnd] = useState<number | null>(null)
  const [verdict, setVerdict] = useState<'correct' | 'wrong'>('correct')
  const [expected, setExpected] = useState<Label | ''>('')
  const [note, setNote] = useState('')
  const [pStart, setPStart] = useState<number | null>(null)
  const [pEnd, setPEnd] = useState<number | null>(null)
  const [pPeople, setPPeople] = useState(1)
  const [pLabel, setPLabel] = useState<Label | ''>('')
  const [pNote, setPNote] = useState('')
  const [pMarks, setPMarks] = useState<PresenceMark[]>([])
  const [pSummary, setPSummary] = useState<PresenceSummary | null>(null)
  const action = useAction()
  const setActionError = action.setError

  const loadPresence = useCallback(async () => {
    const r = await request<{ marks: PresenceMark[]; summary: PresenceSummary }>(`/api/v1/jobs/${jobId}/presence-marks`)
    setPMarks(r.marks); setPSummary(r.summary)
  }, [jobId])
  const loadMarks = useCallback(async () => { setMarks((await request<{ marks: Mark[] }>(`/api/v1/jobs/${jobId}/review-marks`)).marks) }, [jobId])
  useEffect(() => {
    request<{ detections: Detection[]; alert_intervals?: AlertSegment[] }>(`/api/v1/jobs/${jobId}/results/detections`).then((r) => { setDetections(r.detections); setAlertIntervals(r.alert_intervals) }).catch((e) => setActionError(String(e)))
    request<{ summary?: VideoSummary }>(`/api/v1/jobs/${jobId}/results`).then((r) => setSummary(r.summary)).catch((e) => setActionError(String(e)))
    void loadMarks().catch((e) => setActionError(String(e)))
    void loadPresence().catch((e) => setActionError(String(e)))
  }, [jobId, loadMarks, loadPresence, setActionError])

  const segments = useMemo(() => buildSegments(detections), [detections])
  const alerts = useAlertSegments(detections, alertIntervals)
  const alerting = [...new Set(alerts.filter((s) => playbackMs >= s.start && playbackMs < s.end + (alertIntervals ? 0 : 500)).map((s) => s.track_id))]
  const seek = (ms: number) => { if (video.current) video.current.currentTime = ms / 1000 }
  const samplesByTrack = useMemo(() => {
    const groups: Record<string, Detection[]> = {}
    for (const item of detections) (groups[item.track_id] ??= []).push(item)
    for (const rows of Object.values(groups)) rows.sort((a, b) => a.timestamp_ms - b.timestamp_ms)
    return groups
  }, [detections])
  const scoreRows = useMemo(() => {
    const step = 1000 * (summary?.frame_stride ?? 3) / (summary?.fps_metadata || 30)
    const gap = summary?.max_gap_ms ?? 1500
    const rows: Detection[] = []
    for (const items of Object.values(samplesByTrack)) {
      let low = 0, high = items.length
      while (low < high) { const middle = (low + high) >>> 1; if (items[middle].timestamp_ms <= playbackMs) low = middle + 1; else high = middle }
      const left = items[low - 1], right = items[low]
      if (left && ((right && right.timestamp_ms - left.timestamp_ms <= gap) || playbackMs - left.timestamp_ms <= step)) rows.push(left)
    }
    return rows
  }, [samplesByTrack, playbackMs, summary])
  const tracks = Object.keys(segments).sort()
  const current = () => Math.round((video.current?.currentTime ?? 0) * 1000)
  const duration = Math.max(1, durationMs, ...Object.values(segments).flat().map((s) => s.end))
  const chosenTrack = track || tracks[0] || ''
  const save = () => action.run(async () => {
    await post(`/api/v1/jobs/${jobId}/review-marks`, {
      reviewer, track_id: chosenTrack, start_ms: start, end_ms: end, verdict, expected_label: expected || null, note })
    setNote(''); setStart(null); setEnd(null)
    await loadMarks(); onChanged()
    return 'Đã ghi đánh dấu.'
  })
  const remove = (id: string) => action.run(async () => { await request(`/api/v1/jobs/${jobId}/review-marks/${id}`, { method: 'DELETE' }); await loadMarks(); onChanged() })
  const savePresence = () => action.run(async () => {
    await post(`/api/v1/jobs/${jobId}/presence-marks`, {
      reviewer, start_ms: pStart, end_ms: pEnd, people: pPeople, expected_label: pPeople > 0 ? (pLabel || null) : null, note: pNote })
    setPNote(''); setPStart(null); setPEnd(null)
    await loadPresence(); onChanged()
    return 'Đã ghi đoạn có người.'
  })
  const removePresence = (id: string) => action.run(async () => { await request(`/api/v1/jobs/${jobId}/presence-marks/${id}`, { method: 'DELETE' }); await loadPresence(); onChanged() })
  const pick = (trackId: string, segment: Segment) => {
    setTrack(trackId); setStart(segment.start); setEnd(segment.end)
    if (video.current) video.current.currentTime = segment.start / 1000
  }

  return <div className="review-area">
    <Banner error={action.error} notice={action.notice} />
    <VideoStats detections={detections} segments={alerts} durationMs={durationMs} summary={summary} />
    <div className="analysis-layout">
      <div className="analysis-player">
        <div className={`video-shell ${alerting.length ? 'alerting' : ''}`}><video ref={video} controls
          onLoadedMetadata={(event) => setDurationMs(event.currentTarget.duration * 1000 || 0)}
          onTimeUpdate={(event) => setPlaybackMs(event.currentTarget.currentTime * 1000)}
          onSeeked={(event) => setPlaybackMs(event.currentTarget.currentTime * 1000)}
          src={api(`/api/v1/jobs/${jobId}/results/${summary?.render === 'overlay' ? 'source' : 'annotated'}`)} />
          <VideoOverlay video={video} detections={detections} summary={summary} showBoxes={showBoxes} showAllLabels={showAllLabels} /></div>
        <AlertBanner tracks={alerting} />
        <AlertTimeline segments={alerts} durationMs={durationMs} playbackMs={playbackMs} onSeek={seek} />
        <div className="player-foot">
          <LabelLegend />
          {summary?.render === "overlay" && <label className="switch"><input type="checkbox" checked={showAllLabels} onChange={(e) => setShowAllLabels(e.target.checked)} /><span />Hiện mọi nhãn để kiểm tra</label>}
          {summary?.render === 'overlay' && <label className="switch"><input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} /><span />Hiện khung</label>}
        </div>
      </div>
      <aside className="analysis-side">
        <div className="side-head"><b>Các đoạn cảnh báo</b><span className={`pill ${alerts.length ? 'pill-bad' : 'pill-ok'}`}>{alerts.length}</span></div>
        <AlertList segments={alerts} playbackMs={playbackMs} onSeek={seek} onPick={(s) => {
          setTrack(s.track_id); setStart(s.start); setEnd(s.end); seek(s.start)
          markForm.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
        }} />
        <p className="fine-print">“Đánh dấu” điền sẵn người và khoảng thời gian vào form bên dưới; bạn tự chọn model đúng hay sai. Nhãn AI và xác nhận của bạn được lưu riêng.</p>
      </aside>
    </div>
    <div className="review-layout">
      <div className="timeline-box">
        <h4 className="block-title">Nhãn theo từng người</h4>
        <p className="muted">Mỗi dòng là một người do tracker gán ID tạm (không phải danh tính đã xác minh). Màu là nhãn model dự đoán; bấm vào một đoạn để chọn khoảng thời gian cần đánh dấu.</p>
        {tracks.length === 0 && <p className="muted">Không có phát hiện nào trong video.</p>}
        {tracks.map((id) => <div className="track-line" key={id}>
          <span className="track-id">{id.replace('trk-', '#')}</span>
          <div className="track-bar">{segments[id].map((segment) => <button key={`${segment.start}-${segment.label}`}
            className={`seg ${segment.branch === 'temporal' || segment.branch === 'image_tracked' ? '' : 'warmup'}`} title={`${labelText[segment.label]} · ${segment.branch} · ${(segment.start / 1000).toFixed(1)}–${(segment.end / 1000).toFixed(1)}s`}
            style={{ left: `${(segment.start / duration) * 100}%`, width: `${Math.max(0.6, ((segment.end - segment.start) / duration) * 100)}%`, background: labelColor[segment.label] }}
            onClick={() => pick(id, segment)} />)}</div>
        </div>)}
        <div className="legend">{LABELS.map((l) => <span key={l}><i style={{ background: labelColor[l] }} />{labelText[l]}</span>)}<span><i className="hatch" />dự đoán ảnh đơn (khởi động, chưa đủ chuỗi hoặc chưa đủ khung của người đó)</span></div>
      </div>
    <div className="score-panel">
      <h4 className="block-title">Điểm tại thời điểm đang xem</h4>
      <p className="muted">Điểm ba nhãn tại khung đã xử lý gần nhất, thời điểm {(playbackMs / 1000).toFixed(2)}s. Thang 0–1, chưa phải xác suất đã hiệu chuẩn.</p>
      <div className="table-scroll"><table className="data-table"><thead><tr><th>ID</th><th>Nhãn</th><th>YOLO tìm người</th><th>Điểm ba nhãn</th></tr></thead>
        <tbody>{scoreRows.map((row) => <tr key={row.track_id}><td className="mono">{trackName(row.track_id)}</td><td>{row.label ?? '—'}</td><td>{scoreText(row.detector_confidence)}</td><td><ClassScores scores={row.class_scores} /></td></tr>)}
          {scoreRows.length === 0 && <tr><td colSpan={4} className="muted">Không có detection tại thời điểm này.</td></tr>}</tbody></table></div>
    </div>
    </div>
    <div className="mark-form" ref={markForm}>
      <h4>Đánh dấu đoạn video</h4>
      <div className="mark-row">
        <label className="field">Người<select value={chosenTrack} onChange={(e) => setTrack(e.target.value)}>{tracks.map((t) => <option key={t} value={t}>{t}</option>)}</select></label>
        <div className="time-pick"><button className="secondary-button big" onClick={() => setStart(current())}>Đặt đầu</button><b>{start == null ? '—' : `${(start / 1000).toFixed(2)}s`}</b></div>
        <div className="time-pick"><button className="secondary-button big" onClick={() => setEnd(current())}>Đặt cuối</button><b>{end == null ? '—' : `${(end / 1000).toFixed(2)}s`}</b></div>
        <label className="field">Model dự đoán đúng không?<select value={verdict} onChange={(e) => setVerdict(e.target.value as 'correct' | 'wrong')}><option value="correct">Đúng</option><option value="wrong">Sai</option></select></label>
        {verdict === 'wrong' && <label className="field">Nhãn đúng phải là<select value={expected} onChange={(e) => setExpected(e.target.value as Label | '')}><option value="">(chưa rõ)</option>{LABELS.map((l) => <option key={l} value={l}>{labelText[l]}</option>)}</select></label>}
        <label className="field grow">Ghi chú<input value={note} onChange={(e) => setNote(e.target.value)} placeholder="Ví dụ: người đứng trên bờ" /></label>
        <button className="primary-button" disabled={!reviewer.trim() || !chosenTrack || start == null || end == null || end <= start || action.busy} onClick={() => void save()}>Ghi đánh dấu</button>
      </div>
      {!reviewer.trim() && <p className="muted">Điền tên người duyệt ở trên để ghi đánh dấu.</p>}
    </div>
    {marks.length > 0 && <table className="data-table"><thead><tr><th>Người</th><th>Đoạn</th><th>Model dự đoán</th><th>Bạn xác nhận</th><th>Ghi chú</th><th /></tr></thead>
      <tbody>{marks.map((mark) => <tr key={mark.id}><td>{mark.track_id}</td><td>{(mark.start_ms / 1000).toFixed(2)}–{(mark.end_ms / 1000).toFixed(2)}s ({mark.frames} khung)</td>
        <td>{labelText[mark.predicted_label]}</td><td>{mark.verdict === 'correct' ? <Tag tone="ok">đúng</Tag> : <Tag tone="bad">sai → {mark.expected_label ? labelText[mark.expected_label] : 'chưa rõ'}</Tag>}</td>
        <td>{mark.note}</td><td><button className="review-button" onClick={() => void remove(mark.id)}>Xóa</button></td></tr>)}</tbody></table>}
    <div className="mark-form">
      <h4>Đánh dấu có người (kể cả người model không thấy)</h4>
      <p className="muted">Để đo bỏ sót và báo nhầm: ghi "trong đoạn này có N người", không phụ thuộc kết quả model. Chọn 0 người cho đoạn trống. Chỉ cần ghi khoảng thời gian, không cần vẽ khung.</p>
      <div className="mark-row">
        <div className="time-pick"><button className="secondary-button big" onClick={() => setPStart(current())}>Đặt đầu</button><b>{pStart == null ? '—' : `${(pStart / 1000).toFixed(2)}s`}</b></div>
        <div className="time-pick"><button className="secondary-button big" onClick={() => setPEnd(current())}>Đặt cuối</button><b>{pEnd == null ? '—' : `${(pEnd / 1000).toFixed(2)}s`}</b></div>
        <label className="field">Số người thật<input type="number" min={0} max={50} value={pPeople} onChange={(e) => setPPeople(Math.max(0, Math.min(50, Number(e.target.value) || 0)))} /></label>
        {pPeople > 0 && <label className="field">Trạng thái đúng<select value={pLabel} onChange={(e) => setPLabel(e.target.value as Label | '')}><option value="">(chưa rõ)</option>{LABELS.map((l) => <option key={l} value={l}>{labelText[l]}</option>)}</select></label>}
        <label className="field grow">Ghi chú<input value={pNote} onChange={(e) => setPNote(e.target.value)} placeholder="Ví dụ: người ở xa, chỉ thấy cánh tay" /></label>
        <button className="primary-button" disabled={!reviewer.trim() || pStart == null || pEnd == null || pEnd <= pStart || action.busy} onClick={() => void savePresence()}>Ghi đoạn có người</button>
      </div>
    </div>
    {pSummary && pSummary.intervals > 0 && <div className="callout info">
      <b>Phát hiện người:</b> {pSummary.with_people} đoạn có người, {pSummary.empty} đoạn trống. Model thấy ít nhất 1 người ở <b>{pct(pSummary.interval_recall_any_person)}</b> số đoạn có người,
      thấy đủ số người ở <b>{pct(pSummary.interval_recall_full_count)}</b>, trung bình có phát hiện ở <b>{pct(pSummary.mean_frame_detection_rate)}</b> số khung.
      Đoạn trống mà model vẫn báo có người: <b>{pSummary.empty_intervals_with_detections}/{pSummary.empty}</b>.
      {pSummary.label_checked_intervals > 0 && <> Nhãn đúng khi có phát hiện: <b>{pct(pSummary.label_accuracy_when_detected)}</b> ({pSummary.label_checked_intervals} đoạn).</>}
      {' '}Chỉ tính các đoạn bạn đã ghi, không phải kết quả đo trên video chưa xem.</div>}
    {pMarks.length > 0 && <table className="data-table"><thead><tr><th>Đoạn</th><th>Có người thật</th><th>Model thấy</th><th>Khung có phát hiện</th><th>Nhãn đúng / dự đoán</th><th>Ghi chú</th><th /></tr></thead>
      <tbody>{pMarks.map((m) => <tr key={m.id}><td>{(m.start_ms / 1000).toFixed(2)}–{(m.end_ms / 1000).toFixed(2)}s</td>
        <td>{m.people}</td>
        <td>{m.people > 0 && m.detected_people_max === 0 ? <Tag tone="bad">bỏ sót</Tag> : m.people === 0 && m.detected_boxes > 0 ? <Tag tone="bad">báo nhầm {m.detected_boxes} hộp</Tag> : <Tag tone="ok">{m.detected_people_max} người</Tag>}</td>
        <td>{m.frames_with_detection}/{m.expected_frames} ({pct(m.detection_rate)})</td>
        <td>{m.expected_label ? labelText[m.expected_label] : '—'} / {m.predicted_label ? labelText[m.predicted_label] : '—'}</td>
        <td>{m.note}</td><td><button className="review-button" onClick={() => void removePresence(m.id)}>Xóa</button></td></tr>)}</tbody></table>}
  </div>
}

export default function EvaluationPage() {
  const [models, setModels] = useState<ModelRow[]>([])
  const [selected, setSelected] = useState<string>('')
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null)
  const [videoJobs, setVideoJobs] = useState<VideoJob[]>([])
  const [jobId, setJobId] = useState('')
  const [jobModality, setJobModality] = useState<'image' | 'video'>('video')
  const [file, setFile] = useState<File | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const mediaInput = useRef<HTMLInputElement>(null)
  const [key, setKey] = useState<string | null>(null)
  const [activeMediaJob, setActiveMediaJob] = useState<VideoJob | null>(null)
  const [stopPending, setStopPending] = useState(false)
  const [reviewer, setReviewer] = useState(() => localStorage.getItem('reviewer') ?? '')
  const action = useAction()
  const setActionError = action.setError

  const loadModels = useCallback(async () => {
    const [list, jobs] = await Promise.all([request<{ models: ModelRow[] }>('/api/v1/models'), request<{ jobs: VideoJob[] }>('/api/v1/jobs')])
    setModels(list.models)
    setVideoJobs(jobs.jobs.filter((job) => job.job_type === 'inference' && ['image', 'video'].includes(job.mode ?? '') && ['completed', 'sync_pending'].includes(job.state)))
    setSelected((current) => current || list.models[0]?.id || '')
  }, [])
  const loadEvaluation = useCallback(async () => {
    if (!selected) { setEvaluation(null); return }
    setEvaluation(await request<Evaluation>(`/api/v1/models/${selected}/evaluation`))
  }, [selected])
  useEffect(() => { void loadModels().catch((e) => setActionError(String(e))) }, [loadModels, setActionError])
  useEffect(() => { void loadEvaluation().catch((e) => setActionError(String(e))) }, [loadEvaluation, setActionError])
  useEffect(() => { localStorage.setItem('reviewer', reviewer) }, [reviewer])
  useEffect(() => {
    if (!key || jobModality !== 'video') return
    let alive = true
    const poll = async () => {
      try {
        const rows = (await request<{ jobs: VideoJob[] }>('/api/v1/jobs')).jobs
        const match = rows.find((job) => job.idempotency_key === key)
        if (alive && match && match.mode === 'video') setActiveMediaJob(match)
      } catch { /* the upload request reports the primary failure */ }
    }
    void poll()
    const timer = window.setInterval(() => void poll(), 2000)
    return () => { alive = false; window.clearInterval(timer) }
  }, [key, jobModality])

  const model = models.find((m) => m.id === selected)
  const modelJobs = videoJobs.filter((job) => job.model_version_id === selected)

  const sync = () => action.run(async () => { await post(`/api/v1/models/${selected}/sync`); await loadModels(); return 'Đã tải và kiểm tra model về máy (checksum và danh tính khớp registry).' })
  const acceptFile = (incoming: File | null) => {
    if (!incoming) return
    const ext = incoming.name.split('.').pop()?.toLowerCase() ?? ''
    const kind = incoming.type.startsWith('image/') || ['jpg', 'jpeg', 'png', 'webp', 'bmp'].includes(ext)
      ? 'image' : incoming.type.startsWith('video/') || ['mp4', 'avi', 'mov', 'mkv', 'webm'].includes(ext) ? 'video' : null
    if (!kind) { setFile(null); setKey(null); action.setError('Chọn ảnh JPG/PNG/WebP/BMP hoặc video MP4/AVI/MOV/MKV/WebM.'); return }
    setFile(incoming); setJobModality(kind); setJobId(''); setKey(null); setActiveMediaJob(null); action.setError('')
  }
  const runMedia = () => action.run(async () => {
    if (!file) return
    const k = key ?? newKey(`eval-${jobModality}`)
    setKey(k)
    const form = new FormData()
    form.append('file', file)
    let result: MediaResult
    try {
      const endpoint = jobModality === 'video'
        ? `/api/v1/test/video?model_version_id=${selected}&frame_stride=3`
        : `/api/v1/test/image?model_version_id=${selected}`
      result = await request<MediaResult>(endpoint, { method: 'POST', headers: { 'Idempotency-Key': k }, body: form })
    } catch (error) {
      const detail = error && typeof error === 'object' && 'detail' in error ? (error as { detail?: { state?: string } }).detail : undefined
      if (detail?.state === 'cancelled') {
        setActiveMediaJob((job) => job ? { ...job, state: 'cancelled' } : job)
        setKey(null)
        return 'Đã dừng xử lý video; ô compute đã được nhả.'
      }
      throw error
    }
    setKey(null)
    setFile(null); setJobId(result.job_id); setJobModality(result.modality); setActiveMediaJob(null)
    await loadModels(); await loadEvaluation()
    return result.modality === 'video'
      ? `Xử lý xong ${fmt(result.summary?.frame_count)} khung hình. Hãy xem video và đánh dấu đoạn đúng/sai.`
      : `Kiểm tra ảnh xong, phát hiện ${fmt(result.summary?.detection_count)} đối tượng.`
  })
  const review = (decision: 'approve' | 'reject') => action.run(async () => {
    await post(`/api/v1/models/${selected}/review`, { decision, reviewer_id: reviewer, comment: '' })
    await loadModels(); await loadEvaluation()
    return decision === 'approve' ? 'Đã duyệt model làm model chính.' : 'Đã từ chối model; phiên bản vẫn được lưu.'
  })
  const stopVideo = async () => {
    if (!activeMediaJob) return
    setStopPending(true)
    try {
      const result = await post<{ message: string }>(`/api/v1/test/jobs/${activeMediaJob.id}/stop`)
      action.setNotice(result.message)
    } catch (error) { action.setError(error instanceof Error ? error.message : String(error)) }
    finally { setStopPending(false) }
  }

  return <div className="page-stack">
    <header className="page-head"><div className="eyebrow">Đánh giá</div><h1>Xem model trên ảnh và video thật</h1>
      <p className="subtitle">Thử model đang dùng hoặc các phiên bản lịch sử trên ảnh và video; bạn có thể đánh dấu các đoạn đúng/sai để lưu nhận xét.</p></header>
    <Banner error={action.error} notice={action.notice} />
    <Panel kicker="Model" title="Phiên bản model" subtitle="best.pt là model đang dùng; các phiên bản Kaggle còn lại là lịch sử."
      right={<label className="field inline">Tên người duyệt<input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="Tên của bạn" /></label>}>
      {models.length === 0 ? <div className="empty-state">Chưa có model nào. Hoàn tất bước train ở trang Train để có model ứng viên.</div> : <>
        <div className="model-tabs">{models.map((m) => <button key={m.id} className={`model-tab ${m.id === selected ? 'on' : ''}`} onClick={() => { setSelected(m.id); setJobId('') }}>
          <b>{m.display_version}</b><span>{m.status === 'active' ? 'đang dùng' : `${m.status} · ${m.model_kind === 'image_tracked' ? 'ảnh + gộp theo ID' : m.model_kind === 'temporal' ? 'temporal' : 'chưa xác định loại'}`}</span></button>)}</div>
        {model && <>
          <div className="stat-row">
            <div><b>{model.status === 'active' ? 'đang dùng' : model.status}</b>{model.source !== 'imported_h20saver' && <span>trạng thái · {when(model.created_at)}</span>}</div>
            <div><b>{model.cached_locally ? 'có' : 'chưa'}</b><span>đã tải về máy này</span></div>
            <div><b>{fmt(evaluation?.test_runs.images)}</b><span>ảnh đã chạy thử</span></div>
            <div><b>{fmt(evaluation?.test_runs.videos)}</b><span>video đã chạy thử</span></div>
            <div><b>{fmt(evaluation?.human_review.segments)}</b><span>đoạn bạn đã đánh dấu</span></div>
          </div>
          {model.checkpoint && <div className="callout info checkpoint-box">
            <b>{model.source === 'imported_h20saver' ? 'File model đang dùng' : 'Checkpoint dùng chung'}</b>{model.source === 'imported_h20saver' ? ' · best.pt được nhập từ H20Saver và kiểm tra mã băm.' : ' (mọi máy tải đúng file này và được kiểm tra mã băm trước khi dùng):'}
            {model.checkpoint.drive_url && <> <a href={model.checkpoint.drive_url} target="_blank" rel="noreferrer">mở trên Drive</a></>}
            <div>SHA-256 gói: <code>{model.checkpoint.archive_sha256}</code> · {fmt(Math.round(model.checkpoint.size_bytes / 1024))} KiB · trạng thái {model.checkpoint.status}</div>
            {model.checkpoint.weights_sha256 && <div>SHA-256 trọng số: <code>{model.checkpoint.weights_sha256}</code></div>}
          </div>}
          {model.source !== 'imported_h20saver' && <div className="metric-pair"><MetricsBlock title="Validation — nhánh ảnh" metrics={model.metrics.image_validation} />{model.model_kind === 'temporal' && <MetricsBlock title="Validation — nhánh temporal" metrics={model.metrics.temporal_validation} />}</div>}
          <p className="muted">{evaluation?.training_metrics_note}</p>
          <div className="toolbar">
            {model.source === 'imported_h20saver'
              ? <button className="secondary-button big" disabled>Đang dùng best.pt</button>
              : <button className="secondary-button big" disabled={action.busy || model.cached_locally} onClick={() => void sync()}>{model.cached_locally ? 'Đã có trên máy' : 'Tải model về máy để thử'}</button>}
            <div className={`media-dropzone ${dragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
              role="group" aria-label="Chọn ảnh hoặc video để chạy thử"
              onDragOver={(event) => { event.preventDefault(); setDragOver(true) }}
              onDragLeave={(event) => { const target = event.relatedTarget; if (!(target instanceof Node) || !event.currentTarget.contains(target)) setDragOver(false) }}
              onDrop={(event) => { event.preventDefault(); setDragOver(false); acceptFile(event.dataTransfer.files[0] ?? null) }}>
              <input ref={mediaInput} className="visually-hidden-input" type="file" accept=".jpg,.jpeg,.png,.webp,.bmp,.mp4,.avi,.mov,.mkv,.webm"
                disabled={action.busy} onChange={(event: ChangeEvent<HTMLInputElement>) => { acceptFile(event.target.files?.[0] ?? null); event.currentTarget.value = '' }} />
              <div className="media-drop-copy"><b>{file ? file.name : 'Kéo ảnh hoặc video vào đây'}</b>
                <span>{file ? (jobModality === 'image' ? 'Ảnh' : 'Video') : 'JPG, PNG, WebP, BMP, MP4, AVI, MOV, MKV hoặc WebM · tối đa 512 MiB'}</span></div>
              <button type="button" className="secondary-button big" disabled={action.busy} onClick={() => mediaInput.current?.click()}>
                {file ? 'Chọn tệp khác' : 'Chọn tệp'}
              </button>
            </div>
            <button className="primary-button" disabled={!model.cached_locally || !file || action.busy} onClick={() => void runMedia()}>
              {action.busy ? 'Đang xử lý…' : `Chạy thử ${file ? (jobModality === 'image' ? 'ảnh' : 'video') : 'với model này'}`}</button>
          </div>
          {key && jobModality === 'image' && <div className="callout info">Đang kiểm tra ảnh…</div>}
          {key && jobModality === "video" && <LiveVideoPreview key={key} requestKey={key} />}
          {key && jobModality === 'video' && <div className="callout info progress-box">
            <div><b>{activeMediaJob?.state === 'queued' ? 'Đang chờ ô compute' : 'Đang xử lý video'}</b>
              {activeMediaJob?.progress && <> · detector đã xử lý {fmt(activeMediaJob.progress.processed)}/{fmt(Math.ceil(activeMediaJob.progress.total / (activeMediaJob.frame_stride || 3)))} khung
                ({fmt(activeMediaJob.progress.frame)}/{fmt(activeMediaJob.progress.total)} khung video)
                {activeMediaJob.progress.eta_s != null && <> · còn khoảng {Math.ceil(activeMediaJob.progress.eta_s / 60)} phút</>}</>}</div>
            <progress max={activeMediaJob?.progress?.total || 1} value={activeMediaJob?.progress?.frame || 0} />
            {activeMediaJob && <button className="secondary-button big stop" disabled={stopPending || activeMediaJob.state === 'cancellation_requested'} onClick={() => void stopVideo()}>
              {stopPending || activeMediaJob.state === 'cancellation_requested' ? 'Đang dừng…' : 'Dừng'}</button>}
          </div>}
          {activeMediaJob?.state === 'cancelled' && <p className="muted">Đã dừng xử lý video; ô compute đã được nhả.</p>}
          {!model.cached_locally && <p className="muted">Cần tải model về máy trước, và cần PERSON_DETECTOR_WEIGHTS trong .env để chạy thử.</p>}
          {modelJobs.length > 0 && <div className="reopen">Ảnh / video đã chạy: {modelJobs.map((job) => <button key={job.id} className={`secondary-button big ${job.id === jobId ? 'on' : ''}`} onClick={() => { setJobId(job.id); setJobModality(job.mode === 'image' ? 'image' : 'video') }}>{job.mode === 'image' ? 'Ảnh' : 'Video'} · {job.id.slice(0, 8)} · {when(job.created_at)}</button>)}</div>}
        </>}
      </>}
    </Panel>
    {jobId && <Panel kicker={jobModality === 'image' ? 'Kết quả ảnh' : 'Xem và đánh dấu'} title={`${jobModality === 'image' ? 'Ảnh' : 'Video'} ${jobId.slice(0, 8)}`} subtitle={jobModality === 'image' ? 'Ảnh đã gắn khung và nhãn dự đoán.' : 'Đánh dấu là nhận định của bạn về những đoạn video; không phải benchmark.'}>
      {jobModality === 'image' ? <ImageReview key={jobId} jobId={jobId} /> : <VideoReview key={jobId} jobId={jobId} reviewer={reviewer} onChanged={() => void loadEvaluation()} />}
    </Panel>}
    {evaluation && evaluation.human_review.segments > 0 && <Panel kicker="Tổng hợp" title="Nhận định của bạn về model này" subtitle={evaluation.human_review.caveat}>
      <div className="stat-row">
        <div><b>{evaluation.human_review.correct}</b><span>đoạn đúng</span></div><div><b>{evaluation.human_review.wrong}</b><span>đoạn sai</span></div>
        <div><b>{(evaluation.human_review.total_reviewed_ms / 1000).toFixed(0)}s</b><span>tổng thời lượng đã xem</span></div>
        <div><b>{evaluation.test_runs.videos_with_review_marks}/{evaluation.test_runs.videos}</b><span>video có đánh dấu</span></div>
      </div>
      <Matrix labels={LABELS.map((l) => l)} rowTitle="bạn xác nhận" colTitle="model dự đoán"
        rows={LABELS.map((e) => LABELS.map((p) => evaluation.human_review.confusion_expected_by_predicted[e][p]))} />
      <p className="muted">Recall theo lớp (trên các đoạn đã đánh dấu): {LABELS.map((l) => `${labelText[l]} ${pct(evaluation.human_review.per_class[l].recall)} (${evaluation.human_review.per_class[l].reviewed_segments} đoạn)`).join(' · ')}</p>
    </Panel>}
    {model && model.status === 'candidate' && <Panel kicker="Quyết định" title="Duyệt hoặc từ chối model ứng viên" subtitle="Chỉ duyệt khi bạn đã xem model trên video thật. Từ chối không xóa phiên bản.">
      <div className="toolbar">
        <button className="primary-button" disabled={!reviewer.trim() || action.busy} onClick={() => void review('approve')}>Duyệt làm model chính</button>
        <button className="secondary-button big" disabled={!reviewer.trim() || action.busy} onClick={() => void review('reject')}>Từ chối</button>
        {!reviewer.trim() && <span className="muted">Điền tên người duyệt.</span>}
      </div>
      {evaluation && evaluation.decisions.length > 0 && <ul className="findings">{evaluation.decisions.map((d) => <li className="info" key={d.at}>{when(d.at)} · {d.reviewer}: {d.decision}</li>)}</ul>}
    </Panel>}
  </div>
}
