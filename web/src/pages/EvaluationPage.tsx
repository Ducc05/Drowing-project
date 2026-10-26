import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import type { RefObject } from 'react'
import { api, fmt, newKey, pct, post, request, useAction, when } from '../api'
import type { JobSummary } from '../api'
import { Banner, Matrix, Panel, Tag } from '../ui'

const LABELS = ['drowning', 'swimming', 'out_of_water'] as const
type Label = (typeof LABELS)[number]
const labelText: Record<Label, string> = { drowning: 'Đuối nước', swimming: 'Bình thường trong nước', out_of_water: 'Ngoài nước' }
const labelColor: Record<string, string> = { drowning: '#d86a63', swimming: '#5aa6d8', out_of_water: '#8fb78f' }

type Metrics = { accuracy?: number; macro_f1?: number; per_class_recall?: Record<string, number>; confusion_matrix?: number[][]; support?: unknown }
type ModelRow = {
  id: string; display_version: string; status: string; model_kind: 'temporal' | 'image_tracked' | null; temporal_trained: boolean; completion_kind: string
  metrics: { image_validation?: Metrics; temporal_validation?: Metrics; device?: string }; cached_locally: boolean; created_at: string
  checkpoint: { drive_url: string | null; archive_sha256: string; weights_sha256?: string | null; size_bytes: number; status: string } | null
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
type Detection = { track_id: string; timestamp_ms: number; frame_index: number; bbox_xyxy: number[]; label: Label | null; class_scores: number[] | null; prediction_branch: string }
type VideoSummary = { detection_count?: number; frame_count?: number; processed_frames?: number; frame_stride?: number; width?: number; height?: number; fps_metadata?: number; max_gap_ms?: number; render?: 'overlay' | 'video' }
type MediaResult = { job_id: string; state: string; model_version_id: string; files: Record<string, string>; summary?: VideoSummary }
type VideoJob = JobSummary & { mode?: string | null; model_version_id?: string | null; idempotency_key?: string; frame_stride?: number; progress?: { frame: number; total: number; processed: number; elapsed_s: number; eta_s: number | null } | null }
type Segment = { label: Label; start: number; end: number; branch: string }

function VideoOverlay({ video, detections, summary }: { video: RefObject<HTMLVideoElement | null>; detections: Detection[]; summary?: VideoSummary }) {
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
            const label = left.label ?? 'unknown', color = labelColor[label] ?? '#fff'
            const [x1, y1, x2, y2] = box
            ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy)
            const classIndex = LABELS.indexOf(left.label as Label), score = left.class_scores && classIndex >= 0 ? left.class_scores[classIndex] : null
            const caption = `${trackId.replace('trk-', '#')}  ${left.label ? labelText[left.label] : label}${score == null ? '' : `  ${(score * 100).toFixed(0)}%`}`
            ctx.font = '14px system-ui'; const textWidth = ctx.measureText(caption).width, tx = x1 * sx, ty = Math.max(17, y1 * sy - 5)
            ctx.fillStyle = color; ctx.fillRect(tx, ty - 15, textWidth + 8, 19); ctx.fillStyle = '#101820'; ctx.fillText(caption, tx + 4, ty - 1)
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

function MetricsBlock({ title, metrics }: { title: string; metrics?: Metrics }) {
  if (!metrics) return <div className="metric-box"><h4>{title}</h4><p className="muted">Không có.</p></div>
  return <div className="metric-box"><h4>{title}</h4>
    <p>Độ chính xác <b>{pct(metrics.accuracy)}</b> · Macro-F1 <b>{pct(metrics.macro_f1)}</b></p>
    <p className="muted">Recall: {LABELS.map((l) => `${labelText[l]} ${pct(metrics.per_class_recall?.[l])}`).join(' · ')}</p>
    {metrics.confusion_matrix && <Matrix labels={LABELS.map((l) => l)} rows={metrics.confusion_matrix} rowTitle="thật" colTitle="dự đoán" />}</div>
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
  const [detections, setDetections] = useState<Detection[]>([])
  const [summary, setSummary] = useState<VideoSummary>()
  const [marks, setMarks] = useState<Mark[]>([])
  const [track, setTrack] = useState('')
  const [start, setStart] = useState<number | null>(null)
  const [end, setEnd] = useState<number | null>(null)
  const [verdict, setVerdict] = useState<'correct' | 'wrong'>('correct')
  const [expected, setExpected] = useState<Label | ''>('')
  const [note, setNote] = useState('')
  const action = useAction()
  const setActionError = action.setError

  const loadMarks = useCallback(async () => { setMarks((await request<{ marks: Mark[] }>(`/api/v1/jobs/${jobId}/review-marks`)).marks) }, [jobId])
  useEffect(() => {
    request<{ detections: Detection[] }>(`/api/v1/jobs/${jobId}/results/detections`).then((r) => setDetections(r.detections)).catch((e) => setActionError(String(e)))
    request<{ summary?: VideoSummary }>(`/api/v1/jobs/${jobId}/results`).then((r) => setSummary(r.summary)).catch((e) => setActionError(String(e)))
    void loadMarks().catch((e) => setActionError(String(e)))
  }, [jobId, loadMarks, setActionError])

  const segments = useMemo(() => buildSegments(detections), [detections])
  const tracks = Object.keys(segments).sort()
  const current = () => Math.round((video.current?.currentTime ?? 0) * 1000)
  const duration = Math.max(1, ...Object.values(segments).flat().map((s) => s.end))
  const chosenTrack = track || tracks[0] || ''
  const save = () => action.run(async () => {
    await post(`/api/v1/jobs/${jobId}/review-marks`, {
      reviewer, track_id: chosenTrack, start_ms: start, end_ms: end, verdict, expected_label: expected || null, note })
    setNote(''); setStart(null); setEnd(null)
    await loadMarks(); onChanged()
    return 'Đã ghi đánh dấu.'
  })
  const remove = (id: string) => action.run(async () => { await request(`/api/v1/jobs/${jobId}/review-marks/${id}`, { method: 'DELETE' }); await loadMarks(); onChanged() })
  const pick = (trackId: string, segment: Segment) => {
    setTrack(trackId); setStart(segment.start); setEnd(segment.end)
    if (video.current) video.current.currentTime = segment.start / 1000
  }

  return <div className="review-area">
    <Banner error={action.error} notice={action.notice} />
    <div className="review-layout">
      <div className="video-shell"><video ref={video} controls src={api(`/api/v1/jobs/${jobId}/results/${summary?.render === 'overlay' ? 'source' : 'annotated'}`)} />
        <VideoOverlay video={video} detections={detections} summary={summary} /></div>
      <div className="timeline-box">
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
    </div>
    <div className="mark-form">
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
  </div>
}

export default function EvaluationPage() {
  const [models, setModels] = useState<ModelRow[]>([])
  const [selected, setSelected] = useState<string>('')
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null)
  const [videoJobs, setVideoJobs] = useState<VideoJob[]>([])
  const [jobId, setJobId] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [key, setKey] = useState<string | null>(null)
  const [activeVideoJob, setActiveVideoJob] = useState<VideoJob | null>(null)
  const [stopPending, setStopPending] = useState(false)
  const [reviewer, setReviewer] = useState(() => localStorage.getItem('reviewer') ?? '')
  const action = useAction()
  const setActionError = action.setError

  const loadModels = useCallback(async () => {
    const [list, jobs] = await Promise.all([request<{ models: ModelRow[] }>('/api/v1/models'), request<{ jobs: VideoJob[] }>('/api/v1/jobs')])
    setModels(list.models)
    setVideoJobs(jobs.jobs.filter((job) => job.job_type === 'inference' && job.mode === 'video' && ['completed', 'sync_pending'].includes(job.state)))
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
    if (!key) return
    let alive = true
    const poll = async () => {
      try {
        const rows = (await request<{ jobs: VideoJob[] }>('/api/v1/jobs')).jobs
        const match = rows.find((job) => job.idempotency_key === key)
        if (alive && match) setActiveVideoJob(match)
      } catch { /* the upload request reports the primary failure */ }
    }
    void poll()
    const timer = window.setInterval(() => void poll(), 2000)
    return () => { alive = false; window.clearInterval(timer) }
  }, [key])

  const model = models.find((m) => m.id === selected)
  const modelJobs = videoJobs.filter((job) => job.model_version_id === selected)

  const sync = () => action.run(async () => { await post(`/api/v1/models/${selected}/sync`); await loadModels(); return 'Đã tải và kiểm tra model về máy (checksum và danh tính khớp registry).' })
  const runVideo = () => action.run(async () => {
    if (!file) return
    const k = key ?? newKey('eval-video')
    setKey(k)
    const form = new FormData()
    form.append('file', file)
    let result: MediaResult
    try {
      result = await request<MediaResult>(`/api/v1/test/video?model_version_id=${selected}&frame_stride=3`, { method: 'POST', headers: { 'Idempotency-Key': k }, body: form })
    } catch (error) {
      const detail = error && typeof error === 'object' && 'detail' in error ? (error as { detail?: { state?: string } }).detail : undefined
      if (detail?.state === 'cancelled') {
        setActiveVideoJob((job) => job ? { ...job, state: 'cancelled' } : job)
        return 'Đã dừng xử lý video; ô compute đã được nhả.'
      }
      throw error
    } finally { setKey(null) }
    setFile(null); setJobId(result.job_id); setActiveVideoJob(null)
    await loadModels(); await loadEvaluation()
    return `Xử lý xong ${fmt(result.summary?.frame_count)} khung hình. Hãy xem video và đánh dấu đoạn đúng/sai.`
  })
  const review = (decision: 'approve' | 'reject') => action.run(async () => {
    await post(`/api/v1/models/${selected}/review`, { decision, reviewer_id: reviewer, comment: '' })
    await loadModels(); await loadEvaluation()
    return decision === 'approve' ? 'Đã duyệt model làm model chính.' : 'Đã từ chối model; phiên bản vẫn được lưu.'
  })
  const stopVideo = async () => {
    if (!activeVideoJob) return
    setStopPending(true)
    try {
      const result = await post<{ message: string }>(`/api/v1/test/jobs/${activeVideoJob.id}/stop`)
      action.setNotice(result.message)
    } catch (error) { action.setError(error instanceof Error ? error.message : String(error)) }
    finally { setStopPending(false) }
  }

  return <div className="page-stack">
    <header className="page-head"><div className="eyebrow"><span className="eyebrow-line" /> ĐÁNH GIÁ</div><h1>Xem model làm việc trên video thật</h1>
      <p className="subtitle">Model ứng viên không tự trở thành model chính. Bạn chạy thử trên video của mình, đánh dấu đoạn đúng/sai rồi mới quyết định duyệt hoặc từ chối.</p></header>
    <Banner error={action.error} notice={action.notice} />
    <Panel kicker="MODEL" title="Phiên bản model" subtitle="Số liệu do Kaggle báo trên tập validation; không phải kết quả trên video của bạn."
      right={<label className="field inline">Tên người duyệt<input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="Tên của bạn" /></label>}>
      {models.length === 0 ? <div className="empty-state">Chưa có model nào. Hoàn tất bước train ở trang Train để có model ứng viên.</div> : <>
        <div className="model-tabs">{models.map((m) => <button key={m.id} className={`model-tab ${m.id === selected ? 'on' : ''}`} onClick={() => { setSelected(m.id); setJobId('') }}>
          <b>{m.display_version}</b><span>{m.status} · {m.model_kind === 'image_tracked' ? 'ảnh + gộp theo ID' : m.model_kind === 'temporal' ? 'temporal' : 'chưa xác định loại'}</span></button>)}</div>
        {model && <>
          <div className="stat-row">
            <div><b>{model.status}</b><span>trạng thái · tạo {when(model.created_at)}</span></div>
            <div><b>{model.cached_locally ? 'có' : 'chưa'}</b><span>đã tải về máy này</span></div>
            <div><b>{fmt(evaluation?.test_runs.videos)}</b><span>video đã chạy thử</span></div>
            <div><b>{fmt(evaluation?.human_review.segments)}</b><span>đoạn bạn đã đánh dấu</span></div>
          </div>
          {model.checkpoint && <div className="callout info checkpoint-box">
            <b>Checkpoint dùng chung</b> (mọi máy tải đúng file này và được kiểm tra mã băm trước khi dùng):
            {model.checkpoint.drive_url && <> <a href={model.checkpoint.drive_url} target="_blank" rel="noreferrer">mở trên Drive</a></>}
            <div>SHA-256 gói: <code>{model.checkpoint.archive_sha256}</code> · {fmt(Math.round(model.checkpoint.size_bytes / 1024))} KiB · trạng thái {model.checkpoint.status}</div>
            {model.checkpoint.weights_sha256 && <div>SHA-256 trọng số: <code>{model.checkpoint.weights_sha256}</code></div>}
          </div>}
          <div className="metric-pair"><MetricsBlock title="Validation — nhánh ảnh" metrics={model.metrics.image_validation} />{model.model_kind === 'temporal' && <MetricsBlock title="Validation — nhánh temporal" metrics={model.metrics.temporal_validation} />}</div>
          <p className="muted">{evaluation?.training_metrics_note}</p>
          <div className="toolbar">
            <button className="secondary-button big" disabled={action.busy || model.cached_locally} onClick={() => void sync()}>{model.cached_locally ? 'Đã có trên máy' : 'Tải model về máy để thử'}</button>
            <label className="field inline">Video thử (tối đa 512 MiB)<input type="file" accept=".mp4,.avi,.mov,.mkv,.webm" onChange={(e: ChangeEvent<HTMLInputElement>) => { setFile(e.target.files?.[0] ?? null); setKey(null) }} /></label>
            <button className="primary-button" disabled={!model.cached_locally || !file || action.busy} onClick={() => void runVideo()}>{action.busy ? 'Đang xử lý…' : 'Chạy thử với model này'}</button>
          </div>
          {key && <div className="callout info progress-box">
            <div><b>{activeVideoJob?.state === 'queued' ? 'Đang chờ ô compute' : 'Đang xử lý video'}</b>
              {activeVideoJob?.progress && <> · detector đã xử lý {fmt(activeVideoJob.progress.processed)}/{fmt(Math.ceil(activeVideoJob.progress.total / (activeVideoJob.frame_stride || 3)))} khung
                ({fmt(activeVideoJob.progress.frame)}/{fmt(activeVideoJob.progress.total)} khung video)
                {activeVideoJob.progress.eta_s != null && <> · còn khoảng {Math.ceil(activeVideoJob.progress.eta_s / 60)} phút</>}</>}</div>
            <progress max={activeVideoJob?.progress?.total || 1} value={activeVideoJob?.progress?.frame || 0} />
            {activeVideoJob && <button className="secondary-button big stop" disabled={stopPending || activeVideoJob.state === 'cancellation_requested'} onClick={() => void stopVideo()}>
              {stopPending || activeVideoJob.state === 'cancellation_requested' ? 'Đang dừng…' : 'Dừng'}</button>}
          </div>}
          {activeVideoJob?.state === 'cancelled' && <p className="muted">Đã dừng xử lý video; ô compute đã được nhả.</p>}
          {!model.cached_locally && <p className="muted">Cần tải model về máy trước, và cần PERSON_DETECTOR_WEIGHTS trong .env để chạy thử.</p>}
          {modelJobs.length > 0 && <div className="reopen">Video đã chạy: {modelJobs.map((job) => <button key={job.id} className={`secondary-button big ${job.id === jobId ? 'on' : ''}`} onClick={() => setJobId(job.id)}>{job.id.slice(0, 8)} · {when(job.created_at)}</button>)}</div>}
        </>}
      </>}
    </Panel>
    {jobId && <Panel kicker="XEM VÀ ĐÁNH DẤU" title={`Video ${jobId.slice(0, 8)}`} subtitle="Đánh dấu là nhận định của bạn về những đoạn bạn chọn; không phải benchmark.">
      <VideoReview key={jobId} jobId={jobId} reviewer={reviewer} onChanged={() => void loadEvaluation()} />
    </Panel>}
    {evaluation && evaluation.human_review.segments > 0 && <Panel kicker="TỔNG HỢP" title="Nhận định của bạn về model này" subtitle={evaluation.human_review.caveat}>
      <div className="stat-row">
        <div><b>{evaluation.human_review.correct}</b><span>đoạn đúng</span></div><div><b>{evaluation.human_review.wrong}</b><span>đoạn sai</span></div>
        <div><b>{(evaluation.human_review.total_reviewed_ms / 1000).toFixed(0)}s</b><span>tổng thời lượng đã xem</span></div>
        <div><b>{evaluation.test_runs.videos_with_review_marks}/{evaluation.test_runs.videos}</b><span>video có đánh dấu</span></div>
      </div>
      <Matrix labels={LABELS.map((l) => l)} rowTitle="bạn xác nhận" colTitle="model dự đoán"
        rows={LABELS.map((e) => LABELS.map((p) => evaluation.human_review.confusion_expected_by_predicted[e][p]))} />
      <p className="muted">Recall theo lớp (trên các đoạn đã đánh dấu): {LABELS.map((l) => `${labelText[l]} ${pct(evaluation.human_review.per_class[l].recall)} (${evaluation.human_review.per_class[l].reviewed_segments} đoạn)`).join(' · ')}</p>
    </Panel>}
    {model && model.status === 'candidate' && <Panel kicker="QUYẾT ĐỊNH" title="Duyệt hoặc từ chối model ứng viên" subtitle="Chỉ duyệt khi bạn đã xem model trên video thật. Từ chối không xóa phiên bản.">
      <div className="toolbar">
        <button className="primary-button" disabled={!reviewer.trim() || action.busy} onClick={() => void review('approve')}>Duyệt làm model chính</button>
        <button className="secondary-button big" disabled={!reviewer.trim() || action.busy} onClick={() => void review('reject')}>Từ chối</button>
        {!reviewer.trim() && <span className="muted">Điền tên người duyệt.</span>}
      </div>
      {evaluation && evaluation.decisions.length > 0 && <ul className="findings">{evaluation.decisions.map((d) => <li className="info" key={d.at}>{when(d.at)} · {d.reviewer}: {d.decision}</li>)}</ul>}
    </Panel>}
  </div>
}
