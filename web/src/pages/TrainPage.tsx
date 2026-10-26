import { useCallback, useEffect, useRef, useState } from 'react'
import { fmt, newKey, post, request, useAction, usePolling, when } from '../api'
import type { JobEvent, JobSummary } from '../api'
import { Banner, Check, Panel, Tag } from '../ui'
import RunReportView from './RunReportView'

type Mode = 'prepare_only' | 'remote_smoke' | 'train_candidate'
type RunView = { job_id: string; state: string; created_at: string; summary: Record<string, unknown> | null } | null
type Preflight = {
  environment: Record<string, boolean | number | string>
  sources: Array<{ source_id: string; mapping_approved: boolean; groups_verified: boolean; modality: string }>
  stale_sources: string[]
  latest_prepare: RunView
  latest_smoke: RunView
  approved_model: { id: string; display_version: string } | null
  compute_slot_held_by: { job_id: string; job_type: string; mode: string | null; state: string } | null
  queued_kaggle_jobs: number
  modes: Record<Mode, { allowed: boolean; blocked_by: string[] }>
  temporal_window_count_before_split: number
}
type Supervisor = { running: boolean; ticks?: number; errors?: Record<string, string>; last_actions?: Array<{ job_id: string; action: string; state?: string }>; reason?: string }
type JobRow = JobSummary & { mode?: string | null; executor?: string | null; remote_status?: string | null }

const TERMINAL = new Set(['completed', 'failed', 'cancelled'])
function minutesSince(value: string) {
  const seconds = Math.max(0, Math.round((Date.now() - new Date(value).getTime()) / 1000))
  return seconds < 90 ? `${seconds} giây trước` : `${Math.round(seconds / 60)} phút trước`
}
const stateTone = (state: string) => (state === 'completed' ? 'ok' : state === 'failed' || state === 'cancelled' ? 'bad' : state === 'sync_pending' || state === 'remote_unconfirmed' ? 'warn' : 'info') as 'ok' | 'bad' | 'warn' | 'info'
const stateText: Record<string, string> = {
  queued: 'đang chờ gửi lên Kaggle', dispatching: 'đang gửi lên Kaggle', remote_unconfirmed: 'đã gửi, chưa xác nhận trạng thái từ Kaggle',
  running: 'đang chạy trên Kaggle', collecting: 'đang tải kết quả', validating: 'đang kiểm tra kết quả', syncing: 'đang đồng bộ lên Drive',
  sync_pending: 'xong phần tính toán, đang chờ đồng bộ', completed: 'hoàn tất', failed: 'thất bại', cancelled: 'đã hủy',
}

const STEPS: Array<{ mode: Mode; title: string; lead: string }> = [
  { mode: 'prepare_only', title: '1 · Chuẩn bị & kiểm tra dữ liệu trên Kaggle', lead: 'Quét, chuyển đổi, kiểm tra và tạo báo cáo ảnh mẫu trên Kaggle. Không train.' },
  { mode: 'remote_smoke', title: '2 · Kiểm tra GPU & phụ thuộc (smoke)', lead: 'Chạy mô hình thật trên dữ liệu ngẫu nhiên để xác nhận GPU, thư viện và gradient. Không tạo model.' },
  { mode: 'train_candidate', title: '3 · Train model ứng viên', lead: 'Train hai giai đoạn trên GPU Kaggle, tạo model ứng viên. Model chỉ được dùng sau khi bạn xem video thật và duyệt.' },
]

function Timeline({ jobId, active }: { jobId: string; active: boolean }) {
  const [events, setEvents] = useState<JobEvent[]>([])
  const load = useCallback(async () => { setEvents((await request<{ events: JobEvent[] }>(`/api/v1/jobs/${jobId}/events`)).events) }, [jobId])
  usePolling(load, 5000, active)
  return <ol className="timeline">{events.map((event) => <li key={event.sequence}>
    <time>{when(event.created_at)}</time><b>{event.to_state}</b><span>{event.event_type.replaceAll('_', ' ')}</span></li>)}</ol>
}

export default function TrainPage() {
  const [pre, setPre] = useState<Preflight | null>(null)
  const [jobs, setJobs] = useState<JobRow[]>([])
  const [supervisor, setSupervisor] = useState<Supervisor | null>(null)
  const [open, setOpen] = useState<string | null>(null)
  const [newVersion, setNewVersion] = useState(false)
  const [reason, setReason] = useState('')
  const [pretrained, setPretrained] = useState(false)
  const [imageEpochs, setImageEpochs] = useState(5)
  const [temporalEpochs, setTemporalEpochs] = useState(5)
  const [keys, setKeys] = useState<Partial<Record<Mode, string>>>({})
  const action = useAction()
  const setActionError = action.setError

  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)
  const loadFast = useCallback(async () => {
    const [j, s] = await Promise.allSettled([request<{ jobs: JobRow[] }>('/api/v1/jobs'), request<Supervisor>('/api/v1/supervisor/status')])
    if (j.status === 'fulfilled') setJobs(j.value.jobs.filter((job) => job.job_type === 'prepare' || (job.job_type === 'train')))
    if (s.status === 'fulfilled') setSupervisor(s.value)
    setUpdatedAt(new Date())
  }, [])
  const loadSlow = useCallback(async () => {
    try { setPre(await request<Preflight>('/api/v1/train/preflight')) } catch (reason) { setActionError(String(reason)) }
  }, [setActionError])
  const load = useCallback(async () => { await Promise.all([loadFast(), loadSlow()]) }, [loadFast, loadSlow])
  const anyActive = jobs.some((job) => !TERMINAL.has(job.state))
  const anyActiveRef = useRef(false)
  useEffect(() => { anyActiveRef.current = anyActive }, [anyActive])
  // Job state is re-read every 3 s; while something is active the UI also asks the supervisor to check Kaggle every 10 s.
  usePolling(loadFast, 3000, true)
  usePolling(loadSlow, 15000, true)
  const driveJobs = useCallback(async () => {
    if (!anyActiveRef.current) return
    try { await post('/api/v1/supervisor/tick') } catch { /* the next pass retries */ }
    await loadFast()
  }, [loadFast])
  usePolling(driveJobs, 10000, true)

  const submit = (mode: Mode) => action.run(async () => {
    const key = keys[mode] ?? newKey(mode)
    setKeys((k) => ({ ...k, [mode]: key }))
    const result = await post<{ status: string; message?: string }>('/api/v1/train/submit', {
      mode, new_version: newVersion, reason, idempotency_key: key, pretrained, image_epochs: imageEpochs, temporal_epochs: temporalEpochs })
    setKeys((k) => ({ ...k, [mode]: undefined }))
    if (result.status === 'queued') { try { await post('/api/v1/supervisor/tick') } catch { /* background loop will pick it up */ } }
    await load()
    return result.status === 'model_available' || result.status === 'same_training_exists' || result.status === 'already_active' ? (result.message ?? 'Đã có kết quả cho yêu cầu này.') : 'Đã xếp job vào hàng đợi. Tiến trình nền sẽ gửi lên Kaggle khi ô compute rảnh.'
  })
  const cancel = (id: string) => action.run(async () => { const r = await post<{ message?: string }>(`/api/v1/jobs/${id}/cancel`); await load(); return r.message })
  const tick = () => action.run(async () => { await post('/api/v1/supervisor/tick'); await load(); return 'Đã kiểm tra tiến độ các job ngay.' })

  const env = pre?.environment
  return <div className="page-stack">
    <header className="page-head"><div className="eyebrow"><span className="eyebrow-line" /> TRAIN</div><h1>Chạy trên Kaggle theo từng bước</h1>
      <p className="subtitle">Không bỏ qua bước nào: chuẩn bị dữ liệu → bạn duyệt → kiểm tra GPU → train ứng viên. Train thật chỉ chạy trên Kaggle, không có train local.</p></header>
    <Banner error={action.error} notice={action.notice} />

    <Panel kicker="MÔI TRƯỜNG" title="Điều kiện để chạy Kaggle" subtitle="Chỉ hiển thị đã/chưa cấu hình, không bao giờ hiển thị giá trị bí mật.">
      <ul className="checks">
        <Check ok={Boolean(env?.kaggle_cli_installed)} label="Đã cài Kaggle CLI" hint="pip install kaggle" />
        <Check ok={Boolean(env?.kaggle_credentials_present)} label="Có thông tin đăng nhập Kaggle" hint="Tạo token trong Kaggle → Settings → API, lưu vào ~/.kaggle" />
        <Check ok={Boolean(env?.kaggle_kernel_ref_valid)} label="KAGGLE_KERNEL_REF hợp lệ (owner/slug)" hint="Điền trong .env" />
        <Check ok={Number(env?.kaggle_dataset_sources) > 0} label={`Dataset Kaggle gắn vào notebook: ${env?.kaggle_dataset_sources ?? 0}`} hint="Điền KAGGLE_DATASET_SOURCES trong .env" />
        <Check ok={Boolean(env?.drive_folder_configured && env?.drive_token_file_present)} label="Google Drive (thư mục + token)" hint="Cần để xuất bản model ứng viên: drownctl auth drive" />
      </ul>
      <p className="muted">Chạy <code>drownctl doctor</code> trong terminal để xem danh sách đầy đủ và cách sửa.</p>
    </Panel>

    {STEPS.map((step) => {
      const mode = pre?.modes[step.mode]
      const last = step.mode === 'prepare_only' ? pre?.latest_prepare : step.mode === 'remote_smoke' ? pre?.latest_smoke : null
      const blockers = (mode?.blocked_by ?? []).filter((b) => !(step.mode === 'train_candidate' && b.startsWith('an approved model already exists')))
      const candidateNeedsInput = step.mode === 'train_candidate' && (!newVersion || reason.trim().length < 10)
      return <Panel key={step.mode} kicker="BƯỚC" title={step.title} subtitle={step.lead}
        right={last ? <Tag tone={stateTone(last.state)}>lần gần nhất: {stateText[last.state] ?? last.state}</Tag> : <Tag tone="warn">chưa chạy</Tag>}>
        {blockers.length > 0 ? <ul className="findings">{blockers.map((b) => <li className="warn" key={b}>{b}</li>)}</ul>
          : <div className="callout ok">Đủ điều kiện chạy bước này.</div>}
        {step.mode === 'train_candidate' && <>
          {pre?.approved_model && <div className="callout info">Đã có model được duyệt ({pre.approved_model.display_version}). Hệ thống sẽ <b>không</b> train lại trừ khi bạn yêu cầu phiên bản mới.</div>}
          <label className="confirm"><input type="checkbox" checked={newVersion} onChange={(e) => setNewVersion(e.target.checked)} />Tôi yêu cầu train một <b>phiên bản mới</b> một cách có chủ đích.</label>
          <label className="field">Lý do (tối thiểu 10 ký tự)<textarea rows={2} value={reason} onChange={(e) => setReason(e.target.value)} /></label>
          <div className="toolbar wrap">
            <label className="field">Epoch giai đoạn A (ảnh)<input type="number" min={1} max={50} value={imageEpochs} onChange={(e) => setImageEpochs(Number(e.target.value))} /></label>
            <label className="field">Epoch giai đoạn B (temporal)<input type="number" min={1} max={50} value={temporalEpochs} onChange={(e) => setTemporalEpochs(Number(e.target.value))} /></label>
            <label className="confirm"><input type="checkbox" checked={pretrained} onChange={(e) => setPretrained(e.target.checked)} />Khởi tạo encoder bằng trọng số ImageNet (Kaggle tải qua internet; ghi vào hồ sơ model)</label>
          </div>
          <p className="muted">Số cửa sổ temporal đếm được (chưa chia tập): <b>{fmt(pre?.temporal_window_count_before_split)}</b>.</p>
        </>}
        <div className="toolbar">
          <button className="primary-button" disabled={!mode?.allowed || action.busy || candidateNeedsInput} onClick={() => void submit(step.mode)}>
            {step.mode === 'train_candidate' && pre?.approved_model && !newVersion ? 'Kiểm tra model đã có' : 'Xếp vào hàng đợi'}</button>
          {last?.summary && last.job_id && <button className="secondary-button big" onClick={() => setOpen(open === `report:${last.job_id}` ? null : `report:${last.job_id}`)}>Xem báo cáo lần gần nhất</button>}
        </div>
        {last && open === `report:${last.job_id}` && <RunReportView jobId={last.job_id} />}
      </Panel>
    })}

    <Panel kicker="TIẾN ĐỘ" title="Job và tiến trình nền" subtitle="Tiến trình nền gửi job lên Kaggle, theo dõi, tải kết quả và khôi phục sau khi bạn tắt/mở lại ứng dụng."
      right={<div className="tag-row"><span className="muted">Cập nhật lúc {updatedAt ? updatedAt.toLocaleTimeString('vi-VN') : '…'} · tự làm mới mỗi 3 giây</span>
        <button className="secondary-button big" disabled={action.busy} onClick={() => void tick()}>Kiểm tra ngay</button></div>}>
      <div className="stat-row">
        <div><b>{supervisor?.running ? 'đang chạy' : 'tắt'}</b><span>tiến trình nền{supervisor?.ticks != null ? ` · ${supervisor.ticks} lượt` : ''}</span></div>
        <div><b>{pre?.compute_slot_held_by ? `${pre.compute_slot_held_by.job_id.slice(0, 8)}` : 'rảnh'}</b><span>ô compute chung{pre?.compute_slot_held_by ? ` · ${pre.compute_slot_held_by.mode ?? pre.compute_slot_held_by.job_type} (${pre.compute_slot_held_by.state})` : ''}</span></div>
        <div><b>{fmt(pre?.queued_kaggle_jobs)}</b><span>job Kaggle đang chờ</span></div>
      </div>
      {supervisor && !supervisor.running && <div className="callout warn">Tiến trình nền không chạy ({supervisor.reason ?? 'bị tắt bằng SUPERVISOR_ENABLED hoặc chưa có PostgreSQL'}). Job sẽ không tự được gửi hay theo dõi.</div>}
      {Object.keys(supervisor?.errors ?? {}).length > 0 && <ul className="findings">{Object.entries(supervisor!.errors!).map(([id, text]) => <li className="bad" key={id}>{id}: {text}</li>)}</ul>}
      {jobs.length === 0 ? <div className="empty-state">Chưa có job Kaggle nào.</div> : <div className="job-table">{jobs.slice(0, 12).map((job) => <div className="job-block" key={job.id}>
        <div className="job-line">
          <span className="job-type">{job.mode ?? job.job_type}</span>
          <b>{job.id.slice(0, 8)}</b><span className="muted">{when(job.created_at)}</span>
          <Tag tone={stateTone(job.state)}>{stateText[job.state] ?? job.state}</Tag>
          {!TERMINAL.has(job.state) && <span className="live-dot" title="Đang tự cập nhật" />}
          {job.remote_status && <span className="muted">Kaggle: {job.remote_status}</span>}
          <span className="muted">{minutesSince(job.updated_at)}</span>
          {job.remote_ref && <span className="muted">{job.remote_ref}{job.remote_version ? ` v${job.remote_version}` : ''}</span>}
          <span className="grow" />
          {job.state === 'queued' && <button className="review-button" onClick={() => void cancel(job.id)}>Hủy</button>}
          {['running', 'remote_unconfirmed', 'collecting'].includes(job.state) && <span className="muted">Đã tới Kaggle: hủy trên web Kaggle nếu cần</span>}
          <button className="secondary-button big" onClick={() => setOpen(open === job.id ? null : job.id)}>{open === job.id ? 'Ẩn lịch sử' : 'Lịch sử'}</button>
        </div>
        {job.error_summary && <div className="callout bad">{job.error_summary}</div>}
        {open === job.id && <Timeline jobId={job.id} active={!TERMINAL.has(job.state)} />}
      </div>)}</div>}
    </Panel>
  </div>
}
