import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'
import './pages.css'
import { api, fmt, request } from './api'
import CameraPage from './pages/CameraPage'
import DataPage from './pages/DataPage'
import EvaluationPage from './pages/EvaluationPage'
import TrainPage from './pages/TrainPage'
import { ResultVideo } from './videoOverlay'
import { Icon } from './icons'
import type { IconName } from './icons'

type Route = 'overview' | 'data' | 'train' | 'evaluation' | 'camera'
const ROUTES: Route[] = ['overview', 'data', 'train', 'evaluation', 'camera']
const NAV: Array<[Route, IconName, string, string]> = [
  ['overview', 'overview', 'Tổng quan', 'Trạng thái hệ thống'],
  ['camera', 'camera', 'Camera', 'Giám sát trực tiếp'],
  ['evaluation', 'evaluation', 'Đánh giá', 'Chạy thử ảnh, video'],
  ['data', 'data', 'Dữ liệu', 'Duyệt nguồn và nhãn'],
  ['train', 'train', 'Train', 'Kaggle notebook'],
]
function currentRoute(): Route {
  const hash = window.location.hash.replace('#', '')
  return (ROUTES as string[]).includes(hash) ? (hash as Route) : 'overview'
}

type ConfigStatus = {
  app_env: string
  database_configured: boolean
  database_reachable?: boolean | null
  drive_configured: boolean
  kaggle_kernel_configured: boolean
  kaggle_private: boolean
  train_executor: string
  local_training_enabled: boolean
}

type Source = {
  source_id: string
  source_version: string
  format: string
  modality: string
  valid_config: boolean
  root_env: string
  root_configured: boolean
  root_exists: boolean
  mapping_review_status: string
  mapping_actions: Array<{ original_label: string; action: string; canonical_label: string | null }>
}

type Job = { id: string; job_type: string; state: string; phase: string; created_at: string }
type Model = { id: string; model_family: string; display_version: string; status: string; model_kind: 'temporal' | 'image_tracked' | null; cached_locally: boolean; temporal_trained: boolean; completion_kind: string; created_at: string }
type ScanReport = { media_count: number; annotation_count: number; media_counts_by_original_split: Record<string, number>; annotation_counts_by_source_label: Record<string, number>; malformed_annotation_count: number; source_data_modified: boolean }
type TestReadiness = { ready_for_request: boolean; reason?: string; model_version?: string; max_upload_bytes: number }
type MediaResult = { job_id: string; state: string; modality: 'image' | 'video'; model_version_id: string; result_storage?: string; result_sync_error?: string; files: Record<string, string>; summary?: { detection_count?: number; frame_count?: number; render?: 'overlay' | 'video' } }

function date(value: string) { return new Date(value).toLocaleString('vi-VN', { dateStyle: 'short', timeStyle: 'short' }) }
function modelKindText(kind: Model['model_kind']) { return kind === 'image_tracked' ? 'ảnh + gộp theo ID' : kind === 'temporal' ? 'temporal' : 'chưa xác định loại' }
function StatusDot({ ok }: { ok: boolean }) { return <span className={`status-dot ${ok ? 'ok' : 'missing'}`} aria-label={ok ? 'Đã cấu hình' : 'Chưa cấu hình'} /> }
function jobStateTone(state: string) { return state === 'completed' ? 'approved' : ['failed', 'cancelled'].includes(state) ? 'failed' : 'pending' }

function App() {
  const mainContent = useRef<HTMLElement>(null)
  const [route, setRoute] = useState<Route>(currentRoute)
  useEffect(() => {
    const onHash = () => setRoute(currentRoute())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  useEffect(() => { mainContent.current?.scrollTo(0, 0) }, [route])
  const [config, setConfig] = useState<ConfigStatus | null>(null)
  const [sources, setSources] = useState<Source[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [models, setModels] = useState<Model[]>([])
  const [scan, setScan] = useState<Record<string, ScanReport>>({})
  const [busySource, setBusySource] = useState<string | null>(null)
  const [reviewingModel, setReviewingModel] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [apiOnline, setApiOnline] = useState<boolean | null>(null)
  const [testReadiness, setTestReadiness] = useState<TestReadiness | null>(null)
  const [mediaKind, setMediaKind] = useState<'image' | 'video'>('image')
  const [mediaFile, setMediaFile] = useState<File | null>(null)
  const [mediaKey, setMediaKey] = useState<string | null>(null)
  const [testingMedia, setTestingMedia] = useState(false)
  const [mediaResult, setMediaResult] = useState<MediaResult | null>(null)
  const [syncingResult, setSyncingResult] = useState(false)
  const [syncingModel, setSyncingModel] = useState(false)

  const refresh = useCallback(async () => {
    const [health, configResult, sourcesResult, jobsResult, modelsResult, readinessResult] = await Promise.allSettled([
      request<{ status: string }>('/health'),
      request<ConfigStatus>('/api/v1/config-status'),
      request<{ sources: Source[] }>('/api/v1/sources'),
      request<{ jobs: Job[] }>('/api/v1/jobs'),
      request<{ models: Model[] }>('/api/v1/models'),
      request<TestReadiness>('/api/v1/test/readiness'),
    ])
    setApiOnline(health.status === 'fulfilled')
    setError(health.status === 'rejected' ? `API chưa phản hồi: ${String(health.reason)}` : '')
    if (configResult.status === 'fulfilled') setConfig(configResult.value)
    if (sourcesResult.status === 'fulfilled') setSources(sourcesResult.value.sources)
    if (jobsResult.status === 'fulfilled') setJobs(jobsResult.value.jobs)
    if (modelsResult.status === 'fulfilled') setModels(modelsResult.value.models)
    setTestReadiness(readinessResult.status === 'fulfilled' ? readinessResult.value : null)
    setLoading(false)
  }, [])

  useEffect(() => { void refresh() }, [refresh, route])

  const scanSource = async (source: Source) => {
    setBusySource(source.source_id)
    setError('')
    setNotice('Đang quét metadata và nhãn. Dữ liệu nguồn chỉ được đọc.')
    try {
      const result = await request<ScanReport>(`/api/v1/sources/${encodeURIComponent(source.source_id)}/scan`, { method: 'POST' })
      setScan((current) => ({ ...current, [source.source_id]: result }))
      setNotice('Quét xong. Chưa chuyển đổi hoặc thay đổi file nguồn.')
    } catch (reason) {
      setNotice('')
      setError(String(reason))
    } finally { setBusySource(null) }
  }

  const reviewModel = async (model: Model, decision: 'approve' | 'reject') => {
    const reviewer = window.prompt('Tên người duyệt để lưu audit log:')?.trim()
    if (!reviewer) return
    setReviewingModel(model.id)
    setError('')
    try {
      await request(`/api/v1/models/${model.id}/review`, {
        method: 'POST',
        body: JSON.stringify({ decision, reviewer_id: reviewer, comment: '' }),
      })
      setNotice(decision === 'approve' ? 'Đã duyệt model.' : 'Đã từ chối model; candidate vẫn được lưu.')
      await refresh()
    } catch (reason) { setError(String(reason)) }
    finally { setReviewingModel(null) }
  }

  const databaseReady = Boolean(config?.database_configured && config?.database_reachable)
  const approvedModel = models.find((model) => model.status === 'approved' && model.model_kind)

  const syncApprovedModel = async () => {
    setSyncingModel(true)
    setError('')
    try {
      await request('/api/v1/models/sync-approved', { method: 'POST' })
      setNotice('Đã kiểm tra và đồng bộ model approved vào cache local.')
      await refresh()
    } catch (reason) { setError(String(reason)) }
    finally { setSyncingModel(false) }
  }

  const uploadMedia = async () => {
    if (!mediaFile || testingMedia) return
    setTestingMedia(true)
    setError('')
    const key = mediaKey ?? `media-${crypto.randomUUID()}`
    setMediaKey(key)
    const form = new FormData()
    form.append('file', mediaFile)
    try {
      const result = await request<MediaResult>(`/api/v1/test/${mediaKind}`, { method: 'POST', headers: { 'Idempotency-Key': key }, body: form })
      setMediaResult(result)
      setMediaKey(null)
      const messages: Record<string, string> = {
        completed: 'Kiểm tra và đồng bộ kết quả xong.',
        sync_pending: 'Kiểm tra local xong. Kết quả đang chờ đồng bộ Drive.',
        failed: 'Job kiểm tra đã thất bại. Xem trạng thái và lịch sử job; request này không tự chạy lại.',
      }
      setNotice(messages[result.state] ?? `Job đang ở trạng thái ${result.state}. Chưa có kết quả hoàn tất.`)
      await refresh()
    } catch (reason) { setError(String(reason)) }
    finally { setTestingMedia(false) }
  }

  const showResults = async (job: Job) => {
    try {
      const result = await request<MediaResult>(`/api/v1/jobs/${job.id}/results`)
      setMediaResult(result)
      document.getElementById('media-test')?.scrollIntoView({ behavior: 'smooth' })
    } catch (reason) { setError(String(reason)) }
  }

  const syncResults = async () => {
    if (!mediaResult) return
    setSyncingResult(true)
    try {
      setMediaResult(await request<MediaResult>(`/api/v1/jobs/${mediaResult.job_id}/sync-results`, { method: 'POST' }))
      await refresh()
    } catch (reason) { setError(String(reason)) }
    finally { setSyncingResult(false) }
  }

  const navItem = NAV.find(([id]) => id === route)!
  const systemRows: Array<[string, boolean, string]> = [
    ['API local', apiOnline === true, apiOnline === null ? 'đang kết nối' : apiOnline ? 'đang chạy' : 'không phản hồi'],
    ['PostgreSQL', Boolean(config?.database_reachable), config?.database_reachable ? 'đã kết nối' : config?.database_configured ? 'chưa kết nối được' : 'chưa cấu hình'],
    ['Google Drive', Boolean(config?.drive_configured), config?.drive_configured ? 'đã cấu hình' : 'chưa cấu hình'],
    ['Kaggle', Boolean(config?.kaggle_kernel_configured), config?.kaggle_kernel_configured ? 'đã cấu hình' : 'chưa cấu hình'],
  ]

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><Icon name="waves" size={20} /></div><div><strong>Stillwater</strong><small>Giám sát đuối nước</small></div></div>
      <nav className="nav-group" aria-label="Trang">
        {NAV.map(([id, icon, text, hint]) =>
          <a key={id} className={`nav-item ${route === id ? 'selected' : ''}`} href={`#${id}`} title={text} aria-current={route === id ? 'page' : undefined}>
            <Icon name={icon} className="nav-icon" />
            <span className="nav-text"><b>{text}</b><small>{hint}</small></span>
            {id === 'data' ? <span className="nav-count">{sources.length}</span> : null}
          </a>)}
      </nav>
      <div className="sidebar-bottom">
        <div className="system-box">
          <div className="system-title">Hệ thống</div>
          {systemRows.map(([name, ok, text]) => <div className="system-row" key={name}><StatusDot ok={ok} /><span>{name}</span><small>{text}</small></div>)}
        </div>
        <div className="sidebar-foot">Train chỉ chạy trên Kaggle · bản 0.1.0</div>
      </div>
    </aside>

    <main ref={mainContent} className="main-content" id="overview">
      <header className="topbar">
        <div className="topbar-title"><Icon name={navItem[1]} size={17} /><b>{navItem[2]}</b><span>{navItem[3]}</span></div>
        <div className="top-actions">
          <div className={`api-badge ${apiOnline === false ? 'offline' : ''}`}><StatusDot ok={apiOnline === true} />{apiOnline === null ? 'Đang kết nối…' : apiOnline ? 'API đang chạy' : 'API ngoại tuyến'}</div>
          <button className="icon-button" onClick={() => void refresh()} title="Làm mới" aria-label="Làm mới"><Icon name="refresh" size={16} /></button>
        </div>
      </header>

      <div className="page-wrap">
        {route === 'data' && <DataPage />}
        {route === 'train' && <TrainPage />}
        {route === 'evaluation' && <EvaluationPage />}
        {route === 'camera' && <CameraPage />}
        {route === 'overview' && <div className="page-stack">
        <section className="welcome-row">
          <div><div className="eyebrow">Tổng quan</div><h1>Bể bơi hôm nay thế nào?</h1>
            </div>
          <div className="welcome-actions">
            <a className="btn link-button" href="#evaluation"><Icon name="evaluation" size={16} /> Chạy thử video</a>
            <a className="btn btn-primary link-button" href="#camera"><Icon name="camera" size={16} /> Mở camera</a>
          </div>
        </section>
        {(error || notice) && <div className={`message ${error ? 'error' : 'notice'}`} role="status">{error || notice}</div>}

        <section className="metric-grid" aria-label="Trạng thái hệ thống">
          <article className="metric-card"><div className="metric-top"><span>Cơ sở dữ liệu</span><Icon name="data" size={17} /></div><div className="metric-value">{config?.database_reachable ? 'Đã kết nối' : config?.database_configured ? 'Chưa kết nối được' : 'Chưa cấu hình'}</div><div className="metric-foot"><StatusDot ok={Boolean(config?.database_reachable)} /> PostgreSQL dùng chung</div></article>
          <article className="metric-card"><div className="metric-top"><span>Google Drive</span><Icon name="upload" size={17} /></div><div className="metric-value">{config?.drive_configured ? 'Đã cấu hình' : 'Chưa cấu hình'}</div><div className="metric-foot"><StatusDot ok={Boolean(config?.drive_configured)} /> Nơi lưu artifact</div></article>
          <article className="metric-card"><div className="metric-top"><span>Kaggle</span><Icon name="train" size={17} /></div><div className="metric-value">{config?.kaggle_kernel_configured ? 'Đã cấu hình' : 'Chưa cấu hình'}</div><div className="metric-foot"><StatusDot ok={Boolean(config?.kaggle_kernel_configured)} /> Notebook riêng tư · GPU</div></article>
          <article className={`metric-card ${approvedModel ? 'accent' : ''}`}><div className="metric-top"><span>Model đang dùng</span><Icon name="check" size={17} /></div><div className="metric-value">{approvedModel?.display_version ?? 'Chưa có'}</div><div className="metric-foot"><StatusDot ok={Boolean(approvedModel)} /> {approvedModel ? modelKindText(approvedModel.model_kind) : 'Train → Đánh giá → Duyệt'}</div></article>
        </section>

        <section className="panel" id="media-test">
          <div className="panel-header">
            <div><div className="panel-kicker">Chạy thử nhanh</div><h2>Kiểm tra ảnh hoặc video</h2>
              <p>{approvedModel ? `Dùng model đã duyệt ${approvedModel.display_version} (${modelKindText(approvedModel.model_kind)}).` : 'Cần có model đã duyệt: Train → Đánh giá → Duyệt.'}</p></div>
            <div className="tag-row">
              {approvedModel && <button className="btn btn-small" disabled={!databaseReady || syncingModel || testingMedia} onClick={() => void syncApprovedModel()} title={approvedModel.cached_locally ? 'Kiểm tra và đồng bộ lại model đã duyệt' : 'Tải model đã duyệt về máy'}>{syncingModel ? 'Đang đồng bộ…' : approvedModel.cached_locally ? 'Model đã có trên máy' : 'Tải model về máy'}</button>}
              <span className={`state-tag ${testReadiness?.ready_for_request ? 'approved' : 'pending'}`}>{testReadiness?.ready_for_request ? `sẵn sàng · ${testReadiness.model_version}` : 'chưa sẵn sàng'}</span>
            </div>
          </div>
          <div className="media-controls">
            <div className="segmented" role="radiogroup" aria-label="Loại input">
              {(['image', 'video'] as const).map((kind) => <button key={kind} role="radio" aria-checked={mediaKind === kind} className={mediaKind === kind ? 'on' : ''} disabled={testingMedia}
                onClick={() => { setMediaKind(kind); setMediaFile(null); setMediaKey(null); setMediaResult(null) }}>{kind === 'image' ? 'Ảnh' : 'Video'}</button>)}
            </div>
            <label className="file-field"><Icon name="file" size={16} /><span>{mediaFile ? mediaFile.name : mediaKind === 'image' ? 'Chọn ảnh JPG, PNG, WebP, BMP…' : 'Chọn video MP4, AVI, MOV, MKV, WebM…'}</span>
              <input key={mediaKind} type="file" accept={mediaKind === 'image' ? '.jpg,.jpeg,.png,.webp,.bmp' : '.mp4,.avi,.mov,.mkv,.webm'} disabled={testingMedia} onChange={(event) => { setMediaFile(event.target.files?.[0] ?? null); setMediaKey(null); setMediaResult(null) }} /></label>
            <button className="btn btn-primary" disabled={!testReadiness?.ready_for_request || !mediaFile || testingMedia} onClick={() => void uploadMedia()}>{testingMedia ? 'Đang xử lý…' : mediaKey ? 'Thử lại cùng request' : 'Chạy kiểm tra'}</button>
          </div>
          <p className="media-help">{testReadiness?.ready_for_request ? 'Tối đa 512 MiB. Video được xử lý offline; nếu Drive chưa sẵn sàng, kết quả local vẫn xem được.' : testReadiness?.reason ?? 'Cần PostgreSQL đã migrate, model đã duyệt được tải về máy và detector weights trước khi chạy.'}</p>
          {mediaResult && <div className="media-result">
            <div className="result-info"><b>Job {mediaResult.job_id.slice(0, 8)}</b>
              <span>{mediaResult.state} · {mediaResult.result_storage ?? 'đang xử lý'} · {fmt(mediaResult.summary?.detection_count)} lần phát hiện{mediaResult.summary?.frame_count !== undefined ? ` · ${fmt(mediaResult.summary.frame_count)} khung` : ''}</span>
              <div className="result-links">
                {mediaResult.files.annotated && <a href={api(mediaResult.files.annotated)} target="_blank" rel="noreferrer">Mở media kết quả</a>}
                {mediaResult.files.source && <a href={api(mediaResult.files.source)} target="_blank" rel="noreferrer">Mở video gốc</a>}
                {mediaResult.files.detections && <a href={api(mediaResult.files.detections)} target="_blank" rel="noreferrer">Tải detections JSON</a>}
                {mediaResult.state === 'sync_pending' && config?.drive_configured && <button className="btn btn-small" disabled={syncingResult} onClick={() => void syncResults()}>{syncingResult ? 'Đang đồng bộ…' : 'Đồng bộ lại Drive'}</button>}
              </div>
            </div>
            {mediaResult.modality === 'video' && (mediaResult.files.annotated || mediaResult.files.source) && (mediaResult.files.detections
              ? <ResultVideo jobId={mediaResult.job_id} src={mediaResult.files.annotated ?? mediaResult.files.source} />
              : <video controls src={api(mediaResult.files.annotated ?? mediaResult.files.source)} />)}
            {mediaResult.modality === 'image' && mediaResult.files.annotated &&
              <img src={api(mediaResult.files.annotated)} alt="Kết quả kiểm tra ảnh với box và nhãn" />}
          </div>}
        </section>

        <section className="content-grid">
          <div className="panel datasets-panel" id="datasets">
            <div className="panel-header"><div><div className="panel-kicker">Registry</div><h2>Nguồn dữ liệu</h2><p>Thư mục gốc, ánh xạ nhãn và trạng thái duyệt.</p></div><span className="count-pill">{sources.length} nguồn</span></div>
            {loading ? <div className="empty-state">Đang tải registry…</div> : sources.length === 0 ? <div className="empty-state">Chưa có cấu hình nguồn trong configs/datasets.</div> : <div className="source-list">{sources.map((source) => {
              const report = scan[source.source_id]
              return <article className="source-card" key={source.source_id}>
                <div className="source-symbol"><Icon name={source.modality === 'video' ? 'camera' : 'file'} size={16} /></div>
                <div className="source-main"><div className="source-title-row"><h3>{source.source_id.replaceAll('_', ' ')}</h3><span className={`state-tag ${source.mapping_review_status === 'approved' ? 'approved' : 'pending'}`}>{source.mapping_review_status.replaceAll('_', ' ')}</span></div>
                  <div className="source-meta">{source.format} <span>·</span> v{source.source_version} <span>·</span> {source.root_env}: {source.root_exists ? 'thư mục sẵn sàng' : 'chưa có thư mục'}</div>
                  <div className="class-tags">{source.mapping_actions.map((item) => <span key={item.original_label} className="class-tag"><i className={item.action === 'map' ? 'mapped' : 'unmapped'} />{item.original_label}<small>{item.canonical_label ?? item.action}</small></span>)}</div>
                  {report && <div className="scan-summary"><b>{fmt(report.media_count)}</b> ảnh · <b>{fmt(report.annotation_count)}</b> bbox · {fmt(report.malformed_annotation_count)} dòng lỗi · split {Object.entries(report.media_counts_by_original_split).map(([key, value]) => `${key} ${fmt(value)}`).join(' / ')}</div>}
                </div>
                <button className="btn btn-small scan-button" onClick={() => void scanSource(source)} disabled={!source.root_exists || busySource === source.source_id}>{busySource === source.source_id ? 'Đang quét…' : report ? 'Quét lại' : 'Quét'}</button>
              </article>
            })}</div>}
            <div className="panel-note">Nguồn Alanoud hiện là ảnh tĩnh. Ánh xạ nhãn đang chờ duyệt mẫu; crop xem trước chưa đủ điều kiện train.</div>
          </div>

          <div className="right-column">
            <div className="panel model-panel" id="models"><div className="panel-header compact"><div><div className="panel-kicker">Model</div><h2>Các phiên bản</h2></div><span className="count-pill">{models.length}</span></div>
              {models.length === 0 ? <div className="empty-state small"><b>Chưa có model</b><span>Model ứng viên cần train trên Kaggle và được duyệt trước khi dùng.</span></div> : models.slice(0, 4).map((model) => <div className="model-row" key={model.id}><div className="model-version">{model.display_version}</div><div><b>{model.model_family}</b><small>{modelKindText(model.model_kind)} · {date(model.created_at)}</small></div><span className={`state-tag ${model.status === 'approved' ? 'approved' : 'pending'}`}>{model.status}</span>{model.status === 'candidate' && <div className="review-actions"><button className="review-button approve" disabled={reviewingModel === model.id} onClick={() => void reviewModel(model, 'approve')}>Duyệt</button><button className="review-button reject" disabled={reviewingModel === model.id} onClick={() => void reviewModel(model, 'reject')}>Từ chối</button></div>}</div>)}
            </div>
            <div className="panel jobs-panel" id="jobs"><div className="panel-header compact"><div><div className="panel-kicker">Hàng đợi</div><h2>Tác vụ gần đây</h2></div><span className="count-pill">{jobs.length}</span></div>
              {jobs.length === 0 ? <div className="empty-state small"><b>Chưa có tác vụ</b><span>{databaseReady ? 'Các job train, kiểm tra và camera sẽ hiện ở đây.' : 'Cần kết nối PostgreSQL để ghi job bền vững.'}</span></div> : <div className="job-list">{jobs.slice(0, 5).map((job) => <div className="job-row" key={job.id}><span className="job-type">{job.job_type}</span><div className="job-info"><b>{job.id.slice(0, 8)}</b><small>{date(job.created_at)} · {job.phase}</small></div><span className={`state-tag ${jobStateTone(job.state)}`}>{job.state}</span>{job.job_type === 'inference' && ['completed', 'sync_pending'].includes(job.state) && <button className="btn btn-small" onClick={() => void showResults(job)}>Kết quả</button>}</div>)}</div>}
            </div>
          </div>
        </section>
        </div>}
      </div>
    </main>
  </div>
}

export default App
