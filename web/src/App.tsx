import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'
import './pages.css'
import { api, fmt, request } from './api'
import CameraPage from './pages/CameraPage'
import DataPage from './pages/DataPage'
import EvaluationPage from './pages/EvaluationPage'
import TrainPage from './pages/TrainPage'

type Route = 'overview' | 'data' | 'train' | 'evaluation' | 'camera'
const ROUTES: Route[] = ['overview', 'data', 'train', 'evaluation', 'camera']
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

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><span>W</span></div><div><strong>Stillwater</strong><small>Water safety lab</small></div></div>
      <div className="workspace-label">WORKSPACE</div>
      <div className="workspace-switch" aria-label="Workspace hiện tại"><span className="workspace-avatar">D</span><span><b>Drowning project</b><small>Local environment</small></span></div>
      <div className="nav-group"><div className="workspace-label">PROJECT</div>
        {([['overview', '◫', 'Tổng quan'], ['data', '▦', 'Dữ liệu'], ['train', '⌁', 'Train'], ['evaluation', '◈', 'Đánh giá'], ['camera', '◉', 'Camera']] as const).map(([id, icon, text]) =>
          <a key={id} className={`nav-item ${route === id ? 'selected' : ''}`} href={`#${id}`} title={text} aria-label={text} aria-current={route === id ? 'page' : undefined}><span className="nav-icon" aria-hidden="true">{icon}</span> {text}{id === 'data' ? <span className="nav-count">{sources.length}</span> : null}</a>)}
      </div>
      <div className="sidebar-bottom"><div className="profile"><div className="profile-avatar">AI</div><div><b>Local operator</b><small>Project workspace</small></div><span className="profile-menu" aria-hidden="true">•••</span></div><div className="sidebar-foot">DROWNING DETECTION<br /><span>BUILD 0.1.0 · DEV</span></div></div>
    </aside>

    <main ref={mainContent} className="main-content" id="overview">
      <header className="topbar"><div className="breadcrumb"><span>Workspace</span><i>/</i><b>Drowning project</b></div><div className="top-actions"><div className={`api-badge ${apiOnline === false ? 'offline' : 'online'}`}><StatusDot ok={apiOnline === true} />{apiOnline === null ? 'Đang kết nối…' : apiOnline ? 'API local' : 'API ngoại tuyến'}</div><button className="icon-button" onClick={() => void refresh()} title="Làm mới">↻</button><div className="user-avatar">OP</div></div></header>

      <div className="page-wrap">
        {route === 'data' && <DataPage />}
        {route === 'train' && <TrainPage />}
        {route === 'evaluation' && <EvaluationPage />}
        {route === 'camera' && <CameraPage />}
        {route === 'overview' && <>
        <section className="welcome-row"><div><div className="eyebrow"><span className="eyebrow-line" /> MODEL DEVELOPMENT WORKSPACE</div><h1>Xin chào, operator <span>✳</span></h1><p className="subtitle">Theo dõi nguồn dữ liệu, trạng thái model và các tác vụ trong môi trường local.</p></div><a className="primary-button link-button" href="#train"><span>→</span> Mở trang Train</a></section>
        {(error || notice) && <div className={`message ${error ? 'error' : 'notice'}`} role="status">{error || notice}</div>}

        <section className="metric-grid" aria-label="Trạng thái hệ thống">
          <article className="metric-card"><div className="metric-top"><span>DATABASE</span><span className="metric-icon violet">⌘</span></div><div className="metric-value">{config?.database_reachable ? 'Connected' : config?.database_configured ? 'Not ready' : 'Not configured'}</div><div className="metric-foot"><StatusDot ok={Boolean(config?.database_reachable)} /> PostgreSQL chung</div></article>
          <article className="metric-card"><div className="metric-top"><span>GOOGLE DRIVE</span><span className="metric-icon blue">↗</span></div><div className="metric-value">{config?.drive_configured ? 'Configured' : 'Not configured'}</div><div className="metric-foot"><StatusDot ok={Boolean(config?.drive_configured)} /> Artifact storage</div></article>
          <article className="metric-card"><div className="metric-top"><span>KAGGLE WORKER</span><span className="metric-icon green">⌁</span></div><div className="metric-value">{config?.kaggle_kernel_configured ? 'Configured' : 'Not configured'}</div><div className="metric-foot"><StatusDot ok={Boolean(config?.kaggle_kernel_configured)} /> Private · GPU executor</div></article>
          <article className="metric-card"><div className="metric-top"><span>MODEL ĐÃ DUYỆT</span><span className="metric-icon amber">◈</span></div><div className="metric-value">{approvedModel?.display_version ?? 'Chưa có'}</div><div className="metric-foot"><StatusDot ok={Boolean(approvedModel)} /> {approvedModel ? modelKindText(approvedModel.model_kind) : 'Train → Đánh giá → Duyệt'}</div></article>
        </section>

        <section className="panel media-panel" aria-labelledby="use-model-title">
          <div className="panel-header"><div><div className="panel-kicker">MODEL ĐÃ DUYỆT</div><h2 id="use-model-title">Dùng model</h2><p>{approvedModel ? `${approvedModel.display_version} · ${modelKindText(approvedModel.model_kind)}` : 'Train → Đánh giá → Duyệt model làm model chính.'}</p></div></div>
          {approvedModel ? <div className="toolbar">
            <button className="secondary-button big" disabled={!databaseReady || syncingModel || testingMedia} onClick={() => void syncApprovedModel()} title={approvedModel.cached_locally ? 'Kiểm tra và đồng bộ lại model đã duyệt' : 'Tải model đã duyệt về máy'}>{syncingModel ? 'Đang đồng bộ model…' : approvedModel.cached_locally ? 'Đã có trên máy' : 'Đồng bộ về máy'}</button>
            <a className="secondary-button big link-button" href="#evaluation">Chạy trên video / ảnh</a>
            <a className="secondary-button big link-button" href="#camera">Mở Camera</a>
          </div> : <div className="toolbar"><a className="secondary-button big link-button" href="#train">Mở trang Train</a><a className="secondary-button big link-button" href="#evaluation">Mở trang Đánh giá</a></div>}
        </section>

        <section className="panel media-panel" id="media-test">
          <div className="panel-header"><div><div className="panel-kicker">LOCAL INFERENCE</div><h2>Kiểm tra ảnh / video</h2><p>Dùng đúng model đã duyệt và quyền compute chung.</p></div><span className="count-pill">{testReadiness?.ready_for_request ? testReadiness.model_version : 'Chưa sẵn sàng'}</span></div>
          <div className="media-controls">
            <label>Loại input<select value={mediaKind} disabled={testingMedia} onChange={(event) => { setMediaKind(event.target.value as 'image' | 'video'); setMediaFile(null); setMediaKey(null); setMediaResult(null) }}><option value="image">Ảnh</option><option value="video">Video offline</option></select></label>
            <label className="file-field">Tệp kiểm tra<input key={mediaKind} type="file" accept={mediaKind === 'image' ? '.jpg,.jpeg,.png,.webp,.bmp' : '.mp4,.avi,.mov,.mkv,.webm'} disabled={testingMedia} onChange={(event) => { setMediaFile(event.target.files?.[0] ?? null); setMediaKey(null); setMediaResult(null) }} /></label>
            <button className="primary-button" disabled={!testReadiness?.ready_for_request || !mediaFile || testingMedia} onClick={() => void uploadMedia()}>{testingMedia ? 'Đang xử lý…' : mediaKey ? 'Thử lại cùng request' : 'Chạy kiểm tra'}</button>
          </div>
          <p className="media-help">{testReadiness?.ready_for_request ? 'Input tối đa 512 MiB. Video được xử lý offline; khi Drive chưa sẵn sàng, kết quả local vẫn tải được.' : testReadiness?.reason ?? 'Cần PostgreSQL đã migrate, model approved đã sync và detector weights trước khi chạy.'}</p>
          {mediaResult && <div className="media-result">
            <div className="result-info"><b>Job {mediaResult.job_id.slice(0, 8)}</b>
              <span>{mediaResult.state} · {mediaResult.result_storage ?? 'đang xử lý'} · {fmt(mediaResult.summary?.detection_count)} detections{mediaResult.summary?.frame_count !== undefined ? ` · ${fmt(mediaResult.summary.frame_count)} frames` : ''}</span>
              <div className="result-links">
                {mediaResult.files.annotated && <a href={api(mediaResult.files.annotated)} target="_blank" rel="noreferrer">Mở media kết quả</a>}
                {mediaResult.files.source && <a href={api(mediaResult.files.source)} target="_blank" rel="noreferrer">Mở video gốc · overlay ở Đánh giá</a>}
                {mediaResult.files.detections && <a href={api(mediaResult.files.detections)} target="_blank" rel="noreferrer">Tải detections JSON</a>}
                {mediaResult.state === 'sync_pending' && config?.drive_configured && <button className="secondary-button" disabled={syncingResult} onClick={() => void syncResults()}>{syncingResult ? 'Đang đồng bộ…' : 'Đồng bộ lại Drive'}</button>}
              </div>
            </div>
            {mediaResult.modality === 'video' && (mediaResult.files.annotated || mediaResult.files.source) &&
              <video controls src={api(mediaResult.files.annotated ?? mediaResult.files.source)} />}
            {mediaResult.modality === 'image' && mediaResult.files.annotated &&
              <img src={api(mediaResult.files.annotated)} alt="Kết quả kiểm tra ảnh với box và nhãn" />}
          </div>}
        </section>

        <section className="content-grid">
          <div className="panel datasets-panel" id="datasets">
            <div className="panel-header"><div><div className="panel-kicker">DATA REGISTRY</div><h2>Nguồn dữ liệu</h2><p>Roots, mapping và trạng thái duyệt nguồn.</p></div><span className="count-pill">{sources.length} nguồn</span></div>
            {loading ? <div className="empty-state">Đang tải registry…</div> : sources.length === 0 ? <div className="empty-state">Chưa có cấu hình nguồn trong configs/datasets.</div> : <div className="source-list">{sources.map((source) => {
              const report = scan[source.source_id]
              return <article className="source-card" key={source.source_id}>
                <div className="source-symbol">{source.modality === 'video' ? '▣' : '▧'}</div>
                <div className="source-main"><div className="source-title-row"><h3>{source.source_id.replaceAll('_', ' ')}</h3><span className={`state-tag ${source.mapping_review_status === 'approved' ? 'approved' : 'pending'}`}>{source.mapping_review_status.replaceAll('_', ' ')}</span></div>
                  <div className="source-meta">{source.format} <span>·</span> v{source.source_version} <span>·</span> {source.root_env}: {source.root_exists ? 'root sẵn sàng' : 'root chưa có'}</div>
                  <div className="class-tags">{source.mapping_actions.map((item) => <span key={item.original_label} className="class-tag"><i className={item.action === 'map' ? 'mapped' : 'unmapped'} />{item.original_label}<small>{item.canonical_label ?? item.action}</small></span>)}</div>
                  {report && <div className="scan-summary"><b>{fmt(report.media_count)}</b> ảnh · <b>{fmt(report.annotation_count)}</b> bbox · {fmt(report.malformed_annotation_count)} dòng lỗi · split {Object.entries(report.media_counts_by_original_split).map(([key, value]) => `${key} ${fmt(value)}`).join(' / ')}</div>}
                </div>
                <button className="secondary-button scan-button" onClick={() => void scanSource(source)} disabled={!source.root_exists || busySource === source.source_id}>{busySource === source.source_id ? 'Đang quét…' : report ? 'Quét lại' : 'Quét'}</button>
              </article>
            })}</div>}
            <div className="panel-note"><span>ⓘ</span> Nguồn Alanoud hiện là ảnh tĩnh. Mapping đang chờ duyệt mẫu; crop preview không đủ điều kiện train.</div>
          </div>

          <div className="right-column">
            <div className="panel model-panel" id="models"><div className="panel-header compact"><div><div className="panel-kicker">MODEL REGISTRY</div><h2>Model versions</h2></div><span className="count-pill">{models.length}</span></div>
              {models.length === 0 ? <div className="empty-state small"><div className="empty-symbol">◈</div><b>Chưa có model</b><span>Candidate cần train trên Kaggle và được duyệt trước khi dùng.</span></div> : models.slice(0, 4).map((model) => <div className="model-row" key={model.id}><div className="model-version">{model.display_version}</div><div><b>{model.model_family}</b><small>{modelKindText(model.model_kind)}</small></div><span className={`state-tag ${model.status === 'approved' ? 'approved' : 'pending'}`}>{model.status}</span>{model.status === 'candidate' && <div className="review-actions"><button className="review-button approve" disabled={reviewingModel === model.id} onClick={() => void reviewModel(model, 'approve')}>Duyệt</button><button className="review-button reject" disabled={reviewingModel === model.id} onClick={() => void reviewModel(model, 'reject')}>Từ chối</button></div>}</div>)}
            </div>
            <div className="panel jobs-panel" id="jobs"><div className="panel-header compact"><div><div className="panel-kicker">COMPUTE QUEUE</div><h2>Jobs gần đây</h2></div><span className="count-pill">{jobs.length}</span></div>
              {jobs.length === 0 ? <div className="empty-state small"><div className="empty-symbol">◷</div><b>Queue đang trống</b><span>{databaseReady ? 'Chưa có job nào.' : 'Cần kết nối PostgreSQL để ghi job bền vững.'}</span></div> : <div className="job-list">{jobs.slice(0, 5).map((job) => <div className="job-row" key={job.id}><span className="job-type">{job.job_type}</span><div className="job-info"><b>{job.id.slice(0, 8)}</b><small>{date(job.created_at)} · {job.phase}</small></div><span className="state-tag pending">{job.state}</span>{job.job_type === 'inference' && ['completed', 'sync_pending'].includes(job.state) && <button className="secondary-button" onClick={() => void showResults(job)}>Kết quả</button>}</div>)}</div>}
            </div>
          </div>
        </section>
        <footer className="page-footer"><span>Stillwater <b>·</b> Local workspace</span><span>Train executor: <b>Kaggle</b> <i /> Local train: <b>disabled</b></span></footer>
        </>}
      </div>
    </main>
  </div>
}

export default App
