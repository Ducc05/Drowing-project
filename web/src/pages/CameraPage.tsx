import { useCallback, useState } from 'react'
import { api, newKey, post, request, useAction, usePolling } from '../api'
import { Banner, Panel, Tag } from '../ui'

type Record_ = { track_id: string; label: string | null; prediction_branch: string; detector_confidence: number; bbox_xyxy: number[] }
type Status = {
  active: boolean; job_id?: string; source?: string; state?: string; error?: string | null; width?: number; height?: number
  frames_read?: number; frames_processed?: number; frames_dropped?: number; processed_fps?: number | null
  latency_ms_p50?: number | null; latency_ms_p95?: number | null; elapsed_seconds?: number; tracker?: string | null
  records?: Record_[]; last_session?: Status | null
}
type Readiness = { ready_for_request: boolean; reason?: string; model_version?: string }

const labelText: Record<string, string> = { drowning: 'Đuối nước', swimming: 'Bình thường trong nước', out_of_water: 'Ngoài nước' }
const branchText: Record<string, string> = { temporal: 'temporal', image_warmup: 'ảnh đơn (khởi động)',
  image_tracked: 'ảnh gộp theo ID', image_frame: 'ảnh đơn (chưa đủ khung)' }

export default function CameraPage() {
  const [source, setSource] = useState('0')
  const [status, setStatus] = useState<Status>({ active: false })
  const [readiness, setReadiness] = useState<Readiness | null>(null)
  const [probe, setProbe] = useState<string>('')
  const action = useAction()
  const setActionError = action.setError

  const load = useCallback(async () => {
    const [s, r] = await Promise.allSettled([request<Status>('/api/v1/camera/status'), request<Readiness>('/api/v1/test/readiness')])
    if (s.status === 'fulfilled') setStatus(s.value)
    else setActionError(String(s.reason))
    if (r.status === 'fulfilled') setReadiness(r.value)
  }, [setActionError])
  usePolling(load, 1000, true)

  const doProbe = () => action.run(async () => {
    setProbe('')
    const result = await post<{ width: number; height: number; source: string }>('/api/v1/camera/probe', { source })
    setProbe(`Kết nối được ${result.source}: ${result.width}×${result.height}`)
  })
  const start = () => action.run(async () => { setProbe(''); await post('/api/v1/camera/start', { source, idempotency_key: newKey('camera') }); await load(); return 'Camera đã chạy. Ô compute chung đang được giữ.' })
  const stop = () => action.run(async () => { await post('/api/v1/camera/stop'); await load(); return 'Đã dừng camera và xác nhận các luồng xử lý đã thoát.' })

  const active = status.active
  const summary = active ? status : status.last_session
  return <div className="page-stack">
    <header className="page-head"><div className="eyebrow"><span className="eyebrow-line" /> CAMERA</div><h1>Nhận diện từ camera trực tiếp</h1>
      <p className="subtitle">Webcam Windows (số 0–15) hoặc URL luồng http/https/rtsp/rtmp do bạn nhập. Chỉ dùng model đã duyệt; một lúc chỉ một tác vụ compute trên toàn hệ thống.</p></header>
    <Banner error={action.error} notice={action.notice} />
    <Panel kicker="NGUỒN" title="Kết nối camera" subtitle="Thông tin đăng nhập trong URL không bao giờ được hiển thị hay lưu."
      right={<Tag tone={readiness?.ready_for_request ? 'ok' : 'warn'}>{readiness?.ready_for_request ? `model ${readiness.model_version}` : 'chưa sẵn sàng'}</Tag>}>
      <div className="toolbar wrap">
        <label className="field grow">Webcam (0, 1…) hoặc URL luồng<input value={source} disabled={active} onChange={(e) => setSource(e.target.value)} placeholder="0 hoặc http://192.168.1.5:8080/video" /></label>
        <button className="secondary-button big" disabled={active || action.busy || !source.trim()} onClick={() => void doProbe()}>Kiểm tra kết nối</button>
        {!active
          ? <button className="primary-button" disabled={action.busy || !source.trim() || !readiness?.ready_for_request} onClick={() => void start()}>Start</button>
          : <button className="primary-button stop" disabled={action.busy} onClick={() => void stop()}>Stop</button>}
      </div>
      {probe && <div className="callout ok">{probe}</div>}
      {!readiness?.ready_for_request && <div className="callout warn">{readiness?.reason ?? 'Cần model đã duyệt, model được tải về máy và PERSON_DETECTOR_WEIGHTS trước khi chạy camera.'}</div>}
    </Panel>

    {active && <Panel kicker="TRỰC TIẾP" title="Luồng xử lý" subtitle={`${status.source} · ${status.width}×${status.height}`}
      right={<Tag tone={status.state === 'running' ? 'ok' : 'warn'}>{status.state}</Tag>}>
      <div className="camera-layout">
        <img className="camera-view" key={status.job_id} src={api('/api/v1/camera/stream')} alt="Luồng camera đã gắn khung và nhãn" />
        <div className="camera-side">
          <div className="stat-grid">
            <div><b>{status.processed_fps ?? '—'}</b><span>khung/giây xử lý</span></div>
            <div><b>{status.latency_ms_p50 ?? '—'}</b><span>độ trễ p50 (ms)</span></div>
            <div><b>{status.latency_ms_p95 ?? '—'}</b><span>độ trễ p95 (ms)</span></div>
            <div><b>{status.frames_dropped ?? 0}</b><span>khung bị bỏ qua</span></div>
          </div>
          <p className="muted">Tracker: {status.tracker}. ID là ID tạm của tracker, không phải danh tính đã xác minh. Nếu model chậm hơn camera, khung cũ bị bỏ để không tích trễ.</p>
          <table className="data-table"><thead><tr><th>ID</th><th>Nhãn</th><th>Nhánh</th><th>Phát hiện</th></tr></thead>
            <tbody>{(status.records ?? []).map((r) => <tr key={r.track_id}><td>{r.track_id.replace('trk-', '#')}</td><td>{r.label ? labelText[r.label] ?? r.label : '—'}</td>
              <td>{branchText[r.prediction_branch] ?? r.prediction_branch}</td><td>{(r.detector_confidence * 100).toFixed(0)}%</td></tr>)}
              {(status.records ?? []).length === 0 && <tr><td colSpan={4} className="muted">Chưa phát hiện người nào trong khung hình hiện tại.</td></tr>}</tbody></table>
        </div>
      </div>
    </Panel>}

    {!active && summary && <Panel kicker="PHIÊN GẦN NHẤT" title="Kết quả phiên camera" subtitle={summary.source}
      right={<Tag tone={summary.state === 'stopped' ? 'ok' : 'bad'}>{summary.state === 'stopped' ? 'đã dừng bình thường' : `lỗi: ${summary.error ?? summary.state}`}</Tag>}>
      <div className="stat-row">
        <div><b>{summary.elapsed_seconds}s</b><span>thời lượng</span></div><div><b>{summary.frames_processed}</b><span>khung đã xử lý</span></div>
        <div><b>{summary.processed_fps ?? '—'}</b><span>khung/giây</span></div><div><b>{summary.latency_ms_p95 ?? '—'} ms</b><span>độ trễ p95</span></div>
      </div>
      <p className="muted">Số liệu tốc độ này đo trên máy hiện tại với đúng detector và model đang dùng.</p>
    </Panel>}
  </div>
}
