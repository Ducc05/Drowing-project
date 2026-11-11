import { useCallback, useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { api, newKey, post, request, scoreText, useAction, useCellGrid, useConfidence, usePolling, cellGridParam, parseCellGrid } from '../api'
import { Banner, CellGridControl, ClassScores, ConfidenceControl, Panel, Tag } from '../ui'
import { Icon } from '../icons'
import { labelColor, labelVi, trackName, useAlarmSound, useTitleAlert } from '../overlayData'
import type { Label } from '../overlayData'
import { AlertBanner, LabelLegend, ScoreBar } from '../videoOverlay'
import { beginSending, pickSurface, screenShareSupported, stopShare, useScreenShare } from '../screenShare'

type Record_ = { track_id: string; label: string | null; prediction_branch: string; detector_confidence: number; class_scores?: number[] | null; bbox_xyxy: number[]; alert?: boolean }
type Status = {
  active: boolean; job_id?: string; source?: string; state?: string; error?: string | null; width?: number; height?: number
  frames_read?: number; frames_processed?: number; frames_dropped?: number; processed_fps?: number | null
  latency_ms_p50?: number | null; latency_ms_p95?: number | null; elapsed_seconds?: number; tracker?: string | null
  alert_track_ids?: string[]; records?: Record_[]; last_session?: Status | null; confidence_threshold?: number | null
}
type Readiness = { ready_for_request: boolean; reason?: string; model_version?: string; default_confidence?: number; default_cell_grid?: string; cell_grid_supported?: boolean }

const branchText: Record<string, string> = { temporal: 'temporal', image_warmup: 'ảnh đơn (khởi động)',
  image_tracked: 'ảnh gộp theo ID', image_frame: 'ảnh đơn (chưa đủ khung)',
  h20saver_tracked: 'H20Saver gộp theo ID', h20saver_frame: 'H20Saver (chưa đủ khung)' }
const COUNT_CARDS: Array<{ label: Label; icon: 'alert' | 'swim' | 'person'; hint: string }> = [
  { label: 'drowning', icon: 'alert', hint: 'cần quan sát ngay' },
  { label: 'swimming', icon: 'swim', hint: 'dưới nước' },
  { label: 'out_of_water', icon: 'person', hint: 'trên bờ' },
]
const SEND_FPS = 10
const elapsed = (seconds?: number) => {
  const s = Math.round(seconds ?? 0)
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

export default function CameraPage() {
  const [source, setSource] = useState('0')
  const [mode, setMode] = useState<'camera' | 'tab'>('camera')
  const share = useScreenShare()
  const tabMode = mode === 'tab'
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
  const [chosenConfidence, setConfidence] = useConfidence()
  const confidence = chosenConfidence ?? readiness?.default_confidence ?? 0.25
  const [chosenGrid, setCellGrid] = useCellGrid()
  const cellGrid = chosenGrid ?? parseCellGrid(readiness?.default_cell_grid)
  const gridParam = readiness?.cell_grid_supported ? cellGridParam(chosenGrid) : undefined
  const start = () => action.run(async () => {
    setProbe('')
    if (tabMode) {
      // Frames must flow while the server loads the model: it waits for the first frame before it reports "started".
      beginSending(SEND_FPS)
      try { await post('/api/v1/camera/start', { source: 'browser-tab', fps: SEND_FPS, idempotency_key: newKey('camera'), confidence, cell_grid: gridParam }) }
      catch (reason) { stopShare(); throw reason }
    } else {
      await post('/api/v1/camera/start', { source, idempotency_key: newKey('camera'), confidence, cell_grid: gridParam })
    }
    await load()
    return `Camera đã chạy với ngưỡng ${confidence.toFixed(2)}. Ô compute chung đang được giữ.`
  })
  const stop = () => action.run(async () => {
    try { await post('/api/v1/camera/stop') } finally { stopShare() }
    await load()
    return 'Đã dừng camera và xác nhận các luồng xử lý đã thoát.'
  })

  const [sound, setSound] = useState(true)
  const active = status.active
  // While the camera runs, a new threshold is sent after the slider settles; it applies from the next frame.
  const pendingLive = useRef<number | null>(null)
  useEffect(() => {
    if (!active || status.confidence_threshold == null || Math.abs(status.confidence_threshold - confidence) < 0.001) return
    if (pendingLive.current === confidence) return
    const timer = window.setTimeout(() => {
      pendingLive.current = confidence
      post('/api/v1/camera/confidence', { confidence }).catch((reason) => setActionError(String(reason)))
        .finally(() => { pendingLive.current = null })
    }, 350)
    return () => window.clearTimeout(timer)
  }, [active, confidence, status.confidence_threshold, setActionError])
  const summary = active ? status : status.last_session
  const records = active ? status.records ?? [] : []
  const alertRecords = records.filter((r) => r.alert)
  const polledAlerts = active ? status.alert_track_ids ?? alertRecords.map((r) => r.track_id) : []
  const alerted = polledAlerts
  // Alerts that ride on the frame-upload replies arrive even when polling is throttled in a background tab;
  // they drive the sound and the tab title only, so the on-screen lists stay consistent with the polled records.
  const soundAlerts = active && tabMode && share.alertTracks.length ? [...new Set([...polledAlerts, ...share.alertTracks])] : polledAlerts
  useAlarmSound(sound && soundAlerts.length > 0)
  useTitleAlert(soundAlerts.length > 0)
  // The server session ended by itself (stream error): stop capturing in the browser too.
  const sharing = share.phase === 'starting' || share.phase === 'sending'
  const wasActive = useRef(false)
  useEffect(() => {
    if (active) { wasActive.current = true; return }
    if (wasActive.current) { wasActive.current = false; if (share.phase === 'sending') stopShare() }
  }, [active, share.phase])

  return <div className="page-stack">
    <header className="page-head"><div className="eyebrow">Camera</div><h1>Giám sát trực tiếp</h1>
      <p className="subtitle">Webcam Windows (số 0–15) hoặc URL luồng http/https/rtsp/rtmp. Chỉ dùng model đã duyệt; một lúc chỉ chạy một tác vụ compute trên toàn hệ thống.</p></header>
    <Banner error={action.error} notice={action.notice} />

    <section className={`source-bar ${active || sharing ? 'running' : ''}`}>
      <div className="field wide source-mode">Nguồn
        <div className="segmented" role="group" aria-label="Loại nguồn">
          <button type="button" className={tabMode ? '' : 'on'} disabled={active || sharing} onClick={() => setMode('camera')}>Webcam / URL</button>
          <button type="button" className={tabMode ? 'on' : ''} disabled={active || sharing} onClick={() => setMode('tab')}>Quét tab/cửa sổ</button>
        </div>
      </div>
      {tabMode
        ? <div className="field grow">Tab/cửa sổ cần quét
          <div className="tab-picker">
            <button className="btn pick-btn" disabled={active || sharing || action.busy || !screenShareSupported()} onClick={() => void pickSurface()}><Icon name="camera" size={16} /> Chọn tab để quét…</button>
            <span className="muted tab-label" title={share.surfaceLabel ?? undefined}>{share.surfaceLabel ? `Đã chọn: ${share.surfaceLabel}` : 'Chưa chọn tab nào.'}</span>
          </div>
        </div>
        : <label className="field grow">Nguồn camera<input value={source} disabled={active} onChange={(e) => setSource(e.target.value)} placeholder="0 hoặc http://192.168.1.5:8080/video" /></label>}
      <div className="source-actions">
        {!tabMode && <button className="btn" disabled={active || action.busy || !source.trim()} onClick={() => void doProbe()}><Icon name="link" size={16} /> Kiểm tra kết nối</button>}
        {!active
          ? <button className="btn btn-primary" disabled={action.busy || !readiness?.ready_for_request || (tabMode ? share.phase !== 'ready' : !source.trim())} onClick={() => void start()}><Icon name="play" size={16} /> Bắt đầu</button>
          : <button className="btn btn-danger" disabled={action.busy} onClick={() => void stop()}><Icon name="stop" size={16} /> Dừng</button>}
      </div>
      <div className="source-status">
        <Tag tone={readiness?.ready_for_request ? 'ok' : 'warn'}>{readiness?.ready_for_request ? `model ${readiness.model_version}` : 'chưa sẵn sàng'}</Tag>
        <span className="muted source-note">Thông tin đăng nhập trong URL không được hiển thị hay lưu lại.</span>
      </div>
      <div className="wide">
        <ConfidenceControl value={confidence} onChange={setConfidence}
          hint={active
            ? <>Đổi ngưỡng khi camera đang chạy: áp dụng từ khung kế tiếp (đang dùng {status.confidence_threshold != null ? status.confidence_threshold.toFixed(2) : '—'}).</>
            : <>Ô có điểm thấp hơn ngưỡng không được vẽ và không được tính. Cảnh báo đuối nước vẫn cần ô drowning đủ điểm trong đủ thời gian.</>} />
        {readiness?.cell_grid_supported && <CellGridControl value={cellGrid} onChange={setCellGrid} disabled={active}
          hint={active ? 'Đổi lưới cần dừng rồi bật lại camera.' : undefined} />}
      </div>
      {tabMode && !screenShareSupported() && <div className="callout warn wide">Trình duyệt này chưa hỗ trợ quét tab. Hãy dùng Chrome hoặc Edge.</div>}
      {tabMode && <div className="callout info wide tab-help">Mở video ở một tab khác rồi chọn tab đó. App tự loại trừ chính tab này nên không bị lặp hình. Để cảnh báo chắc chắn đến kịp, hãy mở app thành cửa sổ riêng cạnh cửa sổ video (cửa sổ bị che hoàn toàn có thể bị trình duyệt giảm tốc).</div>}
      {tabMode && share.error && <div className="callout warn wide">{share.error}</div>}
      {probe && <div className="callout ok wide">{probe}</div>}
      {!readiness?.ready_for_request && <div className="callout warn wide">{readiness?.reason ?? 'Cần model đã duyệt, model được tải về máy và PERSON_DETECTOR_WEIGHTS trước khi chạy camera.'}</div>}
    </section>

    {active && <>
      <section className="count-strip" aria-label="Số người trong khung hình hiện tại">
        {COUNT_CARDS.map(({ label, icon, hint }) => {
          const rows = records.filter((r) => r.label === label)
          const hot = label === 'drowning' && (rows.length > 0 || alerted.length > 0)
          return <div key={label} className={`count-card ${hot ? 'hot' : ''}`} style={{ '--tone': labelColor[label] } as CSSProperties}>
            <span className="count-icon"><Icon name={icon} size={18} /></span>
            <div><span className="count-name">{labelVi[label]}</span><b>{rows.length}</b></div>
            <small>{label === 'drowning' && alerted.length ? `cảnh báo ${alerted.map(trackName).join(', ')}` : hint}</small>
          </div>
        })}
      </section>

      <div className="monitor">
        <section className={`monitor-frame ${alerted.length ? 'alerting' : ''}`}>
          <header className="monitor-head">
            <span className={`live-dot ${alerted.length ? 'red' : ''}`} />
            <b className="monitor-source">{status.source === 'browser-tab' ? 'Tab trình duyệt' : /^\d+$/.test(status.source ?? '') ? `Webcam ${status.source}` : status.source}</b>
            <span className="muted">{status.width}×{status.height}</span>
            <Tag tone={alerted.length ? 'bad' : status.state === 'running' ? 'ok' : 'warn'}>{alerted.length ? 'đang cảnh báo' : status.state}</Tag>
            <span className="monitor-metrics">
              {status.source === 'browser-tab' && share.phase === 'sending' && <span title="Khung hình trình duyệt đã gửi tới model">gửi {share.framesSent} khung · ~{share.sendFps ?? '—'}/s</span>}
              <span>{status.processed_fps ?? '—'} khung/s</span>
              <span>trễ {status.latency_ms_p50 ?? '—'} ms</span>
              <span className="mono">{elapsed(status.elapsed_seconds)}</span>
            </span>
          </header>
          <div className="monitor-view">
            <img key={status.job_id} src={api('/api/v1/camera/stream')} alt="Luồng camera đã gắn khung và nhãn" />
          </div>
          <footer className="monitor-foot">
            <LabelLegend />
            <label className="switch"><input type="checkbox" checked={sound} onChange={(e) => setSound(e.target.checked)} /><span /><Icon name={sound ? 'bell' : 'bellOff'} size={15} /> Âm báo</label>
          </footer>
        </section>

        <aside className="monitor-side">
          <AlertBanner tracks={alerted} />
          <div className="side-card">
            <div className="side-head"><b>Cảnh báo hiện tại</b><span className={`pill ${alerted.length ? 'pill-bad' : 'pill-ok'}`}>{alerted.length}</span></div>
            {alertRecords.length === 0
              ? <div className="alert-empty"><Icon name="check" size={22} /><b>Chưa ghi nhận dấu hiệu đuối nước</b><span>Cảnh báo bật khi một ID giữ điểm drowning cao đủ lâu.</span></div>
              : <ul className="alert-cards">{alertRecords.map((r) => <li key={r.track_id} className="alert-card live">
                <div className="alert-card-head"><b>Người {trackName(r.track_id)}</b><span className="pill pill-bad">drowning</span></div>
                <div className="alert-card-meta">
                  <span>điểm drowning <b>{scoreText(r.class_scores?.[0])}</b></span>
                  <span>YOLO <b>{scoreText(r.detector_confidence)}</b></span>
                </div>
                {r.class_scores?.[0] != null && <div className="score-meter" aria-hidden="true"><i style={{ width: `${Math.min(100, r.class_scores[0] * 100)}%` }} /></div>}
                <p className="fine-print">Việc xác nhận cảnh báo vẫn cần người trực quan sát trực tiếp.</p>
              </li>)}</ul>}
          </div>

          <div className="side-card">
            <div className="side-head"><b>Người trong khung hình</b><span className="pill">{records.length}</span></div>
            {records.length === 0 ? <p className="muted">Chưa phát hiện người nào trong khung hình hiện tại.</p>
              : <ul className="track-list">{records.map((r) => <li key={r.track_id} className={r.alert ? 'alert' : undefined}>
                <span className="track-dot" style={{ background: r.alert ? 'var(--drowning)' : labelColor[r.label ?? ''] ?? 'var(--text-3)' }} />
                <div className="track-main">
                  <div><b>{trackName(r.track_id)}</b> <span>{r.label ? labelVi[r.label as Label] ?? r.label : 'chưa phân loại'}</span></div>
                  <small>{branchText[r.prediction_branch] ?? r.prediction_branch} · điểm ô <ScoreBar value={r.detector_confidence} /></small>
                  <ClassScores scores={r.class_scores} />
                </div>
              </li>)}</ul>}
          </div>

          <div className="stat-grid">
            <div><b>{status.processed_fps ?? '—'}</b><span>khung/giây xử lý</span></div>
            <div><b>{status.latency_ms_p50 ?? '—'}</b><span>độ trễ p50 (ms)</span></div>
            <div><b>{status.latency_ms_p95 ?? '—'}</b><span>độ trễ p95 (ms)</span></div>
            <div><b>{status.frames_dropped ?? 0}</b><span>khung bị bỏ qua</span></div>
          </div>
          <p className="fine-print">Tracker: {status.tracker ?? '—'}. ID là ID tạm của tracker, không phải danh tính. Nếu model chậm hơn camera, khung cũ bị bỏ để không tích trễ. Điểm ba nhãn ở thang 0–1, chưa hiệu chuẩn; điểm YOLO tìm người tách riêng.</p>
        </aside>
      </div>
    </>}

    {!active && !summary && <section className="monitor-idle">
      <Icon name="camera" size={34} />
      <b>Camera đang tắt</b>
      <span>Nhập nguồn, bấm “Kiểm tra kết nối” rồi “Bắt đầu”. Khung hình và cảnh báo sẽ hiện ở đây.</span>
    </section>}

    {!active && summary && <Panel kicker="Phiên gần nhất" title="Kết quả phiên camera" subtitle={summary.source}
      right={<Tag tone={summary.state === 'stopped' ? 'ok' : 'bad'}>{summary.state === 'stopped' ? 'đã dừng bình thường' : `lỗi: ${summary.error ?? summary.state}`}</Tag>}>
      <div className="stat-row">
        <div><b>{summary.elapsed_seconds}s</b><span>thời lượng</span></div><div><b>{summary.frames_processed}</b><span>khung đã xử lý</span></div>
        <div><b>{summary.processed_fps ?? '—'}</b><span>khung/giây</span></div><div><b>{summary.latency_ms_p95 ?? '—'} ms</b><span>độ trễ p95</span></div>
      </div>
      <p className="muted">Số liệu tốc độ đo trên máy hiện tại với đúng detector và model đang dùng.</p>
    </Panel>}
  </div>
}
