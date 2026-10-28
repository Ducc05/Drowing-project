import { useEffect, useState } from 'react'
import { api, post, request } from './api'

type Job = { id: string; idempotency_key?: string; state: string; progress?: { frame: number; total: number } }
const finished = new Set(['completed', 'sync_pending', 'cancelled', 'failed'])

/** The latest processed frame, not a second video inference request. */
export function LiveVideoPreview({ requestKey, allowStop = false }: { requestKey: string; allowStop?: boolean }) {
  const [job, setJob] = useState<Job>()
  const [src, setSrc] = useState('')
  const [error, setError] = useState('')
  const [stopping, setStopping] = useState(false)
  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>, objectUrl = '', current: Job | undefined, lastStatus = 0
    const poll = async () => {
      try {
        if (Date.now() - lastStatus >= 2000) {
          const result = await request<{ jobs: Job[] }>('/api/v1/jobs', { signal: controller.signal })
          current = result.jobs.find((row) => row.idempotency_key === requestKey)
          lastStatus = Date.now()
          if (!controller.signal.aborted) setJob(current)
        }
        if (current && !finished.has(current.state)) {
          const response = await fetch(api(`/api/v1/test/jobs/${current.id}/preview`), { cache: 'no-store', signal: controller.signal })
          if (response.status === 200) {
            const blob = await response.blob()
            if (!controller.signal.aborted) {
              const next = URL.createObjectURL(blob)
              setSrc(next)
              if (objectUrl) URL.revokeObjectURL(objectUrl)
              objectUrl = next
            }
          } else if (response.status !== 204) {
            throw new Error(`Không tải được hình xem trực tiếp (${response.status}).`)
          }
        }
        if (!controller.signal.aborted) setError('')
      } catch (reason) {
        if (!controller.signal.aborted) setError(String(reason))
      } finally {
        if (!controller.signal.aborted && (!current || !finished.has(current.state))) timer = setTimeout(() => void poll(), 400)
      }
    }
    void poll()
    return () => { controller.abort(); clearTimeout(timer); if (objectUrl) URL.revokeObjectURL(objectUrl) }
  }, [requestKey])

  const stop = async () => {
    if (!job) return
    setStopping(true)
    try { await post(`/api/v1/test/jobs/${job.id}/stop`) }
    catch (reason) { setError(String(reason)) }
    finally { setStopping(false) }
  }
  const ended = job && finished.has(job.state)
  return <section className="live-video-preview" aria-label="Video đang được model xử lý">
    <div className="toolbar"><b>{ended ? 'Đã kết thúc xử lý' : 'Xem model gán nhãn trực tiếp'}</b>
      {job?.progress && <span className="muted">Khung {job.progress.frame}/{job.progress.total}</span>}
      {allowStop && job?.state === 'running' && <button className="btn btn-small btn-danger" disabled={stopping} onClick={() => void stop()}>{stopping ? 'Đang dừng…' : 'Dừng video'}</button>}
    </div>
    {src ? <img src={src} alt="Khung video vừa xử lý với nhãn và khung nhận diện" />
      : <div className="empty-state">{job?.state === 'queued' ? 'Video đang chờ suất xử lý.' : ended ? 'Không có hình xem trước.' : 'Đang tải video và chuẩn bị model…'}</div>}
    <p className="muted">{ended ? `Trạng thái: ${job.state}.` : 'Hình cập nhật theo tốc độ xử lý của model. Nhãn từng khung hình được hiển thị để theo dõi; cảnh báo cần đủ bằng chứng qua nhiều khung.'}</p>
    {error && <p role="status" className="message error">{error}</p>}
  </section>
}
