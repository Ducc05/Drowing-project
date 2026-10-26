import { useEffect, useState } from 'react'
import { fmt, pct, request } from '../api'
import { Bars, Tag } from '../ui'

type SourceEntry = {
  source_id: string; source_version: string; status: string; error?: string; mapping_review?: string; grouping_review?: string
  input?: { status: string; root_relative?: string; candidates?: string[] }
  scan?: { media_count: number; annotation_count: number; malformed_annotation_count: number }
  audit?: { valid: boolean; issue_count: number; exact_duplicate_groups: number; same_box_label_conflicts: number }
  quality?: { near_duplicates?: Record<string, number | boolean | string>; readiness?: { blockers: string[]; classes_present_in_source: string[] } }
  review_sheet_count?: number
  temporal?: { window_count: number; verified_track_count: number; windows_by_label: Record<string, number>; blockers: string[] }
}
type LabelStat = { original_label: string; mapping_action: string; canonical_label: string | null; count: number; by_original_split: Record<string, number>; tiny_box_count: number }
type Quality = { labels: LabelStat[]; counts: { media: number; observations: number }; images_with_multiple_states: number }
type Report = {
  summary: { run_mode: string }
  report: null | {
    run_mode: string; started_at?: string; finished_at?: string; blockers?: string[]; sources?: SourceEntry[]
    environment: { python?: string; torch?: string; cuda_available?: boolean; gpu?: string | null }
    device?: string; gpu_confirmed?: boolean; forward_backward_seconds?: number; all_trainable_parameters_received_gradients?: boolean
  }
  sources_detail: Record<string, { quality: Quality }>
  stored_on_this_installation: boolean
  drive_file_id?: string | null
}

const blockerText: Record<string, string> = {
  mapping_review_not_approved: 'ánh xạ nhãn chưa được bạn duyệt',
  source_groups_not_verified: 'nhóm nguồn chưa được bạn xác nhận',
  no_temporal_video_source_prepared: 'chưa có nguồn video có nhãn (cần để train nhánh temporal)',
  no_temporal_windows_meet_the_sampling_contract: 'video có nhãn nhưng không tạo được cửa sổ temporal nào',
  input_missing: 'không tìm thấy dataset trên Kaggle', input_ambiguous: 'có nhiều thư mục khớp, hệ thống không đoán',
  canonical_audit_failed: 'kiểm tra cấu trúc dữ liệu thất bại', prepare_failed: 'xử lý nguồn này thất bại',
}
function explain(blocker: string) {
  const [source, reason] = blocker.includes(':') ? [blocker.split(':')[0], blocker.split(':').slice(1).join(':')] : ['', blocker]
  return `${source ? `${source}: ` : ''}${blockerText[reason] ?? reason}`
}
function duration(start?: string, end?: string) {
  if (!start || !end) return '—'
  const seconds = Math.round((new Date(end).getTime() - new Date(start).getTime()) / 1000)
  return seconds < 120 ? `${seconds} giây` : `${Math.round(seconds / 60)} phút`
}
const classText: Record<string, string> = { drowning: 'Đuối nước', swimming: 'Bình thường trong nước', out_of_water: 'Ngoài nước' }

function SourceBlock({ entry, detail }: { entry: SourceEntry; detail?: { quality: Quality } }) {
  const quality = detail?.quality
  const near = entry.quality?.near_duplicates
  const total = Math.max(1, quality?.counts.observations ?? 1)
  return <div className="mapping-card">
    <div className="mapping-head"><b>{entry.source_id}</b><Tag tone={entry.status === 'prepared' ? 'ok' : 'bad'}>{entry.status === 'prepared' ? 'đã xử lý' : entry.status}</Tag>
      <Tag tone={entry.mapping_review === 'approved' ? 'ok' : 'warn'}>ánh xạ: {entry.mapping_review === 'approved' ? 'đã duyệt' : 'chưa duyệt'}</Tag>
      <Tag tone={entry.grouping_review === 'verified' ? 'ok' : 'warn'}>nhóm: {entry.grouping_review === 'verified' ? 'đã xác nhận' : 'chưa xác nhận'}</Tag></div>
    {entry.error && <div className="callout bad">{entry.error}</div>}
    {entry.input?.root_relative && <p className="muted">Tìm thấy trên Kaggle tại <code>{entry.input.root_relative}</code></p>}
    {entry.scan && <div className="stat-row">
      <div><b>{fmt(entry.scan.media_count)}</b><span>ảnh/video</span></div>
      <div><b>{fmt(entry.scan.annotation_count)}</b><span>khung người được gán nhãn</span></div>
      <div><b>{fmt(entry.scan.malformed_annotation_count)}</b><span>dòng nhãn lỗi</span></div>
      <div><b>{entry.audit?.valid ? 'hợp lệ' : 'có lỗi'}</b><span>kiểm tra cấu trúc ({fmt(entry.audit?.issue_count)} vấn đề)</span></div>
      <div><b>{fmt(entry.audit?.exact_duplicate_groups)}</b><span>nhóm ảnh trùng hệt</span></div>
      <div><b>{fmt(entry.audit?.same_box_label_conflicts)}</b><span>khung trùng nhưng nhãn mâu thuẫn</span></div>
    </div>}
    {quality && <>
      <h4>Số khung theo nhãn</h4>
      <Bars rows={quality.labels.map((label) => ({ label: `${label.original_label} → ${classText[label.canonical_label ?? ''] ?? label.mapping_action}`, value: label.count, note: pct(label.count / total) }))} />
      <div className="table-scroll"><table className="data-table"><thead><tr><th>Nhãn gốc</th><th>train / valid / test (split gốc)</th><th>Khung quá nhỏ</th></tr></thead>
        <tbody>{quality.labels.map((label) => <tr key={label.original_label}><td>{label.original_label}</td>
          <td>{['train', 'valid', 'test'].map((k) => fmt(label.by_original_split[k])).join(' / ')}</td><td>{fmt(label.tiny_box_count)}</td></tr>)}</tbody></table></div>
    </>}
    {near && (near.checked as boolean) && <div className="callout warn">
      Ảnh gần giống nhau: <b>{fmt(Number(near.clusters_with_multiple_images))}</b> cụm ({fmt(Number(near.images_in_such_clusters))} ảnh), cụm lớn nhất {fmt(Number(near.largest_cluster))} ảnh.
      {Number(near.clusters_crossing_original_splits) > 0 && <> <b>{fmt(Number(near.clusters_crossing_original_splits))} cụm nằm xuyên các split gốc</b> ({fmt(Number(near.images_in_cross_split_clusters))} ảnh): không dùng split gốc để đánh giá.</>}
    </div>}
    {entry.temporal && <div className="callout info">Video: {fmt(entry.temporal.verified_track_count)} người có ID đã xác minh, {fmt(entry.temporal.window_count)} cửa sổ temporal (chưa chia tập).
      {Object.keys(entry.temporal.windows_by_label).length > 0 && <> Theo nhãn: {Object.entries(entry.temporal.windows_by_label).map(([k, v]) => `${classText[k] ?? k} ${fmt(v)}`).join(', ')}.</>}</div>}
    {entry.review_sheet_count != null && <p className="muted">{entry.review_sheet_count} bảng ảnh mẫu đã được tạo; xem và duyệt ở trang <a href="#data">Dữ liệu</a>.</p>}
  </div>
}

export default function RunReportView({ jobId }: { jobId: string }) {
  const [data, setData] = useState<Report | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    request<Report>(`/api/v1/jobs/${jobId}/run-report`).then(setData).catch((reason) => setError(String(reason)))
  }, [jobId])
  if (error) return <div className="callout warn">{error}</div>
  if (!data) return <div className="muted">Đang tải báo cáo…</div>
  const report = data.report
  if (!report) return <div className="callout warn">Báo cáo đầy đủ không có trên máy này{data.drive_file_id ? ' (đã lưu trên Drive)' : ''}; chỉ có bản tóm tắt.</div>
  const env = report.environment
  return <div className="report page-stack">
    <div className="stat-row">
      <div><b>{report.run_mode === 'remote_smoke' ? 'Kiểm tra GPU' : 'Chuẩn bị dữ liệu'}</b><span>loại lần chạy</span></div>
      <div><b>{duration(report.started_at, report.finished_at)}</b><span>thời gian chạy trên Kaggle</span></div>
      <div><b>{env.cuda_available ? (env.gpu ?? 'có GPU') : 'không có GPU'}</b><span>GPU · torch {env.torch ?? '—'} · Python {env.python ?? '—'}</span></div>
      <div><b>{data.drive_file_id ? 'đã lưu' : 'chưa'}</b><span>bản sao trên Google Drive</span></div>
    </div>
    {report.run_mode === 'remote_smoke' && <div className="callout ok">
      Thiết bị chạy: <b>{report.device}</b> · GPU xác nhận: <b>{report.gpu_confirmed ? 'có' : 'KHÔNG'}</b> · mọi tham số nhận gradient: <b>{report.all_trainable_parameters_received_gradients ? 'có' : 'không'}</b> · một lượt tiến/lùi {report.forward_backward_seconds}s. Chỉ là kiểm tra bằng dữ liệu ngẫu nhiên, chưa tạo model.</div>}
    {(report.sources ?? []).map((entry) => <SourceBlock key={entry.source_id} entry={entry} detail={data.sources_detail[entry.source_id]} />)}
    {(report.blockers ?? []).length > 0 && <div>
      <h4>Còn chặn bước tiếp theo</h4>
      <ul className="findings">{report.blockers!.map((b) => <li className="warn" key={b}>{explain(b)}</li>)}</ul></div>}
    <p className="muted">Thống kê mô tả nhãn như đã xuất, không phải độ chính xác của model.</p>
  </div>
}
