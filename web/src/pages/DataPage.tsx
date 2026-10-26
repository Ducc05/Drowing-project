import { useCallback, useEffect, useState } from 'react'
import { api, fmt, pct, post, request, useAction, usePolling } from '../api'
import { Banner, Bars, Panel, Tag } from '../ui'

type Requirement = { source_label: string; canonical_label: string; ambiguous_name: boolean; definition_note_required: boolean; min_samples_viewed: number }
type Sheet = { key: string; kind: 'random' | 'smallest'; original_label: string; mapping_action: string; canonical_label: string | null; sample_count: number; total_for_label: number }
type Percentiles = { min: number; p05: number; p50: number; p95: number; max: number } | null
type LabelStat = {
  original_label: string; mapping_action: string; canonical_label: string | null; count: number
  by_original_split: Record<string, number>; box_area_fraction: Percentiles; box_min_side_px: Percentiles
  tiny_box_count: number; tiny_box_fraction: number
}
type Quality = {
  counts: { media: number; observations: number; mapped_observations: number }
  labels: LabelStat[]
  canonical_by_original_split: Record<string, Record<string, number>>
  boxes_per_image: Percentiles
  images_with_multiple_states: number
  images_with_more_than_10_boxes: number
  image_size_px: { distinct: number; most_common: Array<[string, number]> }
  exact_duplicates: { groups: number; extra_images: number; groups_crossing_original_splits: number }
  near_duplicates: {
    checked: boolean; max_hamming_distance?: number; clusters_with_multiple_images?: number; images_in_such_clusters?: number
    largest_cluster?: number; clusters_crossing_original_splits?: number; images_in_cross_split_clusters?: number
    proposed_group_count?: number; cross_split_examples?: Array<Array<{ path: string; split: string }>>
  }
  image_training_readiness: { mapping_review_approved: boolean; source_groups_verified: boolean; classes_present_in_source: string[]; blockers: string[]; ready_for_split_and_crop_building: boolean }
}
type Review = {
  source_id: string; source_version: string; mapping_review: 'approved' | 'stale' | 'pending'; mapping_reviewer: string | null
  grouping_review: 'verified' | 'pending'; requirements: Requirement[]
  task: { state: string; step?: string; error?: string | null }
  sheets: Sheet[] | null; quality: Quality | null; proposed_groups_available: boolean
}
type SourceRow = { source_id: string; source_version: string; format: string; modality: string; root_exists: boolean; root_env: string }

const classNames: Record<string, string> = { drowning: 'Đuối nước', swimming: 'Bình thường trong nước', out_of_water: 'Ngoài nước' }

function SourceReview({ source }: { source: SourceRow }) {
  const [review, setReview] = useState<Review | null>(null)
  const [reviewer, setReviewer] = useState(() => localStorage.getItem('reviewer') ?? '')
  const [seen, setSeen] = useState<Record<string, boolean>>({})
  const [confirmed, setConfirmed] = useState<Record<string, boolean>>({})
  const [notes, setNotes] = useState<Record<string, string>>({})
  const [groupNote, setGroupNote] = useState('')
  const action = useAction()
  const setActionError = action.setError
  const base = `/api/v1/sources/${encodeURIComponent(source.source_id)}/review`

  const load = useCallback(async () => {
    try { setReview(await request<Review>(base)) } catch (reason) { setActionError(String(reason)) }
  }, [base, setActionError])
  const running = review?.task.state === 'running'
  usePolling(load, 2000, running)
  useEffect(() => { localStorage.setItem('reviewer', reviewer) }, [reviewer])

  const prepare = () => action.run(async () => { await post(`${base}/prepare`); await load(); return 'Đang chuẩn bị ảnh mẫu và báo cáo chất lượng. Dữ liệu gốc chỉ được đọc.' })
  const approveMapping = () => action.run(async () => {
    if (!review) return
    const viewed: Record<string, number> = {}
    for (const requirement of review.requirements) {
      viewed[requirement.source_label] = (review.sheets ?? [])
        .filter((sheet) => sheet.original_label === requirement.source_label && seen[sheet.key])
        .reduce((sum, sheet) => sum + sheet.sample_count, 0)
    }
    await post(`${base}/approve-mapping`, {
      reviewer, confirmed_labels: review.requirements.filter((r) => confirmed[r.source_label]).map((r) => r.source_label),
      samples_viewed: viewed, notes,
    })
    await load()
    return 'Đã ghi nhận duyệt ánh xạ nhãn cho đúng bản ánh xạ hiện tại.'
  })
  const approveGroups = () => action.run(async () => {
    await post(`${base}/approve-grouping`, { reviewer, note: groupNote })
    await load()
    return 'Đã ghi nhận nhóm nguồn. Hãy bấm "Chuẩn bị lại" để dựng lại dữ liệu với các nhóm đã xác nhận.'
  })

  const mappedReady = review?.requirements.every((r) =>
    confirmed[r.source_label] && (review.sheets ?? []).some((s) => s.original_label === r.source_label && seen[s.key])
    && (!r.definition_note_required || (notes[r.source_label] ?? '').trim().length >= 20)) ?? false
  const quality = review?.quality
  const near = quality?.near_duplicates

  return <Panel kicker="DỮ LIỆU NGUỒN" title={source.source_id.replaceAll('_', ' ')}
    subtitle={`${source.format} · v${source.source_version} · ${source.modality}`}
    right={<div className="tag-row">
      <Tag tone={review?.mapping_review === 'approved' ? 'ok' : review?.mapping_review === 'stale' ? 'bad' : 'warn'}>
        ánh xạ: {review?.mapping_review === 'approved' ? `đã duyệt${review.mapping_reviewer ? ` (${review.mapping_reviewer})` : ''}` : review?.mapping_review === 'stale' ? 'đã đổi, cần duyệt lại' : 'chưa duyệt'}</Tag>
      <Tag tone={review?.grouping_review === 'verified' ? 'ok' : 'warn'}>nhóm nguồn: {review?.grouping_review === 'verified' ? 'đã xác nhận' : 'chưa xác nhận'}</Tag>
    </div>}>
    <Banner error={action.error} notice={action.notice} />
    {!source.root_exists && <div className="callout warn">Chưa điền hoặc không tìm thấy thư mục dữ liệu: biến <code>{source.root_env}</code> trong <code>.env</code>.</div>}
    <div className="toolbar">
      <button className="primary-button" disabled={!source.root_exists || running || action.busy} onClick={() => void prepare()}>
        {running ? `Đang chạy: ${review?.task.step ?? ''}` : review?.sheets ? 'Chuẩn bị lại' : 'Chuẩn bị tài liệu duyệt'}</button>
      <span className="muted">Tạo ảnh mẫu thật cho từng nhãn, báo cáo chất lượng và đề xuất nhóm ảnh trùng. Không train, không sửa dữ liệu gốc.</span>
    </div>
    {review?.task.state === 'failed' && <div className="callout bad">Chuẩn bị thất bại: {review.task.error}</div>}

    {quality && <>
      <h3 className="section-title">1. Báo cáo chất lượng</h3>
      <div className="stat-row">
        <div><b>{fmt(quality.counts.media)}</b><span>ảnh/video</span></div>
        <div><b>{fmt(quality.counts.observations)}</b><span>khung người được gán nhãn</span></div>
        <div><b>{fmt(quality.images_with_multiple_states)}</b><span>ảnh có nhiều trạng thái khác nhau</span></div>
        <div><b>{fmt(quality.images_with_more_than_10_boxes)}</b><span>ảnh có hơn 10 khung</span></div>
      </div>
      <Bars rows={quality.labels.map((label) => ({ label: `${label.original_label} → ${label.canonical_label ?? label.mapping_action}`, value: label.count,
        note: pct(label.count / Math.max(1, quality.counts.observations)) }))} />
      <div className="table-scroll"><table className="data-table">
        <thead><tr><th>Nhãn gốc</th><th>Split gốc (train / valid / test)</th><th>Diện tích khung (trung vị · p5–p95)</th><th>Cạnh ngắn nhất px (p5 · trung vị)</th><th>Khung quá nhỏ</th></tr></thead>
        <tbody>{quality.labels.map((label) => <tr key={label.original_label}>
          <td>{label.original_label}</td>
          <td>{['train', 'valid', 'test'].map((s) => fmt(label.by_original_split[s])).join(' / ')}</td>
          <td>{label.box_area_fraction ? `${pct(label.box_area_fraction.p50)} · ${pct(label.box_area_fraction.p05)}–${pct(label.box_area_fraction.p95)}` : '—'}</td>
          <td>{label.box_min_side_px ? `${label.box_min_side_px.p05} · ${label.box_min_side_px.p50}` : '—'}</td>
          <td>{fmt(label.tiny_box_count)} ({pct(label.tiny_box_fraction)})</td></tr>)}</tbody></table></div>
      <ul className="findings">
        {(() => {
          const total = Math.max(1, quality.counts.mapped_observations)
          const smallest = [...quality.labels].filter((l) => l.mapping_action === 'map').sort((a, b) => a.count - b.count)[0]
          return smallest && smallest.count / total < 0.1 ? <li className="bad">Lớp ít nhất ({smallest.original_label}) chỉ chiếm {pct(smallest.count / total)} số khung: mất cân bằng mạnh, kết quả theo từng lớp cần được xem riêng.</li> : null
        })()}
        {quality.exact_duplicates.groups > 0 && <li className="warn">{fmt(quality.exact_duplicates.groups)} nhóm ảnh trùng hệt nhau ({fmt(quality.exact_duplicates.extra_images)} ảnh thừa); {quality.exact_duplicates.groups_crossing_original_splits} nhóm nằm ở nhiều split gốc.</li>}
        {near?.checked && (near.clusters_crossing_original_splits ?? 0) > 0 && <li className="bad">Rò rỉ giữa các split gốc: {fmt(near.clusters_crossing_original_splits)} cụm ảnh gần giống nhau nằm ở nhiều split ({fmt(near.images_in_cross_split_clusters)} ảnh). Không dùng split gốc của nguồn để đánh giá; hệ thống sẽ chia lại theo nhóm.</li>}
        <li className="info">Ảnh một kích thước: {quality.image_size_px.most_common.map(([size, n]) => `${size} (${fmt(n)})`).join(', ')}.</li>
      </ul>
    </>}

    {review?.sheets && <>
      <h3 className="section-title">2. Duyệt ánh xạ nhãn bằng ảnh mẫu thật</h3>
      <p className="muted">Ánh xạ nhãn nguồn → nhãn chuẩn chỉ được dùng để train sau khi bạn xem ảnh mẫu và xác nhận. Hệ thống không tự đoán nghĩa của nhãn.</p>
      {review.requirements.map((requirement) => {
        const sheets = review.sheets!.filter((sheet) => sheet.original_label === requirement.source_label)
        return <div className="mapping-card" key={requirement.source_label}>
          <div className="mapping-head"><b>{requirement.source_label}</b><span>→</span><Tag tone="info">{requirement.canonical_label} · {classNames[requirement.canonical_label] ?? ''}</Tag>
            {requirement.ambiguous_name && <Tag tone="warn">tên dễ hiểu nhầm</Tag>}</div>
          <div className="sheet-grid">{sheets.map((sheet) => <figure key={sheet.key}>
            <img loading="lazy" src={api(`${base}/sheets/${sheet.key}`)} alt={`${sheet.original_label}: ${sheet.kind === 'random' ? 'ngẫu nhiên' : 'khung nhỏ nhất'}`} />
            <figcaption><label><input type="checkbox" checked={Boolean(seen[sheet.key])} onChange={(e) => setSeen({ ...seen, [sheet.key]: e.target.checked })} />
              Đã xem bảng {sheet.kind === 'random' ? `${sheet.sample_count} ảnh ngẫu nhiên` : `${sheet.sample_count} khung nhỏ nhất`} (tổng {fmt(sheet.total_for_label)})</label></figcaption>
          </figure>)}</div>
          {requirement.definition_note_required && <label className="field">Định nghĩa bằng lời của nhãn này trong nguồn (tối thiểu 20 ký tự)
            <textarea rows={2} value={notes[requirement.source_label] ?? ''} onChange={(e) => setNotes({ ...notes, [requirement.source_label]: e.target.value })} /></label>}
          <label className="confirm"><input type="checkbox" checked={Boolean(confirmed[requirement.source_label])}
            onChange={(e) => setConfirmed({ ...confirmed, [requirement.source_label]: e.target.checked })} />
            Tôi đã xem ảnh mẫu và xác nhận “{requirement.source_label}” đúng nghĩa là <b>{requirement.canonical_label}</b>.</label>
        </div>
      })}
      <div className="toolbar">
        <label className="field inline">Tên người duyệt<input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="Tên của bạn (lưu vào audit)" /></label>
        <button className="primary-button" disabled={!mappedReady || !reviewer.trim() || action.busy || review.mapping_review === 'approved'} onClick={() => void approveMapping()}>
          {review.mapping_review === 'approved' ? 'Ánh xạ đã được duyệt' : 'Duyệt ánh xạ này'}</button>
      </div>
    </>}

    {near?.checked && <>
      <h3 className="section-title">3. Nhóm nguồn và chống rò rỉ khi chia tập</h3>
      <p className="muted">Ảnh gần giống nhau (cùng một ảnh gốc bị xoay, đổi màu…) phải nằm chung một tập. Hệ thống đề xuất nhóm bằng so khớp ảnh (dHash, ngưỡng {near.max_hamming_distance} bit).
        Đề xuất này <b>không chứng minh</b> các nhóm đến từ video nguồn khác nhau: khi bạn xác nhận, hệ thống ghi tên bạn và lý do, và vẫn ghi “chưa xác minh tách theo video nguồn”.</p>
      <div className="stat-row">
        <div><b>{fmt(near.proposed_group_count)}</b><span>nhóm đề xuất</span></div>
        <div><b>{fmt(near.clusters_with_multiple_images)}</b><span>cụm nhiều ảnh ({fmt(near.images_in_such_clusters)} ảnh)</span></div>
        <div><b>{fmt(near.largest_cluster)}</b><span>cụm lớn nhất</span></div>
        <div><b>{fmt(near.clusters_crossing_original_splits)}</b><span>cụm vắt qua split gốc</span></div>
      </div>
      {near.cross_split_examples && near.cross_split_examples.length > 0 && <details><summary>Ví dụ cụm vắt qua nhiều split gốc</summary>
        <ul className="paths">{near.cross_split_examples.slice(0, 5).map((cluster, index) => <li key={index}>{cluster.map((item) => `${item.split}: ${item.path.split('/').pop()}`).join('  |  ')}</li>)}</ul></details>}
      <label className="field">Cơ sở để coi các nhóm này là tách biệt (tối thiểu 20 ký tự)
        <textarea rows={2} value={groupNote} onChange={(e) => setGroupNote(e.target.value)} placeholder="Ví dụ: đã xem các cặp mẫu vắt qua split, đều là cùng một ảnh gốc." /></label>
      <div className="toolbar">
        <button className="primary-button" disabled={!review?.proposed_groups_available || groupNote.trim().length < 20 || !reviewer.trim() || action.busy || review?.grouping_review === 'verified'} onClick={() => void approveGroups()}>
          {review?.grouping_review === 'verified' ? 'Nhóm đã được xác nhận' : 'Xác nhận nhóm đề xuất'}</button>
      </div>
    </>}

    {quality && <div className={`callout ${quality.image_training_readiness.ready_for_split_and_crop_building ? 'ok' : 'warn'}`}>
      {quality.image_training_readiness.ready_for_split_and_crop_building
        ? 'Nguồn này đủ điều kiện dựng split và crop huấn luyện (ánh xạ đã duyệt, nhóm đã xác nhận).'
        : `Chưa đủ điều kiện train: ${quality.image_training_readiness.blockers.filter((b) => !b.startsWith('source_alone')).join(', ') || 'đang chờ cập nhật'}.`}
    </div>}
  </Panel>
}

export default function DataPage() {
  const [sources, setSources] = useState<SourceRow[]>([])
  const [error, setError] = useState('')
  useEffect(() => {
    request<{ sources: SourceRow[] }>('/api/v1/sources').then((r) => setSources(r.sources.filter((s) => s.source_id))).catch((e) => setError(String(e)))
  }, [])
  return <div className="page-stack">
    <header className="page-head"><div className="eyebrow"><span className="eyebrow-line" /> DỮ LIỆU</div><h1>Duyệt dữ liệu trước khi train</h1>
      <p className="subtitle">Bạn không phải tự gán nhãn. Bạn chỉ xem ảnh mẫu thật, xác nhận nghĩa của nhãn và xác nhận cách nhóm ảnh trùng; mọi thứ gắn với tên người duyệt.</p></header>
    <Banner error={error} />
    {sources.length === 0 && !error && <div className="empty-state">Chưa có nguồn dữ liệu trong configs/datasets.</div>}
    {sources.map((source) => <SourceReview key={source.source_id} source={source} />)}
  </div>
}
