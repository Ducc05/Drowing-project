import { useEffect, useState } from 'react'
import { api, fmt, pct, post, request, SCORE_LABELS as LABELS, useAction } from '../api'
import { Banner, ClassScores, Matrix, Tag } from '../ui'
import { labelColor, labelText, labelVi } from '../overlayData'
import type { Label } from '../overlayData'
import { ScoreBar, ScoreFilter } from '../videoOverlay'

export type Expected = Label | 'not_person'
type BoxMark = {
  detection_index: number; box_number: number; predicted_label: Label | null; score: number | null
  expected_label: Expected; correct: boolean; reviewer: string; note: string
}
type BoxRow = { detection_index: number; box_number: number; label: Label | null; score: number | null; bbox_xyxy: number[]; mark: BoxMark | null }
type SweepRow = { threshold: number; kept: number; tp: number; fp: number; fn: number; precision: number | null; recall: number | null; f1: number | null; measurable: boolean }
export type BoxSummary = {
  reviewed_boxes: number; correct: number; wrong_label: number; not_person: number; missed: number; real_people: number
  precision: number | null; recall: number | null; f1: number | null; mean_score_correct: number | null; mean_score_wrong: number | null
  run_threshold: number | null; threshold_sweep: SweepRow[]; best_threshold_by_f1: number | null
  per_class: Record<Label, { predicted_boxes: number; expected_boxes: number; precision: number | null; recall_of_boxed: number | null }>
  confusion_expected_by_predicted: Record<Expected, Record<Label, number>>; caveat: string; images_reviewed?: number
}
type BoxView = { confidence_threshold: number | null; boxes: BoxRow[]; marks: BoxMark[]; missed: number; summary: BoxSummary }
type Detail = { label: Label | null; class_scores: number[] | null; detector_confidence?: number; prediction_branch: string }

const expectedText = (value: Expected) => value === 'not_person' ? 'không phải người / ô trùng' : labelVi[value]

/** Precision / recall / F1 headline plus the threshold sweep computed from the reviewed boxes. */
export function BoxSummaryView({ summary, highlight }: { summary: BoxSummary; highlight?: number }) {
  const rows = summary.threshold_sweep.filter((row) => row.measurable)
  return <div className="box-summary">
    <div className="stat-row">
      <div><b>{fmt(summary.reviewed_boxes)}</b><span>ô đã đánh giá{summary.images_reviewed != null ? ` · ${summary.images_reviewed} ảnh` : ''}</span></div>
      <div><b>{summary.correct}</b><span>ô đúng</span></div>
      <div><b>{summary.wrong_label + summary.not_person}</b><span>ô sai ({summary.wrong_label} sai nhãn · {summary.not_person} không phải người)</span></div>
      <div><b>{summary.missed}</b><span>người bị bỏ sót (không có ô)</span></div>
      <div><b>{pct(summary.precision)}</b><span>precision (ô đúng / ô đã đánh giá)</span></div>
      <div><b>{pct(summary.recall)}</b><span>recall (ô đúng / người thật)</span></div>
      <div><b>{summary.f1 == null ? '—' : summary.f1.toFixed(3)}</b><span>F1</span></div>
      <div><b>{summary.mean_score_correct == null ? '—' : summary.mean_score_correct.toFixed(2)} / {summary.mean_score_wrong == null ? '—' : summary.mean_score_wrong.toFixed(2)}</b><span>điểm TB ô đúng / ô sai</span></div>
    </div>
    {rows.length > 0 && summary.reviewed_boxes > 0 && <>
      <h4 className="block-title">Nếu nâng ngưỡng confidence thì sao?</h4>
      <p className="muted">Áp lại từng ngưỡng lên chính các ô đã đánh giá. Chỉ tính được từ ngưỡng lúc chạy ({summary.run_threshold?.toFixed(2) ?? '—'}) trở lên, vì ô thấp hơn chưa từng được tạo ra.
        {summary.best_threshold_by_f1 != null && <> F1 cao nhất trên dữ liệu đã đánh giá: <b>ngưỡng {summary.best_threshold_by_f1.toFixed(2)}</b>.</>}</p>
      <div className="table-scroll"><table className="data-table sweep-table">
        <thead><tr><th>Ngưỡng</th><th>Ô giữ lại</th><th>Đúng (TP)</th><th>Sai (FP)</th><th>Sót (FN)</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
        <tbody>{rows.map((row) => <tr key={row.threshold} className={[
          highlight != null && Math.abs(row.threshold - highlight) < 0.001 ? 'current' : '',
          row.threshold === summary.best_threshold_by_f1 ? 'best' : ''].join(' ').trim() || undefined}>
          <td className="mono">{row.threshold.toFixed(2)}{row.threshold === summary.best_threshold_by_f1 && <span className="pill pill-ok">F1 tốt nhất</span>}</td>
          <td>{row.kept}</td><td>{row.tp}</td><td>{row.fp}</td><td>{row.fn}</td>
          <td><ScoreBar value={row.precision} /></td><td><ScoreBar value={row.recall} /></td><td>{row.f1 == null ? '—' : row.f1.toFixed(3)}</td>
        </tr>)}</tbody></table></div>
    </>}
    {summary.reviewed_boxes > 0 && <>
      <h4 className="block-title">Theo từng nhãn</h4>
      <div className="table-scroll"><table className="data-table">
        <thead><tr><th>Nhãn</th><th>Model dự đoán</th><th>Precision của nhãn</th><th>Bạn xác nhận</th><th>Recall trong số ô</th></tr></thead>
        <tbody>{LABELS.map((label) => <tr key={label}><td><i className="dot" style={{ background: labelColor[label] }} />{labelText[label]}</td>
          <td>{summary.per_class[label].predicted_boxes}</td><td>{pct(summary.per_class[label].precision)}</td>
          <td>{summary.per_class[label].expected_boxes}</td><td>{pct(summary.per_class[label].recall_of_boxed)}</td></tr>)}</tbody></table></div>
      <Matrix labels={[...LABELS]} rowTitle="bạn xác nhận" colTitle="model dự đoán"
        rows={LABELS.map((e) => LABELS.map((p) => summary.confusion_expected_by_predicted[e][p]))} />
      {summary.not_person > 0 && <p className="muted">Ô không phải người theo nhãn model dự đoán: {LABELS.map((p) => `${labelText[p]} ${summary.confusion_expected_by_predicted.not_person[p]}`).join(' · ')}</p>}
    </>}
    <p className="fine-print">{summary.caveat === 'Only boxes someone reviewed are counted; this is a review log, not a benchmark.'
      ? 'Chỉ tính các ô bạn đã đánh giá: đây là nhật ký đánh giá, không phải benchmark trên dữ liệu chưa xem.' : summary.caveat}</p>
  </div>
}

/** Image result reviewed box by box: every box has its own number and score; the reviewer judges each one. */
export function ImageBoxReview({ jobId, reviewer, onChanged }: { jobId: string; reviewer: string; onChanged?: () => void }) {
  const [annotated, setAnnotated] = useState('')
  const [details, setDetails] = useState<Detail[]>([])
  const [view, setView] = useState<BoxView | null>(null)
  const [size, setSize] = useState<[number, number] | null>(null)
  const [selected, setSelected] = useState<number | null>(null)
  const [minScore, setMinScore] = useState<number | null>(null)
  const [missed, setMissed] = useState(0)
  const action = useAction()
  const setActionError = action.setError

  useEffect(() => {
    let alive = true
    Promise.all([
      request<{ files: Record<string, string> }>(`/api/v1/jobs/${jobId}/results`),
      request<{ detections: Detail[] }>(`/api/v1/jobs/${jobId}/results/detections`),
      request<BoxView>(`/api/v1/jobs/${jobId}/box-marks`),
    ]).then(([media, doc, boxes]) => {
      if (!alive) return
      setAnnotated(media.files.annotated ?? ''); setDetails(doc.detections); setView(boxes); setMissed(boxes.missed)
    }).catch((reason) => { if (alive) setActionError(String(reason)) })
    return () => { alive = false }
  }, [jobId, setActionError])

  const judge = (row: BoxRow, expected: Expected) => action.run(async () => {
    setView(await post<BoxView>(`/api/v1/jobs/${jobId}/box-marks`, { reviewer, detection_index: row.detection_index, expected_label: expected, note: '' }))
    onChanged?.()
  })
  const undo = (row: BoxRow) => action.run(async () => {
    setView(await request<BoxView>(`/api/v1/jobs/${jobId}/box-marks/${row.detection_index}`, { method: 'DELETE' }))
    onChanged?.()
  })
  const saveMissed = () => action.run(async () => {
    setView(await request<BoxView>(`/api/v1/jobs/${jobId}/box-marks/missed`, { method: 'PUT', body: JSON.stringify({ reviewer, missed }) }))
    onChanged?.()
    return 'Đã lưu số người bị bỏ sót.'
  })

  const viewThreshold = minScore ?? view?.confidence_threshold ?? 0.05
  const hidden = (row: BoxRow) => row.score != null && row.score < viewThreshold
  const boxes = view?.boxes ?? []
  const reviewed = boxes.filter((row) => row.mark).length

  return <div className="box-review">
    <Banner error={action.error} notice={action.notice} />
    <div className="image-review">
      <div className="box-canvas">
        {annotated && <img className="image-review-preview" src={api(annotated)} alt="Ảnh kết quả, mỗi ô có số thứ tự, nhãn và điểm"
          onLoad={(event) => setSize([event.currentTarget.naturalWidth, event.currentTarget.naturalHeight])} />}
        {size && <svg className="box-layer" viewBox={`0 0 ${size[0]} ${size[1]}`} preserveAspectRatio="xMidYMid meet">
          {boxes.map((row) => {
            const [x1, y1, x2, y2] = row.bbox_xyxy
            const state = hidden(row) ? 'hidden' : row.mark ? (row.mark.correct ? 'correct' : 'wrong') : 'open'
            return <g key={row.detection_index} className={`box-shape ${state} ${selected === row.detection_index ? 'selected' : ''}`}
              onClick={() => setSelected(row.detection_index)}>
              <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} />
            </g>
          })}
        </svg>}
      </div>
      <div className="box-side">
        <ScoreFilter value={viewThreshold} runThreshold={view?.confidence_threshold} onChange={setMinScore} />
        <p className="muted">Bấm vào một ô trên ảnh để chọn dòng tương ứng. Ô mờ: điểm dưới bộ lọc. Viền xanh: bạn xác nhận đúng; đỏ: sai.</p>
        <div className="missed-row">
          <label className="field">Số người model bỏ sót (không có ô)<input type="number" min={0} max={500} value={missed}
            onChange={(e) => setMissed(Math.max(0, Math.min(500, Math.round(Number(e.target.value) || 0))))} /></label>
          <button className="btn btn-small" disabled={!reviewer.trim() || action.busy || missed === view?.missed} onClick={() => void saveMissed()}>Lưu</button>
        </div>
      </div>
    </div>

    <h4 className="block-title">Đánh giá từng ô · {reviewed}/{boxes.length} ô đã đánh giá</h4>
    {!reviewer.trim() && <p className="muted">Điền tên người duyệt ở phía trên để ghi đánh giá.</p>}
    {boxes.length === 0 ? <p className="muted">Model không tạo ô nào ở ngưỡng này. Nếu ảnh có người, hãy nhập số người bị bỏ sót.</p>
      : <div className="table-scroll"><table className="data-table box-table">
        <thead><tr><th>Ô</th><th>Model dự đoán</th><th>Điểm ô</th><th>Điểm ba nhãn</th><th>Bạn đánh giá</th><th /></tr></thead>
        <tbody>{boxes.map((row) => {
          const detail = details[row.detection_index]
          const others: Expected[] = [...LABELS.filter((label) => label !== row.label), 'not_person']
          return <tr key={row.detection_index} className={[selected === row.detection_index ? 'selected' : '', hidden(row) ? 'dimmed' : ''].join(' ').trim() || undefined}
            onClick={() => setSelected(row.detection_index)}>
            <td className="mono"><b>#{row.box_number}</b></td>
            <td>{row.label ? <><i className="dot" style={{ background: labelColor[row.label] }} />{labelText[row.label]}</> : 'chưa phân loại'}</td>
            <td><ScoreBar value={row.score} /></td>
            <td>{detail && <ClassScores scores={detail.class_scores} />}</td>
            <td>{row.mark ? (row.mark.correct ? <Tag tone="ok">đúng</Tag> : <Tag tone="bad">sai → {expectedText(row.mark.expected_label)}</Tag>)
              : <span className="muted">chưa đánh giá</span>}</td>
            <td className="row-actions" onClick={(event) => event.stopPropagation()}>
              <button className="btn btn-small" disabled={!reviewer.trim() || action.busy || !row.label} onClick={() => row.label && void judge(row, row.label)}>Đúng</button>
              <select className="judge-select" aria-label={`Ô #${row.box_number} sai, nhãn đúng là`} value="" disabled={!reviewer.trim() || action.busy}
                onChange={(event) => { if (event.target.value) void judge(row, event.target.value as Expected) }}>
                <option value="">Sai → …</option>
                {others.map((value) => <option key={value} value={value}>{expectedText(value)}</option>)}
              </select>
              {row.mark && <button className="review-button" disabled={action.busy} onClick={() => void undo(row)}>Bỏ</button>}
            </td>
          </tr>
        })}</tbody></table></div>}

    {view && (view.summary.reviewed_boxes > 0 || view.summary.missed > 0) && <>
      <h4 className="block-title">Kết quả đánh giá ảnh này</h4>
      <BoxSummaryView summary={view.summary} highlight={viewThreshold} />
    </>}
  </div>
}
