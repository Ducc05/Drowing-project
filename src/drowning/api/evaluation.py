"""Human review of model output: mark video segments right or wrong, then summarize per model.

Marks are the user's judgement of what the model produced on videos they chose, so the summary is a
review log, not a benchmark: it says how many segments and videos were looked at and nothing about
videos nobody opened.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from drowning.data.labels import CANONICAL_LABELS
from drowning.db.models import Job, ModelVersion, ReviewDecision

CLASS_ORDER = ["drowning", "swimming", "out_of_water"]


class MarkError(ValueError):
    pass


def _detections_doc(job: Job, result_file) -> dict[str, Any]:
    try:
        path: Path = result_file(job.id, "detections")
    except (KeyError, FileNotFoundError) as exc:
        raise MarkError("The detections file is not stored on this installation, so segments cannot be checked") from exc
    return json.loads(path.read_text(encoding="utf-8"))


def _detections(job: Job, result_file) -> list[dict[str, Any]]:
    return _detections_doc(job, result_file).get("detections", [])


def presence_for(doc: dict[str, Any], start_ms: int, end_ms: int) -> dict[str, Any]:
    """What the model produced inside [start_ms, end_ms), whether or not it found anyone."""
    inside = [r for r in doc.get("detections", []) if start_ms <= r.get("timestamp_ms", -1) < end_ms]
    per_frame = Counter(r.get("frame_index") for r in inside)
    fps, stride = float(doc.get("fps_metadata") or 0), max(1, int(doc.get("frame_stride") or 1))
    expected = 0
    if fps > 0:  # frames the pipeline actually looked at in this interval (every ``stride``-th)
        expected = sum(1 for f in range(math.ceil(start_ms * fps / 1000), math.ceil(end_ms * fps / 1000)) if f % stride == 0)
    expected = max(expected, len(per_frame), 1)
    labels = Counter(r["label"] for r in inside if r.get("label"))
    return {"detected_boxes": len(inside), "detected_people_max": max(per_frame.values(), default=0),
            "frames_with_detection": len(per_frame), "expected_frames": expected,
            "detection_rate": round(min(1.0, len(per_frame) / expected), 3),
            "predicted_label": labels.most_common(1)[0][0] if labels else None, "label_votes": dict(labels)}


def summarize_presence(marks: list[dict[str, Any]]) -> dict[str, Any]:
    """Misses and false alarms of the detector, from intervals the reviewer marked as having N people (or none)."""
    with_people = [m for m in marks if m["people"] > 0]
    empty = [m for m in marks if m["people"] == 0]

    def rate(hits: int, total: int) -> float | None:
        return round(hits / total, 4) if total else None
    labelled = [m for m in with_people if m.get("expected_label") and m.get("predicted_label")]
    return {"intervals": len(marks), "with_people": len(with_people), "empty": len(empty),
            "interval_recall_any_person": rate(sum(1 for m in with_people if m["detected_people_max"] >= 1), len(with_people)),
            "interval_recall_full_count": rate(sum(1 for m in with_people if m["detected_people_max"] >= m["people"]), len(with_people)),
            "mean_frame_detection_rate": (round(sum(m["detection_rate"] for m in with_people) / len(with_people), 4)
                                          if with_people else None),
            "empty_intervals_with_detections": sum(1 for m in empty if m["detected_boxes"] > 0),
            "empty_false_alarm_rate": rate(sum(1 for m in empty if m["detected_boxes"] > 0), len(empty)),
            "label_accuracy_when_detected": rate(sum(1 for m in labelled if m["expected_label"] == m["predicted_label"]), len(labelled)),
            "label_checked_intervals": len(labelled),
            "caveat": "Counts only the intervals someone chose to mark, so it is a review log, not a benchmark of unseen videos."}


def predicted_for(records: list[dict[str, Any]], track_id: str, start_ms: int, end_ms: int) -> dict[str, Any]:
    selected = [r for r in records if r.get("track_id") == track_id and start_ms <= r.get("timestamp_ms", -1) < end_ms]
    if not selected:
        raise MarkError("The model produced no detection for that track in that interval")
    labels = Counter(r["label"] for r in selected if r.get("label"))
    branches = Counter(r.get("prediction_branch") for r in selected)
    label, votes = labels.most_common(1)[0]
    return {"predicted_label": label, "frames": len(selected), "label_votes": dict(labels),
            "prediction_branches": dict(branches), "label_agreement": round(votes / len(selected), 3)}


class ReviewMarks:
    def __init__(self, session_factory: sessionmaker):
        self.factory = session_factory

    def _video_job(self, session, job_id: UUID) -> Job:
        job = session.scalar(select(Job).where(Job.id == job_id).with_for_update())
        if job is None:
            raise KeyError(str(job_id))
        payload = job.payload or {}
        if job.job_type != "inference" or payload.get("modality") != "video" or not payload.get("result_files"):
            raise MarkError("Only finished video inference jobs with stored results can be reviewed")
        if job.state not in {"completed", "sync_pending"}:
            raise MarkError("The job has not finished")
        return job

    def list(self, job_id: UUID) -> dict[str, Any]:
        with self.factory() as session:
            job = self._video_job(session, job_id)
            marks = (job.payload or {}).get("review_marks", [])
            return {"job_id": str(job.id), "model_version_id": job.payload.get("model_version_id"),
                    "marks": marks, "summary": summarize(marks)}

    def add(self, job_id: UUID, result_file, *, reviewer: str, track_id: str, start_ms: int, end_ms: int,
            verdict: str, expected_label: str | None, note: str) -> dict[str, Any]:
        if not reviewer.strip():
            raise MarkError("reviewer is required")
        if verdict not in {"correct", "wrong"}:
            raise MarkError("verdict must be correct or wrong")
        if start_ms < 0 or end_ms <= start_ms:
            raise MarkError("the interval must satisfy 0 <= start < end (milliseconds, end exclusive)")
        if expected_label is not None and expected_label not in CANONICAL_LABELS:
            raise MarkError("expected_label must be drowning, swimming or out_of_water")
        with self.factory() as session:
            job = self._video_job(session, job_id)
            computed = predicted_for(_detections(job, result_file), track_id, start_ms, end_ms)
            if verdict == "correct":
                expected_label = computed["predicted_label"]
            mark = {"id": uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(), "reviewer": reviewer.strip(),
                    "track_id": track_id, "start_ms": start_ms, "end_ms": end_ms, "verdict": verdict,
                    "expected_label": expected_label, "note": note.strip()[:1000], **computed,
                    "track_identity_verified": False}
            payload = dict(job.payload)
            payload["review_marks"] = [*payload.get("review_marks", []), mark]
            job.payload = payload
            session.commit()
            return mark

    def presence_list(self, job_id: UUID) -> dict[str, Any]:
        with self.factory() as session:
            job = self._video_job(session, job_id)
            marks = (job.payload or {}).get("presence_marks", [])
            return {"job_id": str(job.id), "marks": marks, "summary": summarize_presence(marks)}

    def presence_add(self, job_id: UUID, result_file, *, reviewer: str, start_ms: int, end_ms: int, people: int,
                     expected_label: str | None, note: str) -> dict[str, Any]:
        """Record 'in this interval there are N people' independently of what the model found (so misses can be counted)."""
        if not reviewer.strip():
            raise MarkError("reviewer is required")
        if start_ms < 0 or end_ms <= start_ms:
            raise MarkError("the interval must satisfy 0 <= start < end (milliseconds, end exclusive)")
        if not isinstance(people, int) or isinstance(people, bool) or not 0 <= people <= 50:
            raise MarkError("people must be a whole number from 0 to 50")
        if expected_label is not None and expected_label not in CANONICAL_LABELS:
            raise MarkError("expected_label must be drowning, swimming or out_of_water")
        if people == 0 and expected_label is not None:
            raise MarkError("an interval with nobody in it has no expected label")
        with self.factory() as session:
            job = self._video_job(session, job_id)
            computed = presence_for(_detections_doc(job, result_file), start_ms, end_ms)
            mark = {"id": uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(), "reviewer": reviewer.strip(),
                    "start_ms": start_ms, "end_ms": end_ms, "people": people, "expected_label": expected_label,
                    "note": note.strip()[:1000], **computed}
            payload = dict(job.payload)
            payload["presence_marks"] = [*payload.get("presence_marks", []), mark]
            job.payload = payload
            session.commit()
            return mark

    def presence_delete(self, job_id: UUID, mark_id: str) -> dict[str, Any]:
        with self.factory() as session:
            job = self._video_job(session, job_id)
            marks = (job.payload or {}).get("presence_marks", [])
            kept = [m for m in marks if m["id"] != mark_id]
            if len(kept) == len(marks):
                raise KeyError(mark_id)
            job.payload = {**job.payload, "presence_marks": kept}
            session.commit()
            return {"deleted": mark_id, "remaining": len(kept)}

    def delete(self, job_id: UUID, mark_id: str) -> dict[str, Any]:
        with self.factory() as session:
            job = self._video_job(session, job_id)
            marks = (job.payload or {}).get("review_marks", [])
            kept = [m for m in marks if m["id"] != mark_id]
            if len(kept) == len(marks):
                raise KeyError(mark_id)
            job.payload = {**job.payload, "review_marks": kept}
            session.commit()
            return {"deleted": mark_id, "remaining": len(kept)}


def summarize(marks: list[dict[str, Any]]) -> dict[str, Any]:
    wrong = [m for m in marks if m["verdict"] == "wrong"]
    known = [m for m in marks if m.get("expected_label")]
    matrix = {e: {p: 0 for p in CLASS_ORDER} for e in CLASS_ORDER}
    for mark in known:
        matrix[mark["expected_label"]][mark["predicted_label"]] += 1
    per_class = {}
    for label in CLASS_ORDER:
        expected_total = sum(matrix[label].values())
        per_class[label] = {"reviewed_segments": expected_total,
                            "recall": round(matrix[label][label] / expected_total, 4) if expected_total else None}
    return {"segments": len(marks), "correct": len(marks) - len(wrong), "wrong": len(wrong),
            "wrong_without_expected_label": sum(1 for m in wrong if not m.get("expected_label")),
            "confusion_expected_by_predicted": matrix, "per_class": per_class,
            "total_reviewed_ms": sum(m["end_ms"] - m["start_ms"] for m in marks),
            "reviewers": sorted({m["reviewer"] for m in marks}),
            "caveat": "A review log of the segments someone chose to mark; not a benchmark of unseen videos."}


def model_evaluation(session_factory: sessionmaker, model_id: UUID) -> dict[str, Any]:
    with session_factory() as session:
        model = session.get(ModelVersion, model_id)
        if model is None:
            raise KeyError(str(model_id))
        jobs = [j for j in session.scalars(select(Job).where(Job.job_type == "inference"))
                if (j.payload or {}).get("model_version_id") == str(model_id)]
        decisions = session.scalars(select(ReviewDecision).where(ReviewDecision.model_version_id == model_id)
                                    .order_by(ReviewDecision.created_at)).all()
        marks = [m for j in jobs for m in (j.payload or {}).get("review_marks", [])]
        videos = [j for j in jobs if (j.payload or {}).get("modality") == "video"]
        return {
            "model_version_id": str(model.id), "display_version": model.display_version, "status": model.status,
            "training_metrics": model.metrics or {},
            "training_metrics_note": "Reported by the Kaggle run on its validation split. Not a measurement on your videos.",
            "test_runs": {"videos": len(videos), "images": sum(1 for j in jobs if (j.payload or {}).get("modality") == "image"),
                          "videos_with_review_marks": sum(1 for j in videos if (j.payload or {}).get("review_marks"))},
            "human_review": summarize(marks),
            "presence_review": summarize_presence([m for j in jobs for m in (j.payload or {}).get("presence_marks", [])]),
            "decisions": [{"decision": d.decision, "reviewer": d.reviewer_id, "comment": d.comment,
                           "at": d.created_at.isoformat()} for d in decisions]}


def imported_model_evaluation(session_factory: sessionmaker, model_id: UUID, display_version: str) -> dict[str, Any]:
    """Review summary for imported weights that have no training run or registry review to attribute."""
    with session_factory() as session:
        jobs = [j for j in session.scalars(select(Job).where(Job.job_type == "inference"))
                if (j.payload or {}).get("model_version_id") == str(model_id)]
        marks = [m for job in jobs for m in (job.payload or {}).get("review_marks", [])]
        videos = [job for job in jobs if (job.payload or {}).get("modality") == "video"]
        return {
            "model_version_id": str(model_id), "display_version": display_version, "status": "active",
            "training_metrics": {},
            "training_metrics_note": "Imported H20Saver weights; this project has no training metrics to attribute to them.",
            "test_runs": {"videos": len(videos),
                          "images": sum(1 for job in jobs if (job.payload or {}).get("modality") == "image"),
                          "videos_with_review_marks": sum(1 for job in videos if (job.payload or {}).get("review_marks"))},
            "human_review": summarize(marks),
            "presence_review": summarize_presence([m for job in jobs for m in (job.payload or {}).get("presence_marks", [])]),
            "decisions": []}
