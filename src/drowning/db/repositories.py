"""Transactional job, event and singleton execution-slot operations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from drowning.db.models import Artifact, ExecutionSlot, Job, JobEvent, ModelRegistry, ModelVersion, ReviewDecision, now_utc
from drowning.domain.jobs import JobState, can_release_execution_slot, validate_transition


class ExecutionSlotBusy(RuntimeError):
    pass


class StaleExecutionOwner(RuntimeError):
    pass


class ModelReviewError(RuntimeError):
    pass


def create_job(session: Session, *, job_type: str, idempotency_key: str, payload: dict[str, Any]) -> tuple[Job, bool]:
    if not idempotency_key.strip():
        raise ValueError("idempotency_key is required")
    existing = session.scalar(select(Job).where(Job.idempotency_key == idempotency_key))
    if existing:
        return existing, False
    job = Job(job_type=job_type, state=JobState.QUEUED.value, phase="queued",
              idempotency_key=idempotency_key, payload=payload)
    session.add(job)
    try:
        session.flush()
        session.add(JobEvent(job_id=job.id, sequence=1, from_state=JobState.CREATED.value,
                             to_state=JobState.QUEUED.value, event_type="job_queued", details={}))
        session.commit()
        session.refresh(job)
        return job, True
    except IntegrityError:
        session.rollback()
        existing = session.scalar(select(Job).where(Job.idempotency_key == idempotency_key))
        if existing:
            return existing, False
        raise


def assert_execution_owner(session: Session, job_id: UUID, owner_id: str, token: int, *,
                            require_slot: bool = False) -> Job:
    if require_slot:
        slot = session.scalar(select(ExecutionSlot).where(ExecutionSlot.slot_id == 1).with_for_update()
                              .execution_options(populate_existing=True))
        if slot is None or (slot.job_id, slot.owner_id, slot.fencing_token) != (job_id, owner_id, token):
            raise StaleExecutionOwner("compute slot no longer matches this job/owner/fencing token")
    job = session.scalar(select(Job).where(Job.id == job_id).with_for_update()
                         .execution_options(populate_existing=True))
    if job is None or (job.owner_id, job.execution_token) != (owner_id, token):
        raise StaleExecutionOwner("job ownership or fencing token has changed")
    return job


def transition_job(session: Session, job_id: UUID, target: str | JobState, *, event_type: str,
                   details: dict[str, Any] | None = None, phase: str | None = None,
                   expected_owner_id: str | None = None, expected_execution_token: int | None = None,
                   require_slot: bool = False) -> Job:
    if (expected_owner_id is None) != (expected_execution_token is None):
        raise ValueError("owner and execution token must be provided together")
    job = (assert_execution_owner(session, job_id, expected_owner_id, expected_execution_token, require_slot=require_slot)
           if expected_owner_id is not None else
           session.scalar(select(Job).where(Job.id == job_id).with_for_update()))
    if job is None:
        raise KeyError(f"unknown job {job_id}")
    old, new = validate_transition(job.state, target)
    sequence = (session.scalar(select(func.max(JobEvent.sequence)).where(JobEvent.job_id == job_id)) or 0) + 1
    job.state = new.value
    if phase is not None:
        job.phase = phase
    job.updated_at = now_utc()
    session.add(JobEvent(job_id=job.id, sequence=sequence, from_state=old.value, to_state=new.value,
                         event_type=event_type, details=details or {}))
    session.commit()
    session.refresh(job)
    return job


def claim_execution_slot(session: Session, job_id: UUID, owner_id: str, *, lease_seconds: int = 90) -> int:
    if lease_seconds <= 0 or not owner_id:
        raise ValueError("owner_id and a positive lease_seconds are required")
    slot = session.scalar(select(ExecutionSlot).where(ExecutionSlot.slot_id == 1).with_for_update())
    if slot is None:
        raise RuntimeError("execution slot singleton is missing; apply database migrations first")
    if slot.job_id is not None:
        raise ExecutionSlotBusy(f"compute slot is held by job {slot.job_id}")
    job = session.scalar(select(Job).where(Job.id == job_id).with_for_update())
    if job is None:
        raise KeyError(f"unknown job {job_id}")
    if job.state != JobState.QUEUED.value:
        raise ValueError("only queued jobs can claim the compute slot")
    slot.fencing_token += 1
    slot.job_id = job.id
    slot.owner_id = owner_id
    slot.lease_until = datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)
    slot.reconciliation_status = "owned"
    slot.updated_at = now_utc()
    job.owner_id = owner_id
    job.execution_token = slot.fencing_token
    session.commit()
    return slot.fencing_token


def release_execution_slot(session: Session, job_id: UUID, owner_id: str, token: int, *,
                           compute_stopped_confirmed: bool) -> None:
    slot = session.scalar(select(ExecutionSlot).where(ExecutionSlot.slot_id == 1).with_for_update())
    if slot is None or slot.job_id != job_id or slot.owner_id != owner_id or slot.fencing_token != token:
        raise StaleExecutionOwner("job/owner/fencing token no longer owns the execution slot")
    job = session.get(Job, job_id)
    if job is None or not can_release_execution_slot(job.state, compute_stopped_confirmed=compute_stopped_confirmed):
        raise ValueError("slot release requires a terminal job state and confirmed compute stop")
    slot.job_id = None
    slot.owner_id = None
    slot.lease_until = None
    slot.reconciliation_status = "clear"
    slot.updated_at = now_utc()
    session.commit()


def review_model_version(session: Session, model_version_id: UUID, *, decision: str,
                         reviewer_id: str, comment: str = "") -> ModelVersion:
    if decision not in {"approve", "reject"} or not reviewer_id.strip():
        raise ValueError("decision must be approve/reject and reviewer_id is required")
    model = session.scalar(select(ModelVersion).where(ModelVersion.id == model_version_id).with_for_update())
    if model is None:
        raise KeyError(f"unknown model version {model_version_id}")
    if model.status != "candidate":
        raise ModelReviewError("only a candidate model can be reviewed")
    if decision == "approve":
        if model.completion_kind != "candidate" or not model.temporal_trained:
            raise ModelReviewError("only a completed temporal candidate can be approved")
        artifact = session.get(Artifact, model.artifact_id) if model.artifact_id else None
        if (artifact is None or artifact.status != "ready" or not artifact.remote_file_id
                or artifact.storage_provider != "google_drive" or len(artifact.sha256) != 64 or artifact.size_bytes <= 0):
            raise ModelReviewError("a verified, ready Drive artifact is required before approval")
        dialect = session.get_bind().dialect.name
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        elif dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert
        else:
            raise ModelReviewError("model approval requires a PostgreSQL-compatible registry transaction")
        session.execute(insert(ModelRegistry).values(model_family=model.model_family, approved_model_id=None,
                                                     generation=0, updated_at=now_utc()).on_conflict_do_nothing(
                                                         index_elements=[ModelRegistry.model_family]))
        registry = session.scalar(select(ModelRegistry).where(ModelRegistry.model_family == model.model_family).with_for_update())
        previous_id = registry.approved_model_id
        if previous_id and previous_id != model.id:
            previous = session.get(ModelVersion, previous_id)
            if previous is not None and previous.status == "approved":
                previous.status = "superseded"
        registry.approved_model_id = model.id
        registry.generation += 1
        registry.updated_at = now_utc()
        model.status = "approved"
    else:
        model.status = "rejected"
    session.add(ReviewDecision(model_version_id=model.id, reviewer_id=reviewer_id.strip(),
                               decision=decision, comment=comment))
    session.commit()
    session.refresh(model)
    return model
