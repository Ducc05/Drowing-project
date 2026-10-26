"""SQLAlchemy schema for the shared PostgreSQL registry and execution slot."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Index, Integer, JSON, LargeBinary, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def now_utc() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class DatasetSource(Base):
    __tablename__ = "dataset_sources"
    source_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    descriptor: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, onupdate=now_utc, nullable=False)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"
    __table_args__ = (UniqueConstraint("source_id", "source_version", "fingerprint", name="uq_dataset_snapshot"),)
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    source_id: Mapped[str] = mapped_column(ForeignKey("dataset_sources.source_id", ondelete="RESTRICT"), nullable=False)
    source_version: Mapped[str] = mapped_column(String(100), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    mapping_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    audit_artifact_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    review_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, nullable=False)


class TrainingRecipe(Base):
    __tablename__ = "training_recipes"
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    recipe_id: Mapped[str] = mapped_column(String(160), unique=True, nullable=False)
    recipe_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, nullable=False)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_job_idempotency_key"),
        CheckConstraint("job_type IN ('train', 'evaluation', 'inference', 'prepare')", name="ck_job_type"),
        CheckConstraint("execution_token >= 0", name="ck_job_fencing_token"),
        Index("ix_jobs_state_created", "state", "created_at"),
    )
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    job_type: Mapped[str] = mapped_column(String(24), nullable=False)
    state: Mapped[str] = mapped_column(String(32), default="created", nullable=False)
    phase: Mapped[str] = mapped_column(String(32), default="queued", nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(200), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    owner_id: Mapped[str | None] = mapped_column(String(160))
    execution_token: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    remote_ref: Mapped[str | None] = mapped_column(String(300))
    remote_version: Mapped[str | None] = mapped_column(String(100))
    remote_status: Mapped[str | None] = mapped_column(String(60))
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_summary: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, onupdate=now_utc, nullable=False)


class JobEvent(Base):
    __tablename__ = "job_events"
    __table_args__ = (UniqueConstraint("job_id", "sequence", name="uq_job_event_sequence"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    from_state: Mapped[str | None] = mapped_column(String(32))
    to_state: Mapped[str] = mapped_column(String(32), nullable=False)
    event_type: Mapped[str] = mapped_column(String(80), nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, nullable=False)


class ExecutionSlot(Base):
    __tablename__ = "execution_slot"
    __table_args__ = (CheckConstraint("slot_id = 1", name="ck_singleton_execution_slot"),)
    slot_id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    job_id: Mapped[UUID | None] = mapped_column(ForeignKey("jobs.id", ondelete="RESTRICT"))
    owner_id: Mapped[str | None] = mapped_column(String(160))
    fencing_token: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reconciliation_status: Mapped[str] = mapped_column(String(32), default="clear", nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, onupdate=now_utc, nullable=False)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("model_family", "display_version", name="uq_model_family_version"),)
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    model_family: Mapped[str] = mapped_column(String(160), nullable=False)
    display_version: Mapped[str] = mapped_column(String(100), nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    class_keys: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    temporal_trained: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    completion_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="candidate", nullable=False)
    parent_version_id: Mapped[UUID | None] = mapped_column(ForeignKey("model_versions.id", ondelete="RESTRICT"))
    training_job_id: Mapped[UUID | None] = mapped_column(ForeignKey("jobs.id", ondelete="RESTRICT"))
    dataset_fingerprint: Mapped[str | None] = mapped_column(String(64))
    artifact_id: Mapped[UUID | None] = mapped_column(Uuid(as_uuid=True))
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, nullable=False)


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    model_family: Mapped[str] = mapped_column(String(160), primary_key=True)
    approved_model_id: Mapped[UUID | None] = mapped_column(ForeignKey("model_versions.id", ondelete="RESTRICT"))
    generation: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, onupdate=now_utc, nullable=False)


class Worker(Base):
    __tablename__ = "workers"
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    installation_id: Mapped[str] = mapped_column(String(160), unique=True, nullable=False)
    code_version: Mapped[str] = mapped_column(String(100), nullable=False)
    cached_model_ids: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    loaded_model_id: Mapped[UUID | None] = mapped_column(ForeignKey("model_versions.id", ondelete="SET NULL"))
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, nullable=False)


class Artifact(Base):
    __tablename__ = "artifacts"
    __table_args__ = (UniqueConstraint("job_id", "role", "sha256", name="uq_job_artifact_hash"),)
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(ForeignKey("jobs.id", ondelete="RESTRICT"), nullable=False)
    role: Mapped[str] = mapped_column(String(80), nullable=False)
    storage_provider: Mapped[str] = mapped_column(String(40), default="google_drive", nullable=False)
    remote_file_id: Mapped[str | None] = mapped_column(String(300))
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="pending", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, nullable=False)


class ArtifactUpload(Base):
    __tablename__ = "artifact_uploads"
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    artifact_id: Mapped[UUID] = mapped_column(ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False)
    upload_key: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(24), default="started", nullable=False)
    uploaded_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, onupdate=now_utc, nullable=False)


class Evaluation(Base):
    __tablename__ = "evaluations"
    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    model_version_id: Mapped[UUID] = mapped_column(ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False)
    evaluation_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    result_artifact_id: Mapped[UUID | None] = mapped_column(ForeignKey("artifacts.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, nullable=False)


class ReviewDecision(Base):
    __tablename__ = "review_decisions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_version_id: Mapped[UUID] = mapped_column(ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False)
    reviewer_id: Mapped[str] = mapped_column(String(160), nullable=False)
    decision: Mapped[str] = mapped_column(String(24), nullable=False)
    comment: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now_utc, nullable=False)
