"""Explicit Kaggle job dispatch and exact-version reconciliation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from drowning.db.models import Job
from drowning.db.repositories import assert_execution_owner, claim_execution_slot, release_execution_slot, transition_job
from drowning.domain.jobs import JobState
from drowning.integrations.kaggle import KaggleCLIClient, KaggleError, RemoteStateUnconfirmed, validate_checkpoint_dataset

_PUSH_VERSION = re.compile(r"(?:kernel\s+)?version(?:\s+number)?\s*[:#]?\s*(\d+)", re.IGNORECASE)
# The CLI has printed this in several shapes across versions; none of them has been checked against a live
# account yet, so any unrecognized output leaves the job remote_unconfirmed (the slot stays held).
_REMOTE_STATUS_PATTERNS = (
    re.compile(r"kernel\s+worker\s+status\s*:\s*\"?(?:KernelWorkerStatus\.)?([A-Za-z_]+)", re.IGNORECASE),
    re.compile(r"has\s+status\s+\"?(?:KernelWorkerStatus\.)?([A-Za-z_]+)", re.IGNORECASE),
)
_REMOTE_STATUS_MAP = {"running": "running", "queued": "running", "cancelrequested": "running",
                      "complete": "completed", "completed": "completed", "error": "failed", "failed": "failed",
                      "cancelled": "cancelled", "canceled": "cancelled", "cancelacknowledged": "cancelled"}


@dataclass(frozen=True)
class DispatchReceipt:
    job_id: str
    state: str
    kernel_ref: str
    kernel_version: str | None
    fencing_token: int
    detail: str


def parse_pushed_version(stdout: str) -> str | None:
    match = _PUSH_VERSION.search(stdout)
    return match.group(1) if match else None


def parse_remote_status(stdout: str) -> str | None:
    for pattern in _REMOTE_STATUS_PATTERNS:
        match = pattern.search(stdout)
        if match:
            return _REMOTE_STATUS_MAP.get(match.group(1).casefold().replace("_", ""))
    return None


class KaggleJobDispatcher:
    """One-command dispatcher; ambiguous pushes keep the global execution slot held."""

    def __init__(self, session_factory: sessionmaker, client: KaggleCLIClient, *, worker_id: str):
        if not worker_id:
            raise ValueError("worker_id is required")
        self.session_factory = session_factory
        self.client = client
        self.worker_id = worker_id

    def _transition(self, session, job_id, target, **kwargs):
        return transition_job(session, job_id, target, expected_owner_id=self.worker_id,
                              expected_execution_token=self._tokens[job_id], require_slot=True, **kwargs)

    def _validate_job(self, job: Job, kernel_dir: Path) -> str:
        if job.state != JobState.QUEUED.value:
            raise ValueError("only queued jobs can be dispatched")
        if job.job_type not in {"prepare", "train"}:
            raise ValueError("Kaggle executor accepts prepare/train jobs only")
        payload = job.payload or {}
        if payload.get("executor", "kaggle") != "kaggle":
            raise ValueError("training executor must be kaggle")
        mode = payload.get("mode", "prepare_only")
        if mode not in {"prepare_only", "remote_smoke", "train_candidate"}:
            raise ValueError("unsupported Kaggle run mode")
        if mode == "train_candidate":
            preflight = payload.get("preflight", {})
            deferred_inputs = (preflight.get("runtime_data_validation") is True
                               and payload.get("dataset_identity_pending") is True
                               and re.fullmatch(r"[a-f0-9]{64}", payload.get("dataset_request_fingerprint") or ""))
            if not (payload.get("new_version") is True and preflight.get("ready") is True
                    and (preflight.get("review_mode") == "advisory"
                         or (preflight.get("mapping_review_approved") is True
                             and preflight.get("source_video_disjointness_verified") is True))
                    and (preflight.get("temporal_window_count", 0) > 0
                         or (payload.get("options") or {}).get("branch") == "image_only"
                         or preflight.get("runtime_data_validation") is True)
                    and (payload.get("dataset_fingerprint") or deferred_inputs)):
                raise ValueError("train_candidate requires explicit new version, review policy and verified or runtime-validated dataset identity")
        metadata_path = kernel_dir / "kernel-metadata.json"
        if not metadata_path.is_file():
            raise ValueError("kernel directory must contain kernel-metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("is_private") is not True:
            raise ValueError("Kaggle kernel metadata must set is_private=true")
        code_file = metadata.get("code_file")
        if not isinstance(code_file, str):
            raise ValueError("Kaggle kernel metadata requires code_file")
        resolved_code = (kernel_dir / code_file).resolve()
        try:
            resolved_code.relative_to(kernel_dir.resolve())
        except ValueError as exc:
            raise ValueError("Kaggle kernel code_file must stay inside kernel directory") from exc
        if not resolved_code.is_file():
            raise ValueError("Kaggle kernel code_file does not exist")
        allowed_files = {metadata_path.resolve(), resolved_code}
        if any(path.is_file() and path.resolve() not in allowed_files for path in kernel_dir.rglob("*")):
            raise ValueError("Kaggle kernel directory may contain only its notebook and kernel-metadata.json")
        notebook = json.loads(resolved_code.read_text(encoding="utf-8"))
        notebook_mode = notebook.get("metadata", {}).get("drowning_project", {}).get("run_mode_default")
        if notebook_mode != mode:
            raise ValueError("Kaggle notebook run mode does not match the durable job payload")
        if mode == "train_candidate" and (payload.get("preflight") or {}).get("runtime_data_validation"):
            if notebook.get("metadata", {}).get("drowning_project", {}).get("runtime_data_validation") is not True:
                raise ValueError("direct training requires a notebook that validates its training inputs")
        parent = notebook.get("metadata", {}).get("drowning_project", {}).get("parent_model")
        if parent != payload.get("parent_model"):
            raise ValueError("Kaggle notebook parent checkpoint does not match the durable job payload")
        if (payload.get("options") or {}).get("continue_from_approved") and not parent:
            raise ValueError("continuation job is missing its pinned parent checkpoint")
        parent_ref = notebook.get("metadata", {}).get("drowning_project", {}).get("parent_dataset_ref")
        if parent_ref != payload.get("parent_dataset_ref"):
            raise ValueError("parent dataset reference does not match the durable job payload")
        if parent:
            if parent_ref not in metadata.get("dataset_sources", []):
                raise ValueError("parent checkpoint dataset is not attached to the notebook")
            validate_checkpoint_dataset(kernel_dir.parent / (kernel_dir.name + "-parent"), parent_ref, parent)
        notebook_options = notebook.get("metadata", {}).get("drowning_project", {}).get("options", {})
        if any(notebook_options.get(key) != value for key, value in (payload.get("options") or {}).items()):
            raise ValueError("Kaggle notebook training options do not match the durable job payload")
        return self.client.validate_kernel_ref(str(metadata.get("id", "")))

    def dispatch(self, job_id: UUID, kernel_dir: Path, *, timeout_seconds: int = 3600) -> DispatchReceipt:
        with self.session_factory() as session:
            job = session.scalar(select(Job).where(Job.id == job_id).with_for_update())
            if job is None:
                raise KeyError(f"unknown job {job_id}")
            kernel_ref = self._validate_job(job, kernel_dir)
            parent_model = (job.payload or {}).get("parent_model")
            parent_dataset_ref = (job.payload or {}).get("parent_dataset_ref")
        # Check the local CLI before taking the shared compute slot.
        self.client.version()
        with self.session_factory() as session:
            token = claim_execution_slot(session, job_id, self.worker_id)
            self._tokens = {job_id: token}
            job = self._transition(session, job_id, JobState.DISPATCHING, event_type="kaggle_push_started",
                                 phase="dispatching", details={"kernel_ref": kernel_ref, "fencing_token": token})
            job = assert_execution_owner(session, job_id, self.worker_id, token, require_slot=True)
            job.remote_ref = kernel_ref
            job.remote_status = "push_started"
            session.commit()
        if parent_model:
            try:
                self.client.push_private_checkpoint_dataset(kernel_dir.parent / (kernel_dir.name + "-parent"),
                                                             parent_dataset_ref, parent_model)
            except Exception as exc:
                # A dataset upload may have reached Kaggle, but no compute kernel was submitted.
                with self.session_factory() as session:
                    self._transition(session, job_id, JobState.FAILED, event_type="parent_dataset_upload_failed",
                                     phase="checkpoint_upload_failed", details={"error_type": type(exc).__name__, "kernel_pushed": False})
                    release_execution_slot(session, job_id, self.worker_id, token, compute_stopped_confirmed=True)
                return DispatchReceipt(str(job_id), JobState.FAILED.value, kernel_ref, None, token,
                                       "parent dataset upload failed; no kernel was pushed")
        try:
            receipt = self.client.push_private_kernel(kernel_dir, timeout_seconds=timeout_seconds)
            version = parse_pushed_version(receipt.stdout)
            details = {"push_exit_code": receipt.returncode, "version_receipt_present": version is not None}
            status_message = "push_accepted_version_unconfirmed" if version else "push_accepted_version_missing"
        except RemoteStateUnconfirmed:
            version = None
            details = {"push_timeout": True, "reconcile_before_retry": True}
            status_message = "push_timeout_remote_unconfirmed"
        except KaggleError:
            # A nonzero CLI result may occur after a server accepted the upload.
            version = None
            details = {"push_error": True, "reconcile_before_retry": True}
            status_message = "push_error_remote_unconfirmed"
        except Exception as exc:
            # After dispatch begins, preserve the global slot if the remote receipt is ambiguous.
            version = None
            details = {"push_error_type": type(exc).__name__, "reconcile_before_retry": True}
            status_message = "push_exception_remote_unconfirmed"
        with self.session_factory() as session:
            job = assert_execution_owner(session, job_id, self.worker_id, token, require_slot=True)
            if job is None or job.state != JobState.DISPATCHING.value or job.owner_id != self.worker_id:
                raise RuntimeError("job ownership changed while Kaggle push was in flight")
            job.remote_version = version
            job.remote_status = status_message
            session.commit()
            self._transition(session, job_id, JobState.REMOTE_UNCONFIRMED,
                           event_type="kaggle_push_receipt", phase="reconcile_remote", details=details)
        return DispatchReceipt(str(job_id), JobState.REMOTE_UNCONFIRMED.value,
                               kernel_ref, version, token, status_message)

    def reconcile(self, job_id: UUID) -> DispatchReceipt:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise KeyError(f"unknown job {job_id}")
            if job.state not in {JobState.REMOTE_UNCONFIRMED.value, JobState.RUNNING.value}:
                raise ValueError("only running/remote_unconfirmed jobs require Kaggle reconciliation")
            current_state = JobState(job.state)
            kernel_ref, version, token = job.remote_ref, job.remote_version, job.execution_token
            assert_execution_owner(session, job_id, self.worker_id, token, require_slot=True)
            self._tokens = {job_id: token}
        if not kernel_ref or not version:
            if current_state == JobState.RUNNING:
                return DispatchReceipt(str(job_id), current_state.value, kernel_ref or "", version, token,
                                       "exact kernel version is unknown; slot remains held and push must not be retried")
            return DispatchReceipt(str(job_id), JobState.REMOTE_UNCONFIRMED.value,
                                   kernel_ref or "", version, token,
                                   "exact kernel version is unknown; slot remains held and push must not be retried")
        result = self.client.status(f"{kernel_ref}/{version}")
        remote_status = parse_remote_status(result.stdout)
        if remote_status is None:
            return DispatchReceipt(str(job_id), JobState.REMOTE_UNCONFIRMED.value, kernel_ref,
                                   version, token, "remote status did not contain a recognized runtime state")
        with self.session_factory() as session:
            job = assert_execution_owner(session, job_id, self.worker_id, token, require_slot=True)
            if job is None or JobState(job.state) not in {JobState.REMOTE_UNCONFIRMED, JobState.RUNNING}:
                raise RuntimeError("job state changed during Kaggle reconciliation")
            job.remote_status = remote_status
            session.commit()
            if remote_status == "running":
                if job.state == JobState.REMOTE_UNCONFIRMED.value:
                    job = self._transition(session, job_id, JobState.RUNNING, event_type="kaggle_remote_running",
                                         phase="remote_compute", details={"kernel_version": version})
            elif remote_status == "completed":
                job = self._transition(session, job_id, JobState.COLLECTING, event_type="kaggle_remote_completed",
                                     phase="collecting_exact_version", details={"kernel_version": version})
            elif remote_status == "failed":
                job = self._transition(session, job_id, JobState.FAILED, event_type="kaggle_remote_failed",
                                     phase="compute_stopped", details={"kernel_version": version})
                release_execution_slot(session, job_id, self.worker_id, token, compute_stopped_confirmed=True)
            else:
                if job.state == JobState.RUNNING.value:
                    job = self._transition(session, job_id, JobState.CANCELLATION_REQUESTED,
                                         event_type="kaggle_remote_cancel_observed", phase="cancellation_requested",
                                         details={"kernel_version": version})
                job = self._transition(session, job_id, JobState.CANCELLED, event_type="kaggle_remote_cancelled",
                                     phase="compute_stopped", details={"kernel_version": version})
                release_execution_slot(session, job_id, self.worker_id, token, compute_stopped_confirmed=True)
        return DispatchReceipt(str(job_id), job.state, kernel_ref, version, token, remote_status)
