"""Durable local media tests using the shared compute slot and pinned approved model."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from drowning.api.inference import (InferenceCancelled, get_approved_model, load_approved_pipeline,
                                    run_image_test, run_video_test, runtime_weights_missing)
from drowning.api.media import persist_upload, safe_local_root
from drowning.config import Settings
from drowning.db.models import Artifact, Job, ModelVersion
from drowning.db.repositories import (ExecutionSlotBusy, StaleExecutionOwner, assert_execution_owner,
                                      claim_execution_slot, create_job, release_execution_slot, transition_job)
from drowning.domain.jobs import JobState
from drowning.installation import installation_id
from drowning.integrations.drive import sha256_file


class MediaRequestConflict(ValueError):
    pass


class MediaSlotBusy(RuntimeError):
    def __init__(self, job_id: UUID):
        self.job_id = job_id
        super().__init__("Compute slot is busy; the uploaded job remains queued. Retry with the same file and Idempotency-Key.")


class LocalInferenceFailed(RuntimeError):
    def __init__(self, job_id: UUID, error_type: str):
        self.job_id = job_id
        super().__init__(f"Local inference failed ({error_type}); see the stored job event.")


class LocalInferenceCancelled(RuntimeError):
    def __init__(self, job_id: UUID):
        self.job_id = job_id
        super().__init__(f"Local inference was cancelled (job {job_id}).")


class LocalMediaTestJobs:
    def __init__(self, session_factory: sessionmaker, settings: Settings, *,
                 inference_factory: Callable[[ModelVersion, dict[str, Any], Path], Any] | None = None,
                 drive_factory: Callable[[], Any] | None = None):
        self.factory = session_factory
        self.settings = settings
        self.inference_factory = inference_factory
        self.drive_factory = drive_factory
        self._fences: dict[UUID, tuple[str, int]] = {}
        self._stop: set[UUID] = set()
        self._preview_lock = threading.Lock()
        self._preview: tuple[UUID, bytes] | None = None
        self.installation = installation_id(Path(settings.input_cache_dir).expanduser().resolve().parent)

    def _owned(self, session, job_id):
        owner, token = self._fences[job_id]
        return assert_execution_owner(session, job_id, owner, token)

    def _job(self, job_id: UUID) -> Job:
        with self.factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise KeyError(str(job_id))
            return job

    def _transition(self, job_id: UUID, state: JobState, phase: str, event: str,
                    details: dict[str, Any] | None = None) -> Job:
        with self.factory() as session:
            owner, token = self._fences[job_id]
            return transition_job(session, job_id, state, phase=phase, event_type=event, details=details or {},
                                  expected_owner_id=owner, expected_execution_token=token)

    def _payload(self, job_id: UUID, **values: Any) -> Job:
        with self.factory() as session:
            job = (self._owned(session, job_id) if job_id in self._fences else
                   session.scalar(select(Job).where(Job.id == job_id).with_for_update()))
            if job is None:
                raise KeyError(str(job_id))
            job.payload = {**job.payload, **values}
            session.commit()
            session.refresh(job)
            return job

    def _release(self, job_id: UUID) -> None:
        with self.factory() as session:
            owner, token = self._fences[job_id]
            release_execution_slot(session, job_id, owner, token,
                                   compute_stopped_confirmed=True)
        self._fences.pop(job_id, None)
        self._clear_preview(job_id)

    def _clear_preview(self, job_id: UUID) -> None:
        with self._preview_lock:
            if self._preview is not None and self._preview[0] == job_id:
                self._preview = None

    def _publish_preview(self, job_id: UUID, jpeg: bytes) -> None:
        with self._preview_lock:
            self._preview = (job_id, jpeg)

    def preview(self, job_id: UUID) -> bytes | None:
        with self._preview_lock:
            if self._preview is not None and self._preview[0] == job_id:
                return self._preview[1]
        job = self._job(job_id)
        if job.job_type != "inference" or job.payload.get("modality") != "video":
            raise KeyError(str(job_id))
        return None

    def _drive(self):
        if self.drive_factory is not None:
            return self.drive_factory()
        from drowning.integrations.drive import GoogleDriveArtifactAdapter, build_drive_service
        return GoogleDriveArtifactAdapter(build_drive_service(Path(self.settings.google_oauth_token_file)))

    def _sync_files(self, job: Job) -> dict[str, str]:
        if not self.settings.drive_folder_id:
            raise MediaRequestConflict("Google Drive output storage is not configured")
        drive = self._drive()
        result = {}
        for role in job.payload["result_files"]:
            path = self.result_file(job.id, role)
            digest = sha256_file(path)
            receipt = drive.upload(
                path, folder_id=self.settings.drive_folder_id, job_id=str(job.id),
                attempt_id=f"attempt-{job.execution_token}", role=f"result_{role}",
                artifact_key=f"{job.id}/attempt-{job.execution_token}/result-{role}")
            if receipt.get("verified") is not True:
                raise RuntimeError("Drive did not return a verified artifact receipt")
            with self.factory() as session:
                self._owned(session, job.id)
                existing = session.scalar(select(Artifact).where(
                    Artifact.job_id == job.id, Artifact.role == f"result_{role}", Artifact.sha256 == digest))
                if existing is None:
                    session.add(Artifact(job_id=job.id, role=f"result_{role}", storage_provider="google_drive",
                                         remote_file_id=receipt["file_id"], sha256=digest,
                                         size_bytes=path.stat().st_size, status="ready"))
                    session.commit()
            result[role] = receipt["file_id"]
        return result

    def submit(self, upload: UploadFile, modality: str, idempotency_key: str,
               model_version_id: UUID | None = None, *, frame_stride: int = 3) -> dict[str, Any]:
        if modality == "video" and not 1 <= frame_stride <= 10:
            raise MediaRequestConflict("frame_stride must be between 1 and 10")
        if len(idempotency_key.strip()) < 8:
            raise MediaRequestConflict("Idempotency-Key must contain at least eight characters")
        temp_path, original_name, digest, size = persist_upload(upload, self.settings.input_cache_dir,
                                                               modality=modality)
        try:
            with self.factory() as session:
                existing = session.scalar(select(Job).where(Job.idempotency_key == idempotency_key))
                if existing is not None:
                    if (existing.job_type != "inference" or existing.payload.get("input_sha256") != digest
                            or existing.payload.get("modality") != modality):
                        raise MediaRequestConflict("Idempotency-Key is already associated with a different request")
                    stored_stride = existing.payload.get("frame_stride")
                    if modality == "video" and stored_stride is not None and int(stored_stride) != frame_stride:
                        raise MediaRequestConflict("Idempotency-Key is already associated with a different frame_stride")
                    if existing.state != JobState.QUEUED.value:
                        return {**self.result(existing.id), "created": False}
                pinned = (UUID(existing.payload["model_version_id"])
                          if existing and existing.payload.get("model_version_id") else model_version_id)
                explicit = bool(existing.payload.get("evaluation")) if existing else model_version_id is not None
                model_record, _, _ = get_approved_model(session, self.settings, pinned_model_id=pinned,
                                                        allow_candidate=explicit)
            missing = None if self.inference_factory is not None else runtime_weights_missing(self.settings, model_record)
            if missing:
                raise MediaRequestConflict(f"{missing}; local test cannot start")
            with self.factory() as session:
                job, created = create_job(session, job_type="inference", idempotency_key=idempotency_key,
                                          payload={"executor": "local", "modality": modality,
                                                   "input_name": original_name, "input_sha256": digest,
                                                   "input_size_bytes": size, "model_version_id": str(model_record.id),
                                                   "frame_stride": frame_stride if modality == "video" else None,
                                                   "installation_id": self.installation, "pid": os.getpid(),
                                                   "evaluation": model_version_id is not None,
                                                   "model_status_at_request": model_record.status})
            if not created:
                if (job.job_type != "inference" or job.payload.get("input_sha256") != digest
                        or job.payload.get("modality") != modality):
                    raise MediaRequestConflict("Idempotency-Key is already associated with a different request")
                if job.state != JobState.QUEUED.value:
                    return {**self.result(job.id), "created": False}
                with self.factory() as session:
                    model_record, _, _ = get_approved_model(session, self.settings,
                                                           pinned_model_id=UUID(job.payload["model_version_id"]),
                                                           allow_candidate=bool(job.payload.get("evaluation")))
            input_root = safe_local_root(self.settings.input_cache_dir)
            input_dir = input_root / "media-tests" / str(job.id)
            input_dir.mkdir(parents=True, exist_ok=True)
            source = input_dir / f"input{Path(original_name).suffix.casefold()}"
            if not source.exists():
                try:
                    temp_path.rename(source)
                except FileExistsError:
                    if sha256_file(source) != digest:
                        raise MediaRequestConflict("Concurrent input checksum differs from the queued request")
            elif sha256_file(source) != digest:
                raise MediaRequestConflict("Stored input checksum differs from the queued request")
            self._payload(job.id, local_input=str(source))
        finally:
            temp_path.unlink(missing_ok=True)

        try:
            with self.factory() as session:
                token = claim_execution_slot(session, job.id, self.settings.worker_id)
                self._fences[job.id] = (self.settings.worker_id, token)
        except ExecutionSlotBusy as exc:
            raise MediaSlotBusy(job.id) from exc

        try:
            self._transition(job.id, JobState.DISPATCHING, "loading_pinned_model", "local_test_claimed",
                             {"model_version_id": str(model_record.id), "execution_token": token})
            with self.factory() as session:
                pipeline, loaded_version = load_approved_pipeline(
                    session, self.settings, pinned_model_id=model_record.id,
                    inference_factory=self.inference_factory, allow_candidate=bool(job.payload.get("evaluation")))
            if modality == "video":
                pipeline.preview_callback = lambda jpeg: self._publish_preview(job.id, jpeg)
            self._transition(job.id, JobState.RUNNING, "local_inference", "local_test_started")
            output_root = safe_local_root(self.settings.output_cache_dir)
            output_dir = output_root / "media-tests" / str(job.id) / f"attempt-{token}"
            def report_progress(progress: dict[str, Any]) -> None:
                self._payload(job.id, progress=progress)

            summary = (run_image_test(pipeline, source, output_dir) if modality == "image"
                       else run_video_test(pipeline, source, output_dir, frame_stride=frame_stride,
                                           progress=report_progress,
                                           should_stop=lambda: job.id in self._stop))
            self._clear_preview(job.id)
            summary = {key: value for key, value in summary.items()
                       if key not in {"video_path", "detections_path"}}
            self._payload(job.id, local_result_dir=str(output_dir), result_files=summary["files"],
                          result_summary=summary, result_storage="local", model_version_id=str(loaded_version.id))
            self._transition(job.id, JobState.COLLECTING, "local_results_ready", "local_test_output_written")
            self._transition(job.id, JobState.VALIDATING, "validating_local_results", "local_test_compute_stopped")
            for role in summary["files"]:
                self.result_file(job.id, role)
        except InferenceCancelled:
            try:
                self._transition(job.id, JobState.CANCELLED, "cancelled", "local_test_cancelled")
            finally:
                self._release(job.id)
                self._stop.discard(job.id)
            raise LocalInferenceCancelled(job.id)
        except StaleExecutionOwner:
            self._clear_preview(job.id)
            self._fences.pop(job.id, None)
            self._stop.discard(job.id)
            raise
        except Exception as exc:
            self._transition(job.id, JobState.FAILED, "local_inference_failed", "local_test_failed",
                             {"error_type": type(exc).__name__})
            self._release(job.id)
            raise LocalInferenceFailed(job.id, type(exc).__name__) from exc
        finally:
            self._clear_preview(job.id)

        if self.settings.drive_folder_id:
            self._transition(job.id, JobState.SYNCING, "syncing_results", "local_results_drive_sync_started")
            try:
                drive_ids = self._sync_files(self._job(job.id))
                self._payload(job.id, drive_result_ids=drive_ids, result_storage="local_and_drive")
                self._transition(job.id, JobState.COMPLETED, "complete", "local_results_synced")
            except StaleExecutionOwner:
                self._fences.pop(job.id, None)
                raise
            except Exception as exc:
                self._payload(job.id, result_sync_error=type(exc).__name__)
                self._transition(job.id, JobState.SYNC_PENDING, "local_results_sync_pending",
                                 "local_results_sync_failed", {"error_type": type(exc).__name__})
        else:
            self._payload(job.id, result_sync_error="drive_not_configured")
            self._transition(job.id, JobState.SYNC_PENDING, "local_results_sync_pending",
                             "local_results_drive_not_configured")
        self._release(job.id)
        return {**self.result(job.id), "created": created}

    def result(self, job_id: UUID) -> dict[str, Any]:
        job = self._job(job_id)
        files = job.payload.get("result_files", {})
        return {"job_id": str(job.id), "state": job.state, "phase": job.phase,
                "modality": job.payload.get("modality"),
                "model_version_id": job.payload.get("model_version_id"),
                "result_storage": job.payload.get("result_storage"),
                "result_sync_error": job.payload.get("result_sync_error"),
                "summary": job.payload.get("result_summary"),
                "files": {role: f"/api/v1/jobs/{job.id}/results/{role}" for role in files}}

    def stop(self, job_id: UUID) -> dict[str, Any]:
        job = self._job(job_id)
        if job.job_type != "inference":
            raise MediaRequestConflict("Only local inference jobs can be stopped here")
        if job.state in {JobState.COLLECTING.value, JobState.VALIDATING.value, JobState.SYNCING.value,
                         JobState.COMPLETED.value, JobState.SYNC_PENDING.value}:
            return {"job_id": str(job.id), "state": job.state, "message": "Job đã xong"}
        if job.state == JobState.CANCELLED.value:
            return {"job_id": str(job.id), "state": job.state, "message": "Job đã hủy"}
        if job.state == JobState.RUNNING.value:
            try:
                self._transition(job_id, JobState.CANCELLATION_REQUESTED, "cancellation_requested",
                                 "local_test_stop_requested")
                # Persist cancellation before the inference thread observes the stop flag.
                self._stop.add(job_id)
            except Exception:
                current = self._job(job_id)
                if current.state not in {JobState.COLLECTING.value, JobState.VALIDATING.value, JobState.SYNCING.value,
                                         JobState.COMPLETED.value, JobState.SYNC_PENDING.value, JobState.CANCELLED.value}:
                    raise
                self._stop.discard(job_id)
                return {"job_id": str(job.id), "state": current.state, "message": "Job đã xong"}
            return {"job_id": str(job.id), "state": JobState.CANCELLATION_REQUESTED.value,
                    "message": "Đang dừng job"}
        if job.state == JobState.CANCELLATION_REQUESTED.value:
            return {"job_id": str(job.id), "state": job.state, "message": "Đang dừng job"}
        raise MediaRequestConflict(f"Job hiện không thể dừng (state={job.state})")

    def result_file(self, job_id: UUID, role: str) -> Path:
        job = self._job(job_id)
        files = job.payload.get("result_files", {})
        if role not in files or not job.payload.get("local_result_dir"):
            raise KeyError(role)
        output_root = Path(self.settings.output_cache_dir).expanduser().resolve()
        directory = Path(job.payload["local_result_dir"]).resolve()
        try:
            directory.relative_to(output_root)
        except ValueError as exc:
            raise MediaRequestConflict("Stored result path is outside the configured result cache") from exc
        filename = files[role]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise MediaRequestConflict("Stored result filename is invalid")
        path = (directory / filename).resolve()
        try:
            path.relative_to(directory)
        except ValueError as exc:
            raise MediaRequestConflict("Stored result file escapes its job directory") from exc
        if not path.is_file():
            raise FileNotFoundError(path.name)
        return path

    def retry_sync(self, job_id: UUID) -> dict[str, Any]:
        job = self._job(job_id)
        if job.job_type != "inference" or job.state != JobState.SYNC_PENDING.value:
            raise MediaRequestConflict("Only local inference jobs in sync_pending can retry output synchronization")
        if not self.settings.drive_folder_id:
            raise MediaRequestConflict("Google Drive output storage is not configured")
        self._fences[job_id] = (job.owner_id, job.execution_token)
        self._transition(job_id, JobState.SYNCING, "syncing_results", "local_results_sync_retry")
        try:
            drive_ids = self._sync_files(job)
            self._payload(job_id, drive_result_ids=drive_ids, result_storage="local_and_drive",
                          result_sync_error=None)
            self._transition(job_id, JobState.COMPLETED, "complete", "local_results_synced")
        except StaleExecutionOwner:
            raise
        except Exception as exc:
            self._payload(job_id, result_sync_error=type(exc).__name__)
            self._transition(job_id, JobState.SYNC_PENDING, "local_results_sync_pending",
                             "local_results_sync_failed", {"error_type": type(exc).__name__})
            raise
        finally:
            self._fences.pop(job_id, None)
        return self.result(job_id)
