"""Collect the exact output of a completed prepare_only / remote_smoke Kaggle run."""

from __future__ import annotations

import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from drowning.db.models import Artifact, ExecutionSlot, Job
from drowning.db.repositories import assert_execution_owner, release_execution_slot, transition_job
from drowning.domain.jobs import JobState
from drowning.integrations.drive import sha256_file
from drowning.orchestration.kaggle_run import PACKAGE_NAME, REPORT_NAME

_ALLOWED_SUFFIXES = {".json", ".jpg", ".jpeg", ".png", ".txt"}
_MAX_BYTES, _MAX_FILES = 1024 * 1024 * 1024, 5000


def verify_run_archive(archive_path: Path, extract_to: Path, *, expected_mode: str) -> dict[str, Any]:
    if not zipfile.is_zipfile(archive_path):
        raise ValueError("Kaggle output does not contain a valid run-output ZIP archive")
    if extract_to.exists():
        raise FileExistsError(f"Refusing to overwrite run-report extraction: {extract_to}")
    with zipfile.ZipFile(archive_path) as archive:
        entries, total, seen = [], 0, set()
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError("Run output ZIP contains an unsafe path")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Run output ZIP may not contain symbolic links")
            if relative.suffix.casefold() not in _ALLOWED_SUFFIXES:
                raise ValueError(f"Run output ZIP contains a disallowed file type: {name}")
            if name in seen:
                raise ValueError(f"Run output ZIP contains a duplicate path: {name}")
            seen.add(name)
            total += info.file_size
            if total > _MAX_BYTES or len(seen) > _MAX_FILES:
                raise ValueError("Run output ZIP exceeds the size or file-count limit")
            entries.append(info)
        if REPORT_NAME not in seen:
            raise ValueError(f"Run output ZIP is missing {REPORT_NAME}")
        if archive.testzip() is not None:
            raise ValueError("Run output ZIP failed CRC validation")
        extract_to.mkdir(parents=True)
        try:
            for info in entries:
                destination = extract_to.joinpath(*PurePosixPath(info.filename).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source_stream, destination.open("xb") as target_stream:
                    shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)
        except Exception:
            shutil.rmtree(extract_to, ignore_errors=True)
            raise
    report = json.loads((extract_to / REPORT_NAME).read_text(encoding="utf-8"))
    if (report.get("schema_version") != 1 or report.get("run_mode") != expected_mode
            or report.get("training_started") is not False or report.get("model_candidate") is not False):
        shutil.rmtree(extract_to, ignore_errors=True)
        raise ValueError("Run report does not match the job: wrong mode, or it claims training/a model candidate")
    return {"report": report, "archive_sha256": sha256_file(archive_path), "archive_size": archive_path.stat().st_size}


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"run_mode": report["run_mode"], "finished_at": report.get("finished_at"),
                               "code_fingerprint": report.get("code_fingerprint"),
                               "environment": {k: report.get("environment", {}).get(k)
                                               for k in ("python", "torch", "cuda_available", "gpu")}}
    if report.get("train_blocked"):
        summary.update(train_blocked=True, blockers=report.get("blockers", []),
                       excluded_sources=report.get("input_report", {}).get("excluded_sources", {}),
                       image_crops_by_split=report.get("input_report", {}).get("image_crops_by_split"),
                       temporal_windows_by_split=report.get("input_report", {}).get("temporal_windows_by_split"))
    elif report["run_mode"] == "prepare_only":
        summary.update(blockers=report.get("blockers", []), advisories=report.get("advisories", []),
                       incremental_prepare=report.get("incremental_prepare"),
                       review_mode=report.get("review_mode", "enforced"),
                       temporal_window_count_before_split=report.get("temporal_window_count_before_split", 0),
                       sources=[{k: entry.get(k) for k in ("source_id", "source_version", "status", "mapping_review",
                                                           "grouping_review", "mapping_hash", "canonical_sha256", "modality",
                                                           "prepare_fingerprint", "data_fingerprint", "configuration_fingerprint",
                                                           "processor_fingerprint", "runtime_fingerprint", "cache_status",
                                                           "cache_origin_job_id")}
                                for entry in report.get("sources", [])])
    else:
        summary.update(gpu_confirmed=bool(report.get("gpu_confirmed")), device=report.get("device"),
                       gradients_ok=bool(report.get("all_trainable_parameters_received_gradients")))
    return summary


class RunOutputCollector:
    def __init__(self, session_factory: sessionmaker, kaggle: Any, *, cache_root: Path, worker_id: str,
                 drive: Any | None = None, drive_folder_id: str = ""):
        if not worker_id:
            raise ValueError("worker_id is required")
        self.session_factory, self.kaggle = session_factory, kaggle
        self.cache_root = cache_root.resolve()
        self.worker_id, self.drive, self.drive_folder_id = worker_id, drive, drive_folder_id
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _job(self, job_id: UUID) -> Job:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise KeyError(f"unknown job {job_id}")
            return job

    def _transition(self, job: Job, target: JobState, event: str, phase: str, details: dict[str, Any] | None = None) -> Job:
        with self.session_factory() as session:
            return transition_job(session, job.id, target, event_type=event, phase=phase, details=details,
                                  expected_owner_id=job.owner_id, expected_execution_token=job.execution_token)

    def _payload(self, job: Job, **values: Any) -> None:
        with self.session_factory() as session:
            current = assert_execution_owner(session, job.id, job.owner_id, job.execution_token)
            current.payload = {**(current.payload or {}), **values}
            session.commit()

    def _release(self, job: Job) -> None:
        with self.session_factory() as session:
            release_execution_slot(session, job.id, job.owner_id, job.execution_token, compute_stopped_confirmed=True)

    @staticmethod
    def _mode(job: Job) -> str:
        mode = (job.payload or {}).get("mode")
        if job.job_type not in {"prepare", "train"} or mode not in {"prepare_only", "remote_smoke"}:
            raise ValueError("only prepare_only/remote_smoke jobs produce a run-output archive")
        return mode

    def _upload(self, job: Job, archive: Path, digest: str) -> str:
        receipt = self.drive.upload(archive, folder_id=self.drive_folder_id, job_id=str(job.id),
                                    attempt_id=f"attempt-{job.execution_token}", role="run_output",
                                    artifact_key=f"{job.id}/attempt-{job.execution_token}/run-output")
        if receipt.get("verified") is not True or not receipt.get("file_id"):
            raise ValueError("Drive run-output upload did not return a verified artifact receipt")
        with self.session_factory() as session:
            assert_execution_owner(session, job.id, job.owner_id, job.execution_token)
            if session.scalar(select(Artifact).where(Artifact.job_id == job.id, Artifact.role == "run_output",
                                                     Artifact.sha256 == digest)) is None:
                session.add(Artifact(job_id=job.id, role="run_output", storage_provider="google_drive",
                                     remote_file_id=receipt["file_id"], sha256=digest,
                                     size_bytes=archive.stat().st_size, status="ready"))
                session.commit()
        return receipt["file_id"]

    def _download(self, job: Job, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=True)
        archive = destination / PACKAGE_NAME
        if not archive.is_file():
            self.kaggle.collect_exact_output(f"{job.remote_ref}/{job.remote_version}", destination)
        if not archive.is_file():
            raise FileNotFoundError(PACKAGE_NAME)
        return archive

    def _verify_into_cache(self, job: Job, archive: Path, mode: str) -> dict[str, Any]:
        report_dir = self.cache_root / "run-reports" / str(job.id)
        shutil.rmtree(report_dir, ignore_errors=True)
        report_dir.parent.mkdir(parents=True, exist_ok=True)
        verified = verify_run_archive(archive, report_dir, expected_mode=mode)
        summary = summarize_report(verified["report"])
        self._payload(job, run_summary=summary, run_archive=str(archive), run_report_dir=str(report_dir),
                      run_archive_sha256=verified["archive_sha256"], result_storage="local")
        return {"summary": summary, "digest": verified["archive_sha256"]}

    def _fail(self, job_id: UUID, event: str, exc: BaseException) -> None:
        current = self._job(job_id)
        if current.state in {JobState.VALIDATING.value, JobState.SYNCING.value}:
            failed = self._transition(current, JobState.FAILED, event, "validation_failed", {"error_type": type(exc).__name__})
            self._release(failed)

    def collect(self, job_id: UUID) -> dict[str, Any]:
        job = self._job(job_id)
        mode = self._mode(job)
        if job.state != JobState.COLLECTING.value or not job.remote_ref or not job.remote_version:
            raise ValueError("run collection requires a completed exact-version Kaggle job")
        with self.session_factory() as session:
            assert_execution_owner(session, job.id, job.owner_id, job.execution_token, require_slot=True)
        try:
            archive = self._download(job, self.cache_root / "kaggle-output" / str(job.id) / f"version-{job.remote_version}")
        except Exception as exc:
            # The remote already finished, so compute is confirmed stopped: free the slot and retry the download later.
            pending = self._transition(job, JobState.SYNC_PENDING, "run_output_collect_failed", "sync_pending",
                                       {"error_type": type(exc).__name__})
            self._release(pending)
            raise
        job = self._transition(job, JobState.VALIDATING, "run_output_collected", "validating_run_output",
                               {"kernel_version": job.remote_version})
        try:
            checked = self._verify_into_cache(job, archive, mode)
        except Exception as exc:
            self._fail(job_id, "run_output_invalid", exc)
            raise
        return self._finish(job, archive, checked["digest"], checked["summary"])

    def _finish(self, job: Job, archive: Path, digest: str, summary: dict[str, Any]) -> dict[str, Any]:
        job = self._job(job.id)
        if self.drive is not None and self.drive_folder_id:
            if job.state == JobState.VALIDATING.value:
                job = self._transition(job, JobState.SYNCING, "run_output_drive_sync_started", "artifact_sync")
            try:
                file_id = self._upload(job, archive, digest)
                self._payload(job, result_storage="local_and_drive", drive_file_id=file_id)
            except Exception as exc:
                pending = self._transition(job, JobState.SYNC_PENDING, "run_output_sync_pending", "sync_pending",
                                           {"error_type": type(exc).__name__})
                if self._slot_held_by(job):
                    self._release(pending)
                return {"job_id": str(job.id), "state": "sync_pending", "summary": summary}
            job = self._job(job.id)
        done = self._transition(job, JobState.COMPLETED, "run_output_collected", "complete",
                                {"mode": summary["run_mode"], "storage": "drive" if self.drive else "local"})
        if self._slot_held_by(done):
            self._release(done)
        return {"job_id": str(job.id), "state": "completed", "summary": summary}

    def _slot_held_by(self, job: Job) -> bool:
        with self.session_factory() as session:
            slot = session.get(ExecutionSlot, 1)
            return slot is not None and slot.job_id == job.id

    def retry_sync(self, job_id: UUID) -> dict[str, Any]:
        """Finish a sync_pending run: re-download the exact output if needed, verify, upload, complete."""
        job = self._job(job_id)
        mode = self._mode(job)
        if job.state != JobState.SYNC_PENDING.value:
            raise ValueError("only sync_pending run jobs can retry")
        archive_value = (job.payload or {}).get("run_archive")
        have_archive = bool(archive_value) and Path(archive_value).is_file()
        if have_archive and sha256_file(Path(archive_value)) != (job.payload or {}).get("run_archive_sha256"):
            raise ValueError("local run archive checksum changed; refusing to upload")
        if not have_archive and (not job.remote_ref or not job.remote_version):
            raise ValueError("no local archive and no exact Kaggle version to collect again")
        job = self._transition(job, JobState.SYNCING, "run_output_sync_retry", "artifact_sync")
        try:
            if have_archive:
                archive = Path(archive_value)
                checked = {"digest": (job.payload or {}).get("run_archive_sha256"),
                           "summary": (job.payload or {}).get("run_summary")}
            else:
                archive = self._download(job, self.cache_root / "kaggle-output" / str(job.id) / f"retry-{job.execution_token}-{job.remote_version}")
                checked = self._verify_into_cache(job, archive, mode)
        except ValueError as exc:
            self._fail(job_id, "run_output_invalid", exc)
            raise
        except Exception as exc:
            self._transition(self._job(job_id), JobState.SYNC_PENDING, "run_output_sync_retry_failed", "sync_pending",
                             {"error_type": type(exc).__name__})
            raise
        if self.drive is None or not self.drive_folder_id:
            done = self._transition(self._job(job_id), JobState.COMPLETED, "run_output_collected", "complete", {"storage": "local"})
            return {"job_id": str(done.id), "state": "completed", "summary": checked["summary"]}
        try:
            file_id = self._upload(self._job(job_id), archive, checked["digest"])
        except Exception as exc:
            self._transition(self._job(job_id), JobState.SYNC_PENDING, "run_output_sync_retry_failed", "sync_pending",
                             {"error_type": type(exc).__name__})
            raise
        self._payload(self._job(job_id), result_storage="local_and_drive", drive_file_id=file_id)
        done = self._transition(self._job(job_id), JobState.COMPLETED, "run_output_sync_completed", "complete")
        return {"job_id": str(done.id), "state": "completed", "summary": checked["summary"]}
