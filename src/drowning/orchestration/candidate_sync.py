"""Collect one exact Kaggle candidate archive, verify it, and sync via Drive."""

from __future__ import annotations

import json
import hashlib
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from drowning.db.models import Artifact, Job, ModelVersion, now_utc
from drowning.db.repositories import assert_execution_owner, release_execution_slot, transition_job
from drowning.domain.jobs import JobState
from drowning.integrations.drive import GoogleDriveArtifactAdapter
from drowning.integrations.kaggle import KaggleCLIClient
from drowning.ml.bundle import load_bundle, sha256_file

_EXPECTED_FILES = {"model-bundle/manifest.json", "model-bundle/weights.pt", "metrics.json"}
_MAX_ARCHIVE_BYTES = 8 * 1024 * 1024 * 1024


def verify_candidate_archive(archive_path: Path, extract_to: Path) -> dict[str, Any]:
    if not zipfile.is_zipfile(archive_path):
        raise ValueError("Kaggle output does not contain a valid candidate ZIP archive")
    if extract_to.exists():
        raise FileExistsError(f"Refusing to overwrite candidate extraction: {extract_to}")
    total_size = 0
    seen = set()
    with zipfile.ZipFile(archive_path) as archive:
        entries = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError("Candidate ZIP contains an unsafe path")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("Candidate ZIP may not contain symbolic links")
            if name not in _EXPECTED_FILES:
                raise ValueError(f"Candidate ZIP contains an unexpected file: {name}")
            if name in seen:
                raise ValueError(f"Candidate ZIP contains a duplicate path: {name}")
            seen.add(name)
            total_size += info.file_size
            if total_size > _MAX_ARCHIVE_BYTES:
                raise ValueError("Candidate ZIP exceeds the maximum uncompressed size")
            entries.append(info)
        if seen != _EXPECTED_FILES:
            raise ValueError("Candidate ZIP is missing manifest, weights, or metrics")
        if archive.testzip() is not None:
            raise ValueError("Candidate ZIP failed CRC validation")
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
    model_dir = extract_to / "model-bundle"
    model, manifest = load_bundle(model_dir, require_temporal=True, map_location="cpu")
    del model
    metrics = json.loads((extract_to / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("completion_kind") != "candidate" or metrics.get("approved") is not False:
        raise ValueError("Candidate metrics must be explicitly unapproved candidate metrics")
    if metrics.get("dataset_fingerprint") != manifest.get("dataset_fingerprint"):
        raise ValueError("Candidate metrics and bundle dataset fingerprints do not match")
    if manifest.get("model_version_id") and metrics.get("candidate_id") != manifest["model_version_id"]:
        raise ValueError("Candidate metrics and bundle model-version IDs do not match")
    if not manifest.get("code_fingerprint") or not manifest.get("recipe_fingerprint"):
        raise ValueError("Candidate bundle is missing exact code/recipe fingerprints")
    if metrics.get("model_kind", "temporal") != manifest.get("model_kind", "temporal"):
        raise ValueError("Candidate metrics and bundle model kinds do not match")
    if metrics.get("test_split_used") is not False:
        raise ValueError("Candidate generation must not use the holdout test split")
    return {"manifest": manifest, "metrics": metrics, "archive_sha256": sha256_file(archive_path),
            "archive_size": archive_path.stat().st_size}


class CandidateArtifactSync:
    def __init__(self, session_factory: sessionmaker, kaggle: KaggleCLIClient,
                 drive: GoogleDriveArtifactAdapter, *, drive_folder_id: str,
                 cache_root: Path, worker_id: str):
        if not drive_folder_id or not worker_id:
            raise ValueError("Drive folder ID and worker_id are required")
        self.session_factory = session_factory
        self.kaggle = kaggle
        self.drive = drive
        self.drive_folder_id = drive_folder_id
        self.cache_root = cache_root.resolve()
        self.worker_id = worker_id
        self._fences: dict[UUID, tuple[str, int]] = {}
        try:
            self.cache_root.relative_to(Path("D:/DrowningDemo").resolve())
            protected_demo = True
        except ValueError:
            protected_demo = False
        if protected_demo or any(part.casefold() == "sources" for part in self.cache_root.parts):
            raise ValueError("candidate cache must not be under a sources/ directory")
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _job(self, job_id: UUID) -> Job:
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise KeyError(f"unknown job {job_id}")
            return job

    def _transition(self, job_id: UUID, target: JobState, *, event_type: str, phase: str,
                    details: dict[str, Any] | None = None) -> Job:
        with self.session_factory() as session:
            owner, token = self._fences[job_id]
            return transition_job(session, job_id, target, event_type=event_type, phase=phase, details=details,
                                  expected_owner_id=owner, expected_execution_token=token)

    def _owned(self, session, job_id):
        owner, token = self._fences[job_id]
        return assert_execution_owner(session, job_id, owner, token)

    def _capture(self, job, *, require_slot=False):
        self._fences[job.id] = (self.worker_id, job.execution_token)
        with self.session_factory() as session:
            assert_execution_owner(session, job.id, self.worker_id, job.execution_token, require_slot=require_slot)

    def _release_slot(self, job: Job) -> None:
        with self.session_factory() as session:
            release_execution_slot(session, job.id, job.owner_id or self.worker_id,
                                   job.execution_token, compute_stopped_confirmed=True)

    def _register_artifact_and_candidate(self, job: Job, artifact_info: dict[str, Any],
                                         verified: dict[str, Any]) -> ModelVersion:
        manifest, metrics = verified["manifest"], verified["metrics"]
        with self.session_factory() as session:
            self._owned(session, job.id)
            current = session.get(Job, job.id)
            payload = dict(current.payload or {})
            if payload.get("dataset_identity_pending") is True:
                actual = manifest.get("dataset_fingerprint") or ""
                request = payload.get("training_request_identity") or ""
                if not re.fullmatch(r"[a-f0-9]{64}", actual) or not re.fullmatch(r"[a-f0-9]{64}", request):
                    raise ValueError("Direct training output is missing its verified dataset or request identity")
                identity = hashlib.sha256(json.dumps({"training_request_identity": request,
                                                      "dataset_fingerprint": actual}, sort_keys=True).encode()).hexdigest()
                payload.update(dataset_fingerprint=actual, dataset_identity_pending=False,
                               dataset_identity_source="training_notebook", training_identity=identity)
                current.payload = payload
            job.payload = payload
            artifact = session.scalar(select(Artifact).where(
                Artifact.job_id == job.id, Artifact.role == "candidate_bundle",
                Artifact.sha256 == verified["archive_sha256"]))
            if artifact is None:
                artifact = Artifact(job_id=job.id, role="candidate_bundle", storage_provider="google_drive",
                                    remote_file_id=artifact_info["file_id"], sha256=verified["archive_sha256"],
                                    size_bytes=verified["archive_size"], status="ready")
                session.add(artifact)
                session.flush()
            model = session.scalar(select(ModelVersion).where(ModelVersion.training_job_id == job.id))
            if model is None:
                display_version = f"kaggle-{job.remote_version}"
                version_id = UUID(str(manifest["model_version_id"])) if manifest.get("model_version_id") else uuid4()
                model = ModelVersion(
                    id=version_id,
                    model_family=manifest["model_family"], display_version=display_version,
                    schema_version=manifest["model_schema_version"], class_keys=manifest["class_keys"],
                    temporal_trained=bool(manifest.get("temporal_trained")), completion_kind="candidate", status="candidate",
                    training_job_id=job.id, dataset_fingerprint=manifest["dataset_fingerprint"],
                    artifact_id=artifact.id, metrics={**{k: v for k, v in metrics.items() if k != "validation_samples"}, "provenance": {
                        "training_job_id": str(job.id), "training_identity": (job.payload or {}).get("training_identity"),
                        "training_request_identity": (job.payload or {}).get("training_request_identity"),
                        "dataset_identity_source": (job.payload or {}).get("dataset_identity_source"),
                        "options": (job.payload or {}).get("options"), "seed": metrics.get("seed"),
                        "recipe_fingerprint": manifest.get("recipe_fingerprint"), "code_fingerprint": manifest.get("code_fingerprint"),
                        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
                        "weights_sha256": manifest.get("files", {}).get("weights.pt", {}).get("sha256"),
                        "archive_sha256": verified["archive_sha256"], "initialization": manifest.get("initialization"),
                        "runtime_versions": manifest.get("runtime_versions"), "determinism": metrics.get("determinism"),
                        "kernel_ref": job.remote_ref, "kernel_version": job.remote_version,
                        "drive_file_id": artifact.remote_file_id}})
                session.add(model)
            session.commit()
            session.refresh(model)
            return model

    def _upload_verified_archive(self, job: Job, archive_path: Path, verified: dict[str, Any]) -> ModelVersion:
        artifact_key = f"{job.id}/attempt-{job.execution_token}/candidate-bundle"
        drive_receipt = self.drive.upload(
            archive_path, folder_id=self.drive_folder_id, job_id=str(job.id),
            attempt_id=f"attempt-{job.execution_token}", role="candidate_bundle", artifact_key=artifact_key)
        if drive_receipt.get("verified") is not True or not drive_receipt.get("file_id"):
            raise ValueError("Drive candidate upload did not return a verified artifact receipt")
        return self._register_artifact_and_candidate(job, drive_receipt, verified)

    def collect(self, job_id: UUID) -> dict[str, Any]:
        job = self._job(job_id)
        if job.state != JobState.COLLECTING.value or not job.remote_ref or not job.remote_version:
            raise ValueError("candidate collection requires a completed exact-version Kaggle job")
        if job.job_type != "train" or (job.payload or {}).get("mode") != "train_candidate":
            raise ValueError("only train_candidate jobs produce a model bundle")
        self._capture(job, require_slot=True)
        destination = self.cache_root / "kaggle-output" / str(job.id) / f"version-{job.remote_version}"
        destination.mkdir(parents=True, exist_ok=True)
        archive_path = destination / "candidate-output.zip"
        try:
            with self.session_factory() as session:
                current = self._owned(session, job_id)
                if current is None or current.state != JobState.COLLECTING.value:
                    raise RuntimeError("job changed while output was collected")
                current.payload = {**(current.payload or {}), "local_candidate_archive": str(archive_path)}
                session.commit()
            if not archive_path.is_file():
                self.kaggle.collect_exact_output(f"{job.remote_ref}/{job.remote_version}", destination)
            blocked_archive = destination / "run-output.zip"
            if not archive_path.is_file() and blocked_archive.is_file():
                return self._finish_blocked(job, blocked_archive)
            self._transition(job_id, JobState.VALIDATING, event_type="candidate_output_collected",
                             phase="validating_bundle", details={"kernel_version": job.remote_version})
            with tempfile.TemporaryDirectory(dir=self.cache_root) as temp:
                verified = verify_candidate_archive(archive_path, Path(temp) / "unpacked")
            self._transition(job_id, JobState.SYNCING, event_type="candidate_drive_sync_started",
                             phase="bundle_validated", details={"archive_sha256": verified["archive_sha256"]})
            with self.session_factory() as session:
                current = self._owned(session, job_id)
                current.phase = "artifact_sync"
                current.updated_at = now_utc()
                session.commit()
            model = self._upload_verified_archive(job, archive_path, verified)
            self._transition(job_id, JobState.COMPLETED, event_type="candidate_collected",
                             phase="complete", details={"model_version_id": str(model.id)})
            self._release_slot(job)
            return {"job_id": str(job_id), "state": "completed", "model_version_id": str(model.id),
                    "model_status": "candidate", "approved": False}
        except Exception as exc:
            current = self._job(job_id)
            if current.state == JobState.VALIDATING.value:
                try:
                    failed = self._transition(job_id, JobState.FAILED, event_type="candidate_validation_failed",
                                              phase="validation_failed", details={"error_type": type(exc).__name__})
                    self._release_slot(failed)
                except Exception:
                    pass
            elif current.state in {JobState.COLLECTING.value, JobState.SYNCING.value}:
                try:
                    pending = self._transition(job_id, JobState.SYNC_PENDING,
                                               event_type="candidate_sync_pending", phase="sync_pending",
                                               details={"error_type": type(exc).__name__})
                    with self.session_factory() as session:
                        refreshed = self._owned(session, job_id)
                        refreshed.error_code = "candidate_sync_pending"
                        refreshed.error_summary = type(exc).__name__
                        session.commit()
                    self._release_slot(pending)
                except Exception:
                    pass
            raise

    def _finish_blocked(self, job: Job, archive: Path) -> dict[str, Any]:
        """The notebook declined to train and said why; record that as a readable failure."""
        from drowning.orchestration.run_collector import summarize_report, verify_run_archive

        job_id = job.id
        self._transition(job_id, JobState.VALIDATING, event_type="train_blocked_report_collected",
                         phase="validating_blocked_report", details={"kernel_version": job.remote_version})
        report_dir = self.cache_root / "run-reports" / str(job_id)
        shutil.rmtree(report_dir, ignore_errors=True)
        report_dir.parent.mkdir(parents=True, exist_ok=True)
        verified = verify_run_archive(archive, report_dir, expected_mode="train_candidate")
        summary = summarize_report(verified["report"])
        with self.session_factory() as session:
            current = self._owned(session, job_id)
            current.payload = {**(current.payload or {}), "run_summary": summary, "run_report_dir": str(report_dir),
                               "run_archive": str(archive)}
            current.error_code = "train_inputs_blocked"
            current.error_summary = "Training did not start: " + ", ".join(summary.get("blockers", []))[:900]
            session.commit()
        failed = self._transition(job_id, JobState.FAILED, event_type="train_inputs_blocked", phase="train_blocked",
                                  details={"blockers": summary.get("blockers", [])})
        self._release_slot(failed)
        return {"job_id": str(job_id), "state": "failed", "train_blocked": True, "blockers": summary.get("blockers", [])}

    def retry_sync(self, job_id: UUID) -> dict[str, Any]:
        job = self._job(job_id)
        if job.state != JobState.SYNC_PENDING.value:
            raise ValueError("only sync_pending candidate jobs can retry Drive sync")
        self._capture(job)
        archive_value = (job.payload or {}).get("local_candidate_archive")
        if not archive_value:
            raise ValueError("sync_pending job has no local candidate archive; collect exact version first")
        archive_path = Path(archive_value)
        if not archive_path.is_file():
            destination = archive_path.parent / f"retry-{uuid4().hex}"
            destination.mkdir(parents=True, exist_ok=False)
            archive_path = destination / "candidate-output.zip"
            self.kaggle.collect_exact_output(f"{job.remote_ref}/{job.remote_version}", destination)
            with self.session_factory() as session:
                current = self._owned(session, job_id)
                current.payload = {**(current.payload or {}), "local_candidate_archive": str(archive_path)}
                session.commit()
        with tempfile.TemporaryDirectory(dir=self.cache_root) as temp:
            verified = verify_candidate_archive(archive_path, Path(temp) / "unpacked")
        self._transition(job_id, JobState.SYNCING, event_type="candidate_sync_retry",
                         phase="artifact_sync", details={})
        with tempfile.TemporaryDirectory(dir=self.cache_root) as temp:
            try:
                model = self._upload_verified_archive(job, archive_path, verified)
            except Exception as exc:
                self._transition(job_id, JobState.SYNC_PENDING,
                                 event_type="candidate_sync_retry_failed", phase="sync_pending",
                                 details={"error_type": type(exc).__name__})
                raise
        self._transition(job_id, JobState.COMPLETED, event_type="candidate_sync_completed",
                         phase="complete", details={"model_version_id": str(model.id)})
        return {"job_id": str(job_id), "state": "completed", "model_version_id": str(model.id),
                "model_status": "candidate", "approved": False}
