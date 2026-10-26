"""Train page backend: preflight gates, Kaggle job submission, and honest cancellation."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from drowning.api.review import SourceReviewService
from drowning.config import Settings
from drowning.data.registry import load_source
from drowning.data.review import is_mapping_approved, load_grouping_review, mapping_fingerprint
from drowning.db.models import ExecutionSlot, Job, ModelRegistry, ModelVersion
from drowning.db.repositories import create_job, transition_job
from drowning.domain.jobs import JobState
from drowning.integrations.kaggle import find_kaggle

MODES = ("prepare_only", "remote_smoke", "train_candidate")
_KERNEL_REF = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
_TERMINAL = {"completed", "failed", "cancelled"}


class TrainBlocked(RuntimeError):
    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


class TrainService:
    def __init__(self, session_factory: sessionmaker, settings: Settings, *, project_root: Path,
                 reviews: SourceReviewService, which: Callable[[str], str | None] | None = None,
                 home: Path | None = None, environ: dict[str, str] | None = None):
        self.factory, self.settings, self.project_root, self.reviews = session_factory, settings, project_root, reviews
        self.which, self.home = which or (lambda _name: find_kaggle()), home or Path.home()
        self.environ = os.environ if environ is None else environ

    # -- environment -----------------------------------------------------------------------
    def _environment(self) -> dict[str, Any]:
        kaggle_files = [self.home / ".kaggle" / "kaggle.json", self.home / ".config" / "kaggle" / "kaggle.json",
                        self.home / ".kaggle" / "access_token"]
        credentials = ((bool(self.environ.get("KAGGLE_USERNAME")) and bool(self.environ.get("KAGGLE_KEY")))
                       or bool(self.environ.get("KAGGLE_API_TOKEN")) or any(path.is_file() for path in kaggle_files))
        token_file = Path(self.settings.google_oauth_token_file).expanduser()
        return {
            "kaggle_cli_installed": bool(self.which("kaggle")),
            "kaggle_credentials_present": credentials,
            "kaggle_kernel_ref_valid": bool(_KERNEL_REF.fullmatch(self.settings.kaggle_kernel_ref or "")),
            "kaggle_dataset_sources": len(self.settings.kaggle_dataset_sources),
            "kaggle_notebook_private": self.settings.kaggle_private,
            "drive_folder_configured": bool(self.settings.drive_folder_id),
            "drive_token_file_present": token_file.is_file(),
            "worker_id": self.settings.worker_id,
        }

    def _sources(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.reviews.sources_dir.glob("*.yaml")):
            source = load_source(path, self.reviews.reviews_dir)
            media_ok = load_grouping_review(source, self.reviews.reviews_dir) is not None or (
                source.get("grouping", {}).get("status") == "verified")
            rows.append({"source_id": source.get("source_id"), "source_version": str(source.get("source_version")),
                         "modality": source.get("modality"), "format": source.get("format"),
                         "mapping_hash": mapping_fingerprint(source), "mapping_approved": is_mapping_approved(source),
                         "groups_verified": media_ok})
        return rows

    def _latest(self, session, mode: str, *, completed_only: bool = True) -> Job | None:
        jobs = session.scalars(select(Job).where(Job.job_type.in_(["prepare", "train"])).order_by(Job.created_at.desc())).all()
        for job in jobs:
            if (job.payload or {}).get("mode") != mode:
                continue
            if completed_only and not (job.state == "completed" and (job.payload or {}).get("run_summary")):
                continue
            return job
        return None

    def _run_view(self, job: Job | None) -> dict[str, Any] | None:
        if job is None:
            return None
        return {"job_id": str(job.id), "state": job.state, "created_at": job.created_at.isoformat(),
                "summary": (job.payload or {}).get("run_summary")}

    # -- preflight -------------------------------------------------------------------------
    def preflight(self) -> dict[str, Any]:
        environment, sources = self._environment(), self._sources()
        with self.factory() as session:
            prepare = self._latest(session, "prepare_only")
            smoke = self._latest(session, "remote_smoke")
            registry = session.get(ModelRegistry, self.settings.model_family)
            approved = session.get(ModelVersion, registry.approved_model_id) if registry and registry.approved_model_id else None
            slot = session.get(ExecutionSlot, 1)
            active = session.get(Job, slot.job_id) if slot is not None and slot.job_id else None
            recent = session.scalars(select(Job).where(Job.job_type.in_(["prepare", "train"]))
                                     .order_by(Job.created_at.desc()).limit(1)).first()
            queued = [j for j in session.scalars(select(Job).where(Job.state == "queued")) if (j.payload or {}).get("executor") == "kaggle"]
        prepare_summary = (prepare.payload or {}).get("run_summary") if prepare else None
        stale, prepared_sources = [], {}
        if prepare_summary:
            prepared_sources = {s["source_id"]: s for s in prepare_summary.get("sources", [])}
            for row in sources:
                seen = prepared_sources.get(row["source_id"])
                if seen is None or seen.get("status") != "prepared" or seen.get("mapping_hash") != row["mapping_hash"]:
                    stale.append(row["source_id"])
        approvals_missing = [f"{r['source_id']}: " + ("mapping not approved" if not r["mapping_approved"] else "source groups not verified")
                             for r in sources if not (r["mapping_approved"] and r["groups_verified"])]
        base = []
        if not environment["kaggle_cli_installed"]:
            base.append("Kaggle CLI is not installed (pip install kaggle)")
        if not environment["kaggle_credentials_present"]:
            base.append("Kaggle credentials are missing (kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)")
        if not environment["kaggle_kernel_ref_valid"]:
            base.append("KAGGLE_KERNEL_REF must be owner/slug")
        if not environment["kaggle_dataset_sources"]:
            base.append("KAGGLE_DATASET_SOURCES lists no private Kaggle datasets to attach")
        if not environment["kaggle_notebook_private"]:
            base.append("KAGGLE_NOTEBOOK_PRIVATE must stay true")
        modes = {"prepare_only": {"blocked_by": list(base)}}
        smoke_block = list(base)
        if prepare is None:
            smoke_block.append("run prepare_only first and review its report")
        elif stale:
            smoke_block.append("the mapping changed after the last prepare run: " + ", ".join(stale))
        smoke_block += approvals_missing
        modes["remote_smoke"] = {"blocked_by": smoke_block}
        train_block = list(smoke_block)
        if smoke is None:
            train_block.append("run remote_smoke first")
        elif not (smoke.payload["run_summary"].get("gpu_confirmed") and smoke.payload["run_summary"].get("gradients_ok")):
            train_block.append("the last remote_smoke did not confirm a GPU with gradients")
        windows = (prepare_summary or {}).get("temporal_window_count_before_split", 0)
        if prepare_summary and windows == 0:
            train_block.append("no temporal windows: a labeled video source with person/time annotations is required")
        if not environment["drive_folder_configured"] or not environment["drive_token_file_present"]:
            train_block.append("Google Drive must be configured to publish a candidate")
        if approved is not None:
            train_block.append("an approved model already exists; training a new one needs an explicit new version")
        modes["train_candidate"] = {"blocked_by": train_block, "requires_new_version_flag": True}
        for entry in modes.values():
            effective = [b for b in entry["blocked_by"] if not b.startswith("an approved model already exists")]
            entry["allowed"] = not effective
        return {
            "environment": environment, "sources": sources, "stale_sources": stale,
            "latest_prepare": self._run_view(prepare), "latest_smoke": self._run_view(smoke),
            "latest_job": ({"job_id": str(recent.id), "mode": (recent.payload or {}).get("mode"), "state": recent.state}
                           if recent else None),
            "approved_model": ({"id": str(approved.id), "display_version": approved.display_version} if approved else None),
            "compute_slot_held_by": ({"job_id": str(active.id), "job_type": active.job_type,
                                      "mode": (active.payload or {}).get("mode") or (active.payload or {}).get("modality"),
                                      "state": active.state} if active else None),
            "queued_kaggle_jobs": len(queued), "modes": modes,
            "temporal_window_count_before_split": windows}

    # -- submission ------------------------------------------------------------------------
    def _materializer(self):
        root = str(self.project_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        return importlib.import_module("notebooks.materialize_kernel")

    def _notebook_builder(self):
        self._materializer()  # puts the project root on sys.path
        return importlib.import_module("notebooks.build_notebook")

    def _training_identity(self, payload: dict[str, Any], view: dict[str, Any]) -> str:
        """Hash of everything that decides the trained weights: data, mappings, recipes, code, options, seed."""
        recipes = self.project_root / "configs" / "recipes"
        document = {"dataset": payload["dataset_fingerprint"],
                    "mappings": sorted((s["source_id"], s["mapping_hash"]) for s in view["sources"]),
                    "recipes": [hashlib.sha256((recipes / name).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()
                                for name in ("image.yaml", "temporal.yaml")],
                    "code": self._notebook_builder().source_bundle_sha256(), "options": payload.get("options"),
                    "seed": 20261026}
        return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()

    def _active_run(self, mode: str, idempotency_key: str) -> dict[str, Any] | None:
        """A check that is already queued or running is returned instead of stacking another one behind it."""
        with self.factory() as session:
            for job in session.scalars(select(Job).where(Job.job_type.in_(["prepare", "train"])).order_by(Job.created_at)):
                if (job.payload or {}).get("mode") == mode and (job.payload or {}).get("executor") == "kaggle"                         and job.state not in _TERMINAL and job.idempotency_key != idempotency_key:
                    return {"status": "already_active", "created": False,
                            "job": {"id": str(job.id), "state": job.state, "mode": mode},
                            "message": "Bước này đã được xếp hàng hoặc đang chạy trên Kaggle; không tạo thêm job. "
                                       "Theo dõi ở danh sách Job bên dưới."}
        return None

    def _same_training(self, identity: str) -> dict[str, Any] | None:
        """An identical request is answered with the existing run/model instead of training again."""
        with self.factory() as session:
            for job in session.scalars(select(Job).where(Job.job_type == "train").order_by(Job.created_at.desc())):
                if (job.payload or {}).get("training_identity") != identity or job.state in {"failed", "cancelled"}:
                    continue
                model = session.scalar(select(ModelVersion).where(ModelVersion.training_job_id == job.id))
                return {"status": "same_training_exists", "created": False,
                        "job": {"id": str(job.id), "state": job.state, "mode": "train_candidate"},
                        "model_version_id": str(model.id) if model else None,
                        "message": "The same data, mappings, recipes, code, options and seed were already submitted. "
                                   "Training again is not needed: use that model so everyone gets identical weights."}
        return None

    def _dataset_fingerprint(self, summary: dict[str, Any]) -> str:
        digest = hashlib.sha256()
        for entry in sorted(summary.get("sources", []), key=lambda e: e["source_id"]):
            digest.update(f"{entry['source_id']}|{entry.get('canonical_sha256')}|{entry.get('mapping_hash')}\n".encode())
        return digest.hexdigest()

    def submit(self, mode: str, *, new_version: bool, reason: str, idempotency_key: str,
               options: dict[str, Any] | None = None) -> dict[str, Any]:
        if mode not in MODES:
            raise TrainBlocked([f"mode must be one of {', '.join(MODES)}"])
        if len(idempotency_key.strip()) < 8:
            raise TrainBlocked(["idempotency_key must contain at least eight characters"])
        view = self.preflight()
        if mode == "train_candidate" and view["approved_model"] and not new_version:
            return {"status": "model_available", "job": None, "approved_model": view["approved_model"],
                    "message": "An approved model already exists, so no training job was created. Use Test, or request a new version explicitly."}
        blocked = [reason_ for reason_ in view["modes"][mode]["blocked_by"]
                   if not (mode == "train_candidate" and new_version and reason_.startswith("an approved model already exists"))]
        if mode == "train_candidate":
            if not new_version:
                blocked.append("train_candidate must be requested as an explicit new version")
            if len(reason.strip()) < 10:
                blocked.append("a reason (10+ characters) is required for a new model version")
        if blocked:
            raise TrainBlocked(blocked)
        # Kaggle's push timeout is the kernel's maximum run time, so training needs far longer than a check.
        if mode in {"prepare_only", "remote_smoke"}:
            active = self._active_run(mode, idempotency_key)
            if active is not None:
                return active
        payload: dict[str, Any] = {"mode": mode, "executor": "kaggle", "new_version": bool(new_version),
                                   "reason": reason.strip(), "requested_via": "web",
                                   "push_timeout_seconds": {"prepare_only": 3600, "remote_smoke": 1800, "train_candidate": 28800}[mode]}
        if mode == "train_candidate":
            try:
                payload["options"] = self._notebook_builder().normalize_options(options)
            except (ValueError, KeyError) as exc:
                raise TrainBlocked([f"invalid training options: {exc}"]) from exc
        summary = (view["latest_prepare"] or {}).get("summary") or {}
        if mode == "train_candidate":
            payload.update(dataset_fingerprint=self._dataset_fingerprint(summary), preflight={
                "ready": True, "mapping_review_approved": True,
                "source_video_disjointness_verified": all(s["groups_verified"] for s in view["sources"]),
                "temporal_window_count": view["temporal_window_count_before_split"]})
            payload["training_identity"] = self._training_identity(payload, view)
            existing = self._same_training(payload["training_identity"])
            if existing is not None:
                return existing
        with self.factory() as session:
            job, created = create_job(session, job_type="prepare" if mode == "prepare_only" else "train",
                                      idempotency_key=idempotency_key, payload=payload)
        if created:
            kernel_root = Path(self.settings.kaggle_kernel_dir).expanduser()
            target = kernel_root / "jobs" / f"{str(job.id)[:8]}-{mode}"
            try:
                path = self._materializer().materialize_kernel(self.settings.kaggle_kernel_ref, target, mode=mode,
                                                               dataset_sources=list(self.settings.kaggle_dataset_sources),
                                                               options=payload.get("options"))
            except Exception as exc:
                with self.factory() as session:
                    transition_job(session, job.id, JobState.FAILED, event_type="kernel_materialization_failed",
                                   phase="not_dispatched", details={"error_type": type(exc).__name__})
                raise TrainBlocked([f"could not prepare the Kaggle kernel files ({type(exc).__name__}: {exc})"]) from exc
            with self.factory() as session:
                stored = session.get(Job, job.id)
                stored.payload = {**stored.payload, "kernel_dir": str(path.resolve())}
                session.commit()
                session.refresh(stored)
                job = stored
        return {"status": "queued" if created else "duplicate_request", "created": created,
                "job": {"id": str(job.id), "state": job.state, "mode": mode},
                "dispatch": "the background supervisor pushes queued jobs when the compute slot is free"}

    # -- cancellation ----------------------------------------------------------------------
    def cancel(self, job_id: UUID) -> dict[str, Any]:
        with self.factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                raise KeyError(str(job_id))
            if job.state == JobState.QUEUED.value:
                job = transition_job(session, job_id, JobState.CANCELLED, event_type="cancelled_before_dispatch",
                                     phase="cancelled", details={"remote_started": False})
                return {"job_id": str(job.id), "state": job.state, "remote_cancelled": False,
                        "message": "Cancelled before anything was pushed to Kaggle."}
            if job.state in _TERMINAL:
                return {"job_id": str(job.id), "state": job.state, "message": "The job is already finished."}
        raise TrainBlocked(["A job that already reached Kaggle cannot be cancelled from here: the Kaggle CLI has no cancel "
                            "command. Stop the session in the Kaggle web interface; the supervisor will observe the "
                            "cancelled state, keep the compute slot until then, and report it."])
