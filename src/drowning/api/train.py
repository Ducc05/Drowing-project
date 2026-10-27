"""Train page backend: preflight gates, Kaggle job submission, and honest cancellation."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from drowning.api.review import SourceReviewService
from drowning.config import Settings
from drowning.data.registry import load_source
from drowning.data.review import is_mapping_approved, load_grouping_review, mapping_fingerprint
from drowning.data.review_rules import gating_mode
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
                 home: Path | None = None, environ: dict[str, str] | None = None, policy_path: Path | None = None):
        self.factory, self.settings, self.project_root, self.reviews = session_factory, settings, project_root, reviews
        self.which, self.home = which or (lambda _name: find_kaggle()), home or Path.home()
        self.environ = os.environ if environ is None else environ
        self.policy_path = policy_path or (project_root / "configs" / "review_policy.yaml")

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
        advisory = gating_mode(self.policy_path) == "advisory"
        unconfirmed = [f"{r['source_id']}: " + ("mapping not confirmed" if not r["mapping_approved"] else "source groups not confirmed")
                       for r in sources if not (r["mapping_approved"] and r["groups_verified"])]
        approvals_missing = [] if advisory else [
            f"{r['source_id']}: " + ("mapping not approved" if not r["mapping_approved"] else "source groups not verified")
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
        # Diagnostic jobs are optional. The training notebook validates its own
        # selected inputs and requires CUDA before starting the training loop.
        modes["remote_smoke"] = {"blocked_by": list(base)}
        train_block = list(base) + approvals_missing
        windows = (prepare_summary or {}).get("temporal_window_count_before_split", 0)
        # Old reports cannot establish which windows the current data produces.
        branch_blocked = {"temporal": [], "image_only": []}
        if not environment["drive_folder_configured"] or not environment["drive_token_file_present"]:
            train_block.append("Google Drive must be configured to publish a candidate")
        if approved is not None:
            train_block.append("an approved model already exists; training a new one needs an explicit new version")
        modes["train_candidate"] = {"blocked_by": train_block, "requires_new_version_flag": True,
                                    "requires_prepare": False, "requires_smoke": False,
                                    "branch_blocked_by": branch_blocked}
        for entry in modes.values():
            effective = [b for b in entry["blocked_by"] if not b.startswith("an approved model already exists")]
            entry["allowed"] = not effective
        return {
            "capabilities": {"continue_from_approved": True, "source_selection": True,
                             "incremental_prepare": True, "direct_train": True},
            "environment": environment, "sources": sources, "stale_sources": stale,
            "latest_prepare": self._run_view(prepare), "latest_smoke": self._run_view(smoke),
            "latest_job": ({"job_id": str(recent.id), "mode": (recent.payload or {}).get("mode"), "state": recent.state}
                           if recent else None),
            "approved_model": ({"id": str(approved.id), "display_version": approved.display_version,
                                "model_kind": approved.model_kind,
                                "cached_locally": (Path(self.settings.approved_model_bundle_dir) / str(approved.id) / "manifest.json").is_file()}
                               if approved else None),
            "compute_slot_held_by": ({"job_id": str(active.id), "job_type": active.job_type,
                                      "mode": (active.payload or {}).get("mode") or (active.payload or {}).get("modality"),
                                      "state": active.state} if active else None),
            "queued_kaggle_jobs": len(queued), "modes": modes,
            "review_mode": "advisory" if advisory else "enforced",
            "unconfirmed_sources": unconfirmed if advisory else [],
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

    def _prepare_cache(self) -> dict:
        """Recover snapshots only from locally available, checksum-verified run archives."""
        from drowning.data.prepare_cache import MANIFEST, bundle_snapshots, file_sha256
        from drowning.orchestration.run_collector import verify_run_archive

        with self.factory() as session:
            jobs = session.scalars(select(Job).where(Job.state == "completed")
                                   .order_by(Job.created_at.desc())).all()
            records = [(str(j.id), dict(j.payload or {})) for j in jobs
                       if (j.payload or {}).get("mode") == "prepare_only"][:20]
        with tempfile.TemporaryDirectory(prefix="drown-prepare-cache-") as temporary:
            candidates = []
            for job_id, payload in records:
                archive = Path(payload.get("run_archive") or "")
                digest = payload.get("run_archive_sha256")
                if not digest or not archive.is_file():
                    continue
                try:
                    if file_sha256(archive) != digest:
                        continue
                    with zipfile.ZipFile(archive) as bundle:
                        if not any(name.endswith("/" + MANIFEST) for name in bundle.namelist()):
                            continue  # legacy output has no content-bound cache identity
                    extracted = Path(temporary) / job_id
                    verify_run_archive(archive, extracted, expected_mode="prepare_only")
                    candidates.extend((job_id, p.parent) for p in sorted(extracted.rglob(MANIFEST)))
                except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                    continue
            return bundle_snapshots(candidates)

    def _training_identity(self, payload: dict[str, Any], view: dict[str, Any]) -> str:
        """Hash of everything that decides the trained weights: data, mappings, recipes, code, options, seed."""
        recipes = self.project_root / "configs" / "recipes"
        document = {"dataset": payload["dataset_fingerprint"],
                    "dataset_request": payload.get("dataset_request_fingerprint"),
                    "mappings": sorted((s["source_id"], s["mapping_hash"]) for s in view["sources"]),
                    "recipes": [hashlib.sha256((recipes / name).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()
                                for name in ("image.yaml", "temporal.yaml")],
                    "code": self._notebook_builder().source_bundle_sha256(), "options": payload.get("options"),
                    "seed": 20261026, "parent_model": payload.get("parent_model")}
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

    def _parent_checkpoint(self, approved: dict[str, Any] | None, branch: str) -> tuple[Path, dict[str, Any]]:
        """Capture the single approved model; never download or change the registry during submission."""
        from drowning.api.inference import get_approved_model, InferenceUnavailable
        from drowning.api.model_sync import _check_identity
        from drowning.ml.bundle import load_bundle

        if not approved:
            raise TrainBlocked(["Train tiếp cần một model chính đã duyệt."])
        try:
            with self.factory() as session:
                model, directory, _ = get_approved_model(session, self.settings, pinned_model_id=UUID(approved["id"]))
                if model.status != "approved":
                    raise ValueError("model chính đã thay đổi; tải lại trang Train")
                expected = "image_tracked" if branch == "image_only" else "temporal"
                if model.model_kind != expected:
                    raise ValueError("loại model gốc không khớp chế độ train")
                loaded, manifest = load_bundle(directory, require_temporal=True, map_location="cpu")
                del loaded
                _check_identity(manifest, model)
                weights_sha = manifest["files"]["weights.pt"]["sha256"]
                recorded_sha = (model.metrics or {}).get("provenance", {}).get("weights_sha256")
                if recorded_sha and weights_sha != recorded_sha:
                    raise ValueError("checkpoint khác checksum đã đăng ký")
                return directory, {"id": str(model.id), "model_kind": model.model_kind,
                                   "weights_sha256": weights_sha,
                                   "manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()}
        except (InferenceUnavailable, ValueError, OSError, RuntimeError) as exc:
            raise TrainBlocked([f"Không thể nạp model gốc: {exc}. Đồng bộ model đã duyệt về máy trước."]) from exc

    def submit(self, mode: str, *, new_version: bool, reason: str, idempotency_key: str,
               options: dict[str, Any] | None = None) -> dict[str, Any]:
        if mode not in MODES:
            raise TrainBlocked([f"mode must be one of {', '.join(MODES)}"])
        if len(idempotency_key.strip()) < 8:
            raise TrainBlocked(["idempotency_key must contain at least eight characters"])
        with self.factory() as session:
            prior = session.scalar(select(Job).where(Job.idempotency_key == idempotency_key))
            if prior is not None:
                old = prior.payload or {}
                same = (old.get("executor") == "kaggle" and old.get("mode") == mode
                        and old.get("new_version") == bool(new_version) and old.get("reason") == reason.strip())
                if mode == "train_candidate":
                    try:
                        same = same and self._notebook_builder().normalize_options(old.get("options")) == self._notebook_builder().normalize_options(options)
                    except ValueError as exc:
                        raise TrainBlocked([str(exc)]) from exc
                if not same:
                    raise TrainBlocked(["Idempotency-Key is already associated with a different training request"])
                return {"status": "duplicate_request", "created": False,
                        "job": {"id": str(prior.id), "state": prior.state, "mode": mode}}
        view = self.preflight()
        if mode == "train_candidate" and view["approved_model"] and not new_version:
            return {"status": "model_available", "job": None, "approved_model": view["approved_model"],
                    "message": "An approved model already exists, so no training job was created. Use Test, or request a new version explicitly."}
        blocked = [reason_ for reason_ in view["modes"][mode]["blocked_by"]
                   if not (mode == "train_candidate" and new_version and reason_.startswith("an approved model already exists"))]
        if mode == "train_candidate":
            try:
                chosen = self._notebook_builder().normalize_options(options)
            except (ValueError, KeyError) as exc:
                raise TrainBlocked([f"invalid training options: {exc}"]) from exc
            blocked += view["modes"]["train_candidate"]["branch_blocked_by"][chosen["branch"]]
            if not new_version:
                blocked.append("train_candidate must be requested as an explicit new version")
            if len(reason.strip()) < 10:
                blocked.append("a reason (10+ characters) is required for a new model version")
            if chosen.get("source_ids") is not None:
                unknown = set(chosen["source_ids"]) - {row["source_id"] for row in view["sources"]}
                if unknown:
                    blocked.append("unknown dataset source IDs: " + ", ".join(sorted(unknown)))
        if blocked:
            raise TrainBlocked(blocked)
        # This is the remote execution deadline, not the upload request timeout.
        # Data preparation can exceed an hour; allow prepare/train the documented
        # 12-hour CPU/GPU session budget instead of ending them prematurely.
        if mode in {"prepare_only", "remote_smoke"}:
            active = self._active_run(mode, idempotency_key)
            if active is not None:
                return active
        payload: dict[str, Any] = {"mode": mode, "executor": "kaggle", "new_version": bool(new_version),
                                   "reason": reason.strip(), "requested_via": "web",
                                   "push_timeout_seconds": {"prepare_only": 43200, "remote_smoke": 1800, "train_candidate": 43200}[mode]}
        if mode == "train_candidate":
            try:
                payload["options"] = self._notebook_builder().normalize_options(options)
            except (ValueError, KeyError) as exc:
                raise TrainBlocked([f"invalid training options: {exc}"]) from exc
        summary = (view["latest_prepare"] or {}).get("summary") or {}
        parent_dir = None
        if mode == "train_candidate":
            if payload["options"]["continue_from_approved"]:
                parent_dir, payload["parent_model"] = self._parent_checkpoint(view["approved_model"], payload["options"]["branch"])
            selected = payload["options"].get("source_ids")
            selected_sources = [s for s in view["sources"] if selected is None or s["source_id"] in selected]
            if selected is not None:
                summary = {**summary, "sources": [s for s in summary.get("sources", []) if s["source_id"] in selected]}
            seen = {s["source_id"]: s for s in summary.get("sources", [])}
            prepared_snapshot = bool(selected_sources) and all(
                s["source_id"] in seen and seen[s["source_id"]].get("status") == "prepared"
                and seen[s["source_id"]].get("mapping_hash") == s["mapping_hash"]
                and seen[s["source_id"]].get("source_version") == s["source_version"]
                and re.fullmatch(r"[a-f0-9]{64}", seen[s["source_id"]].get("canonical_sha256") or "")
                for s in selected_sources)
            request_identity = {"sources": [(s["source_id"], s["source_version"], s["mapping_hash"]) for s in selected_sources],
                                "attached_datasets": sorted(self.settings.kaggle_dataset_sources)}
            request_fingerprint = hashlib.sha256(json.dumps(request_identity, sort_keys=True).encode()).hexdigest()
            payload.update(dataset_fingerprint=self._dataset_fingerprint(summary) if prepared_snapshot else None,
                           dataset_request_fingerprint=request_fingerprint,
                           dataset_identity_pending=not prepared_snapshot, preflight={
                "ready": True, "review_mode": view["review_mode"],
                "mapping_review_approved": all(s["mapping_approved"] for s in selected_sources),
                "source_video_disjointness_verified": all(s["groups_verified"] for s in selected_sources),
                "temporal_window_count": view["temporal_window_count_before_split"] if prepared_snapshot else 0,
                "runtime_data_validation": True})
            identity = self._training_identity(payload, {**view, "sources": selected_sources})
            if prepared_snapshot:
                payload["training_identity"] = identity
                existing = self._same_training(identity)
                if existing is not None:
                    return existing
            else:
                # A request hash is not a verified data-content hash. Resolve
                # that identity from the notebook's actual candidate output.
                payload["training_request_identity"] = identity
                payload["training_identity"] = None
        with self.factory() as session:
            job, created = create_job(session, job_type="prepare" if mode == "prepare_only" else "train",
                                      idempotency_key=idempotency_key, payload=payload)
        if created:
            kernel_root = Path(self.settings.kaggle_kernel_dir).expanduser()
            target = kernel_root / "jobs" / f"{str(job.id)[:8]}-{mode}"
            parent_dataset_ref = (self.settings.kaggle_kernel_ref.split("/")[0] + "/drown-parent-" + job.id.hex
                                  if parent_dir is not None else None)
            try:
                prepare_cache = self._prepare_cache() if mode == "prepare_only" else None
                path = self._materializer().materialize_kernel(self.settings.kaggle_kernel_ref, target, mode=mode,
                                                               dataset_sources=list(self.settings.kaggle_dataset_sources),
                                                               options=payload.get("options"), parent_bundle_dir=parent_dir,
                                                               parent_model=payload.get("parent_model"),
                                                               parent_dataset_ref=parent_dataset_ref, prepare_cache=prepare_cache)
            except Exception as exc:
                with self.factory() as session:
                    transition_job(session, job.id, JobState.FAILED, event_type="kernel_materialization_failed",
                                   phase="not_dispatched", details={"error_type": type(exc).__name__})
                raise TrainBlocked([f"could not prepare the Kaggle kernel files ({type(exc).__name__}: {exc})"]) from exc
            with self.factory() as session:
                stored = session.get(Job, job.id)
                stored.payload = {**stored.payload, "kernel_dir": str(path.resolve()),
                                  "parent_dataset_ref": parent_dataset_ref,
                                  "prepare_cache_source_versions": (prepare_cache or {}).get("source_versions", 0)}
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
