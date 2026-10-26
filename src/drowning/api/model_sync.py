"""Install the registry-approved Drive archive into an immutable version cache."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from drowning.api.inference import InferenceUnavailable
from drowning.api.media import safe_local_root
from drowning.db.models import Artifact, ModelRegistry, ModelVersion
from drowning.integrations.drive import sha256_file
from drowning.ml.bundle import load_bundle
from drowning.orchestration.candidate_sync import verify_candidate_archive


def _check_identity(manifest, model):
    if (UUID(str(manifest.get("model_version_id", ""))) != model.id
            or manifest.get("model_family") != model.model_family
            or manifest.get("class_keys") != model.class_keys
            or manifest.get("dataset_fingerprint") != model.dataset_fingerprint):
        raise ValueError("Downloaded model identity/ontology/data fingerprint differs from the registry")


def sync_approved_model(session: Session, settings, drive) -> dict:
    registry = session.get(ModelRegistry, settings.model_family)
    model = session.get(ModelVersion, registry.approved_model_id) if registry and registry.approved_model_id else None
    if model is None or model.status != "approved":
        raise InferenceUnavailable("No approved temporal candidate is available to synchronize")
    return _sync_model(session, settings, drive, model)


def sync_model_version(session: Session, settings, drive, model_version_id: UUID) -> dict:
    """Install one explicitly chosen version (including an unapproved candidate) for local evaluation."""
    model = session.get(ModelVersion, model_version_id)
    if model is None or model.status not in {"approved", "candidate", "superseded"}:
        raise InferenceUnavailable("That model version is not available to synchronize")
    return _sync_model(session, settings, drive, model)


def _sync_model(session: Session, settings, drive, model: ModelVersion) -> dict:
    if model.model_kind is None or model.model_family != settings.model_family:
        raise InferenceUnavailable("Only a completed candidate of the active model family can be synchronized")
    artifact = session.get(Artifact, model.artifact_id) if model.artifact_id else None
    if (artifact is None or artifact.status != "ready" or artifact.storage_provider != "google_drive"
            or not artifact.remote_file_id or len(artifact.sha256) != 64 or artifact.size_bytes <= 0):
        raise InferenceUnavailable("Approved model has no verified Drive archive")
    # End the read transaction before network/file operations; the captured model stays pinned.
    session.expunge(model)
    session.expunge(artifact)
    session.rollback()
    root = safe_local_root(settings.approved_model_bundle_dir)
    destination = root / str(model.id)
    if destination.exists():
        loaded, manifest = load_bundle(destination, require_temporal=True, map_location="cpu")
        del loaded
        _check_identity(manifest, model)
        return {"model_version_id": str(model.id), "cached": True, "downloaded": False, "model_loaded": False,
                "status": model.status}
    with tempfile.TemporaryDirectory(prefix=".model-sync-", dir=root) as temp:
        directory = Path(temp)
        archive = directory / "candidate.zip"
        receipt = drive.download(artifact.remote_file_id, archive, expected_sha256=artifact.sha256)
        if (receipt.get("verified") is not True or not archive.is_file()
                or archive.stat().st_size != artifact.size_bytes or sha256_file(archive) != artifact.sha256):
            raise ValueError("Downloaded approved archive failed registry size/checksum verification")
        unpacked = directory / "unpacked"
        verified = verify_candidate_archive(archive, unpacked)
        _check_identity(verified["manifest"], model)
        # Atomic directory install on the same filesystem, preserving all previous versions.
        try:
            (unpacked / "model-bundle").rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            loaded, manifest = load_bundle(destination, require_temporal=True, map_location="cpu")
            del loaded
            _check_identity(manifest, model)
    return {"model_version_id": str(model.id), "cached": True, "downloaded": True, "model_loaded": False,
            "status": model.status}
