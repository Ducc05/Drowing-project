"""Versioned model bundle save/load with checksum and schema validation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

from drowning.ml.model import CLASS_KEYS, MODEL_SCHEMA_VERSION, ONTOLOGY_ID, ONTOLOGY_VERSION, DrowningTemporalModel, torch


MODEL_KINDS = ("temporal", "image_tracked")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    packages = ("torch", "torchvision", "Pillow", "numpy")
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package.lower()] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package.lower()] = "not-installed"
    return versions


def save_bundle(model: DrowningTemporalModel, destination: Path, *, completion_kind: str,
                temporal_trained: bool, preprocessing: dict[str, Any], dataset_fingerprint: str | None = None,
                temporal_contract: dict[str, Any] | None = None,
                recipe_fingerprint: str | None = None, code_fingerprint: str | None = None,
                training_job_id: str | None = None, remote_kernel_ref: str | None = None,
                remote_kernel_version: str | None = None, parent_model_version: str | None = None,
                model_version_id: str | None = None, model_kind: str = "temporal",
                aggregation: dict[str, Any] | None = None) -> dict[str, Any]:
    if model_kind not in MODEL_KINDS:
        raise ValueError(f"model_kind must be one of {', '.join(MODEL_KINDS)}")
    if completion_kind not in {"synthetic_smoke", "candidate", "approved"}:
        raise ValueError("completion_kind must be synthetic_smoke, candidate, or approved")
    if completion_kind == "approved":
        raise ValueError("Approval is a registry decision and cannot be assigned while saving a bundle")
    preprocessing_keys = {"size", "context_fraction", "mean", "std"}
    if not isinstance(preprocessing, dict) or not preprocessing_keys.issubset(preprocessing):
        raise ValueError("model bundles require size, context_fraction, mean and std preprocessing metadata")
    if len(preprocessing["mean"]) != 3 or len(preprocessing["std"]) != 3:
        raise ValueError("preprocessing mean/std must have three channels")
    if model_kind == "image_tracked":
        from drowning.inference.aggregation import validate_aggregation

        if temporal_trained:
            raise ValueError("an image_tracked bundle has no trained temporal head")
        problems = validate_aggregation(aggregation)
        if problems:
            raise ValueError("image_tracked bundles require a valid aggregation contract: " + "; ".join(problems))
    if temporal_trained:
        required_contract = {"sequence_length", "sample_period_ms", "max_track_gap_ms", "causal"}
        if not isinstance(temporal_contract, dict) or not required_contract.issubset(temporal_contract):
            raise ValueError("trained temporal bundles require the exact sampling/causal contract")
        if temporal_contract.get("causal") is not True:
            raise ValueError("only causal temporal bundles are supported")
    target = destination.resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {target}")
    try:
        target.relative_to(Path("D:/DrowningDemo").resolve())
        protected_demo = True
    except ValueError:
        protected_demo = False
    if protected_demo or any(part.casefold() == "sources" for part in target.parts):
        raise ValueError("Model bundle output is under a protected path")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        weights_path = temp_dir / "weights.pt"
        torch.save(model.state_dict(), weights_path)
        manifest = {
            "bundle_schema_version": 1,
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "model_family": "drowning-person-temporal-v1",
            "encoder": "mobilenet_v3_small",
            "temporal_head": "causal_gru_cell",
            "hidden_size": model.hidden_size,
            "class_keys": list(CLASS_KEYS),
            "ontology_id": ONTOLOGY_ID,
            "ontology_version": ONTOLOGY_VERSION,
            "preprocessing": preprocessing,
            "temporal_contract": temporal_contract,
            "runtime_versions": runtime_versions(),
            "dataset_fingerprint": dataset_fingerprint,
            "recipe_fingerprint": recipe_fingerprint,
            "code_fingerprint": code_fingerprint,
            "training_job_id": training_job_id,
            "remote_kernel_ref": remote_kernel_ref,
            "remote_kernel_version": remote_kernel_version,
            "parent_model_version": parent_model_version,
            "model_version_id": model_version_id,
            "initialization": {"encoder_pretrained": model.pretrained_encoder,
                               "weights_source": "torchvision_default_imagenet" if model.pretrained_encoder else "random_initialization"},
            "model_kind": model_kind,
            "aggregation": aggregation if model_kind == "image_tracked" else None,
            "temporal_trained": bool(temporal_trained),
            "completion_kind": completion_kind,
            "files": {"weights.pt": {"sha256": sha256_file(weights_path), "bytes": weights_path.stat().st_size}},
        }
        (temp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_dir.rename(target)
        return manifest
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def validate_manifest(manifest: dict[str, Any], *, require_temporal: bool = True, allow_smoke: bool = False) -> list[str]:
    errors: list[str] = []
    if manifest.get("bundle_schema_version") != 1:
        errors.append("unsupported bundle_schema_version")
    if manifest.get("model_schema_version") != MODEL_SCHEMA_VERSION:
        errors.append("unsupported model_schema_version")
    if tuple(manifest.get("class_keys", ())) != CLASS_KEYS:
        errors.append("class order does not match ontology version 1")
    if manifest.get("ontology_id") != ONTOLOGY_ID or manifest.get("ontology_version") != ONTOLOGY_VERSION:
        errors.append("ontology does not match expected contract")
    preprocessing = manifest.get("preprocessing")
    if not isinstance(preprocessing, dict) or not {"size", "context_fraction", "mean", "std"}.issubset(preprocessing):
        errors.append("bundle preprocessing contract is missing")
    elif len(preprocessing.get("mean", [])) != 3 or len(preprocessing.get("std", [])) != 3:
        errors.append("bundle preprocessing mean/std must have three channels")
    kind = manifest.get("model_kind", "temporal")
    if kind not in MODEL_KINDS:
        errors.append("unknown model_kind")
    if kind == "image_tracked":
        # Image encoder + per-track aggregation: usable for inference without a trained temporal head.
        from drowning.inference.aggregation import validate_aggregation

        if manifest.get("temporal_trained"):
            errors.append("an image_tracked bundle cannot claim a trained temporal head")
        errors.extend(validate_aggregation(manifest.get("aggregation")))
    elif require_temporal and not manifest.get("temporal_trained", False):
        errors.append("bundle has no trained temporal head")
    if require_temporal and kind != "image_tracked":
        contract = manifest.get("temporal_contract")
        required_contract = {"sequence_length", "sample_period_ms", "max_track_gap_ms", "causal"}
        if not isinstance(contract, dict) or not required_contract.issubset(contract) or contract.get("causal") is not True:
            errors.append("bundle temporal sampling/causal contract is missing or invalid")
        elif (not isinstance(contract.get("sequence_length"), int) or contract["sequence_length"] < 2
              or not isinstance(contract.get("sample_period_ms"), int) or contract["sample_period_ms"] <= 0
              or not isinstance(contract.get("max_track_gap_ms"), int)
              or contract["max_track_gap_ms"] < contract["sample_period_ms"]):
            errors.append("bundle temporal sampling values are invalid")
    if manifest.get("completion_kind") == "synthetic_smoke" and not allow_smoke:
        errors.append("synthetic smoke bundles cannot be used as model candidates")
    return errors


def load_bundle(source: Path, *, require_temporal: bool = True, allow_smoke: bool = False, map_location: str = "cpu") -> tuple[DrowningTemporalModel, dict[str, Any]]:
    bundle_dir = source.resolve()
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Bundle is missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate_manifest(manifest, require_temporal=require_temporal, allow_smoke=allow_smoke)
    if errors:
        raise ValueError("Invalid model bundle: " + "; ".join(errors))
    weights = bundle_dir / "weights.pt"
    expected = manifest.get("files", {}).get("weights.pt", {})
    if not weights.is_file() or weights.stat().st_size != expected.get("bytes") or sha256_file(weights) != expected.get("sha256"):
        raise ValueError("Model weights are missing or fail checksum validation")
    model = DrowningTemporalModel(hidden_size=int(manifest["hidden_size"]))
    state = torch.load(weights, map_location=map_location, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, manifest
