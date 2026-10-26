"""Approved-model resolution and local media inference helpers for the API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from PIL import Image, ImageDraw, ImageOps
from sqlalchemy.orm import Session

from drowning.config import Settings
from drowning.db.models import Artifact, ModelRegistry, ModelVersion


class InferenceUnavailable(RuntimeError):
    pass


def get_approved_model(session: Session, settings: Settings, *, pinned_model_id: UUID | None = None,
                       allow_candidate: bool = False) -> tuple[ModelVersion, Path, dict[str, Any]]:
    """Resolve the approved model, or an explicitly pinned version (a candidate only for evaluation)."""
    if pinned_model_id is None:
        registry = session.get(ModelRegistry, settings.model_family)
        if registry is None or registry.approved_model_id is None:
            raise InferenceUnavailable("No approved model is registered for this model family")
        model_record = session.get(ModelVersion, registry.approved_model_id)
        if model_record is None or model_record.status != "approved":
            raise InferenceUnavailable("Approved model pointer is missing or inconsistent")
    else:
        model_record = session.get(ModelVersion, pinned_model_id)
        allowed = {"approved", "superseded"} | ({"candidate"} if allow_candidate else set())
        if model_record is None or model_record.status not in allowed:
            raise InferenceUnavailable("Pinned job model version is unavailable")
    if (not model_record.temporal_trained or model_record.completion_kind != "candidate"
            or model_record.artifact_id is None or model_record.model_family != settings.model_family):
        raise InferenceUnavailable("Approved model is not a verified temporal candidate")
    artifact = session.get(Artifact, model_record.artifact_id)
    if artifact is None or artifact.status != "ready" or not artifact.remote_file_id:
        raise InferenceUnavailable("Approved model has no ready Drive artifact")
    bundle_dir = Path(settings.approved_model_bundle_dir).expanduser().resolve()
    version_dir = bundle_dir / str(model_record.id)
    if version_dir.is_dir():
        bundle_dir = version_dir
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise InferenceUnavailable("Approved model bundle is not synchronized into the local cache")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceUnavailable("Approved model bundle manifest is unreadable") from exc
    try:
        manifest_id = UUID(str(manifest.get("model_version_id", "")))
    except ValueError as exc:
        raise InferenceUnavailable("Local model bundle has no valid registry model-version ID") from exc
    if manifest_id != model_record.id:
        raise InferenceUnavailable("Local model bundle does not match the approved registry version")
    return model_record, bundle_dir, manifest


def load_approved_pipeline(session: Session, settings: Settings, *,
                           inference_factory: Callable[[ModelVersion, dict[str, Any], Path], Any] | None = None,
                           pinned_model_id: UUID | None = None, allow_candidate: bool = False):
    model_record, bundle_dir, manifest = get_approved_model(session, settings, pinned_model_id=pinned_model_id,
                                                            allow_candidate=allow_candidate)
    if inference_factory is not None:
        return inference_factory(model_record, manifest, bundle_dir), model_record
    if not settings.detector_weights_configured:
        raise InferenceUnavailable("PERSON_DETECTOR_WEIGHTS is not configured")
    try:
        from drowning.inference.detector import UltralyticsPersonDetector
        from drowning.inference.pipeline import PersonStateInference
        from drowning.ml.bundle import load_bundle

        model, loaded_manifest = load_bundle(bundle_dir, require_temporal=True, map_location=settings.model_device)
        detector = UltralyticsPersonDetector(Path(settings.person_detector_weights_path),
                                             person_class_id=settings.detector_class_id)
        pipeline = PersonStateInference(model, loaded_manifest, detector,
                                        model_version=model_record.display_version, device=settings.model_device)
        return pipeline, model_record
    except ImportError as exc:
        raise InferenceUnavailable("Install model, vision and detector dependencies before local inference") from exc
    except (ValueError, OSError, RuntimeError) as exc:
        raise InferenceUnavailable(f"Unable to load the pinned approved model/detector ({type(exc).__name__})") from exc


def run_image_test(pipeline: Any, source: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    result = pipeline.infer_image(image)
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for item in result["detections"]:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        draw.rectangle((x1, y1, x2, y2), outline=(52, 163, 135), width=3)
        draw.text((x1, max(0, y1 - 16)), f"{item.get('label') or 'unknown'} | {item['prediction_branch']}",
                  fill=(17, 35, 42), stroke_width=1, stroke_fill=(255, 255, 255))
    annotated_path = output_dir / "annotated.png"
    detections_path = output_dir / "detections.json"
    annotated.save(annotated_path, format="PNG", optimize=True)
    detections_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"model_version": result["model_version"], "detection_count": len(result["detections"]),
            "files": {"annotated": annotated_path.name, "detections": detections_path.name},
            "scores_are_calibrated_probabilities": False}


def run_video_test(pipeline: Any, source: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    output_video = output_dir / "annotated.webm"
    result = pipeline.infer_video(source, output_video)
    return {**result, "files": {"annotated": output_video.name,
                                "detections": Path(result["detections_path"]).name},
            "scores_are_calibrated_probabilities": False}
