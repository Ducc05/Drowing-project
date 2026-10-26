"""Read-only loader and validator for versioned dataset source descriptors."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from drowning.data.labels import CANONICAL_LABELS, resolve_label
from drowning.data.review import apply_reviews

SUPPORTED_FORMATS = {"yolo_detection", "coco_detection", "pascal_voc", "cvat_video", "reviewed_jsonl", "raw_media"}
SUPPORTED_MODALITIES = {"image", "video", "mixed"}
MAPPING_ACTIONS = {"map", "ignore", "context", "review"}


class SourceConfigError(ValueError):
    pass


def load_source(path: Path, reviews_dir: Path | None = None) -> dict[str, Any]:
    """Load a descriptor and overlay valid human reviews from the project's ``reviews/`` directory.

    The overlay applies only for the standard ``<project>/configs/datasets/*.yaml`` layout (or an
    explicit ``reviews_dir``), so descriptors elsewhere never inherit approvals by accident.
    """
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SourceConfigError(f"Cannot read source config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceConfigError("Source config must be a YAML object")
    resolved = path.resolve()
    if reviews_dir is None and resolved.parent.name == "datasets" and resolved.parent.parent.name == "configs":
        reviews_dir = resolved.parents[2] / "reviews"
    if reviews_dir is not None and isinstance(value.get("mapping"), dict) and "source_id" in value and "source_version" in value:
        value = apply_reviews(value, reviews_dir)
    return value


def validate_source(source: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ("schema_version", "source_id", "source_version", "format", "modality", "paths", "mapping")
    for key in required:
        if key not in source:
            errors.append(f"missing required field: {key}")
    if source.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if source.get("format") not in SUPPORTED_FORMATS:
        errors.append(f"unsupported format: {source.get('format')!r}")
    if source.get("modality") not in SUPPORTED_MODALITIES:
        errors.append(f"unsupported modality: {source.get('modality')!r}")
    paths = source.get("paths")
    if not isinstance(paths, dict) or not isinstance(paths.get("root_env"), str) or not paths["root_env"].strip():
        errors.append("paths.root_env must name an environment variable")
    mapping = source.get("mapping")
    if not isinstance(mapping, dict):
        errors.append("mapping must be an object")
    else:
        seen: set[str] = set()
        for label, rule in mapping.items():
            if not isinstance(label, str) or not isinstance(rule, dict):
                errors.append("mapping entries must map label strings to rule objects")
                continue
            decision_key = label.strip().casefold().replace("_", " ").replace("-", " ")
            if decision_key in seen:
                errors.append(f"mapping contains normalized duplicate label: {label!r}")
            seen.add(decision_key)
            action = rule.get("action")
            if action not in MAPPING_ACTIONS:
                errors.append(f"mapping[{label!r}].action must be one of {sorted(MAPPING_ACTIONS)}")
            if action == "map" and rule.get("target") not in CANONICAL_LABELS:
                errors.append(f"mapping[{label!r}] has invalid canonical target")
    return errors


def resolve_source_root(source: dict[str, Any], environ: dict[str, str] | None = None) -> Path | None:
    env = os.environ if environ is None else environ
    root_key = source.get("paths", {}).get("root_env")
    raw_root = env.get(root_key, "") if root_key else ""
    return Path(raw_root).expanduser() if raw_root.strip() else None


def inspect_source(source: dict[str, Any], environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Report configuration and exact mapping outcomes without changing any input."""
    errors = validate_source(source)
    root = resolve_source_root(source, environ)
    mapping = source.get("mapping", {}) if isinstance(source.get("mapping"), dict) else {}
    decisions = [resolve_label(str(label), mapping).__dict__ for label in mapping]
    return {
        "source_id": source.get("source_id"),
        "source_version": source.get("source_version"),
        "format": source.get("format"),
        "modality": source.get("modality"),
        "valid_config": not errors,
        "errors": errors,
        "root_env": source.get("paths", {}).get("root_env") if isinstance(source.get("paths"), dict) else None,
        "root_configured": root is not None,
        "root_exists": root.is_dir() if root else False,
        "mapping": decisions,
        "unknown_labels_default_to": "review",
        "conversion_performed": False,
    }


def read_class_names(source: dict[str, Any], root: Path) -> list[tuple[int | str, str]]:
    """Read declared class names from supported YOLO or COCO metadata files."""
    path_config = source.get("paths", {})
    declared = (source.get("source_labels") or {}).get("class_names")
    if declared is not None and not path_config.get("class_list"):
        # For datasets that were uploaded without data.yaml: the descriptor states which index is which name.
        items = list(declared.items()) if isinstance(declared, dict) else list(enumerate(declared)) if isinstance(declared, list) else []
        if not items or any(name is None or not str(name).strip() for _, name in items):
            raise SourceConfigError("source_labels.class_names must be a nonempty list or an id-to-name mapping")
        return [(class_id, str(name)) for class_id, name in items]
    spec = path_config.get("class_list")
    if not isinstance(spec, str) or ":" not in spec:
        raise SourceConfigError("paths.class_list must be a relative_file:field reference")
    relative, field = spec.split(":", 1)
    metadata_path = (root / relative).resolve()
    try:
        metadata_path.relative_to(root.resolve())
    except ValueError as exc:
        raise SourceConfigError("Class-list path escapes the configured source root") from exc
    if not metadata_path.is_file():
        raise SourceConfigError(f"Class-list file not found: {relative}")
    if metadata_path.suffix.lower() == ".json":
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        data = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    for part in field.split("."):
        if not isinstance(data, dict) or part not in data:
            raise SourceConfigError(f"Field {field!r} not found in {relative}")
        data = data[part]
    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, list):
        if field.endswith("categories"):
            items = [(entry.get("id"), entry.get("name")) for entry in data if isinstance(entry, dict)]
        else:
            items = list(enumerate(data))
    else:
        raise SourceConfigError("Class-list field must be a list or object")
    if not items or any(name is None or not str(name).strip() for _, name in items):
        raise SourceConfigError("Class-list contains no usable class names")
    return [(class_id, str(name)) for class_id, name in items]
