"""Validate canonical JSONL records and report exact-content label conflicts."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from drowning.data.labels import CANONICAL_LABELS

_ACTIONS = {"map", "ignore", "context", "review"}


def audit_jsonl(paths: list[Path]) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    media_by_id: dict[str, dict[str, Any]] = {}
    observations: list[dict[str, Any]] = []
    mapping_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    record_counts: Counter[str] = Counter()
    duplicate_record_keys: set[tuple[str, str]] = set()

    for input_path in paths:
        try:
            lines = input_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            issues.append({"file": str(input_path), "line": None, "code": "file_read_error", "detail": str(exc)})
            continue
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append({"file": str(input_path), "line": line_number, "code": "invalid_json", "detail": exc.msg})
                continue
            if not isinstance(record, dict):
                issues.append({"file": str(input_path), "line": line_number, "code": "record_not_object"})
                continue
            entity = record.get("entity")
            record_counts[str(entity)] += 1
            source_id = str(record.get("source_id", ""))
            source_counts[source_id] += 1
            if record.get("schema_version") != 1:
                issues.append({"file": str(input_path), "line": line_number, "code": "unsupported_schema_version"})
            if entity == "media":
                media_id = record.get("media_id")
                if not isinstance(media_id, str) or not media_id:
                    issues.append({"file": str(input_path), "line": line_number, "code": "missing_media_id"})
                elif media_id in media_by_id:
                    issues.append({"file": str(input_path), "line": line_number, "code": "duplicate_media_id", "media_id": media_id})
                else:
                    media_by_id[media_id] = record
                if not isinstance(record.get("content_sha256"), str) or len(record["content_sha256"]) != 64:
                    issues.append({"file": str(input_path), "line": line_number, "code": "invalid_content_sha256"})
                for dimension in ("width", "height"):
                    if not isinstance(record.get(dimension), int) or record[dimension] <= 0:
                        issues.append({"file": str(input_path), "line": line_number, "code": f"invalid_{dimension}"})
            elif entity == "observation":
                observations.append({"record": record, "file": str(input_path), "line": line_number})
                observation_id = record.get("observation_id")
                key = ("observation", str(observation_id))
                if not observation_id:
                    issues.append({"file": str(input_path), "line": line_number, "code": "missing_observation_id"})
                elif key in duplicate_record_keys:
                    issues.append({"file": str(input_path), "line": line_number, "code": "duplicate_observation_id", "observation_id": observation_id})
                duplicate_record_keys.add(key)
                action, canonical = record.get("mapping_action"), record.get("canonical_label")
                if action not in _ACTIONS:
                    issues.append({"file": str(input_path), "line": line_number, "code": "invalid_mapping_action"})
                if action == "map":
                    if canonical not in CANONICAL_LABELS:
                        issues.append({"file": str(input_path), "line": line_number, "code": "invalid_canonical_label"})
                    else:
                        class_counts[f"{source_id}:{canonical}"] += 1
                elif canonical is not None:
                    issues.append({"file": str(input_path), "line": line_number, "code": "nonmapped_record_has_canonical_label"})
                mapping_counts[f"{source_id}:{action}"] += 1
                bbox = record.get("bbox")
                outside_marker = record.get("outside") is True and bbox is None
                if not outside_marker and (not isinstance(bbox, list) or len(bbox) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox)):
                    issues.append({"file": str(input_path), "line": line_number, "code": "invalid_bbox"})
                elif isinstance(bbox, list) and (bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
                    issues.append({"file": str(input_path), "line": line_number, "code": "nonpositive_bbox_area"})
            else:
                issues.append({"file": str(input_path), "line": line_number, "code": "unknown_entity", "entity": entity})

    for item in observations:
        record = item["record"]
        media = media_by_id.get(record.get("media_id"))
        if media is None:
            issues.append({"file": item["file"], "line": item["line"], "code": "missing_media_reference", "media_id": record.get("media_id")})
            continue
        bbox = record.get("bbox")
        if record.get("outside") is True and bbox is None:
            continue
        if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox):
            if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > media.get("width", 0) or bbox[3] > media.get("height", 0):
                issues.append({"file": item["file"], "line": item["line"], "code": "bbox_outside_media", "media_id": record.get("media_id")})

    if sum(record_counts.values()) == 0 and not issues:
        issues.append({"file": None, "line": None, "code": "no_records"})

    media_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for media in media_by_id.values():
        fingerprint = media.get("content_sha256")
        if isinstance(fingerprint, str) and len(fingerprint) == 64:
            media_by_hash[fingerprint].append(media)
    duplicate_groups = []
    conflicts = []
    cross_split_duplicate_groups = 0
    cross_split_duplicate_extra_media = 0
    for fingerprint, group in media_by_hash.items():
        if len(group) < 2:
            continue
        group_ids = {media.get("media_id") for media in group}
        splits = sorted({str(media.get("original_export_split") or "unknown") for media in group})
        cross_split = len([split for split in splits if split != "unknown"]) > 1
        if cross_split:
            cross_split_duplicate_groups += 1
            cross_split_duplicate_extra_media += len(group) - 1
        duplicate_groups.append({
            "content_sha256": fingerprint,
            "media_ids": sorted(group_ids),
            "sources": sorted({str(media.get("source_id")) for media in group}),
            "original_export_splits": splits,
            "cross_split_duplicate": cross_split,
        })
        labels_by_box: dict[tuple[str, tuple[float, ...]], set[str]] = defaultdict(set)
        for item in observations:
            record = item["record"]
            if record.get("media_id") in group_ids and record.get("mapping_action") == "map" and record.get("canonical_label"):
                bbox = record.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                    labels_by_box[(str(record.get("media_id")), tuple(float(v) for v in bbox))].add(record["canonical_label"])
        label_sets: dict[tuple[float, ...], set[str]] = defaultdict(set)
        for (_, bbox), labels in labels_by_box.items():
            label_sets[bbox].update(labels)
        for bbox, labels in label_sets.items():
            if len(labels) > 1:
                conflicts.append({"content_sha256": fingerprint, "bbox": list(bbox), "labels": sorted(labels), "disposition": "review_all_matching_annotations"})

    return {
        "schema_version": 1,
        "input_files": [str(path) for path in paths],
        "record_counts": dict(sorted(record_counts.items())),
        "canonical_class_counts_by_source": dict(sorted(class_counts.items())),
        "mapping_action_counts_by_source": dict(sorted(mapping_counts.items())),
        "record_counts_by_source": dict(sorted(source_counts.items())),
        "exact_duplicate_content_groups": duplicate_groups,
        "cross_split_duplicate_group_count": cross_split_duplicate_groups,
        "cross_split_duplicate_extra_media_count": cross_split_duplicate_extra_media,
        "same_box_label_conflicts": conflicts,
        "issues": issues,
        "valid": not issues,
        "automatic_quarantine_performed": False,
        "source_video_disjointness_verified": False,
    }
