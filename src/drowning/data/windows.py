"""Causal temporal-window preflight and manifest builder for reviewed tracks."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from drowning.data.labels import CANONICAL_LABELS


def build_temporal_windows(canonical_jsonl: Path, *, sequence_length: int, sample_period_ms: int,
                           max_track_gap_ms: int, min_valid_ratio: float = 0.75,
                           split_assignments: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    if sequence_length < 2 or sample_period_ms <= 0 or max_track_gap_ms < sample_period_ms:
        raise ValueError("invalid temporal sampling contract")
    if not 0 < min_valid_ratio <= 1:
        raise ValueError("min_valid_ratio must be in (0,1]")
    media: dict[str, dict[str, Any]] = {}
    observations: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for line_number, line in enumerate(canonical_jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append({"line": line_number, "code": "invalid_json", "detail": exc.msg})
            continue
        if record.get("entity") == "media":
            media[record.get("media_id")] = record
        elif record.get("entity") == "observation":
            observations.append(record)
    split_assignments = split_assignments or {}
    missing_split_media = []
    split_by_group: dict[str, set[str]] = defaultdict(set)
    split_by_hash: dict[str, set[str]] = defaultdict(set)
    split_issues: list[dict[str, Any]] = []
    for media_id, item in media.items():
        assignment = split_assignments.get(str(media_id))
        if assignment is None:
            missing_split_media.append(str(media_id))
            continue
        split = assignment.get("split")
        assigned_group = assignment.get("global_group_id")
        group_id = item.get("global_group_id")
        if split not in {"train", "validation", "test"}:
            split_issues.append({"media_id": media_id, "code": "invalid_split_assignment"})
            continue
        if item.get("grouping_status") != "verified" or not group_id or assigned_group != group_id:
            split_issues.append({"media_id": media_id, "code": "unverified_or_mismatched_group"})
        else:
            split_by_group[str(group_id)].add(split)
        fingerprint = item.get("content_sha256")
        if isinstance(fingerprint, str):
            split_by_hash[fingerprint].add(split)
    for group_id, assigned in split_by_group.items():
        if len(assigned) > 1:
            split_issues.append({"global_group_id": group_id, "code": "source_group_crosses_splits"})
    for fingerprint, assigned in split_by_hash.items():
        if len(assigned) > 1:
            split_issues.append({"content_sha256": fingerprint, "code": "exact_duplicate_crosses_splits"})
    split_verified = not missing_split_media and not split_issues and bool(media)
    tracks: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    temporal_media_ids = {media_id for media_id, item in media.items() if item.get("modality") == "video"}
    for observation in observations:
        media_id = observation.get("media_id")
        if media_id not in temporal_media_ids:
            continue
        person_id = observation.get("person_id")
        timestamp = observation.get("timestamp_ms")
        if not person_id or observation.get("identity_verified") is not True:
            issues.append({"observation_id": observation.get("observation_id"), "code": "unverified_person_identity"})
            continue
        if not isinstance(timestamp, int) or timestamp < 0:
            issues.append({"observation_id": observation.get("observation_id"), "code": "invalid_timestamp"})
            continue
        if observation.get("outside") is True:
            tracks[(media_id, str(person_id))].append(observation)
            continue
        if observation.get("mapping_action") != "map" or observation.get("canonical_label") not in CANONICAL_LABELS or observation.get("annotation_status") != "reviewed":
            issues.append({"observation_id": observation.get("observation_id"), "code": "state_label_not_reviewed"})
            continue
        tracks[(media_id, str(person_id))].append(observation)

    windows: list[dict[str, Any]] = []
    for (media_id, person_id), items in sorted(tracks.items()):
        items.sort(key=lambda item: item["timestamp_ms"])
        if len({item["timestamp_ms"] for item in items}) != len(items):
            issues.append({"media_id": media_id, "person_id": person_id, "code": "duplicate_person_timestamp"})
            continue
        segments: list[list[dict[str, Any]]] = []
        for item in items:
            if item.get("outside") is True:
                if segments and segments[-1]:
                    segments.append([])
                continue
            if not segments or not segments[-1] or item["timestamp_ms"] - segments[-1][-1]["timestamp_ms"] > max_track_gap_ms:
                segments.append([])
            segments[-1].append(item)
        segments = [segment for segment in segments if segment]
        source = media[media_id]
        for segment in segments:
            for anchor_index, anchor in enumerate(segment):
                anchor_time = anchor["timestamp_ms"]
                selected: list[dict[str, Any] | None] = []
                used: set[str] = set()
                for offset in range(sequence_length - 1, -1, -1):
                    target_time = anchor_time - offset * sample_period_ms
                    candidates = [item for item in segment[:anchor_index + 1]
                                  if item["timestamp_ms"] <= target_time and target_time - item["timestamp_ms"] <= sample_period_ms // 2]
                    candidate = candidates[-1] if candidates else None
                    if candidate and candidate["observation_id"] in used:
                        candidate = None
                    if candidate:
                        used.add(candidate["observation_id"])
                    selected.append(candidate)
                selected[-1] = anchor
                valid_count = sum(item is not None for item in selected)
                if valid_count / sequence_length < min_valid_ratio:
                    continue
                split_assignment = split_assignments.get(str(media_id), {})
                grouping_verified = (source.get("grouping_status") == "verified" and bool(source.get("global_group_id"))
                                     and split_assignment.get("global_group_id") == source.get("global_group_id"))
                windows.append({
                    "schema_version": 1,
                    "window_id": f"{media_id}:{person_id}:{anchor['observation_id']}:{sequence_length}:{sample_period_ms}",
                    "media_id": media_id,
                    "source_id": source.get("source_id"),
                    "source_version": source.get("source_version"),
                    "global_group_id": source.get("global_group_id"),
                    "grouping_status": source.get("grouping_status", "unknown"),
                    "person_id": person_id,
                    "identity_verified": True,
                    "anchor_observation_id": anchor["observation_id"],
                    "anchor_timestamp_ms": anchor_time,
                    "label": anchor["canonical_label"],
                    "observation_ids": [item["observation_id"] if item else None for item in selected],
                    "timestamps_ms": [item["timestamp_ms"] if item else None for item in selected],
                    "valid_mask": [item is not None for item in selected],
                    "sequence_length": sequence_length,
                    "sample_period_ms": sample_period_ms,
                    "causal": True,
                    "split": split_assignment.get("split"),
                    "training_eligible": grouping_verified and split_verified,
                })
    blockers = []
    if not temporal_media_ids:
        blockers.append("no_temporal_video_annotations")
    if not windows:
        blockers.append("no_windows_meet_verified_track_sampling_contract")
    if any(not window["training_eligible"] for window in windows):
        blockers.append("source_video_group_identity_unverified")
    if not split_verified:
        blockers.append("verified_group_split_manifest_required_or_invalid")
    return {
        "schema_version": 1,
        "status": "blocked" if blockers or issues else "ready_for_review",
        "temporal_media_count": len(temporal_media_ids),
        "verified_track_count": len(tracks),
        "window_count": len(windows),
        "windows": windows,
        "blockers": sorted(set(blockers)),
        "issues": issues,
        "source_video_disjointness_verified": split_verified,
        "training_started": False,
        "split_manifest_missing_media_count": len(missing_split_media),
        "split_manifest_issues": split_issues,
    }
