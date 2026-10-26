"""Deterministic source-group split planning that refuses unknown identities."""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Iterable


class SplitBlocked(ValueError):
    pass


class _DisjointSet:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            low, high = sorted((a, b))
            self.parent[high] = low


def build_group_split(media_records: list[dict[str, Any]], *, seed: int = 20261026,
                      holdout_group_ids: set[str] | None = None,
                      train_ratio: float = 0.70, validation_ratio: float = 0.15,
                      test_ratio: float = 0.15) -> dict[str, Any]:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0 or ratio >= 1 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("train/validation/test ratios must be positive and sum to 1")
    if not media_records:
        raise SplitBlocked("no media records were supplied")
    group_for_media: dict[str, str] = {}
    unknown: list[str] = []
    for media in media_records:
        media_id, group_id = media.get("media_id"), media.get("global_group_id")
        if not media_id:
            raise SplitBlocked("media record is missing media_id")
        if media.get("grouping_status") != "verified" or not group_id:
            unknown.append(str(media_id))
        else:
            group_for_media[str(media_id)] = str(group_id)
    if unknown:
        raise SplitBlocked(f"source-video/session grouping is unverified for {len(unknown)} media records")
    group_ids = set(group_for_media.values())
    sets = _DisjointSet(group_ids)
    hash_groups: dict[str, list[str]] = {}
    for media in media_records:
        fingerprint = media.get("content_sha256")
        if isinstance(fingerprint, str) and len(fingerprint) == 64:
            hash_groups.setdefault(fingerprint, []).append(group_for_media[str(media["media_id"])])
    for linked_groups in hash_groups.values():
        for group_id in linked_groups[1:]:
            sets.union(linked_groups[0], group_id)
    cluster_for_group = {group_id: sets.find(group_id) for group_id in group_ids}
    requested_holdout = holdout_group_ids or set()
    missing_holdout = requested_holdout - group_ids
    if missing_holdout:
        raise SplitBlocked("requested holdout group not present: " + ", ".join(sorted(missing_holdout)))
    holdout_clusters = {cluster_for_group[group_id] for group_id in requested_holdout}
    group_clusters = sorted(set(cluster_for_group.values()))
    if len(group_clusters) < 3:
        raise SplitBlocked("at least three verified split groups are required")
    assignments: dict[str, str] = {}
    for cluster in holdout_clusters:
        assignments[cluster] = "test"
    remaining = [cluster for cluster in group_clusters if cluster not in holdout_clusters]
    if len(remaining) < 2:
        raise SplitBlocked("at least two non-holdout groups are required for train and validation")
    remaining.sort(key=lambda cluster: hashlib.sha256(f"{seed}:{cluster}".encode("utf-8")).digest())
    if holdout_clusters:
        validation_count = max(1, round(len(remaining) * validation_ratio / (train_ratio + validation_ratio)))
        validation_count = min(validation_count, len(remaining) - 1)
        for index, cluster in enumerate(remaining):
            assignments[cluster] = "validation" if index < validation_count else "train"
    else:
        test_count = min(max(1, round(len(remaining) * test_ratio)), len(remaining) - 2)
        validation_count = min(max(1, round(len(remaining) * validation_ratio)), len(remaining) - test_count - 1)
        for index, cluster in enumerate(remaining):
            assignments[cluster] = "test" if index < test_count else "validation" if index < test_count + validation_count else "train"
    manifest = []
    for media in sorted(media_records, key=lambda item: str(item["media_id"])):
        group_id = group_for_media[str(media["media_id"])]
        cluster = cluster_for_group[group_id]
        manifest.append({
            "schema_version": 1,
            "media_id": media["media_id"],
            "global_group_id": group_id,
            "split_group_id": cluster,
            "split": assignments[cluster],
            "split_reason": "user_holdout" if cluster in holdout_clusters else "stable_group_hash",
            "seed": seed,
        })
    counts = Counter(item["split"] for item in manifest)
    return {
        "schema_version": 1,
        "status": "ready_for_review",
        "seed": seed,
        "requested_ratios": {"train": train_ratio, "validation": validation_ratio, "test": test_ratio},
        "verified_group_count": len(group_clusters),
        "media_count": len(manifest),
        "media_counts_by_split": dict(sorted(counts.items())),
        "source_video_disjointness_verified": True,
        "manifest": manifest,
    }
