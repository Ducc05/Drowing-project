"""Quality report for a converted source: label balance, box sizes, duplicates and split leakage.

Near-duplicate detection (difference hash) produces *proposed* groups. A proposal is evidence, not
proof that groups are source-disjoint; it becomes a verified grouping only through
``record_grouping_review`` by a named reviewer.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from drowning.data.adapters import _safe_file
from drowning.data.review import is_mapping_approved, load_grouping_review, mapping_fingerprint
from drowning.data.samples import read_canonical

TINY_SIDE_PX = 16
TINY_AREA_FRACTION = 0.0005


def dhash(path: Path) -> int:
    """64-bit difference hash of the luma channel (8 rows of 8 horizontal gradients)."""
    with Image.open(path) as image:
        small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    gradients = (pixels[:, :-1] > pixels[:, 1:]).ravel()
    return int("".join("1" if bit else "0" for bit in gradients), 2)


def cluster_near_duplicates(hashes: dict[str, int], max_distance: int) -> list[list[str]]:
    """Group keys whose hashes differ in at most ``max_distance`` bits (single-linkage).

    Pigeonhole: two hashes within d bits agree exactly on at least one of d+1 chunks, so candidate
    pairs are found by bucketing on each chunk and then verified with the exact distance.
    """
    if max_distance < 0:
        raise ValueError("max_distance must be nonnegative")
    chunks = max_distance + 1
    bounds = [round(i * 64 / chunks) for i in range(chunks + 1)]
    parent = {key: key for key in hashes}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for index in range(chunks):
        width = bounds[index + 1] - bounds[index]
        buckets: dict[int, list[str]] = defaultdict(list)
        for key, value in hashes.items():
            buckets[(value >> bounds[index]) & ((1 << width) - 1)].append(key)
        for members in buckets.values():
            for i, left in enumerate(members):
                for right in members[i + 1:]:
                    if bin(hashes[left] ^ hashes[right]).count("1") <= max_distance:
                        a, b = find(left), find(right)
                        if a != b:
                            parent[max(a, b)] = min(a, b)
    groups: dict[str, list[str]] = defaultdict(list)
    for key in hashes:
        groups[find(key)].append(key)
    return [sorted(members) for members in groups.values()]


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def at(q: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 5)

    return {"min": round(ordered[0], 5), "p05": at(0.05), "p50": at(0.5), "p95": at(0.95), "max": round(ordered[-1], 5)}


def build_quality_report(source: dict[str, Any], canonical_jsonl: Path, raw_root: Path | None = None, *,
                         near_duplicate_distance: int = 2, workers: int = 8,
                         reviews_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    """Return ``(report, proposed_groups_by_path)``; ``raw_root=None`` skips pixel-based checks."""
    media, observations = read_canonical(canonical_jsonl)
    mapped = [o for o in observations if o.get("mapping_action") == "map" and o.get("canonical_label")]
    original_split = {mid: (m.get("original_export_split") or "unknown") for mid, m in media.items()}

    by_label: dict[str, dict[str, Any]] = {}
    for observation in observations:
        entry = by_label.setdefault(str(observation["original_label"]), {
            "original_label": observation["original_label"], "original_label_id": observation.get("original_label_id"),
            "mapping_action": observation.get("mapping_action"), "canonical_label": observation.get("canonical_label"),
            "count": 0, "by_original_split": Counter(), "area_fraction": [], "min_side_px": [], "tiny": 0})
        entry["count"] += 1
        entry["by_original_split"][original_split.get(observation["media_id"], "unknown")] += 1
        box, record = observation.get("bbox"), media.get(observation["media_id"])
        if isinstance(box, list) and record:
            width, height = box[2] - box[0], box[3] - box[1]
            fraction = width * height / (record["width"] * record["height"])
            entry["area_fraction"].append(fraction)
            entry["min_side_px"].append(min(width, height))
            entry["tiny"] += min(width, height) < TINY_SIDE_PX or fraction < TINY_AREA_FRACTION
    labels = []
    for entry in sorted(by_label.values(), key=lambda e: str(e["original_label"])):
        labels.append({**{k: entry[k] for k in ("original_label", "original_label_id", "mapping_action", "canonical_label", "count")},
                       "by_original_split": dict(sorted(entry["by_original_split"].items())),
                       "box_area_fraction": _percentiles(entry["area_fraction"]),
                       "box_min_side_px": _percentiles(entry["min_side_px"]),
                       "tiny_box_count": entry["tiny"],
                       "tiny_box_fraction": round(entry["tiny"] / entry["count"], 4) if entry["count"] else 0})

    boxes_per_media = Counter(o["media_id"] for o in observations if isinstance(o.get("bbox"), list))
    classes_per_media: dict[str, set[str]] = defaultdict(set)
    for observation in mapped:
        classes_per_media[observation["media_id"]].add(observation["canonical_label"])
    canonical_by_split: dict[str, Counter] = defaultdict(Counter)
    for observation in mapped:
        canonical_by_split[original_split.get(observation["media_id"], "unknown")][observation["canonical_label"]] += 1

    by_hash: dict[str, list[str]] = defaultdict(list)
    for media_id, record in media.items():
        by_hash[record["content_sha256"]].append(media_id)
    exact_groups = [ids for ids in by_hash.values() if len(ids) > 1]
    exact_cross_split = sum(len({original_split[i] for i in ids} - {"unknown"}) > 1 for ids in exact_groups)

    report: dict[str, Any] = {
        "schema_version": 1, "source_id": source["source_id"], "source_version": str(source["source_version"]),
        "mapping_hash": mapping_fingerprint(source),
        "counts": {"media": len(media), "observations": len(observations), "mapped_observations": len(mapped),
                   "review_or_ignored_observations": len(observations) - len(mapped)},
        "labels": labels,
        "canonical_by_original_split": {s: dict(sorted(c.items())) for s, c in sorted(canonical_by_split.items())},
        "boxes_per_image": _percentiles([float(c) for c in boxes_per_media.values()]),
        "images_with_multiple_states": sum(len(c) > 1 for c in classes_per_media.values()),
        "images_with_more_than_10_boxes": sum(c > 10 for c in boxes_per_media.values()),
        "image_size_px": {"distinct": len({(m["width"], m["height"]) for m in media.values()}),
                          "most_common": [[f"{w}x{h}", n] for (w, h), n in
                                          Counter((m["width"], m["height"]) for m in media.values()).most_common(5)]},
        "exact_duplicates": {"groups": len(exact_groups), "extra_images": sum(len(g) - 1 for g in exact_groups),
                             "groups_crossing_original_splits": exact_cross_split},
        "near_duplicates": {"checked": False, "reason": "raw_root not supplied"},
        "limitations": ["Box and class statistics describe annotations as exported; they are not accuracy measures.",
                        "Near-duplicate clusters are proposals; they do not prove source-video disjointness."],
    }
    proposed: dict[str, str] = {}
    if raw_root is not None and any(m.get("modality", "image") == "image" for m in media.values()):
        paths = {mid: _safe_file(raw_root, m["raw_relative_path"]) for mid, m in media.items()
                 if m.get("modality", "image") == "image"}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dhash") as pool:
            hashes = dict(zip(paths, pool.map(dhash, paths.values())))
        clusters = cluster_near_duplicates(hashes, near_duplicate_distance)
        multi = [c for c in clusters if len(c) > 1]
        cross = [c for c in multi if len({original_split[i] for i in c} - {"unknown"}) > 1]
        for cluster in clusters:
            group_id = "ndg-" + hashlib.sha1("|".join(media[i]["raw_relative_path"] for i in cluster).encode()).hexdigest()[:12]
            for media_id in cluster:
                proposed[media[media_id]["raw_relative_path"]] = group_id
        sizes = sorted((len(c) for c in multi), reverse=True)
        report["near_duplicates"] = {
            "checked": True, "method": "dhash64_single_linkage", "max_hamming_distance": near_duplicate_distance,
            "clusters_with_multiple_images": len(multi), "images_in_such_clusters": sum(sizes),
            "largest_cluster": sizes[0] if sizes else 1,
            "oversized_clusters_over_1pct_of_images": sum(s > max(5, len(media) * 0.01) for s in sizes),
            "clusters_crossing_original_splits": len(cross),
            "images_in_cross_split_clusters": sum(len(c) for c in cross),
            "cross_split_examples": [[{"path": media[i]["raw_relative_path"], "split": original_split[i]} for i in c[:4]]
                                     for c in sorted(cross, key=len, reverse=True)[:8]],
            "proposed_group_count": len(set(proposed.values())),
        }

    for media_id, record in media.items():
        if record.get("modality") == "video":
            # One video is one proposed group; two videos from the same session must be merged by hand.
            proposed[record["raw_relative_path"]] = "vid-" + hashlib.sha1(record["raw_relative_path"].encode()).hexdigest()[:12]
    if any(m.get("modality") == "video" for m in media.values()):
        report["video_groups"] = {"proposed_group_count": sum(m.get("modality") == "video" for m in media.values()),
                                  "method": "one_group_per_video_file",
                                  "caveat": "Videos recorded in the same session or scene are not independent; merge them in a manual group file."}
    mapping_ok = is_mapping_approved(source)
    grouping_ok = (load_grouping_review(source, reviews_dir) is not None
                   or (bool(media) and all(m.get("grouping_status") == "verified" and m.get("global_group_id")
                                            for m in media.values())))
    present = {o["canonical_label"] for o in mapped}
    blockers = []
    if not mapping_ok:
        blockers.append("mapping_review_not_approved")
    if not grouping_ok:
        blockers.append("source_groups_not_verified")
    if present != {"drowning", "swimming", "out_of_water"}:
        blockers.append("source_alone_does_not_cover_all_three_classes_(ok_if_other_sources_cover_the_rest)")
    report["image_training_readiness"] = {
        "mapping_review_approved": mapping_ok, "source_groups_verified": grouping_ok,
        "classes_present_in_source": sorted(present), "blockers": blockers,
        "ready_for_split_and_crop_building": mapping_ok and grouping_ok}
    return report, proposed


def dominant_class_ratio(report: dict[str, Any]) -> float | None:
    counts = [label["count"] for label in report["labels"] if label["mapping_action"] == "map"]
    return round(max(counts) / sum(counts), 4) if counts and sum(counts) else None


def label_count_summary(report: dict[str, Any]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for label in report["labels"]:
        if label["mapping_action"] == "map":
            totals[label["canonical_label"]] += label["count"]
    return dict(totals)

