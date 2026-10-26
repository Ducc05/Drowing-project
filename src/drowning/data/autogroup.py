"""Automatic source groups for sources nobody has confirmed.

When review is advisory a source can be used before anyone confirms its groups. The system then builds the
same near-duplicate groups it would have proposed (images) or puts each video in its own group, so a split
still never separates near-identical images. These groups carry ``grouping_assurance`` so reports can say they
were not confirmed, and they never claim that the groups come from different source videos.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from drowning.data.adapters import _safe_file
from drowning.data.quality import cluster_near_duplicates, dhash

ASSURANCE_IMAGES = "auto_near_duplicate_unconfirmed"
ASSURANCE_VIDEOS = "auto_one_group_per_video_unconfirmed"


def assign_automatic_groups(canonical_jsonl: Path, raw_root: Path, *, distance: int = 2, workers: int = 8) -> dict[str, Any]:
    """Rewrite media records that have no verified group; records that already have one are left alone."""
    lines = canonical_jsonl.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    pending = [r for r in records if r.get("entity") == "media" and not (r.get("grouping_status") == "verified" and r.get("global_group_id"))]
    images = [r for r in pending if r.get("modality", "image") == "image"]
    videos = [r for r in pending if r.get("modality") == "video"]
    group_of: dict[str, tuple[str, str]] = {}
    clusters = 0
    if images:
        paths = {r["media_id"]: _safe_file(raw_root, r["raw_relative_path"]) for r in images}
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="autogroup") as pool:
            hashes = dict(zip(paths, pool.map(dhash, paths.values())))
        by_id = {r["media_id"]: r for r in images}
        for cluster in cluster_near_duplicates(hashes, distance):
            key = "|".join(sorted(by_id[m]["raw_relative_path"] for m in cluster))
            gid = "auto-" + hashlib.sha1(key.encode()).hexdigest()[:12]
            clusters += len(cluster) > 1
            for media_id in cluster:
                group_of[media_id] = (gid, ASSURANCE_IMAGES)
    for record in videos:
        group_of[record["media_id"]] = ("autovid-" + hashlib.sha1(record["raw_relative_path"].encode()).hexdigest()[:12], ASSURANCE_VIDEOS)
    for record in records:
        if record.get("entity") == "media" and record["media_id"] in group_of:
            record["global_group_id"], record["grouping_assurance"] = group_of[record["media_id"]]
            record["grouping_status"] = "verified"
    temp = canonical_jsonl.with_name(canonical_jsonl.name + f".{os.getpid()}.tmp")
    temp.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records), encoding="utf-8", newline="\n")
    temp.replace(canonical_jsonl)
    return {"images_grouped": len(images), "multi_image_clusters": clusters, "videos_grouped": len(videos),
            "assurance": {"images": ASSURANCE_IMAGES if images else None, "videos": ASSURANCE_VIDEOS if videos else None},
            "source_video_disjointness_verified": False}
