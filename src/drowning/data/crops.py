"""Deterministic person-crop previews and reviewed image training samples."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from drowning.data.audit import audit_jsonl
from drowning.data.adapters import _safe_file, _sha256_file
from drowning.data.registry import SourceConfigError, validate_source


def _protected_output(target: Path, raw_root: Path) -> None:
    target = target.resolve()
    try:
        target.relative_to(raw_root.resolve())
        inside_raw_root = True
    except ValueError:
        inside_raw_root = False
    if inside_raw_root:
        raise SourceConfigError("Derived crops must be outside the raw source root")
    try:
        target.relative_to(Path("D:/DrowningDemo").resolve())
        protected_demo = True
    except ValueError:
        protected_demo = False
    if protected_demo or any(part.casefold() == "sources" for part in target.parts):
        raise SourceConfigError("A protected D:\\DrowningDemo or sources/ path cannot be used for derived crops")
    if target.exists():
        raise SourceConfigError(f"Refusing to overwrite existing crop output: {target}")


def _crop_image(image_path: Path, bbox: list[float], context_fraction: float, size: int) -> Image.Image:
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    return _crop_pil(image, bbox, context_fraction, size)


def _crop_pil(image: Image.Image, bbox: list[float], context_fraction: float, size: int) -> Image.Image:
    if context_fraction < 0 or size < 16:
        raise ValueError("context_fraction must be nonnegative and crop size at least 16")
    if len(bbox) != 4 or any(not isinstance(value, (int, float)) for value in bbox):
        raise ValueError("bbox must contain four numeric xyxy values")
    width, height = image.size
    x1, y1, x2, y2 = map(float, bbox)
    if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError("bbox is invalid or outside the decoded image")
    pad_x, pad_y = (x2 - x1) * context_fraction, (y2 - y1) * context_fraction
    crop_box = (max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
                min(width, int(x2 + pad_x + 0.999)), min(height, int(y2 + pad_y + 0.999)))
    crop = image.crop(crop_box)
    return ImageOps.pad(crop, (size, size), method=Image.Resampling.LANCZOS, color=(114, 114, 114))


def build_crops(source: dict[str, Any], canonical_jsonl: Path, raw_root: Path, output_dir: Path,
                *, preview_only: bool = False, limit: int | None = None,
                split_manifest: Path | None = None,
                context_fraction: float = 0.15, size: int = 224) -> dict[str, Any]:
    errors = validate_source(source)
    if errors:
        raise SourceConfigError("Invalid source descriptor: " + "; ".join(errors))
    review = source.get("mapping_review", {})
    approved = isinstance(review, dict) and review.get("status") == "approved" and not review.get("sample_review_required", True)
    if not approved and not preview_only:
        raise SourceConfigError("Source mapping is not approved; use --preview-only for non-training review crops")
    if not preview_only and split_manifest is None:
        raise SourceConfigError("Training crops require a verified --split-manifest")
    if preview_only and (limit is None or limit <= 0):
        raise SourceConfigError("Preview crops require an explicit positive --limit")
    _protected_output(output_dir, raw_root)
    audit = audit_jsonl([canonical_jsonl])
    if not audit["valid"]:
        raise SourceConfigError("Canonical JSONL failed audit; resolve structural/data issues before cropping")
    media: dict[str, dict[str, Any]] = {}
    observations: list[dict[str, Any]] = []
    splits: dict[str, str] = {}
    split_groups: dict[str, str] = {}
    if split_manifest is not None:
        for line_number, line in enumerate(split_manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SourceConfigError(f"Invalid split manifest at line {line_number}") from exc
            media_id, split = item.get("media_id"), item.get("split")
            if not media_id or split not in {"train", "validation", "test"}:
                raise SourceConfigError(f"Split manifest line {line_number} lacks a valid media_id/split")
            if media_id in splits and splits[media_id] != split:
                raise SourceConfigError(f"Media {media_id} has conflicting split assignments")
            splits[media_id] = split
            if item.get("global_group_id"):
                split_groups[media_id] = str(item["global_group_id"])
    for line_number, line in enumerate(canonical_jsonl.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SourceConfigError(f"Invalid canonical JSONL at line {line_number}") from exc
        if record.get("entity") == "media":
            media[record["media_id"]] = record
        elif record.get("entity") == "observation" and record.get("mapping_action") in ({"map", "review"} if preview_only else {"map"}):
            observations.append(record)
    if not preview_only:
        missing_assignments = set(media) - set(splits)
        if missing_assignments:
            raise SourceConfigError(f"Split manifest is missing {len(missing_assignments)} media records")
        split_by_group: dict[str, set[str]] = defaultdict(set)
        split_by_hash: dict[str, set[str]] = defaultdict(set)
        for media_id, media_record in media.items():
            group_id = media_record.get("global_group_id")
            if (media_record.get("grouping_status") != "verified" or not group_id
                    or split_groups.get(media_id) != group_id):
                raise SourceConfigError(f"Media {media_id} has no matching verified global source group")
            split_by_group[group_id].add(splits[media_id])
            fingerprint = media_record.get("content_sha256")
            if isinstance(fingerprint, str):
                split_by_hash[fingerprint].add(splits[media_id])
        if any(len(values) > 1 for values in split_by_group.values()):
            raise SourceConfigError("Split manifest places one verified source group across multiple splits")
        if any(len(values) > 1 for values in split_by_hash.values()):
            raise SourceConfigError("Split manifest places exact duplicate media across multiple splits")
    conflict_hashes = {item["content_sha256"] for item in audit["exact_duplicate_content_groups"]
                       if any(conflict.get("content_sha256") == item["content_sha256"] for conflict in audit["same_box_label_conflicts"])}
    if preview_only:
        per_label: dict[str, list[dict[str, Any]]] = {}
        for record in observations:
            per_label.setdefault(record.get("canonical_label") or "review", []).append(record)
        for records in per_label.values():
            records.sort(key=lambda record: record["observation_id"])
        selected: list[dict[str, Any]] = []
        labels = sorted(per_label)
        while labels and len(selected) < limit:
            remaining = []
            for label in labels:
                if per_label[label]:
                    selected.append(per_label[label].pop(0))
                    if len(selected) >= limit:
                        break
                if per_label[label]:
                    remaining.append(label)
            labels = remaining
        observations = selected
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    manifest_path = temp_dir / "crop_manifest.jsonl"
    written = 0
    eligible = 0
    skipped: list[dict[str, str]] = []
    try:
        with manifest_path.open("x", encoding="utf-8", newline="\n") as manifest_stream:
            for observation in observations:
                media_record = media[observation["media_id"]]
                label = observation.get("canonical_label") or "review"
                assigned_split = splits.get(media_record["media_id"])
                canonical_group = media_record.get("global_group_id")
                group_verified = (media_record.get("grouping_status") == "verified" and bool(canonical_group)
                                  and split_groups.get(media_record["media_id"]) == canonical_group)
                review_required = (not approved or media_record.get("content_sha256") in conflict_hashes
                                   or assigned_split is None or not group_verified)
                training_eligible = (approved and not review_required
                                     and observation.get("annotation_status") == "reviewed")
                relative = Path(media_record["raw_relative_path"])
                image_path = _safe_file(raw_root, relative)
                crop_relative = Path("preview" if preview_only else "samples") / label / f"{observation['observation_id']}.jpg"
                try:
                    crop = _crop_image(image_path, observation["bbox"], context_fraction, size)
                except (OSError, ValueError) as exc:
                    skipped.append({"observation_id": observation.get("observation_id", ""), "reason": str(exc)})
                    continue
                target = temp_dir / crop_relative
                target.parent.mkdir(parents=True, exist_ok=True)
                crop.save(target, format="JPEG", quality=92, optimize=True)
                manifest_record = {
                    "schema_version": 1,
                    "crop_id": observation["observation_id"],
                    "media_id": media_record["media_id"],
                    "observation_id": observation["observation_id"],
                    "source_id": media_record["source_id"],
                    "source_version": media_record["source_version"],
                    "source_relative_path": media_record["raw_relative_path"],
                    "source_content_sha256": media_record["content_sha256"],
                    "original_export_split": media_record.get("original_export_split"),
                    "split": assigned_split if assigned_split else media_record.get("original_export_split"),
                    "crop_relative_path": crop_relative.as_posix(),
                    "crop_sha256": _sha256_file(target),
                    "bbox_xyxy": observation["bbox"],
                    "context_fraction": context_fraction,
                    "output_size": [size, size],
                    "canonical_label": label,
                    "training_eligible": training_eligible,
                    "review_required": review_required,
                }
                manifest_stream.write(json.dumps(manifest_record, ensure_ascii=False, sort_keys=True) + "\n")
                written += 1
                eligible += int(training_eligible)
        (temp_dir / "summary.json").write_text(json.dumps({
            "source_id": source["source_id"],
            "preview_only": preview_only,
            "mapping_approved": approved,
            "crop_count": written,
            "training_eligible_count": eligible,
            "split_assigned_count": sum(1 for obs in observations if obs.get("media_id") in splits),
            "skipped_count": len(skipped),
            "skipped_samples": skipped[:50],
            "raw_source_modified": False,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_dir.rename(output_dir.resolve())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {
        "output_dir": str(output_dir.resolve()), "crop_count": written,
        "training_eligible_count": eligible, "skipped_count": len(skipped),
        "split_assigned_count": sum(1 for obs in observations if obs.get("media_id") in splits),
        "preview_only": preview_only, "mapping_approved": approved,
    }


def select_video_frames(observations: list[dict[str, Any]], *, stride_ms: int, max_per_video: int) -> list[dict[str, Any]]:
    """Choose which annotated video frames become still-image samples.

    Adjacent frames of one person are nearly identical, so a sample is kept per person track every ``stride_ms``.
    The first frame and every frame where the person's label changes are always kept (state changes are the rare,
    informative moments). A video above ``max_per_video`` is thinned evenly, change frames first, so one long video
    cannot dominate training. Frames never leave their video, so the group split still keeps them in one split.
    """
    if stride_ms <= 0 or max_per_video <= 0:
        raise ValueError("stride_ms and max_per_video must be positive")
    tracks: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        if isinstance(observation.get("frame_index"), int) and not isinstance(observation.get("frame_index"), bool):
            tracks[(observation["media_id"], str(observation.get("person_id")))].append(observation)

    def clock(observation: dict[str, Any]) -> float:
        value = observation.get("timestamp_ms")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else observation["frame_index"] * 40.0

    kept_by_media: dict[str, list[tuple[bool, dict[str, Any]]]] = defaultdict(list)
    for (media_id, _person), items in sorted(tracks.items()):
        items.sort(key=lambda item: (item["frame_index"], item.get("observation_id", "")))
        last_label, last_time = None, None
        for item in items:
            changed = last_label is not None and item.get("canonical_label") != last_label
            if last_time is None or changed or clock(item) - last_time >= stride_ms:
                kept_by_media[media_id].append((changed, item))
                last_time = clock(item)
            last_label = item.get("canonical_label")
    selected: list[dict[str, Any]] = []
    for media_id in sorted(kept_by_media):
        entries = kept_by_media[media_id]
        if len(entries) > max_per_video:
            changes = [entry for entry in entries if entry[0]][:max_per_video]
            rest = [entry for entry in entries if not entry[0]]
            room = max_per_video - len(changes)
            step = len(rest) / room if room else 0
            picked = [rest[int(i * step)] for i in range(room)] if room else []
            entries = sorted(changes + picked, key=lambda entry: (entry[1]["frame_index"], entry[1].get("observation_id", "")))
        selected.extend(item for _changed, item in entries)
    return selected


def build_video_frame_crops(source: dict[str, Any], canonical_jsonl: Path, raw_root: Path, output_dir: Path,
                            *, split_manifest: Path, stride_ms: int = 500, max_per_video: int = 300,
                            context_fraction: float = 0.15, size: int = 224) -> dict[str, Any]:
    """Cut reviewed video annotations into still person crops, with the same eligibility rules as image crops."""
    import cv2

    errors = validate_source(source)
    if errors:
        raise SourceConfigError("Invalid source descriptor: " + "; ".join(errors))
    review = source.get("mapping_review", {})
    if not (isinstance(review, dict) and review.get("status") == "approved" and not review.get("sample_review_required", True)):
        raise SourceConfigError("Source mapping is not approved; video frames cannot become training samples")
    _protected_output(output_dir, raw_root)
    splits: dict[str, str] = {}
    split_groups: dict[str, str] = {}
    for line in split_manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            splits[item["media_id"]] = item["split"]
            if item.get("global_group_id"):
                split_groups[item["media_id"]] = str(item["global_group_id"])
    media: dict[str, dict[str, Any]] = {}
    observations: list[dict[str, Any]] = []
    for line in canonical_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("entity") == "media" and record.get("modality") == "video":
            media[record["media_id"]] = record
        elif record.get("entity") == "observation" and record.get("mapping_action") == "map":
            observations.append(record)
    for media_id, media_record in media.items():
        group = media_record.get("global_group_id")
        if (media_id not in splits or media_record.get("grouping_status") != "verified" or not group
                or split_groups.get(media_id) != group):
            raise SourceConfigError(f"Video {media_id} has no matching verified global source group")
    chosen = select_video_frames([o for o in observations if o.get("media_id") in media and o.get("annotation_status") == "reviewed"
                                  and o.get("canonical_label")], stride_ms=stride_ms, max_per_video=max_per_video)
    by_media: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in chosen:
        by_media[observation["media_id"]].append(observation)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    written, skipped = 0, []
    try:
        with (temp_dir / "crop_manifest.jsonl").open("x", encoding="utf-8", newline="\n") as manifest_stream:
            for media_id in sorted(by_media):
                media_record = media[media_id]
                wanted = {o["frame_index"]: [] for o in by_media[media_id]}
                for observation in by_media[media_id]:
                    wanted[observation["frame_index"]].append(observation)
                capture = cv2.VideoCapture(str(_safe_file(raw_root, Path(media_record["raw_relative_path"]))))
                if not capture.isOpened():
                    capture.release()
                    skipped.append({"observation_id": media_id, "reason": "video could not be opened"})
                    continue
                try:
                    last_needed = max(wanted)
                    for frame_index in range(last_needed + 1):
                        if frame_index not in wanted:
                            if not capture.grab():
                                break
                            continue
                        ok, bgr = capture.read()
                        if not ok:
                            break
                        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                        for observation in wanted[frame_index]:
                            x1, y1, x2, y2 = (float(v) for v in observation["bbox"])
                            box = [max(0.0, x1), max(0.0, y1), min(float(image.width), x2), min(float(image.height), y2)]
                            try:
                                crop = _crop_pil(image, box, context_fraction, size)
                            except (ValueError, TypeError) as exc:
                                skipped.append({"observation_id": observation.get("observation_id", ""), "reason": str(exc)})
                                continue
                            label = observation["canonical_label"]
                            relative = Path("frames") / label / f"{observation['observation_id']}.jpg"
                            target = temp_dir / relative
                            target.parent.mkdir(parents=True, exist_ok=True)
                            crop.save(target, format="JPEG", quality=92, optimize=True)
                            manifest_stream.write(json.dumps({
                                "schema_version": 1, "crop_id": observation["observation_id"], "media_id": media_id,
                                "observation_id": observation["observation_id"], "source_id": media_record["source_id"],
                                "source_version": media_record["source_version"], "origin": "video_frame",
                                "frame_index": frame_index, "timestamp_ms": observation.get("timestamp_ms"),
                                "person_id": observation.get("person_id"), "split": splits[media_id],
                                "crop_relative_path": relative.as_posix(), "crop_sha256": _sha256_file(target),
                                "bbox_xyxy": box, "context_fraction": context_fraction, "output_size": [size, size],
                                "canonical_label": label, "training_eligible": True, "review_required": False,
                            }, ensure_ascii=False, sort_keys=True) + "\n")
                            written += 1
                finally:
                    capture.release()
        (temp_dir / "summary.json").write_text(json.dumps({
            "source_id": source["source_id"], "origin": "video_frame", "stride_ms": stride_ms, "max_per_video": max_per_video,
            "videos": len(by_media), "crop_count": written, "skipped_count": len(skipped), "skipped_samples": skipped[:50],
            "raw_source_modified": False}, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_dir.rename(output_dir.resolve())
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {"output_dir": str(output_dir.resolve()), "crop_count": written, "videos": len(by_media), "skipped_count": len(skipped)}
