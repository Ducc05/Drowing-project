"""Conservative image annotation adapters for YOLO detection and COCO."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit
from defusedxml import ElementTree as ET

from PIL import Image, ImageOps, UnidentifiedImageError

from drowning.data.labels import resolve_label
from drowning.data.review import mapping_fingerprint
from drowning.data.registry import SourceConfigError, read_class_names, validate_source

_ID_NAMESPACE = uuid.UUID("a49d48b3-cc2b-4486-bf97-adf4f44e6eb5")
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _stable_id(*parts: object) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, "\x1f".join(map(str, parts))))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(root: Path, relative: str | Path, *, resolved_root: Path | None = None) -> Path:
    root_resolved = resolved_root if resolved_root is not None else root.resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise SourceConfigError(f"Source media path escapes root: {relative}") from exc
    if not target.is_file():
        raise SourceConfigError(f"Source media file not found: {relative}")
    return target


def _image_info(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).size
    except (OSError, UnidentifiedImageError) as exc:
        raise SourceConfigError(f"Invalid or unreadable image: {path.name}") from exc


_mapping_fingerprint = mapping_fingerprint


def _safe_provenance(source: dict[str, Any]) -> dict[str, Any]:
    provenance = source.get("provenance", {})
    safe: dict[str, Any] = {}
    if isinstance(provenance, dict):
        for key in ("url", "license_declared", "license_verified_for_all_media"):
            if key not in provenance:
                continue
            value = provenance[key]
            if key == "url" and isinstance(value, str):
                parsed = urlsplit(value)
                if parsed.scheme and parsed.netloc:
                    host = parsed.hostname or ""
                    if parsed.port:
                        host = f"{host}:{parsed.port}"
                    value = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
            safe[key] = value
    return safe


def _load_grouping_map(source: dict[str, Any], root: Path) -> dict[str, str]:
    reviewed = source.get("_reviewed_groups")
    if isinstance(reviewed, dict) and reviewed:
        # Reviewer-verified assignment from reviews/; keyed by raw relative path.
        return {Path(key).as_posix(): value for key, value in reviewed.items()
                if isinstance(value, str) and value.strip()}
    grouping = source.get("grouping", {})
    if not isinstance(grouping, dict) or not grouping.get("metadata_file"):
        return {}
    metadata_path = _safe_file(root, grouping["metadata_file"])
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SourceConfigError("Invalid source grouping metadata JSON") from exc
    entries = data.get("by_path", data) if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        raise SourceConfigError("Grouping metadata must map raw relative paths to global group IDs")
    result: dict[str, str] = {}
    for relative, value in entries.items():
        if isinstance(value, dict):
            group_id = value.get("global_group_id")
            status = value.get("status", grouping.get("status", "unknown"))
        else:
            group_id, status = value, grouping.get("status", "unknown")
        if isinstance(group_id, str) and group_id.strip() and status == "verified":
            result[Path(relative).as_posix()] = group_id
    return result


def _media_record(source: dict[str, Any], root: Path, image_path: Path, relative: str,
                  original_export_split: str | None = None,
                  grouping_map: dict[str, str] | None = None) -> tuple[dict[str, Any], str, int, int]:
    width, height = _image_info(image_path)
    content_hash = _sha256_file(image_path)
    media_id = _stable_id(source["source_id"], source["source_version"], relative, content_hash)
    global_group_id = (grouping_map or {}).get(Path(relative).as_posix())
    return ({
        "entity": "media",
        "schema_version": 1,
        "media_id": media_id,
        "source_id": source["source_id"],
        "source_version": str(source["source_version"]),
        "raw_relative_path": relative.replace("\\", "/"),
        "content_sha256": content_hash,
        "modality": "image",
        "width": width,
        "height": height,
        "original_export_split": original_export_split,
        "global_group_id": global_group_id,
        "grouping_status": "verified" if global_group_id else "unknown",
        "domain": source.get("domain"),
        "viewpoint": source.get("viewpoint"),
        "overlay_present": None,
        "provenance": _safe_provenance(source),
    }, media_id, width, height)


def _observation(
    source: dict[str, Any], media_id: str, annotation_key: object, frame_index: int | None,
    source_class_id: object, source_label: str, bbox: list[float] | None, mapping_hash: str,
    *, iscrowd: bool = False, is_ignore: bool = False,
    annotation_flags: dict[str, Any] | None = None,
    timestamp_ms: int | None = None, person_id: str | None = None,
    identity_verified: bool = False, occluded: bool | None = None,
    outside: bool | None = None,
) -> dict[str, Any]:
    decision = resolve_label(source_label, source.get("mapping", {}))
    mapping_review = source.get("mapping_review", {})
    mapping_approved = isinstance(mapping_review, dict) and mapping_review.get("status") == "approved" and not mapping_review.get("sample_review_required", True)
    annotation_status = "reviewed" if decision.action == "map" and mapping_approved else "source_unreviewed" if decision.action == "map" else decision.action
    if iscrowd or is_ignore:
        annotation_status = "ignored_annotation"
    return {
        "entity": "observation",
        "schema_version": 1,
        "observation_id": _stable_id(media_id, "annotation", annotation_key),
        "media_id": media_id,
        "source_id": source["source_id"],
        "source_version": str(source["source_version"]),
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "person_id": person_id,
        "identity_verified": identity_verified,
        "bbox_format": "xyxy_pixels",
        "bbox": bbox,
        "original_label_id": source_class_id,
        "original_label": source_label,
        "canonical_label": decision.canonical_label,
        "mapping_action": decision.action,
        "mapping_reason": decision.reason,
        "mapping_version": mapping_hash,
        "annotation_status": annotation_status,
        "occluded": occluded,
        "outside": outside,
        "iscrowd": bool(iscrowd),
        "ignore": bool(is_ignore),
        "annotation_flags": annotation_flags or {},
        "global_group_id": None,
        "grouping_status": "unknown",
        "split": None,
    }


def yolo_annotation_files(root: Path) -> Iterator[tuple[Path, Path, str]]:
    """Pair YOLO label files to sibling images only when path is unambiguous."""
    root_resolved = root.resolve()
    label_files = sorted((path for path in root.rglob("*.txt") if path.parent.name.casefold() == "labels"), key=lambda path: path.relative_to(root).as_posix())
    if not label_files:
        raise SourceConfigError("No YOLO label files found under a directory named 'labels'")
    image_indexes: dict[Path, dict[str, list[Path]]] = {}
    for label_file in label_files:
        _safe_file(root, label_file.relative_to(root), resolved_root=root_resolved)
        image_dir = label_file.parent.parent / "images"
        if image_dir not in image_indexes:
            index: dict[str, list[Path]] = {}
            if image_dir.is_dir():
                with os.scandir(image_dir) as entries:
                    for entry in entries:
                        candidate = Path(entry.path)
                        if candidate.suffix.casefold() in _IMAGE_EXTENSIONS and entry.is_file():
                            index.setdefault(os.path.normcase(candidate.stem), []).append(candidate)
            image_indexes[image_dir] = index
        matches = image_indexes[image_dir].get(os.path.normcase(label_file.stem), [])
        if len(matches) != 1:
            raise SourceConfigError(f"Expected exactly one sibling image for label file {label_file.relative_to(root)}; found {len(matches)}")
        relative_image = Path(os.path.relpath(matches[0], root)).as_posix()
        image_path = _safe_file(root, relative_image, resolved_root=root_resolved)
        yield label_file, image_path, relative_image


def convert_yolo_records(source: dict[str, Any], root: Path) -> Iterator[dict[str, Any]]:
    errors = validate_source(source)
    if errors:
        raise SourceConfigError("Invalid source descriptor: " + "; ".join(errors))
    if source.get("format") != "yolo_detection":
        raise SourceConfigError("YOLO converter requires format=yolo_detection")
    classes = read_class_names(source, root)
    class_names: dict[int, str] = {}
    for class_id, name in classes:
        try:
            numeric_id = int(class_id)
        except (ValueError, TypeError) as exc:
            raise SourceConfigError(f"YOLO class ID must be an integer: {class_id!r}") from exc
        if numeric_id in class_names:
            raise SourceConfigError(f"Duplicate YOLO class ID: {numeric_id}")
        class_names[numeric_id] = name
    mapping_hash = _mapping_fingerprint(source)
    grouping_map = _load_grouping_map(source, root)
    for label_file, image_path, relative_image in yolo_annotation_files(root):
        path_parts = Path(relative_image).parts
        export_split = next((part for part in path_parts if part.casefold() in {"train", "val", "valid", "validation", "test"}), None)
        if export_split == "val":
            export_split = "valid"
        if export_split == "validation":
            export_split = "valid"
        media, media_id, width, height = _media_record(source, root, image_path, relative_image, export_split, grouping_map)
        yield media
        for line_number, raw_line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise SourceConfigError(f"{label_file.name}:{line_number}: expected 5 YOLO detection fields; segmentation/pose is unsupported")
            try:
                class_id = int(fields[0])
                cx, cy, box_w, box_h = map(float, fields[1:])
            except ValueError as exc:
                raise SourceConfigError(f"{label_file.name}:{line_number}: invalid numeric YOLO annotation") from exc
            if class_id not in class_names:
                raise SourceConfigError(f"{label_file.name}:{line_number}: class id {class_id} is absent from the declared class list")
            if not all(math.isfinite(value) for value in (cx, cy, box_w, box_h)):
                raise SourceConfigError(f"{label_file.name}:{line_number}: bbox contains a non-finite number")
            if box_w <= 0 or box_h <= 0 or cx < 0 or cy < 0 or cx > 1 or cy > 1 or box_w > 1 or box_h > 1:
                raise SourceConfigError(f"{label_file.name}:{line_number}: normalized bbox is outside allowed range")
            x1, y1 = (cx - box_w / 2) * width, (cy - box_h / 2) * height
            x2, y2 = (cx + box_w / 2) * width, (cy + box_h / 2) * height
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                raise SourceConfigError(f"{label_file.name}:{line_number}: bbox extends outside the image")
            yield _observation(source, media_id, f"{label_file.relative_to(root).as_posix()}:{line_number}", None,
                               class_id, class_names[class_id], [x1, y1, x2, y2], mapping_hash)


def convert_coco_records(source: dict[str, Any], root: Path) -> Iterator[dict[str, Any]]:
    errors = validate_source(source)
    if errors:
        raise SourceConfigError("Invalid source descriptor: " + "; ".join(errors))
    if source.get("format") != "coco_detection":
        raise SourceConfigError("COCO converter requires format=coco_detection")
    annotation_relative = source.get("paths", {}).get("annotations")
    image_relative = source.get("paths", {}).get("images")
    if not isinstance(annotation_relative, str) or not isinstance(image_relative, str):
        raise SourceConfigError("COCO source requires paths.annotations and paths.images")
    grouping_map = _load_grouping_map(source, root)
    annotation_file = _safe_file(root, annotation_relative)
    try:
        coco = json.loads(annotation_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SourceConfigError(f"Invalid COCO annotation JSON: {annotation_relative}") from exc
    categories = coco.get("categories", [])
    category_names: dict[int, str] = {}
    for category in categories:
        try:
            category_id = int(category["id"])
            name = str(category["name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SourceConfigError("COCO categories require integer id and name") from exc
        if category_id in category_names:
            raise SourceConfigError(f"Duplicate COCO category id: {category_id}")
        category_names[category_id] = name
    images: dict[object, dict[str, Any]] = {}
    media_records: dict[object, tuple[dict[str, Any], str, int, int]] = {}
    for image in coco.get("images", []):
        if "id" not in image or "file_name" not in image:
            raise SourceConfigError("COCO images require id and file_name")
        image_id = image["id"]
        if image_id in images:
            raise SourceConfigError(f"Duplicate COCO image id: {image_id}")
        images[image_id] = image
        relative = Path(image_relative) / str(image["file_name"])
        image_path = _safe_file(root, relative)
        # Keep the already validated relative name. On Windows, resolve() may
        # return a differently cased/8.3 form that pathlib.relative_to rejects.
        relative_source_path = (Path(image_relative) / str(image["file_name"])).as_posix()
        media_records[image_id] = _media_record(source, root, image_path, relative_source_path,
                                                grouping_map=grouping_map)
    mapping_hash = _mapping_fingerprint(source)
    observations_by_image: dict[object, list[dict[str, Any]]] = {image_id: [] for image_id in images}
    for index, annotation in enumerate(coco.get("annotations", [])):
        image_id, category_id = annotation.get("image_id"), annotation.get("category_id")
        if image_id not in images:
            raise SourceConfigError(f"COCO annotation {annotation.get('id', index)} references unknown image id {image_id!r}")
        if category_id not in category_names:
            raise SourceConfigError(f"COCO annotation {annotation.get('id', index)} references unknown category id {category_id!r}")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise SourceConfigError(f"COCO annotation {annotation.get('id', index)} has invalid bbox")
        try:
            x, y, box_w, box_h = map(float, bbox)
        except (TypeError, ValueError) as exc:
            raise SourceConfigError(f"COCO annotation {annotation.get('id', index)} has non-numeric bbox") from exc
        if not all(math.isfinite(value) for value in (x, y, box_w, box_h)) or box_w <= 0 or box_h <= 0:
            raise SourceConfigError(f"COCO annotation {annotation.get('id', index)} has invalid bbox values")
        _, media_id, width, height = media_records[image_id]
        x2, y2 = x + box_w, y + box_h
        if x < 0 or y < 0 or x2 > width or y2 > height:
            raise SourceConfigError(f"COCO annotation {annotation.get('id', index)} bbox extends outside image {image_id}")
        observations_by_image[image_id].append(_observation(
            source, media_id, annotation.get("id", index), None, category_id, category_names[category_id],
            [x, y, x2, y2], mapping_hash,
            iscrowd=bool(annotation.get("iscrowd", 0)), is_ignore=bool(annotation.get("ignore", 0)),
        ))
    for image_id, media_data in media_records.items():
        media_record = media_data[0]
        yield media_record
        yield from observations_by_image[image_id]


def _voc_files(source: dict[str, Any], root: Path) -> tuple[list[Path], str, str]:
    paths = source.get("paths", {})
    xml_dir, image_dir = paths.get("annotations"), paths.get("images")
    if not isinstance(xml_dir, str) or not isinstance(image_dir, str):
        raise SourceConfigError("Pascal VOC source requires paths.annotations and paths.images")
    safe_xml_dir = (root / xml_dir).resolve()
    safe_image_dir = (root / image_dir).resolve()
    for candidate in (safe_xml_dir, safe_image_dir):
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise SourceConfigError("VOC annotation/image directory escapes the source root") from exc
    if not safe_xml_dir.is_dir() or not safe_image_dir.is_dir():
        raise SourceConfigError("VOC annotation or image directory does not exist")
    files = sorted(safe_xml_dir.rglob("*.xml"), key=lambda path: path.relative_to(safe_xml_dir).as_posix())
    if not files:
        raise SourceConfigError("No Pascal VOC XML files found")
    return files, xml_dir, image_dir


def _voc_image(xml_root: Any, xml_path: Path, root: Path, image_dir: str) -> tuple[Path, str]:
    filename = xml_root.findtext("filename")
    relative = Path(image_dir) / (filename.strip() if filename else f"{xml_path.stem}.jpg")
    if not filename:
        options = [Path(image_dir) / f"{xml_path.stem}{ext}" for ext in _IMAGE_EXTENSIONS]
        matches = [candidate for candidate in options if (root / candidate).is_file()]
        if len(matches) != 1:
            raise SourceConfigError(f"VOC XML {xml_path.name}: expected exactly one matching image; found {len(matches)}")
        relative = matches[0]
    image_path = _safe_file(root, relative)
    return image_path, relative.as_posix()


def convert_voc_records(source: dict[str, Any], root: Path) -> Iterator[dict[str, Any]]:
    errors = validate_source(source)
    if errors:
        raise SourceConfigError("Invalid source descriptor: " + "; ".join(errors))
    if source.get("format") != "pascal_voc":
        raise SourceConfigError("Pascal VOC converter requires format=pascal_voc")
    xml_files, _xml_dir, image_dir = _voc_files(source, root)
    mapping_hash = _mapping_fingerprint(source)
    grouping_map = _load_grouping_map(source, root)
    for xml_path in xml_files:
        xml_relative = Path(os.path.relpath(xml_path, root))
        xml_path = _safe_file(root, xml_relative)
        try:
            xml_root = ET.parse(xml_path).getroot()
        except Exception as exc:
            raise SourceConfigError(f"Invalid VOC XML: {xml_path.name}") from exc
        image_path, relative_image = _voc_image(xml_root, xml_path, root, image_dir)
        width, height = _image_info(image_path)
        annotation_dimensions = xml_root.find("size")
        if annotation_dimensions is not None:
            try:
                annotation_width = int(annotation_dimensions.findtext("width", "0"))
                annotation_height = int(annotation_dimensions.findtext("height", "0"))
            except ValueError as exc:
                raise SourceConfigError(f"VOC XML {xml_path.name}: invalid size metadata") from exc
            if annotation_width and annotation_height and (annotation_width, annotation_height) != (width, height):
                raise SourceConfigError(f"VOC XML {xml_path.name}: annotation size does not match decoded image size")
        relative_parts = Path(relative_image).parts
        split = next((part.casefold() for part in relative_parts if part.casefold() in {"train", "val", "valid", "validation", "test"}), None)
        split = "valid" if split in {"val", "validation"} else split
        media, media_id, _, _ = _media_record(source, root, image_path, relative_image, split, grouping_map)
        yield media
        for index, object_node in enumerate(xml_root.findall("object")):
            label = (object_node.findtext("name") or "").strip()
            box = object_node.find("bndbox")
            if not label or box is None:
                raise SourceConfigError(f"VOC XML {xml_path.name}: object requires name and bndbox")
            try:
                xmin = float(box.findtext("xmin", "nan")) - 1.0
                ymin = float(box.findtext("ymin", "nan")) - 1.0
                xmax = float(box.findtext("xmax", "nan"))
                ymax = float(box.findtext("ymax", "nan"))
            except ValueError as exc:
                raise SourceConfigError(f"VOC XML {xml_path.name}: non-numeric bbox") from exc
            if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)) or xmin < 0 or ymin < 0 or xmax <= xmin or ymax <= ymin or xmax > width or ymax > height:
                raise SourceConfigError(f"VOC XML {xml_path.name}: bbox invalid or outside image under VOC 1-based inclusive convention")
            flags = {
                "difficult": (object_node.findtext("difficult", "0").strip() == "1"),
                "truncated": (object_node.findtext("truncated", "0").strip() == "1"),
            }
            observation = _observation(source, media_id, f"{xml_relative.as_posix()}:{index}", None,
                                      None, label, [xmin, ymin, xmax, ymax], mapping_hash,
                                      annotation_flags=flags)
            if flags["difficult"] and observation["mapping_action"] == "map":
                observation["annotation_status"] = "difficult_review"
            yield observation


def _video_media_record(source: dict[str, Any], video_path: Path, relative: str,
                        width: int, height: int, fps: float, frame_count: int,
                        group_id: str | None = None) -> tuple[dict[str, Any], str]:
    fingerprint = _sha256_file(video_path)
    media_id = _stable_id(source["source_id"], source["source_version"], relative, fingerprint)
    grouping = source.get("grouping", {})
    grouping = grouping if isinstance(grouping, dict) else {}
    grouping_status = "verified" if group_id else grouping.get("status", "unknown")
    if not group_id:
        group_id = grouping.get("global_group_id") if grouping_status == "verified" else None
    record = {
        "entity": "media", "schema_version": 1, "media_id": media_id,
        "source_id": source["source_id"], "source_version": str(source["source_version"]),
        "raw_relative_path": relative.replace("\\", "/"), "content_sha256": fingerprint,
        "modality": "video", "width": width, "height": height, "fps": fps,
        "duration_ms": int(round(frame_count * 1000 / fps)) if fps > 0 else None,
        "frame_count": frame_count, "original_export_split": None,
        "global_group_id": group_id, "grouping_status": grouping_status if group_id else "unknown",
        "domain": source.get("domain"), "viewpoint": source.get("viewpoint"),
        "overlay_present": None, "provenance": _safe_provenance(source),
    }
    return record, media_id


def _video_timestamps(source: dict[str, Any], root: Path, frame_count: int, fps: float,
                      timestamps_file: str | None = None) -> dict[int, int]:
    timing = source.get("timing", {})
    if not isinstance(timing, dict):
        raise SourceConfigError("CVAT video source requires a timing contract")
    mode = timing.get("mode")
    if mode == "explicit_timestamps":
        filename = timestamps_file or source.get("paths", {}).get("timestamps")
        if not isinstance(filename, str):
            raise SourceConfigError("explicit_timestamps requires paths.timestamps JSON")
        path = _safe_file(root, filename)
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            values = {index: value for index, value in enumerate(data)}
        elif isinstance(data, dict):
            values = {int(key): value for key, value in data.items()}
        else:
            raise SourceConfigError("timestamp file must be a JSON list or frame-index object")
        result = {}
        for frame in range(frame_count):
            if frame not in values or not isinstance(values[frame], int) or values[frame] < 0:
                raise SourceConfigError(f"timestamp file is missing a valid PTS for frame {frame}")
            result[frame] = values[frame]
        return result
    if mode == "constant_frame_rate" and timing.get("cfr_verified") is True and fps > 0:
        return {frame: int(round(frame * 1000 / fps)) for frame in range(frame_count)}
    raise SourceConfigError("CVAT video requires explicit PTS timestamps or verified constant-frame-rate timing")


def cvat_tasks(source: dict[str, Any]) -> list[dict[str, Any]]:
    """Video/annotation pairs of a CVAT source: an explicit ``tasks`` list, or the single ``paths`` pair."""
    tasks = source.get("tasks")
    if tasks is not None:
        if not isinstance(tasks, list) or not tasks:
            raise SourceConfigError("tasks must be a nonempty list of {annotations, video[, timestamps, global_group_id]}")
        for task in tasks:
            if not isinstance(task, dict) or not isinstance(task.get("annotations"), str) or not isinstance(task.get("video"), str):
                raise SourceConfigError("each CVAT task requires string annotations and video paths")
        return tasks
    paths = source.get("paths", {})
    if not isinstance(paths.get("annotations"), str) or not isinstance(paths.get("video"), str):
        raise SourceConfigError("CVAT source requires paths.annotations and paths.video, or a tasks list")
    return [{"annotations": paths["annotations"], "video": paths["video"], "timestamps": paths.get("timestamps")}]


def _read_video_geometry(video_path: Path) -> tuple[int, int, int, float]:
    try:
        import cv2
    except ImportError as exc:
        raise SourceConfigError("Install the optional vision dependencies to read video dimensions") from exc
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise SourceConfigError(f"Unable to decode video {video_path.name}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise SourceConfigError(f"Video {video_path.name} has invalid decoded dimensions or frame count")
    return width, height, frame_count, fps


def convert_cvat_video_records(source: dict[str, Any], root: Path) -> Iterator[dict[str, Any]]:
    errors = validate_source(source)
    if errors:
        raise SourceConfigError("Invalid source descriptor: " + "; ".join(errors))
    if source.get("format") != "cvat_video":
        raise SourceConfigError("CVAT converter requires format=cvat_video")
    identity_review = source.get("identity_review", {})
    identity_verified = isinstance(identity_review, dict) and identity_review.get("status") == "verified" and not identity_review.get("sample_review_required", True)
    interpolation = source.get("interpolation", {})
    allow_interpolated = isinstance(interpolation, dict) and interpolation.get("policy") == "reviewed_interpolation" and interpolation.get("reviewed") is True
    mapping_hash = _mapping_fingerprint(source)
    grouping_map = _load_grouping_map(source, root)
    for task in cvat_tasks(source):
        yield from _convert_cvat_task(source, root, task, identity_verified=identity_verified,
                                      allow_interpolated=allow_interpolated, mapping_hash=mapping_hash,
                                      grouping_map=grouping_map)


def _convert_cvat_task(source: dict[str, Any], root: Path, task: dict[str, Any], *, identity_verified: bool,
                       allow_interpolated: bool, mapping_hash: str, grouping_map: dict[str, str]) -> Iterator[dict[str, Any]]:
    xml_path = _safe_file(root, task["annotations"])
    video_path = _safe_file(root, task["video"])
    width, height, frame_count, fps = _read_video_geometry(video_path)
    timestamps = _video_timestamps(source, root, frame_count, fps, task.get("timestamps"))
    try:
        annotation_root = ET.parse(xml_path).getroot()
    except Exception as exc:
        raise SourceConfigError(f"Invalid CVAT video XML: {xml_path.name}") from exc
    relative_video = Path(task["video"]).as_posix()  # not relpath(): resolve() may return an 8.3/long-name form
    group_id = task.get("global_group_id") or grouping_map.get(relative_video)
    media_record, media_id = _video_media_record(source, video_path, relative_video, width, height, fps, frame_count,
                                                 group_id=group_id)
    yield media_record
    for track in annotation_root.findall("track"):
        track_id = track.get("id")
        track_label = (track.get("label") or "").strip()
        if track_id is None or not track_label:
            raise SourceConfigError("CVAT track requires numeric id and label")
        try:
            numeric_track_id = int(track_id)
        except ValueError as exc:
            raise SourceConfigError(f"CVAT track id must be an integer: {track_id!r}") from exc
        person_id = f"{media_id}:track:{numeric_track_id}"
        for index, box in enumerate(track.findall("box")):
            try:
                frame = int(box.get("frame", "-1"))
                outside = box.get("outside", "0") == "1"
                occluded = box.get("occluded", "0") == "1"
                keyframe = box.get("keyframe", "1") != "0"
                coords = [float(box.get(key, "nan")) for key in ("xtl", "ytl", "xbr", "ybr")]
            except ValueError as exc:
                raise SourceConfigError(f"CVAT track {track_id}: invalid box attributes") from exc
            if not 0 <= frame < frame_count:
                raise SourceConfigError(f"CVAT track {track_id}: frame index outside video")
            if not keyframe and not allow_interpolated:
                continue
            label = track_label
            for attribute in box.findall("attribute"):
                if (attribute.get("name") or "").casefold() in {"state", "water_state", "class"} and (attribute.text or "").strip():
                    label = attribute.text.strip()
                    break
            if outside:
                observation = _observation(source, media_id, f"track:{track_id}:box:{index}", frame,
                                          track_id, label, None, mapping_hash, timestamp_ms=timestamps[frame],
                                          person_id=person_id, identity_verified=identity_verified,
                                          occluded=occluded, outside=True,
                                          annotation_flags={"cvat_keyframe": keyframe, "outside_marker": True})
                observation["mapping_action"] = "review"
                observation["canonical_label"] = None
                observation["annotation_status"] = "outside_marker"
                yield observation
                continue
            x1, y1, x2, y2 = coords
            if not all(math.isfinite(value) for value in coords) or x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
                raise SourceConfigError(f"CVAT track {track_id} frame {frame}: bbox outside decoded video frame")
            yield _observation(source, media_id, f"track:{track_id}:box:{index}", frame,
                              track_id, label, coords, mapping_hash,
                              timestamp_ms=timestamps[frame], person_id=person_id,
                              identity_verified=identity_verified, occluded=occluded, outside=False,
                              annotation_flags={"cvat_keyframe": keyframe, "interpolated": not keyframe})


def _read_jsonl_rows(path: Path, what: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SourceConfigError(f"{what} line {line_number} is not valid JSON") from exc
        if not isinstance(row, dict):
            raise SourceConfigError(f"{what} line {line_number} must be an object")
        rows.append(row)
    return rows


def _read_reviewed_jsonl(source: dict[str, Any], root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = source.get("paths", {})
    if not isinstance(paths.get("media"), str) or not isinstance(paths.get("annotations"), str):
        raise SourceConfigError("reviewed_jsonl source requires paths.media and paths.annotations")
    media_rows = _read_jsonl_rows(_safe_file(root, paths["media"]), "media list")
    seen: set[str] = set()
    for row in media_rows:
        path = row.get("path")
        if not isinstance(path, str) or not path or path in seen:
            raise SourceConfigError("every media row needs a unique string path")
        seen.add(path)
    return media_rows, _read_jsonl_rows(_safe_file(root, paths["annotations"]), "annotation list")


def convert_reviewed_jsonl_records(source: dict[str, Any], root: Path) -> Iterator[dict[str, Any]]:
    """Reviewed per-person annotations in a plain JSONL contract (see docs/DATASET_ONBOARDING.md).

    Media row: ``{"path", "modality": "image"|"video", "group_id"?}``. Annotation row: ``{"path", "label", "bbox":
    [x1,y1,x2,y2]}`` for images, plus ``frame``, ``timestamp_ms`` and ``person_id`` for video (``outside: true``
    rows carry no bbox and break a person's track). Nothing is inferred: missing timestamps or identities fail.
    """
    errors = validate_source(source)
    if errors:
        raise SourceConfigError("Invalid source descriptor: " + "; ".join(errors))
    if source.get("format") != "reviewed_jsonl":
        raise SourceConfigError("reviewed_jsonl converter requires format=reviewed_jsonl")
    media_rows, annotation_rows = _read_reviewed_jsonl(source, root)
    identity_review = source.get("identity_review", {})
    identity_verified = isinstance(identity_review, dict) and identity_review.get("status") == "verified" and not identity_review.get("sample_review_required", True)
    mapping_hash = _mapping_fingerprint(source)
    grouping_map = _load_grouping_map(source, root)
    by_path: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(annotation_rows):
        by_path.setdefault(str(row.get("path")), []).append((index, row))
    unknown = set(by_path) - {row["path"] for row in media_rows}
    if unknown:
        raise SourceConfigError(f"annotations reference {len(unknown)} path(s) missing from the media list")
    for media_row in media_rows:
        relative = media_row["path"]
        file_path = _safe_file(root, relative)
        modality = media_row.get("modality", source.get("modality"))
        if modality == "video":
            width, height, frame_count, fps = _read_video_geometry(file_path)
            group_id = media_row.get("group_id") or grouping_map.get(Path(relative).as_posix())
            media, media_id = _video_media_record(source, file_path, relative, width, height, fps, frame_count, group_id=group_id)
        elif modality == "image":
            media, media_id, width, height = _media_record(source, root, file_path, relative, media_row.get("split"),
                                                           {Path(relative).as_posix(): media_row["group_id"]} if media_row.get("group_id") else grouping_map)
            frame_count = None
        else:
            raise SourceConfigError(f"media {relative!r} needs modality image or video")
        yield media
        for index, row in by_path.get(relative, []):
            label, outside = row.get("label"), row.get("outside") is True
            if not isinstance(label, str) or not label.strip():
                raise SourceConfigError(f"annotation {index} of {relative!r} has no label")
            person_id = timestamp = frame = None
            if modality == "video":
                frame, timestamp = row.get("frame"), row.get("timestamp_ms")
                if not isinstance(frame, int) or isinstance(frame, bool) or not 0 <= frame < frame_count:
                    raise SourceConfigError(f"annotation {index} of {relative!r}: frame must be an integer inside the video")
                if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
                    raise SourceConfigError(f"annotation {index} of {relative!r}: explicit timestamp_ms is required")
                raw_person = row.get("person_id")
                if raw_person is None or str(raw_person).strip() == "":
                    raise SourceConfigError(f"annotation {index} of {relative!r}: person_id is required for video")
                person_id = f"{media_id}:person:{raw_person}"
            bbox = None
            if not outside:
                bbox = row.get("bbox")
                if (not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bbox)
                        or bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1] or bbox[2] > width or bbox[3] > height):
                    raise SourceConfigError(f"annotation {index} of {relative!r}: bbox must be finite xyxy pixels inside the frame")
                bbox = [float(v) for v in bbox]
            observation = _observation(source, media_id, f"jsonl:{index}", frame, row.get("label_id"), label, bbox, mapping_hash,
                                       timestamp_ms=timestamp, person_id=person_id,
                                       identity_verified=identity_verified and modality == "video",
                                       occluded=bool(row.get("occluded")) if modality == "video" else None,
                                       outside=outside if modality == "video" else None,
                                       annotation_flags={"source_row": index})
            if outside:
                observation.update(mapping_action="review", canonical_label=None, annotation_status="outside_marker")
            yield observation


def scan_source(source: dict[str, Any], root: Path) -> dict[str, Any]:
    """Count source media and label decisions without writing or decoding media."""
    errors = validate_source(source)
    if errors:
        raise SourceConfigError("Invalid source descriptor: " + "; ".join(errors))
    classes = read_class_names(source, root) if source.get("format") in {"yolo_detection", "coco_detection"} else []
    class_map = {str(class_id): name for class_id, name in classes}
    class_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    malformed: list[dict[str, Any]] = []
    media_count = 0
    annotation_count = 0
    if source.get("format") == "yolo_detection":
        pairs = list(yolo_annotation_files(root))
        def read_label(pair):
            return pair, pair[0].read_text(encoding="utf-8").splitlines()
        # Small label-file opens dominate Windows scans. Bound disk concurrency,
        # preserve source order, and keep path containment checks before reads.
        with ThreadPoolExecutor(max_workers=8, thread_name_prefix="label-scan") as pool:
            label_rows = list(pool.map(read_label, pairs))
        for (label_file, _image, relative_image), lines in label_rows:
            media_count += 1
            split = next((p.casefold() for p in Path(relative_image).parts if p.casefold() in {"train", "val", "valid", "validation", "test"}), "unknown")
            split_counts["valid" if split in {"val", "validation"} else split] += 1
            for line_number, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                annotation_count += 1
                fields = line.split()
                if len(fields) != 5 or not fields[0].lstrip("+-").isdigit():
                    malformed.append({"file": str(label_file), "line": line_number, "reason": "expected YOLO detection row with numeric class id and four bbox values"})
                    continue
                try:
                    cx, cy, box_w, box_h = map(float, fields[1:])
                    if not all(math.isfinite(v) for v in (cx, cy, box_w, box_h)):
                        raise ValueError("non-finite bbox")
                    if (box_w <= 0 or box_h <= 0 or cx - box_w / 2 < 0 or cy - box_h / 2 < 0
                            or cx + box_w / 2 > 1 or cy + box_h / 2 > 1):
                        raise ValueError("bbox extends outside image or has non-positive size")
                except ValueError as exc:
                    malformed.append({"file": str(label_file), "line": line_number, "reason": f"invalid normalized bbox: {exc}"})
                    continue
                source_label = class_map.get(fields[0])
                if source_label is None:
                    malformed.append({"file": str(label_file), "line": line_number, "reason": f"class id {fields[0]} missing from class list"})
                    continue
                decision = resolve_label(source_label, source.get("mapping", {}))
                class_counts[source_label] += 1
                action_counts[decision.action] += 1
    elif source.get("format") == "coco_detection":
        annotation_path = _safe_file(root, source.get("paths", {}).get("annotations", ""))
        try:
            coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SourceConfigError("Invalid COCO annotation JSON") from exc
        categories = {str(category.get("id")): str(category.get("name", "")) for category in coco.get("categories", [])}
        media_count = len(coco.get("images", []))
        annotation_count = len(coco.get("annotations", []))
        for annotation in coco.get("annotations", []):
            source_label = categories.get(str(annotation.get("category_id")))
            if not source_label:
                malformed.append({"annotation_id": annotation.get("id"), "reason": "category id missing from category list"})
                continue
            decision = resolve_label(source_label, source.get("mapping", {}))
            class_counts[source_label] += 1
            action_counts[decision.action] += 1
    elif source.get("format") == "pascal_voc":
        xml_files, _xml_dir, image_dir = _voc_files(source, root)
        media_count = len(xml_files)
        for xml_path in xml_files:
            try:
                xml_root = ET.parse(xml_path).getroot()
                _voc_image(xml_root, xml_path, root, image_dir)
            except Exception as exc:
                malformed.append({"file": str(xml_path), "reason": f"invalid VOC XML or image reference: {exc}"})
                continue
            for object_node in xml_root.findall("object"):
                source_label = (object_node.findtext("name") or "").strip()
                if not source_label or object_node.find("bndbox") is None:
                    malformed.append({"file": str(xml_path), "reason": "object requires name and bndbox"})
                    continue
                decision = resolve_label(source_label, source.get("mapping", {}))
                class_counts[source_label] += 1
                action_counts[decision.action] += 1
                annotation_count += 1
    elif source.get("format") == "cvat_video":
        tasks = cvat_tasks(source)
        media_count = len(tasks)
        for task in tasks:
            annotation_path = _safe_file(root, task["annotations"])
            _safe_file(root, task["video"])
            try:
                cvat_root = ET.parse(annotation_path).getroot()
            except Exception as exc:
                raise SourceConfigError("Invalid CVAT video XML") from exc
            for track in cvat_root.findall("track"):
                track_label = (track.get("label") or "").strip()
                for box in track.findall("box"):
                    if box.get("outside", "0") == "1":
                        continue
                    annotation_count += 1
                    source_label = track_label
                    for attribute in box.findall("attribute"):
                        if (attribute.get("name") or "").casefold() in {"state", "water_state", "class"} and (attribute.text or "").strip():
                            source_label = attribute.text.strip()
                            break
                    if not source_label:
                        malformed.append({"track_id": track.get("id"), "frame": box.get("frame"), "reason": "missing state label"})
                        continue
                    decision = resolve_label(source_label, source.get("mapping", {}))
                    class_counts[source_label] += 1
                    action_counts[decision.action] += 1
    elif source.get("format") == "reviewed_jsonl":
        media_rows, annotation_rows = _read_reviewed_jsonl(source, root)
        media_count = len(media_rows)
        known = {row["path"] for row in media_rows}
        for row in annotation_rows:
            if row.get("path") not in known:
                malformed.append({"path": str(row.get("path")), "reason": "annotation references a path missing from the media list"})
                continue
            if row.get("outside") is True:
                continue
            label = row.get("label")
            if not isinstance(label, str) or not label.strip():
                malformed.append({"path": row["path"], "reason": "annotation has no label"})
                continue
            annotation_count += 1
            decision = resolve_label(label, source.get("mapping", {}))
            class_counts[label] += 1
            action_counts[decision.action] += 1
    else:
        raise SourceConfigError(f"Scanning is not implemented for format={source.get('format')!r}")
    return {
        "source_id": source["source_id"],
        "source_version": str(source["source_version"]),
        "format": source["format"],
        "media_count": media_count,
        "annotation_count": annotation_count,
        "media_counts_by_original_split": dict(sorted(split_counts.items())),
        "annotation_counts_by_source_label": dict(sorted(class_counts.items())),
        "annotation_counts_by_mapping_action": dict(sorted(action_counts.items())),
        "malformed_annotation_count": len(malformed),
        "malformed_annotations_sample": malformed[:20],
        "source_data_modified": False,
    }


def convert_to_jsonl(source: dict[str, Any], root: Path, output: Path) -> dict[str, Any]:
    """Write canonical records outside the raw root, refusing to overwrite."""
    root_resolved = root.resolve()
    output_resolved = output.resolve()
    try:
        output_resolved.relative_to(root_resolved)
    except ValueError:
        pass
    else:
        raise SourceConfigError("Derived output must be outside the raw source root")
    protected_demo = Path("D:/DrowningDemo").resolve()
    try:
        output_resolved.relative_to(protected_demo)
    except ValueError:
        pass
    else:
        raise SourceConfigError("D:\\DrowningDemo is protected and cannot be a conversion output")
    if any(part.casefold() == "sources" for part in output_resolved.parts):
        raise SourceConfigError("A sources/ directory is protected and cannot be a conversion output")
    if output_resolved.exists():
        raise SourceConfigError(f"Refusing to overwrite existing output: {output_resolved}")
    if source.get("format") == "yolo_detection":
        records = convert_yolo_records(source, root)
    elif source.get("format") == "coco_detection":
        records = convert_coco_records(source, root)
    elif source.get("format") == "pascal_voc":
        records = convert_voc_records(source, root)
    elif source.get("format") == "cvat_video":
        records = convert_cvat_video_records(source, root)
    elif source.get("format") == "reviewed_jsonl":
        records = convert_reviewed_jsonl_records(source, root)
    else:
        raise SourceConfigError(f"Conversion is not implemented for format={source.get('format')!r}")
    output_resolved.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_resolved.with_name(output_resolved.name + f".{os.getpid()}.tmp")
    counts: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    record_count = 0
    try:
        with temp_path.open("x", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                record_count += 1
                counts[record["entity"]] += 1
                if record["entity"] == "observation":
                    labels[f"{record['mapping_action']}:{record['original_label']}"] += 1
        temp_path.rename(output_resolved)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return {
        "source_id": source["source_id"],
        "output": str(output_resolved),
        "record_count": record_count,
        "media_count": counts["media"],
        "observation_count": counts["observation"],
        "mapping_counts": dict(sorted(labels.items())),
        "conversion_performed": True,
    }
