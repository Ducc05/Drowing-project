"""Contact sheets of real annotated crops, so a person can check what each source label means."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from drowning.data.adapters import _safe_file
from drowning.data.crops import _crop_image, _crop_pil, _protected_output
from drowning.data.registry import SourceConfigError
from drowning.data.review import mapping_fingerprint


def read_canonical(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    media: dict[str, dict[str, Any]] = {}
    observations: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SourceConfigError(f"Invalid canonical JSONL at line {line_number}") from exc
            if record.get("entity") == "media":
                media[record["media_id"]] = record
            elif record.get("entity") == "observation":
                observations.append(record)
    return media, observations


def _video_frame(path: Path, frame_index: int) -> Image.Image:
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise SourceConfigError(f"Unable to open video {path.name}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, bgr = capture.read()
        if not ok:
            raise SourceConfigError(f"Unable to decode frame {frame_index} of {path.name}")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()


def crop_for_observation(raw_root: Path, record: dict[str, Any], observation: dict[str, Any], tile: int) -> Image.Image:
    media_path = _safe_file(raw_root, record["raw_relative_path"])
    if record.get("modality") == "video":
        return _crop_pil(_video_frame(media_path, observation["frame_index"]), observation["bbox"], 0.15, tile)
    return _crop_image(media_path, observation["bbox"], 0.15, tile)


def _sheet_key(label: str, kind: str) -> str:
    safe = "".join(c if c.isalnum() else "-" for c in label.casefold()).strip("-") or "label"
    return f"{safe}--{kind}"


def build_review_sheets(source: dict[str, Any], canonical_jsonl: Path, raw_root: Path, output_dir: Path, *,
                        per_label: int = 24, columns: int = 6, tile: int = 160,
                        seed: int = 20261026) -> dict[str, Any]:
    """Write one random sheet and one smallest-boxes sheet per source label; idempotent per mapping."""
    if per_label < 10 or columns < 1 or tile < 64:
        raise ValueError("per_label must be at least 10 so reviewers can meet the minimum sample count")
    fingerprint = mapping_fingerprint(source)
    manifest_path = output_dir / "samples.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("mapping_hash") == fingerprint and existing.get("seed") == seed:
            return existing
        raise SourceConfigError("Review sheet directory already holds sheets for a different mapping/seed")
    _protected_output(output_dir, raw_root)
    media, observations = read_canonical(canonical_jsonl)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        if isinstance(observation.get("bbox"), list) and observation.get("media_id") in media:
            by_label.setdefault(str(observation["original_label"]), []).append(observation)
    if not by_label:
        raise SourceConfigError("Canonical file has no boxed observations to sample")
    temp_dir = output_dir.with_name(output_dir.name + f".{os.getpid()}.tmp")
    temp_dir.mkdir(parents=True)
    sheets: list[dict[str, Any]] = []
    try:
        for label in sorted(by_label):
            rows = sorted(by_label[label], key=lambda item: item["observation_id"])
            action, canonical = rows[0].get("mapping_action"), rows[0].get("canonical_label")
            area = lambda item: (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1])
            choices = {
                "random": random.Random(f"{seed}:{label}").sample(rows, min(per_label, len(rows))),
                "smallest": sorted(rows, key=lambda item: (area(item), item["observation_id"]))[:per_label],
            }
            for kind, chosen in choices.items():
                key = _sheet_key(label, kind)
                grid_rows = (len(chosen) + columns - 1) // columns
                sheet = Image.new("RGB", (columns * tile, grid_rows * tile), (32, 36, 40))
                items = []
                for index, observation in enumerate(chosen):
                    record = media[observation["media_id"]]
                    crop = crop_for_observation(raw_root, record, observation, tile)
                    ImageDraw.Draw(crop).text((4, 3), str(index + 1), fill=(255, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
                    sheet.paste(crop, ((index % columns) * tile, (index // columns) * tile))
                    items.append({"index": index + 1, "observation_id": observation["observation_id"],
                                  "media_id": observation["media_id"], "raw_relative_path": record["raw_relative_path"],
                                  "frame_index": observation.get("frame_index"), "bbox": observation["bbox"], "box_area_px": round(area(observation), 1)})
                sheet.save(temp_dir / f"{key}.jpg", format="JPEG", quality=88)
                sheets.append({"key": key, "kind": kind, "original_label": label, "mapping_action": action,
                               "canonical_label": canonical, "file": f"{key}.jpg", "sample_count": len(chosen),
                               "total_for_label": len(rows), "items": items})
        manifest = {"schema_version": 1, "source_id": source["source_id"], "source_version": str(source["source_version"]),
                    "mapping_hash": fingerprint, "seed": seed, "tile_px": tile, "columns": columns, "sheets": sheets,
                    "training_eligible": False}
        (temp_dir / "samples.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_dir.rename(output_dir)
    except Exception:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return manifest


_SPLIT_COLORS = {"train": (66, 133, 244), "valid": (244, 160, 0), "test": (219, 68, 55), "unknown": (150, 150, 150)}


def _group_tile(raw_root: Path, record: dict[str, Any], tile: int) -> Image.Image:
    from PIL import ImageOps
    with Image.open(_safe_file(raw_root, record["raw_relative_path"])) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image = ImageOps.pad(image, (tile, tile), method=Image.Resampling.LANCZOS, color=(32, 36, 40))
    split = str(record.get("original_export_split") or "unknown")
    draw = ImageDraw.Draw(image)
    color = _SPLIT_COLORS.get(split, _SPLIT_COLORS["unknown"])
    draw.rectangle((0, 0, tile - 1, tile - 1), outline=color, width=5)
    draw.rectangle((0, tile - 18, 54, tile), fill=color)
    draw.text((4, tile - 16), split[:5], fill=(255, 255, 255))
    return image


def build_group_sheets(source: dict[str, Any], canonical_jsonl: Path, raw_root: Path, proposed: dict[str, str],
                       output_dir: Path, *, rows: int = 12, per_row: int = 6, tile: int = 150,
                       seed: int = 20261026) -> dict[str, Any]:
    """Evidence for the question "are images placed in one group really the same photo or scene?".

    Three sheets: clusters that cross the export's own splits (one row per cluster, border colour = split),
    ordinary same-split clusters, and a sample of the largest cluster (to expose chained, unrelated images).
    """
    manifest_path = output_dir / "groups.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    media, _ = read_canonical(canonical_jsonl)
    by_group: dict[str, list[dict[str, Any]]] = {}
    for record in media.values():
        if record.get("modality", "image") == "image" and record["raw_relative_path"] in proposed:
            by_group.setdefault(proposed[record["raw_relative_path"]], []).append(record)
    clusters = {gid: sorted(items, key=lambda r: r["raw_relative_path"]) for gid, items in by_group.items() if len(items) > 1}
    if not clusters:
        raise SourceConfigError("There are no multi-image clusters to show")
    crossing = {gid: items for gid, items in clusters.items() if len({r.get("original_export_split") for r in items} - {None}) > 1}
    within = {gid: items for gid, items in clusters.items() if gid not in crossing}
    rng = random.Random(f"{seed}:groups")
    largest_id = max(clusters, key=lambda gid: (len(clusters[gid]), gid))

    def pick(pool: dict[str, list[dict[str, Any]]]) -> list[str]:
        ids = sorted(pool)
        rng.shuffle(ids)
        return ids[:rows]

    sets = {"cross-split": pick(crossing), "same-split": pick(within), "largest": [largest_id]}
    temp = output_dir.with_name(output_dir.name + f".{os.getpid()}.tmp")
    temp.mkdir(parents=True)
    sheets = []
    try:
        for key, ids in sets.items():
            if not ids:
                continue
            if key == "largest":
                members = clusters[largest_id]
                chosen = rng.sample(members, min(len(members), rows * per_row // 2))
                grid = [[chosen[i:i + per_row] for i in range(0, len(chosen), per_row)]]
                grid = grid[0]
            else:
                grid = []
                for gid in ids:
                    members = clusters[gid]
                    grid.append(rng.sample(members, min(per_row, len(members))))
            sheet = Image.new("RGB", (per_row * tile, len(grid) * tile), (20, 22, 24))
            items = []
            for row_index, row in enumerate(grid):
                for column, record in enumerate(row):
                    sheet.paste(_group_tile(raw_root, record, tile), (column * tile, row_index * tile))
                items.append({"row": row_index + 1, "images": [{"path": r["raw_relative_path"], "split": r.get("original_export_split")} for r in row],
                              "cluster_size": len(clusters[ids[row_index]]) if key != "largest" else len(clusters[largest_id])})
            file_name = f"groups--{key}.jpg"
            sheet.save(temp / file_name, format="JPEG", quality=88)
            sheets.append({"key": f"groups--{key}", "file": file_name, "rows": len(grid), "kind": key,
                           "clusters_in_export": len(crossing) if key == "cross-split" else len(within) if key == "same-split" else 1,
                           "items": items})
        manifest = {"schema_version": 1, "source_id": source["source_id"], "seed": seed,
                    "legend": {"border colour = split of the original export": {k: "#%02x%02x%02x" % v for k, v in _SPLIT_COLORS.items()}},
                    "question": "Within one row, are the images the same photo or scene (rotated, recoloured, cropped)?",
                    "sheets": sheets}
        (temp / "groups.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.rename(output_dir)
    except Exception:
        import shutil
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return manifest
