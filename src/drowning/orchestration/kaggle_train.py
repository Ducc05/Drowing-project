"""Training-input assembly and candidate training for the Kaggle notebook (a library, so it is testable).

``prepare_training_inputs`` uses only sources whose mapping a person approved and whose groups are
verified; everything else is listed as excluded with the reason. Splits are made by verified group
across all sources before any crop or window exists. ``run_train_candidate`` trains two stages,
evaluates on the validation split only, never reads the test split, and refuses to run without a GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import uuid
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from drowning.data.adapters import convert_to_jsonl
from drowning.data.audit import audit_jsonl
from drowning.data.crops import build_crops, build_video_frame_crops
from drowning.data.grouping import SplitBlocked, build_group_split
from drowning.data.leakage import near_copy_report
from drowning.data.registry import SourceConfigError, load_source, validate_source
from drowning.data.autogroup import assign_automatic_groups
from drowning.data.review import advisory_mapping_review, is_mapping_approved, mapping_fingerprint, mapping_review
from drowning.data.review_rules import gating_mode
from drowning.data.samples import read_canonical
from drowning.data.windows import build_temporal_windows
from drowning.orchestration.kaggle_run import discover_source_root, sha256_file

LABELS = ("drowning", "swimming", "out_of_water")
SPLITS_NEEDING_ALL_CLASSES = ("train", "validation")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_training_inputs(*, sources_dir: Path, reviews_dir: Path, input_root: Path, work_dir: Path,
                            recipes_dir: Path, seed: int = 20261026, branch: str = "temporal",
                            frame_stride_ms: int = 500, max_frames_per_video: int = 300,
                            only_sources: set[str] | None = None,
                            holdout_sources: list[str] | None = None) -> dict[str, Any]:
    if branch not in {"temporal", "image_only"}:
        raise ValueError("branch must be temporal or image_only")
    image_only = branch == "image_only"
    work_dir.mkdir(parents=True, exist_ok=True)
    advisory = gating_mode(sources_dir.parent / "review_policy.yaml") == "advisory"
    image_recipe = yaml.safe_load((recipes_dir / "image.yaml").read_text(encoding="utf-8"))
    temporal_recipe = yaml.safe_load((recipes_dir / "temporal.yaml").read_text(encoding="utf-8"))
    sampling, image_input = temporal_recipe["sampling"], image_recipe["input"]
    included: list[dict[str, Any]] = []
    excluded: dict[str, str] = {}
    for path in sorted(sources_dir.glob("*.yaml")):
        source = load_source(path, reviews_dir)
        source_id = str(source.get("source_id"))
        if only_sources is not None and source_id not in only_sources:
            continue
        if validate_source(source):
            excluded[source_id] = "invalid_descriptor"
            continue
        found = discover_source_root(source, input_root)
        if found["status"] != "found":
            excluded[source_id] = f"input_{found['status']}"
            continue
        if not is_mapping_approved(source):
            if not advisory:
                excluded[source_id] = "mapping_review_not_approved"
                continue
            source["mapping_review"] = advisory_mapping_review()
        canonical = work_dir / f"{source_id}@{source['source_version']}.jsonl"
        try:
            convert_to_jsonl(source, found["root"], canonical)
        except SourceConfigError as exc:
            excluded[source_id] = f"conversion_failed: {exc}"[:300]
            continue
        if not audit_jsonl([canonical])["valid"]:
            excluded[source_id] = "canonical_audit_failed"
            continue
        media, _ = read_canonical(canonical)
        ungrouped = not media or any(m.get("grouping_status") != "verified" or not m.get("global_group_id") for m in media.values())
        auto_groups = None
        if ungrouped and media and advisory:
            auto_groups = assign_automatic_groups(canonical, found["root"])
            media, _ = read_canonical(canonical)
            ungrouped = any(m.get("grouping_status") != "verified" or not m.get("global_group_id") for m in media.values())
        if ungrouped:
            excluded[source_id] = "source_groups_not_verified"
            continue
        included.append({"source": source, "root": found["root"], "canonical": canonical, "media": media,
                         "review": {"mapping": mapping_review(source), "auto_groups": auto_groups}})

    blockers: list[str] = []
    report: dict[str, Any] = {"schema_version": 1, "seed": seed, "branch": branch, "excluded_sources": excluded,
                              "included_sources": [i["source"]["source_id"] for i in included],
                              "review_mode": "advisory" if advisory else "enforced",
                              "unconfirmed": {i["source"]["source_id"]: {
                                  "mapping": i["review"]["mapping"], "groups": (i["review"]["auto_groups"] or {}).get("assurance")}
                                  for i in included if i["review"]["mapping"] != "confirmed" or i["review"]["auto_groups"]}}
    if not included:
        blockers.append("no_source_ready_for_training")
        return {"blockers": blockers, "report": {**report, "blockers": blockers}, "dataset_fingerprint": None}

    merged = work_dir / "merged.jsonl"
    with merged.open("w", encoding="utf-8", newline="\n") as stream:
        for item in included:
            stream.write(item["canonical"].read_text(encoding="utf-8"))
    all_media = [m for item in included for m in item["media"].values()]
    try:
        split = build_group_split(all_media, seed=seed)
    except SplitBlocked as exc:
        blockers.append(f"split_blocked: {exc}")
        return {"blockers": blockers, "report": {**report, "blockers": blockers}, "dataset_fingerprint": None}
    holdout = set(holdout_sources or [])
    if holdout:
        # A whole source is kept out of training and used only to measure how the model does on data it never saw
        # from a different origin. Its images that are byte-identical to a training image are excluded, not scored.
        if not image_only:
            raise ValueError("holdout_sources needs the image-only branch")
        source_of = {m["media_id"]: m.get("source_id") for m in all_media}
        sha_of = {m["media_id"]: m.get("content_sha256") for m in all_media}
        present = set(source_of.values())
        if holdout - present or not (present - holdout):
            blockers.append("holdout_source_not_found:" + ",".join(sorted(holdout - present)) if holdout - present
                            else "every_source_is_held_out")
            return {"blockers": blockers, "report": {**report, "blockers": blockers}, "dataset_fingerprint": None}
        train_side_hashes = {sha_of[row["media_id"]] for row in split["manifest"]
                             if source_of.get(row["media_id"]) not in holdout and sha_of.get(row["media_id"])}
        excluded_duplicates = 0
        for row in split["manifest"]:
            if source_of.get(row["media_id"]) in holdout:
                duplicate = sha_of.get(row["media_id"]) in train_side_hashes
                row["split"], row["split_reason"] = ("excluded", "exact_duplicate_of_training_image") if duplicate else ("heldout", "held_out_source")
                excluded_duplicates += int(duplicate)
        split["media_counts_by_split"] = dict(sorted(Counter(row["split"] for row in split["manifest"]).items()))
        report["holdout_sources"] = sorted(holdout)
        report["holdout_exact_duplicates_excluded"] = excluded_duplicates
    split_path = work_dir / "split_manifest.jsonl"
    _write_jsonl(split_path, split["manifest"])
    report["split_media_counts"] = split["media_counts_by_split"]
    report["source_video_disjointness_verified"] = False
    report["split_basis"] = "reviewer-verified groups; the source video of loose images remains unknown"
    # How many validation / held-out images have a near copy (mirrored or turned too) among the training images.
    # A lower bound; informational, it never blocks a run.
    split_of = {row["media_id"]: row["split"] for row in split["manifest"]}
    leak_paths: dict[str, list[Path]] = defaultdict(list)
    for item in included:
        for media_id, media in item["media"].items():
            if media.get("modality", "image") == "image" and split_of.get(media_id) in {"train", "validation", "heldout"}:
                leak_paths[split_of[media_id]].append(Path(item["root"]) / media["raw_relative_path"])
    try:
        stride = max(1, len(leak_paths["train"]) // 20000)
        report["leakage_check"] = {name: near_copy_report(leak_paths["train"][::stride], leak_paths[name][::max(1, len(leak_paths[name]) // 4000)])
                                   for name in ("validation", "heldout") if leak_paths.get(name) and leak_paths.get("train")}
    except (OSError, ValueError) as exc:
        report["leakage_check"] = {"error": f"{type(exc).__name__}"}

    # -- image crops (one directory per image-bearing source, one merged manifest) ---------------
    crops_root = work_dir / "crops"
    crops_root.mkdir(exist_ok=True)
    crop_rows: list[dict[str, Any]] = []
    for index, item in enumerate(included):
        image_ids = {mid for mid, m in item["media"].items() if m.get("modality") == "image"}
        if not image_ids:
            continue
        canonical = item["canonical"]
        if len(image_ids) != len(item["media"]):
            canonical = work_dir / f"{canonical.stem}.images-only.jsonl"
            _write_jsonl(canonical, [r for r in _read_jsonl(item["canonical"])
                                     if r.get("media_id") in image_ids])
        directory = f"{index:02d}-{item['source']['source_id']}"
        build_crops(item["source"], canonical, item["root"], crops_root / directory, split_manifest=split_path,
                    context_fraction=float(image_input["crop_context_fraction"]), size=int(image_input["size"][0]))
        for row in _read_jsonl(crops_root / directory / "crop_manifest.jsonl"):
            row["crop_relative_path"] = f"{directory}/{row['crop_relative_path']}"
            crop_rows.append(row)
    crop_manifest = crops_root / "crop_manifest.jsonl"
    _write_jsonl(crop_manifest, crop_rows)

    # -- video frames as still images (image-only mode): a video source teaches the image head before the loose images do --
    video_crops_root = work_dir / "video_crops"
    video_rows: list[dict[str, Any]] = []
    video_crops_root.mkdir(exist_ok=True)
    if image_only:
        for index, item in enumerate(included):
            if not any(m.get("modality") == "video" for m in item["media"].values()):
                continue
            directory = f"{index:02d}-{item['source']['source_id']}"
            build_video_frame_crops(item["source"], item["canonical"], item["root"], video_crops_root / directory,
                                    split_manifest=split_path, stride_ms=frame_stride_ms, max_per_video=max_frames_per_video,
                                    context_fraction=float(image_input["crop_context_fraction"]), size=int(image_input["size"][0]))
            for row in _read_jsonl(video_crops_root / directory / "crop_manifest.jsonl"):
                row["crop_relative_path"] = f"{directory}/{row['crop_relative_path']}"
                video_rows.append(row)
    video_crop_manifest = video_crops_root / "crop_manifest.jsonl"
    _write_jsonl(video_crop_manifest, video_rows)

    # -- temporal windows -------------------------------------------------------------------------
    assignments = {row["media_id"]: {"split": row["split"], "global_group_id": row["global_group_id"]} for row in split["manifest"]}
    if image_only:  # no windows are built: the image head is trained alone and tracking only aggregates at inference
        window_report = {"windows": [], "issues": []}
    else:
        window_report = build_temporal_windows(
            merged, sequence_length=int(sampling["sequence_length"]), sample_period_ms=int(sampling["sample_period_ms"]),
            max_track_gap_ms=int(sampling["max_track_gap_ms"]), min_valid_ratio=float(sampling["min_valid_ratio"]),
            split_assignments=assignments)
    windows = [w for w in window_report["windows"] if w["training_eligible"]]
    window_manifest = work_dir / "window_manifest.jsonl"
    _write_jsonl(window_manifest, windows)
    video_roots = {i["source"]["source_id"]: i["root"] for i in included
                   if any(m.get("modality") == "video" for m in i["media"].values())}

    # -- coverage -----------------------------------------------------------------------------------
    image_cov: dict[str, Counter] = {s: Counter() for s in ("train", "validation", "test", "heldout")}
    for row in crop_rows:
        if row.get("training_eligible") is True:
            image_cov.setdefault(row["split"], Counter())[row["canonical_label"]] += 1
    window_cov: dict[str, Counter] = {s: Counter() for s in ("train", "validation", "test")}
    for window in windows:
        window_cov[window["split"]][window["label"]] += 1
    video_cov: dict[str, Counter] = {s: Counter() for s in ("train", "validation", "test")}
    for row in video_rows:
        if row.get("training_eligible") is True:
            video_cov.setdefault(row["split"], Counter())[row["canonical_label"]] += 1
    report["image_crops_by_split"] = {s: dict(c) for s, c in image_cov.items()}
    report["video_frame_crops_by_split"] = {s: dict(c) for s, c in video_cov.items()}
    report["video_frame_sampling"] = {"stride_ms": frame_stride_ms, "max_frames_per_video": max_frames_per_video}
    has_images, has_video_frames = bool(image_cov["train"]), bool(video_cov["train"])
    report["training_phases"] = (["video_frames"] if has_video_frames else []) + (["images"] if has_images else [])
    report["temporal_windows_by_split"] = {s: dict(c) for s, c in window_cov.items()}
    report["temporal_window_issues"] = sorted({issue["code"] for issue in window_report["issues"]})
    if image_only:
        if not has_images and not has_video_frames:
            blockers.append("no_training_crops")
    elif not crop_rows:
        blockers.append("no_image_crops")
    if not windows and not image_only:
        blockers.append("no_temporal_windows")
    for split_name in SPLITS_NEEDING_ALL_CLASSES:
        for kind, coverage in (("image", image_cov), ("video_frame", video_cov), ("temporal", window_cov)):
            missing = sorted(set(LABELS) - set(coverage[split_name]))
            if missing and (kind == "image" and (image_cov["train"] or image_cov["validation"])
                            or kind == "video_frame" and (video_cov["train"] or video_cov["validation"])
                            or kind == "temporal" and windows and not image_only):
                blockers.append(f"{kind}_{split_name}_split_missing_classes:{','.join(missing)}")
    hasher = hashlib.sha256()
    hasher.update(f"branch:{branch}".encode())
    for part in (merged, split_path, crop_manifest, video_crop_manifest, window_manifest, recipes_dir / "image.yaml",
                 recipes_dir / "temporal.yaml"):
        hasher.update(sha256_file(part).encode())
    for item in sorted(included, key=lambda i: i["source"]["source_id"]):
        hasher.update(f"{item['source']['source_id']}:{mapping_fingerprint(item['source'])}".encode())
    fingerprint = hasher.hexdigest()
    report["dataset_fingerprint"] = fingerprint
    report["blockers"] = blockers
    return {"blockers": blockers, "report": report, "dataset_fingerprint": fingerprint, "recipes": (image_recipe, temporal_recipe),
            "paths": {"merged": merged, "split": split_path, "crop_manifest": crop_manifest, "window_manifest": window_manifest,
                      "video_crop_manifest": video_crop_manifest},
            "video_roots": video_roots}


def class_weights(counts: Counter, *, floor: float = 0.25, ceiling: float = 8.0) -> list[float]:
    """Inverse-frequency weights from the training split only, normalized to mean 1 and clipped."""
    total = sum(counts.values())
    raw = [total / (len(LABELS) * counts[label]) if counts[label] else 1.0 for label in LABELS]
    return [round(min(ceiling, max(floor, value)), 4) for value in raw]


def build_metrics(*, completion_kind: str, candidate_id: str, dataset_fingerprint: str, **extra: Any) -> dict[str, Any]:
    """The fixed envelope that ``verify_candidate_archive`` checks, plus run-specific results."""
    return {"completion_kind": completion_kind, "approved": False, "candidate_id": candidate_id,
            "dataset_fingerprint": dataset_fingerprint, "test_split_used": False, **extra}


def initialize_training_model(inputs: dict[str, Any], *, device: str, options: dict[str, Any],
                              parent_bundle_dir: Path | None = None):
    """Initialize fresh weights or load one verified parent; optimizers are always new."""
    from drowning.ml.bundle import load_bundle
    from drowning.ml.model import DrowningTemporalModel

    continuing = bool(options.get("continue_from_approved"))
    if continuing != (parent_bundle_dir is not None):
        raise ValueError("continuation requires exactly one pinned parent checkpoint")
    if not continuing:
        return DrowningTemporalModel(hidden_size=int(inputs["recipes"][1]["model"]["hidden_size"]),
                                      pretrained_encoder=bool(options.get("pretrained"))).to(device)
    if options.get("pretrained"):
        raise ValueError("disable ImageNet initialization when continuing a checkpoint")
    model, manifest = load_bundle(parent_bundle_dir, require_temporal=True, map_location=device)
    expected_kind = "image_tracked" if options.get("branch", "temporal") == "image_only" else "temporal"
    if manifest.get("model_kind", "temporal") != expected_kind:
        raise ValueError("parent checkpoint model kind does not match the training branch")
    cfg = inputs["recipes"][0]["input"]
    preprocessing = manifest["preprocessing"]
    actual_size = preprocessing["size"]
    actual_size = [actual_size, actual_size] if isinstance(actual_size, int) else actual_size
    if (actual_size != cfg["size"] or preprocessing["mean"] != cfg["mean"]
            or preprocessing["std"] != cfg["std"] or preprocessing["context_fraction"] != cfg["crop_context_fraction"]):
        raise ValueError("parent checkpoint preprocessing differs from the training recipe")
    if int(manifest["hidden_size"]) != int(inputs["recipes"][1]["model"]["hidden_size"]):
        raise ValueError("parent checkpoint architecture differs from the training recipe")
    if expected_kind == "temporal":
        sampling = inputs["recipes"][1]["sampling"]
        expected = {key: int(sampling[key]) for key in ("sequence_length", "sample_period_ms", "max_track_gap_ms")}
        if any(manifest["temporal_contract"].get(key) != value for key, value in expected.items()):
            raise ValueError("parent checkpoint temporal sampling differs from the training recipe")
    return model.to(device)


def _validation_samples(data: dict[str, Any], heldout_sets: dict[str, Any], limit: int = 3000) -> dict[str, list[dict[str, Any]]]:
    """Which validation / held-out crops were scored (source file, box, label), so another model can be scored on the same ones.

    Kept compact and sampled evenly when large; it travels in the run's metrics.json and is not stored in the registry.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    pools = {f"{name}:validation": sets["val"].rows for name, sets in data.items()}
    pools.update({f"{name}:heldout": dataset.rows for name, dataset in heldout_sets.items()})
    for key, rows in pools.items():
        stride = max(1, -(-len(rows) // limit))
        out[key] = [{"s": row.get("source_id"), "p": row.get("source_relative_path"), "f": row.get("frame_index"),
                     "b": [round(float(v), 1) for v in row["bbox_xyxy"]], "l": row["canonical_label"]} for row in rows[::stride]]
    return out


def _evaluate_by_source(model: Any, dataset: Any, *, batch: int, device: str) -> dict[str, Any]:
    """Accuracy, per-class recall and confusion matrix for each originating source, so one source cannot hide another."""
    from torch.utils.data import DataLoader, Subset

    from drowning.ml.training import evaluate_image_model

    indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset.rows):
        indices[str(row.get("source_id") or "unknown")].append(index)
    return {source: evaluate_image_model(model, DataLoader(Subset(dataset, members), batch_size=batch, shuffle=False, num_workers=0),
                                         device=device) for source, members in sorted(indices.items())}


def _train_two_stage(inputs: dict[str, Any], *, device: str, options: dict[str, Any], seed: int,
                     parent_bundle_dir: Path | None = None) -> dict[str, Any]:
    """Image stage then causal temporal stage (needs labeled video windows)."""
    import torch
    from torch.utils.data import DataLoader

    from drowning.ml.datasets import CanonicalImageCropDataset, TemporalVideoWindowDataset
    from drowning.ml.model import DrowningTemporalModel
    from drowning.ml.training import evaluate_image_model, evaluate_temporal_model, train_stage_a_epoch, train_stage_b_epoch

    image_recipe, temporal_recipe = inputs["recipes"]
    paths, input_cfg = inputs["paths"], image_recipe["input"]
    size = int(input_cfg["size"][0])
    image_train = CanonicalImageCropDataset(paths["crop_manifest"], split="train", output_size=size, mean=input_cfg["mean"], std=input_cfg["std"])
    image_val = CanonicalImageCropDataset(paths["crop_manifest"], split="validation", output_size=size, mean=input_cfg["mean"], std=input_cfg["std"])
    window_kwargs = dict(video_root=inputs["video_roots"], output_size=size, context_fraction=input_cfg["crop_context_fraction"],
                         mean=input_cfg["mean"], std=input_cfg["std"])
    window_train = TemporalVideoWindowDataset(paths["merged"], paths["window_manifest"], split="train", **window_kwargs)
    window_val = TemporalVideoWindowDataset(paths["merged"], paths["window_manifest"], split="validation", **window_kwargs)
    try:
        image_weights = torch.tensor(class_weights(Counter(r["canonical_label"] for r in image_train.rows)), device=device)
        temporal_weights = torch.tensor(class_weights(Counter(w["label"] for w in window_train.windows)), device=device)
        image_batch, temporal_batch = int(image_recipe["training"]["batch_size"]), int(temporal_recipe["training"]["batch_size"])
        loaders = {
            "image_train": DataLoader(image_train, batch_size=image_batch, shuffle=True, num_workers=0,
                                      generator=torch.Generator().manual_seed(seed)),
            "image_val": DataLoader(image_val, batch_size=image_batch, shuffle=False, num_workers=0),
            "window_train": DataLoader(window_train, batch_size=temporal_batch, shuffle=True, num_workers=0,
                                       generator=torch.Generator().manual_seed(seed + 1)),
            "window_val": DataLoader(window_val, batch_size=temporal_batch, shuffle=False, num_workers=0)}
        model = initialize_training_model(inputs, device=device, options=options, parent_bundle_dir=parent_bundle_dir)
        image_epochs, temporal_epochs = int(options.get("image_epochs", 5)), int(options.get("temporal_epochs", 5))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(image_recipe["training"]["learning_rate"]))
        stage_a = [train_stage_a_epoch(model, loaders["image_train"], optimizer, device=device, class_weights=image_weights)
                   for _ in range(image_epochs)]
        image_after_a = evaluate_image_model(model, loaders["image_val"], device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(temporal_recipe["training"]["learning_rate"]))
        stage_b = [train_stage_b_epoch(model, loaders["window_train"], optimizer, device=device, class_weights=temporal_weights,
                                       image_aux_weight=float(temporal_recipe["training"]["image_auxiliary_loss_weight"]))
                   for _ in range(temporal_epochs)]
        image_after_b = evaluate_image_model(model, loaders["image_val"], device=device)
        temporal_val = evaluate_temporal_model(model, loaders["window_val"], device=device)
        samples = {"image_train": len(image_train), "image_validation": len(image_val),
                   "windows_train": len(window_train), "windows_validation": len(window_val)}
    finally:
        window_train.close()
        window_val.close()
    return {"model": model, "epochs": {"image_epochs": image_epochs, "temporal_epochs": temporal_epochs}, "samples": samples,
            "class_weights": {"image": dict(zip(LABELS, image_weights.tolist())), "temporal": dict(zip(LABELS, temporal_weights.tolist()))},
            "results": {"image_stage_history": stage_a, "temporal_stage_history": stage_b,
                        "image_validation_after_stage_a": image_after_a, "image_validation": image_after_b,
                        "temporal_validation": temporal_val,
                        "forgetting_check": "image_validation_after_stage_a vs image_validation shows what stage B did to the image head"}}


def _train_image_only(inputs: dict[str, Any], *, device: str, options: dict[str, Any], seed: int,
                      parent_bundle_dir: Path | None = None) -> dict[str, Any]:
    """Image head only. Phase 1 uses frames cut from labeled videos (when there are any), phase 2 uses loose images.

    The images come last on purpose: they are the more numerous and varied data, and the video phase gives the
    encoder motion-blur / water-surface variety first. ``video_frame_validation`` is measured again after the image
    phase so any forgetting of the video domain is visible in the report.
    """
    import torch
    from torch.utils.data import DataLoader

    from drowning.ml.datasets import CanonicalImageCropDataset
    from drowning.ml.model import DrowningTemporalModel
    from drowning.ml.training import evaluate_image_model, train_stage_a_epoch

    image_recipe, temporal_recipe = inputs["recipes"]
    paths, input_cfg = inputs["paths"], image_recipe["input"]
    size, batch = int(input_cfg["size"][0]), int(image_recipe["training"]["batch_size"])
    kwargs = dict(output_size=size, mean=input_cfg["mean"], std=input_cfg["std"])
    phases = inputs["report"]["training_phases"]
    manifests = {"video_frames": paths["video_crop_manifest"], "images": paths["crop_manifest"]}
    epochs = {"video_frames": int(options.get("video_epochs", 3)), "images": int(options.get("image_epochs", 5))}
    augment = bool(options.get("augment", True))
    data = {name: {"train": CanonicalImageCropDataset(manifests[name], split="train", augment=augment, augment_seed=seed, **kwargs),
                   "val": CanonicalImageCropDataset(manifests[name], split="validation", **kwargs)} for name in phases}
    model = initialize_training_model(inputs, device=device, options=options, parent_bundle_dir=parent_bundle_dir)

    def validate(name: str) -> dict[str, Any]:
        return evaluate_image_model(model, DataLoader(data[name]["val"], batch_size=batch, shuffle=False, num_workers=0), device=device)

    histories: dict[str, list] = {}
    selection: dict[str, Any] = {}
    patience = int(options.get("patience", 3))
    weights: dict[str, Any] = {}
    after_phase: dict[str, dict[str, Any]] = {}
    before_images = None
    for index, name in enumerate(phases):
        train_set = data[name]["train"]
        weights[name] = torch.tensor(class_weights(Counter(r["canonical_label"] for r in train_set.rows)), device=device)
        loader = DataLoader(train_set, batch_size=batch, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed + index))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(image_recipe["training"]["learning_rate"]))
        history: list[dict[str, Any]] = []
        best_state, best_score, best_epoch, stale = None, -1.0, 0, 0
        for epoch in range(epochs[name]):
            train_set.set_epoch(epoch)
            history.append(train_stage_a_epoch(model, loader, optimizer, device=device, class_weights=weights[name]))
            score = float(validate(name)["macro_f1"])
            history[-1]["validation_macro_f1"] = round(score, 4)
            if score > best_score + 1e-4:
                best_score, best_epoch, stale = score, epoch + 1, 0
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if patience > 0 and stale >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)  # keep the epoch with the best validation macro-F1, not the last one
        selection[name] = {"best_epoch": best_epoch, "epochs_run": len(history), "epochs_requested": epochs[name],
                           "best_validation_macro_f1": round(best_score, 4), "early_stopped": len(history) < epochs[name]}
        histories[name] = history
        after_phase[name] = validate(name)
        if name == "video_frames" and "images" in data:
            before_images = validate("images")
    final = {name: validate(name) for name in phases}
    by_source = {name: _evaluate_by_source(model, data[name]["val"], batch=batch, device=device) for name in phases}
    heldout: dict[str, Any] = {}
    heldout_sets: dict[str, Any] = {}
    for name, manifest in manifests.items():
        try:
            heldout_set = CanonicalImageCropDataset(manifest, split="heldout", **kwargs)
        except ValueError:
            continue  # no held-out rows in this manifest
        heldout_sets[name] = heldout_set
        heldout[name] = {"samples": len(heldout_set), "overall": evaluate_image_model(
            model, DataLoader(heldout_set, batch_size=batch, shuffle=False, num_workers=0), device=device),
            "by_source": _evaluate_by_source(model, heldout_set, batch=batch, device=device)}
    results: dict[str, Any] = {
        "training_phases": phases, "temporal_stage_history": [], "temporal_validation": None, "forgetting_check": None,
        "image_stage_history": histories.get("images", []), "video_frame_stage_history": histories.get("video_frames", []),
        "image_validation": final.get("images"), "video_frame_validation": final.get("video_frames"),
        # What the encoder scored on loose images right after the video phase, before any image was trained on.
        "image_validation_after_stage_a": before_images,
        "video_frame_validation_after_video_phase": after_phase.get("video_frames"),
        "image_validation_after_image_phase": after_phase.get("images"),
        "epoch_selection": selection,
        "epoch_selection_note": ("The reported validation is the one the epoch was chosen on, so it is slightly optimistic. "
                                 "Held-out sources and the untouched test split are never used for selection."),
        "validation_by_source": by_source, "heldout": heldout or None,
        "validation_samples": _validation_samples(data, heldout_sets),
        "augmentation": {"enabled": augment, "applied_to": "train split only", "seeded": True}}
    samples = {"image_train": len(data["images"]["train"]) if "images" in data else 0,
               "image_validation": len(data["images"]["val"]) if "images" in data else 0,
               "video_frames_train": len(data["video_frames"]["train"]) if "video_frames" in data else 0,
               "video_frames_validation": len(data["video_frames"]["val"]) if "video_frames" in data else 0,
               "windows_train": 0, "windows_validation": 0}
    return {"model": model, "epochs": {"image_epochs": epochs["images"] if "images" in data else 0,
                                       "video_epochs": epochs["video_frames"] if "video_frames" in data else 0, "temporal_epochs": 0},
            "samples": samples,
            "class_weights": {"image": dict(zip(LABELS, weights["images"].tolist())) if "images" in weights else None,
                              "video_frames": dict(zip(LABELS, weights["video_frames"].tolist())) if "video_frames" in weights else None,
                              "temporal": None},
            "results": results}


def run_train_candidate(inputs: dict[str, Any], *, device: str, options: dict[str, Any], output_dir: Path,
                        archive_path: Path, code_fingerprint: str, job_context: dict[str, Any] | None = None,
                        seed: int = 20261026, parent_bundle_dir: Path | None = None) -> dict[str, Any]:
    contract_test = bool(options.get("contract_test"))
    image_only = options.get("branch", "temporal") == "image_only"
    if (inputs["report"].get("branch", "temporal") == "image_only") != image_only:
        raise RuntimeError("training inputs were prepared for a different branch than the requested options")
    if inputs["blockers"]:
        raise RuntimeError("training inputs are blocked: " + ", ".join(inputs["blockers"]))
    if device != "cuda" and not contract_test:
        raise RuntimeError("Candidate training requires the Kaggle GPU runtime; no CPU fallback is allowed")
    import torch

    from drowning.inference.aggregation import DEFAULT_AGGREGATION
    from drowning.ml.bundle import save_bundle

    image_recipe, temporal_recipe = inputs["recipes"]
    input_cfg = image_recipe["input"]
    random.seed(seed)
    torch.manual_seed(seed)
    # Same inputs must give the same weights: fixed seeds, deterministic kernels, seeded shuffling.
    # (On GPU this still holds only for the same GPU model and torch/CUDA versions; see metrics["determinism"].)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    started = time.time()
    parent = json.loads((parent_bundle_dir / "manifest.json").read_text(encoding="utf-8")) if parent_bundle_dir else None
    trained = (_train_image_only if image_only else _train_two_stage)(inputs, device=device, options=options, seed=seed,
                                                                    parent_bundle_dir=parent_bundle_dir)
    model = trained["model"]

    recipe_fingerprint = hashlib.sha256(json.dumps([image_recipe, temporal_recipe], sort_keys=True).encode()).hexdigest()
    candidate_id = uuid.uuid4().hex
    sampling = temporal_recipe["sampling"]
    kind = "synthetic_smoke" if contract_test else "candidate"
    output_dir.mkdir(parents=True, exist_ok=False)
    context = job_context or {}
    save_bundle(model, output_dir / "model-bundle", completion_kind=kind, temporal_trained=not image_only,
                model_kind="image_tracked" if image_only else "temporal",
                aggregation=dict(DEFAULT_AGGREGATION) if image_only else None,
                preprocessing={"size": input_cfg["size"], "context_fraction": input_cfg["crop_context_fraction"],
                               "mean": input_cfg["mean"], "std": input_cfg["std"]},
                dataset_fingerprint=inputs["dataset_fingerprint"], recipe_fingerprint=recipe_fingerprint,
                code_fingerprint=code_fingerprint, training_job_id=context.get("training_job_id"),
                remote_kernel_ref=context.get("kernel_ref"), remote_kernel_version=context.get("kernel_version"),
                parent_model_version=parent.get("model_version_id") if parent else None,
                model_version_id=candidate_id,
                temporal_contract=None if image_only else {
                    "sequence_length": int(sampling["sequence_length"]), "sample_period_ms": int(sampling["sample_period_ms"]),
                    "max_track_gap_ms": int(sampling["max_track_gap_ms"]), "causal": True})
    metrics = build_metrics(
        completion_kind=kind, candidate_id=candidate_id, dataset_fingerprint=inputs["dataset_fingerprint"],
        model_kind="image_tracked" if image_only else "temporal", branch="image_only" if image_only else "temporal",
        device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu-contract-test",
        pretrained_encoder=bool(options.get("pretrained")), seed=seed, wall_seconds=round(time.time() - started, 1),
        initialization="approved_checkpoint" if parent else ("imagenet" if options.get("pretrained") else "random"),
        parent_model_version=parent.get("model_version_id") if parent else None,
        parent_weights_sha256=parent["files"]["weights.pt"]["sha256"] if parent else None,
        optimizer_reset=True,
        **trained["epochs"], class_weights_from_train_split=trained["class_weights"], samples=trained["samples"],
        **trained["results"],
        determinism={"seed": seed, "deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False,
                     "seeded_shuffle": True, "torch": torch.__version__, "cuda": torch.version.cuda,
                     "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                     "caveat": "Bit-identical retraining is expected only on the same GPU model and library versions."},
        aggregation=(dict(DEFAULT_AGGREGATION) if image_only else None),
        aggregation_note=("Per-track aggregation parameters are unbenchmarked defaults: no labeled video was used to tune or "
                          "validate them, and the temporal head is untrained." if image_only else None),
        input_report=inputs["report"])
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output_dir.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(output_dir).as_posix())
    return metrics


def write_blocked_report(output_dir: Path, archive_path: Path, *, blockers: list[str], report: dict[str, Any],
                         code_fingerprint: str) -> dict[str, Any]:
    """A train run that cannot start still ends cleanly with a readable report instead of a crashed kernel."""
    from drowning.orchestration.kaggle_run import REPORT_NAME, _environment, package_output

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "run_mode": "train_candidate", "train_blocked": True, "blockers": blockers,
               "code_fingerprint": code_fingerprint, "environment": _environment(), "input_report": report,
               "training_started": False, "model_candidate": False}
    (output_dir / REPORT_NAME).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    package_output(output_dir, archive_path)
    return payload
