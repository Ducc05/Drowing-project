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
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from drowning.data.adapters import convert_to_jsonl
from drowning.data.audit import audit_jsonl
from drowning.data.crops import build_crops
from drowning.data.grouping import SplitBlocked, build_group_split
from drowning.data.registry import SourceConfigError, load_source, validate_source
from drowning.data.review import is_mapping_approved, mapping_fingerprint
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
                            recipes_dir: Path, seed: int = 20261026) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    image_recipe = yaml.safe_load((recipes_dir / "image.yaml").read_text(encoding="utf-8"))
    temporal_recipe = yaml.safe_load((recipes_dir / "temporal.yaml").read_text(encoding="utf-8"))
    sampling, image_input = temporal_recipe["sampling"], image_recipe["input"]
    included: list[dict[str, Any]] = []
    excluded: dict[str, str] = {}
    for path in sorted(sources_dir.glob("*.yaml")):
        source = load_source(path, reviews_dir)
        source_id = str(source.get("source_id"))
        if validate_source(source):
            excluded[source_id] = "invalid_descriptor"
            continue
        found = discover_source_root(source, input_root)
        if found["status"] != "found":
            excluded[source_id] = f"input_{found['status']}"
            continue
        if not is_mapping_approved(source):
            excluded[source_id] = "mapping_review_not_approved"
            continue
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
        if not media or any(m.get("grouping_status") != "verified" or not m.get("global_group_id") for m in media.values()):
            excluded[source_id] = "source_groups_not_verified"
            continue
        included.append({"source": source, "root": found["root"], "canonical": canonical, "media": media})

    blockers: list[str] = []
    report: dict[str, Any] = {"schema_version": 1, "seed": seed, "excluded_sources": excluded,
                              "included_sources": [i["source"]["source_id"] for i in included]}
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
    split_path = work_dir / "split_manifest.jsonl"
    _write_jsonl(split_path, split["manifest"])
    report["split_media_counts"] = split["media_counts_by_split"]
    report["source_video_disjointness_verified"] = False
    report["split_basis"] = "reviewer-verified groups; the source video of loose images remains unknown"

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

    # -- temporal windows -------------------------------------------------------------------------
    assignments = {row["media_id"]: {"split": row["split"], "global_group_id": row["global_group_id"]} for row in split["manifest"]}
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
    image_cov: dict[str, Counter] = {s: Counter() for s in ("train", "validation", "test")}
    for row in crop_rows:
        if row.get("training_eligible") is True:
            image_cov[row["split"]][row["canonical_label"]] += 1
    window_cov: dict[str, Counter] = {s: Counter() for s in ("train", "validation", "test")}
    for window in windows:
        window_cov[window["split"]][window["label"]] += 1
    report["image_crops_by_split"] = {s: dict(c) for s, c in image_cov.items()}
    report["temporal_windows_by_split"] = {s: dict(c) for s, c in window_cov.items()}
    report["temporal_window_issues"] = sorted({issue["code"] for issue in window_report["issues"]})
    if not crop_rows:
        blockers.append("no_image_crops")
    if not windows:
        blockers.append("no_temporal_windows")
    for split_name in SPLITS_NEEDING_ALL_CLASSES:
        for kind, coverage in (("image", image_cov), ("temporal", window_cov)):
            missing = sorted(set(LABELS) - set(coverage[split_name]))
            if missing and (kind == "image" and crop_rows or kind == "temporal" and windows):
                blockers.append(f"{kind}_{split_name}_split_missing_classes:{','.join(missing)}")
    hasher = hashlib.sha256()
    for part in (merged, split_path, crop_manifest, window_manifest, recipes_dir / "image.yaml", recipes_dir / "temporal.yaml"):
        hasher.update(sha256_file(part).encode())
    for item in sorted(included, key=lambda i: i["source"]["source_id"]):
        hasher.update(f"{item['source']['source_id']}:{mapping_fingerprint(item['source'])}".encode())
    fingerprint = hasher.hexdigest()
    report["dataset_fingerprint"] = fingerprint
    report["blockers"] = blockers
    return {"blockers": blockers, "report": report, "dataset_fingerprint": fingerprint, "recipes": (image_recipe, temporal_recipe),
            "paths": {"merged": merged, "split": split_path, "crop_manifest": crop_manifest, "window_manifest": window_manifest},
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


def run_train_candidate(inputs: dict[str, Any], *, device: str, options: dict[str, Any], output_dir: Path,
                        archive_path: Path, code_fingerprint: str, job_context: dict[str, Any] | None = None,
                        seed: int = 20261026) -> dict[str, Any]:
    contract_test = bool(options.get("contract_test"))
    if inputs["blockers"]:
        raise RuntimeError("training inputs are blocked: " + ", ".join(inputs["blockers"]))
    if device != "cuda" and not contract_test:
        raise RuntimeError("Candidate training requires the Kaggle GPU runtime; no CPU fallback is allowed")
    import torch
    from torch.utils.data import DataLoader

    from drowning.ml.bundle import save_bundle
    from drowning.ml.datasets import CanonicalImageCropDataset, TemporalVideoWindowDataset
    from drowning.ml.model import DrowningTemporalModel
    from drowning.ml.training import (evaluate_image_model, evaluate_temporal_model, train_stage_a_epoch,
                                      train_stage_b_epoch)

    image_recipe, temporal_recipe = inputs["recipes"]
    paths, input_cfg = inputs["paths"], image_recipe["input"]
    size = int(input_cfg["size"][0])
    random.seed(seed)
    torch.manual_seed(seed)
    # Same inputs must give the same weights: fixed seeds, deterministic kernels, seeded shuffling.
    # (On GPU this still holds only for the same GPU model and torch/CUDA versions; see metrics["determinism"].)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    started = time.time()
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
        model = DrowningTemporalModel(hidden_size=int(temporal_recipe["model"]["hidden_size"]),
                                      pretrained_encoder=bool(options.get("pretrained"))).to(device)
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
    finally:
        window_train.close()
        window_val.close()

    recipe_fingerprint = hashlib.sha256(json.dumps([image_recipe, temporal_recipe], sort_keys=True).encode()).hexdigest()
    candidate_id = uuid.uuid4().hex
    sampling = temporal_recipe["sampling"]
    kind = "synthetic_smoke" if contract_test else "candidate"
    output_dir.mkdir(parents=True, exist_ok=False)
    context = job_context or {}
    save_bundle(model, output_dir / "model-bundle", completion_kind=kind, temporal_trained=True,
                preprocessing={"size": input_cfg["size"], "context_fraction": input_cfg["crop_context_fraction"],
                               "mean": input_cfg["mean"], "std": input_cfg["std"]},
                dataset_fingerprint=inputs["dataset_fingerprint"], recipe_fingerprint=recipe_fingerprint,
                code_fingerprint=code_fingerprint, training_job_id=context.get("training_job_id"),
                remote_kernel_ref=context.get("kernel_ref"), remote_kernel_version=context.get("kernel_version"),
                model_version_id=candidate_id,
                temporal_contract={"sequence_length": int(sampling["sequence_length"]), "sample_period_ms": int(sampling["sample_period_ms"]),
                                   "max_track_gap_ms": int(sampling["max_track_gap_ms"]), "causal": True})
    metrics = build_metrics(
        completion_kind=kind, candidate_id=candidate_id, dataset_fingerprint=inputs["dataset_fingerprint"],
        device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu-contract-test",
        pretrained_encoder=bool(options.get("pretrained")), seed=seed, wall_seconds=round(time.time() - started, 1),
        image_epochs=image_epochs, temporal_epochs=temporal_epochs,
        class_weights_from_train_split={"image": dict(zip(LABELS, image_weights.tolist())), "temporal": dict(zip(LABELS, temporal_weights.tolist()))},
        samples={"image_train": len(image_train), "image_validation": len(image_val),
                 "windows_train": len(window_train), "windows_validation": len(window_val)},
        image_stage_history=stage_a, temporal_stage_history=stage_b,
        image_validation_after_stage_a=image_after_a, image_validation=image_after_b, temporal_validation=temporal_val,
        determinism={"seed": seed, "deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False,
                     "seeded_shuffle": True, "torch": torch.__version__, "cuda": torch.version.cuda,
                     "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                     "caveat": "Bit-identical retraining is expected only on the same GPU model and library versions."},
        forgetting_check="image_validation_after_stage_a vs image_validation shows what stage B did to the image head",
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
