"""Work performed inside a Kaggle notebook run, written as a plain library so it can be tested locally.

``run_prepare`` inspects every registered source found under the attached inputs (scan, convert,
audit, review sheets, quality report) and never trains. ``run_smoke`` is a synthetic GPU/dependency
check whose result is never a model candidate. Both write a small, secret-free output package.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from drowning.data.adapters import convert_to_jsonl, scan_source
from drowning.data.audit import audit_jsonl
from drowning.data.labels import normalize_label
from drowning.data.quality import build_quality_report
from drowning.data.registry import SourceConfigError, load_source, read_class_names, validate_source
from drowning.data.windows import build_temporal_windows
from drowning.data.review import is_mapping_approved, load_grouping_review, mapping_fingerprint
from drowning.data.review_rules import gating_mode
from drowning.data.samples import build_review_sheets

REPORT_NAME = "run-report.json"
PACKAGE_NAME = "run-output.zip"
MAX_DISCOVERY_DEPTH = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marker_file(source: dict[str, Any]) -> str | None:
    paths = source.get("paths", {})
    spec = paths.get("class_list")
    if isinstance(spec, str) and ":" in spec:
        return spec.split(":", 1)[0]
    for key in ("annotations", "video", "media"):
        if isinstance(paths.get(key), str):
            return paths[key]
    tasks = source.get("tasks")
    if isinstance(tasks, list) and tasks and isinstance(tasks[0], dict) and isinstance(tasks[0].get("annotations"), str):
        return tasks[0]["annotations"]
    return None


def discover_source_root(source: dict[str, Any], input_root: Path) -> dict[str, Any]:
    """Find the attached dataset directory for a source without guessing between look-alikes.

    A candidate is a directory under ``input_root`` that contains the source's marker file (for
    example ``data.yaml``). When ``source_labels.expected_names`` is declared, a candidate must
    declare exactly those class names. Zero or several remaining candidates are reported, not chosen.
    """
    marker = _marker_file(source)
    if marker is None:
        return {"status": "no_marker_in_descriptor", "root": None, "candidates": []}
    base_depth = len(input_root.parts)
    candidates: list[Path] = []
    for path in sorted(input_root.rglob(Path(marker).name)):
        directory = path.parent
        for _ in range(len(Path(marker).parts) - 1):
            directory = directory.parent
        if len(directory.parts) - base_depth <= MAX_DISCOVERY_DEPTH and (directory / marker).is_file() and directory not in candidates:
            candidates.append(directory)
    expected = source.get("source_labels", {}).get("expected_names")
    if expected and source.get("paths", {}).get("class_list"):
        wanted = sorted(normalize_label(name) for name in expected)
        matching = []
        for directory in candidates:
            try:
                names = sorted(normalize_label(name) for _, name in read_class_names(source, directory))
            except (SourceConfigError, OSError, ValueError):
                continue
            if names == wanted:
                matching.append(directory)
        candidates = matching
    relative = [directory.relative_to(input_root).as_posix() or "." for directory in candidates]
    if len(candidates) == 1:
        return {"status": "found", "root": candidates[0], "candidates": relative}
    return {"status": "ambiguous" if candidates else "missing", "root": None, "candidates": relative}


def _suggestions(label: str) -> list[str]:
    """Spelling-level hints for a reviewer only; never used to map anything automatically."""
    import difflib
    keys = ["drowning", "swimming", "out of water"]
    return difflib.get_close_matches(normalize_label(label), keys, n=1, cutoff=0.7)


def _environment() -> dict[str, Any]:
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["torch"] = None
    for module in ("cv2", "PIL", "yaml", "defusedxml"):
        try:
            imported = __import__(module)
            info[module] = getattr(imported, "__version__", "present")
        except ImportError:
            info[module] = None
    return info


def _temporal_summary(canonical: Path, recipe_path: Path | None) -> dict[str, Any]:
    """Window counts for a video source, computed without any split (so never training-eligible)."""
    import yaml
    sampling = {"sequence_length": 16, "sample_period_ms": 250, "max_track_gap_ms": 1000, "min_valid_ratio": 0.75}
    if recipe_path is not None and recipe_path.is_file():
        sampling.update(yaml.safe_load(recipe_path.read_text(encoding="utf-8")).get("sampling", {}))
    report = build_temporal_windows(canonical, sequence_length=int(sampling["sequence_length"]),
                                    sample_period_ms=int(sampling["sample_period_ms"]),
                                    max_track_gap_ms=int(sampling["max_track_gap_ms"]),
                                    min_valid_ratio=float(sampling["min_valid_ratio"]), split_assignments=None)
    labels: dict[str, int] = {}
    for window in report["windows"]:
        labels[window["label"]] = labels.get(window["label"], 0) + 1
    return {"sampling": sampling, "temporal_media_count": report["temporal_media_count"],
            "verified_track_count": report["verified_track_count"], "window_count": report["window_count"],
            "windows_by_label": dict(sorted(labels.items())), "issue_count": len(report["issues"]),
            "issue_codes": sorted({issue["code"] for issue in report["issues"]}), "blockers": report["blockers"],
            "training_eligible_window_count": 0,
            "note": "Counted without a split manifest; windows become training-eligible only after verified groups are split."}


def _prepare_source(source: dict[str, Any], root: Path, source_dir: Path, work_dir: Path,
                    reviews_dir: Path, recipe_path: Path | None = None) -> dict[str, Any]:
    source_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    scan = scan_source(source, root)
    (source_dir / "scan.json").write_text(json.dumps(scan, ensure_ascii=False, indent=2), encoding="utf-8")
    canonical = work_dir / f"{source['source_id']}@{source['source_version']}.jsonl"
    convert_to_jsonl(source, root, canonical)
    audit = audit_jsonl([canonical])
    (source_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = build_review_sheets(source, canonical, root, source_dir / "sheets")
    report, proposed = build_quality_report(source, canonical, root, reviews_dir=reviews_dir)
    (source_dir / "quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if proposed:
        (source_dir / "proposed-groups.json").write_text(json.dumps({
            "method": "near_duplicate_clusters_user_reviewed", "evidence": report["near_duplicates"],
            "by_path": proposed}, ensure_ascii=False), encoding="utf-8")
    mapping_rows = [{
        "original_label": label["original_label"], "original_label_id": label["original_label_id"],
        "action": label["mapping_action"], "canonical_label": label["canonical_label"], "count": label["count"],
        "spelling_suggestions_not_applied": _suggestions(str(label["original_label"])) if label["mapping_action"] == "review" else []}
        for label in report["labels"]]
    (source_dir / "mapping-report.json").write_text(json.dumps({
        "source_id": source["source_id"], "mapping_hash": mapping_fingerprint(source), "labels": mapping_rows,
        "review_approved": is_mapping_approved(source)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    result.update({
        "scan": {k: scan[k] for k in ("media_count", "annotation_count", "malformed_annotation_count")},
        "audit": {"valid": audit["valid"], "issue_count": len(audit["issues"]),
                  "exact_duplicate_groups": len(audit["exact_duplicate_content_groups"]),
                  "same_box_label_conflicts": len(audit["same_box_label_conflicts"])},
        "quality": {"counts": report["counts"], "near_duplicates": {
            k: v for k, v in report["near_duplicates"].items() if k != "cross_split_examples"},
            "readiness": report["image_training_readiness"]},
        "review_sheet_count": len(manifest["sheets"]),
        "canonical_sha256": sha256_file(canonical), "canonical_records": report["counts"]["media"] + report["counts"]["observations"],
        "modality": source.get("modality"), "format": source.get("format"),
    })
    if source.get("modality") == "video":
        result["temporal"] = _temporal_summary(canonical, recipe_path)
        (source_dir / "temporal.json").write_text(json.dumps(result["temporal"], ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run_prepare(*, sources_dir: Path, reviews_dir: Path, input_root: Path, output_dir: Path, work_dir: Path,
                code_fingerprint: str, only_sources: set[str] | None = None) -> dict[str, Any]:
    """Inspect every registered source present under ``input_root``; writes ``run-report.json``."""
    started = datetime.now(timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    advisory = gating_mode(sources_dir.parent / "review_policy.yaml") == "advisory"
    sources: list[dict[str, Any]] = []
    blockers: set[str] = set()
    advisories: set[str] = set()
    for path in sorted(sources_dir.glob("*.yaml")):
        source = load_source(path, reviews_dir)
        entry: dict[str, Any] = {"source_id": source.get("source_id"), "source_version": str(source.get("source_version")),
                                 "descriptor": path.name, "mapping_hash": mapping_fingerprint(source) if isinstance(source.get("mapping"), dict) else None}
        if only_sources is not None and source.get("source_id") not in only_sources:
            continue
        errors = validate_source(source)
        if errors:
            entry.update(status="invalid_descriptor", errors=errors)
            blockers.add(f"{entry['source_id']}:invalid_descriptor")
            sources.append(entry)
            continue
        entry["mapping_review"] = "approved" if is_mapping_approved(source) else "pending"
        entry["grouping_review"] = ("verified" if (load_grouping_review(source, reviews_dir) is not None
                                                    or source.get("grouping", {}).get("status") == "verified") else "pending")
        found = discover_source_root(source, input_root)
        entry["input"] = {"status": found["status"], "candidates": found["candidates"]}
        if found["status"] != "found":
            entry["status"] = f"input_{found['status']}"
            blockers.add(f"{entry['source_id']}:input_{found['status']}")
            sources.append(entry)
            continue
        entry["input"]["root_relative"] = found["candidates"][0]
        try:
            entry.update(_prepare_source(source, found["root"], output_dir / "per-source" / f"{source['source_id']}@{source['source_version']}",
                                         work_dir, reviews_dir, sources_dir.parent / "recipes" / "temporal.yaml"))
            entry["status"] = "prepared"
            for blocker in entry["quality"]["readiness"]["blockers"]:
                if blocker.startswith("source_alone"):
                    continue
                unconfirmed = blocker in {"mapping_review_not_approved", "source_groups_not_verified"}
                (advisories if advisory and unconfirmed else blockers).add(f"{entry['source_id']}:{blocker}")
            if not entry["audit"]["valid"]:
                blockers.add(f"{entry['source_id']}:canonical_audit_failed")
        except Exception as exc:  # recorded, never swallowed: the run report is the evidence
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:500])
            blockers.add(f"{entry['source_id']}:prepare_failed")
        sources.append(entry)
    prepared = [s for s in sources if s["status"] == "prepared"]
    if not sources:
        blockers.add("no_registered_sources")
    elif not prepared:
        blockers.add("no_source_prepared")
    video_sources = [s for s in prepared if s.get("modality") == "video"]
    if not video_sources:
        blockers.add("no_temporal_video_source_prepared")
    temporal_windows = sum(s["temporal"]["window_count"] for s in video_sources)
    if video_sources and temporal_windows == 0:
        blockers.add("no_temporal_windows_meet_the_sampling_contract")
    report = {
        "schema_version": 1, "run_mode": "prepare_only", "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(), "code_fingerprint": code_fingerprint,
        "environment": _environment(), "sources": sources, "temporal_video_sources_prepared": len(video_sources),
        "temporal_window_count_before_split": temporal_windows,
        "blockers": sorted(blockers), "review_mode": "advisory" if advisory else "enforced",
        "advisories": sorted(advisories), "training_started": False, "model_candidate": False,
        "note": "Statistics describe annotations as exported, not model accuracy. Review sheets and group proposals "
                "require a person's approval before any training input is built."}
    (output_dir / REPORT_NAME).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_smoke(*, output_dir: Path, code_fingerprint: str) -> dict[str, Any]:
    """Forward/backward through the real model on random tensors; verifies GPU and dependencies only."""
    import time
    import torch
    from drowning.ml.model import DrowningTemporalModel

    started = datetime.now(timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DrowningTemporalModel(hidden_size=16).to(device)
    frames = torch.randn(2, 4, 3, 64, 64, device=device)
    timer = time.perf_counter()
    output = model(frames)
    loss = output["image_logits"].mean() + output["temporal_logits"].mean()
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    report = {
        "schema_version": 1, "run_mode": "remote_smoke", "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(), "code_fingerprint": code_fingerprint,
        "environment": _environment(), "device": device, "gpu_confirmed": device == "cuda",
        "image_logits_shape": list(output["image_logits"].shape),
        "temporal_logits_shape": list(output["temporal_logits"].shape),
        "all_trainable_parameters_received_gradients": all(g is not None for g in grads),
        "forward_backward_seconds": round(time.perf_counter() - timer, 4),
        "synthetic_input_only": True, "model_candidate": False, "training_started": False}
    (output_dir / REPORT_NAME).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def package_output(output_dir: Path, archive_path: Path) -> Path:
    """Zip ``output_dir`` with forward-slash relative names; ``run-report.json`` must exist."""
    if not (output_dir / REPORT_NAME).is_file():
        raise FileNotFoundError(REPORT_NAME)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir).as_posix())
    return archive_path
