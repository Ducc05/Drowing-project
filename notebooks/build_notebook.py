"""Build a private Kaggle notebook that embeds the current core source bundle.

The notebook is thin: every step lives in ``drowning.orchestration.kaggle_run`` /
``kaggle_train`` (tested locally) and the cells only call them.
"""

from __future__ import annotations

import argparse
import base64
import json
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ("prepare_only", "remote_smoke", "train_candidate")
BRANCHES = ("temporal", "image_only")
# The web UI only trains the image head (video frames first, then images); "temporal" stays reachable through the API.
DEFAULT_OPTIONS = {"pretrained": False, "image_epochs": 5, "temporal_epochs": 5, "branch": "image_only",
                   "video_epochs": 3, "frame_stride_ms": 500, "max_frames_per_video": 300,
                   "continue_from_approved": False, "source_ids": None,
                   "augment": True, "patience": 3, "holdout_sources": None}


_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
CRLF, LF = bytes([13, 10]), bytes([10])


def _add_deterministic(archive: zipfile.ZipFile, path: Path, name: str) -> None:
    """Fixed timestamps/permissions and LF line endings, so the bundle hash is the same on every machine."""
    data = path.read_bytes()
    if path.suffix in {".py", ".yaml", ".json"}:
        data = data.replace(CRLF, LF)
    info = zipfile.ZipInfo(name, date_time=_FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def source_bundle() -> bytes:
    memory = BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / "src" / "drowning").rglob("*.py")):
            _add_deterministic(archive, path, path.relative_to(ROOT).as_posix())
        for path in sorted((ROOT / "configs" / "recipes").glob("*.yaml")):
            _add_deterministic(archive, path, path.relative_to(ROOT).as_posix())
        for path in sorted((ROOT / "configs" / "datasets").glob("*.yaml")):
            _add_deterministic(archive, path, path.relative_to(ROOT).as_posix())
        _add_deterministic(archive, ROOT / "configs" / "ontology.yaml", "configs/ontology.yaml")
        _add_deterministic(archive, ROOT / "configs" / "review_policy.yaml", "configs/review_policy.yaml")
        # Approvals are bound to the mapping/group hashes they reviewed, so shipping them is safe.
        for path in sorted((ROOT / "reviews").glob("*.json")) if (ROOT / "reviews").is_dir() else []:
            _add_deterministic(archive, path, path.relative_to(ROOT).as_posix())
    return memory.getvalue()


def source_bundle_sha256() -> str:
    import hashlib
    return hashlib.sha256(source_bundle()).hexdigest()


def normalize_options(options: dict | None) -> dict:
    merged = {**DEFAULT_OPTIONS, **(options or {})}
    unknown = set(merged) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError(f"unknown run options: {sorted(unknown)}")
    if not isinstance(merged["pretrained"], bool):
        raise ValueError("pretrained must be true or false")
    if not isinstance(merged["augment"], bool):
        raise ValueError("augment must be true or false")
    if not isinstance(merged["patience"], int) or isinstance(merged["patience"], bool) or not 0 <= merged["patience"] <= 20:
        raise ValueError("patience must be an integer between 0 (never stop early) and 20")
    holdout = merged["holdout_sources"]
    if holdout is not None:
        if not isinstance(holdout, list) or not holdout or any(not isinstance(s, str) or not s.strip() or len(s) > 160 for s in holdout):
            raise ValueError("holdout_sources must be null or a nonempty list of source IDs")
        merged["holdout_sources"] = sorted(set(holdout))
        if merged["source_ids"] is not None and set(holdout) - set(merged["source_ids"]):
            raise ValueError("holdout_sources must be among source_ids")
    if not isinstance(merged["continue_from_approved"], bool):
        raise ValueError("continue_from_approved must be true or false")
    if merged["continue_from_approved"] and merged["pretrained"]:
        raise ValueError("continuation already initializes from the approved checkpoint; disable ImageNet initialization")
    sources = merged["source_ids"]
    if sources is not None:
        if not isinstance(sources, list) or not sources or any(not isinstance(s, str) or not s.strip() or len(s) > 160 for s in sources):
            raise ValueError("source_ids must be null (all sources) or a nonempty list of source IDs")
        merged["source_ids"] = sorted(set(sources))
    if merged["branch"] not in BRANCHES:
        raise ValueError(f"branch must be one of {', '.join(BRANCHES)}")
    for key in ("video_epochs",):
        if not isinstance(merged[key], int) or isinstance(merged[key], bool) or not 1 <= merged[key] <= 50:
            raise ValueError(f"{key} must be an integer between 1 and 50")
    if (not isinstance(merged["frame_stride_ms"], int) or isinstance(merged["frame_stride_ms"], bool)
            or not 100 <= merged["frame_stride_ms"] <= 5000):
        raise ValueError("frame_stride_ms must be an integer between 100 and 5000")
    if (not isinstance(merged["max_frames_per_video"], int) or isinstance(merged["max_frames_per_video"], bool)
            or not 10 <= merged["max_frames_per_video"] <= 5000):
        raise ValueError("max_frames_per_video must be an integer between 10 and 5000")
    for key in ("image_epochs", "temporal_epochs"):
        if not isinstance(merged[key], int) or isinstance(merged[key], bool) or not 1 <= merged[key] <= 50:
            raise ValueError(f"{key} must be an integer between 1 and 50")
    return merged


def build_notebook(run_mode: str = "prepare_only", options: dict | None = None, *,
                   parent_dataset_ref: str | None = None, parent_model: dict | None = None) -> dict:
    if run_mode not in MODES:
        raise ValueError("run_mode must be prepare_only, remote_smoke, or train_candidate")
    run_options = normalize_options(options)
    if run_options["continue_from_approved"]:
        if run_mode != "train_candidate" or not parent_dataset_ref or not parent_model:
            raise ValueError("continuation requires a pinned approved checkpoint")
    elif parent_dataset_ref is not None or parent_model is not None:
        raise ValueError("parent checkpoint supplied for a fresh training run")
    payload = base64.b64encode(source_bundle()).decode("ascii")
    setup_cell = f'''# Generated from the current repository source. Do not edit the embedded bundle by hand.
import base64, hashlib, io, os, sys, zipfile, importlib.util, json
from pathlib import Path

SOURCE_BUNDLE_B64 = {payload!r}
SOURCE_BUNDLE_SHA256 = hashlib.sha256(base64.b64decode(SOURCE_BUNDLE_B64)).hexdigest()
# Code goes to scratch space so only the run output ends up in /kaggle/working.
WORK_DIR = Path(os.environ.get("DROWN_CODE_DIR", "/kaggle/temp/drowning-project"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
INPUT_ROOT = Path(os.environ.get("DROWN_INPUT_ROOT", "/kaggle/input"))
OUTPUT_DIR = Path(os.environ.get("DROWN_OUTPUT_DIR", "/kaggle/working/run-output"))
SCRATCH_DIR = Path(os.environ.get("DROWN_SCRATCH_DIR", "/kaggle/temp/derived"))
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_BUNDLE_B64))) as archive:
    archive.extractall(WORK_DIR)
sys.path.insert(0, str(WORK_DIR / "src"))
import subprocess
for _module, _package in (("yaml", "PyYAML"), ("PIL", "Pillow"), ("defusedxml", "defusedxml")):
    if importlib.util.find_spec(_module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _package], check=False)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic CUDA matmuls
RUN_MODE = os.environ.get("DROWN_RUN_MODE", {run_mode!r})
RUN_OPTIONS = {run_options!r}
PARENT_MODEL = {parent_model!r}
PARENT_DATASET_REF = {parent_dataset_ref!r}
PARENT_MODEL_DIR = None
if RUN_OPTIONS.get("continue_from_approved"):
    _slug = PARENT_DATASET_REF.split("/")[1]
    _matches = [p for p in INPUT_ROOT.rglob("manifest.json") if _slug in p.parts]
    if len(_matches) != 1:
        raise RuntimeError("Expected exactly one attached parent checkpoint manifest")
    _manifest_path = _matches[0]
    if hashlib.sha256(_manifest_path.read_bytes()).hexdigest() != PARENT_MODEL["manifest_sha256"]:
        raise RuntimeError("Pinned parent manifest checksum mismatch")
    _manifest = json.loads(_manifest_path.read_text(encoding="utf-8"))
    if str(_manifest["model_version_id"]).replace("-", "") != PARENT_MODEL["id"].replace("-", ""):
        raise RuntimeError("Pinned parent model identity mismatch")
    PARENT_MODEL_DIR = _manifest_path.parent
    if hashlib.sha256((PARENT_MODEL_DIR / "weights.pt").read_bytes()).hexdigest() != PARENT_MODEL["weights_sha256"]:
        raise RuntimeError("Pinned parent weights checksum mismatch")
assert RUN_MODE in {{"prepare_only", "remote_smoke", "train_candidate"}}, RUN_MODE
print({{"run_mode": RUN_MODE, "options": RUN_OPTIONS, "default_mode": "prepare_only", "private_kernel": True}})
'''
    environment_cell = '''import json
from drowning.orchestration.kaggle_run import _environment
print(json.dumps(_environment(), indent=2))
'''
    run_cell = '''from drowning.orchestration.kaggle_run import package_output, run_prepare, run_smoke
RUN_ARCHIVE = Path(os.environ.get("DROWN_ARCHIVE_PATH", "/kaggle/working/run-output.zip"))
if RUN_MODE == "prepare_only":
    run_report = run_prepare(sources_dir=WORK_DIR / "configs" / "datasets", reviews_dir=WORK_DIR / "reviews",
                             input_root=INPUT_ROOT, output_dir=OUTPUT_DIR, work_dir=SCRATCH_DIR,
                             code_fingerprint=SOURCE_BUNDLE_SHA256)
    package_output(OUTPUT_DIR, RUN_ARCHIVE)
    print(json.dumps({"run_mode": RUN_MODE, "blockers": run_report["blockers"],
                      "sources": [{key: entry.get(key) for key in ("source_id", "status", "mapping_review", "grouping_review")}
                                  for entry in run_report["sources"]],
                      "archive": str(RUN_ARCHIVE), "training_started": False}, ensure_ascii=False, indent=2))
elif RUN_MODE == "remote_smoke":
    run_report = run_smoke(output_dir=OUTPUT_DIR, code_fingerprint=SOURCE_BUNDLE_SHA256)
    package_output(OUTPUT_DIR, RUN_ARCHIVE)
    print(json.dumps({key: run_report[key] for key in ("device", "gpu_confirmed", "model_candidate")}, indent=2))
else:
    print("train_candidate runs in the next cell")
'''
    train_cell = '''if RUN_MODE == "train_candidate":
    import torch
    from drowning.orchestration.kaggle_train import prepare_training_inputs, run_train_candidate, write_blocked_report
    inputs = prepare_training_inputs(sources_dir=WORK_DIR / "configs" / "datasets", reviews_dir=WORK_DIR / "reviews",
                                     input_root=INPUT_ROOT, work_dir=SCRATCH_DIR, recipes_dir=WORK_DIR / "configs" / "recipes",
                                     branch=RUN_OPTIONS.get("branch", "image_only"),
                                     frame_stride_ms=RUN_OPTIONS.get("frame_stride_ms", 500),
                                     max_frames_per_video=RUN_OPTIONS.get("max_frames_per_video", 300),
                                     only_sources=set(RUN_OPTIONS["source_ids"]) if RUN_OPTIONS.get("source_ids") else None,
                                     holdout_sources=RUN_OPTIONS.get("holdout_sources"))
    if Path("/kaggle/working").is_dir():
        Path("/kaggle/working/train-input-report.json").write_text(
            json.dumps(inputs["report"], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(inputs["report"], ensure_ascii=False, indent=2, default=str)[:8000])
    if inputs["blockers"]:
        # End cleanly with a readable report: a crashed kernel would hide why nothing was trained.
        write_blocked_report(OUTPUT_DIR, RUN_ARCHIVE, blockers=inputs["blockers"], report=inputs["report"],
                             code_fingerprint=SOURCE_BUNDLE_SHA256)
        print("TRAINING NOT STARTED:", inputs["blockers"])
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Candidate training requires the selected Kaggle GPU runtime; no CPU fallback is allowed")
        metrics = run_train_candidate(inputs, device="cuda", options=RUN_OPTIONS,
                                      parent_bundle_dir=PARENT_MODEL_DIR,
                                      output_dir=Path("/kaggle/working/drowning-candidate"),
                                      archive_path=Path("/kaggle/working/candidate-output.zip"),
                                      code_fingerprint=SOURCE_BUNDLE_SHA256,
                                      job_context={"kernel_ref": os.environ.get("KAGGLE_KERNEL_ID"),
                                                   "kernel_version": os.environ.get("KAGGLE_KERNEL_VERSION")})
        print(json.dumps({"candidate_archive": "/kaggle/working/candidate-output.zip", "approved": False,
                          "image_validation": metrics["image_validation"], "temporal_validation": metrics.get("temporal_validation")},
                         ensure_ascii=False, indent=2, default=str))
'''

    def code(source: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}

    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Drowning temporal Kaggle worker\n",
            "Private GPU notebook. Default mode is `prepare_only`; this notebook never silently trains.\n",
            "`remote_smoke` runs the real model on random tensors to confirm the GPU. `train_candidate` uses only sources whose "
            "label mapping and source groups a person approved, and never touches the test split.\n",
        ]},
        code(setup_cell), code(environment_cell), code(run_cell), code(train_cell),
    ]
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.11"},
                     "drowning_project": {"run_mode_default": run_mode, "private": True, "training_executor": "kaggle",
                                          "parent_model": parent_model,
                                          "parent_dataset_ref": parent_dataset_ref,
                                          "options": run_options}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks" / "Drowning_Temporal_Kaggle.ipynb")
    parser.add_argument("--mode", choices=MODES, default="prepare_only")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--branch", choices=BRANCHES, default="image_only")
    parser.add_argument("--image-epochs", type=int, default=DEFAULT_OPTIONS["image_epochs"])
    parser.add_argument("--temporal-epochs", type=int, default=DEFAULT_OPTIONS["temporal_epochs"])
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook(args.mode, {"pretrained": args.pretrained, "image_epochs": args.image_epochs,
                                          "temporal_epochs": args.temporal_epochs})
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
