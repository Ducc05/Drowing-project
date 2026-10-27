"""Create a Kaggle-ready private kernel directory without pushing or running it."""

from __future__ import annotations

import argparse
import json
import os
import hashlib
import shutil
from pathlib import Path

from drowning.config import load_env_file
from .build_notebook import build_notebook

ROOT = Path(__file__).resolve().parents[1]


def materialize_kernel(kernel_ref: str, output_dir: Path, *, mode: str = "prepare_only",
                       dataset_sources: list[str] | None = None, options: dict | None = None,
                       parent_bundle_dir: Path | None = None, parent_model: dict | None = None,
                       parent_dataset_ref: str | None = None, prepare_cache: dict | None = None) -> Path:
    if not kernel_ref or "/" not in kernel_ref or len(kernel_ref.split("/")) != 2:
        raise ValueError("KAGGLE_KERNEL_REF must be configured as owner/slug")
    if mode not in {"prepare_only", "remote_smoke", "train_candidate"}:
        raise ValueError("unsupported notebook mode")
    target = output_dir.resolve()
    try:
        target.relative_to(Path("D:/DrowningDemo").resolve())
        protected_demo = True
    except ValueError:
        protected_demo = False
    if protected_demo or any(part.casefold() == "sources" for part in target.parts):
        raise ValueError("Protected D:\\DrowningDemo or sources/ output path")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing Kaggle kernel directory: {target}")
    if parent_bundle_dir is not None:
        from drowning.ml.bundle import load_bundle
        if not parent_model or not parent_dataset_ref:
            raise ValueError("parent checkpoint staging requires a pinned identity and private dataset reference")
        loaded, manifest = load_bundle(parent_bundle_dir, require_temporal=True, map_location="cpu")
        del loaded
        if (str(manifest["model_version_id"]).replace("-", "") != parent_model["id"].replace("-", "")
                or manifest["files"]["weights.pt"]["sha256"] != parent_model["weights_sha256"]):
            raise ValueError("parent checkpoint changed after it was pinned")
        staging = target.parent / (target.name + "-parent")
        staging.mkdir(parents=True, exist_ok=False)
        for name in ("manifest.json", "weights.pt"):
            shutil.copyfile(parent_bundle_dir / name, staging / name)
        if (hashlib.sha256((staging / "manifest.json").read_bytes()).hexdigest() != parent_model["manifest_sha256"]
                or hashlib.sha256((staging / "weights.pt").read_bytes()).hexdigest() != parent_model["weights_sha256"]):
            raise ValueError("staged parent checkpoint checksum mismatch")
        (staging / "dataset-metadata.json").write_text(json.dumps({
            "id": parent_dataset_ref, "title": "Drowning parent " + parent_model["id"][:8],
            "description": "Private model checkpoint artifact for this DrowningProject training job.",
            "licenses": [{"name": "other"}]}), encoding="utf-8")
    notebook = build_notebook(mode, options, parent_dataset_ref=parent_dataset_ref, parent_model=parent_model,
                              prepare_cache=prepare_cache)
    metadata_template = json.loads((ROOT / "notebooks" / "kernel-metadata.template.json").read_text(encoding="utf-8"))
    metadata_template["id"] = kernel_ref
    metadata_template["code_file"] = "Drowning_Temporal_Kaggle.ipynb"
    metadata_template["dataset_sources"] = list(dataset_sources or []) + ([parent_dataset_ref] if parent_dataset_ref else [])
    target.mkdir(parents=True)
    (target / metadata_template["code_file"]).write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    (target / "kernel-metadata.json").write_text(json.dumps(metadata_template, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize local Kaggle kernel files; this does not push or run them")
    parser.add_argument("--kernel-ref", default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "notebooks" / "kernels" / "drowning-temporal")
    parser.add_argument("--mode", choices=("prepare_only", "remote_smoke", "train_candidate"), default="prepare_only")
    parser.add_argument("--dataset-source", action="append", default=[])
    args = parser.parse_args()
    load_env_file(Path.cwd() / ".env")
    kernel_ref = args.kernel_ref or os.environ.get("KAGGLE_KERNEL_REF", "")
    path = materialize_kernel(kernel_ref, args.output_dir, mode=args.mode, dataset_sources=args.dataset_source)
    print(json.dumps({"kernel_directory": str(path), "mode": args.mode,
                      "private": True, "remote_action": "none"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
