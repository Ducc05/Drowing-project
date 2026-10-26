"""Create a Kaggle-ready private kernel directory without pushing or running it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from drowning.config import load_env_file
from .build_notebook import build_notebook

ROOT = Path(__file__).resolve().parents[1]


def materialize_kernel(kernel_ref: str, output_dir: Path, *, mode: str = "prepare_only",
                       dataset_sources: list[str] | None = None, options: dict | None = None) -> Path:
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
    notebook = build_notebook(mode, options)
    metadata_template = json.loads((ROOT / "notebooks" / "kernel-metadata.template.json").read_text(encoding="utf-8"))
    metadata_template["id"] = kernel_ref
    metadata_template["code_file"] = "Drowning_Temporal_Kaggle.ipynb"
    metadata_template["dataset_sources"] = dataset_sources or []
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
