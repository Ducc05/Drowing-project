"""Content-verified per-source prepare reports, portable between Kaggle jobs."""
from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import re
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

MANIFEST = "prepare-cache.json"
SCHEMA = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def _text_sha256(path: Path) -> str:
    # The embedded source bundle normalizes CRLF to LF across Windows/Linux.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_fingerprints(source: dict, root: Path, configs_dir: Path) -> dict[str, str]:
    """Hash bytes, not timestamps, directory names, or a user-maintained version number."""
    resolved_root = root.resolve()
    paths = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix())

    def identify(path: Path) -> tuple[str, str]:
        path.resolve().relative_to(resolved_root)
        if path.is_symlink():
            raise ValueError("Prepare cache does not accept symlinked source files")
        return path.relative_to(root).as_posix(), file_sha256(path)

    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="prepare-fingerprint") as pool:
        data = list(pool.map(identify, paths))
    config = [(p.relative_to(configs_dir).as_posix(), _text_sha256(p))
              for p in sorted(configs_dir.rglob("*.yaml"))
              if p.parent.name == "recipes" or p.name in {"ontology.yaml", "review_policy.yaml"}]
    package = Path(__file__).resolve().parents[1]
    processors = sorted((package / "data").glob("*.py")) + [package / "orchestration" / "kaggle_run.py"]
    versions = {}
    for name in ("Pillow", "numpy", "PyYAML", "defusedxml", "opencv-python", "opencv-python-headless"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"data_fingerprint": _digest(data),
              "configuration_fingerprint": _digest({"source": source, "settings": config}),
              "processor_fingerprint": _digest([(p.relative_to(package).as_posix(), _text_sha256(p)) for p in processors]),
              "runtime_fingerprint": _digest(versions)}
    result["prepare_fingerprint"] = _digest({"schema": SCHEMA, **result})
    return result


def _safe_member(folder: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe prepare cache member")
    target = folder.joinpath(*relative.parts)
    target.resolve().relative_to(folder.resolve())
    if target.is_symlink():
        raise ValueError("Symlinks are not accepted in prepare cache")
    return target


def read_snapshot(folder: Path, fingerprint: str | None = None) -> dict:
    document = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Invalid prepare snapshot document")
    entry, files = document.get("entry"), document.get("files")
    identity = document.get("prepare_fingerprint")
    if not isinstance(entry, dict) or not isinstance(files, dict) or not isinstance(entry.get("audit"), dict):
        raise ValueError("Invalid prepare snapshot contents")
    if (document.get("schema_version") != SCHEMA or not re.fullmatch(r"[a-f0-9]{64}", identity)
            or (fingerprint is not None and identity != fingerprint)
            or entry.get("prepare_fingerprint") != identity or entry.get("status") != "prepared"
            or entry.get("audit", {}).get("valid") is not True
            or not isinstance(files, dict) or len(files) > 5000
            or not {"scan.json", "audit.json", "quality.json", "mapping-report.json"}.issubset(files)):
        raise ValueError("Incomplete or incompatible prepare snapshot")
    for name, expected in files.items():
        path = _safe_member(folder, name)
        if path.suffix.lower() not in {".json", ".png", ".jpg", ".jpeg", ".txt"} or file_sha256(path) != expected:
            raise ValueError("Prepare cache checksum mismatch")
    return document


def write_snapshot(folder: Path, entry: dict) -> None:
    files = {p.relative_to(folder).as_posix(): file_sha256(p)
             for p in sorted(folder.rglob("*")) if p.is_file() and p.name != MANIFEST}
    document = {"schema_version": SCHEMA, "prepare_fingerprint": entry["prepare_fingerprint"],
                "entry": entry, "files": files}
    (folder / MANIFEST).write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def restore_snapshot(cache_dir: Path | None, output_dir: Path, fingerprint: str,
                     source_id: str) -> dict | None:
    if cache_dir is None or not cache_dir.is_dir():
        return None
    for manifest in sorted(cache_dir.rglob(MANIFEST)):
        try:
            # Reject mismatched identities before reading all cached image bytes.
            candidate = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(candidate, dict) or candidate.get("prepare_fingerprint") != fingerprint:
                continue
            document = read_snapshot(manifest.parent, fingerprint)
            if document["entry"].get("source_id") != source_id:
                continue
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            if output_dir.exists():
                raise FileExistsError(output_dir)
            with tempfile.TemporaryDirectory(prefix=".prepare-restore-", dir=output_dir.parent) as staging:
                staged = Path(staging) / "source"
                staged.mkdir()
                for name in [*document["files"], MANIFEST]:
                    target = _safe_member(staged, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(_safe_member(manifest.parent, name), target)
                staged.rename(output_dir)
            entry = dict(document["entry"])
            entry.update(cache_status="reused", cache_origin_job_id=document.get("origin_job_id"))
            return entry
        except (OSError, ValueError, KeyError, TypeError):
            # An absent, old, incomplete or corrupted cache always causes real preparation.
            continue
    return None


def bundle_snapshots(candidates: Iterable[tuple[str, Path]], *, max_bytes: int = 8 * 1024 * 1024) -> dict:
    """Package only verified report files, never raw media, credentials, or model weights."""
    memory, seen, total = BytesIO(), set(), 0
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for job_id, folder in candidates:
            try:
                document = read_snapshot(folder)
                key = document["prepare_fingerprint"]
                if key in seen:
                    continue
                members = [(name, _safe_member(folder, name)) for name in document["files"]]
                size = sum(p.stat().st_size for _, p in members)
                if total + size > max_bytes:
                    continue
                document["origin_job_id"] = job_id
                payload = json.dumps(document, ensure_ascii=False).encode()
                if total + size + len(payload) > max_bytes:
                    continue
                for name, path in members:
                    archive.write(path, f"{key}/{name}")
                archive.writestr(f"{key}/{MANIFEST}", payload)
                total += size + len(payload)
                seen.add(key)
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return {"payload": base64.b64encode(memory.getvalue()).decode("ascii") if seen else "",
            "source_versions": len(seen)}
