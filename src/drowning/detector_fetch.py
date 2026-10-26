"""One-command download of the default person detector, verified against a pinned SHA-256.

The system never downloads weights implicitly; this runs only when the user invokes it.
The pinned hash was recorded from the official Ultralytics release asset (its size matches
the GitHub asset listing); GitHub offers no published digest to cross-check it against.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path
from typing import Any, Callable

DETECTOR = {
    "name": "yolo11n.pt",
    "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "bytes": 5613764,
    "note": "COCO-pretrained YOLO11 nano; class 0 is person. A general-purpose starting point, not tuned for pools.",
}
_MAX_BYTES = 200 * 1024 * 1024


class DetectorFetchError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_detector(target_dir: Path, *, spec: dict[str, Any] = DETECTOR,
                   opener: Callable[..., Any] = urllib.request.urlopen) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / spec["name"]
    if target.is_file() and target.stat().st_size == spec["bytes"] and _sha256(target) == spec["sha256"]:
        return {"path": str(target), "downloaded": False, "sha256": spec["sha256"]}
    if target.exists():
        raise DetectorFetchError(f"{target} exists but does not match the pinned file; move it away first")
    temp = target.with_name(target.name + f".{os.getpid()}.part")
    try:
        with opener(spec["url"], timeout=120) as response, temp.open("wb") as stream:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > _MAX_BYTES:
                    raise DetectorFetchError("download is larger than expected")
                stream.write(chunk)
        if temp.stat().st_size != spec["bytes"] or _sha256(temp) != spec["sha256"]:
            raise DetectorFetchError("downloaded file failed the pinned size/SHA-256 check and was discarded")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return {"path": str(target), "downloaded": True, "sha256": spec["sha256"]}


def set_env_value(env_path: Path, key: str, value: str, *, only_if_empty: bool = True) -> bool:
    """Set ``key`` in a .env file, keeping every other line. Returns True when the file changed."""
    # Bytes, not read_text: universal newlines would silently turn CRLF into LF.
    lines = env_path.read_bytes().decode("utf-8").split("\n") if env_path.exists() else []
    for index, line in enumerate(lines):
        if "=" in line and line.split("=", 1)[0].strip() == key:
            current = line.split("=", 1)[1].strip()
            if current and only_if_empty:
                return False
            lines[index] = f"{key}={value}" + ("\r" if line.endswith("\r") else "")
            break
    else:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines), encoding="utf-8", newline="")
    return True
