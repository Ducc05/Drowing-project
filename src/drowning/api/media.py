"""Bounded local upload handling and annotated-image output helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from PIL import Image

MAX_MEDIA_UPLOAD_BYTES = 512 * 1024 * 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def safe_local_root(path: str | Path) -> Path:
    if not str(path).strip():
        raise ValueError("Local inference cache path must be configured")
    root = Path(path).expanduser().resolve()
    try:
        root.relative_to(Path("D:/DrowningDemo").resolve())
        protected_demo = True
    except ValueError:
        protected_demo = False
    if protected_demo or any(part.casefold() == "sources" for part in root.parts):
        raise ValueError("Local inference cache cannot be under D:\\DrowningDemo or sources/")
    root.mkdir(parents=True, exist_ok=True)
    return root


def persist_upload(upload: UploadFile, cache_dir: str | Path, *, modality: str) -> tuple[Path, str, str, int]:
    if modality not in {"image", "video"}:
        raise ValueError("unsupported media modality")
    filename = Path(upload.filename or "").name
    extension = Path(filename).suffix.casefold()
    if not filename or extension not in (IMAGE_EXTENSIONS if modality == "image" else VIDEO_EXTENSIONS):
        raise HTTPException(status_code=415, detail=f"Unsupported {modality} file extension")
    root = safe_local_root(cache_dir) / "incoming"
    root.mkdir(parents=True, exist_ok=True)
    temp_path = root / f"{uuid4().hex}{extension}"
    digest, total = hashlib.sha256(), 0
    try:
        with temp_path.open("xb") as stream:
            while chunk := upload.file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_MEDIA_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Media upload exceeds 512 MiB")
                digest.update(chunk)
                stream.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="Uploaded media file is empty")
        if modality == "image":
            try:
                with Image.open(temp_path) as image:
                    image.verify()
            except Exception as exc:
                raise HTTPException(status_code=415, detail="Uploaded image cannot be decoded") from exc
        else:
            try:
                import cv2
                capture = cv2.VideoCapture(str(temp_path))
                valid = capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
                capture.release()
            except ImportError as exc:
                raise HTTPException(status_code=503, detail="Install OpenCV vision support to test video") from exc
            if not valid:
                raise HTTPException(status_code=415, detail="Uploaded video cannot be decoded")
        return temp_path, filename, digest.hexdigest(), total
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
