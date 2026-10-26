"""Google Drive v3 resumable artifact upload/download with checksum verification."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

class DriveError(RuntimeError):
    pass


class RemoteUploadUnconfirmed(DriveError):
    """Upload connection ended without a server receipt; reconcile before retrying."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_output_path(path: Path) -> Path:
    target = path.resolve()
    try:
        target.relative_to(Path("D:/DrowningDemo").resolve())
        protected_demo = True
    except ValueError:
        protected_demo = False
    if protected_demo or any(part.casefold() == "sources" for part in target.parts):
        raise DriveError("Protected D:\\DrowningDemo or sources/ output path")
    return target


class GoogleDriveArtifactAdapter:
    def __init__(self, service: Any, *, media_upload_factory: Callable[..., Any] | None = None,
                 media_download_factory: Callable[..., Any] | None = None, chunksize: int = 8 * 1024 * 1024):
        self.service = service
        self.media_upload_factory = media_upload_factory
        self.media_download_factory = media_download_factory
        if chunksize < 256 * 1024 or chunksize % (256 * 1024):
            raise ValueError("Drive resumable chunk size must be a multiple of 256 KiB")
        self.chunksize = chunksize

    def _upload_media(self, path: Path) -> Any:
        factory = self.media_upload_factory
        if factory is None:
            try:
                from googleapiclient.http import MediaFileUpload
            except ImportError as exc:
                raise DriveError("Install optional Drive dependencies with `pip install -e '.[drive]'`") from exc
            factory = MediaFileUpload
        return factory(str(path), resumable=True, chunksize=self.chunksize)

    def _download_media(self, request: Any, destination: Any) -> Any:
        factory = self.media_download_factory
        if factory is None:
            try:
                from googleapiclient.http import MediaIoBaseDownload
            except ImportError as exc:
                raise DriveError("Install optional Drive dependencies with `pip install -e '.[drive]'`") from exc
            factory = MediaIoBaseDownload
        return factory(destination, request, chunksize=self.chunksize)

    def reconcile(self, folder_id: str, artifact_key: str) -> dict[str, Any] | None:
        query_key = artifact_key.replace("\\", "\\\\").replace("'", "\\'")
        query = f"'{folder_id}' in parents and trashed = false and appProperties has {{ key='artifactKey' and value='{query_key}' }}"
        response = self.service.files().list(q=query, fields="files(id,name,size,md5Checksum,appProperties)", pageSize=10).execute()
        files = response.get("files", [])
        if len(files) > 1:
            raise DriveError("Multiple Drive files match one artifact key; manual reconciliation is required")
        return files[0] if files else None

    def upload(self, path: Path, *, folder_id: str, job_id: str, attempt_id: str,
               role: str, artifact_key: str) -> dict[str, Any]:
        if not folder_id.strip() or not job_id or not attempt_id or not role or not artifact_key:
            raise ValueError("folder_id, job_id, attempt_id, role and artifact_key are required")
        source = path.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        sha256, md5, size = sha256_file(source), md5_file(source), source.stat().st_size
        existing = self.reconcile(folder_id, artifact_key)
        if existing is not None:
            return self._verify_remote(existing, sha256=sha256, md5=md5, size=size, artifact_key=artifact_key)
        body = {
            "name": f"{job_id}-{attempt_id}-{role}-{sha256[:12]}{source.suffix}",
            "parents": [folder_id],
            "appProperties": {"artifactKey": artifact_key, "sha256": sha256,
                              "jobId": job_id, "attemptId": attempt_id, "role": role},
        }
        try:
            request = self.service.files().create(body=body, media_body=self._upload_media(source),
                                                  fields="id,name,size,md5Checksum,appProperties")
            response = None
            while response is None:
                _progress, response = request.next_chunk()
            verified = self.service.files().get(fileId=response["id"],
                                                fields="id,name,size,md5Checksum,appProperties").execute()
        except Exception as exc:
            raise RemoteUploadUnconfirmed("Drive upload did not return a verified receipt; reconcile artifact key before retrying") from exc
        return self._verify_remote(verified, sha256=sha256, md5=md5, size=size, artifact_key=artifact_key)

    @staticmethod
    def _verify_remote(metadata: dict[str, Any], *, sha256: str, md5: str, size: int,
                       artifact_key: str) -> dict[str, Any]:
        properties = metadata.get("appProperties", {})
        if (properties.get("artifactKey") != artifact_key or properties.get("sha256") != sha256
                or int(metadata.get("size", -1)) != size or metadata.get("md5Checksum") != md5):
            raise DriveError("Drive artifact metadata/checksum does not match the local file")
        return {"file_id": metadata["id"], "name": metadata["name"], "size_bytes": size,
                "sha256": sha256, "md5_checksum": md5, "verified": True}

    def download(self, file_id: str, destination: Path, *, expected_sha256: str) -> dict[str, Any]:
        if not file_id or len(expected_sha256) != 64:
            raise ValueError("file_id and expected SHA-256 are required")
        target = _safe_output_path(destination)
        if target.exists():
            raise DriveError("Refusing to overwrite an existing local artifact")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{target.name}-", suffix=".part", dir=target.parent)
        os.close(descriptor)
        temp_path = Path(temp_name)
        try:
            request = self.service.files().get_media(fileId=file_id)
            with temp_path.open("wb") as stream:
                downloader = self._download_media(request, stream)
                done = False
                while not done:
                    _progress, done = downloader.next_chunk()
            received_sha = sha256_file(temp_path)
            metadata = self.service.files().get(fileId=file_id, fields="id,name,size,md5Checksum").execute()
            if received_sha != expected_sha256 or temp_path.stat().st_size != int(metadata.get("size", -1)):
                raise DriveError("Downloaded artifact failed size or SHA-256 verification")
            if metadata.get("md5Checksum") != md5_file(temp_path):
                raise DriveError("Downloaded artifact failed Drive MD5 verification")
            temp_path.rename(target)
            return {"file_id": file_id, "path": str(target), "sha256": received_sha,
                    "size_bytes": target.stat().st_size, "verified": True}
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise


def build_drive_service(token_file: Path) -> Any:
    """Build Drive API v3 client from an existing OAuth token file; never prints credentials."""
    if not token_file.is_file():
        raise DriveError("Google OAuth token file is missing; configure it locally first")
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        credentials = Credentials.from_authorized_user_file(str(token_file), scopes=["https://www.googleapis.com/auth/drive"])
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        if not credentials.valid:
            raise DriveError("Google OAuth credentials are expired or incomplete")
        return build("drive", "v3", credentials=credentials, cache_discovery=False)
    except DriveError:
        raise
    except Exception as exc:
        raise DriveError("Unable to initialize Google Drive API client") from exc


DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"


def authorize_drive(client_secret_file: Path, token_file: Path, *, force: bool = False,
                    flow_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    """One-time browser consent that writes the OAuth token file; prints no credentials."""
    if token_file.exists() and not force:
        raise DriveError("A token file already exists; pass --force to replace it")
    if not client_secret_file.is_file():
        raise DriveError("OAuth client file is missing; download the Desktop client JSON from Google Cloud Console")
    if flow_factory is None:
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as exc:
            raise DriveError('Install Drive support first: pip install -e ".[drive]"') from exc
        flow_factory = InstalledAppFlow.from_client_secrets_file
    flow = flow_factory(str(client_secret_file), scopes=[DRIVE_SCOPE])
    credentials = flow.run_local_server(port=0)
    token_file.parent.mkdir(parents=True, exist_ok=True)
    temp = token_file.with_name(token_file.name + ".tmp")
    temp.write_text(credentials.to_json(), encoding="utf-8")
    temp.replace(token_file)
    return {"token_file_written": True, "scope": DRIVE_SCOPE, "has_refresh_token": bool(getattr(credentials, "refresh_token", None))}


def check_folder_access(service: Any, folder_id: str) -> str | None:
    """Return a problem description, or None when the signed-in account can add files to the folder."""
    try:
        meta = service.files().get(fileId=folder_id, fields="id,mimeType,capabilities(canAddChildren)",
                                   supportsAllDrives=True).execute()
    except Exception as exc:
        return f"Không đọc được thư mục Drive ({type(exc).__name__})"
    if meta.get("mimeType") != "application/vnd.google-apps.folder":
        return "GOOGLE_DRIVE_FOLDER_ID không phải là một thư mục"
    if not meta.get("capabilities", {}).get("canAddChildren"):
        return "Tài khoản đã đăng nhập không có quyền thêm file vào thư mục này"
    return None
