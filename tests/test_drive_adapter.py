import hashlib
import tempfile
import unittest
from pathlib import Path

from drowning.integrations.drive import DriveError, GoogleDriveArtifactAdapter, RemoteUploadUnconfirmed, md5_file, sha256_file


class FakeRequest:
    def __init__(self, operation):
        self.operation = operation

    def execute(self):
        return self.operation()

    def next_chunk(self):
        return None, self.operation()


class FakeDownloader:
    def __init__(self, stream, request):
        self.stream, self.request, self.done = stream, request, False

    def next_chunk(self):
        if not self.done:
            self.stream.write(self.request.content)
            self.done = True
        return None, self.done


class FakeFiles:
    def __init__(self):
        self.items = {}
        self.create_count = 0
        self.fail_upload = False

    def list(self, q, fields, pageSize):
        key = q.split("value='")[-1].split("'")[0]
        return FakeRequest(lambda: {"files": [item["metadata"] for item in self.items.values() if item["metadata"]["appProperties"]["artifactKey"] == key]})

    def create(self, body, media_body, fields):
        self.create_count += 1
        file_id = f"drive-{self.create_count}"
        content = Path(media_body.path).read_bytes()
        metadata = {
            "id": file_id, "name": body["name"], "size": str(len(content)),
            "md5Checksum": hashlib.md5(content, usedforsecurity=False).hexdigest(),
            "appProperties": body["appProperties"],
        }
        def finish():
            if self.fail_upload:
                raise OSError("connection lost")
            self.items[file_id] = {"metadata": metadata, "content": content}
            return metadata
        return FakeRequest(finish)

    def get(self, fileId, fields):
        return FakeRequest(lambda: self.items[fileId]["metadata"])

    def get_media(self, fileId):
        return type("MediaRequest", (), {"content": self.items[fileId]["content"]})()


class FakeDriveService:
    def __init__(self):
        self.file_api = FakeFiles()

    def files(self):
        return self.file_api


class DriveAdapterTests(unittest.TestCase):
    def test_resumable_upload_reconciles_idempotently_and_download_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "weights.pt"
            source.write_bytes(b"model-artifact-bytes")
            service = FakeDriveService()
            adapter = GoogleDriveArtifactAdapter(
                service,
                media_upload_factory=lambda path, **kwargs: type("Upload", (), {"path": path})(),
                media_download_factory=lambda stream, request, **kwargs: FakeDownloader(stream, request),
            )
            first = adapter.upload(source, folder_id="folder", job_id="job-1", attempt_id="attempt-1",
                                   role="model", artifact_key="job-1/attempt-1/model")
            second = adapter.upload(source, folder_id="folder", job_id="job-1", attempt_id="attempt-1",
                                    role="model", artifact_key="job-1/attempt-1/model")
            self.assertEqual(first["file_id"], second["file_id"])
            self.assertTrue(first["verified"])
            self.assertEqual(service.file_api.create_count, 1)
            target = root / "downloaded.pt"
            downloaded = adapter.download(first["file_id"], target, expected_sha256=sha256_file(source))
            self.assertTrue(downloaded["verified"])
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertEqual(downloaded["md5_checksum"] if "md5_checksum" in downloaded else md5_file(target), md5_file(source))

    def test_interrupted_upload_is_unconfirmed_and_not_silently_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "artifact.bin"
            source.write_bytes(b"bytes")
            service = FakeDriveService()
            service.file_api.fail_upload = True
            adapter = GoogleDriveArtifactAdapter(
                service, media_upload_factory=lambda path, **kwargs: type("Upload", (), {"path": path})())
            with self.assertRaises(RemoteUploadUnconfirmed):
                adapter.upload(source, folder_id="folder", job_id="job", attempt_id="attempt",
                               role="output", artifact_key="job/attempt/output")
            self.assertEqual(service.file_api.create_count, 1)

    def test_refuses_existing_or_protected_download_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            service = FakeDriveService()
            adapter = GoogleDriveArtifactAdapter(service)
            existing = Path(directory) / "exists"
            existing.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(DriveError, "overwrite"):
                adapter.download("missing", existing, expected_sha256="a" * 64)
            with self.assertRaisesRegex(DriveError, "Protected"):
                adapter.download("missing", Path("D:/DrowningDemo/output.bin"), expected_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
