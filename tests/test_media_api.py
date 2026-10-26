"""HTTP flow tests with temporary SQLite, media files, and an explicit fake predictor."""
import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import httpx
import numpy as np
from PIL import Image
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from drowning.api.app import create_app
from drowning.config import Settings
from drowning.db.models import Artifact, Base, ExecutionSlot, Job, ModelRegistry, ModelVersion

ROOT = Path(__file__).resolve().parents[1]


def request(app, method, path, **kwargs):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(run())


def png(color="blue"):
    stream = io.BytesIO()
    Image.new("RGB", (48, 40), color).save(stream, format="PNG")
    return stream.getvalue()


class FakePipeline:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def infer_image(self, image):
        self.calls += 1
        if self.fail:
            raise RuntimeError("fixture predictor failure")
        return {"model_version": "fixture-v1", "detections": [
            {"bbox_xyxy": [3, 4, 24, 32], "label": "swimming", "prediction_branch": "image",
             "identity_verified": False, "class_scores": [0.1, 0.8, 0.1]}]}

    def infer_video(self, source, destination):
        self.calls += 1
        capture = cv2.VideoCapture(str(source))
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"VP80"), 10, (48, 40))
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            count += 1
        capture.release()
        writer.release()
        detections = destination.with_suffix(".detections.json")
        detections.write_text(json.dumps({"model_version": "fixture-v1", "frame_count": count,
                                          "scores_are_calibrated_probabilities": False}), encoding="utf-8")
        return {"model_version": "fixture-v1", "frame_count": count, "detection_count": 0,
                "video_path": str(destination), "detections_path": str(detections)}


class FakeDrive:
    def __init__(self):
        self.fail = False
        self.keys = []

    def upload(self, path, **kwargs):
        self.keys.append(kwargs["artifact_key"])
        if self.fail:
            raise RuntimeError("fixture Drive unavailable")
        return {"verified": True, "file_id": f"drive-{path.name}"}


class MediaApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
            connection.execute(text("INSERT INTO alembic_version VALUES ('20261026_01')"))
        self.model_id = uuid4()
        self.bundle = self.root / "approved"
        self.bundle.mkdir()
        # Deliberately use notebook-style UUID hex to exercise normalization.
        (self.bundle / "manifest.json").write_text(json.dumps({"model_version_id": self.model_id.hex}), encoding="utf-8")
        with Session(self.engine) as session:
            source_job = Job(job_type="train", state="completed", phase="complete",
                             idempotency_key="fixture-source-job", payload={})
            session.add(source_job)
            session.flush()
            artifact = Artifact(job_id=source_job.id, role="candidate_bundle", storage_provider="google_drive",
                                remote_file_id="fixture-drive-model", sha256="a" * 64, size_bytes=100, status="ready")
            session.add(artifact)
            session.flush()
            session.add(ModelVersion(id=self.model_id, model_family="drowning-person-temporal-v1",
                                      display_version="fixture-v1", schema_version=1,
                                      class_keys=["drowning", "swimming", "out_of_water"],
                                      temporal_trained=True, completion_kind="candidate", status="approved",
                                      artifact_id=artifact.id, metrics={}))
            session.flush()
            session.add(ModelRegistry(model_family="drowning-person-temporal-v1",
                                       approved_model_id=self.model_id, generation=1))
            session.add(ExecutionSlot(slot_id=1, fencing_token=0, reconciliation_status="clear"))
            session.commit()
        self.pipeline = FakePipeline()
        self.drive = FakeDrive()
        self.env = {"APPROVED_MODEL_BUNDLE_DIR": str(self.bundle),
                    "INPUT_CACHE_DIR": str(self.root / "inputs"),
                    "OUTPUT_CACHE_DIR": str(self.root / "results"), "WORKER_ID": "fixture-worker"}
        self.app = self.make_app()

    def make_app(self, drive=False):
        settings = Settings.from_env({**self.env, "GOOGLE_DRIVE_FOLDER_ID": "fixture-folder" if drive else ""})
        return create_app(settings=settings, engine=self.engine, config_dir=ROOT / "configs" / "datasets",
                          env_file=self.root / "missing.env",
                          inference_factory=lambda *args: self.pipeline, drive_factory=lambda: self.drive)

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def upload(self, key, content=None, app=None):
        return request(app or self.app, "POST", "/api/v1/test/image",
                       headers={"Idempotency-Key": key},
                       files={"file": ("person.png", content or png(), "image/png")})

    def test_image_upload_result_download_and_idempotency(self):
        response = self.upload("image-upload-001")
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["state"], "sync_pending")
        self.assertEqual(result["result_storage"], "local")
        self.assertEqual(result["model_version_id"], str(self.model_id))
        self.assertEqual(request(self.app, "GET", result["files"]["annotated"]).status_code, 200)
        detections = request(self.app, "GET", result["files"]["detections"])
        self.assertEqual(detections.json()["detections"][0]["prediction_branch"], "image")
        duplicate = self.upload("image-upload-001")
        self.assertEqual(duplicate.json()["job_id"], result["job_id"])
        self.assertFalse(duplicate.json()["created"])
        self.assertEqual(self.pipeline.calls, 1)
        mismatch = self.upload("image-upload-001", png("red"))
        self.assertEqual(mismatch.status_code, 409)
        (self.bundle / "manifest.json").unlink()
        replay = self.upload("image-upload-001")
        self.assertEqual(replay.status_code, 200, replay.text)
        self.assertEqual(replay.json()["job_id"], result["job_id"])
        self.assertEqual(self.pipeline.calls, 1)
        with Session(self.engine) as session:
            self.assertIsNone(session.get(ExecutionSlot, 1).job_id)

    def test_busy_slot_preserves_queued_input_and_can_resume_same_request(self):
        with Session(self.engine) as session:
            busy = Job(job_type="prepare", state="running", phase="compute",
                       idempotency_key="fixture-busy", payload={})
            session.add(busy)
            session.flush()
            slot = session.get(ExecutionSlot, 1)
            slot.job_id, slot.owner_id = busy.id, "other-worker"
            session.commit()
        first = self.upload("image-upload-busy")
        self.assertEqual(first.status_code, 409, first.text)
        job_id = first.json()["detail"]["job_id"]
        with Session(self.engine) as session:
            queued = session.get(Job, UUID(job_id))
            self.assertEqual(queued.state, "queued")
            self.assertTrue(Path(queued.payload["local_input"]).is_file())
            slot = session.get(ExecutionSlot, 1)
            slot.job_id, slot.owner_id = None, None
            session.commit()
        resumed = self.upload("image-upload-busy")
        self.assertEqual(resumed.status_code, 200, resumed.text)
        self.assertEqual(resumed.json()["job_id"], job_id)
        self.assertFalse(resumed.json()["created"])
        self.assertEqual(self.pipeline.calls, 1)

    def test_invalid_media_and_missing_approved_model_are_explicit(self):
        bad = self.upload("image-upload-corrupt", b"not-an-image")
        self.assertEqual(bad.status_code, 415)
        self.assertEqual(self.pipeline.calls, 0)
        with Session(self.engine) as session:
            session.delete(session.get(ModelRegistry, "drowning-person-temporal-v1"))
            session.commit()
        missing = self.upload("image-upload-no-model")
        self.assertEqual(missing.status_code, 409)
        self.assertIn("No approved model", missing.text)

    def test_compute_failure_releases_slot_and_records_failed_job(self):
        self.pipeline.fail = True
        response = self.upload("image-upload-failure")
        self.assertEqual(response.status_code, 500)
        job_id = UUID(response.json()["detail"]["job_id"])
        with Session(self.engine) as session:
            self.assertEqual(session.get(Job, job_id).state, "failed")
            self.assertIsNone(session.get(ExecutionSlot, 1).job_id)
        replay = self.upload("image-upload-failure")
        self.assertEqual(replay.status_code, 200)
        self.assertEqual(replay.json()["state"], "failed")
        self.assertFalse(replay.json()["created"])
        self.assertEqual(self.pipeline.calls, 1)

    def test_drive_sync_retry_does_not_repeat_inference(self):
        app = self.make_app(drive=True)
        self.drive.fail = True
        first = self.upload("image-upload-drive", app=app)
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["state"], "sync_pending")
        self.drive.fail = False
        retried = request(app, "POST", f"/api/v1/jobs/{first.json()['job_id']}/sync-results")
        self.assertEqual(retried.status_code, 200, retried.text)
        self.assertEqual(retried.json()["state"], "completed")
        self.assertEqual(self.pipeline.calls, 1)
        with Session(self.engine) as session:
            artifacts = session.scalars(select(Artifact).where(Artifact.job_id == UUID(first.json()["job_id"]))).all()
            self.assertEqual(len(artifacts), 2)
            self.assertTrue(all(item.status == "ready" for item in artifacts))

    def test_video_upload_returns_decodable_annotated_media_and_json(self):
        source = self.root / "input.avi"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (48, 40))
        self.assertTrue(writer.isOpened())
        for _ in range(3):
            writer.write(np.zeros((40, 48, 3), dtype=np.uint8))
        writer.release()
        response = request(self.app, "POST", "/api/v1/test/video",
                           headers={"Idempotency-Key": "video-upload-001"},
                           files={"file": ("input.avi", source.read_bytes(), "video/x-msvideo")})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["summary"]["frame_count"], 3)
        output = request(self.app, "GET", result["files"]["annotated"])
        self.assertEqual(output.headers["content-type"], "video/webm")
        self.assertIn(b"V_VP8", output.content)
        saved = self.root / "download.webm"
        saved.write_bytes(output.content)
        capture = cv2.VideoCapture(str(saved))
        self.assertTrue(capture.isOpened())
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
        capture.release()
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_result_pointer_cannot_escape_result_cache(self):
        result = self.upload("image-upload-path").json()
        with Session(self.engine) as session:
            job = session.get(Job, UUID(result["job_id"]))
            job.payload = {**job.payload, "local_result_dir": str(self.root)}
            session.commit()
        response = request(self.app, "GET", result["files"]["annotated"])
        self.assertEqual(response.status_code, 409)

    def test_exif_orientation_is_applied_before_annotated_output(self):
        buffer = io.BytesIO()
        image = Image.new("RGB", (48, 40), "blue")
        exif = Image.Exif()
        exif[274] = 6
        image.save(buffer, format="JPEG", exif=exif)
        response = request(self.app, "POST", "/api/v1/test/image",
                           headers={"Idempotency-Key": "image-exif-orientation"},
                           files={"file": ("oriented.jpg", buffer.getvalue(), "image/jpeg")})
        self.assertEqual(response.status_code, 200, response.text)
        output = request(self.app, "GET", response.json()["files"]["annotated"])
        with Image.open(io.BytesIO(output.content)) as annotated:
            self.assertEqual(annotated.size, (40, 48))


if __name__ == "__main__":
    unittest.main()
