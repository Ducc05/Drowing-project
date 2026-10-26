"""Camera session and Start/Stop API tests with a scripted capture device (no real webcam)."""
import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from uuid import UUID, uuid4

import httpx
import numpy as np
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from drowning.api.app import create_app
from drowning.config import Settings
from drowning.db.models import Artifact, Base, ExecutionSlot, Job, JobEvent, ModelRegistry, ModelVersion
from drowning.inference.camera import (CameraConnectError, CameraSession, CameraSourceError, CameraStopUnconfirmed,
                                        parse_camera_source, probe_camera, redact_source)

ROOT = Path(__file__).resolve().parents[1]


class ScriptedCapture:
    """Behaves like cv2.VideoCapture; yields small frames until closed or ``fail_after`` is hit."""

    instances = []

    def __init__(self, source, *, opens=True, fail_after=None, frame_delay=0.005):
        self.source, self.opens, self.fail_after, self.frame_delay = source, opens, fail_after, frame_delay
        self.reads = 0
        self.released = False
        ScriptedCapture.instances.append(self)

    def isOpened(self):
        return self.opens and not self.released

    def read(self):
        time.sleep(self.frame_delay)
        if self.released or (self.fail_after is not None and self.reads >= self.fail_after):
            return False, None
        self.reads += 1
        return True, np.full((48, 64, 3), self.reads % 255, dtype=np.uint8)

    def release(self):
        self.released = True


class FakeStreamPipeline:
    def __init__(self, delay=0.0, hang=False):
        self.delay, self.hang = delay, hang
        self.resets = 0
        self.timestamps = []
        self.tracker = type("T", (), {"name": "fixture_tracker"})()

    def reset_stream(self):
        self.resets += 1

    def process_frame(self, image, timestamp_ms, frame_index):
        if self.hang:
            time.sleep(5)
        time.sleep(self.delay)
        self.timestamps.append(timestamp_ms)
        return [{"frame_index": frame_index, "timestamp_ms": timestamp_ms, "track_id": "trk-000001",
                 "bbox_xyxy": [4, 4, 30, 40], "label": "swimming", "prediction_branch": "image_warmup",
                 "detector_confidence": 0.9, "identity_verified": False, "temporal_available": False}]

    @staticmethod
    def draw_annotations(bgr, records):
        return bgr


def wait_for(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


class SourceParsingTests(unittest.TestCase):
    def test_accepts_webcam_index_and_stream_urls_and_redacts_credentials(self):
        self.assertEqual(parse_camera_source("0"), (0, "webcam:0"))
        source, display = parse_camera_source("rtsp://user:secret@192.168.1.5:554/live?token=abc")
        self.assertEqual(display, "rtsp://192.168.1.5:554/live")
        self.assertNotIn("secret", display + redact_source(source))
        self.assertEqual(parse_camera_source("http://phone.local:8080/video")[1], "http://phone.local:8080/video")

    def test_rejects_file_paths_and_other_schemes(self):
        for bad in ("", "16", "C:\\videos\\private.mp4", "file:///etc/passwd", "/dev/video0", "ftp://host/x", "rtsp://"):
            with self.assertRaises(CameraSourceError, msg=bad):
                parse_camera_source(bad)


class CameraSessionTests(unittest.TestCase):
    def setUp(self):
        ScriptedCapture.instances.clear()

    def session(self, pipeline=None, **capture_kwargs):
        return CameraSession(pipeline or FakeStreamPipeline(), 0, display_source="webcam:0",
                             capture_factory=lambda source: ScriptedCapture(source, **capture_kwargs))

    def test_probe_reports_geometry_and_releases_device(self):
        info = probe_camera(0, lambda s: ScriptedCapture(s))
        self.assertEqual((info["width"], info["height"]), (64, 48))
        self.assertTrue(ScriptedCapture.instances[-1].released)

    def test_probe_fails_when_device_cannot_open_or_returns_no_frame(self):
        with self.assertRaises(CameraConnectError):
            probe_camera(0, lambda s: ScriptedCapture(s, opens=False))
        with self.assertRaises(CameraConnectError):
            probe_camera(0, lambda s: ScriptedCapture(s, fail_after=0))
        self.assertTrue(all(c.released for c in ScriptedCapture.instances))

    def test_start_processes_frames_with_increasing_timestamps_then_stop_releases_device(self):
        pipeline = FakeStreamPipeline()
        session = self.session(pipeline)
        session.start()
        self.assertTrue(wait_for(lambda: session.frames_processed >= 5))
        self.assertIsNotNone(session.latest_jpeg())
        self.assertEqual(session.latest_records()[0]["track_id"], "trk-000001")
        summary = session.stop()
        self.assertEqual(summary["state"], "stopped")
        self.assertEqual(pipeline.resets, 1)
        self.assertTrue(ScriptedCapture.instances[-1].released)
        self.assertFalse(session.is_alive())
        self.assertEqual(pipeline.timestamps, sorted(set(pipeline.timestamps)))
        self.assertEqual(summary["tracker"], "fixture_tracker")
        self.assertGreater(summary["frames_processed"], 0)

    def test_slow_model_drops_frames_instead_of_accumulating_latency(self):
        session = self.session(FakeStreamPipeline(delay=0.05))
        session.start()
        time.sleep(0.6)
        summary = session.stop()
        self.assertGreater(summary["frames_dropped"], 0)
        self.assertLess(summary["frames_processed"], summary["frames_read"])

    def test_stream_failure_is_reported_as_failed_not_stopped(self):
        session = self.session(fail_after=8)
        session.start()
        self.assertTrue(wait_for(lambda: session.state == "failed", timeout=5))
        summary = session.stop()
        self.assertEqual(summary["state"], "failed")
        self.assertEqual(summary["error"], "CameraConnectError")

    def test_stop_that_cannot_confirm_exit_does_not_claim_stopped(self):
        session = self.session(FakeStreamPipeline(hang=True))
        session.start()
        time.sleep(0.2)
        with self.assertRaises(CameraStopUnconfirmed):
            session.stop(timeout=0.2)
        self.assertNotEqual(session.state, "stopped")
        self.assertFalse(ScriptedCapture.instances[-1].released)

    def test_mjpeg_chunks_are_multipart_jpegs_and_end_after_stop(self):
        session = self.session()
        session.start()
        self.assertTrue(wait_for(lambda: session.latest_jpeg() is not None))
        chunk = next(session.mjpeg_chunks())
        self.assertTrue(chunk.startswith(b"--frame\r\nContent-Type: image/jpeg"))
        self.assertIn(b"\xff\xd8", chunk)
        session.stop()
        self.assertEqual(list(session.mjpeg_chunks()), [])


def call(app, method, path, **kwargs):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(run())


class CameraApiTests(unittest.TestCase):
    def setUp(self):
        ScriptedCapture.instances.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
            connection.execute(text("INSERT INTO alembic_version VALUES ('20261026_01')"))
        self.model_id = uuid4()
        bundle = self.root / "approved"
        bundle.mkdir()
        (bundle / "manifest.json").write_text(json.dumps({"model_version_id": str(self.model_id)}), encoding="utf-8")
        with Session(self.engine) as session:
            job = Job(job_type="train", state="completed", phase="complete", idempotency_key="src", payload={})
            session.add(job)
            session.flush()
            artifact = Artifact(job_id=job.id, role="candidate_bundle", remote_file_id="drive-1", sha256="a" * 64,
                                size_bytes=10, status="ready")
            session.add(artifact)
            session.flush()
            session.add(ModelVersion(id=self.model_id, model_family="drowning-person-temporal-v1", display_version="v1",
                                      schema_version=1, class_keys=["drowning", "swimming", "out_of_water"],
                                      temporal_trained=True, completion_kind="candidate", status="approved",
                                      artifact_id=artifact.id, metrics={}))
            session.flush()
            session.add(ModelRegistry(model_family="drowning-person-temporal-v1", approved_model_id=self.model_id,
                                       generation=1))
            session.add(ExecutionSlot(slot_id=1, fencing_token=0, reconciliation_status="clear"))
            session.commit()
        self.pipeline = FakeStreamPipeline()
        self.capture_kwargs = {}
        settings = Settings.from_env({"APPROVED_MODEL_BUNDLE_DIR": str(bundle), "INPUT_CACHE_DIR": str(self.root / "in"),
                                      "OUTPUT_CACHE_DIR": str(self.root / "out"), "WORKER_ID": "fixture-worker"})
        self.app = create_app(settings=settings, engine=self.engine, config_dir=ROOT / "configs" / "datasets",
                              env_file=self.root / "missing.env", inference_factory=lambda *a: self.pipeline,
                              camera_capture_factory=lambda s: ScriptedCapture(s, **self.capture_kwargs))

    def tearDown(self):
        service = self.app.state.camera
        if service._session is not None:
            try:
                service.stop()
            except Exception:
                pass
        self.engine.dispose()
        self.temp.cleanup()

    def job_state(self):
        with Session(self.engine) as session:
            job = session.scalars(select(Job).where(Job.job_type == "inference")).one()
            return job.state, job.payload

    def slot_free(self):
        with Session(self.engine) as session:
            return session.get(ExecutionSlot, 1).job_id is None

    def test_start_status_stop_cycle_holds_then_releases_the_compute_slot(self):
        started = call(self.app, "POST", "/api/v1/camera/start",
                       json={"source": "rtsp://u:p@cam.local/live", "idempotency_key": "camera-session-1"})
        self.assertEqual(started.status_code, 200, started.text)
        self.assertTrue(started.json()["active"])
        self.assertNotIn("p@", started.text)
        self.assertEqual(self.job_state()[0], "running")
        self.assertFalse(self.slot_free())
        time.sleep(0.2)
        status = call(self.app, "GET", "/api/v1/camera/status").json()
        self.assertTrue(status["active"])
        self.assertGreater(status["frames_processed"], 0)
        stopped = call(self.app, "POST", "/api/v1/camera/stop")
        self.assertEqual(stopped.status_code, 200, stopped.text)
        self.assertEqual(stopped.json()["state"], "stopped")
        state, payload = self.job_state()
        self.assertEqual(state, "completed")
        self.assertEqual(payload["camera_summary"]["source"], "rtsp://cam.local/live")
        self.assertTrue(self.slot_free())
        self.assertFalse(call(self.app, "GET", "/api/v1/camera/status").json()["active"])

    def test_second_start_and_busy_slot_are_rejected_without_leaving_queued_jobs(self):
        call(self.app, "POST", "/api/v1/camera/start", json={"source": "0", "idempotency_key": "camera-session-1"})
        again = call(self.app, "POST", "/api/v1/camera/start", json={"source": "1", "idempotency_key": "camera-session-2"})
        self.assertEqual(again.status_code, 409)
        call(self.app, "POST", "/api/v1/camera/stop")
        with Session(self.engine) as session:
            slot = session.get(ExecutionSlot, 1)
            blocker = Job(job_type="train", state="running", phase="x", idempotency_key="blocker", payload={})
            session.add(blocker)
            session.flush()
            slot.job_id, slot.owner_id, slot.fencing_token = blocker.id, "other-machine", 9
            session.commit()
        busy = call(self.app, "POST", "/api/v1/camera/start", json={"source": "0", "idempotency_key": "camera-session-3"})
        self.assertEqual(busy.status_code, 409)
        with Session(self.engine) as session:
            states = {j.idempotency_key: j.state for j in session.scalars(select(Job))}
        self.assertEqual(states["camera-session-3"], "cancelled")

    def test_unreachable_camera_fails_job_and_frees_slot(self):
        self.capture_kwargs = {"opens": False}
        response = call(self.app, "POST", "/api/v1/camera/start", json={"source": "3", "idempotency_key": "camera-session-1"})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(self.job_state()[0], "failed")
        self.assertTrue(self.slot_free())

    def test_invalid_source_and_missing_session_are_client_errors(self):
        bad = call(self.app, "POST", "/api/v1/camera/start", json={"source": "C:/x.mp4", "idempotency_key": "camera-session-1"})
        self.assertEqual(bad.status_code, 422)
        self.assertEqual(call(self.app, "POST", "/api/v1/camera/stop").status_code, 409)
        self.assertEqual(call(self.app, "GET", "/api/v1/camera/stream").status_code, 409)

    def test_stream_dying_marks_job_failed_and_status_frees_slot(self):
        self.capture_kwargs = {"fail_after": 6}
        call(self.app, "POST", "/api/v1/camera/start", json={"source": "0", "idempotency_key": "camera-session-1"})
        self.assertTrue(wait_for(lambda: not call(self.app, "GET", "/api/v1/camera/status").json()["active"], timeout=6))
        self.assertEqual(self.job_state()[0], "failed")
        self.assertTrue(self.slot_free())

    def test_unconfirmed_stop_keeps_slot_and_requests_cancellation(self):
        self.pipeline = FakeStreamPipeline(hang=True)
        self.app.state.camera.stop_timeout = 0.2
        call(self.app, "POST", "/api/v1/camera/start", json={"source": "0", "idempotency_key": "camera-session-1"})
        time.sleep(0.2)
        response = call(self.app, "POST", "/api/v1/camera/stop")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["detail"]["compute_stop_confirmed"])
        self.assertEqual(self.job_state()[0], "cancellation_requested")
        self.assertFalse(self.slot_free())


if __name__ == "__main__":
    unittest.main()
