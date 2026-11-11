"""Camera session and Start/Stop API tests with a scripted capture device (no real webcam)."""
import asyncio
import io
import json
import tempfile
import threading
import time
import unittest
import unittest.mock
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
from drowning.api.camera import CameraNotRunning
from drowning.config import Settings
from drowning.db.models import Artifact, Base, ExecutionSlot, Job, JobEvent, ModelRegistry, ModelVersion
from drowning.inference.camera import (MAX_FRAME_BYTES, CameraConnectError, CameraSession, CameraSourceError,
                                        CameraStopUnconfirmed, PushCapture, decode_frame, parse_camera_source,
                                        probe_camera, redact_source)

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

    def get(self, prop_id):
        return 0.0


class FakeStreamPipeline:
    def __init__(self, delay=0.0, hang=False):
        self.delay, self.hang = delay, hang
        self.resets = 0
        self.timestamps = []
        self.live_fps = []
        self.tracker = type("T", (), {"name": "fixture_tracker"})()

    def configure_live_stream(self, fps):
        self.live_fps.append(fps)

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

    def test_browser_tab_is_a_named_source(self):
        self.assertEqual(parse_camera_source("browser-tab"), ("browser-tab", "browser-tab"))
        self.assertEqual(parse_camera_source(" browser-tab "), ("browser-tab", "browser-tab"))
        for bad in ("browser-tabs", "Browser-Tab", "browser-tab/x"):
            with self.assertRaises(CameraSourceError, msg=bad):
                parse_camera_source(bad)


def jpeg_bytes(width=64, height=48, value=90):
    ok, encoded = cv2.imencode(".jpg", np.full((height, width, 3), value, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


class PushCaptureTests(unittest.TestCase):
    def test_read_returns_only_new_frames_and_latest_wins(self):
        capture = PushCapture(read_timeout=0.05)
        capture.push(np.full((4, 4, 3), 1, dtype=np.uint8))
        capture.push(np.full((4, 4, 3), 2, dtype=np.uint8))
        ok, frame = capture.read()
        self.assertTrue(ok)
        self.assertEqual(int(frame[0, 0, 0]), 2)
        self.assertEqual(capture.read(), (False, None))

    def test_first_read_waits_for_first_frame(self):
        capture = PushCapture(first_frame_timeout=2.0, read_timeout=0.05)
        threading.Timer(0.1, lambda: capture.push(np.zeros((4, 6, 3), dtype=np.uint8))).start()
        ok, frame = capture.read()
        self.assertTrue(ok)
        self.assertEqual(frame.shape, (4, 6, 3))
        self.assertEqual(capture.shape, (4, 6))
        empty = PushCapture(first_frame_timeout=0.1)
        self.assertEqual(empty.read(), (False, None))

    def test_release_unblocks_reader_and_marks_closed(self):
        capture = PushCapture(read_timeout=5.0, first_frame_timeout=5.0)
        capture.push(np.zeros((2, 2, 3), dtype=np.uint8))
        capture.read()
        threading.Timer(0.1, capture.release).start()
        started = time.monotonic()
        self.assertEqual(capture.read(), (False, None))
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertFalse(capture.isOpened())
        self.assertTrue(capture.released)

    def test_get_reports_nominal_fps(self):
        self.assertEqual(PushCapture(fps=12).get(5), 12.0)
        self.assertEqual(PushCapture(fps=12).get(3), 0.0)

    def test_pushed_frame_is_copied_for_the_reader(self):
        capture = PushCapture(read_timeout=0.05)
        source = np.zeros((2, 2, 3), dtype=np.uint8)
        capture.push(source)
        source[:] = 9
        self.assertEqual(int(capture.read()[1][0, 0, 0]), 0)


class DecodeFrameTests(unittest.TestCase):
    def test_valid_jpeg_decodes_to_bgr(self):
        self.assertEqual(decode_frame(jpeg_bytes(64, 48)).shape, (48, 64, 3))

    def test_garbage_empty_and_oversized_payloads_are_rejected(self):
        for bad in (b"", b"not an image", b"x" * (MAX_FRAME_BYTES + 1)):
            with self.assertRaises(CameraSourceError):
                decode_frame(bad)

    def test_huge_declared_size_is_rejected_before_decoding(self):
        buffer = io.BytesIO()
        Image.new("RGB", (4000, 4000), (10, 10, 10)).save(buffer, format="JPEG", quality=20)
        data = buffer.getvalue()
        self.assertLess(len(data), MAX_FRAME_BYTES)
        with unittest.mock.patch("cv2.imdecode") as imdecode:
            with self.assertRaises(CameraSourceError):
                decode_frame(data)
            imdecode.assert_not_called()


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
        self.app.state.camera.push_read_timeout = 0.05
        self.feeding = threading.Event()
        self.feeder = None

    def feed(self, jpeg=None, interval=0.02):
        """Behave like the browser: keep posting frames, ignoring errors, until ``stop_feed``."""
        jpeg = jpeg or jpeg_bytes()
        service = self.app.state.camera
        self.feeding.clear()

        def run():
            while not self.feeding.is_set():
                try:
                    service.push_frame(jpeg)
                except Exception:
                    pass
                time.sleep(interval)

        self.feeder = threading.Thread(target=run, daemon=True)
        self.feeder.start()

    def stop_feed(self):
        self.feeding.set()
        if self.feeder is not None:
            self.feeder.join(2)

    def start_tab(self, key="camera-tab-1", **extra):
        return call(self.app, "POST", "/api/v1/camera/start",
                    json={"source": "browser-tab", "idempotency_key": key, **extra})

    def tearDown(self):
        # Other test classes reuse this tearDown without the feed helpers, so stay self-contained.
        feeding = getattr(self, "feeding", None)
        if feeding is not None:
            feeding.set()
            if self.feeder is not None:
                self.feeder.join(2)
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

    def test_browser_tab_start_status_stop_cycle(self):
        self.feed()
        started = self.start_tab()
        self.assertEqual(started.status_code, 200, started.text)
        self.assertTrue(started.json()["active"])
        state, payload = self.job_state()
        self.assertEqual(state, "running")
        self.assertEqual(payload["capture"], "browser_tab")
        self.assertEqual(payload["source"], "browser-tab")
        self.assertFalse(self.slot_free())
        self.assertTrue(wait_for(lambda: call(self.app, "GET", "/api/v1/camera/status").json()["frames_processed"] > 0))
        stopped = call(self.app, "POST", "/api/v1/camera/stop")
        self.assertEqual(stopped.status_code, 200, stopped.text)
        self.assertEqual(self.job_state()[0], "completed")
        self.assertTrue(self.slot_free())

    def test_frames_without_a_browser_tab_session_are_rejected(self):
        service = self.app.state.camera
        with self.assertRaises(CameraNotRunning):
            service.push_frame(jpeg_bytes())
        call(self.app, "POST", "/api/v1/camera/start", json={"source": "0", "idempotency_key": "camera-session-1"})
        with self.assertRaises(CameraNotRunning):
            service.push_frame(jpeg_bytes())

    def test_wrong_size_frame_is_rejected_but_session_survives(self):
        self.feed()
        self.start_tab()
        with self.assertRaises(CameraSourceError):
            self.app.state.camera.push_frame(jpeg_bytes(32, 24))
        before = call(self.app, "GET", "/api/v1/camera/status").json()
        self.assertTrue(before["active"])
        self.assertTrue(wait_for(lambda: call(self.app, "GET", "/api/v1/camera/status").json()["frames_processed"]
                                 > before["frames_processed"]))

    def test_huge_declared_size_frame_is_rejected(self):
        self.feed()
        self.start_tab()
        buffer = io.BytesIO()
        Image.new("RGB", (4000, 4000), (10, 10, 10)).save(buffer, format="JPEG", quality=20)
        with self.assertRaises(CameraSourceError):
            self.app.state.camera.push_frame(buffer.getvalue())
        self.assertTrue(call(self.app, "GET", "/api/v1/camera/status").json()["active"])

    def test_stopping_the_feed_fails_the_session_and_frees_the_slot(self):
        self.feed()
        self.start_tab()
        self.assertTrue(wait_for(lambda: call(self.app, "GET", "/api/v1/camera/status").json()["frames_processed"] > 0))
        self.stop_feed()
        self.assertTrue(wait_for(lambda: not call(self.app, "GET", "/api/v1/camera/status").json()["active"], 10))
        self.assertEqual(self.job_state()[0], "failed")
        self.assertTrue(self.slot_free())

    def test_start_with_no_frames_fails_and_frees_the_slot(self):
        self.app.state.camera.push_first_frame_timeout = 0.2
        response = self.start_tab()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(self.job_state()[0], "failed")
        self.assertTrue(self.slot_free())

    def test_probe_rejects_browser_tab(self):
        response = call(self.app, "POST", "/api/v1/camera/probe", json={"source": "browser-tab"})
        self.assertEqual(response.status_code, 422)

    def test_fps_is_passed_to_the_pipeline_clock(self):
        self.feed()
        self.start_tab(fps=15)
        self.assertEqual(self.pipeline.live_fps, [15.0])

    def post_frame(self, data, **kwargs):
        return call(self.app, "POST", "/api/v1/camera/frame", content=data,
                    headers={"Content-Type": "image/jpeg"}, **kwargs)

    def test_frame_route_returns_alert_state_and_counts(self):
        self.feed()
        self.start_tab()
        response = self.post_frame(jpeg_bytes())
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIsInstance(body["alert_track_ids"], list)
        self.assertGreaterEqual(body["frames_processed"], 0)

    def test_frame_route_conflict_without_session(self):
        response = self.post_frame(jpeg_bytes())
        self.assertEqual(response.status_code, 409)
        self.assertIsInstance(response.json()["detail"], str)

    def test_frame_route_rejects_oversized_body_even_with_a_running_session(self):
        self.feed()
        self.start_tab()
        self.assertEqual(self.post_frame(b"x" * (MAX_FRAME_BYTES + 1)).status_code, 413)

    def test_frame_route_stops_reading_a_chunked_body_once_over_the_cap(self):
        self.feed()
        self.start_tab()
        yielded = []

        async def body():
            for index in range(40):
                yielded.append(index)
                yield b"x" * (MAX_FRAME_BYTES // 4)

        response = call(self.app, "POST", "/api/v1/camera/frame", content=body(), headers={"Content-Type": "image/jpeg"})
        self.assertEqual(response.status_code, 413)
        self.assertLess(len(yielded), 40, "the whole oversized body was read before rejecting it")

    def test_frame_route_requires_the_image_jpeg_content_type(self):
        self.feed()
        self.start_tab()
        for headers in ({"Content-Type": "text/plain"}, {"Content-Type": "application/x-www-form-urlencoded"}, {}):
            response = call(self.app, "POST", "/api/v1/camera/frame", content=jpeg_bytes(), headers=headers)
            self.assertEqual(response.status_code, 415, headers)
        accepted = call(self.app, "POST", "/api/v1/camera/frame", content=jpeg_bytes(),
                        headers={"Content-Type": "image/jpeg; charset=binary"})
        self.assertEqual(accepted.status_code, 200)

    def test_frame_route_rejects_garbage_image(self):
        self.feed()
        self.start_tab()
        self.assertEqual(self.post_frame(b"not an image").status_code, 422)

    def test_frame_route_does_not_touch_the_database_per_frame(self):
        self.feed()
        self.start_tab()
        with unittest.mock.patch("drowning.api.app.check_db") as check:
            for _ in range(3):
                self.assertEqual(self.post_frame(jpeg_bytes()).status_code, 200)
            self.assertEqual(self.post_frame(b"not an image").status_code, 422)
            check.assert_not_called()

    def test_start_accepts_fps_and_validates_range(self):
        for bad in (0, 31):
            self.assertEqual(self.start_tab(key=f"camera-tab-bad{bad}", fps=bad).status_code, 422)
        self.feed()
        self.assertEqual(self.start_tab(key="camera-tab-good", fps=15).status_code, 200)


if __name__ == "__main__":
    unittest.main()
