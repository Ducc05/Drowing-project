"""Per-run choice of the cell grid (toggle + columns x rows): API validation, storage and hand-over to the loader."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import select
from sqlalchemy.orm import Session

from drowning.api.inference import USE_SETTINGS, cell_grid_info, load_approved_pipeline, resolve_cell_grid
from drowning.config import Settings
from drowning.db.models import Job
from drowning.inference.cells import format_cell_grid
from tests import test_camera, test_evaluation
from tests.test_camera import call
from tests.test_media_api import png, request


class CellGridHelperTests(unittest.TestCase):
    def test_format_and_resolve(self):
        self.assertEqual((format_cell_grid((6, 4)), format_cell_grid(None)), ("6x4", "off"))
        settings = Settings.from_env({})
        self.assertEqual(resolve_cell_grid(settings, None), (6, 4))
        self.assertIsNone(resolve_cell_grid(settings, "off"))
        self.assertEqual(resolve_cell_grid(settings, " 8X6 "), (8, 6))
        for bad in ("0x4", "33x4", "abc", "6x", ""):
            with self.assertRaises(ValueError, msg=bad):
                resolve_cell_grid(settings, bad)

    def test_info_reports_default_range_and_engine_support(self):
        info = cell_grid_info(Settings.from_env({}))
        self.assertEqual((info["default_cell_grid"], info["cell_grid_range"], info["cell_grid_supported"]),
                         ("6x4", [1, 32], False))
        h20 = cell_grid_info(Settings.from_env({"DETECTION_ENGINE": "h20saver", "H20SAVER_CELL_GRID": "off"}))
        self.assertEqual((h20["default_cell_grid"], h20["cell_grid_supported"]), ("off", True))

    def test_requested_grid_reaches_the_h20saver_loader(self):
        identity = SimpleNamespace(display_version="h20", weights_sha256="s" * 64, model_kind="h20saver")
        settings = Settings.from_env({"DETECTION_ENGINE": "h20saver"})
        with patch("drowning.api.inference._h20saver_model", return_value=(identity, Path("best.pt"))), \
                patch("drowning.inference.h20saver.load_h20saver_pipeline",
                      return_value=(object(), identity)) as loader:
            load_approved_pipeline(None, settings, cell_grid=None)
            self.assertIsNone(loader.call_args.kwargs["cell_grid"])
            load_approved_pipeline(None, settings, cell_grid=(8, 6))
            self.assertEqual(loader.call_args.kwargs["cell_grid"], (8, 6))
            load_approved_pipeline(None, settings)
            self.assertEqual(loader.call_args.kwargs["cell_grid"], (6, 4))
        self.assertIsNotNone(USE_SETTINGS)


class VideoCellGridApiTests(unittest.TestCase):
    setUp_base = test_evaluation.EvaluationTests.setUp
    tearDown = test_evaluation.EvaluationTests.tearDown
    video_job = test_evaluation.EvaluationTests.video_job

    def setUp(self):
        self.setUp_base()
        self.assertEqual(request(self.app, "POST", f"/api/v1/models/{self.model_id}/sync").status_code, 200)

    def post_video(self, key, grid):
        params = {"model_version_id": str(self.model_id), "cell_grid": grid}
        return request(self.app, "POST", "/api/v1/test/video", params=params, headers={"Idempotency-Key": key},
                       files={"file": ("clip.avi", (self.root / "clip.avi").read_bytes(), "video/x-msvideo")})

    def inference_jobs(self):
        with Session(self.engine) as session:
            return session.scalars(select(Job).where(Job.job_type == "inference")).all()

    def test_video_job_stores_normalised_grid_and_rejects_a_changed_one(self):
        response = self.post_video("grid-video-0001", " 8X6 ")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["cell_grid"], "8x6")
        listed = request(self.app, "GET", "/api/v1/jobs").json()["jobs"]
        self.assertEqual(next(j for j in listed if j["id"] == response.json()["job_id"])["cell_grid"], "8x6")
        conflict = self.post_video("grid-video-0001", "off")
        self.assertEqual(conflict.status_code, 409, conflict.text)
        self.assertIn("cell_grid", conflict.text)

    def test_default_grid_is_stored_when_none_is_chosen(self):
        response = self.video_job("grid-video-0002")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["cell_grid"], "6x4")

    def test_invalid_grid_creates_no_job(self):
        for bad in ("0x4", "33x4", "abc"):
            self.assertEqual(self.post_video("grid-video-0003", bad).status_code, 409, bad)
        self.assertEqual(self.inference_jobs(), [])

    def test_image_ignores_cell_grid(self):
        response = request(self.app, "POST", "/api/v1/test/image",
                           params={"model_version_id": str(self.model_id), "cell_grid": "off"},
                           headers={"Idempotency-Key": "grid-image-0001"},
                           files={"file": ("pool.png", png(), "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json().get("cell_grid"))
        self.assertNotIn("cell_grid", self.inference_jobs()[0].payload)

    def test_readiness_reports_grid_defaults(self):
        ready = request(self.app, "GET", "/api/v1/test/readiness").json()
        self.assertEqual((ready["default_cell_grid"], ready["cell_grid_range"], ready["cell_grid_supported"]),
                         ("6x4", [1, 32], False))


class CameraCellGridTests(unittest.TestCase):
    setUp = test_camera.CameraApiTests.setUp
    tearDown = test_camera.CameraApiTests.tearDown
    job_state = test_camera.CameraApiTests.job_state

    def test_camera_start_stores_grid_and_rejects_invalid(self):
        started = call(self.app, "POST", "/api/v1/camera/start",
                       json={"source": "0", "idempotency_key": "camera-grid-1", "cell_grid": "5x3"})
        self.assertEqual(started.status_code, 200, started.text)
        self.assertEqual(self.job_state()[1]["cell_grid"], "5x3")
        call(self.app, "POST", "/api/v1/camera/stop")

    def test_invalid_camera_grid_is_422_and_creates_no_job(self):
        bad = call(self.app, "POST", "/api/v1/camera/start",
                   json={"source": "0", "idempotency_key": "camera-grid-2", "cell_grid": "9x"})
        self.assertEqual(bad.status_code, 422, bad.text)
        with Session(self.engine) as session:
            self.assertEqual(session.scalars(select(Job).where(Job.job_type == "inference")).all(), [])


if __name__ == "__main__":
    unittest.main()
