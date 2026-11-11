"""Choosing H20Saver weights from the models folder: scan, list, and per-run pinning (image/video + camera)."""
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from PIL import Image
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from drowning.api.app import create_app
from drowning.config import Settings
from drowning.db.models import Base, ExecutionSlot, Job
from drowning.inference.h20saver import H20SaverInference, h20saver_identity
from drowning.inference.local_weights import scan_local_weights
from drowning.inference.tracking import Box
from tests.test_camera import FakeStreamPipeline, ScriptedCapture, call
from tests.test_h20saver import FakeDetector


class ModelsFolder(unittest.TestCase):
    """models/h20saver/best.pt is the configured default; the rest are alternatives the user can pick."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.models = self.root / "models"
        self.default = self.write("h20saver/best.pt", b"default weights")
        self.new = self.write("best-new/new.pt", b"new weights")
        self.write("best-new/z-copy.pt", b"new weights")  # same content as new.pt: the same model
        self.write("old.zip", b"not weights")  # downloads and unpacked checkpoint folders are not models
        self.write("h20saver_best/best/data.pkl", b"unpacked")

    def tearDown(self):
        self.temp.cleanup()

    def write(self, relative, content):
        path = self.models / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path


class ScanTests(ModelsFolder):
    def test_lists_only_pt_files_default_first_and_names_are_relative(self):
        found = scan_local_weights(self.models, self.default)
        self.assertEqual([item.name for item in found], ["h20saver/best.pt", "best-new/new.pt"])
        self.assertEqual([item.is_default for item in found], [True, False])
        self.assertEqual(found[1].path, self.new.resolve())

    def test_identity_is_the_weights_checksum_so_copies_collapse_to_one_model(self):
        found = scan_local_weights(self.models, self.default)
        self.assertEqual(found[1].identity.id, h20saver_identity(self.new).id)
        self.assertEqual(len({item.identity.id for item in found}), len(found))

    def test_default_outside_the_models_folder_is_still_listed_first_by_file_name(self):
        elsewhere = self.root / "elsewhere" / "pinned.pt"
        elsewhere.parent.mkdir()
        elsewhere.write_bytes(b"outside weights")
        found = scan_local_weights(self.models, elsewhere)
        self.assertEqual(found[0].name, "pinned.pt")
        self.assertTrue(found[0].is_default)
        self.assertEqual(len(found), 3)

    def test_without_a_folder_only_the_default_is_listed_and_nothing_is_scanned(self):
        found = scan_local_weights(None, self.default)
        self.assertEqual([item.name for item in found], ["best.pt"])
        self.assertEqual(scan_local_weights(None, None), [])
        self.assertEqual(scan_local_weights(self.root / "missing", None), [])

    def test_models_dir_comes_from_the_environment(self):
        self.assertEqual(Settings.from_env({"H20SAVER_MODELS_DIR": " ./models "}).h20saver_models_dir, "./models")
        self.assertEqual(Settings.from_env({}).h20saver_models_dir, "")


class ApiFixture(ModelsFolder):
    def setUp(self):
        super().setUp()
        self.engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
            connection.execute(text("INSERT INTO alembic_version VALUES ('20261026_01')"))
        with Session(self.engine) as session:
            session.add(ExecutionSlot(slot_id=1, fencing_token=0, reconciliation_status="clear"))
            session.commit()
        config = Settings.from_env({"DETECTION_ENGINE": "h20saver", "H20SAVER_WEIGHTS": str(self.default),
                                    "H20SAVER_MODELS_DIR": str(self.models),
                                    "INPUT_CACHE_DIR": str(self.root / "inputs"),
                                    "OUTPUT_CACHE_DIR": str(self.root / "results"),
                                    "WORKER_ID": "fixture-worker", "SUPERVISOR_ENABLED": "false"})
        self.app = create_app(settings=config, engine=self.engine, env_file=self.root / "missing.env",
                              camera_capture_factory=lambda source: ScriptedCapture(source))
        self.app.state.camera.push_read_timeout = 0.05
        self.loaded = []

    def tearDown(self):
        service = self.app.state.camera
        if service._session is not None:
            try:
                service.stop()
            except Exception:
                pass
        self.engine.dispose()
        super().tearDown()

    def loader(self, make_pipeline):
        def load(path, **kwargs):
            self.loaded.append(Path(path))
            return make_pipeline(), h20saver_identity(Path(path))
        return load

    def inference_jobs(self):
        with Session(self.engine) as session:
            return list(session.scalars(select(Job).where(Job.job_type == "inference")))


class ModelListTests(ApiFixture):
    def test_every_local_weights_file_is_listed_and_only_the_default_is_marked_in_use(self):
        models = call(self.app, "GET", "/api/v1/models").json()["models"]
        self.assertEqual([m["display_version"] for m in models], ["h20saver/best.pt", "best-new/new.pt"])
        self.assertEqual([m["status"] for m in models], ["active", "available"])
        self.assertEqual({m["source"] for m in models}, {"imported_h20saver"})
        self.assertEqual({m["model_kind"] for m in models}, {"h20saver"})
        self.assertEqual(models[1]["id"], str(h20saver_identity(self.new).id))
        self.assertEqual(models[1]["checkpoint"]["size_bytes"], self.new.stat().st_size)

    def test_evaluation_summary_works_for_a_non_default_local_model(self):
        new_id = h20saver_identity(self.new).id
        response = call(self.app, "GET", f"/api/v1/models/{new_id}/evaluation")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual((body["display_version"], body["model_version_id"]), ("best-new/new.pt", str(new_id)))
        self.assertEqual(call(self.app, "GET", f"/api/v1/models/{uuid4()}/evaluation").status_code, 404)


def png_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (48, 40), "blue").save(stream, format="PNG")
    return stream.getvalue()


class ImageRunTests(ApiFixture):
    def fake_image_pipeline(self):
        return H20SaverInference(FakeDetector([[(Box(4, 4, 30, 30, confidence=0.9), 0)]]), model_version="fixture")

    def post_image(self, key, **query):
        with patch("drowning.inference.h20saver.load_h20saver_pipeline",
                   side_effect=self.loader(self.fake_image_pipeline)):
            return call(self.app, "POST", "/api/v1/test/image", params=query,
                        headers={"Idempotency-Key": key}, files={"file": ("p.png", png_bytes(), "image/png")})

    def test_run_pinned_to_a_local_model_loads_that_file_and_records_its_id(self):
        new_id = h20saver_identity(self.new).id
        response = self.post_image("local-model-image-1", model_version_id=str(new_id))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.loaded, [self.new.resolve()])
        self.assertEqual(response.json()["model_version_id"], str(new_id))

    def test_run_without_a_choice_still_uses_the_configured_default(self):
        response = self.post_image("local-model-image-2")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.loaded, [self.default.resolve()])
        self.assertEqual(response.json()["model_version_id"], str(h20saver_identity(self.default).id))

    def test_unknown_model_id_is_refused_and_nothing_is_loaded(self):
        response = self.post_image("local-model-image-3", model_version_id=str(uuid4()))
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.loaded, [])


class CameraRunTests(ApiFixture):
    def start(self, key, **extra):
        with patch("drowning.inference.h20saver.load_h20saver_pipeline",
                   side_effect=self.loader(FakeStreamPipeline)):
            return call(self.app, "POST", "/api/v1/camera/start",
                        json={"source": "rtsp://cam.local/live", "idempotency_key": key, **extra})

    def test_camera_started_with_a_local_model_runs_that_file(self):
        new_id = h20saver_identity(self.new).id
        started = self.start("camera-local-model-1", model_version_id=str(new_id))
        self.assertEqual(started.status_code, 200, started.text)
        self.assertEqual(self.loaded, [self.new.resolve()])
        self.assertEqual(self.inference_jobs()[0].payload["model_version_id"], str(new_id))
        self.assertEqual(call(self.app, "POST", "/api/v1/camera/stop").status_code, 200)

    def test_camera_without_a_choice_runs_the_configured_default(self):
        started = self.start("camera-local-model-2")
        self.assertEqual(started.status_code, 200, started.text)
        self.assertEqual(self.loaded, [self.default.resolve()])
        self.assertEqual(call(self.app, "POST", "/api/v1/camera/stop").status_code, 200)

    def test_camera_refuses_an_unknown_model_and_leaves_no_job_or_held_slot(self):
        refused = self.start("camera-local-model-3", model_version_id=str(uuid4()))
        self.assertEqual(refused.status_code, 409, refused.text)
        self.assertEqual(self.loaded, [])
        self.assertEqual(self.inference_jobs(), [])
        with Session(self.engine) as session:
            self.assertIsNone(session.get(ExecutionSlot, 1).job_id)


if __name__ == "__main__":
    unittest.main()
