"""H20Saver engine contracts with a fake YOLO model; no real weights are loaded or evaluated."""
import asyncio
import io
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
import torch
from PIL import Image
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from drowning.api.app import create_app
from drowning.api.inference import InferenceUnavailable, get_approved_model, runtime_weights_missing
from drowning.config import Settings
from drowning.db.models import Base, ExecutionSlot
from drowning.inference.h20saver import (H20SaverDetector, H20SaverInference, class_mapping, detection_scores,
                                         h20saver_identity)
from drowning.inference.pipeline import PersonStateInference
from drowning.inference.tracking import Box

H20SAVER_NAMES = {0: "drowning", 1: "out of water", 2: "swimming"}  # order stored in the H20Saver checkpoint


def fake_boxes(rows):
    """rows: [(x1, y1, x2, y2, confidence, class_id)] in the checkpoint's own class IDs."""
    return SimpleNamespace(boxes=[SimpleNamespace(xyxy=torch.tensor([row[:4]]), conf=torch.tensor([row[4]]),
                                                  cls=torch.tensor([float(row[5])])) for row in rows])


class FakeYOLO:
    names = H20SAVER_NAMES

    def __init__(self, frames=None):
        self.frames = list(frames or [])
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return [fake_boxes(self.frames.pop(0) if self.frames else [])]


class FakeDetector:
    """Feeds pre-decided (box, ontology index) detections to the pipeline, one list per frame."""

    def __init__(self, frames):
        self.frames = list(frames)

    def detect(self, image):
        return self.frames.pop(0)


def settings(**overrides):
    env = {"DETECTION_ENGINE": "h20saver", **overrides}
    return Settings.from_env(env)


class ClassMappingTests(unittest.TestCase):
    def test_checkpoint_names_map_exactly_to_ontology_order(self):
        # ontology v1: 0 drowning, 1 swimming, 2 out_of_water
        self.assertEqual(class_mapping(H20SAVER_NAMES), {0: 0, 1: 2, 2: 1})

    def test_other_class_lists_are_refused(self):
        with self.assertRaises(ValueError):
            class_mapping({0: "person"})
        with self.assertRaises(ValueError):
            class_mapping({0: "drowning", 1: "swimming"})

    def test_detection_scores_put_the_confidence_on_the_detected_class(self):
        scores = detection_scores(1, 0.8)
        self.assertAlmostEqual(scores[1], 0.8)
        self.assertAlmostEqual(sum(scores), 1.0)


class DetectorTests(unittest.TestCase):
    def test_detector_sends_bgr_and_returns_ontology_indices(self):
        model = FakeYOLO([[(1, 2, 30, 40, 0.7, 1)]])
        detector = H20SaverDetector(Path("unused.pt"), confidence=0.3, model=model)
        result = detector.detect(Image.new("RGB", (2, 2), (10, 20, 30)))
        image, kwargs = model.calls[0]
        self.assertEqual(image[0, 0].tolist(), [30, 20, 10])
        self.assertTrue(image.flags["C_CONTIGUOUS"])
        self.assertEqual(kwargs["conf"], 0.3)
        (box, class_index), = result
        self.assertEqual(class_index, 2)  # checkpoint "out of water" -> ontology out_of_water
        self.assertAlmostEqual(box.confidence, 0.7, places=5)


class PipelineTests(unittest.TestCase):
    def run_frames(self, classes, step_ms=125):
        frames = [[(Box(10, 10, 50, 80, confidence=0.8), class_index)] for class_index in classes]
        pipeline = H20SaverInference(FakeDetector(frames), model_version="h20saver-test")
        image = Image.new("RGB", (100, 100))
        return [pipeline.process_frame(image, index * step_ms, index)[0] for index in range(len(classes))]

    def test_sustained_drowning_raises_an_alert_only_after_the_minimum_duration(self):
        records = self.run_frames([0] * 30)
        first = next(index for index, record in enumerate(records) if record["alert"])
        self.assertGreaterEqual(first * 125, 1500)
        self.assertTrue(all(record["alert"] for record in records[first:]))
        self.assertEqual(len({record["track_id"] for record in records}), 1)
        self.assertEqual(records[-1]["prediction_branch"], "h20saver_tracked")

    def test_a_single_drowning_frame_among_swimming_never_alerts(self):
        records = self.run_frames([1] * 10 + [0] + [1] * 20)
        self.assertFalse(any(record["alert"] for record in records))
        self.assertEqual(records[10]["frame_label"], "drowning")
        self.assertEqual(records[10]["label"], "swimming")

    def test_inherits_the_shared_video_and_annotation_interface(self):
        pipeline = H20SaverInference(FakeDetector([]), model_version="h20saver-test")
        self.assertIsInstance(pipeline, PersonStateInference)
        pipeline.reset_stream()
        self.assertEqual(pipeline.tracker.name, "bytetrack_kalman_provisional_ids")

    def test_infer_image_reports_detector_class_and_confidence(self):
        pipeline = H20SaverInference(FakeDetector([[(Box(0, 0, 5, 5, confidence=0.9), 0)]]),
                                     model_version="h20saver-test")
        result = pipeline.infer_image(Image.new("RGB", (8, 8)))
        item, = result["detections"]
        self.assertEqual((item["label"], item["prediction_branch"]), ("drowning", "h20saver_frame"))
        self.assertAlmostEqual(item["class_scores"][0], 0.9)


class AlertDrawingTests(unittest.TestCase):
    def test_alert_record_draws_a_red_banner(self):
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        record = {"track_id": "trk-000001", "bbox_xyxy": [40, 50, 90, 110], "label": "drowning",
                  "class_scores": [0.9, 0.05, 0.05], "prediction_branch": "h20saver_tracked",
                  "detector_confidence": 0.9, "alert": True}
        annotated = PersonStateInference.draw_annotations(frame, [record])
        blue, green, red = annotated[5, 195].tolist()
        self.assertGreater(red, 150)
        self.assertLess(max(blue, green), 80)
        self.assertEqual(PersonStateInference.draw_annotations(frame, [{**record, "alert": False}])[5, 195].tolist(),
                         [0, 0, 0])


class EngineResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.weights = Path(self.temp.name) / "best.pt"
        self.weights.write_bytes(b"h20saver test fixture only")

    def tearDown(self):
        self.temp.cleanup()

    def test_default_model_is_the_h20saver_weights_pinned_by_checksum(self):
        config = settings(H20SAVER_WEIGHTS=str(self.weights))
        model, directory, manifest = get_approved_model(None, config)
        self.assertEqual(model.model_kind, "h20saver")
        self.assertEqual(model.id, h20saver_identity(self.weights).id)
        self.assertEqual(directory, self.weights.parent)
        self.assertEqual(manifest["weights_sha256"], model.weights_sha256)
        self.assertIsNone(runtime_weights_missing(config, model))
        # Re-resolving the pinned job model gives the same identity.
        self.assertEqual(get_approved_model(None, config, pinned_model_id=model.id)[0], model)

    def test_missing_weights_block_only_the_default_h20saver_model(self):
        with self.assertRaisesRegex(InferenceUnavailable, "H20SAVER_WEIGHTS"):
            get_approved_model(None, settings())
        with self.assertRaisesRegex(InferenceUnavailable, "missing"):
            get_approved_model(None, settings(H20SAVER_WEIGHTS=str(self.weights) + ".absent"))

    def test_a_pinned_registry_model_still_uses_the_registry(self):
        class EmptySession:
            def get(self, *args):
                return None

        with self.assertRaisesRegex(InferenceUnavailable, "Pinned job model version is unavailable"):
            get_approved_model(EmptySession(), settings(H20SAVER_WEIGHTS=str(self.weights)),
                               pinned_model_id=uuid.uuid4())

    def test_classifier_engine_still_requires_the_person_detector(self):
        record = SimpleNamespace(model_kind="image_tracked")
        self.assertIn("PERSON_DETECTOR_WEIGHTS", runtime_weights_missing(Settings.from_env({}), record))

    def test_invalid_engine_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            Settings.from_env({"DETECTION_ENGINE": "other"})
        with self.assertRaises(ValueError):
            Settings.from_env({"H20SAVER_CONFIDENCE": "2"})


class H20SaverApiTests(unittest.TestCase):
    """HTTP flow with temporary SQLite and no approved registry model: the engine alone makes the API ready."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.weights = root / "best.pt"
        self.weights.write_bytes(b"h20saver api fixture only")
        self.engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
            connection.execute(text("INSERT INTO alembic_version VALUES ('20261026_01')"))
        with Session(self.engine) as session:
            session.add(ExecutionSlot(slot_id=1, fencing_token=0, reconciliation_status="clear"))
            session.commit()
        config = settings(H20SAVER_WEIGHTS=str(self.weights), INPUT_CACHE_DIR=str(root / "inputs"),
                          OUTPUT_CACHE_DIR=str(root / "results"), WORKER_ID="fixture-worker",
                          SUPERVISOR_ENABLED="false")
        self.app = create_app(settings=config, engine=self.engine, env_file=root / "missing.env")

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def call(self, method, path, **kwargs):
        async def run():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://fixture") as client:
                return await client.request(method, path, **kwargs)
        return asyncio.run(run())

    def test_readiness_and_image_upload_use_the_h20saver_engine(self):
        identity = h20saver_identity(self.weights)
        ready = self.call("GET", "/api/v1/test/readiness").json()
        self.assertTrue(ready["ready_for_request"], ready)
        self.assertEqual((ready["model_version"], ready["detection_engine"]), (identity.display_version, "h20saver"))

        fake = H20SaverInference(FakeDetector([[(Box(4, 4, 30, 30, confidence=0.9), 0)]]),
                                 model_version=identity.display_version)
        stream = io.BytesIO()
        Image.new("RGB", (48, 40), "blue").save(stream, format="PNG")
        with patch("drowning.inference.h20saver.load_h20saver_pipeline", return_value=(fake, identity)):
            response = self.call("POST", "/api/v1/test/image", headers={"Idempotency-Key": "h20saver-image-001"},
                                 files={"file": ("person.png", stream.getvalue(), "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["model_version_id"], str(identity.id))
        detections = self.call("GET", f"/api/v1/jobs/{result['job_id']}/results/detections").json()["detections"]
        self.assertEqual([item["label"] for item in detections], ["drowning"])


if __name__ == "__main__":
    unittest.main()
