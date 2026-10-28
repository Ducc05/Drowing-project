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
from drowning.db.models import Base, ExecutionSlot, ModelRegistry, ModelVersion
from drowning.inference.h20saver import (H20SaverDetector, H20SaverInference, class_mapping, detection_scores,
                                         h20saver_identity, DrowningMonitor, LiveDrowningMonitor)
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

    def track(self, image):
        return [("trk-000001", box, index) for box, index in self.detect(image)]


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
        self.assertEqual(scores, [0.0, 0.8, 0.0])


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
        pipeline.configure_stream(1000 / step_ms)
        image = Image.new("RGB", (100, 100))
        return [pipeline.process_frame(image, index * step_ms, index)[0] for index in range(len(classes))]

    def test_sustained_drowning_raises_an_alert_only_after_the_minimum_duration(self):
        records = self.run_frames([0] * 30)
        first = next(index for index, record in enumerate(records) if record["alert"])
        self.assertEqual(first, 11)
        self.assertTrue(all(record["alert"] for record in records[first:]))
        self.assertEqual(len({record["track_id"] for record in records}), 1)
        self.assertEqual(records[-1]["prediction_branch"], "h20saver_tracked")

    def test_a_single_drowning_frame_among_swimming_never_alerts(self):
        records = self.run_frames([1] * 10 + [0] + [1] * 20)
        self.assertFalse(any(record["alert"] for record in records))
        self.assertEqual(records[10]["frame_label"], "drowning")
        self.assertEqual(records[10]["label"], "drowning")  # raw label is retained; alarm stays off

    def test_inherits_the_shared_video_and_annotation_interface(self):
        pipeline = H20SaverInference(FakeDetector([]), model_version="h20saver-test")
        self.assertIsInstance(pipeline, PersonStateInference)
        pipeline.reset_stream()
        self.assertEqual(pipeline.tracker.name, "ultralytics_bytetrack")

    def test_infer_image_reports_detector_class_and_confidence(self):
        pipeline = H20SaverInference(FakeDetector([[(Box(0, 0, 5, 5, confidence=0.9), 0)]]),
                                     model_version="h20saver-test")
        result = pipeline.infer_image(Image.new("RGB", (8, 8)))
        item, = result["detections"]
        self.assertEqual((item["label"], item["prediction_branch"]), ("drowning", "h20saver_frame"))
        self.assertAlmostEqual(item["class_scores"][0], 0.9)


class ReferenceMonitorTests(unittest.TestCase):
    def test_warmup_hysteresis_and_disappearing_track_expiry(self):
        monitor = DrowningMonitor(10)
        for i in range(6):
            self.assertFalse(monitor.update(i, [1], [True]))
        self.assertEqual(monitor.update(6, [1], [True]), {1})
        for i in range(7, 14):
            self.assertEqual(monitor.update(i, [1], [False]), {1})
        self.assertFalse(monitor.update(14, [1], [False]))
        for i in range(15, 25):
            monitor.update(i, [1], [True])
        self.assertEqual(monitor.update(34, [], []), {1})
        self.assertFalse(monitor.update(35, [], []))

    def test_low_confidence_never_inverts_label_or_creates_alarm(self):
        for class_index in range(3):
            frames = [[(Box(0, 0, 10, 10, confidence=.25), class_index)] for _ in range(90)]
            pipeline = H20SaverInference(FakeDetector(frames), model_version="fixture")
            for i in range(90):
                record, = pipeline.process_frame(Image.new("RGB", (10, 10)), i * 33, i)
                self.assertEqual(record["label"], ["drowning", "swimming", "out_of_water"][class_index])
                self.assertFalse(record["alert"])
                self.assertEqual(record["class_scores"][class_index], .25)

    def test_tracks_keep_independent_votes_and_reset_clears_history(self):
        monitor = DrowningMonitor(10)
        for i in range(10):
            monitor.update(i, [1, 2], [True, False])
        self.assertEqual(monitor.alarmed, {1})
        pipeline = H20SaverInference(FakeDetector([]), model_version="fixture")
        pipeline.monitor = monitor
        pipeline.reset_stream()
        self.assertEqual(pipeline.monitor.history, {})

    def test_detector_uses_persistent_ultralytics_tracking(self):
        model = FakeYOLO()
        model.track = lambda image, **kwargs: (model.calls.append((image, kwargs)) or [SimpleNamespace(
            boxes=SimpleNamespace(id=torch.tensor([7.]), xyxy=torch.tensor([[1., 2., 3., 4.]]),
                                  conf=torch.tensor([.7]), cls=torch.tensor([1.])) )])
        detector = H20SaverDetector(Path("unused.pt"), model=model)
        track_id, box, label = detector.track(Image.new("RGB", (2, 2), (10, 20, 30)))[0]
        self.assertEqual((track_id, label), ("trk-000007", 2))
        image, kwargs = model.calls[0]
        self.assertEqual(image[0, 0].tolist(), [30, 20, 10])
        self.assertEqual((kwargs["persist"], kwargs["tracker"], kwargs["imgsz"]), (True, "bytetrack.yaml", 640))

    def test_video_forces_every_frame_and_initializes_two_second_window(self):
        pipeline = H20SaverInference(FakeDetector([]), model_version="fixture")
        pipeline.configure_stream(25)
        self.assertEqual(pipeline.monitor.window_frames, 50)
        with patch.object(PersonStateInference, "infer_video", return_value={}) as infer:
            pipeline.infer_video(Path("fixture.mp4"), Path("result.webm"), frame_stride=3)
        self.assertEqual(infer.call_args.kwargs["frame_stride"], 1)


class PersistentAlarmTests(unittest.TestCase):
    def test_live_window_uses_elapsed_time_not_processed_frame_count(self):
        for step in (50, 100, 200):
            monitor = LiveDrowningMonitor(1000 / 30)
            first = None
            for timestamp in range(0, 2001, step):
                if monitor.update(timestamp, ["a"], [True]) and first is None:
                    first = timestamp
            self.assertIsNotNone(first)
            self.assertLessEqual(abs(first - 1400), step)
            self.assertEqual(monitor.update(4000, [], []), {"a"})
            self.assertFalse(monitor.update(4001, [], []))

    def test_live_long_gap_requires_new_evidence(self):
        monitor = LiveDrowningMonitor(100)
        for timestamp in range(0, 2000, 100):
            monitor.update(timestamp, [1], [True])
        self.assertEqual(monitor.alarmed, {1})
        self.assertFalse(monitor.update(5000, [1], [True]))

    def test_synthetic_video_persists_missing_track_alarm_without_fake_boxes(self):
        import cv2
        import json
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "fixture.avi", Path(folder) / "result.webm"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 8, (32, 32))
            for _ in range(40):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()
            original = source.read_bytes()
            frames = [[(Box(1, 1, 20, 20, confidence=.8), 0)] for _ in range(12)] + [[] for _ in range(28)]
            pipeline = H20SaverInference(FakeDetector(frames), model_version="synthetic")
            previews = []
            pipeline.preview_callback = previews.append
            result = pipeline.infer_video(source, output, frame_stride=3)
            self.assertTrue(previews)
            self.assertEqual(cv2.imdecode(np.frombuffer(previews[0], dtype=np.uint8), cv2.IMREAD_COLOR).shape[:2], (32, 32))
            data = json.loads(Path(result["detections_path"]).read_text())
            self.assertEqual(result["processed_frames"], 40)
            self.assertEqual(data["frame_stride"], 1)
            self.assertEqual(len(data["detections"]), 12)
            interval, = data["alert_intervals"]
            self.assertEqual(interval["start"], 1375)
            self.assertEqual(interval["end"], 3500)
            self.assertEqual(source.read_bytes(), original)
            pipeline.reset_stream()
            self.assertEqual(pipeline.result_metadata()["alert_intervals"], [])


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

    def test_model_list_replaces_legacy_approved_row_with_imported_best_pt(self):
        legacy_id = uuid.uuid4()
        with Session(self.engine) as session:
            session.add(ModelVersion(id=legacy_id, model_family="drowning-person-temporal-v1",
                                     display_version="kaggle-8", schema_version=1,
                                     class_keys=["drowning", "swimming", "out_of_water"],
                                     temporal_trained=False, completion_kind="candidate", status="approved",
                                     metrics={"model_kind": "image_tracked"}))
            session.add(ModelRegistry(model_family="drowning-person-temporal-v1",
                                      approved_model_id=legacy_id, generation=1))
            session.commit()
        identity = h20saver_identity(self.weights)
        models = self.call("GET", "/api/v1/models").json()["models"]
        self.assertEqual([model["display_version"] for model in models], ["best.pt"])
        current = models[0]
        self.assertEqual((current["id"], current["status"], current["model_kind"], current["source"]),
                         (str(identity.id), "active", "h20saver", "imported_h20saver"))
        self.assertTrue(current["cached_locally"])
        self.assertEqual(current["checkpoint"]["weights_sha256"], identity.weights_sha256)
        evaluation = self.call("GET", f"/api/v1/models/{identity.id}/evaluation").json()
        self.assertEqual((evaluation["display_version"], evaluation["status"]), ("best.pt", "active"))
        self.assertEqual(evaluation["training_metrics"], {})
        self.assertIn("Imported H20Saver", evaluation["training_metrics_note"])


if __name__ == "__main__":
    unittest.main()
