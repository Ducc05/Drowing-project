import json
import tempfile
import unittest
from pathlib import Path

try:
    import cv2
    import torch
    from PIL import Image
    from drowning.inference.detector import PersonDetector
    from drowning.inference.pipeline import PersonStateInference
    from drowning.inference.tracking import Box, GreedyIoUTracker
    from drowning.ml.model import CLASS_KEYS, ONTOLOGY_ID, ONTOLOGY_VERSION, MODEL_SCHEMA_VERSION
    from drowning.ml.preprocessing import crop_person, image_to_tensor
except ImportError:
    cv2 = torch = Image = None
    PersonStateInference = GreedyIoUTracker = Box = None
    crop_person = image_to_tensor = None
    CLASS_KEYS = ()
    ONTOLOGY_ID = ""
    ONTOLOGY_VERSION = MODEL_SCHEMA_VERSION = 0


class FixedPersonDetector:
    def detect(self, image):
        return [Box(4, 4, min(24, image.width - 1), min(26, image.height - 1), 0.9)]


class FakeClassifier(torch.nn.Module if torch is not None else object):
    def forward_images(self, images):
        scores = torch.zeros((images.shape[0], len(CLASS_KEYS)), device=images.device)
        scores[:, 1] = 2.0
        return scores

    def forward(self, frames, valid_mask=None):
        batch, timesteps = frames.shape[:2]
        image_logits = torch.zeros((batch, timesteps, len(CLASS_KEYS)), device=frames.device)
        temporal_logits = torch.zeros_like(image_logits)
        temporal_logits[:, :, 2] = 3.0
        return {"image_logits": image_logits, "temporal_logits": temporal_logits}


def approved_test_manifest():
    return {
        "bundle_schema_version": 1,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "model_family": "drowning-person-temporal-v1",
        "encoder": "mobilenet_v3_small",
        "temporal_head": "causal_gru_cell",
        "hidden_size": 8,
        "class_keys": list(CLASS_KEYS),
        "ontology_id": ONTOLOGY_ID,
        "ontology_version": ONTOLOGY_VERSION,
        "temporal_trained": True,
        "completion_kind": "candidate",
        "preprocessing": {"size": 32, "context_fraction": 0.15,
                          "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "temporal_contract": {"sequence_length": 2, "sample_period_ms": 100,
                              "max_track_gap_ms": 500, "causal": True},
    }


class InferencePipelineTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "install optional ML dependencies to test preprocessing")
    def test_person_crop_and_tensor_follow_rgb_normalization_contract(self):
        image = Image.new("RGB", (50, 60), color=(255, 0, 0))
        crop = crop_person(image, (10, 10, 30, 40), context_fraction=0.1, output_size=32)
        tensor = image_to_tensor(crop, size=32)
        self.assertEqual(crop.size, (32, 32))
        self.assertEqual(tuple(tensor.shape), (3, 32, 32))
        self.assertGreater(float(tensor[0].mean()), float(tensor[1].mean()))

    @unittest.skipIf(torch is None, "install optional ML dependencies to test inference")
    def test_image_inference_keeps_each_person_detection_and_uses_fixed_class_order(self):
        model = FakeClassifier()
        pipeline = PersonStateInference(model, approved_test_manifest(), FixedPersonDetector(),
                                         model_version="candidate-v1", sequence_length=2,
                                         sample_period_ms=100, max_track_gap_ms=500, input_size=32)
        result = pipeline.infer_image(Image.new("RGB", (32, 32), "blue"))
        self.assertEqual(result["detections"][0]["label"], "swimming")
        self.assertEqual(result["detections"][0]["prediction_branch"], "image")
        self.assertFalse(result["detections"][0]["identity_verified"])

    @unittest.skipIf(cv2 is None or torch is None, "install optional vision and ML dependencies to test video plumbing")
    def test_video_pipeline_writes_annotated_video_and_marks_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
            self.assertTrue(writer.isOpened())
            for _ in range(4):
                writer.write(__import__("numpy").zeros((32, 32, 3), dtype="uint8"))
            writer.release()
            manifest = approved_test_manifest()
            manifest["temporal_contract"]["sample_period_ms"] = 1
            manifest["temporal_contract"]["max_track_gap_ms"] = 500
            pipeline = PersonStateInference(FakeClassifier(), manifest, FixedPersonDetector(),
                                             model_version="candidate-v1", sequence_length=2,
                                             sample_period_ms=1, max_track_gap_ms=500, input_size=32)
            result = pipeline.infer_video(source, root / "result.webm")
            self.assertEqual(result["frame_count"], 4)
            self.assertTrue((root / "result.webm").is_file())
            self.assertIn(b"V_VP8", (root / "result.webm").read_bytes())
            capture = cv2.VideoCapture(str(root / "result.webm"))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
            capture.release()
            records = json.loads((root / "result.detections.json").read_text(encoding="utf-8"))
            self.assertTrue(any(item["prediction_branch"] == "image_warmup" for item in records["detections"]))
            self.assertTrue(any(item["prediction_branch"] == "temporal" for item in records["detections"]))
            self.assertFalse(records["identity_verified"])
            self.assertFalse(records["scores_are_calibrated_probabilities"])

    @unittest.skipIf(torch is None, "install optional ML dependencies to test bundle guard")
    def test_inference_refuses_synthetic_smoke_bundle(self):
        manifest = approved_test_manifest()
        manifest["completion_kind"] = "synthetic_smoke"
        with self.assertRaisesRegex(ValueError, "synthetic smoke"):
            PersonStateInference(FakeClassifier(), manifest, FixedPersonDetector(), model_version="smoke")

    @unittest.skipIf(torch is None, "install optional ML dependencies to test tracker")
    def test_iou_tracker_reuses_track_id_and_expires_missing_tracks(self):
        tracker = GreedyIoUTracker(min_iou=0.2, max_missed_frames=1)
        first = tracker.update([Box(0, 0, 10, 10)])[0][0]
        second = tracker.update([Box(1, 0, 11, 10)])[0][0]
        self.assertEqual(first, second)
        tracker.update([])
        tracker.update([])
        new_id = tracker.update([Box(1, 0, 11, 10)])[0][0]
        self.assertNotEqual(first, new_id)


if __name__ == "__main__":
    unittest.main()
