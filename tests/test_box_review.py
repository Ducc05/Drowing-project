"""Confidence threshold selection and per-box ("từng ô") review of image results, with fake inference only."""
import time
import unittest
from types import SimpleNamespace

from sqlalchemy import select
from sqlalchemy.orm import Session

from drowning.api.evaluation import SWEEP_THRESHOLDS, summarize_boxes
from drowning.api.inference import apply_confidence_threshold, box_score, resolve_confidence
from drowning.config import Settings
from drowning.db.models import Job
from tests import test_camera, test_evaluation
from tests.test_camera import call
from tests.test_media_api import FakePipeline, png, request

# Four boxes: scores 0.9 / 0.7 / 0.45 / 0.3 (the last without detector confidence: its label score is used).
BOXES = [
    {"bbox_xyxy": [1, 1, 10, 10], "label": "drowning", "detector_confidence": 0.9, "prediction_branch": "h20saver_frame",
     "class_scores": [0.9, 0.0, 0.0], "score_semantics": "winning_class_confidence_only"},
    {"bbox_xyxy": [12, 1, 20, 10], "label": "swimming", "detector_confidence": 0.7, "prediction_branch": "h20saver_frame",
     "class_scores": [0.0, 0.7, 0.0], "score_semantics": "winning_class_confidence_only"},
    {"bbox_xyxy": [1, 12, 10, 20], "label": "swimming", "detector_confidence": 0.45, "prediction_branch": "h20saver_frame",
     "class_scores": [0.0, 0.45, 0.0], "score_semantics": "winning_class_confidence_only"},
    {"bbox_xyxy": [12, 12, 20, 20], "label": "out_of_water", "prediction_branch": "image",
     "class_scores": [0.2, 0.5, 0.3]},
]


class BoxPipeline(FakePipeline):
    def __init__(self):
        super().__init__()
        self.detector = SimpleNamespace(confidence=0.25)

    def infer_image(self, image):
        self.calls += 1
        return {"model_version": "fixture-v1", "detections": [dict(box) for box in BOXES]}


class ConfidenceHelperTests(unittest.TestCase):
    def test_default_follows_the_engine_and_range_is_enforced(self):
        self.assertEqual(resolve_confidence(Settings.from_env({}), None), 0.25)
        self.assertEqual(resolve_confidence(Settings.from_env({"DETECTION_ENGINE": "h20saver",
                                                               "H20SAVER_CONFIDENCE": "0.4"}), None), 0.4)
        self.assertEqual(resolve_confidence(Settings.from_env({}), 0.55), 0.55)
        for bad in (0.0, 0.01, 0.96, 1.5):
            with self.assertRaises(ValueError):
                resolve_confidence(Settings.from_env({}), bad)

    def test_threshold_is_applied_to_the_detector_and_recorded_on_the_pipeline(self):
        pipeline = BoxPipeline()
        apply_confidence_threshold(pipeline, 0.6)
        self.assertEqual((pipeline.detector.confidence, pipeline.confidence_threshold), (0.6, 0.6))
        bare = SimpleNamespace()  # pipelines without a detector still record the value
        apply_confidence_threshold(bare, 0.3)
        self.assertEqual(bare.confidence_threshold, 0.3)

    def test_box_score_prefers_detector_confidence_then_label_score(self):
        self.assertEqual(box_score(BOXES[0]), 0.9)
        self.assertEqual(box_score(BOXES[3]), 0.3)
        self.assertIsNone(box_score({"label": None, "class_scores": None}))


class BoxSummaryMathTests(unittest.TestCase):
    def marks(self):
        return [{"detection_index": 0, "predicted_label": "drowning", "expected_label": "drowning", "correct": True, "score": 0.9},
                {"detection_index": 1, "predicted_label": "swimming", "expected_label": "swimming", "correct": True, "score": 0.7},
                {"detection_index": 2, "predicted_label": "swimming", "expected_label": "not_person", "correct": False, "score": 0.45},
                {"detection_index": 3, "predicted_label": "out_of_water", "expected_label": "drowning", "correct": False, "score": 0.3}]

    def test_precision_recall_f1_and_confusion_count_each_box_once(self):
        summary = summarize_boxes(self.marks(), missed=1, run_threshold=0.25)
        # 2 correct, 1 wrong label (still a real person), 1 not a person, 1 person without a box -> 4 real people
        self.assertEqual((summary["correct"], summary["wrong_label"], summary["not_person"], summary["real_people"]), (2, 1, 1, 4))
        self.assertEqual((summary["precision"], summary["recall"], summary["f1"]), (0.5, 0.5, 0.5))
        self.assertEqual(summary["confusion_expected_by_predicted"]["drowning"]["out_of_water"], 1)
        self.assertEqual(summary["confusion_expected_by_predicted"]["not_person"]["swimming"], 1)
        self.assertEqual(summary["per_class"]["swimming"]["precision"], 0.5)
        self.assertEqual((summary["mean_score_correct"], summary["mean_score_wrong"]), (0.8, 0.375))

    def test_threshold_sweep_drops_low_boxes_and_marks_unmeasurable_thresholds(self):
        sweep = {row["threshold"]: row for row in summarize_boxes(self.marks(), missed=1, run_threshold=0.25)["threshold_sweep"]}
        self.assertEqual([row["threshold"] for row in sweep.values()], SWEEP_THRESHOLDS)
        self.assertFalse(sweep[0.2]["measurable"])          # below the run threshold: those boxes were never produced
        self.assertTrue(sweep[0.25]["measurable"])
        self.assertEqual((sweep[0.5]["kept"], sweep[0.5]["tp"], sweep[0.5]["fp"], sweep[0.5]["fn"]), (2, 2, 0, 2))
        self.assertEqual((sweep[0.5]["precision"], sweep[0.5]["recall"]), (1.0, 0.5))
        self.assertEqual(sweep[0.95]["kept"], 0)
        self.assertIsNone(sweep[0.95]["precision"])
        self.assertEqual(summarize_boxes(self.marks(), missed=1, run_threshold=0.25)["best_threshold_by_f1"], 0.7)

    def test_empty_review_has_no_rates(self):
        summary = summarize_boxes([])
        self.assertEqual((summary["reviewed_boxes"], summary["precision"], summary["recall"], summary["best_threshold_by_f1"]),
                         (0, None, None, None))


class ConfidenceAndBoxReviewApiTests(unittest.TestCase):
    setUp_base = test_evaluation.EvaluationTests.setUp
    tearDown = test_evaluation.EvaluationTests.tearDown
    video_job = test_evaluation.EvaluationTests.video_job

    def setUp(self):
        self.setUp_base()
        self.pipeline = BoxPipeline()
        self.assertEqual(request(self.app, "POST", f"/api/v1/models/{self.model_id}/sync").status_code, 200)

    def image_job(self, key="box-image-0001", **params):
        params = {"model_version_id": str(self.model_id), **params}
        return request(self.app, "POST", "/api/v1/test/image", params=params, headers={"Idempotency-Key": key},
                       files={"file": ("pool.png", png(), "image/png")})

    def mark(self, job_id, index, expected, reviewer="đức"):
        return request(self.app, "POST", f"/api/v1/jobs/{job_id}/box-marks",
                       json={"reviewer": reviewer, "detection_index": index, "expected_label": expected})

    def test_readiness_reports_the_default_threshold_and_range(self):
        ready = request(self.app, "GET", "/api/v1/test/readiness").json()
        self.assertEqual((ready["default_confidence"], ready["confidence_range"]), (0.25, [0.05, 0.95]))

    def test_chosen_threshold_reaches_the_detector_and_is_stored_with_the_job(self):
        response = self.image_job(confidence=0.4)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["confidence_threshold"], 0.4)
        self.assertEqual(self.pipeline.detector.confidence, 0.4)
        doc = request(self.app, "GET", f"/api/v1/jobs/{result['job_id']}/results/detections").json()
        self.assertEqual(doc["confidence_threshold"], 0.4)
        self.assertEqual([d["box_number"] for d in doc["detections"]], [1, 2, 3, 4])
        self.assertEqual([d["box_score"] for d in doc["detections"]], [0.9, 0.7, 0.45, 0.3])
        jobs = request(self.app, "GET", "/api/v1/jobs").json()["jobs"]
        self.assertEqual(next(j for j in jobs if j["id"] == result["job_id"])["confidence_threshold"], 0.4)
        # same request key with another threshold is a different request
        conflict = self.image_job(confidence=0.6)
        self.assertEqual(conflict.status_code, 409, conflict.text)
        self.assertIn("confidence", conflict.text)

    def test_out_of_range_threshold_is_rejected_before_any_job_is_created(self):
        self.assertEqual(self.image_job(confidence=0.99).status_code, 422)
        self.assertEqual(self.image_job(confidence=0).status_code, 422)
        with Session(self.engine) as session:
            self.assertEqual(session.scalars(select(Job).where(Job.job_type == "inference")).all(), [])

    def test_default_threshold_is_used_when_none_is_chosen(self):
        result = self.image_job().json()
        self.assertEqual((result["confidence_threshold"], self.pipeline.detector.confidence), (0.25, 0.25))

    def test_each_box_is_judged_once_and_summarized_per_image_and_per_model(self):
        job_id = self.image_job(confidence=0.25).json()["job_id"]
        listed = request(self.app, "GET", f"/api/v1/jobs/{job_id}/box-marks").json()
        self.assertEqual([b["box_number"] for b in listed["boxes"]], [1, 2, 3, 4])
        self.assertEqual(listed["confidence_threshold"], 0.25)
        self.assertEqual(self.mark(job_id, 0, "drowning").status_code, 200)
        self.assertEqual(self.mark(job_id, 1, "drowning").status_code, 200)   # first judged wrong ...
        self.mark(job_id, 1, "swimming")                                          # ... then corrected: replaces, not adds
        self.mark(job_id, 2, "not_person")
        view = self.mark(job_id, 3, "drowning").json()
        self.assertEqual(len(view["marks"]), 4)
        self.assertEqual([b["mark"]["correct"] for b in view["boxes"]], [True, True, False, False])
        missed = request(self.app, "PUT", f"/api/v1/jobs/{job_id}/box-marks/missed", json={"reviewer": "đức", "missed": 1})
        self.assertEqual(missed.status_code, 200, missed.text)
        summary = missed.json()["summary"]
        self.assertEqual((summary["precision"], summary["recall"], summary["real_people"]), (0.5, 0.5, 4))
        evaluation = request(self.app, "GET", f"/api/v1/models/{self.model_id}/evaluation").json()
        self.assertEqual((evaluation["box_review"]["reviewed_boxes"], evaluation["box_review"]["images_reviewed"],
                          evaluation["box_review"]["run_threshold"]), (4, 1, 0.25))
        removed = request(self.app, "DELETE", f"/api/v1/jobs/{job_id}/box-marks/2")
        self.assertEqual(len(removed.json()["marks"]), 3)
        self.assertEqual(request(self.app, "DELETE", f"/api/v1/jobs/{job_id}/box-marks/2").status_code, 404)

    def test_invalid_box_marks_are_rejected(self):
        job_id = self.image_job().json()["job_id"]
        self.assertEqual(self.mark(job_id, 9, "drowning").status_code, 422)          # no such box
        self.assertEqual(self.mark(job_id, 0, "floating").status_code, 422)          # not an ontology label
        video = self.video_job().json()["job_id"]
        self.assertEqual(self.mark(video, 0, "drowning").status_code, 422)            # videos are reviewed per track
        self.assertEqual(request(self.app, "PUT", f"/api/v1/jobs/{job_id}/box-marks/missed",
                                 json={"reviewer": "đức", "missed": -1}).status_code, 422)


class CameraConfidenceTests(unittest.TestCase):
    setUp_base = test_camera.CameraApiTests.setUp
    tearDown = test_camera.CameraApiTests.tearDown
    job_state = test_camera.CameraApiTests.job_state

    def setUp(self):
        self.setUp_base()
        self.pipeline.detector = SimpleNamespace(confidence=0.25)

    def test_threshold_is_set_at_start_and_can_change_while_running(self):
        started = call(self.app, "POST", "/api/v1/camera/start",
                       json={"source": "0", "idempotency_key": "camera-conf-1", "confidence": 0.35})
        self.assertEqual(started.status_code, 200, started.text)
        self.assertEqual((started.json()["confidence_threshold"], self.pipeline.detector.confidence), (0.35, 0.35))
        changed = call(self.app, "POST", "/api/v1/camera/confidence", json={"confidence": 0.6})
        self.assertEqual(changed.status_code, 200, changed.text)
        self.assertEqual(self.pipeline.detector.confidence, 0.6)
        time.sleep(0.05)
        self.assertEqual(call(self.app, "GET", "/api/v1/camera/status").json()["confidence_threshold"], 0.6)
        self.assertEqual(self.job_state()[1]["confidence_threshold"], 0.6)
        self.assertEqual(call(self.app, "POST", "/api/v1/camera/confidence", json={"confidence": 1.2}).status_code, 422)
        call(self.app, "POST", "/api/v1/camera/stop")
        self.assertEqual(call(self.app, "POST", "/api/v1/camera/confidence", json={"confidence": 0.5}).status_code, 409)


if __name__ == "__main__":
    unittest.main()
