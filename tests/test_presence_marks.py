"""Presence marks: 'there are N people here', independent of what the detector found, so misses are countable."""
import unittest
from uuid import uuid4

from drowning.api.evaluation import presence_for, summarize_presence
from tests.test_evaluation import EvaluationTests
from tests.test_media_api import request


class PresenceMarkApiTests(unittest.TestCase):
    setUp = EvaluationTests.setUp
    tearDown = EvaluationTests.tearDown
    video_job = EvaluationTests.video_job

    def start(self):
        self.assertEqual(request(self.app, "POST", f"/api/v1/models/{self.model_id}/sync").status_code, 200)
        job = self.video_job()
        self.assertEqual(job.status_code, 200, job.text)
        return job.json()["job_id"]

    def mark(self, job_id, **body):
        payload = {"reviewer": "đức", "start_ms": 0, "end_ms": 500, "people": 2, "expected_label": "swimming", "note": ""}
        payload.update(body)
        return request(self.app, "POST", f"/api/v1/jobs/{job_id}/presence-marks", json=payload)

    def test_misses_false_alarms_and_label_checks_are_computed_from_the_stored_detections(self):
        job_id = self.start()
        seen = self.mark(job_id, start_ms=0, end_ms=500, people=2, expected_label="drowning")
        self.assertEqual(seen.status_code, 200, seen.text)
        # the fixture has two boxes on each of 5 frames (100 ms apart): both people were found, one labelled drowning, one swimming
        self.assertEqual((seen.json()["detected_people_max"], seen.json()["frames_with_detection"]), (2, 5))
        missed = self.mark(job_id, start_ms=1000, end_ms=2000, people=1, expected_label="swimming").json()
        self.assertEqual((missed["detected_boxes"], missed["detected_people_max"], missed["predicted_label"]), (0, 0, None))
        false_alarm = self.mark(job_id, start_ms=0, end_ms=300, people=0, expected_label=None).json()
        self.assertGreater(false_alarm["detected_boxes"], 0)
        listed = request(self.app, "GET", f"/api/v1/jobs/{job_id}/presence-marks").json()
        summary = listed["summary"]
        self.assertEqual((summary["intervals"], summary["with_people"], summary["empty"]), (3, 2, 1))
        self.assertEqual(summary["interval_recall_any_person"], 0.5)      # one of the two occupied intervals was missed
        self.assertEqual(summary["interval_recall_full_count"], 0.5)
        self.assertEqual(summary["empty_false_alarm_rate"], 1.0)           # the empty interval still had detections
        self.assertEqual(summary["label_checked_intervals"], 1)
        evaluation = request(self.app, "GET", f"/api/v1/models/{self.model_id}/evaluation").json()
        self.assertEqual(evaluation["presence_review"]["intervals"], 3)
        removed = request(self.app, "DELETE", f"/api/v1/jobs/{job_id}/presence-marks/{missed['id']}")
        self.assertEqual(removed.json()["remaining"], 2)
        self.assertEqual(request(self.app, "DELETE", f"/api/v1/jobs/{job_id}/presence-marks/nope").status_code, 404)

    def test_invalid_marks_are_rejected_with_a_clear_reason(self):
        job_id = self.start()
        self.assertEqual(self.mark(job_id, start_ms=500, end_ms=500).status_code, 422)
        self.assertEqual(self.mark(job_id, people=51).status_code, 422)
        self.assertEqual(self.mark(job_id, expected_label="floating").status_code, 422)
        empty_with_label = self.mark(job_id, people=0, expected_label="swimming")
        self.assertEqual(empty_with_label.status_code, 422)
        self.assertIn("nobody", empty_with_label.text)
        self.assertEqual(request(self.app, "POST", f"/api/v1/jobs/{uuid4()}/presence-marks",
                                 json={"reviewer": "x", "start_ms": 0, "end_ms": 5, "people": 1}).status_code, 404)


class PresenceMathTests(unittest.TestCase):
    def test_expected_frames_follow_fps_and_stride_and_never_divide_by_zero(self):
        doc = {"fps_metadata": 30.0, "frame_stride": 3, "detections": [
            {"frame_index": 0, "timestamp_ms": 0, "label": "swimming"}, {"frame_index": 3, "timestamp_ms": 100, "label": "swimming"}]}
        result = presence_for(doc, 0, 1000)  # frames 0..29, every 3rd -> 10 looked at; 2 had a person
        self.assertEqual((result["expected_frames"], result["frames_with_detection"], result["detection_rate"]), (10, 2, 0.2))
        self.assertEqual(presence_for({"detections": []}, 0, 10)["detection_rate"], 0.0)  # no fps: still safe

    def test_empty_summary_has_no_rates(self):
        summary = summarize_presence([])
        self.assertEqual((summary["intervals"], summary["interval_recall_any_person"], summary["empty_false_alarm_rate"]), (0, None, None))


if __name__ == "__main__":
    unittest.main()
