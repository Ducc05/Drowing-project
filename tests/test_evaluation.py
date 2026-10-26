"""Candidate evaluation: sync an unapproved candidate, test with it explicitly, mark video segments."""
import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from drowning.api.app import create_app
from drowning.config import Settings
from drowning.db.models import Artifact, Base, ExecutionSlot, Job, ModelVersion
from drowning.integrations.drive import sha256_file
from tests.test_candidate_sync import make_candidate_archive
from tests.test_cvat_adapter import make_video
from tests.test_media_api import FakePipeline, png, request
from tests.test_model_sync import DownloadDrive

DETECTIONS = [{"frame_index": i, "timestamp_ms": i * 100, "track_id": track, "label": label,
               "prediction_branch": "temporal", "bbox_xyxy": [1, 1, 20, 30]}
              for i in range(5) for track, label in (("trk-000001", "swimming"), ("trk-000002", "drowning"))]


class VideoPipeline(FakePipeline):
    def infer_video(self, source, destination, **kwargs):
        result = super().infer_video(source, destination, **kwargs)
        Path(result["detections_path"]).write_text(json.dumps({
            "model_version": "fixture", "detections": DETECTIONS, "scores_are_calibrated_probabilities": False}),
            encoding="utf-8")
        return result


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model_id = uuid4()
        self.archive = make_candidate_archive(self.root, self.model_id.hex)
        manifest = json.loads((self.root / "bundle/model-bundle/manifest.json").read_text(encoding="utf-8"))
        self.metrics = json.loads((self.root / "bundle/metrics.json").read_text(encoding="utf-8"))
        self.engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
            connection.execute(text("INSERT INTO alembic_version VALUES ('20261026_01')"))
        with Session(self.engine) as session:
            job = Job(job_type="train", state="completed", phase="complete", idempotency_key="src", payload={})
            session.add(job)
            session.flush()
            artifact = Artifact(job_id=job.id, role="candidate_bundle", remote_file_id="drive-archive", status="ready",
                                sha256=sha256_file(self.archive), size_bytes=self.archive.stat().st_size)
            session.add(artifact)
            session.flush()
            session.add(ModelVersion(id=self.model_id, model_family=manifest["model_family"], display_version="cand-1",
                                      schema_version=1, class_keys=manifest["class_keys"], temporal_trained=True,
                                      completion_kind="candidate", status="candidate", artifact_id=artifact.id,
                                      dataset_fingerprint=manifest["dataset_fingerprint"], metrics=self.metrics))
            session.add(ExecutionSlot(slot_id=1, fencing_token=0, reconciliation_status="clear"))
            session.commit()
        self.drive, self.pipeline = DownloadDrive(self.archive), VideoPipeline()
        settings = Settings.from_env({"APPROVED_MODEL_BUNDLE_DIR": str(self.root / "cache"),
                                      "INPUT_CACHE_DIR": str(self.root / "inputs"), "OUTPUT_CACHE_DIR": str(self.root / "results"),
                                      "WORKER_ID": "eval-worker"})
        self.app = create_app(settings=settings, engine=self.engine, env_file=self.root / "missing.env",
                              drive_factory=lambda: self.drive, inference_factory=lambda *a: self.pipeline)
        make_video(self.root / "clip.avi")

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def video_job(self, key="eval-video-1", model=True):
        params = {"model_version_id": str(self.model_id)} if model else {}
        return request(self.app, "POST", "/api/v1/test/video", params=params, headers={"Idempotency-Key": key},
                       files={"file": ("clip.avi", (self.root / "clip.avi").read_bytes(), "video/x-msvideo")})

    def test_every_model_exposes_a_verifiable_checkpoint_reference(self):
        listed = request(self.app, "GET", "/api/v1/models").json()["models"][0]["checkpoint"]
        self.assertEqual(listed["drive_url"], "https://drive.google.com/file/d/drive-archive/view")
        self.assertEqual(len(listed["archive_sha256"]), 64)
        detail = request(self.app, "GET", f"/api/v1/models/{self.model_id}/checkpoint")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["checkpoint"]["drive_file_id"], "drive-archive")
        self.assertEqual(request(self.app, "GET", f"/api/v1/models/{uuid4()}/checkpoint").status_code, 404)

    def test_candidate_is_listed_with_metrics_synced_and_tested_only_when_explicitly_selected(self):
        models = request(self.app, "GET", "/api/v1/models").json()["models"]
        self.assertEqual(models[0]["status"], "candidate")
        self.assertEqual(models[0]["metrics"]["temporal_validation"]["macro_f1"], 0.2)
        self.assertFalse(models[0]["cached_locally"])
        self.assertEqual(self.video_job(model=True).status_code, 409)  # not synced yet
        synced = request(self.app, "POST", f"/api/v1/models/{self.model_id}/sync")
        self.assertEqual(synced.status_code, 200, synced.text)
        self.assertEqual(synced.json()["status"], "candidate")
        self.assertTrue(request(self.app, "GET", "/api/v1/models").json()["models"][0]["cached_locally"])
        implicit = self.video_job("eval-video-implicit", model=False)
        self.assertEqual(implicit.status_code, 409)
        self.assertIn("No approved model", implicit.text)
        explicit = self.video_job()
        self.assertEqual(explicit.status_code, 200, explicit.text)
        self.assertEqual(explicit.json()["model_version_id"], str(self.model_id))
        with Session(self.engine) as session:
            job = session.query(Job).filter(Job.idempotency_key == "eval-video-1").one()
            self.assertTrue(job.payload["evaluation"])
            self.assertEqual(job.payload["model_status_at_request"], "candidate")
            self.assertEqual(session.get(ExecutionSlot, 1).job_id, None)
        self.assertEqual(request(self.app, "POST", f"/api/v1/models/{uuid4()}/sync").status_code, 409)

    def mark(self, job_id, **overrides):
        body = {"reviewer": "Dung", "track_id": "trk-000001", "start_ms": 0, "end_ms": 300, "verdict": "correct"}
        body.update(overrides)
        return request(self.app, "POST", f"/api/v1/jobs/{job_id}/review-marks", json=body)

    def test_review_marks_use_server_side_predictions_and_summarize_per_class(self):
        request(self.app, "POST", f"/api/v1/models/{self.model_id}/sync")
        job_id = self.video_job().json()["job_id"]
        correct = self.mark(job_id)
        self.assertEqual(correct.status_code, 200, correct.text)
        self.assertEqual((correct.json()["predicted_label"], correct.json()["expected_label"]), ("swimming", "swimming"))
        self.assertEqual(correct.json()["frames"], 3)
        wrong = self.mark(job_id, track_id="trk-000002", end_ms=500, verdict="wrong", expected_label="out_of_water",
                          note="Person is standing on the deck")
        self.assertEqual(wrong.status_code, 200, wrong.text)
        self.assertEqual(wrong.json()["predicted_label"], "drowning")
        self.assertEqual(self.mark(job_id, track_id="trk-000001", start_ms=2000, end_ms=3000).status_code, 422)
        self.assertEqual(self.mark(job_id, track_id="trk-404").status_code, 422)
        self.assertEqual(self.mark(job_id, start_ms=300, end_ms=300).status_code, 422)
        self.assertEqual(self.mark(job_id, reviewer=" ").status_code, 422)
        self.assertEqual(self.mark(job_id, verdict="maybe").status_code, 422)
        listing = request(self.app, "GET", f"/api/v1/jobs/{job_id}/review-marks").json()
        summary = listing["summary"]
        self.assertEqual((summary["segments"], summary["correct"], summary["wrong"]), (2, 1, 1))
        self.assertEqual(summary["confusion_expected_by_predicted"]["out_of_water"]["drowning"], 1)
        self.assertEqual(summary["per_class"]["swimming"]["recall"], 1.0)
        self.assertEqual(summary["per_class"]["out_of_water"]["recall"], 0.0)
        self.assertIsNone(summary["per_class"]["drowning"]["recall"])
        self.assertIn("not a benchmark", summary["caveat"])
        evaluation = request(self.app, "GET", f"/api/v1/models/{self.model_id}/evaluation").json()
        self.assertEqual(evaluation["human_review"]["segments"], 2)
        self.assertEqual(evaluation["test_runs"], {"videos": 1, "images": 0, "videos_with_review_marks": 1})
        self.assertEqual(evaluation["training_metrics"]["temporal_validation"]["macro_f1"], 0.2)
        deleted = request(self.app, "DELETE", f"/api/v1/jobs/{job_id}/review-marks/{wrong.json()['id']}")
        self.assertEqual(deleted.json()["remaining"], 1)
        self.assertEqual(request(self.app, "DELETE", f"/api/v1/jobs/{job_id}/review-marks/nope").status_code, 404)

    def test_wrong_without_expected_label_is_counted_but_stays_out_of_the_confusion_matrix(self):
        request(self.app, "POST", f"/api/v1/models/{self.model_id}/sync")
        job_id = self.video_job().json()["job_id"]
        self.mark(job_id, verdict="wrong")
        summary = request(self.app, "GET", f"/api/v1/jobs/{job_id}/review-marks").json()["summary"]
        self.assertEqual((summary["wrong"], summary["wrong_without_expected_label"]), (1, 1))
        self.assertEqual(sum(sum(row.values()) for row in summary["confusion_expected_by_predicted"].values()), 0)

    def test_image_jobs_and_unknown_jobs_cannot_be_marked(self):
        request(self.app, "POST", f"/api/v1/models/{self.model_id}/sync")
        image = request(self.app, "POST", "/api/v1/test/image", params={"model_version_id": str(self.model_id)},
                        headers={"Idempotency-Key": "eval-image-1"}, files={"file": ("a.png", png(), "image/png")})
        self.assertEqual(image.status_code, 200, image.text)
        self.assertEqual(self.mark(image.json()["job_id"]).status_code, 422)
        self.assertEqual(self.mark(uuid4()).status_code, 404)
        self.assertEqual(request(self.app, "GET", f"/api/v1/models/{uuid4()}/evaluation").status_code, 404)


if __name__ == "__main__":
    unittest.main()
