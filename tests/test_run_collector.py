import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from drowning.db.models import Artifact, Base, ExecutionSlot, Job
from drowning.domain.jobs import JobState
from drowning.integrations.drive import RemoteUploadUnconfirmed
from drowning.orchestration.kaggle_run import PACKAGE_NAME, REPORT_NAME, package_output, run_prepare, run_smoke
from drowning.orchestration.run_collector import RunOutputCollector, verify_run_archive
from tests.test_kaggle_run import ROOT, make_yolo_export


class FakeKaggle:
    def __init__(self, archive: Path):
        self.archive, self.refs = archive, []

    def collect_exact_output(self, ref, destination):
        self.refs.append(ref)
        shutil.copyfile(self.archive, destination / PACKAGE_NAME)


class FakeDrive:
    def __init__(self):
        self.fail = False
        self.uploads = []

    def upload(self, path, **kwargs):
        if self.fail:
            raise RemoteUploadUnconfirmed("fake interrupted upload")
        self.uploads.append(kwargs["role"])
        return {"verified": True, "file_id": "drive-run-output"}


def zip_with(path: Path, files: dict):
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return path


class RunCollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        inputs = self.base / "inputs"
        make_yolo_export(inputs / "pool-export")
        run_prepare(sources_dir=ROOT / "configs" / "datasets", reviews_dir=self.base / "reviews", input_root=inputs,
                    output_dir=self.base / "prepare-out", work_dir=self.base / "work", code_fingerprint="a" * 64)
        self.prepare_archive = package_output(self.base / "prepare-out", self.base / "prepare.zip")
        run_smoke(output_dir=self.base / "smoke-out", code_fingerprint="a" * 64)
        self.smoke_archive = package_output(self.base / "smoke-out", self.base / "smoke.zip")
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.drive = FakeDrive()

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def collector(self, archive, drive=True):
        self.kaggle = FakeKaggle(archive)
        return RunOutputCollector(self.factory, self.kaggle, cache_root=self.base / "cache", worker_id="worker-a",
                                  drive=self.drive if drive else None, drive_folder_id="folder" if drive else "")

    def collecting_job(self, mode="prepare_only", job_type="prepare"):
        job_id = uuid4()
        with self.factory() as session:
            session.add(Job(id=job_id, job_type=job_type, state=JobState.COLLECTING.value, phase="collecting",
                            idempotency_key=f"run-{job_id}", payload={"mode": mode, "executor": "kaggle"},
                            owner_id="worker-a", execution_token=3, remote_ref="owner/kernel", remote_version="5",
                            remote_status="completed"))
            session.flush()
            session.add(ExecutionSlot(slot_id=1, job_id=job_id, owner_id="worker-a", fencing_token=3,
                                      reconciliation_status="owned"))
            session.commit()
        return job_id

    def state(self, job_id):
        with self.factory() as session:
            return session.get(Job, job_id).state, session.get(ExecutionSlot, 1).job_id

    def test_prepare_output_is_verified_summarized_synced_and_slot_released(self):
        job_id = self.collecting_job()
        result = self.collector(self.prepare_archive).collect(job_id)
        self.assertEqual(result["state"], "completed")
        self.assertEqual(self.kaggle.refs, ["owner/kernel/5"])
        self.assertEqual(self.state(job_id), ("completed", None))
        self.assertEqual(self.drive.uploads, ["run_output"])
        with self.factory() as session:
            job = session.get(Job, job_id)
            summary = job.payload["run_summary"]
            self.assertEqual(summary["run_mode"], "prepare_only")
            self.assertEqual(summary["sources"][0]["status"], "prepared")
            self.assertIn("no_temporal_video_source_prepared", summary["blockers"])
            self.assertEqual(job.payload["result_storage"], "local_and_drive")
            self.assertTrue((Path(job.payload["run_report_dir"]) / REPORT_NAME).is_file())
            self.assertEqual(len(session.scalars(select(Artifact)).all()), 1)

    def test_smoke_output_records_gpu_confirmation_without_a_model(self):
        job_id = self.collecting_job("remote_smoke", "train")
        self.collector(self.smoke_archive, drive=False).collect(job_id)
        self.assertEqual(self.state(job_id), ("completed", None))
        with self.factory() as session:
            summary = session.get(Job, job_id).payload["run_summary"]
            self.assertEqual(summary["run_mode"], "remote_smoke")
            self.assertEqual(summary["gpu_confirmed"], summary["environment"]["cuda_available"])
            self.assertTrue(summary["gradients_ok"])
            self.assertEqual(session.get(Job, job_id).payload["result_storage"], "local")
            self.assertEqual(session.scalars(select(Artifact)).all(), [])

    def test_drive_failure_goes_sync_pending_frees_slot_and_retry_needs_no_compute(self):
        job_id = self.collecting_job()
        collector = self.collector(self.prepare_archive)
        self.drive.fail = True
        self.assertEqual(collector.collect(job_id)["state"], "sync_pending")
        self.assertEqual(self.state(job_id), ("sync_pending", None))
        self.drive.fail = False
        self.assertEqual(collector.retry_sync(job_id)["state"], "completed")
        self.assertEqual(self.kaggle.refs, ["owner/kernel/5"])
        self.assertEqual(self.drive.uploads, ["run_output"])

    def test_download_failure_goes_sync_pending_and_retry_collects_the_exact_version_again(self):
        job_id = self.collecting_job()
        collector = self.collector(self.prepare_archive, drive=False)
        original = self.kaggle.collect_exact_output
        self.kaggle.collect_exact_output = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("network down"))
        with self.assertRaises(RuntimeError):
            collector.collect(job_id)
        self.assertEqual(self.state(job_id), ("sync_pending", None))
        self.kaggle.collect_exact_output = original
        self.assertEqual(collector.retry_sync(job_id)["state"], "completed")
        self.assertEqual(self.state(job_id), ("completed", None))
        with self.factory() as session:
            self.assertEqual(session.get(Job, job_id).payload["run_summary"]["run_mode"], "prepare_only")

    def test_wrong_mode_or_invalid_archives_fail_the_job_and_release_the_slot(self):
        mismatch = self.collecting_job("remote_smoke", "train")
        with self.assertRaisesRegex(ValueError, "wrong mode"):
            self.collector(self.prepare_archive).collect(mismatch)
        self.assertEqual(self.state(mismatch), ("failed", None))

    def test_unsafe_oversized_or_dishonest_archives_are_rejected(self):
        good = json.dumps({"schema_version": 1, "run_mode": "prepare_only", "training_started": False, "model_candidate": False})
        cases = {
            "traversal": ({"../escape.json": "{}", REPORT_NAME: good}, "unsafe path"),
            "wrong type": ({"payload.exe": "x", REPORT_NAME: good}, "disallowed file type"),
            "no report": ({"other.json": "{}"}, "missing"),
            "claims training": ({REPORT_NAME: json.dumps({"schema_version": 1, "run_mode": "prepare_only",
                                                           "training_started": True, "model_candidate": False})}, "wrong mode"),
            "claims candidate": ({REPORT_NAME: json.dumps({"schema_version": 1, "run_mode": "prepare_only",
                                                            "training_started": False, "model_candidate": True})}, "wrong mode"),
        }
        for name, (files, message) in cases.items():
            archive = zip_with(self.base / f"{name.replace(' ', '_')}.zip", files)
            with self.assertRaisesRegex(ValueError, message, msg=name):
                verify_run_archive(archive, self.base / f"x-{name.replace(' ', '_')}", expected_mode="prepare_only")
        with self.assertRaises(ValueError):
            verify_run_archive(self.base / "missing.zip", self.base / "y", expected_mode="prepare_only")

    def test_collection_requires_collecting_state_and_exact_version(self):
        job_id = self.collecting_job()
        with self.factory() as session:
            session.get(Job, job_id).remote_version = None
            session.commit()
        with self.assertRaisesRegex(ValueError, "exact-version"):
            self.collector(self.prepare_archive).collect(job_id)
        self.assertEqual(self.state(job_id)[0], "collecting")

    def test_candidate_jobs_are_not_handled_by_this_collector(self):
        job_id = self.collecting_job("train_candidate", "train")
        with self.assertRaisesRegex(ValueError, "prepare_only/remote_smoke"):
            self.collector(self.prepare_archive).collect(job_id)


if __name__ == "__main__":
    unittest.main()
