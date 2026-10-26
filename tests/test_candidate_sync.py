import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from uuid import UUID, uuid4

import torch
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from drowning.db.models import Artifact, Base, ExecutionSlot, Job, ModelVersion
from drowning.domain.jobs import JobState
from drowning.db.repositories import StaleExecutionOwner
from drowning.integrations.drive import RemoteUploadUnconfirmed
from drowning.ml.bundle import save_bundle
from drowning.ml.model import DrowningTemporalModel
from drowning.orchestration.candidate_sync import CandidateArtifactSync, verify_candidate_archive


class FakeCollectorKaggle:
    def __init__(self, archive):
        self.archive = archive
        self.ref_seen = None

    def collect_exact_output(self, ref, destination):
        self.ref_seen = ref
        shutil.copyfile(self.archive, destination / "candidate-output.zip")
        return None


class FakeArtifactDrive:
    def __init__(self):
        self.fail_next = False
        self.uploads = 0

    def upload(self, path, **kwargs):
        self.uploads += 1
        if self.fail_next:
            self.fail_next = False
            raise RemoteUploadUnconfirmed("fake interrupted upload")
        return {"file_id": "drive-candidate-1", "verified": True, "sha256": "unused"}


def make_candidate_archive(directory: Path, model_version_id=None) -> Path:
    bundle_dir = directory / "bundle" / "model-bundle"
    bundle_dir.parent.mkdir(parents=True)
    model = DrowningTemporalModel(hidden_size=8).eval()
    save_bundle(model, bundle_dir, completion_kind="candidate", temporal_trained=True,
                preprocessing={"size": [32, 32], "context_fraction": 0.15,
                               "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                dataset_fingerprint="d" * 64,
                temporal_contract={"sequence_length": 2, "sample_period_ms": 100,
                                   "max_track_gap_ms": 500, "causal": True},
                recipe_fingerprint="r" * 64, code_fingerprint="c" * 64, model_version_id=model_version_id)
    metrics = {"completion_kind": "candidate", "approved": False,
               "dataset_fingerprint": "d" * 64, "test_split_used": False,
               "image_validation": {"macro_f1": 0.3}, "temporal_validation": {"macro_f1": 0.2}}
    if model_version_id:
        metrics["candidate_id"] = model_version_id
    (directory / "bundle" / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    archive = directory / "candidate-output.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path in sorted((directory / "bundle").rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(directory / "bundle").as_posix())
    return archive


class CandidateSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.archive = make_candidate_archive(self.root)
        self.kaggle = FakeCollectorKaggle(self.archive)
        self.drive = FakeArtifactDrive()
        self.sync = CandidateArtifactSync(self.factory, self.kaggle, self.drive,
                                          drive_folder_id="folder-id", cache_root=self.root / "cache",
                                          worker_id="worker-a")

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def make_collecting_job(self):
        job_id = uuid4()
        with self.factory() as session:
            job = Job(id=job_id, job_type="train", state=JobState.COLLECTING.value, phase="collecting",
                      idempotency_key=f"candidate-{job_id}", payload={"mode": "train_candidate"},
                      owner_id="worker-a", execution_token=4, remote_ref="owner/kernel", remote_version="17",
                      remote_status="completed")
            session.add(job)
            session.flush()
            session.add(ExecutionSlot(slot_id=1, job_id=job_id, owner_id="worker-a", fencing_token=4,
                                      reconciliation_status="owned"))
            session.commit()
        return job_id

    def test_exact_candidate_archive_verifies_bundle_and_holdout_contract(self):
        result = verify_candidate_archive(self.archive, self.root / "verify")
        self.assertEqual(result["manifest"]["completion_kind"], "candidate")
        self.assertFalse(result["metrics"]["approved"])
        self.assertEqual(len(result["archive_sha256"]), 64)

    def test_stale_upload_receipt_cannot_register_artifact_or_candidate(self):
        job_id = self.make_collecting_job()
        def stale_upload(*args, **kwargs):
            with self.factory() as session:
                job = session.get(Job, job_id)
                job.owner_id, job.execution_token = "new-worker", 5
                slot = session.get(ExecutionSlot, 1)
                slot.owner_id, slot.fencing_token = "new-worker", 5
                session.commit()
            return {"verified": True, "file_id": "stale-output"}
        self.drive.upload = stale_upload
        with self.assertRaises(StaleExecutionOwner):
            self.sync.collect(job_id)
        with self.factory() as session:
            self.assertEqual(session.scalars(select(Artifact)).all(), [])
            self.assertEqual(session.scalars(select(ModelVersion)).all(), [])
            self.assertEqual(session.get(ExecutionSlot, 1).owner_id, "new-worker")

    def test_unverified_drive_receipt_cannot_publish_candidate(self):
        job_id = self.make_collecting_job()
        self.drive.upload = lambda *args, **kwargs: {"verified": False, "file_id": "unverified-output"}
        with self.assertRaisesRegex(ValueError, "verified artifact receipt"):
            self.sync.collect(job_id)
        with self.factory() as session:
            self.assertEqual(session.scalars(select(ModelVersion)).all(), [])
            self.assertEqual(session.scalars(select(Artifact)).all(), [])
            self.assertEqual(session.get(Job, job_id).state, "sync_pending")
            self.assertIsNone(session.get(ExecutionSlot, 1).job_id)

    def test_candidate_row_carries_checkpoint_provenance_for_sharing(self):
        job_id = self.make_collecting_job()
        with self.factory() as session:
            job = session.get(Job, job_id)
            job.payload = {**job.payload, "training_identity": "i" * 64, "options": {"pretrained": False}}
            session.commit()
        result = self.sync.collect(job_id)
        with self.factory() as session:
            model = session.get(ModelVersion, UUID(result["model_version_id"]))
            provenance = model.metrics["provenance"]
            artifact = session.get(Artifact, model.artifact_id)
        archive_manifest = verify_candidate_archive(self.archive, self.root / "again")["manifest"]
        self.assertEqual(provenance["weights_sha256"], archive_manifest["files"]["weights.pt"]["sha256"])
        self.assertEqual((provenance["training_identity"], provenance["drive_file_id"]), ("i" * 64, "drive-candidate-1"))
        self.assertEqual(provenance["code_fingerprint"], "c" * 64)
        self.assertEqual(provenance["archive_sha256"], artifact.sha256)

    def test_collect_exact_output_registers_candidate_and_releases_confirmed_compute(self):
        job_id = self.make_collecting_job()
        result = self.sync.collect(job_id)
        self.assertEqual(self.kaggle.ref_seen, "owner/kernel/17")
        self.assertEqual(result["state"], "completed")
        self.assertFalse(result["approved"])
        with self.factory() as session:
            job = session.get(Job, job_id)
            slot = session.get(ExecutionSlot, 1)
            model = session.scalar(select(ModelVersion).where(ModelVersion.training_job_id == job_id))
            artifact = session.scalar(select(Artifact).where(Artifact.job_id == job_id))
            self.assertEqual(job.state, "completed")
            self.assertIsNone(slot.job_id)
            self.assertEqual(model.status, "candidate")
            self.assertEqual(artifact.status, "ready")

    def test_drive_failure_becomes_sync_pending_then_idempotent_retry(self):
        job_id = self.make_collecting_job()
        self.drive.fail_next = True
        with self.assertRaises(RemoteUploadUnconfirmed):
            self.sync.collect(job_id)
        with self.factory() as session:
            job = session.get(Job, job_id)
            self.assertEqual(job.state, "sync_pending")
            self.assertIsNone(session.get(ExecutionSlot, 1).job_id)
            self.assertTrue(Path(job.payload["local_candidate_archive"]).is_file())
        result = self.sync.retry_sync(job_id)
        self.assertEqual(result["state"], "completed")
        self.assertEqual(self.drive.uploads, 2)
        with self.factory() as session:
            model = session.scalar(select(ModelVersion).where(ModelVersion.training_job_id == job_id))
            self.assertEqual(model.status, "candidate")

    def test_candidate_archive_rejects_extra_secret_file(self):
        bad = self.root / "bad.zip"
        with zipfile.ZipFile(bad, "w") as archive:
            archive.write(self.archive, "candidate-output.zip")
            archive.writestr(".env", "DATABASE_URL=secret")
        with self.assertRaisesRegex(ValueError, "unexpected file"):
            verify_candidate_archive(bad, self.root / "bad-extract")


if __name__ == "__main__":
    unittest.main()
