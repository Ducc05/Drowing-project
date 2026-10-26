"""Real bundle/archive/file checks with fake Drive and fake inference, no training."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from drowning.api.app import create_app
from drowning.api.inference import get_approved_model
from drowning.config import Settings
from drowning.db.models import Artifact, Base, ExecutionSlot, Job, ModelVersion
from drowning.integrations.drive import sha256_file
from tests.test_candidate_sync import make_candidate_archive
from tests.test_media_api import FakePipeline, png, request


class DownloadDrive:
    def __init__(self, archive):
        self.archive, self.calls, self.corrupt = archive, 0, False

    def download(self, file_id, destination, **kwargs):
        self.calls += 1
        if self.corrupt:
            destination.write_bytes(b"interrupted fake download")
        else:
            shutil.copyfile(self.archive, destination)
        return {"verified": True}


class ApprovedModelSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model_id = uuid4()
        self.archive = make_candidate_archive(self.root, self.model_id.hex)
        self.drive, self.pipeline = DownloadDrive(self.archive), FakePipeline()
        self.engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        with self.engine.begin() as connection:
            connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
            connection.execute(text("INSERT INTO alembic_version VALUES ('20261026_01')"))
        manifest = json.loads((self.root / "bundle/model-bundle/manifest.json").read_text(encoding="utf-8"))
        with Session(self.engine) as session:
            job = Job(job_type="train", state="completed", phase="complete", idempotency_key="sync-source", payload={})
            session.add(job)
            session.flush()
            artifact = Artifact(job_id=job.id, role="candidate_bundle", storage_provider="google_drive",
                                remote_file_id="fake-approved-archive", status="ready",
                                sha256=sha256_file(self.archive), size_bytes=self.archive.stat().st_size)
            session.add(artifact)
            session.flush()
            session.add(ModelVersion(id=self.model_id, model_family=manifest["model_family"], display_version="sync-fixture",
                                      schema_version=1, class_keys=manifest["class_keys"], temporal_trained=True,
                                      completion_kind="candidate", status="candidate", artifact_id=artifact.id,
                                      dataset_fingerprint=manifest["dataset_fingerprint"], metrics={}))
            session.add(ExecutionSlot(slot_id=1, fencing_token=0, reconciliation_status="clear"))
            session.commit()
        self.settings = Settings.from_env({"APPROVED_MODEL_BUNDLE_DIR": str(self.root / "cache"),
                                           "INPUT_CACHE_DIR": str(self.root / "inputs"),
                                           "OUTPUT_CACHE_DIR": str(self.root / "results"), "WORKER_ID": "sync-fixture-worker"})
        self.app = create_app(settings=self.settings, engine=self.engine,
                              env_file=self.root / "missing.env", drive_factory=lambda: self.drive,
                              inference_factory=lambda *args: self.pipeline)

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def approve(self):
        response = request(self.app, "POST", f"/api/v1/models/{self.model_id}/review",
                           json={"decision": "approve", "reviewer_id": "fixture-reviewer"})
        self.assertEqual(response.status_code, 200, response.text)

    def test_candidate_approve_sync_upload_download_entire_flow(self):
        self.approve()
        for expected_downloaded in (True, False):
            response = request(self.app, "POST", "/api/v1/models/sync-approved")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["downloaded"], expected_downloaded)
        self.assertEqual(self.drive.calls, 1)
        with Session(self.engine) as session:
            model, directory, manifest = get_approved_model(session, self.settings)
            self.assertEqual(directory.name, str(self.model_id))
            self.assertEqual(model.id, self.model_id)
        response = request(self.app, "POST", "/api/v1/test/image", headers={"Idempotency-Key": "whole-flow-fixture"},
                           files={"file": ("fixture.png", png(), "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["state"], "sync_pending")  # Drive output folder intentionally unset.
        output = request(self.app, "GET", response.json()["files"]["annotated"])
        self.assertEqual(output.status_code, 200)
        self.assertEqual(output.headers["content-type"], "image/png")

    def test_interrupted_or_wrong_checksum_download_preserves_existing_versions(self):
        self.approve()
        prior = self.root / "cache" / "older-version"
        prior.mkdir(parents=True)
        (prior / "keep.txt").write_text("existing cache", encoding="utf-8")
        self.drive.corrupt = True
        response = request(self.app, "POST", "/api/v1/models/sync-approved")
        self.assertEqual(response.status_code, 503)
        self.assertFalse((prior.parent / str(self.model_id)).exists())
        self.assertEqual((prior / "keep.txt").read_text(), "existing cache")
        self.assertEqual(list(prior.parent.glob(".model-sync-*")), [])
        self.drive.corrupt = False
        self.assertEqual(request(self.app, "POST", "/api/v1/models/sync-approved").status_code, 200)

    def test_candidate_cannot_be_installed_as_approved(self):
        response = request(self.app, "POST", "/api/v1/models/sync-approved")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.drive.calls, 0)
