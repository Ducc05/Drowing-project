import asyncio
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import httpx
import yaml
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from drowning.api.app import create_app
from drowning.api.review import SourceReviewService
from drowning.api.train import TrainBlocked, TrainService
from drowning.config import Settings
from drowning.data.registry import load_source
from drowning.data.review import record_grouping_review, record_mapping_review
from drowning.db.models import Artifact, Base, ExecutionSlot, Job, ModelRegistry, ModelVersion
from drowning.orchestration.kaggle_dispatch import KaggleJobDispatcher
from drowning.orchestration.kaggle_run import run_prepare, package_output, sha256_file
from drowning.orchestration.run_collector import summarize_report
from tests.test_kaggle_run import ROOT, make_yolo_export

NAMES = ["Drowning", "Swimming", "out of water"]
ENV = {"KAGGLE_KERNEL_REF": "owner/drowning-temporal", "KAGGLE_DATASET_SOURCES": "owner/pool-data",
       "GOOGLE_DRIVE_FOLDER_ID": "folder"}


class FakeKaggle:
    @staticmethod
    def validate_kernel_ref(ref, exact_version=False):
        return ref


class TrainServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.sources = self.base / "configs" / "datasets"
        self.sources.mkdir(parents=True)
        shutil.copy(ROOT / "configs" / "datasets" / "alanoud_swimming_drowning_v4.yaml", self.sources)
        self.reviews_dir = self.base / "reviews"
        self.token = self.base / "google-token.json"
        self.token.write_text("{}", encoding="utf-8")
        self.engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.factory() as session:
            session.add(ExecutionSlot(slot_id=1, fencing_token=0, reconciliation_status="clear"))
            session.commit()
        self.reviews = SourceReviewService(self.sources, reviews_dir=self.reviews_dir, work_dir=self.base / "work")

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def policy(self, mode):
        path = self.base / f"policy-{mode}.yaml"
        path.write_text(yaml.safe_dump({"version": 1, "gating": {"mode": mode}}), encoding="utf-8")
        return path

    def service(self, env=None, *, configured=True, cli=True, credentials=True, drive=True, mode="enforced"):
        values = {**ENV, "KAGGLE_KERNEL_DIR": str(self.base / "kernels"), "GOOGLE_OAUTH_TOKEN_FILE": str(self.token)}
        if not configured:
            values = {k: v for k, v in values.items() if not k.startswith("KAGGLE_KERNEL_REF") and k != "KAGGLE_DATASET_SOURCES"}
        if not drive:
            values.pop("GOOGLE_DRIVE_FOLDER_ID", None)
        values.update(env or {})
        return TrainService(self.factory, Settings.from_env(values), project_root=ROOT, reviews=self.reviews,
                            which=(lambda name: "kaggle" if cli else None), home=self.base / "home",
                            environ={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"} if credentials else {},
                            policy_path=self.policy(mode))

    def approve_everything(self):
        source = load_source(self.sources / "alanoud_swimming_drowning_v4.yaml", self.reviews_dir)
        record_mapping_review(source, reviewer="Dung", confirmed_labels=NAMES, samples_viewed={n: 24 for n in NAMES},
                              reviews_dir=self.reviews_dir)
        record_grouping_review(source, reviewer="Dung", by_path={"a.jpg": "g1"}, method="manual", evidence={},
                               note="Checked cross-split pairs on the review sheets by hand.", reviews_dir=self.reviews_dir)

    def completed_prepare(self, *, windows=0):
        inputs = self.base / "inputs"
        if not inputs.exists():
            make_yolo_export(inputs / "pool-export")
        output = self.base / f"prep-{uuid4().hex[:6]}"
        report = run_prepare(sources_dir=self.sources, reviews_dir=self.reviews_dir, input_root=inputs,
                             output_dir=output, work_dir=self.base / f"work-prep-{uuid4().hex[:6]}",
                             code_fingerprint="a" * 64)
        summary = summarize_report(report)
        summary["temporal_window_count_before_split"] = windows
        archive = package_output(output, self.base / f"prepare-{uuid4().hex}.zip")
        return self.insert_job("prepare", "prepare_only", "completed", run_summary=summary,
                               run_archive=str(archive), run_archive_sha256=sha256_file(archive))

    def completed_smoke(self, gpu=True):
        summary = {"run_mode": "remote_smoke", "gpu_confirmed": gpu, "gradients_ok": True, "device": "cuda" if gpu else "cpu"}
        return self.insert_job("train", "remote_smoke", "completed", run_summary=summary)

    def insert_job(self, job_type, mode, state, **payload):
        with self.factory() as session:
            job = Job(job_type=job_type, state=state, phase="x", idempotency_key=f"seed-{uuid4()}",
                      payload={"mode": mode, "executor": "kaggle", **payload})
            session.add(job)
            session.commit()
            return job.id

    # -- gating ----------------------------------------------------------------------------
    def test_nothing_is_allowed_until_kaggle_is_configured(self):
        view = self.service(configured=False, cli=False, credentials=False).preflight()
        for mode in ("prepare_only", "remote_smoke", "train_candidate"):
            self.assertFalse(view["modes"][mode]["allowed"], mode)
        joined = " ".join(view["modes"]["prepare_only"]["blocked_by"])
        for text in ("Kaggle CLI", "credentials", "KAGGLE_KERNEL_REF", "KAGGLE_DATASET_SOURCES"):
            self.assertIn(text, joined)
        self.assertFalse(view["environment"]["kaggle_cli_installed"])

    def test_prepare_is_allowed_once_configured_but_smoke_needs_a_reviewed_prepare(self):
        service = self.service()
        view = service.preflight()
        self.assertTrue(view["modes"]["prepare_only"]["allowed"])
        self.assertIn("run prepare_only first", " ".join(view["modes"]["remote_smoke"]["blocked_by"]))
        self.completed_prepare()
        blocked = " ".join(service.preflight()["modes"]["remote_smoke"]["blocked_by"])
        self.assertIn("mapping not approved", blocked)
        self.approve_everything()
        # approvals change the mapping_review overlay but not the mapping, so the prepare is still current
        view = service.preflight()
        self.assertEqual(view["stale_sources"], [])
        self.assertTrue(view["modes"]["remote_smoke"]["allowed"], view["modes"]["remote_smoke"])

    def test_advisory_review_never_blocks_smoke_or_train_and_says_what_is_unconfirmed(self):
        service = self.service(mode="advisory")
        self.completed_prepare(windows=6)          # nothing was approved or verified
        self.completed_smoke()
        view = service.preflight()
        self.assertEqual(view["review_mode"], "advisory")
        self.assertTrue(view["modes"]["remote_smoke"]["allowed"], view["modes"]["remote_smoke"])
        self.assertTrue(view["modes"]["train_candidate"]["allowed"], view["modes"]["train_candidate"])
        self.assertTrue(any("not confirmed" in text for text in view["unconfirmed_sources"]))
        result = service.submit("train_candidate", new_version=True, reason="Advisory review, nobody confirmed yet",
                                idempotency_key="train-advisory-1")
        self.assertEqual(result["status"], "queued")
        with self.factory() as session:
            job = session.get(Job, UUID_FROM(result["job"]["id"]))
        self.assertEqual(job.payload["preflight"]["review_mode"], "advisory")
        self.assertFalse(job.payload["preflight"]["mapping_review_approved"])
        dispatcher = KaggleJobDispatcher(self.factory, FakeKaggle(), worker_id="w")
        self.assertEqual(dispatcher._validate_job(job, Path(job.payload["kernel_dir"])), "owner/drowning-temporal")

    def test_enforced_mode_still_requires_review_when_asked(self):
        service = self.service(mode="enforced")
        self.completed_prepare(windows=6)
        self.completed_smoke()
        view = service.preflight()
        self.assertFalse(view["modes"]["train_candidate"]["allowed"])
        self.assertIn("mapping not approved", " ".join(view["modes"]["train_candidate"]["blocked_by"]))

    def test_changed_mapping_makes_the_last_prepare_stale(self):
        service = self.service()
        self.completed_prepare()
        self.approve_everything()
        path = self.sources / "alanoud_swimming_drowning_v4.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data["mapping"]["Swimming"] = {"action": "review"}
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        view = service.preflight()
        self.assertEqual(view["stale_sources"], ["alanoud_swimming_drowning"])
        self.assertIn("mapping changed after the last prepare", " ".join(view["modes"]["remote_smoke"]["blocked_by"]))

    # -- submission ------------------------------------------------------------------------
    def test_prepare_carries_verified_source_cache_and_ignores_tampered_archives(self):
        service = self.service()
        prior_id = self.completed_prepare()
        first = service.submit("prepare_only", new_version=False, reason="", idempotency_key="prepare-cached-1")
        with self.factory() as session:
            job = session.get(Job, UUID_FROM(first["job"]["id"]))
            self.assertEqual(job.payload["prepare_cache_source_versions"], 1)
            kernel_dir = Path(job.payload["kernel_dir"])
            metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
            notebook = json.loads((kernel_dir / metadata["code_file"]).read_text(encoding="utf-8"))
            self.assertEqual(notebook["metadata"]["drowning_project"]["prepare_cache_source_versions"], 1)
            job.state = "cancelled"
            archive = Path(session.get(Job, prior_id).payload["run_archive"])
            session.commit()
        archive.write_bytes(archive.read_bytes() + b"tampered")
        second = service.submit("prepare_only", new_version=False, reason="", idempotency_key="prepare-cached-2")
        with self.factory() as session:
            self.assertEqual(session.get(Job, UUID_FROM(second["job"]["id"])).payload["prepare_cache_source_versions"], 0)

    def test_prepare_submission_writes_a_private_kernel_and_is_idempotent(self):
        service = self.service()
        result = service.submit("prepare_only", new_version=False, reason="", idempotency_key="prepare-request-1")
        self.assertEqual((result["status"], result["created"]), ("queued", True))
        again = service.submit("prepare_only", new_version=False, reason="", idempotency_key="prepare-request-1")
        self.assertEqual((again["status"], again["job"]["id"]), ("duplicate_request", result["job"]["id"]))
        with self.factory() as session:
            job = session.scalars(select(Job)).one()
        kernel_dir = Path(job.payload["kernel_dir"])
        metadata = json.loads((kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertTrue(metadata["is_private"])
        self.assertEqual(metadata["dataset_sources"], ["owner/pool-data"])
        notebook = json.loads((kernel_dir / metadata["code_file"]).read_text(encoding="utf-8"))
        self.assertEqual(notebook["metadata"]["drowning_project"]["run_mode_default"], "prepare_only")
        self.assertEqual((job.job_type, job.state), ("prepare", "queued"))
        self.assertEqual(job.payload["push_timeout_seconds"], 43200)
        dispatcher = KaggleJobDispatcher(self.factory, FakeKaggle(), worker_id="w")
        self.assertEqual(dispatcher._validate_job(job, kernel_dir), "owner/drowning-temporal")

    def test_a_check_already_queued_or_running_is_not_duplicated_by_more_clicks(self):
        service = self.service()
        first = service.submit("prepare_only", new_version=False, reason="", idempotency_key="prepare-click-1")
        clicked_again = service.submit("prepare_only", new_version=False, reason="", idempotency_key="prepare-click-2")
        self.assertEqual((clicked_again["status"], clicked_again["job"]["id"]), ("already_active", first["job"]["id"]))
        with self.factory() as session:
            self.assertEqual(len(session.scalars(select(Job)).all()), 1)
            session.get(Job, UUID_FROM(first["job"]["id"])).state = "completed"
            session.commit()
        later = service.submit("prepare_only", new_version=False, reason="", idempotency_key="prepare-click-3")
        self.assertEqual(later["status"], "queued")

    def test_blocked_submission_reports_every_reason_and_creates_nothing(self):
        with self.assertRaises(TrainBlocked) as caught:
            self.service(credentials=False).submit("prepare_only", new_version=False, reason="", idempotency_key="prepare-request-2")
        self.assertTrue(any("credentials" in reason for reason in caught.exception.reasons))
        with self.assertRaises(TrainBlocked):
            self.service().submit("remote_smoke", new_version=False, reason="", idempotency_key="smoke-request-1")
        with self.factory() as session:
            self.assertEqual(session.scalars(select(Job)).all(), [])

    def test_train_candidate_needs_explicit_new_version_smoke_gpu_windows_and_passes_the_dispatcher_gate(self):
        service = self.service()
        self.completed_prepare(windows=0)
        self.approve_everything()
        smoke_id = self.completed_smoke(gpu=False)
        with self.assertRaises(TrainBlocked) as caught:
            service.submit("train_candidate", new_version=True, reason="first candidate", idempotency_key="train-request-1",
                           options={"branch": "temporal"})
        reasons = " ".join(caught.exception.reasons)
        for text in ("did not confirm a GPU", "no temporal windows"):
            self.assertIn(text, reasons)
        with self.factory() as session:
            session.delete(session.get(Job, smoke_id))
            session.commit()
        self.completed_smoke(gpu=True)
        self.completed_prepare(windows=6)
        with self.assertRaises(TrainBlocked) as caught:
            service.submit("train_candidate", new_version=False, reason="x", idempotency_key="train-request-2")
        self.assertIn("explicit new version", " ".join(caught.exception.reasons))
        result = service.submit("train_candidate", new_version=True, reason="First temporal candidate on reviewed data",
                                idempotency_key="train-request-3")
        self.assertEqual(result["status"], "queued")
        with self.factory() as session:
            job = session.get(Job, UUID_FROM(result["job"]["id"]))
        self.assertEqual(job.payload["preflight"]["temporal_window_count"], 6)
        self.assertEqual(len(job.payload["dataset_fingerprint"]), 64)
        dispatcher = KaggleJobDispatcher(self.factory, FakeKaggle(), worker_id="w")
        self.assertEqual(dispatcher._validate_job(job, Path(job.payload["kernel_dir"])), "owner/drowning-temporal")

    def test_image_mode_is_the_default_and_needs_no_temporal_windows_but_passes_the_dispatcher_gate(self):
        service = self.service()
        self.completed_prepare(windows=0)
        self.approve_everything()
        self.completed_smoke()
        with self.assertRaises(TrainBlocked) as caught:
            service.submit("train_candidate", new_version=True, reason="temporal on image-only data", idempotency_key="train-temporal-1",
                           options={"branch": "temporal"})
        self.assertIn("no temporal windows", " ".join(caught.exception.reasons))
        result = service.submit("train_candidate", new_version=True, reason="Image mode on the labeled image data",
                                idempotency_key="train-image-1")
        self.assertEqual(result["status"], "queued")
        with self.factory() as session:
            job = session.get(Job, UUID_FROM(result["job"]["id"]))
        self.assertEqual((job.payload["options"]["branch"], job.payload["preflight"]["temporal_window_count"]), ("image_only", 0))
        self.assertEqual(job.payload["push_timeout_seconds"], 43200)
        dispatcher = KaggleJobDispatcher(self.factory, FakeKaggle(), worker_id="w")
        self.assertEqual(dispatcher._validate_job(job, Path(job.payload["kernel_dir"])), "owner/drowning-temporal")

    def test_identical_training_requests_reuse_the_existing_run_instead_of_training_again(self):
        service = self.service()
        self.completed_prepare(windows=6)
        self.approve_everything()
        self.completed_smoke()
        first = service.submit("train_candidate", new_version=True, reason="First temporal candidate on reviewed data",
                               idempotency_key="train-same-1")
        self.assertEqual(first["status"], "queued")
        with self.factory() as session:
            job = session.get(Job, UUID_FROM(first["job"]["id"]))
            identity = job.payload["training_identity"]
        self.assertEqual(len(identity), 64)
        again = service.submit("train_candidate", new_version=True, reason="Someone else asks the same thing",
                               idempotency_key="train-same-2")
        self.assertEqual((again["status"], again["job"]["id"]), ("same_training_exists", first["job"]["id"]))
        with self.factory() as session:
            self.assertEqual(len(session.scalars(select(Job).where(Job.job_type == "train", Job.payload["training_identity"].as_string() == identity)).all()), 1)
        changed = service.submit("train_candidate", new_version=True, reason="Different epochs on purpose",
                                 idempotency_key="train-same-3", options={"image_epochs": 7})
        self.assertEqual(changed["status"], "queued")
        with self.factory() as session:
            failed = session.get(Job, UUID_FROM(first["job"]["id"]))
            failed.state = "failed"
            session.commit()
        retry = service.submit("train_candidate", new_version=True, reason="Retry after the failure",
                               idempotency_key="train-same-4")
        self.assertEqual(retry["status"], "queued")

    def test_existing_approved_model_means_no_training_job_unless_new_version_is_explicit(self):
        service = self.service()
        self.completed_prepare(windows=6)
        self.approve_everything()
        self.completed_smoke()
        model_id = uuid4()
        with self.factory() as session:
            session.add(ModelVersion(id=model_id, model_family="drowning-person-temporal-v1", display_version="v1",
                                     schema_version=1, class_keys=NAMES, temporal_trained=True, completion_kind="candidate",
                                     status="approved", metrics={}))
            session.flush()
            session.add(ModelRegistry(model_family="drowning-person-temporal-v1", approved_model_id=model_id, generation=1))
            session.commit()
        result = service.submit("train_candidate", new_version=False, reason="", idempotency_key="train-request-4")
        self.assertEqual((result["status"], result["job"]), ("model_available", None))
        with self.factory() as session:
            self.assertEqual(len(session.scalars(select(Job).where(Job.idempotency_key == "train-request-4")).all()), 0)
        queued = service.submit("train_candidate", new_version=True, reason="Retrain with new reviewed videos",
                                idempotency_key="train-request-5")
        self.assertEqual(queued["status"], "queued")

    def test_continuation_pins_the_approved_checkpoint_and_selected_datasets(self):
        from tests.test_training_continuation import make_parent_bundle

        cache, model_id = self.base / "cache", uuid4()
        _, manifest, _ = make_parent_bundle(cache / str(model_id), str(model_id))
        with self.factory() as session:
            origin = Job(job_type="train", state="completed", phase="complete", idempotency_key="parent-training", payload={})
            session.add(origin)
            session.flush()
            artifact = Artifact(job_id=origin.id, role="candidate_bundle", storage_provider="google_drive",
                                remote_file_id="fixture-parent", sha256="a" * 64, size_bytes=123, status="ready")
            session.add(artifact)
            session.flush()
            session.add(ModelVersion(id=model_id, model_family=manifest["model_family"], display_version="base-v1",
                                     schema_version=1, class_keys=manifest["class_keys"], temporal_trained=False,
                                     completion_kind="candidate", status="approved", artifact_id=artifact.id,
                                     dataset_fingerprint=manifest["dataset_fingerprint"], metrics={"model_kind": "image_tracked"}))
            session.flush()
            session.add(ModelRegistry(model_family=manifest["model_family"], approved_model_id=model_id, generation=1))
            session.commit()
        self.completed_prepare()
        self.completed_smoke()
        service = self.service(env={"APPROVED_MODEL_BUNDLE_DIR": str(cache)}, mode="advisory")
        source_id = service.preflight()["sources"][0]["source_id"]
        result = service.submit("train_candidate", new_version=True, reason="Continue with the selected new data",
                                idempotency_key="continue-parent-1",
                                options={"continue_from_approved": True, "source_ids": [source_id]})
        with self.factory() as session:
            job = session.get(Job, UUID_FROM(result["job"]["id"]))
        self.assertEqual(job.payload["parent_model"]["id"], str(model_id))
        self.assertEqual(job.payload["options"]["source_ids"], [source_id])
        path = Path(job.payload["kernel_dir"])
        notebook_path = path / "Drowning_Temporal_Kaggle.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["metadata"]["drowning_project"]["parent_model"], job.payload["parent_model"])
        self.assertEqual(notebook["metadata"]["drowning_project"]["parent_dataset_ref"], job.payload["parent_dataset_ref"])
        kernel_metadata = json.loads((path / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertIn(job.payload["parent_dataset_ref"], kernel_metadata["dataset_sources"])
        dispatcher = KaggleJobDispatcher(self.factory, FakeKaggle(), worker_id="w")
        dispatcher._validate_job(job, path)
        with self.factory() as session:
            session.get(ModelVersion, model_id).status = "superseded"
            session.get(ModelRegistry, manifest["model_family"]).approved_model_id = None
            session.commit()
        replay = service.submit("train_candidate", new_version=True, reason="Continue with the selected new data",
                                idempotency_key="continue-parent-1",
                                options={"continue_from_approved": True, "source_ids": [source_id]})
        self.assertEqual(replay["job"]["id"], str(job.id))
        with self.assertRaisesRegex(TrainBlocked, "different training request"):
            service.submit("train_candidate", new_version=True, reason="Continue with the selected new data",
                           idempotency_key="continue-parent-1", options={"continue_from_approved": False})
        notebook["metadata"]["drowning_project"]["parent_model"]["id"] = str(uuid4())
        notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "parent checkpoint"):
            dispatcher._validate_job(job, path)
        notebook["metadata"]["drowning_project"]["parent_model"] = job.payload["parent_model"]
        notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
        from drowning.integrations.kaggle import KaggleError
        class FailedUpload(FakeKaggle):
            def version(self): pass
            def push_private_checkpoint_dataset(self, *args): raise KaggleError("fixture upload failed")
            def push_private_kernel(self, *args, **kwargs): raise AssertionError("kernel must not start after upload failure")
        failed = KaggleJobDispatcher(self.factory, FailedUpload(), worker_id="w").dispatch(job.id, path)
        self.assertEqual(failed.state, "failed")
        with self.factory() as session:
            self.assertIsNone(session.get(ExecutionSlot, 1).job_id)

    def test_continuation_without_an_approved_model_and_unknown_sources_are_blocked(self):
        self.completed_prepare()
        self.completed_smoke()
        service = self.service(mode="advisory")
        with self.assertRaises(TrainBlocked):
            service.submit("train_candidate", new_version=True, reason="Continue the approved model",
                           idempotency_key="no-parent-1", options={"continue_from_approved": True})
        with self.assertRaisesRegex(TrainBlocked, "unknown dataset"):
            service.submit("train_candidate", new_version=True, reason="Train new dataset selection",
                           idempotency_key="unknown-source-1", options={"source_ids": ["missing"]})

    def test_cancel_only_before_dispatch(self):
        service = self.service()
        queued = service.submit("prepare_only", new_version=False, reason="", idempotency_key="prepare-request-3")["job"]["id"]
        cancelled = service.cancel(UUID_FROM(queued))
        self.assertEqual((cancelled["state"], cancelled["remote_cancelled"]), ("cancelled", False))
        running = self.insert_job("prepare", "prepare_only", "running")
        with self.assertRaisesRegex(TrainBlocked, "no cancel"):
            service.cancel(running)
        done = self.insert_job("prepare", "prepare_only", "completed")
        self.assertEqual(service.cancel(done)["state"], "completed")
        with self.assertRaises(KeyError):
            service.cancel(uuid4())

    def test_http_preflight_and_submit_use_the_real_project(self):
        settings = Settings.from_env({**ENV, "KAGGLE_KERNEL_DIR": str(self.base / "kernels"),
                                      "INPUT_CACHE_DIR": str(self.base / "inputs")})
        with self.factory() as session:
            session.add(Job(job_type="inference", state="completed", phase="x", idempotency_key="warm", payload={}))
            session.commit()
        from sqlalchemy import text
        with self.engine.begin() as connection:
            connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
            connection.execute(text("INSERT INTO alembic_version VALUES ('20261026_01')"))
        app = create_app(settings=settings, engine=self.engine, config_dir=ROOT / "configs" / "datasets",
                         env_file=self.base / "missing.env")
        app.state.train.which = lambda name: "kaggle"
        app.state.train.environ = {"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}

        async def call(method, path, **kwargs):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
                return await client.request(method, path, **kwargs)

        view = asyncio.run(call("GET", "/api/v1/train/preflight")).json()
        self.assertTrue(view["modes"]["prepare_only"]["allowed"])
        blocked = asyncio.run(call("POST", "/api/v1/train/submit", json={"mode": "remote_smoke", "idempotency_key": "smoke-http-1"}))
        self.assertEqual(blocked.status_code, 409)
        self.assertIn("run prepare_only first", " ".join(blocked.json()["detail"]["reasons"]))
        queued = asyncio.run(call("POST", "/api/v1/train/submit", json={"mode": "prepare_only", "idempotency_key": "prepare-http-1"}))
        self.assertEqual(queued.status_code, 200, queued.text)
        job_id = queued.json()["job"]["id"]
        self.assertEqual(asyncio.run(call("POST", f"/api/v1/jobs/{job_id}/cancel")).json()["state"], "cancelled")
        self.assertEqual(asyncio.run(call("GET", f"/api/v1/jobs/{job_id}/run-report")).status_code, 404)
        self.assertEqual(asyncio.run(call("GET", "/api/v1/supervisor/status")).json()["running"], False)


def UUID_FROM(value):
    from uuid import UUID
    return UUID(value)


if __name__ == "__main__":
    unittest.main()
