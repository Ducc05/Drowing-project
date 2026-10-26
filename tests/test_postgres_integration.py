"""Real PostgreSQL checks in an isolated, disposable schema of TEST_DATABASE_URL."""
import multiprocessing
import re
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from drowning.config import load_env_file
from drowning.db.models import Artifact, ExecutionSlot, Job, ModelRegistry, ModelVersion
from drowning.db.repositories import (ExecutionSlotBusy, StaleExecutionOwner, claim_execution_slot,
                                      create_job, review_model_version, transition_job)
from drowning.db.session import check_db

ROOT = Path(__file__).resolve().parents[1]


def concurrent_worker(url, schema, operation, value, barrier, queue):
    engine = create_engine(url, hide_parameters=True, connect_args={"options": f"-csearch_path={schema}"})
    try:
        with Session(engine) as session:
            barrier.wait(timeout=15)
            if operation == "claim":
                token = claim_execution_slot(session, UUID(value), f"worker-{multiprocessing.current_process().pid}")
                queue.put({"outcome": "claimed", "token": token, "job_id": value})
            elif operation == "idempotency":
                job, created = create_job(session, job_type="prepare", idempotency_key=value, payload={})
                queue.put({"outcome": "created" if created else "existing", "job_id": str(job.id)})
            else:
                model = review_model_version(session, UUID(value), decision="approve",
                                             reviewer_id=f"reviewer-{multiprocessing.current_process().pid}")
                queue.put({"outcome": "approved", "model_id": str(model.id)})
    except ExecutionSlotBusy:
        queue.put({"outcome": "busy"})
    except Exception as exc:
        queue.put({"outcome": "error", "error_type": type(exc).__name__})
    finally:
        engine.dispose()


class PostgreSQLIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = {}
        load_env_file(ROOT / ".env", env)
        cls.url = env.get("TEST_DATABASE_URL", "")
        if not cls.url:
            raise unittest.SkipTest("TEST_DATABASE_URL is not configured; real PostgreSQL is unverified")
        parsed = make_url(cls.url)
        if parsed.get_backend_name() != "postgresql" or not (parsed.database or "").endswith("_test"):
            raise unittest.SkipTest("TEST_DATABASE_URL must target a dedicated PostgreSQL *_test database")
        cls.admin = create_engine(cls.url, hide_parameters=True)
        with cls.admin.connect() as connection:
            cls.server_version = connection.execute(text("SHOW server_version")).scalar_one()

    @classmethod
    def tearDownClass(cls):
        cls.admin.dispose()

    def setUp(self):
        self.schema = f"drowning_review_{uuid4().hex}"
        with self.admin.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{self.schema}"')
        self.engine = create_engine(self.url, hide_parameters=True,
                                    connect_args={"options": f"-csearch_path={self.schema}"})
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "migrations"))
        with self.engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")

    def tearDown(self):
        self.engine.dispose()
        if not re.fullmatch(r"drowning_review_[0-9a-f]{32}", self.schema):
            raise RuntimeError("Refusing to remove an unexpected test schema")
        with self.admin.begin() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA "{self.schema}" CASCADE')

    def run_workers(self, operation, values):
        context = multiprocessing.get_context("spawn")
        barrier, queue = context.Barrier(2), context.Queue()
        workers = [context.Process(target=concurrent_worker,
                                   args=(self.url, self.schema, operation, value, barrier, queue))
                   for value in values]
        try:
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=30)
                self.assertFalse(worker.is_alive(), "isolated PostgreSQL worker did not finish")
                self.assertEqual(worker.exitcode, 0)
            return [queue.get(timeout=5) for _ in workers]
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=5)
            queue.close()

    def test_real_migration_head_and_singleton_seed(self):
        self.assertTrue(check_db(self.engine))
        with Session(self.engine) as session:
            slot = session.get(ExecutionSlot, 1)
            self.assertIsNone(slot.job_id)
            self.assertEqual(slot.fencing_token, 0)

    def test_two_processes_compete_for_one_global_compute_slot(self):
        with Session(self.engine) as session:
            first, _ = create_job(session, job_type="train", idempotency_key="concurrent-train", payload={})
            second, _ = create_job(session, job_type="inference", idempotency_key="concurrent-test", payload={})
            ids = [str(first.id), str(second.id)]
        results = self.run_workers("claim", ids)
        self.assertEqual(sorted(item["outcome"] for item in results), ["busy", "claimed"], results)
        with Session(self.engine) as session:
            self.assertEqual(session.get(ExecutionSlot, 1).fencing_token, 1)

    def test_two_processes_create_one_idempotent_job(self):
        results = self.run_workers("idempotency", ["same-request-1234"] * 2)
        self.assertEqual(sorted(item["outcome"] for item in results), ["created", "existing"], results)
        self.assertEqual(results[0]["job_id"], results[1]["job_id"])

    def test_stale_token_cannot_publish_state_on_real_postgresql(self):
        with Session(self.engine) as session:
            job, _ = create_job(session, job_type="inference", idempotency_key="stale-token", payload={})
            job_id = job.id
            token = claim_execution_slot(session, job_id, "worker-old")
            job.execution_token = token + 1
            job.owner_id = "worker-new"
            slot = session.get(ExecutionSlot, 1)
            slot.fencing_token, slot.owner_id = token + 1, "worker-new"
            session.commit()
        with Session(self.engine) as session:
            with self.assertRaises(StaleExecutionOwner):
                transition_job(session, job_id, "dispatching", event_type="stale-output",
                               expected_owner_id="worker-old", expected_execution_token=token, require_slot=True)
            session.rollback()
            self.assertEqual(session.get(Job, job_id).state, "queued")

    def test_expired_lease_does_not_release_unconfirmed_compute(self):
        with Session(self.engine) as session:
            first, _ = create_job(session, job_type="train", idempotency_key="lease-first", payload={})
            second, _ = create_job(session, job_type="inference", idempotency_key="lease-second", payload={})
            claim_execution_slot(session, first.id, "remote-worker")
            slot = session.get(ExecutionSlot, 1)
            slot.lease_until = datetime.now(timezone.utc) - timedelta(hours=1)
            session.commit()
            with self.assertRaises(ExecutionSlotBusy):
                claim_execution_slot(session, second.id, "new-worker")

    def test_simultaneous_first_model_approvals_serialize_registry_pointer(self):
        ids = []
        with Session(self.engine) as session:
            for index in range(2):
                job, _ = create_job(session, job_type="train", idempotency_key=f"approve-{index}", payload={})
                artifact = Artifact(job_id=job.id, role="candidate_bundle", storage_provider="google_drive",
                                    remote_file_id=f"fixture-drive-{index}", sha256=str(index) * 64,
                                    size_bytes=100, status="ready")
                session.add(artifact)
                session.flush()
                model = ModelVersion(model_family="review-family", display_version=f"v{index}",
                                     schema_version=1, class_keys=["drowning", "swimming", "out_of_water"],
                                     temporal_trained=True, completion_kind="candidate", status="candidate",
                                     artifact_id=artifact.id, metrics={})
                session.add(model)
                session.commit()
                ids.append(str(model.id))
        results = self.run_workers("approve", ids)
        self.assertEqual([item["outcome"] for item in results].count("approved"), 2, results)
        with Session(self.engine) as session:
            registry = session.get(ModelRegistry, "review-family")
            self.assertEqual(registry.generation, 2)
            versions = session.scalars(select(ModelVersion)).all()
            self.assertEqual(sorted(item.status for item in versions), ["approved", "superseded"])


if __name__ == "__main__":
    unittest.main()
