import unittest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from drowning.db.models import Artifact, Base, ExecutionSlot, Job, JobEvent, ModelRegistry, ModelVersion
from drowning.db.repositories import ExecutionSlotBusy, ModelReviewError, StaleExecutionOwner, claim_execution_slot, create_job, release_execution_slot, review_model_version, transition_job
from drowning.domain.jobs import InvalidJobTransition, can_release_execution_slot, validate_transition
from drowning.db.session import check_db


class JobPersistenceContractTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        with Session(self.engine) as session:
            session.add(ExecutionSlot(slot_id=1, fencing_token=0, reconciliation_status="clear"))
            session.commit()

    def tearDown(self):
        self.engine.dispose()

    def test_stale_worker_cannot_transition_or_append_event(self):
        with Session(self.engine) as session:
            job, _ = create_job(session, job_type="train", idempotency_key="stale-transition", payload={})
            token = claim_execution_slot(session, job.id, "worker-a")
            with self.assertRaises(StaleExecutionOwner):
                transition_job(session, job.id, "dispatching", event_type="stale-write",
                               expected_owner_id="worker-a", expected_execution_token=token + 1,
                               require_slot=True)
            session.rollback()
            self.assertEqual(session.get(Job, job.id).state, "queued")
            self.assertEqual(len(session.scalars(select(JobEvent)).all()), 1)

    def test_idempotency_key_reuses_same_job(self):
        with Session(self.engine) as session:
            first, created1 = create_job(session, job_type="prepare", idempotency_key="request-1234", payload={"mode": "prepare_only"})
            second, created2 = create_job(session, job_type="prepare", idempotency_key="request-1234", payload={"mode": "train_candidate"})
            self.assertTrue(created1)
            self.assertFalse(created2)
            self.assertEqual(first.id, second.id)
            self.assertEqual(second.payload, {"mode": "prepare_only"})
            self.assertEqual(len(session.scalars(select(Job)).all()), 1)

    def test_singleton_slot_fencing_and_confirmed_release(self):
        with Session(self.engine) as session:
            job1, _ = create_job(session, job_type="train", idempotency_key="request-one", payload={})
            job2, _ = create_job(session, job_type="evaluation", idempotency_key="request-two", payload={})
            token = claim_execution_slot(session, job1.id, "worker-a")
            self.assertEqual(token, 1)
            with self.assertRaises(ExecutionSlotBusy):
                claim_execution_slot(session, job2.id, "worker-b")
            for state in ("dispatching", "running", "collecting", "validating", "syncing", "completed"):
                transition_job(session, job1.id, state, event_type="test_transition")
            with self.assertRaisesRegex(ValueError, "confirmed compute stop"):
                release_execution_slot(session, job1.id, "worker-a", token, compute_stopped_confirmed=False)
            release_execution_slot(session, job1.id, "worker-a", token, compute_stopped_confirmed=True)
            slot = session.get(ExecutionSlot, 1)
            self.assertIsNone(slot.job_id)
            self.assertEqual(slot.fencing_token, 1)

    def test_stale_owner_cannot_release_reclaimed_slot(self):
        with Session(self.engine) as session:
            job, _ = create_job(session, job_type="train", idempotency_key="request-stale", payload={})
            token = claim_execution_slot(session, job.id, "worker-a")
            with self.assertRaises(StaleExecutionOwner):
                release_execution_slot(session, job.id, "worker-b", token + 1, compute_stopped_confirmed=True)
            with self.assertRaises(InvalidJobTransition):
                transition_job(session, job.id, "completed", event_type="skip_remote")
            self.assertFalse(can_release_execution_slot("remote_unconfirmed", compute_stopped_confirmed=True))
            self.assertEqual(token, 1)

    def test_job_transition_order_is_enforced(self):
        with self.assertRaises(InvalidJobTransition):
            validate_transition("queued", "completed")
        self.assertEqual(validate_transition("queued", "dispatching")[1].value, "dispatching")
        self.assertTrue(can_release_execution_slot("sync_pending", compute_stopped_confirmed=True))
        self.assertFalse(can_release_execution_slot("sync_pending", compute_stopped_confirmed=False))

    def test_database_readiness_requires_the_expected_migration_revision(self):
        self.assertFalse(check_db(self.engine))
        with self.engine.begin() as connection:
            connection.exec_driver_sql("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
            connection.exec_driver_sql("INSERT INTO alembic_version (version_num) VALUES ('old-revision')")
        self.assertFalse(check_db(self.engine))
        with self.engine.begin() as connection:
            connection.exec_driver_sql("DELETE FROM alembic_version")
            connection.exec_driver_sql("INSERT INTO alembic_version (version_num) VALUES ('20261026_01')")
        self.assertTrue(check_db(self.engine))

    def test_approval_requires_verified_drive_artifact_then_atomically_updates_pointer(self):
        with Session(self.engine) as session:
            job, _ = create_job(session, job_type="train", idempotency_key="model-review-job", payload={})
            version = ModelVersion(model_family="family", display_version="v1", schema_version=1,
                                   class_keys=["drowning", "swimming", "out_of_water"], temporal_trained=True,
                                   completion_kind="candidate", status="candidate", training_job_id=job.id,
                                   metrics={})
            session.add(version)
            session.commit()
            with self.assertRaisesRegex(ModelReviewError, "verified, ready Drive artifact"):
                review_model_version(session, version.id, decision="approve", reviewer_id="reviewer")
            artifact = Artifact(job_id=job.id, role="model_bundle", storage_provider="google_drive",
                                remote_file_id="drive-file-id", sha256="a" * 64, size_bytes=100, status="ready")
            session.add(artifact)
            session.commit()
            version.artifact_id = artifact.id
            session.commit()
            reviewed = review_model_version(session, version.id, decision="approve", reviewer_id="reviewer", comment="tested")
            registry = session.get(ModelRegistry, "family")
            self.assertEqual(reviewed.status, "approved")
            self.assertEqual(registry.approved_model_id, version.id)
            self.assertEqual(registry.generation, 1)

    def test_reject_keeps_candidate_record(self):
        with Session(self.engine) as session:
            job, _ = create_job(session, job_type="train", idempotency_key="model-reject-job", payload={})
            version = ModelVersion(model_family="family", display_version="v2", schema_version=1,
                                   class_keys=["drowning", "swimming", "out_of_water"], temporal_trained=True,
                                   completion_kind="candidate", status="candidate", training_job_id=job.id,
                                   metrics={})
            session.add(version)
            session.commit()
            reviewed = review_model_version(session, version.id, decision="reject", reviewer_id="reviewer")
            self.assertEqual(reviewed.status, "rejected")
            self.assertIsNotNone(session.get(ModelVersion, version.id))


if __name__ == "__main__":
    unittest.main()
