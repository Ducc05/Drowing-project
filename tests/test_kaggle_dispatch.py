import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from uuid import UUID

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from drowning.db.models import Base, ExecutionSlot, Job
from drowning.db.repositories import StaleExecutionOwner, create_job
from drowning.integrations.kaggle import KaggleCommandResult, RemoteStateUnconfirmed
from drowning.orchestration.kaggle_dispatch import KaggleJobDispatcher, parse_pushed_version, parse_remote_status


class FakeKaggleClient:
    def __init__(self):
        self.status_outputs = []
        self.push_count = 0
        self.timeout = False

    @staticmethod
    def validate_kernel_ref(kernel_ref):
        return kernel_ref

    def version(self):
        return KaggleCommandResult(("kaggle", "--version"), 0, "Kaggle CLI", "")

    def push_private_kernel(self, kernel_dir, timeout_seconds):
        self.push_count += 1
        if self.timeout:
            raise RemoteStateUnconfirmed("timeout")
        return KaggleCommandResult(("kaggle", "kernels", "push"), 0, "Kernel Version: 12", "")

    def status(self, kernel_ref):
        output = self.status_outputs.pop(0)
        return KaggleCommandResult(("kaggle", "kernels", "status", kernel_ref), 0, output, "")


class KaggleOutputParsingTests(unittest.TestCase):
    def test_status_lines_in_every_known_shape_are_recognized_and_unknown_ones_are_not(self):
        cases = {
            "Kernel Worker Status: Running": "running", "owner/slug has status \"KernelWorkerStatus.RUNNING\"": "running",
            "owner/slug has status \"KernelWorkerStatus.QUEUED\"": "running", "owner/slug has status \"complete\"": "completed",
            "owner/slug has status \"KernelWorkerStatus.COMPLETE\"": "completed", "owner/slug has status \"KernelWorkerStatus.ERROR\"": "failed",
            "owner/slug has status \"KernelWorkerStatus.CANCEL_REQUESTED\"": "running",
            "owner/slug has status \"KernelWorkerStatus.CANCEL_ACKNOWLEDGED\"": "cancelled",
            "owner/slug has status \"KernelWorkerStatus.SOMETHING_NEW\"": None, "no status here": None, "": None}
        for text, expected in cases.items():
            self.assertEqual(parse_remote_status(text), expected, text)

    def test_push_receipt_version_is_read_from_the_usual_sentence(self):
        self.assertEqual(parse_pushed_version("Kernel version 3 successfully pushed. Please check progress at https://x"), "3")
        self.assertEqual(parse_pushed_version("Kernel Version: 12"), "12")
        self.assertIsNone(parse_pushed_version("pushed"))


class KaggleDispatchTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.session_factory() as session:
            session.add(ExecutionSlot(slot_id=1, fencing_token=0, reconciliation_status="clear"))
            session.commit()
        self.temp = tempfile.TemporaryDirectory()
        self.kernel_dir = Path(self.temp.name)
        notebook = {"metadata": {"drowning_project": {"run_mode_default": "prepare_only"}}}
        (self.kernel_dir / "worker.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
        (self.kernel_dir / "kernel-metadata.json").write_text(json.dumps({
            "id": "owner/drowning-temporal", "code_file": "worker.ipynb", "is_private": True,
        }), encoding="utf-8")
        self.client = FakeKaggleClient()
        self.dispatcher = KaggleJobDispatcher(self.session_factory, self.client, worker_id="worker-a")

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def create_prepare_job(self):
        with self.session_factory() as session:
            job, _ = create_job(session, job_type="prepare", idempotency_key="dispatch-prepare-1",
                                payload={"mode": "prepare_only", "executor": "kaggle"})
            return job.id

    def test_dispatch_keeps_slot_until_exact_remote_status_is_known(self):
        job_id = self.create_prepare_job()
        receipt = self.dispatcher.dispatch(job_id, self.kernel_dir)
        self.assertEqual(receipt.state, "remote_unconfirmed")
        self.assertEqual(receipt.kernel_version, "12")
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            self.assertEqual(job.state, "remote_unconfirmed")
            self.assertEqual(session.get(ExecutionSlot, 1).job_id, job_id)
        self.client.status_outputs = ["Kernel Worker Status: Running", "Kernel Worker Status: Complete"]
        running = self.dispatcher.reconcile(job_id)
        self.assertEqual(running.state, "running")
        completed = self.dispatcher.reconcile(job_id)
        self.assertEqual(completed.state, "collecting")
        self.assertEqual(self.client.push_count, 1)
        with self.session_factory() as session:
            self.assertEqual(session.get(ExecutionSlot, 1).job_id, job_id)

    def test_timed_out_push_is_not_retried_or_reconciled_to_latest(self):
        job_id = self.create_prepare_job()
        self.client.timeout = True
        receipt = self.dispatcher.dispatch(job_id, self.kernel_dir)
        self.assertIsNone(receipt.kernel_version)
        reconciliation = self.dispatcher.reconcile(job_id)
        self.assertEqual(reconciliation.state, "remote_unconfirmed")
        self.assertEqual(self.client.push_count, 1)
        self.assertEqual(self.client.status_outputs, [])
        with self.session_factory() as session:
            self.assertEqual(session.get(ExecutionSlot, 1).job_id, job_id)

    def test_stale_push_receipt_cannot_overwrite_new_execution_owner(self):
        job_id = self.create_prepare_job()
        def stale_push(*args, **kwargs):
            with self.session_factory() as session:
                job = session.get(Job, job_id)
                job.owner_id, job.execution_token = "worker-new", 2
                slot = session.get(ExecutionSlot, 1)
                slot.owner_id, slot.fencing_token = "worker-new", 2
                session.commit()
            return KaggleCommandResult(("kaggle",), 0, "Kernel Version: 999", "")
        self.client.push_private_kernel = stale_push
        with self.assertRaises(StaleExecutionOwner):
            self.dispatcher.dispatch(job_id, self.kernel_dir)
        with self.session_factory() as session:
            job = session.get(Job, job_id)
            self.assertIsNone(job.remote_version)
            self.assertEqual(job.owner_id, "worker-new")
            self.assertEqual(session.get(ExecutionSlot, 1).owner_id, "worker-new")

    def test_candidate_dispatch_requires_reviewed_preflight_before_claim(self):
        with self.session_factory() as session:
            job, _ = create_job(session, job_type="train", idempotency_key="dispatch-candidate-1",
                                payload={"mode": "train_candidate", "executor": "kaggle"})
            job_id = job.id
        with self.assertRaisesRegex(ValueError, "explicit new version"):
            self.dispatcher.dispatch(job_id, self.kernel_dir)
        self.assertEqual(self.client.push_count, 0)
        with self.session_factory() as session:
            self.assertIsNone(session.get(ExecutionSlot, 1).job_id)

    def test_kernel_directory_must_be_private_and_mode_matched(self):
        job_id = self.create_prepare_job()
        metadata = json.loads((self.kernel_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
        metadata["is_private"] = False
        (self.kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "is_private"):
            self.dispatcher.dispatch(job_id, self.kernel_dir)
        self.assertEqual(self.client.push_count, 0)
        metadata["is_private"] = True
        (self.kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (self.kernel_dir / ".env").write_text("DATABASE_URL=must-not-upload", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "only its notebook"):
            self.dispatcher.dispatch(job_id, self.kernel_dir)
        with self.session_factory() as session:
            self.assertIsNone(session.get(ExecutionSlot, 1).job_id)
        self.assertEqual(self.client.push_count, 0)


if __name__ == "__main__":
    unittest.main()
