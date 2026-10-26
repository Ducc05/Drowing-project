import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from drowning.db.models import Base, ExecutionSlot, Job, JobEvent
from drowning.db.repositories import create_job
from drowning.domain.jobs import JobState
from drowning.integrations.kaggle import KaggleCommandResult
from drowning.orchestration.kaggle_run import PACKAGE_NAME, package_output, run_prepare
from drowning.orchestration.supervisor import JobSupervisor
from tests.test_kaggle_run import ROOT, make_yolo_export


class ScriptedKaggle:
    def __init__(self, archive: Path):
        self.archive, self.pushes, self.statuses, self.collect_failures = archive, 0, [], 0

    @staticmethod
    def validate_kernel_ref(ref, exact_version=False):
        return ref

    def version(self):
        return KaggleCommandResult(("kaggle", "--version"), 0, "Kaggle CLI", "")

    def push_private_kernel(self, kernel_dir, timeout_seconds):
        self.pushes += 1
        return KaggleCommandResult(("kaggle", "kernels", "push"), 0, "Kernel Version: 7", "")

    def status(self, ref):
        return KaggleCommandResult(("kaggle", "kernels", "status", ref), 0, self.statuses.pop(0), "")

    def collect_exact_output(self, ref, destination):
        if self.collect_failures:
            self.collect_failures -= 1
            raise RuntimeError("network down")
        shutil.copyfile(self.archive, destination / PACKAGE_NAME)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.engine = create_engine("sqlite+pysqlite:///:memory:", poolclass=StaticPool,
                                    connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.factory() as session:
            session.add(ExecutionSlot(slot_id=1, fencing_token=0, reconciliation_status="clear"))
            session.commit()
        self.clock = Clock()
        inputs = self.base / "inputs"
        make_yolo_export(inputs / "pool-export")
        run_prepare(sources_dir=ROOT / "configs" / "datasets", reviews_dir=self.base / "reviews", input_root=inputs,
                    output_dir=self.base / "out", work_dir=self.base / "work", code_fingerprint="a" * 64)
        self.kaggle = ScriptedKaggle(package_output(self.base / "out", self.base / "run.zip"))
        self.kernel_dir = self.base / "kernel"
        self.kernel_dir.mkdir()
        (self.kernel_dir / "worker.ipynb").write_text(json.dumps(
            {"metadata": {"drowning_project": {"run_mode_default": "prepare_only"}}}), encoding="utf-8")
        (self.kernel_dir / "kernel-metadata.json").write_text(json.dumps(
            {"id": "owner/drowning-temporal", "code_file": "worker.ipynb", "is_private": True}), encoding="utf-8")

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def supervisor(self, **overrides):
        args = dict(worker_id="worker-a", installation="inst-1", kaggle_factory=lambda: self.kaggle,
                    cache_root=self.base / "cache", poll_seconds=0, clock=self.clock, current_pid=100,
                    is_alive=lambda pid: False)
        args.update(overrides)
        return JobSupervisor(self.factory, **args)

    def local_job(self, state, *, installation="inst-1", pid=4242, slot=True):
        job_id = uuid4()
        with self.factory() as session:
            session.add(Job(id=job_id, job_type="inference", state=state, phase="x", idempotency_key=f"l-{job_id}",
                            payload={"executor": "local", "installation_id": installation, "pid": pid},
                            owner_id="worker-a", execution_token=2))
            session.flush()
            if slot:
                held = session.get(ExecutionSlot, 1)
                held.job_id, held.owner_id, held.fencing_token = job_id, "worker-a", 2
            session.commit()
        return job_id

    def job(self, job_id):
        with self.factory() as session:
            return session.get(Job, job_id), session.get(ExecutionSlot, 1).job_id

    # -- local recovery --------------------------------------------------------------------
    def test_local_job_of_a_dead_process_is_failed_and_frees_the_slot(self):
        job_id = self.local_job("running")
        self.assertEqual(self.supervisor().tick()[0]["action"], "recovered_local_running_to_failed")
        job, slot_job = self.job(job_id)
        self.assertEqual((job.state, slot_job), ("failed", None))
        with self.factory() as session:
            event = session.scalars(select(JobEvent).where(JobEvent.job_id == job_id)).one()
            self.assertEqual(event.event_type, "local_process_lost_recovered")

    def test_running_syncing_and_cancelling_jobs_recover_to_the_right_state(self):
        for state, expected in (("syncing", "sync_pending"), ("cancellation_requested", "cancelled"),
                                ("dispatching", "failed")):
            job_id = self.local_job(state)
            self.supervisor().tick()
            self.assertEqual(self.job(job_id)[0].state, expected, state)

    def test_live_tracked_or_foreign_jobs_are_left_alone(self):
        alive = self.local_job("running")
        self.supervisor(is_alive=lambda pid: True).tick()
        self.assertEqual(self.job(alive)[0].state, "running")
        self.supervisor(is_tracked_locally=lambda job_id: True).tick()
        self.assertEqual(self.job(alive)[0].state, "running")
        self.supervisor(current_pid=4242).tick()
        self.assertEqual(self.job(alive)[0].state, "running")
        with self.factory() as session:
            job = session.get(Job, alive)
            job.payload = {**job.payload, "installation_id": "another-machine"}
            session.commit()
        self.supervisor().tick()
        self.assertEqual(self.job(alive)[0].state, "running")

    # -- Kaggle ----------------------------------------------------------------------------
    def queued_prepare(self):
        with self.factory() as session:
            job, _ = create_job(session, job_type="prepare", idempotency_key=f"p-{uuid4()}", payload={
                "mode": "prepare_only", "executor": "kaggle", "kernel_dir": str(self.kernel_dir)})
            return job.id

    def test_queued_job_is_dispatched_reconciled_collected_and_completed_across_ticks(self):
        job_id = self.queued_prepare()
        supervisor = self.supervisor()
        self.kaggle.statuses = ["Kernel Worker Status: Running", "Kernel Worker Status: Running",
                                "Kernel Worker Status: Complete"]
        self.assertEqual([a["action"] for a in supervisor.tick()], ["dispatched"])
        self.assertEqual(self.kaggle.pushes, 1)
        self.assertEqual(self.job(job_id)[0].state, "remote_unconfirmed")
        self.assertEqual([a["state"] for a in supervisor.tick()], ["running"])
        self.assertEqual([a["state"] for a in supervisor.tick()], ["running"])
        self.assertEqual([a["state"] for a in supervisor.tick()], ["collecting"])
        self.assertEqual([a["action"] for a in supervisor.tick()], ["collected"])
        job, slot_job = self.job(job_id)
        self.assertEqual((job.state, slot_job), ("completed", None))
        self.assertEqual(job.payload["run_summary"]["run_mode"], "prepare_only")
        self.assertEqual(self.kaggle.pushes, 1)
        self.assertEqual(supervisor.tick(), [])

    def test_forced_tick_checks_kaggle_now_and_overlapping_ticks_are_skipped(self):
        job_id = self.queued_prepare()
        supervisor = self.supervisor(poll_seconds=3600)
        self.kaggle.statuses = ["Kernel Worker Status: Running", "Kernel Worker Status: Complete"]
        supervisor.tick()
        self.assertEqual([a["state"] for a in supervisor.tick(force=True)], ["running"])
        self.assertEqual(supervisor.tick(), [])  # poll interval not elapsed, not forced
        self.assertEqual([a["state"] for a in supervisor.tick(force=True)], ["collecting"])
        supervisor._tick_lock.acquire()
        try:
            self.assertEqual(supervisor.tick(force=True), [])
        finally:
            supervisor._tick_lock.release()
        self.assertEqual(self.job(job_id)[0].state, "collecting")

    def test_queued_job_waits_while_the_compute_slot_is_busy(self):
        self.local_job("running", pid=100)
        job_id = self.queued_prepare()
        self.assertEqual(self.supervisor().tick(), [])
        self.assertEqual((self.kaggle.pushes, self.job(job_id)[0].state), (0, "queued"))

    def test_collect_failure_frees_the_slot_backs_off_and_is_retried_without_a_second_push(self):
        job_id = self.queued_prepare()
        supervisor = self.supervisor()
        self.kaggle.statuses = ["Kernel Worker Status: Complete"]
        supervisor.tick()
        supervisor.tick()
        self.assertEqual(self.job(job_id)[0].state, "collecting")
        self.kaggle.collect_failures = 1
        supervisor.tick()
        self.assertIn(str(job_id)[:8], supervisor.status()["errors"])
        self.assertEqual(self.job(job_id)[0].state, "sync_pending")
        self.assertIsNone(self.job(job_id)[1])
        self.assertEqual(supervisor.tick(), [])  # still inside the backoff window
        self.clock.now += 60
        self.assertEqual([a["action"] for a in supervisor.tick()], ["sync_retried"])
        self.assertEqual(self.job(job_id)[0].state, "completed")
        self.assertEqual(supervisor.status()["errors"], {})
        self.assertEqual(self.kaggle.pushes, 1)

    def test_jobs_owned_by_another_worker_are_not_touched(self):
        job_id = self.queued_prepare()
        supervisor = self.supervisor()
        supervisor.tick()
        with self.factory() as session:
            session.get(Job, job_id).owner_id = "someone-else"
            session.commit()
        self.kaggle.statuses = ["Kernel Worker Status: Complete"]
        self.assertEqual(supervisor.tick(), [])
        self.assertEqual(self.kaggle.statuses, ["Kernel Worker Status: Complete"])

    def test_without_a_kaggle_factory_only_local_recovery_runs(self):
        self.queued_prepare()
        self.assertEqual(self.supervisor(kaggle_factory=None).tick(), [])
        self.assertEqual(self.kaggle.pushes, 0)

    def test_background_thread_runs_ticks_and_stops_cleanly(self):
        job_id = self.local_job("running")
        supervisor = self.supervisor(interval_seconds=0.05)
        supervisor.start()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and self.job(job_id)[0].state != "failed":
            time.sleep(0.02)
        supervisor.stop()
        self.assertEqual(self.job(job_id)[0].state, "failed")
        self.assertFalse(supervisor.status()["running"])
        self.assertGreaterEqual(supervisor.status()["ticks"], 1)


if __name__ == "__main__":
    unittest.main()
