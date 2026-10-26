"""Background supervisor: crash recovery for local jobs and progress for this worker's Kaggle jobs.

Every action goes through the same fenced repository calls as the manual CLI, so the supervisor
cannot publish stale outputs or free a compute slot it does not own. It only advances jobs owned by
its own worker ID: another machine's job is shown but never touched, and a remote that may still be
running is never treated as stopped.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from drowning.db.models import ExecutionSlot, Job
from drowning.db.repositories import release_execution_slot, transition_job
from drowning.domain.jobs import JobState
from drowning.installation import pid_alive

_LOCAL_RECOVERABLE = {JobState.DISPATCHING.value, JobState.RUNNING.value, JobState.COLLECTING.value,
                      JobState.VALIDATING.value, JobState.SYNCING.value, JobState.CANCELLATION_REQUESTED.value}
_KAGGLE_STATES = {JobState.QUEUED.value, JobState.REMOTE_UNCONFIRMED.value, JobState.RUNNING.value,
                  JobState.COLLECTING.value, JobState.SYNC_PENDING.value}


class JobSupervisor:
    def __init__(self, session_factory: sessionmaker, *, worker_id: str, installation: str,
                 kaggle_factory: Callable[[], Any] | None = None, drive_factory: Callable[[], Any] | None = None,
                 drive_folder_id: str = "", cache_root: Path | None = None, interval_seconds: float = 30.0,
                 poll_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic,
                 is_alive: Callable[[int], bool] = pid_alive, current_pid: int | None = None,
                 is_tracked_locally: Callable[[UUID], bool] = lambda _job_id: False):
        self.factory, self.worker_id, self.installation = session_factory, worker_id, installation
        self.kaggle_factory, self.drive_factory = kaggle_factory, drive_factory
        self.drive_folder_id, self.cache_root = drive_folder_id, cache_root
        self.interval, self.poll_seconds, self.clock = interval_seconds, poll_seconds, clock
        self.is_alive, self.current_pid = is_alive, current_pid if current_pid is not None else os.getpid()
        self.is_tracked_locally = is_tracked_locally
        self._failures: dict[UUID, tuple[int, float]] = {}
        self._last_poll: dict[UUID, float] = {}
        self._stop = threading.Event()
        self._tick_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.last_tick_wall: float | None = None
        self.last_actions: list[dict[str, Any]] = []
        self.last_errors: dict[str, str] = {}
        self.ticks = 0

    # -- lifecycle -------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="job-supervisor", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # the loop must survive a bad tick
                self.last_errors["tick"] = type(exc).__name__
            self._stop.wait(self.interval)

    def status(self) -> dict[str, Any]:
        return {"running": self._thread is not None and self._thread.is_alive(), "worker_id": self.worker_id,
                "ticks": self.ticks, "interval_seconds": self.interval, "last_tick_age_seconds":
                    None if self.last_tick_wall is None else round(time.time() - self.last_tick_wall, 1),
                "last_actions": self.last_actions[-20:], "errors": dict(self.last_errors),
                "scope": "recovers this installation's local jobs and advances Kaggle jobs owned by this worker ID"}

    # -- tick ------------------------------------------------------------------------------
    def tick(self, force: bool = False) -> list[dict[str, Any]]:
        """One pass over this worker's jobs. ``force`` re-checks Kaggle now instead of waiting out the poll interval.
        A pass already in progress (background loop or another caller) is not repeated."""
        if not self._tick_lock.acquire(blocking=False):
            return []
        try:
            actions = self._recover_local_jobs() + self._advance_kaggle_jobs(force)
            self.ticks += 1
            self.last_tick_wall = time.time()
            self.last_actions = (self.last_actions + actions)[-100:]
            return actions
        finally:
            self._tick_lock.release()

    def _backoff_ok(self, job_id: UUID) -> bool:
        failures, retry_at = self._failures.get(job_id, (0, 0.0))
        return failures == 0 or self.clock() >= retry_at

    def _record(self, job_id: UUID, exc: BaseException | None) -> None:
        key = str(job_id)[:8]
        if exc is None:
            self._failures.pop(job_id, None)
            self.last_errors.pop(key, None)
            return
        failures = self._failures.get(job_id, (0, 0.0))[0] + 1
        self._failures[job_id] = (failures, self.clock() + min(1800.0, 30.0 * 2 ** (failures - 1)))
        self.last_errors[key] = f"{type(exc).__name__}: {exc}"[:300]

    # -- local crash recovery --------------------------------------------------------------
    def _recover_local_jobs(self) -> list[dict[str, Any]]:
        actions = []
        with self.factory() as session:
            candidates = [job for job in session.scalars(select(Job).where(Job.state.in_(_LOCAL_RECOVERABLE)))
                          if (job.payload or {}).get("executor") == "local"
                          and (job.payload or {}).get("installation_id") == self.installation]
        for job in candidates:
            pid = int((job.payload or {}).get("pid") or 0)
            if self.is_tracked_locally(job.id) or pid == self.current_pid or (pid and self.is_alive(pid)):
                continue
            target = (JobState.SYNC_PENDING if job.state == JobState.SYNCING.value else
                      JobState.CANCELLED if job.state == JobState.CANCELLATION_REQUESTED.value else JobState.FAILED)
            try:
                with self.factory() as session:
                    transition_job(session, job.id, target, event_type="local_process_lost_recovered",
                                   phase="recovered_after_restart",
                                   details={"previous_state": job.state, "owner_process_gone": True},
                                   expected_owner_id=job.owner_id, expected_execution_token=job.execution_token)
                    slot = session.get(ExecutionSlot, 1)
                    if slot is not None and slot.job_id == job.id:
                        release_execution_slot(session, job.id, job.owner_id, job.execution_token,
                                               compute_stopped_confirmed=True)
                actions.append({"job_id": str(job.id), "action": f"recovered_local_{job.state}_to_{target.value}"})
            except Exception as exc:
                self._record(job.id, exc)
        return actions

    # -- Kaggle jobs -----------------------------------------------------------------------
    def _advance_kaggle_jobs(self, force: bool = False) -> list[dict[str, Any]]:
        if self.kaggle_factory is None:
            return []
        with self.factory() as session:
            jobs = [job for job in session.scalars(select(Job).where(Job.state.in_(_KAGGLE_STATES)).order_by(Job.created_at))
                    if job.job_type in {"prepare", "train"} and (job.payload or {}).get("executor") == "kaggle"]
            slot_busy = (session.get(ExecutionSlot, 1) or ExecutionSlot(slot_id=1)).job_id is not None
        actions: list[dict[str, Any]] = []
        dispatched = False
        for job in jobs:
            if not self._backoff_ok(job.id):
                continue
            try:
                if job.state == JobState.QUEUED.value:
                    if slot_busy or dispatched or not (job.payload or {}).get("kernel_dir"):
                        continue
                    dispatched = True
                    actions.append(self._dispatch(job))
                elif job.owner_id != self.worker_id:
                    continue
                elif job.state in {JobState.REMOTE_UNCONFIRMED.value, JobState.RUNNING.value}:
                    if not force and self.clock() - self._last_poll.get(job.id, -1e9) < self.poll_seconds:
                        continue
                    self._last_poll[job.id] = self.clock()
                    actions.append(self._reconcile(job))
                elif job.state == JobState.COLLECTING.value:
                    actions.append(self._collect(job))
                elif job.state == JobState.SYNC_PENDING.value:
                    actions.append(self._retry_sync(job))
                self._record(job.id, None)
            except Exception as exc:
                self._record(job.id, exc)
        return [action for action in actions if action]

    def _dispatch(self, job: Job) -> dict[str, Any]:
        from drowning.orchestration.kaggle_dispatch import KaggleJobDispatcher
        receipt = KaggleJobDispatcher(self.factory, self.kaggle_factory(), worker_id=self.worker_id).dispatch(
            job.id, Path(job.payload["kernel_dir"]), timeout_seconds=int(job.payload.get("push_timeout_seconds", 3600)))
        return {"job_id": str(job.id), "action": "dispatched", "state": receipt.state, "kernel_version": receipt.kernel_version}

    def _reconcile(self, job: Job) -> dict[str, Any]:
        from drowning.orchestration.kaggle_dispatch import KaggleJobDispatcher
        receipt = KaggleJobDispatcher(self.factory, self.kaggle_factory(), worker_id=self.worker_id).reconcile(job.id)
        return {"job_id": str(job.id), "action": "reconciled", "state": receipt.state, "detail": receipt.detail}

    def _collector(self, job: Job):
        mode = (job.payload or {}).get("mode")
        drive = None
        if self.drive_folder_id and self.drive_factory is not None:
            try:
                drive = self.drive_factory()
            except Exception as exc:
                if mode == "train_candidate":
                    raise
                self.last_errors["drive"] = f"{type(exc).__name__}: {exc}"[:300]
        cache = self.cache_root or Path("./local-cache/results")
        if mode == "train_candidate":
            if drive is None:
                raise RuntimeError("Google Drive is required to publish a model candidate")
            from drowning.orchestration.candidate_sync import CandidateArtifactSync
            return CandidateArtifactSync(self.factory, self.kaggle_factory(), drive, drive_folder_id=self.drive_folder_id,
                                         cache_root=cache, worker_id=self.worker_id)
        from drowning.orchestration.run_collector import RunOutputCollector
        return RunOutputCollector(self.factory, self.kaggle_factory(), cache_root=cache, worker_id=self.worker_id,
                                  drive=drive, drive_folder_id=self.drive_folder_id)

    def _collect(self, job: Job) -> dict[str, Any]:
        result = self._collector(job).collect(job.id)
        return {"job_id": str(job.id), "action": "collected", "state": result["state"]}

    def _retry_sync(self, job: Job) -> dict[str, Any]:
        if (job.payload or {}).get("mode") == "train_candidate" and not self.drive_folder_id:
            return {}
        result = self._collector(job).retry_sync(job.id)
        return {"job_id": str(job.id), "action": "sync_retried", "state": result["state"]}
