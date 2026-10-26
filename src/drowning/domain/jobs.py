"""Explicit compute job state machine and safe execution-slot release rules."""

from __future__ import annotations

from enum import StrEnum


class JobState(StrEnum):
    CREATED = "created"
    QUEUED = "queued"
    DISPATCHING = "dispatching"
    RUNNING = "running"
    COLLECTING = "collecting"
    VALIDATING = "validating"
    SYNCING = "syncing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLATION_REQUESTED = "cancellation_requested"
    REMOTE_UNCONFIRMED = "remote_unconfirmed"
    SYNC_PENDING = "sync_pending"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


_TRANSITIONS = {
    JobState.CREATED: {JobState.QUEUED, JobState.FAILED, JobState.CANCELLED},
    JobState.QUEUED: {JobState.DISPATCHING, JobState.CANCELLED, JobState.FAILED},
    JobState.DISPATCHING: {JobState.RUNNING, JobState.REMOTE_UNCONFIRMED, JobState.FAILED, JobState.CANCELLATION_REQUESTED},
    JobState.RUNNING: {JobState.COLLECTING, JobState.FAILED, JobState.CANCELLATION_REQUESTED, JobState.INTERRUPTED, JobState.REMOTE_UNCONFIRMED},
    JobState.COLLECTING: {JobState.VALIDATING, JobState.SYNC_PENDING, JobState.FAILED, JobState.REMOTE_UNCONFIRMED},
    JobState.VALIDATING: {JobState.SYNCING, JobState.SYNC_PENDING, JobState.COMPLETED, JobState.FAILED},
    JobState.SYNCING: {JobState.COMPLETED, JobState.SYNC_PENDING, JobState.FAILED},
    JobState.SYNC_PENDING: {JobState.SYNCING, JobState.FAILED},
    JobState.CANCELLATION_REQUESTED: {JobState.CANCELLED, JobState.REMOTE_UNCONFIRMED, JobState.FAILED},
    JobState.INTERRUPTED: {JobState.QUEUED, JobState.FAILED, JobState.REMOTE_UNCONFIRMED},
    JobState.REMOTE_UNCONFIRMED: {JobState.RUNNING, JobState.COLLECTING, JobState.CANCELLED, JobState.FAILED, JobState.INTERRUPTED},
    JobState.COMPLETED: set(),
    JobState.FAILED: set(),
    JobState.CANCELLED: set(),
}


class InvalidJobTransition(ValueError):
    pass


def validate_transition(current: str | JobState, target: str | JobState) -> tuple[JobState, JobState]:
    old, new = JobState(current), JobState(target)
    if new not in _TRANSITIONS[old]:
        raise InvalidJobTransition(f"job cannot transition from {old.value} to {new.value}")
    return old, new


def can_release_execution_slot(state: str | JobState, *, compute_stopped_confirmed: bool) -> bool:
    current = JobState(state)
    # Sync can retry after compute has been independently confirmed stopped.
    return current in {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED, JobState.SYNC_PENDING} and compute_stopped_confirmed
