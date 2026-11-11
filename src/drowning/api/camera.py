"""Camera Start/Stop service: one live session per installation, owning the shared compute slot."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable
from uuid import UUID

from sqlalchemy.orm import sessionmaker

from drowning.api.inference import (InferenceUnavailable, apply_confidence_threshold, get_approved_model,
                                    load_approved_pipeline, resolve_cell_grid, resolve_confidence,
                                    runtime_weights_missing)
from drowning.inference.cells import format_cell_grid
from drowning.config import Settings
from drowning.db.models import Job
from drowning.db.repositories import (ExecutionSlotBusy, StaleExecutionOwner, claim_execution_slot, create_job,
                                      release_execution_slot, transition_job)
from drowning.domain.jobs import JobState
from drowning.inference.camera import (BROWSER_TAB_SOURCE, CameraConnectError, CameraSession, CameraSourceError,
                                        CameraStopUnconfirmed, PushCapture, decode_frame, parse_camera_source,
                                        probe_camera)
from drowning.installation import installation_id


class CameraBusy(RuntimeError):
    def __init__(self, job_id: UUID | None, reason: str):
        self.job_id = job_id
        super().__init__(reason)


class CameraNotRunning(RuntimeError):
    pass


class CameraService:
    def __init__(self, session_factory: sessionmaker, settings: Settings, *,
                 inference_factory: Callable[..., Any] | None = None,
                 capture_factory: Callable[[Any], Any] | None = None, stop_timeout: float = 10.0,
                 push_read_timeout: float = 1.0, push_first_frame_timeout: float = 20.0):
        self.factory = session_factory
        self.settings = settings
        self.inference_factory = inference_factory
        self.capture_factory = capture_factory
        self.stop_timeout = stop_timeout
        self.push_read_timeout = push_read_timeout
        self.push_first_frame_timeout = push_first_frame_timeout
        self._push: PushCapture | None = None
        self.installation = installation_id(Path(settings.input_cache_dir).expanduser().resolve().parent)
        self._lock = threading.RLock()
        self._session: CameraSession | None = None
        self._job_id: UUID | None = None
        self._fence: tuple[str, int] | None = None
        self._last_summary: dict[str, Any] | None = None

    # -- helpers ---------------------------------------------------------------------------
    def _transition(self, state: JobState, phase: str, event: str, details: dict[str, Any] | None = None) -> Job:
        assert self._job_id is not None and self._fence is not None
        with self.factory() as session:
            return transition_job(session, self._job_id, state, phase=phase, event_type=event,
                                  details=details or {}, expected_owner_id=self._fence[0],
                                  expected_execution_token=self._fence[1])

    def _payload(self, **values: Any) -> None:
        assert self._job_id is not None
        with self.factory() as session:
            job = session.get(Job, self._job_id)
            job.payload = {**job.payload, **values}
            session.commit()

    def _release(self) -> None:
        assert self._job_id is not None and self._fence is not None
        with self.factory() as session:
            release_execution_slot(session, self._job_id, self._fence[0], self._fence[1],
                                   compute_stopped_confirmed=True)

    def _forget(self) -> None:
        self._session, self._job_id, self._fence, self._push = None, None, None, None

    # -- operations ------------------------------------------------------------------------
    def probe(self, raw_source: str) -> dict[str, Any]:
        source, display = parse_camera_source(raw_source)
        if source == BROWSER_TAB_SOURCE:
            raise CameraSourceError("nguồn browser-tab không cần kiểm tra kết nối")
        with self._lock:
            if self._session is not None and self._session.is_alive():
                raise CameraBusy(self._job_id, "A camera session is already running; stop it before probing")
        return {**probe_camera(source, self.capture_factory), "source": display}

    def start(self, raw_source: str, idempotency_key: str, *, confidence: float | None = None,
              cell_grid: str | None = None, fps: int | None = None) -> dict[str, Any]:
        source, display = parse_camera_source(raw_source)
        is_tab = source == BROWSER_TAB_SOURCE
        if len(idempotency_key.strip()) < 8:
            raise CameraSourceError("idempotency_key must contain at least eight characters")
        try:
            threshold = resolve_confidence(self.settings, confidence)
        except ValueError as exc:
            raise CameraSourceError(str(exc)) from exc
        try:
            grid = resolve_cell_grid(self.settings, cell_grid)
        except ValueError as exc:
            raise CameraSourceError(f"cell_grid: {exc}") from exc
        with self._lock:
            if self._session is not None:
                self._finalize_if_dead()
            if self._session is not None:
                raise CameraBusy(self._job_id, "A camera session is already active on this installation")
            with self.factory() as session:
                model_record, _, _ = get_approved_model(session, self.settings)
            missing = None if self.inference_factory is not None else runtime_weights_missing(self.settings, model_record)
            if missing:
                raise InferenceUnavailable(f"{missing}; camera test cannot start")
            with self.factory() as session:
                job, created = create_job(session, job_type="inference", idempotency_key=idempotency_key, payload={
                    "executor": "local", "modality": "camera", "source": display,
                    "model_version_id": str(model_record.id), "installation_id": self.installation,
                    "confidence_threshold": threshold, "cell_grid": format_cell_grid(grid), "pid": os.getpid(),
                    **({"capture": "browser_tab", "send_fps": fps or 10} if is_tab else {})})
            if not created:
                raise CameraBusy(job.id, "This Idempotency-Key was already used; start a new session with a new key")
            try:
                with self.factory() as session:
                    token = claim_execution_slot(session, job.id, self.settings.worker_id)
            except ExecutionSlotBusy as exc:
                with self.factory() as session:
                    transition_job(session, job.id, JobState.CANCELLED, phase="slot_busy",
                                   event_type="camera_slot_busy")
                raise CameraBusy(job.id, "The shared compute slot is busy; the camera request was cancelled") from exc
            self._job_id, self._fence = job.id, (self.settings.worker_id, token)
            capture_factory = self.capture_factory
            if is_tab:
                # Created before the model loads so early browser frames are accepted rather than rejected.
                self._push = push = PushCapture(fps or 10, read_timeout=self.push_read_timeout,
                                                first_frame_timeout=self.push_first_frame_timeout)
                capture_factory = lambda _source: push  # noqa: E731
            camera: CameraSession | None = None
            try:
                self._transition(JobState.DISPATCHING, "loading_pinned_model", "camera_claimed",
                                 {"model_version_id": str(model_record.id), "execution_token": token})
                with self.factory() as session:
                    pipeline, loaded = load_approved_pipeline(session, self.settings, pinned_model_id=model_record.id,
                                                              inference_factory=self.inference_factory, cell_grid=grid)
                apply_confidence_threshold(pipeline, threshold)
                camera = CameraSession(pipeline, source, display_source=display, capture_factory=capture_factory)
                geometry = camera.start()
                self._session = camera
                self._payload(model_version_id=str(loaded.id), camera_geometry=geometry)
                self._transition(JobState.RUNNING, "camera_running", "camera_started", geometry)
            except StaleExecutionOwner:
                self._session = None
                self._push = None
                raise
            except Exception as exc:
                self._session = None
                if camera is not None:
                    try:
                        camera.stop(timeout=self.stop_timeout)
                    except CameraStopUnconfirmed:
                        # Keep the slot: the device may still be in use.
                        raise
                self._fail_and_release("camera_start_failed", exc)
                if isinstance(exc, (CameraConnectError, InferenceUnavailable)):
                    raise
                raise CameraConnectError(f"Camera start failed ({type(exc).__name__})") from exc
            return self.status()

    def _fail_and_release(self, event: str, exc: BaseException | None = None, *, summary: dict[str, Any] | None = None) -> None:
        details = {"error_type": type(exc).__name__} if exc is not None else {}
        if summary is not None:
            self._payload(camera_summary=summary)
        job = self._transition(JobState.FAILED, "camera_failed", event, details)
        self._release()
        self._forget()
        del job

    def _finalize_if_dead(self) -> None:
        """If the camera thread died by itself, record the failure and free the slot once stopped."""
        session = self._session
        if session is None or session.state != "failed":
            return
        try:
            summary = session.stop(timeout=self.stop_timeout)
        except CameraStopUnconfirmed:
            return
        self._last_summary = summary
        current = self._job_state()
        if current == JobState.RUNNING.value:
            self._fail_and_release("camera_stream_failed", summary=summary)

    def _job_state(self) -> str | None:
        if self._job_id is None:
            return None
        with self.factory() as session:
            job = session.get(Job, self._job_id)
            return job.state if job else None

    def stop(self) -> dict[str, Any]:
        with self._lock:
            session = self._session
            if session is None:
                raise CameraNotRunning("No camera session is running on this installation")
            job_id = self._job_id
            try:
                summary = session.stop(timeout=self.stop_timeout)
            except CameraStopUnconfirmed:
                if self._job_state() == JobState.RUNNING.value:
                    self._transition(JobState.CANCELLATION_REQUESTED, "camera_stop_unconfirmed",
                                     "camera_stop_requested", {"unconfirmed": True})
                raise
            self._last_summary = summary
            state = self._job_state()
            if state == JobState.RUNNING.value and summary["state"] == "stopped":
                self._payload(camera_summary=summary, result_storage="local_summary_only")
                self._transition(JobState.COLLECTING, "camera_summary_ready", "camera_stopped")
                self._transition(JobState.VALIDATING, "validating_camera_summary", "camera_compute_stopped")
                self._transition(JobState.COMPLETED, "complete", "camera_session_completed")
                self._release()
                self._forget()
            elif state == JobState.RUNNING.value:
                self._fail_and_release("camera_stream_failed", summary=summary)
            elif state == JobState.CANCELLATION_REQUESTED.value:
                self._transition(JobState.CANCELLED, "compute_stopped", "camera_stop_confirmed")
                self._release()
                self._forget()
            return {**summary, "job_id": str(job_id) if job_id else None}

    def status(self) -> dict[str, Any]:
        with self._lock:
            if self._session is not None:
                self._finalize_if_dead()
            if self._session is None:
                return {"active": False, "last_session": self._last_summary}
            return {"active": True, "job_id": str(self._job_id), **self._session.summary(),
                    "records": self._session.latest_records()}

    def set_confidence(self, confidence: float) -> dict[str, Any]:
        """Change the per-box threshold of the running session; it applies from the next processed frame."""
        try:
            threshold = resolve_confidence(self.settings, confidence)
        except ValueError as exc:
            raise CameraSourceError(str(exc)) from exc
        with self._lock:
            if self._session is None:
                raise CameraNotRunning("No camera session is running on this installation")
            apply_confidence_threshold(self._session.pipeline, threshold)
            self._payload(confidence_threshold=threshold)
            return {"confidence_threshold": threshold}

    def push_frame(self, jpeg: bytes) -> dict[str, Any]:
        """Accept one browser-captured JPEG for the running ``browser-tab`` session.

        Deliberately does not take ``self._lock``: ``start`` holds it while the model loads and the
        browser keeps posting frames during that time.
        """
        push = self._push
        if push is None or push.released:
            raise CameraNotRunning("No browser-tab camera session is running on this installation")
        frame = decode_frame(jpeg)
        if push.shape is not None and tuple(frame.shape[:2]) != push.shape:
            raise CameraSourceError(f"frame size {frame.shape[1]}x{frame.shape[0]} differs from the session's "
                                    f"{push.shape[1]}x{push.shape[0]}")
        push.push(frame)
        session = self._session
        summary = session.summary() if session is not None else {}
        return {"alert_track_ids": list(summary.get("alert_track_ids", [])),
                "frames_processed": int(summary.get("frames_processed", 0))}

    def owns(self, job_id: UUID) -> bool:
        return self._job_id == job_id and self._session is not None

    def stream(self):
        with self._lock:
            session = self._session
        if session is None:
            raise CameraNotRunning("No camera session is running on this installation")
        return session.mjpeg_chunks()
