"""Live camera session: latest-frame capture, per-frame inference and an MJPEG preview.

A reader thread keeps only the newest frame so a slow model drops frames instead of building
up latency; a worker thread runs detection/tracking/classification. ``stop`` only reports a
confirmed stop after both threads have exited and the capture device has been released.
"""

from __future__ import annotations

import io
import re
import threading
import time
from collections import deque
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from PIL import Image

BROWSER_TAB_SOURCE = "browser-tab"
MAX_FRAME_BYTES = 2_000_000
MAX_FRAME_SIDE = 1920
_ALLOWED_SCHEMES = {"http", "https", "rtsp", "rtmp"}
_MAX_CAMERA_INDEX = 15
_STREAM_BOUNDARY = "frame"


class CameraSourceError(ValueError):
    """The requested camera source is not an allowed webcam index or stream URL."""


class CameraConnectError(RuntimeError):
    """The camera could not be opened or produced no frame."""


class CameraStopUnconfirmed(RuntimeError):
    """A camera thread did not exit in time; compute must not be reported as stopped."""


def parse_camera_source(raw: str) -> tuple[int | str, str]:
    """Return ``(opencv_source, display_source)`` for a webcam index or network stream URL.

    Local file paths are refused so this endpoint cannot be used to read arbitrary files, and
    credentials embedded in a URL are removed from every displayed or stored form.
    """
    value = (raw or "").strip()
    if value == BROWSER_TAB_SOURCE:
        return BROWSER_TAB_SOURCE, BROWSER_TAB_SOURCE
    if re.fullmatch(r"\d{1,2}", value):
        index = int(value)
        if index > _MAX_CAMERA_INDEX:
            raise CameraSourceError(f"webcam index must be between 0 and {_MAX_CAMERA_INDEX}")
        return index, f"webcam:{index}"
    parts = urlsplit(value)
    if parts.scheme.lower() not in _ALLOWED_SCHEMES or not parts.hostname:
        raise CameraSourceError("camera source must be a webcam index or an http/https/rtsp/rtmp URL")
    return value, redact_source(value)


def redact_source(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def open_capture(source: int | str, capture_factory: Callable[[Any], Any] | None = None) -> Any:
    if capture_factory is None:
        import cv2
        capture_factory = cv2.VideoCapture
    capture = capture_factory(source)
    if not capture.isOpened():
        capture.release()
        raise CameraConnectError("Unable to open the camera or stream")
    return capture


def probe_camera(source: int | str, capture_factory: Callable[[Any], Any] | None = None) -> dict[str, Any]:
    """Open the source, read one frame and release it; used to validate a connection."""
    capture = open_capture(source, capture_factory)
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise CameraConnectError("The camera opened but returned no frame")
        height, width = frame.shape[:2]
        return {"connected": True, "width": int(width), "height": int(height)}
    finally:
        capture.release()


def decode_frame(jpeg: bytes, *, max_side: int = MAX_FRAME_SIDE) -> np.ndarray:
    """Decode a browser-supplied JPEG to a BGR frame, rejecting oversized or invalid input before decoding."""
    import cv2

    if not jpeg:
        raise CameraSourceError("frame is empty")
    if len(jpeg) > MAX_FRAME_BYTES:
        raise CameraSourceError("frame is larger than the allowed size")
    try:
        with Image.open(io.BytesIO(jpeg)) as header:
            width, height = header.size
    except Exception as exc:
        raise CameraSourceError("frame is not a valid image") from exc
    if max(width, height) > max_side:
        raise CameraSourceError(f"frame is larger than {max_side} pixels on its longest side")
    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise CameraSourceError("frame is not a valid image")
    return frame


class PushCapture:
    """VideoCapture-like source fed by ``push``; ``read`` returns only frames it has not returned yet.

    Only the newest frame is kept, so a slow model drops frames instead of accumulating latency.
    """

    def __init__(self, fps: float = 10.0, *, read_timeout: float = 1.0, first_frame_timeout: float = 20.0):
        self.fps = float(fps)
        self.read_timeout = read_timeout
        self.first_frame_timeout = first_frame_timeout
        self.released = False
        self.shape: tuple[int, int] | None = None
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._read_seq = 0
        self._has_read = False

    def isOpened(self) -> bool:
        return not self.released

    def push(self, frame: np.ndarray) -> None:
        with self._condition:
            if self.released:
                return
            if self.shape is None:
                self.shape = (int(frame.shape[0]), int(frame.shape[1]))
            self._frame = frame.copy()
            self._seq += 1
            self._condition.notify_all()

    def read(self) -> tuple[bool, np.ndarray | None]:
        timeout = self.read_timeout if self._has_read else self.first_frame_timeout
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._seq == self._read_seq and not self.released:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._condition.wait(remaining)
            if self.released or self._frame is None:
                return False, None
            self._read_seq = self._seq
            self._has_read = True
            return True, self._frame.copy()

    def release(self) -> None:
        with self._condition:
            self.released = True
            self._condition.notify_all()

    def get(self, prop_id: int) -> float:
        return self.fps if prop_id == 5 else 0.0  # 5 == cv2.CAP_PROP_FPS


class CameraSession:
    def __init__(self, pipeline: Any, source: int | str, *, display_source: str,
                 capture_factory: Callable[[Any], Any] | None = None, jpeg_quality: int = 80,
                 clock: Callable[[], float] = time.monotonic):
        self.pipeline = pipeline
        self.source = source
        self.display_source = display_source
        self.capture_factory = capture_factory
        self.jpeg_quality = jpeg_quality
        self.clock = clock
        self.state = "created"
        self.error: str | None = None
        self.frames_read = 0
        self.frames_processed = 0
        self.detections_total = 0
        self.width = self.height = 0
        self.started_wall: float | None = None
        self.stopped_wall: float | None = None
        self._capture: Any = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._latest_frame: Any = None
        self._latest_seq = 0
        self._latest_jpeg: bytes | None = None
        self._latest_records: list[dict[str, Any]] = []
        self._latest_alert_tracks: list[str] = []
        self._latencies_ms: deque[float] = deque(maxlen=300)
        self._reader: threading.Thread | None = None
        self._worker: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------------------------
    def start(self) -> dict[str, Any]:
        """Open the camera and fail fast if it yields no frame; then start both threads."""
        if self.state != "created":
            raise RuntimeError("a camera session can be started only once")
        self.state = "starting"
        capture = open_capture(self.source, self.capture_factory)
        try:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise CameraConnectError("The camera opened but returned no frame")
        except Exception as exc:
            capture.release()
            self.state, self.error = "failed", type(exc).__name__
            raise
        self.height, self.width = frame.shape[:2]
        self._capture = capture
        self._latest_frame, self._latest_seq, self.frames_read = frame, 1, 1
        self.pipeline.reset_stream()
        if hasattr(self.pipeline, "configure_live_stream"):
            import cv2
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            self.pipeline.configure_live_stream(fps if 0 < fps < 1000 else 30.0)
        self.started_wall = self.clock()
        self._reader = threading.Thread(target=self._read_loop, name="camera-reader", daemon=True)
        self._worker = threading.Thread(target=self._process_loop, name="camera-worker", daemon=True)
        self.state = "running"
        self._reader.start()
        self._worker.start()
        return {"width": self.width, "height": self.height}

    def stop(self, timeout: float = 10.0) -> dict[str, Any]:
        """Stop both threads and release the device. Raises if exit cannot be confirmed."""
        self._stop.set()
        with self._frame_ready:
            self._frame_ready.notify_all()
        deadline = self.clock() + timeout
        for thread in (self._reader, self._worker):
            if thread is not None and thread is not threading.current_thread():
                thread.join(max(0.0, deadline - self.clock()))
        alive = [t.name for t in (self._reader, self._worker) if t is not None and t.is_alive()]
        if alive:
            if self.state == "running":
                self.state = "stopping"
            raise CameraStopUnconfirmed("camera thread(s) still running: " + ", ".join(alive))
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self.state != "failed":
            self.state = "stopped"
        self.stopped_wall = self.clock()
        return self.summary()

    def is_alive(self) -> bool:
        return any(t is not None and t.is_alive() for t in (self._reader, self._worker))

    # -- threads ---------------------------------------------------------------------------
    def _fail(self, exc: BaseException) -> None:
        self.error = type(exc).__name__
        self.state = "failed"
        self._stop.set()
        with self._frame_ready:
            self._frame_ready.notify_all()

    def _read_loop(self) -> None:
        consecutive_failures = 0
        try:
            while not self._stop.is_set():
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        raise CameraConnectError("camera stream stopped delivering frames")
                    time.sleep(0.02)
                    continue
                consecutive_failures = 0
                with self._frame_ready:
                    self._latest_frame = frame
                    self._latest_seq += 1
                    self.frames_read += 1
                    self._frame_ready.notify_all()
        except Exception as exc:
            self._fail(exc)

    def _process_loop(self) -> None:
        import cv2

        processed_seq, last_timestamp = 0, -1
        try:
            while not self._stop.is_set():
                with self._frame_ready:
                    while self._latest_seq == processed_seq and not self._stop.is_set():
                        self._frame_ready.wait(0.25)
                    if self._stop.is_set():
                        break
                    frame, processed_seq = self._latest_frame, self._latest_seq
                began = self.clock()
                timestamp_ms = int(round((began - self.started_wall) * 1000))
                timestamp_ms = max(timestamp_ms, last_timestamp + 1)
                last_timestamp = timestamp_ms
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                records = self.pipeline.process_frame(image, timestamp_ms, self.frames_processed)
                annotated = self.pipeline.draw_annotations(frame, records)
                encoded_ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                with self._lock:
                    self.frames_processed += 1
                    self.detections_total += len(records)
                    self._latest_records = records
                    self._latest_alert_tracks = list(getattr(self.pipeline, "active_alert_tracks",
                        [row["track_id"] for row in records if row.get("alert")]))
                    if encoded_ok:
                        self._latest_jpeg = encoded.tobytes()
                    self._latencies_ms.append((self.clock() - began) * 1000)
        except Exception as exc:
            self._fail(exc)

    # -- observation -----------------------------------------------------------------------
    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def latest_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._latest_records)

    def mjpeg_chunks(self, poll_seconds: float = 0.05) -> Iterator[bytes]:
        """Multipart MJPEG body; ends when the session stops or fails."""
        last = None
        while self.state in {"running", "starting"} and not self._stop.is_set():
            jpeg = self.latest_jpeg()
            if jpeg is not None and jpeg is not last:
                last = jpeg
                yield (f"--{_STREAM_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
                       .encode("ascii") + jpeg + b"\r\n")
            else:
                time.sleep(poll_seconds)

    def summary(self) -> dict[str, Any]:
        with self._lock:
            latencies = sorted(self._latencies_ms)
            frames_processed, detections = self.frames_processed, self.detections_total
            alert_tracks = list(self._latest_alert_tracks)
        end = self.stopped_wall if self.stopped_wall is not None else self.clock()
        elapsed = max(0.0, end - self.started_wall) if self.started_wall is not None else 0.0

        def percentile(q: float) -> float | None:
            return round(latencies[min(len(latencies) - 1, int(q * len(latencies)))], 2) if latencies else None

        return {"source": self.display_source, "state": self.state, "error": self.error,
                "width": self.width, "height": self.height, "frames_read": self.frames_read,
                "frames_processed": frames_processed, "frames_dropped": max(0, self.frames_read - frames_processed),
                "detections_total": detections, "elapsed_seconds": round(elapsed, 2),
                "processed_fps": round(frames_processed / elapsed, 2) if elapsed > 0 else None,
                "latency_ms_p50": percentile(0.5), "latency_ms_p95": percentile(0.95),
                "tracker": getattr(getattr(self.pipeline, "tracker", None), "name", None),
                "timestamps": "monotonic_wall_clock_ms",
                "scores_are_calibrated_probabilities": False, "alert_track_ids": alert_tracks,
                "confidence_threshold": getattr(self.pipeline, "confidence_threshold", None)}

    status = summary


STREAM_MEDIA_TYPE = f"multipart/x-mixed-replace; boundary={_STREAM_BOUNDARY}"
