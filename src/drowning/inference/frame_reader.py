"""Decode video frames on a background thread so decoding overlaps with model inference.

OpenCV releases the GIL while decoding and PyTorch while running the model, so a small bounded
queue is enough to hide the decode time. Frames are delivered strictly in order and the queue size
bounds memory (a 1080p BGR frame is ~6 MB).
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class FrameItem:
    index: int
    frame: np.ndarray | None
    pos_msec: float
    process: bool  # False: a skipped frame, delivered only so a rewritten video keeps its duration


_END = object()


class FrameReader:
    def __init__(self, capture: Any, *, pos_msec_prop: int, frame_stride: int = 1, keep_skipped: bool = False,
                 max_frames: int | None = None, queue_size: int = 8):
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        self.capture, self.pos_msec_prop = capture, pos_msec_prop
        self.frame_stride, self.keep_skipped, self.max_frames = frame_stride, keep_skipped, max_frames
        self.frames_read = 0  # frames successfully grabbed (and decoded when they were decoded)
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="frame-reader", daemon=True)

    def __enter__(self) -> "FrameReader":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _put(self, item: object) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            index = 0
            while not self._stop.is_set() and (self.max_frames is None or index < self.max_frames):
                if not self.capture.grab():
                    break
                process = index % self.frame_stride == 0
                if process or self.keep_skipped:
                    ok, frame = self.capture.retrieve()
                    if not ok:
                        break
                    item = FrameItem(index, frame, float(self.capture.get(self.pos_msec_prop)), process)
                    if not self._put(item):
                        return
                index += 1
                self.frames_read = index
        except BaseException as exc:  # re-raised in the consumer
            self._put(exc)
            return
        self._put(_END)

    def __iter__(self) -> Iterator[FrameItem]:
        while True:
            item = self._queue.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def close(self) -> None:
        self._stop.set()
        while self._thread.is_alive():
            try:
                self._queue.get_nowait()  # unblock a producer waiting on a full queue
            except queue.Empty:
                pass
            self._thread.join(timeout=0.05)
        self.capture.release()
