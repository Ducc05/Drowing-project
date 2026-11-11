"""Motion condition for drowning alarms.

A person who is really drowning struggles in place or drifts slowly; a rescuer (or any strong swimmer) whom the
detector mislabels as drowning keeps travelling. Per track this measures how far the box centre has wandered in
a sliding time window, in *body lengths* (the box's longer side), so the limit does not depend on camera distance
or resolution. A track that travelled more than ``max_body_lengths`` is not confirmed as drowning. Like the cell
gate this can only remove alarms, never add them. Camera pan/zoom is not compensated for.
"""

from __future__ import annotations

import math
from collections import deque
from statistics import median
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # keep this module (used by config parsing) free of NumPy
    from drowning.inference.tracking import Box

MAX_BODY_LENGTHS_LIMIT = 20.0
_OFF = {"off", "none", "false", "0", "no", "disabled"}


def parse_motion_limit(text: str) -> float | None:
    """Parse the body-length limit; ``off`` disables the gate, anything invalid raises."""
    value = text.strip().casefold()
    if value in _OFF:
        return None
    try:
        limit = float(value)
    except ValueError as exc:
        raise ValueError("motion limit must be a number of body lengths or off") from exc
    if not 0 < limit <= MAX_BODY_LENGTHS_LIMIT:
        raise ValueError(f"motion limit must be in (0, {MAX_BODY_LENGTHS_LIMIT:g}] body lengths")
    return limit


class MotionGate:
    """Sliding-window travelled distance per track, in body lengths. Timestamps are milliseconds."""

    def __init__(self, window_ms: float, max_body_lengths: float):
        if window_ms <= 0:
            raise ValueError("motion window must be positive")
        if max_body_lengths <= 0:
            raise ValueError("motion limit must be positive")
        self.window_ms = float(window_ms)
        self.max_body_lengths = float(max_body_lengths)
        self._samples: dict[str, deque[tuple[float, float, float, float]]] = {}

    def reset(self) -> None:
        self._samples.clear()

    def update(self, timestamp_ms: float, detections: Iterable[tuple[str, "Box"]]) -> dict[str, float]:
        """Record this frame's boxes; return the current body lengths travelled for every track seen now."""
        seen: dict[str, float] = {}
        for track_id, box in detections:
            size = max(box.x2 - box.x1, box.y2 - box.y1, 1e-6)
            history = self._samples.setdefault(track_id, deque())
            if history and timestamp_ms <= history[-1][0]:
                history.clear()  # time went backwards: not one continuous observation
            history.append((timestamp_ms, (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2, size))
            seen[track_id] = self.body_lengths(track_id, timestamp_ms)
        for track_id in list(self._samples):
            history = self._samples[track_id]
            if timestamp_ms - history[-1][0] > self.window_ms:
                del self._samples[track_id]
        return seen

    def body_lengths(self, track_id: str, now_ms: float) -> float:
        """Diagonal of the box that holds the track's centres in the window ending at ``now_ms``, per body length."""
        cutoff = now_ms - self.window_ms
        points = [(cx, cy, size) for t, cx, cy, size in self._samples.get(track_id, ()) if cutoff < t <= now_ms]
        if len(points) < 2:
            return 0.0
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / median(p[2] for p in points)

    def travelling(self, track_id: str, now_ms: float) -> bool:
        return self.body_lengths(track_id, now_ms) > self.max_body_lengths
