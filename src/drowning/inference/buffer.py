"""Causal per-track frame buffers with explicit discontinuity resets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FrameCrop:
    track_id: str
    timestamp_ms: int
    crop: Any
    model_version: str
    scene_id: str | None = None


class TemporalWindowBuffer:
    """Sample prior crops per tracker ID; tracker IDs are provisional identities."""

    def __init__(self, sequence_length: int, sample_period_ms: int, max_gap_ms: int):
        if sequence_length < 2 or sample_period_ms <= 0 or max_gap_ms < sample_period_ms:
            raise ValueError("sequence_length >= 2 and max_gap_ms >= sample_period_ms > 0 are required")
        self.sequence_length = sequence_length
        self.sample_period_ms = sample_period_ms
        self.max_gap_ms = max_gap_ms
        self._frames: dict[str, deque[FrameCrop]] = {}
        self._last_seen: dict[str, FrameCrop] = {}

    def reset(self, track_id: str | None = None) -> None:
        if track_id is None:
            self._frames.clear()
            self._last_seen.clear()
        else:
            self._frames.pop(track_id, None)
            self._last_seen.pop(track_id, None)

    def push(self, frame: FrameCrop) -> tuple[FrameCrop, ...] | None:
        if not frame.track_id or not frame.model_version or frame.timestamp_ms < 0:
            raise ValueError("track_id/model_version must be nonempty and timestamp_ms nonnegative")
        last = self._last_seen.get(frame.track_id)
        if last is not None:
            if frame.timestamp_ms <= last.timestamp_ms:
                raise ValueError("timestamps must increase for each track")
            reset_required = (
                frame.model_version != last.model_version
                or frame.scene_id != last.scene_id
                or frame.timestamp_ms - last.timestamp_ms > self.max_gap_ms
            )
            if reset_required:
                self.reset(frame.track_id)
                last = None
        self._last_seen[frame.track_id] = frame
        queue = self._frames.setdefault(frame.track_id, deque(maxlen=self.sequence_length))
        if not queue or frame.timestamp_ms - queue[-1].timestamp_ms >= self.sample_period_ms:
            queue.append(frame)
        if len(queue) < self.sequence_length:
            return None
        return tuple(queue)

    def tracks(self) -> tuple[str, ...]:
        return tuple(sorted(self._frames))
