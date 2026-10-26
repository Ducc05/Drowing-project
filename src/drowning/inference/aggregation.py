"""Per-track aggregation of an image classifier's per-frame scores.

An image model sees one frame at a time. Aggregating its scores over a tracker's provisional ID gives a steadier
label and an alert state, but it does not make the model understand motion, and a tracker ID switch resets the
history of that person. Parameters live in the model bundle (``aggregation``) and are unbenchmarked starting values
until they have been tuned on validation video.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

CLASSES = ("drowning", "swimming", "out_of_water")
REQUIRED_KEYS = {"method", "tau_ms", "max_gap_ms", "min_frames", "alert_label", "enter_probability", "exit_probability",
                 "min_alert_duration_ms"}
DEFAULT_AGGREGATION = {
    "method": "ema_probabilities", "tau_ms": 2000, "max_gap_ms": 1500, "min_frames": 5, "alert_label": "drowning",
    "enter_probability": 0.60, "exit_probability": 0.40, "min_alert_duration_ms": 1500,
}


def validate_aggregation(config: Any) -> list[str]:
    if not isinstance(config, dict) or not REQUIRED_KEYS.issubset(config):
        return ["aggregation contract is missing"]
    errors = []
    if config["method"] != "ema_probabilities":
        errors.append("unsupported aggregation method")
    for key in ("tau_ms", "max_gap_ms", "min_alert_duration_ms"):
        if not isinstance(config[key], (int, float)) or isinstance(config[key], bool) or config[key] <= 0:
            errors.append(f"aggregation {key} must be positive")
    if not isinstance(config["min_frames"], int) or isinstance(config["min_frames"], bool) or config["min_frames"] < 1:
        errors.append("aggregation min_frames must be a positive integer")
    if config["alert_label"] not in CLASSES:
        errors.append("aggregation alert_label must be a class key")
    enter, exit_ = config["enter_probability"], config["exit_probability"]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v <= 1 for v in (enter, exit_)) or exit_ >= enter:
        errors.append("aggregation needs 0 < exit_probability < enter_probability <= 1")
    return errors


@dataclass
class _Track:
    ema: list[float]
    last_ms: int
    first_ms: int
    frames: int = 1
    above_since: int | None = None
    alert: bool = False


@dataclass
class TrackAggregator:
    config: dict[str, Any]
    _tracks: dict[str, _Track] = field(default_factory=dict)

    def __post_init__(self) -> None:
        problems = validate_aggregation(self.config)
        if problems:
            raise ValueError("; ".join(problems))

    def reset(self, track_id: str | None = None) -> None:
        if track_id is None:
            self._tracks.clear()
        else:
            self._tracks.pop(track_id, None)

    def update(self, track_id: str, timestamp_ms: int, probabilities: list[float]) -> dict[str, Any]:
        """Fold one frame's class probabilities into the track and return the smoothed state."""
        cfg = self.config
        alert_index = CLASSES.index(cfg["alert_label"])
        track = self._tracks.get(track_id)
        if track is not None and (timestamp_ms <= track.last_ms or timestamp_ms - track.last_ms > cfg["max_gap_ms"]):
            track = None  # out of order or a long gap: not the same continuous observation
        if track is None:
            track = _Track(ema=list(probabilities), last_ms=timestamp_ms, first_ms=timestamp_ms)
            self._tracks[track_id] = track
        else:
            alpha = 1.0 - math.exp(-(timestamp_ms - track.last_ms) / cfg["tau_ms"])
            track.ema = [(1 - alpha) * old + alpha * new for old, new in zip(track.ema, probabilities)]
            track.last_ms = timestamp_ms
            track.frames += 1
        ready = track.frames >= cfg["min_frames"]
        probability = track.ema[alert_index]
        if ready and probability >= cfg["enter_probability"]:
            if track.above_since is None:
                track.above_since = timestamp_ms
            if timestamp_ms - track.above_since >= cfg["min_alert_duration_ms"]:
                track.alert = True
        elif track.alert and probability >= cfg["exit_probability"]:
            pass  # hysteresis: an active alert holds until the probability falls below the exit level
        else:
            track.above_since, track.alert = None, False
        label = CLASSES[max(range(len(CLASSES)), key=track.ema.__getitem__)] if ready else None
        return {"smoothed_scores": [round(v, 5) for v in track.ema], "label": label, "alert": track.alert and ready,
                "track_frames": track.frames, "track_ready": ready, "observed_ms": timestamp_ms - track.first_ms}
