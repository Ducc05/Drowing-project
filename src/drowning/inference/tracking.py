"""Deterministic trackers. IDs are provisional (never verified identity) and can switch.

``ByteTracker`` is the default for video/camera inference; ``GreedyIoUTracker`` is kept as a
baseline for comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0


@dataclass
class _Track:
    track_id: str
    box: Box
    missed_frames: int = 0


def intersection_over_union(left: Box, right: Box) -> float:
    x1, y1 = max(left.x1, right.x1), max(left.y1, right.y1)
    x2, y2 = min(left.x2, right.x2), min(left.y2, right.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    area_right = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


class GreedyIoUTracker:
    """Frame-local greedy matching for demos; not a validated identity tracker."""

    name = "greedy_iou_provisional_ids"

    def __init__(self, *, min_iou: float = 0.3, max_missed_frames: int = 8):
        if not 0 <= min_iou <= 1 or max_missed_frames < 0:
            raise ValueError("invalid IoU tracker parameters")
        self.min_iou = min_iou
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[str, _Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, detections: list[Box]) -> list[tuple[str, Box]]:
        candidates = sorted(
            ((intersection_over_union(track.box, detection), track_id, index)
             for track_id, track in self._tracks.items()
             for index, detection in enumerate(detections)),
            reverse=True,
        )
        matched_tracks: set[str] = set()
        matched_detections: set[int] = set()
        assignments: dict[int, str] = {}
        for iou, track_id, index in candidates:
            if iou < self.min_iou:
                break
            if track_id in matched_tracks or index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(index)
            assignments[index] = track_id
            track = self._tracks[track_id]
            track.box = detections[index]
            track.missed_frames = 0
        for track_id, track in list(self._tracks.items()):
            if track_id not in matched_tracks:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    del self._tracks[track_id]
        for index, detection in enumerate(detections):
            if index not in assignments:
                track_id = f"trk-{self._next_id:06d}"
                self._next_id += 1
                self._tracks[track_id] = _Track(track_id, detection)
                assignments[index] = track_id
        return [(assignments[index], detection) for index, detection in enumerate(detections)]


def _iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pairwise IoU for ``(n,4)`` and ``(m,4)`` xyxy arrays."""
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)))
    x1 = np.maximum(left[:, None, 0], right[None, :, 0])
    y1 = np.maximum(left[:, None, 1], right[None, :, 1])
    x2 = np.minimum(left[:, None, 2], right[None, :, 2])
    y2 = np.minimum(left[:, None, 3], right[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_l = np.clip(left[:, 2] - left[:, 0], 0, None) * np.clip(left[:, 3] - left[:, 1], 0, None)
    area_r = np.clip(right[:, 2] - right[:, 0], 0, None) * np.clip(right[:, 3] - right[:, 1], 0, None)
    union = area_l[:, None] + area_r[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def linear_sum_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment (Hungarian algorithm with potentials) for a rectangular matrix."""
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return []
    transposed = rows > cols
    matrix = cost.T if transposed else cost
    n, m = matrix.shape
    inf = float("inf")
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    owner = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        owner[0] = i
        j0 = 0
        minv = np.full(m + 1, inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = owner[j0]
            delta, j1 = inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = matrix[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j], way[j] = current, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
    pairs = [(int(owner[j]) - 1, j - 1) for j in range(1, m + 1) if owner[j]]
    return sorted((c, r) if transposed else (r, c) for r, c in pairs)


class _KalmanTrack:
    """Constant-velocity Kalman filter over (cx, cy, aspect, height)."""

    _POSITION_WEIGHT, _VELOCITY_WEIGHT = 1.0 / 20, 1.0 / 160
    _TRANSITION = np.eye(8)
    _TRANSITION[:4, 4:] = np.eye(4)
    _MEASURE = np.eye(4, 8)

    def __init__(self, track_id: str, box: Box):
        self.track_id = track_id
        self.box = box
        height = max(box.y2 - box.y1, 1e-3)
        measurement = np.array([(box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2, (box.x2 - box.x1) / height, height])
        self.mean = np.concatenate([measurement, np.zeros(4)])
        std = [2 * self._POSITION_WEIGHT * height] * 2 + [1e-2] + [2 * self._POSITION_WEIGHT * height]
        std += [10 * self._VELOCITY_WEIGHT * height] * 2 + [1e-5] + [10 * self._VELOCITY_WEIGHT * height]
        self.covariance = np.diag(np.square(std))
        self.lost_frames = 0

    def xyxy(self) -> np.ndarray:
        cx, cy, aspect, height = self.mean[:4]
        width = aspect * height
        return np.array([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])

    def predict(self) -> None:
        height = max(self.mean[3], 1e-3)
        std_pos = [self._POSITION_WEIGHT * height] * 2 + [1e-2, self._POSITION_WEIGHT * height]
        std_vel = [self._VELOCITY_WEIGHT * height] * 2 + [1e-5, self._VELOCITY_WEIGHT * height]
        process = np.diag(np.square(np.concatenate([std_pos, std_vel])))
        self.mean = self._TRANSITION @ self.mean
        self.covariance = self._TRANSITION @ self.covariance @ self._TRANSITION.T + process
        self.mean[3] = max(self.mean[3], 1e-3)

    def update(self, box: Box) -> None:
        height = max(box.y2 - box.y1, 1e-3)
        measurement = np.array([(box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2, (box.x2 - box.x1) / height, height])
        std = [self._POSITION_WEIGHT * self.mean[3]] * 2 + [1e-1, self._POSITION_WEIGHT * self.mean[3]]
        innovation_cov = self._MEASURE @ self.covariance @ self._MEASURE.T + np.diag(np.square(std))
        gain = np.linalg.solve(innovation_cov.T, (self.covariance @ self._MEASURE.T).T).T
        self.mean = self.mean + gain @ (measurement - self._MEASURE @ self.mean)
        self.covariance = self.covariance - gain @ innovation_cov @ gain.T
        self.box = box
        self.lost_frames = 0


def _associate(tracks: list[_KalmanTrack], boxes: np.ndarray, min_iou: float
               ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks or len(boxes) == 0:
        return [], list(range(len(tracks))), list(range(len(boxes)))
    iou = _iou_matrix(np.array([track.xyxy() for track in tracks]), boxes)
    matches = [(r, c) for r, c in linear_sum_assignment(1.0 - iou) if iou[r, c] >= min_iou]
    matched_rows, matched_cols = {r for r, _ in matches}, {c for _, c in matches}
    return (matches, [r for r in range(len(tracks)) if r not in matched_rows],
            [c for c in range(len(boxes)) if c not in matched_cols])


class ByteTracker:
    """ByteTrack-style association: Kalman prediction, Hungarian matching, two confidence tiers.

    Detections at or above ``high_threshold`` are matched first against all live tracks; the
    remaining tracks may then claim low-confidence detections, which keeps a swimmer's ID through
    a momentary confidence dip. Every detection the detector returned is kept: an unmatched one
    starts a new provisional track instead of being discarded. IDs are not verified identities.
    """

    name = "bytetrack_kalman_provisional_ids"

    def __init__(self, *, high_threshold: float = 0.5, first_match_iou: float = 0.2,
                 second_match_iou: float = 0.5, max_lost_frames: int = 30):
        if not 0 <= high_threshold <= 1 or not 0 < first_match_iou <= 1 or not 0 < second_match_iou <= 1:
            raise ValueError("invalid ByteTracker thresholds")
        if max_lost_frames < 0:
            raise ValueError("max_lost_frames must be nonnegative")
        self.high_threshold = high_threshold
        self.first_match_iou = first_match_iou
        self.second_match_iou = second_match_iou
        self.max_lost_frames = max_lost_frames
        self._tracks: list[_KalmanTrack] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, detections: list[Box]) -> list[tuple[str, Box]]:
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections], dtype=float).reshape(-1, 4)
        high = [i for i, d in enumerate(detections) if d.confidence >= self.high_threshold]
        low = [i for i, d in enumerate(detections) if d.confidence < self.high_threshold]
        for track in self._tracks:
            track.predict()
        assignments: dict[int, _KalmanTrack] = {}

        matches, unmatched_tracks, _ = _associate(self._tracks, boxes[high], self.first_match_iou)
        for row, col in matches:
            assignments[high[col]] = self._tracks[row]
        remaining = [self._tracks[row] for row in unmatched_tracks]
        matches, _, _ = _associate(remaining, boxes[low], self.second_match_iou)
        for row, col in matches:
            assignments[low[col]] = remaining[row]

        for index, track in assignments.items():
            track.update(detections[index])
        matched = {id(track) for track in assignments.values()}
        for track in self._tracks:
            if id(track) not in matched:
                track.lost_frames += 1
        self._tracks = [t for t in self._tracks if t.lost_frames <= self.max_lost_frames]
        for index in range(len(detections)):
            if index not in assignments:
                track = _KalmanTrack(f"trk-{self._next_id:06d}", detections[index])
                self._next_id += 1
                self._tracks.append(track)
                assignments[index] = track
        return [(assignments[index].track_id, detections[index]) for index in range(len(detections))]
