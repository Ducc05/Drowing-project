"""H20Saver inference with reference Ultralytics tracking and windowed alarm votes."""

from __future__ import annotations

import hashlib
import uuid
from collections import deque
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from drowning.inference.aggregation import CLASSES
from drowning.inference.cells import CellGate, CellGrid
from drowning.inference.motion import MotionGate
from drowning.inference.pipeline import PersonStateInference
from drowning.inference.tracking import Box

# Exact class names stored in the H20Saver checkpoint -> ontology v1 keys. No fuzzy matching.
H20SAVER_CLASS_MAP = {"drowning": "drowning", "swimming": "swimming", "out of water": "out_of_water"}

# Reference H20Saver rule: a drowning box only votes for an alarm at this score. Boxes below the
# user-chosen display/detection threshold are never produced, so a higher threshold also raises this bar.
DROWNING_VOTE_CONFIDENCE = 0.5

_MODEL_NAMESPACE = uuid.UUID("5d0f3b8e-6a3c-4c1e-9a55-2f1b1f0c4d20")


def class_mapping(names: Any) -> dict[int, int]:
    """Map the checkpoint's class IDs to ontology indices; refuse a checkpoint with other classes."""
    items = names.items() if isinstance(names, dict) else enumerate(names)
    mapping: dict[int, int] = {}
    for class_id, name in items:
        key = H20SAVER_CLASS_MAP.get(str(name).strip().casefold())
        if key is None:
            raise ValueError(f"H20Saver checkpoint has an unknown class {name!r}")
        mapping[int(class_id)] = CLASSES.index(key)
    if sorted(mapping.values()) != list(range(len(CLASSES))):
        raise ValueError("H20Saver checkpoint must contain exactly drowning, swimming and out of water")
    return mapping


def detection_scores(class_index: int, confidence: float) -> list[float]:
    # Only the winning detector score is observed; other class scores are unknown.
    return [confidence if index == class_index else 0.0 for index in range(len(CLASSES))]


@dataclass(frozen=True)
class H20SaverIdentity:
    """Stands in for a registry ``ModelVersion``: the engine is pinned by its weights checksum."""

    id: uuid.UUID
    display_version: str
    weights_sha256: str
    status: str = "external"
    model_kind: str = "h20saver"


_identity_cache: dict[tuple[str, int, int], H20SaverIdentity] = {}


def h20saver_identity(weights_path: Path) -> H20SaverIdentity:
    path = weights_path.resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _identity_cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        sha = digest.hexdigest()
        _identity_cache[key] = H20SaverIdentity(id=uuid.uuid5(_MODEL_NAMESPACE, sha),
                                                display_version=f"h20saver-{sha[:12]}", weights_sha256=sha)
    return _identity_cache[key]


class H20SaverDetector:
    """Load local H20Saver weights; never download them implicitly."""

    def __init__(self, weights_path: Path, *, confidence: float = 0.25, device: str = "cpu", model: Any = None):
        if model is None:
            if not weights_path.is_file():
                raise FileNotFoundError(f"H20Saver weights not found: {weights_path}")
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Install optional Ultralytics support before using H20Saver") from exc
            model = YOLO(str(weights_path))
        if not 0 <= confidence <= 1:
            raise ValueError("confidence threshold must be in [0,1]")
        self.model = model
        self.confidence = confidence
        self.device = device
        self.class_map = class_mapping(model.names)

    def detect(self, image: Image.Image) -> list[tuple[Box, int]]:
        """Return ``(box, ontology class index)`` per detected person."""
        # Ultralytics treats NumPy input as BGR (see UltralyticsPersonDetector).
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[..., ::-1])
        results = self.model.predict(bgr, conf=self.confidence, imgsz=640, device=self.device, verbose=False)
        detections: list[tuple[Box, int]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                coords = box.xyxy[0].detach().cpu().tolist()
                confidence = float(box.conf[0].detach().cpu())
                class_index = self.class_map[int(box.cls[0].detach().cpu())]
                detections.append((Box(*map(float, coords), confidence=confidence), class_index))
        return detections

    def track(self, image):
        # Video decoding already yields BGR ndarrays: pass them straight through (no RGB round trip).
        bgr = (np.ascontiguousarray(image) if isinstance(image, np.ndarray)
               else np.ascontiguousarray(np.asarray(image.convert("RGB"))[..., ::-1]))
        result = self.model.track(bgr, persist=True, tracker="bytetrack.yaml", conf=self.confidence,
                                  imgsz=640, device=self.device, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        return [(f"trk-{int(track_id):06d}", Box(*map(float, coords), confidence=float(confidence)),
                 self.class_map[int(class_id)])
                for track_id, coords, confidence, class_id in zip(boxes.id.int().tolist(), boxes.xyxy.tolist(),
                                                                 boxes.conf.tolist(), boxes.cls.int().tolist())]


class DrowningMonitor:
    """Reference H20Saver window votes, including full-window warmup and hysteresis."""
    def __init__(self, window_frames=60, alarm_ratio=0.7, clear_ratio=0.3):
        self.window_frames = window_frames
        self.alarm_ratio, self.clear_ratio = alarm_ratio, clear_ratio
        self.history, self.last_seen, self.alarmed = {}, {}, set()
        self.ratios = {}  # latest vote ratio per track (diagnostics)

    def update(self, frame_index, track_ids, votes):
        for track_id, vote in zip(track_ids, votes):
            history = self.history.setdefault(track_id, deque(maxlen=self.window_frames))
            history.append(vote)
            self.last_seen[track_id] = frame_index
            ratio = sum(history) / self.window_frames
            self.ratios[track_id] = ratio
            if ratio >= self.alarm_ratio:
                self.alarmed.add(track_id)
            elif ratio < self.clear_ratio:
                self.alarmed.discard(track_id)
        for track_id, seen in list(self.last_seen.items()):
            if frame_index - seen > self.window_frames:
                del self.history[track_id], self.last_seen[track_id]
                self.ratios.pop(track_id, None)
                self.alarmed.discard(track_id)
        return self.alarmed


class LiveDrowningMonitor:
    """Time-weighted votes (two seconds by default) for latest-frame cameras with variable processing FPS."""
    def __init__(self, sample_ms=1000 / 30, window_ms=2000, alarm_ratio=0.7, clear_ratio=0.3):
        self.sample_ms = sample_ms
        self.window_ms = window_ms
        self.alarm_ratio, self.clear_ratio = alarm_ratio, clear_ratio
        self.history, self.last_seen, self.alarmed = {}, {}, set()
        self.ratios = {}

    def update(self, timestamp_ms, track_ids, votes):
        for track_id, vote in zip(track_ids, votes):
            previous = self.last_seen.get(track_id, timestamp_ms - self.sample_ms)
            if timestamp_ms - previous >= self.window_ms or timestamp_ms <= previous:
                self.history.pop(track_id, None)
                self.alarmed.discard(track_id)
                previous = timestamp_ms - self.sample_ms
            history = self.history.setdefault(track_id, deque())
            history.append((previous, timestamp_ms, vote))
            self.last_seen[track_id] = timestamp_ms
            cutoff = timestamp_ms - self.window_ms
            while history and history[0][1] <= cutoff:
                history.popleft()
            ratio = sum(end - max(start, cutoff) for start, end, voted in history if voted) / self.window_ms
            self.ratios[track_id] = ratio
            if ratio >= self.alarm_ratio:
                self.alarmed.add(track_id)
            elif ratio < self.clear_ratio:
                self.alarmed.discard(track_id)
        for track_id, seen in list(self.last_seen.items()):
            if timestamp_ms - seen > self.window_ms:
                del self.history[track_id], self.last_seen[track_id]
                self.ratios.pop(track_id, None)
                self.alarmed.discard(track_id)
        return self.alarmed


def frame_size(image):
    """(width, height) of a PIL image or an OpenCV/NumPy frame."""
    if isinstance(image, np.ndarray):
        return image.shape[1], image.shape[0]
    return image.size


class H20SaverInference(PersonStateInference):
    def __init__(self, detector, *, model_version, cell_grid=None, cell_window_s=3.0,
                 motion_max_body_lengths=None, motion_window_s=3.0):
        """Optional gates that can only remove alarms (None disables each):

        ``cell_grid=(cols, rows)`` confirms drowning alarms per cell (see ``inference.cells``);
        ``motion_max_body_lengths`` refuses a track that travelled further than that in ``motion_window_s``
        (see ``inference.motion``).
        """
        if not model_version:
            raise ValueError("model_version must be pinned for the inference job")
        if cell_window_s <= 0:
            raise ValueError("cell_window_s must be positive")
        if motion_window_s <= 0:
            raise ValueError("motion_window_s must be positive")
        if motion_max_body_lengths is not None and motion_max_body_lengths <= 0:
            raise ValueError("motion_max_body_lengths must be positive")
        self.motion_max_body_lengths = None if motion_max_body_lengths is None else float(motion_max_body_lengths)
        self.motion_window_s = float(motion_window_s)
        self.motion_gate = None
        self.cell_grid = CellGrid(*cell_grid) if cell_grid else None
        self.cell_window_s = float(cell_window_s)
        self.cell_gate = None
        self._confirmed = set()
        self.detector, self.model_version = detector, model_version
        self.model_kind = "h20saver"
        self.accepts_bgr = True  # infer_video may hand frames over as OpenCV BGR arrays
        self.frame_stride = 1
        self.tracker = SimpleNamespace(name="ultralytics_bytetrack")
        self.monitor = DrowningMonitor()
        self.label_mode = "alarm"
        self.frame_ms = 1000 / 30
        self._alert_intervals = []
        self._open_alerts = {}
        self._rebuild_gates(live=False)

    def _rebuild_gates(self, *, live):
        """Fresh gate state: per-cell vote windows in the per-track monitor's clock, and motion history."""
        self._confirmed = set()
        self.motion_gate = (None if self.motion_max_body_lengths is None
                            else MotionGate(self.motion_window_s * 1000, self.motion_max_body_lengths))
        self._cell_intervals, self._open_cells = [], {}
        if self.cell_grid is None:
            self.cell_gate = None
        elif live:
            self.cell_gate = CellGate(self.cell_grid, LiveDrowningMonitor(self.frame_ms, window_ms=self.cell_window_s * 1000))
        else:
            self.cell_gate = CellGate(self.cell_grid, DrowningMonitor(max(1, round(self.cell_window_s * 1000 / self.frame_ms))))

    def reset_stream(self):
        self.monitor = DrowningMonitor(max(1, round(2000 / self.frame_ms)))
        self._alert_intervals = []
        self._open_alerts = {}
        self._rebuild_gates(live=False)
        predictor = getattr(getattr(self.detector, "model", None), "predictor", None)
        for tracker in getattr(predictor, "trackers", []):
            tracker.reset()

    def configure_stream(self, fps, frame_stride=None):
        """Offline video: the 2 s vote window holds ``2*fps/stride`` processed samples.

        With ``frame_stride=1`` this is exactly the reference per-frame window. A larger stride keeps the
        same duration and the same alarm/clear ratios; ``frame_ms`` becomes the sample period.
        """
        self.frame_stride = max(1, int(frame_stride or self.frame_stride))
        self.frame_ms = 1000 * self.frame_stride / fps
        self.monitor = DrowningMonitor(max(1, round(2 * fps / self.frame_stride)))
        self._rebuild_gates(live=False)

    def configure_live_stream(self, fps):
        self.frame_stride = 1
        self.frame_ms = 1000 / fps
        self.monitor = LiveDrowningMonitor(self.frame_ms)
        self._rebuild_gates(live=True)

    @property
    def active_alert_tracks(self):
        return sorted(self._confirmed)

    @property
    def cell_overlay(self):
        """Grid and currently alarmed cells for the renderers, or None when the gate is off."""
        if self.cell_gate is None:
            return None
        return {"cols": self.cell_grid.cols, "rows": self.cell_grid.rows, "hot": sorted(self.cell_gate.alarmed_cells)}

    def result_metadata(self):
        metadata = {"alert_intervals": self._alert_intervals, "inference_policy": "h20saver_reference_votes_v1",
                    "drowning_vote_confidence": DROWNING_VOTE_CONFIDENCE}
        if self.motion_gate is not None:
            metadata["inference_policy"] = "h20saver_motion_gated_votes_v1"
            metadata["motion_gate"] = {"max_body_lengths": self.motion_max_body_lengths,
                                       "window_s": self.motion_window_s}
        if self.cell_grid is not None:
            metadata["inference_policy"] = ("h20saver_cell_motion_gated_votes_v1" if self.motion_gate is not None
                                            else "h20saver_cell_gated_votes_v1")
            metadata["cell_gate"] = {"cols": self.cell_grid.cols, "rows": self.cell_grid.rows,
                                     "window_s": self.cell_window_s}
            metadata["cell_intervals"] = self._cell_intervals
        return metadata

    def infer_video(self, video_path, output_video, *, frame_stride=3, **kwargs):
        self.frame_stride = frame_stride  # read by configure_stream, which the base class calls with the fps
        self.reset_stream()
        return super().infer_video(video_path, output_video, frame_stride=frame_stride, **kwargs)

    def infer_image(self, image):
        output = []
        for index, (box, class_index) in enumerate(self.detector.detect(image)):
            output.append({"detection_id": f"image-{index}", "bbox_xyxy": [box.x1, box.y1, box.x2, box.y2],
                           "detector_confidence": box.confidence, "identity_verified": False,
                           "prediction_branch": "h20saver_frame", "temporal_available": False,
                           "label": CLASSES[class_index], "class_scores": detection_scores(class_index, box.confidence),
                           "score_semantics": "winning_class_confidence_only"})
        return {"model_version": self.model_version, "detections": output,
                "prediction_note": "Winning class confidence only; zeros mean unobserved, not probabilities"}

    def process_frame(self, image, timestamp_ms, frame_index):
        detections = self.detector.track(image)
        ids = [track_id for track_id, _, _ in detections]
        votes = [class_index == 0 and box.confidence >= DROWNING_VOTE_CONFIDENCE for _, box, class_index in detections]
        clock = timestamp_ms if isinstance(self.monitor, LiveDrowningMonitor) else frame_index // self.frame_stride
        track_alarmed = self.monitor.update(clock, ids, votes)
        if self.cell_gate is None:
            cell_confirmed = track_alarmed
        else:
            # A track alarm only counts while the cell(s) it occupies hold sustained drowning evidence.
            cell_confirmed = self.cell_gate.update(clock, frame_size(image),
                                                   [(tid, box, vote) for (tid, box, _), vote in zip(detections, votes)],
                                                   track_alarmed)
        motion = {}
        if self.motion_gate is None:
            alarmed = cell_confirmed
        else:
            # ... and only while that track is not travelling: a swimmer who keeps moving is not a drowning person.
            motion = self.motion_gate.update(timestamp_ms, [(tid, box) for tid, box, _ in detections])
            alarmed = {tid for tid in cell_confirmed if not self.motion_gate.travelling(tid, timestamp_ms)}
        self._confirmed = set(alarmed)
        if self.cell_gate is not None:
            hot = self.cell_gate.alarmed_cells
            for cell in list(self._open_cells):
                if cell not in hot:
                    self._open_cells.pop(cell)["end"] = timestamp_ms
            for cell in sorted(hot):
                segment = self._open_cells.get(cell)
                if segment is None:
                    segment = {"cell": list(cell), "start": timestamp_ms, "end": timestamp_ms}
                    self._open_cells[cell] = segment
                    self._cell_intervals.append(segment)
                segment["end"] = timestamp_ms + self.frame_ms
        for track_id in list(self._open_alerts):
            if track_id not in alarmed:
                self._open_alerts.pop(track_id)["end"] = timestamp_ms
        confidence_by_id = {tid: box.confidence for tid, box, cls in detections if cls == 0}
        for track_id in sorted(alarmed):
            segment = self._open_alerts.get(track_id)
            if segment is None:
                segment = {"track_id": track_id, "start": timestamp_ms, "end": timestamp_ms, "peak": None}
                self._open_alerts[track_id] = segment
                self._alert_intervals.append(segment)
            segment["end"] = timestamp_ms + self.frame_ms
            if track_id in confidence_by_id:
                segment["peak"] = max(segment["peak"] or 0, confidence_by_id[track_id])
        return [{"frame_index": frame_index, "timestamp_ms": timestamp_ms, "track_id": track_id,
                 "identity_verified": False, "bbox_xyxy": [box.x1, box.y1, box.x2, box.y2],
                 "detector_confidence": box.confidence, "prediction_branch": "h20saver_tracked",
                 "temporal_available": False, "label": CLASSES[class_index],
                 "frame_label": CLASSES[class_index], "class_scores": detection_scores(class_index, box.confidence),
                 "score_semantics": "winning_class_confidence_only", "label_mode": self.label_mode,
                 "alert": track_id in alarmed, "alert_suppressed_by_cell": track_id in track_alarmed and track_id not in cell_confirmed,
                 "alert_suppressed_by_motion": track_id in cell_confirmed and track_id not in alarmed,
                 "track_frames": len(self.monitor.history[track_id]),
                 # Why a drowning label may not be an alert yet: this box's vote, and its track's vote ratio
                 # (an alarm needs >= 0.7 of the window, and a new track ID starts again from zero).
                 "drowning_vote": vote, "drowning_ratio": round(self.monitor.ratios.get(track_id, 0.0), 2),
                 **({"motion": round(motion[track_id], 2)} if track_id in motion else {})}
                for (track_id, box, class_index), vote in zip(detections, votes)]

    def draw_annotations(self, bgr, records):
        import cv2
        frame = bgr.copy()
        if self.cell_overlay:
            from drowning.inference.preview import draw_cells
            draw_cells(frame, self.cell_overlay)
        colors = {"drowning": (0, 165, 255), "swimming": (80, 200, 80), "out_of_water": (170, 170, 170)}
        for record in records:
            x1, y1, x2, y2 = map(int, record["bbox_xyxy"])
            alerted = record["alert"]
            color = (0, 0, 255) if alerted else colors[record["label"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if alerted else 2)
            # Every box shows its own score; the alert decision is still per track over time.
            caption = (f"{record['track_id'].replace('trk-', '#')} "
                       + ("DROWNING ALERT" if alerted else record["label"])
                       + f" {record['detector_confidence']:.2f}")
            (width, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, .5, 2)
            cv2.putText(frame, caption, (max(0, min(x1, frame.shape[1] - width - 2)), max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2, cv2.LINE_AA)
        if self._confirmed:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 200), -1)
            cv2.putText(frame, "WARNING: DROWNING DETECTED", (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, (255, 255, 255), 2, cv2.LINE_AA)
        return frame


def load_h20saver_pipeline(weights_path: Path, *, confidence: float, device: str, cell_grid=None,
                           cell_window_s: float = 3.0, motion_max_body_lengths=None,
                           motion_window_s: float = 3.0) -> tuple[H20SaverInference, H20SaverIdentity]:
    identity = h20saver_identity(weights_path)
    detector = H20SaverDetector(weights_path, confidence=confidence, device=device)
    return H20SaverInference(detector, model_version=identity.display_version, cell_grid=cell_grid,
                             cell_window_s=cell_window_s, motion_max_body_lengths=motion_max_body_lengths,
                             motion_window_s=motion_window_s), identity
