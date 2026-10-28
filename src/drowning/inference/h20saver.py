"""External H20Saver engine: one YOLO detector that both finds people and labels their state.

H20Saver (https://github.com/esonhjz/H20Saver, YOLO11x fine-tuned on drowning / swimming / out of water)
replaces the person detector *and* the approved classifier when ``DETECTION_ENGINE=h20saver``. Tracking, the
per-track alert and the web UI stay in this project: each detection's class confidence is folded into the
existing ``TrackAggregator`` so an alert needs a sustained, hysteresis-gated drowning score on one tracker ID.

The confidence of a YOLO box is a post-NMS detector score for its winning class, not a probability over the
three classes. ``detection_scores`` spreads the remainder evenly over the other classes so the aggregator has a
valid vector; reports keep ``scores_are_calibrated_probabilities = False``.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from drowning.inference.aggregation import CLASSES, TrackAggregator, validate_aggregation
from drowning.inference.pipeline import PersonStateInference
from drowning.inference.tracking import Box, ByteTracker

# Exact class names stored in the H20Saver checkpoint -> ontology v1 keys. No fuzzy matching.
H20SAVER_CLASS_MAP = {"drowning": "drowning", "swimming": "swimming", "out of water": "out_of_water"}

# Unbenchmarked starting values: a drowning score of >= 0.5 (EMA over ~1 s) held for 1.5 s raises the alert.
H20SAVER_AGGREGATION = {
    "method": "ema_probabilities", "tau_ms": 1000, "max_gap_ms": 1500, "min_frames": 5, "alert_label": "drowning",
    "enter_probability": 0.50, "exit_probability": 0.30, "min_alert_duration_ms": 1500,
}

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
    others = (1.0 - confidence) / (len(CLASSES) - 1)
    return [confidence if index == class_index else others for index in range(len(CLASSES))]


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
        results = self.model.predict(bgr, conf=self.confidence, device=self.device, verbose=False)
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


class H20SaverInference(PersonStateInference):
    """Same interface and video/camera handling as ``PersonStateInference``, with H20Saver doing recognition.

    The parent constructor validates a Kaggle model bundle, which this engine does not have, so it is not called;
    ``infer_video``, ``draw_annotations`` and ``reset_stream`` are inherited unchanged.
    """

    def __init__(self, detector: H20SaverDetector, *, model_version: str,
                 aggregation: dict[str, Any] | None = None, tracker: Any | None = None):
        if not model_version:
            raise ValueError("model_version must be pinned for the inference job")
        config = dict(aggregation or H20SAVER_AGGREGATION)
        problems = validate_aggregation(config)
        if problems:
            raise ValueError("; ".join(problems))
        self.detector = detector
        self.model_version = model_version
        self.model_kind = "h20saver"
        self.buffer = None
        self.aggregator = TrackAggregator(config)
        self.tracker = tracker if tracker is not None else ByteTracker()

    def infer_image(self, image: Image.Image) -> dict[str, Any]:
        output = []
        for index, (box, class_index) in enumerate(self.detector.detect(image.convert("RGB"))):
            output.append({"detection_id": f"image-{index}", "bbox_xyxy": [box.x1, box.y1, box.x2, box.y2],
                           "detector_confidence": box.confidence, "identity_verified": False,
                           "prediction_branch": "h20saver_frame", "temporal_available": False,
                           "label": CLASSES[class_index],
                           "class_scores": detection_scores(class_index, box.confidence)})
        return {"model_version": self.model_version, "detections": output,
                "prediction_note": "H20Saver detector class confidences, not calibrated drowning probabilities"}

    def process_frame(self, image: Image.Image, timestamp_ms: int, frame_index: int) -> list[dict[str, Any]]:
        detections = self.detector.detect(image)
        tracked = self.tracker.update([box for box, _ in detections])
        records = []
        # ByteTracker returns one (track_id, box) per input detection, in input order.
        for (track_id, box), (_, class_index) in zip(tracked, detections):
            frame_scores = detection_scores(class_index, box.confidence)
            state = self.aggregator.update(str(track_id), timestamp_ms, frame_scores)
            ready = state["track_ready"]
            records.append({"frame_index": frame_index, "timestamp_ms": timestamp_ms, "track_id": track_id,
                            "identity_verified": False, "bbox_xyxy": [box.x1, box.y1, box.x2, box.y2],
                            "detector_confidence": box.confidence,
                            "prediction_branch": "h20saver_tracked" if ready else "h20saver_frame",
                            "temporal_available": False,
                            "label": state["label"] if ready else CLASSES[class_index],
                            "class_scores": state["smoothed_scores"] if ready else frame_scores,
                            "frame_label": CLASSES[class_index], "frame_class_scores": frame_scores,
                            "alert": state["alert"], "track_frames": state["track_frames"]})
        return records


def load_h20saver_pipeline(weights_path: Path, *, confidence: float, device: str) -> tuple[H20SaverInference, H20SaverIdentity]:
    identity = h20saver_identity(weights_path)
    detector = H20SaverDetector(weights_path, confidence=confidence, device=device)
    return H20SaverInference(detector, model_version=identity.display_version), identity
