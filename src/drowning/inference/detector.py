"""Detector interface and optional explicit-weight Ultralytics person detector."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from drowning.inference.tracking import Box


class PersonDetector(Protocol):
    def detect(self, image: Image.Image) -> list[Box]: ...


class UltralyticsPersonDetector:
    """Load user-provided local weights; never download a detector implicitly."""

    def __init__(self, weights_path: Path, *, person_class_id: int = 0, confidence: float = 0.25):
        if not weights_path.is_file():
            raise FileNotFoundError(f"person detector weights not found: {weights_path}")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence threshold must be in [0,1]")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install optional Ultralytics support before using this detector") from exc
        self.model = YOLO(str(weights_path))
        self.person_class_id = person_class_id
        self.confidence = confidence
        names = self.model.names
        if isinstance(names, dict) and person_class_id in names and str(names[person_class_id]).casefold() not in {"person", "people"}:
            raise ValueError("configured detector class ID is not named person")

    def detect(self, image: Image.Image) -> list[Box]:
        results = self.model.predict(np.asarray(image.convert("RGB")), classes=[self.person_class_id],
                                     conf=self.confidence, verbose=False)
        boxes: list[Box] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                coords = box.xyxy[0].detach().cpu().tolist()
                confidence = float(box.conf[0].detach().cpu())
                boxes.append(Box(*map(float, coords), confidence=confidence))
        return boxes
