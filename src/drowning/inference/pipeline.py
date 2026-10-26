"""Local image/video inference using a pinned temporal bundle and injected person detector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from drowning.data.labels import CANONICAL_LABELS
from drowning.inference.aggregation import TrackAggregator
from drowning.inference.buffer import FrameCrop, TemporalWindowBuffer
from drowning.inference.detector import PersonDetector
from drowning.inference.tracking import Box, ByteTracker
from drowning.ml.bundle import validate_manifest
from drowning.ml.model import CLASS_KEYS, DrowningTemporalModel, torch
from drowning.ml.preprocessing import crop_person, image_to_tensor


def _protect_output(path: Path) -> Path:
    target = path.resolve()
    try:
        target.relative_to(Path("D:/DrowningDemo").resolve())
        under_demo = True
    except ValueError:
        under_demo = False
    if under_demo or any(part.casefold() == "sources" for part in target.parts):
        raise ValueError("Protected D:\\DrowningDemo or sources/ output path")
    return target


class PersonStateInference:
    def __init__(self, model: DrowningTemporalModel, manifest: dict[str, Any], detector: PersonDetector,
                 *, model_version: str, device: str = "cpu", sequence_length: int | None = None,
                 sample_period_ms: int | None = None, max_track_gap_ms: int | None = None,
                 context_fraction: float | None = None, input_size: int | None = None,
                 tracker: Any | None = None):
        errors = validate_manifest(manifest, require_temporal=True)
        if errors:
            raise ValueError("Model cannot be used for temporal inference: " + "; ".join(errors))
        if not model_version:
            raise ValueError("model_version must be pinned for the inference job")
        self.model_kind = manifest.get("model_kind", "temporal")
        temporal_contract = manifest["temporal_contract"] if self.model_kind == "temporal" else None
        preprocessing = manifest["preprocessing"]
        configured_size = preprocessing["size"]
        if isinstance(configured_size, (list, tuple)):
            if len(configured_size) != 2 or configured_size[0] != configured_size[1]:
                raise ValueError("inference currently requires a square model input")
            configured_size = int(configured_size[0])
        contract_values = (sequence_length, sample_period_ms, max_track_gap_ms)
        manifest_values = ((temporal_contract["sequence_length"], temporal_contract["sample_period_ms"], temporal_contract["max_track_gap_ms"])
                           if temporal_contract else (None, None, None))
        if any(given is not None and given != expected for given, expected in zip(contract_values, manifest_values)):
            raise ValueError("inference sampling parameters must exactly match the pinned model bundle")
        if context_fraction is not None and context_fraction != preprocessing["context_fraction"]:
            raise ValueError("crop context must match the pinned model bundle")
        if input_size is not None and input_size != configured_size:
            raise ValueError("input size must match the pinned model bundle")
        self.model = model.to(device).eval()
        self.manifest = manifest
        self.detector = detector
        self.model_version = model_version
        self.device = device
        if temporal_contract:
            self.sequence_length = int(temporal_contract["sequence_length"])
            self.sample_period_ms = int(temporal_contract["sample_period_ms"])
            self.max_track_gap_ms = int(temporal_contract["max_track_gap_ms"])
        else:
            self.sequence_length = self.sample_period_ms = self.max_track_gap_ms = None
        self.context_fraction = float(preprocessing["context_fraction"])
        self.input_size = int(configured_size)
        self.mean = preprocessing["mean"]
        self.std = preprocessing["std"]
        self.buffer = (TemporalWindowBuffer(self.sequence_length, self.sample_period_ms, self.max_track_gap_ms)
                       if temporal_contract else None)
        # image_tracked: the image head scores each frame, and the tracker ID aggregates a person's scores over time.
        self.aggregator = TrackAggregator(manifest["aggregation"]) if self.model_kind == "image_tracked" else None
        self.tracker = tracker if tracker is not None else ByteTracker()

    @torch.inference_mode()
    def infer_image(self, image: Image.Image) -> dict[str, Any]:
        rgb = image.convert("RGB")
        detections = self.detector.detect(rgb)
        crops = [crop_person(rgb, (box.x1, box.y1, box.x2, box.y2),
                            context_fraction=self.context_fraction, output_size=self.input_size)
                 for box in detections]
        tensors = [image_to_tensor(crop, size=self.input_size, mean=self.mean, std=self.std) for crop in crops]
        scores = self.model.forward_images(torch.stack(tensors).to(self.device)) if tensors else None
        output = []
        for index, box in enumerate(detections):
            item = {"detection_id": f"image-{index}", "bbox_xyxy": [box.x1, box.y1, box.x2, box.y2],
                    "detector_confidence": box.confidence, "identity_verified": False,
                    "prediction_branch": "image", "temporal_available": False}
            if scores is not None:
                item.update(self._prediction(scores[index]))
            else:
                item.update({"label": None, "class_scores": None})
            output.append(item)
        return {"model_version": self.model_version, "detections": output,
                "prediction_note": "scores are model outputs, not calibrated drowning probabilities"}

    def reset_stream(self) -> None:
        """Forget tracker and per-track history before an unrelated video or camera session."""
        self.tracker.reset()
        if self.buffer is not None:
            self.buffer.reset()
        if self.aggregator is not None:
            self.aggregator.reset()

    @torch.inference_mode()
    def process_frame(self, image: Image.Image, timestamp_ms: int, frame_index: int) -> list[dict[str, Any]]:
        """Detect, track and classify one RGB frame; one record per detected person."""
        boxes = self.detector.detect(image)
        records = []
        for track_id, box in self.tracker.update(boxes):
            crop = crop_person(image, (box.x1, box.y1, box.x2, box.y2),
                               context_fraction=self.context_fraction, output_size=self.input_size)
            tensor = image_to_tensor(crop, size=self.input_size, mean=self.mean, std=self.std)
            if self.aggregator is not None:
                frame_probs = torch.softmax(self.model.forward_images(tensor.unsqueeze(0).to(self.device))[0].float(), dim=-1)
                state = self.aggregator.update(str(track_id), timestamp_ms, frame_probs.detach().cpu().tolist())
                frame = self._prediction(frame_probs.log())
                ready = state["track_ready"]
                records.append({"frame_index": frame_index, "timestamp_ms": timestamp_ms, "track_id": track_id,
                                "identity_verified": False, "bbox_xyxy": [box.x1, box.y1, box.x2, box.y2],
                                "detector_confidence": box.confidence,
                                "prediction_branch": "image_tracked" if ready else "image_frame", "temporal_available": False,
                                "label": state["label"] if ready else frame["label"],
                                "class_scores": state["smoothed_scores"] if ready else frame["class_scores"],
                                "frame_label": frame["label"], "frame_class_scores": frame["class_scores"],
                                "alert": state["alert"], "track_frames": state["track_frames"]})
                continue
            window = self.buffer.push(FrameCrop(track_id, timestamp_ms, tensor, self.model_version))
            if window is not None:
                frames = torch.stack([item.crop for item in window]).unsqueeze(0).to(self.device)
                logits, branch = self.model(frames)["temporal_logits"][0, -1], "temporal"
            else:
                logits, branch = self.model.forward_images(tensor.unsqueeze(0).to(self.device))[0], "image_warmup"
            records.append({"frame_index": frame_index, "timestamp_ms": timestamp_ms,
                            "track_id": track_id, "identity_verified": False,
                            "bbox_xyxy": [box.x1, box.y1, box.x2, box.y2],
                            "detector_confidence": box.confidence,
                            "prediction_branch": branch, "temporal_available": branch == "temporal",
                            **self._prediction(logits)})
        return records

    @staticmethod
    def draw_annotations(bgr: Any, records: list[dict[str, Any]]) -> Any:
        import cv2

        annotated = bgr.copy()
        for record in records:
            x1, y1, x2, y2 = (int(value) for value in record["bbox_xyxy"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (101, 207, 184), 2)
            caption = f"{record['track_id']} | {record.get('label') or 'unknown'} | {record['prediction_branch']}"
            if record.get("alert"):
                caption += " | ALERT"
            cv2.putText(annotated, caption, (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (255, 255, 255), 2, cv2.LINE_AA)
        return annotated

    @torch.inference_mode()
    def infer_video(self, video_path: Path, output_video: Path, *, max_frames: int | None = None) -> dict[str, Any]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Install OpenCV support to process local video") from exc
        source = video_path.resolve()
        target = _protect_output(output_video)
        results_path = target.with_suffix(".detections.json")
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists() or results_path.exists():
            raise FileExistsError("Refusing to overwrite an existing video or detections file")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_video = target.with_name(f".{target.stem}.pending{target.suffix}")
        temp_results = results_path.with_name(f".{results_path.name}.pending")
        if temp_video.exists() or temp_results.exists():
            raise FileExistsError("A previous incomplete inference output needs manual review")
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise ValueError("Unable to open input video")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            capture.release()
            raise ValueError("Input video has no usable FPS metadata")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = "VP80" if target.suffix.casefold() == ".webm" else "mp4v"
        writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            writer.release()
            temp_video.unlink(missing_ok=True)
            raise ValueError("Unable to create output video writer")
        detections_json: list[dict[str, Any]] = []
        frame_index, last_timestamp = 0, -1
        timestamp_fallback_used = False
        try:
            while max_frames is None or frame_index < max_frames:
                ok, bgr = capture.read()
                if not ok:
                    break
                frame_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                if frame_ms <= last_timestamp:
                    frame_ms = frame_index * 1000.0 / fps
                    timestamp_fallback_used = True
                timestamp = max(0, int(round(frame_ms)))
                last_timestamp = timestamp
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                records = self.process_frame(image, timestamp, frame_index)
                detections_json.extend(records)
                annotated = self.draw_annotations(bgr, records)
                writer.write(annotated)
                frame_index += 1
        except Exception:
            capture.release()
            writer.release()
            temp_video.unlink(missing_ok=True)
            raise
        finally:
            capture.release()
            writer.release()
        if frame_index == 0:
            temp_video.unlink(missing_ok=True)
            raise ValueError("Input video contains no decodable frames")
        payload = {
            "schema_version": 1,
            "model_version": self.model_version,
            "video_file": source.name,
            "frame_count": frame_index,
            "detection_count": len(detections_json),
            "fps_metadata": fps,
            "timestamp_source": "opencv_container_position_ms" if not timestamp_fallback_used else "opencv_with_frame_index_fallback",
            "timestamp_quality": "fallback_used" if timestamp_fallback_used else "container_position_ms",
            "tracker": self.tracker.name,
            "identity_verified": False,
            "scores_are_calibrated_probabilities": False,
            "detections": detections_json,
        }
        temp_results.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            temp_video.rename(target)
            temp_results.rename(results_path)
        except Exception:
            target.unlink(missing_ok=True)
            temp_video.unlink(missing_ok=True)
            temp_results.unlink(missing_ok=True)
            raise
        return {"video_path": str(target), "detections_path": str(results_path), "frame_count": frame_index,
                "detection_count": len(detections_json), "model_version": self.model_version,
                "timestamp_quality": payload["timestamp_quality"]}

    @staticmethod
    def _prediction(logits: torch.Tensor) -> dict[str, Any]:
        scores = torch.softmax(logits.float(), dim=-1).detach().cpu().tolist()
        class_index = int(max(range(len(scores)), key=scores.__getitem__))
        return {"label": CLASS_KEYS[class_index], "class_scores": scores}
