"""Training-ready datasets that consume only explicitly eligible crop/window manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
from PIL import Image
from torch.utils.data import Dataset

from drowning.data.adapters import _safe_file
from drowning.data.labels import CANONICAL_LABELS
from drowning.ml.augment import augment_crop, sample_rng
from drowning.ml.model import CLASS_KEYS, torch
from drowning.ml.preprocessing import crop_person, image_to_tensor


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path.name}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"JSONL row at {path.name}:{line_number} must be an object")
        records.append(record)
    return records


class CanonicalImageCropDataset(Dataset):
    def __init__(self, crop_manifest: Path, *, split: str, output_size: int = 224,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), augment: bool = False, augment_seed: int = 0):
        self.augment, self.augment_seed, self._epoch = bool(augment), int(augment_seed), 0
        self.root = crop_manifest.parent.resolve()
        self.rows = [row for row in _read_jsonl(crop_manifest)
                     if row.get("split") == split and row.get("training_eligible") is True]
        if not self.rows:
            raise ValueError(f"No training-eligible crop samples in split={split!r}")
        self.output_size, self.mean, self.std = output_size, mean, std
        for row in self.rows:
            if row.get("canonical_label") not in CANONICAL_LABELS or not row.get("crop_sha256"):
                raise ValueError("crop manifest contains an invalid label or missing checksum")
            _safe_file(self.root, row["crop_relative_path"])

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        """Augmentation differs per epoch but is reproducible: it depends only on (seed, epoch, index)."""
        self._epoch = int(epoch)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = _safe_file(self.root, row["crop_relative_path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["crop_sha256"]:
            raise ValueError(f"Crop checksum mismatch: {path.name}")
        with Image.open(path) as image:
            if self.augment:
                image = augment_crop(image, sample_rng(self.augment_seed, self._epoch, index))
            tensor = image_to_tensor(image, size=self.output_size, mean=self.mean, std=self.std)
        return tensor, CLASS_KEYS.index(row["canonical_label"])


class TemporalVideoWindowDataset(Dataset):
    """Decode verified per-person windows by frame index; never joins different tracks."""

    def __init__(self, canonical_jsonl: Path, window_manifest: Path, *, video_root: Path | dict[str, Path],
                 split: str, output_size: int = 224, context_fraction: float = 0.15,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.media = {row["media_id"]: row for row in _read_jsonl(canonical_jsonl) if row.get("entity") == "media"}
        self.observations = {row["observation_id"]: row for row in _read_jsonl(canonical_jsonl)
                             if row.get("entity") == "observation" and row.get("observation_id")}
        self.windows = [row for row in _read_jsonl(window_manifest)
                        if row.get("split") == split and row.get("training_eligible") is True]
        if not self.windows:
            raise ValueError(f"No training-eligible temporal windows in split={split!r}")
        # One root for every video, or a mapping source_id -> root when several datasets are attached.
        self.video_root = ({key: value.resolve() for key, value in video_root.items()}
                           if isinstance(video_root, dict) else video_root.resolve())
        self.output_size, self.context_fraction, self.mean, self.std = output_size, context_fraction, mean, std
        self._captures: dict[Path, Any] = {}
        self._zero = torch.zeros((3, output_size, output_size), dtype=torch.float32)
        for window in self.windows:
            media = self.media.get(window.get("media_id"))
            if not media or media.get("modality") != "video":
                raise ValueError("temporal window references missing/non-video media")
            if media.get("grouping_status") != "verified" or window.get("global_group_id") != media.get("global_group_id"):
                raise ValueError("temporal window global source group does not match verified media")
            if not window.get("identity_verified") or not window.get("causal"):
                raise ValueError("temporal window must have verified identity and be causal")

    def __len__(self) -> int:
        return len(self.windows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_captures"] = {}
        return state

    def _read_frame(self, video_path: Path, frame_index: int) -> Image.Image:
        capture = self._captures.get(video_path)
        if capture is None:
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                capture.release()
                raise ValueError(f"Unable to open video {video_path.name}")
            self._captures[video_path] = capture
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, bgr = capture.read()
        if not ok:
            raise ValueError(f"Unable to decode frame {frame_index} from {video_path.name}")
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def __getitem__(self, index: int):
        window = self.windows[index]
        media = self.media[window["media_id"]]
        root = self.video_root[media["source_id"]] if isinstance(self.video_root, dict) else self.video_root
        video_path = _safe_file(root, media["raw_relative_path"])
        observation_ids = window["observation_ids"]
        valid_mask = list(window["valid_mask"])
        if len(observation_ids) != window["sequence_length"] or len(valid_mask) != len(observation_ids):
            raise ValueError("temporal window length does not match its sequence contract")
        frames, labels = [], []
        expected_person = window["person_id"]
        for observation_id, valid in zip(observation_ids, valid_mask):
            if not valid or observation_id is None:
                frames.append(self._zero)
                labels.append(-100)
                continue
            observation = self.observations.get(observation_id)
            if (not observation or observation.get("media_id") != window["media_id"]
                    or observation.get("person_id") != expected_person
                    or observation.get("identity_verified") is not True
                    or not isinstance(observation.get("frame_index"), int)):
                raise ValueError("window observation does not match its verified video/person/frame contract")
            label = observation.get("canonical_label")
            if observation.get("mapping_action") != "map" or label not in CANONICAL_LABELS or observation.get("annotation_status") != "reviewed":
                raise ValueError("window observation label is not reviewed for supervised training")
            image = self._read_frame(video_path, observation["frame_index"])
            crop = crop_person(image, observation["bbox"], context_fraction=self.context_fraction, output_size=self.output_size)
            frames.append(image_to_tensor(crop, size=self.output_size, mean=self.mean, std=self.std))
            labels.append(CLASS_KEYS.index(label))
        return torch.stack(frames), torch.tensor(labels, dtype=torch.long), torch.tensor(valid_mask, dtype=torch.bool)

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()
