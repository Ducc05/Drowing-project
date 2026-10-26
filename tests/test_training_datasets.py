import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from drowning.ml.datasets import CanonicalImageCropDataset, TemporalVideoWindowDataset


class TrainingDatasetTests(unittest.TestCase):
    def test_image_dataset_requires_eligible_manifest_and_validates_crop_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crop = root / "train" / "drowning" / "crop.jpg"
            crop.parent.mkdir(parents=True)
            Image.new("RGB", (32, 32), "blue").save(crop)
            digest = hashlib.sha256(crop.read_bytes()).hexdigest()
            manifest = root / "crop_manifest.jsonl"
            manifest.write_text(json.dumps({"split": "train", "training_eligible": True,
                                            "crop_relative_path": "train/drowning/crop.jpg",
                                            "crop_sha256": digest, "canonical_label": "drowning"}) + "\n",
                                encoding="utf-8")
            dataset = CanonicalImageCropDataset(manifest, split="train", output_size=32)
            image, label = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 32, 32))
            self.assertEqual(label, 0)
            with self.assertRaisesRegex(ValueError, "No training-eligible"):
                CanonicalImageCropDataset(manifest, split="validation", output_size=32)
            crop.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                dataset[0]

    def test_temporal_dataset_decodes_only_verified_person_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_path = root / "video.avi"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (40, 40))
            self.assertTrue(writer.isOpened())
            for index in range(3):
                frame = np.zeros((40, 40, 3), dtype=np.uint8)
                frame[:, :, index] = 180
                writer.write(frame)
            writer.release()
            media = {"entity": "media", "media_id": "video-1", "source_id": "video-source",
                     "source_version": "v1", "modality": "video", "raw_relative_path": "video.avi",
                     "grouping_status": "verified", "global_group_id": "capture-1"}
            observations = []
            for index in range(2):
                observations.append({"entity": "observation", "observation_id": f"obs-{index}",
                                     "media_id": "video-1", "frame_index": index,
                                     "person_id": "person-1", "identity_verified": True,
                                     "bbox": [2, 2, 30, 35], "mapping_action": "map",
                                     "canonical_label": "drowning" if index == 0 else "swimming",
                                     "annotation_status": "reviewed"})
            canonical = root / "canonical.jsonl"
            canonical.write_text("\n".join(json.dumps(row) for row in [media, *observations]) + "\n", encoding="utf-8")
            window = {"window_id": "window-1", "media_id": "video-1", "global_group_id": "capture-1",
                      "grouping_status": "verified", "person_id": "person-1", "identity_verified": True,
                      "causal": True, "sequence_length": 2, "split": "train", "training_eligible": True,
                      "observation_ids": ["obs-0", "obs-1"], "valid_mask": [True, True]}
            window_file = root / "windows.jsonl"
            window_file.write_text(json.dumps(window) + "\n", encoding="utf-8")
            dataset = TemporalVideoWindowDataset(canonical, window_file, video_root=root, split="train", output_size=32)
            frames, labels, valid = dataset[0]
            self.assertEqual(tuple(frames.shape), (2, 3, 32, 32))
            self.assertEqual(labels.tolist(), [0, 1])
            self.assertEqual(valid.tolist(), [True, True])
            dataset.close()

    def test_temporal_dataset_rejects_unverified_group_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = {"entity": "media", "media_id": "video-1", "modality": "video",
                     "raw_relative_path": "video.avi", "grouping_status": "unknown"}
            obs = {"entity": "observation", "observation_id": "obs-1", "media_id": "video-1",
                   "person_id": "person-1", "identity_verified": True}
            canonical = root / "canonical.jsonl"
            canonical.write_text(json.dumps(media) + "\n" + json.dumps(obs), encoding="utf-8")
            window_file = root / "windows.jsonl"
            window_file.write_text(json.dumps({"window_id": "w", "media_id": "video-1", "global_group_id": None,
                                               "grouping_status": "unknown", "person_id": "person-1",
                                               "identity_verified": True, "causal": True, "training_eligible": True,
                                               "split": "train", "observation_ids": [], "valid_mask": []}) + "\n",
                                    encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "global source group"):
                TemporalVideoWindowDataset(canonical, window_file, video_root=root, split="train")


if __name__ == "__main__":
    unittest.main()
