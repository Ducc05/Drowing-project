import json
import tempfile
import unittest
from pathlib import Path

from drowning.data.windows import build_temporal_windows


def write_video_records(path: Path, timestamps=(0, 100, 200, 300, 400), *, verified=True, group=True):
    media = {
        "entity": "media", "schema_version": 1, "media_id": "video-1", "source_id": "video-fixture",
        "source_version": "v1", "modality": "video", "global_group_id": "capture-1" if group else None,
        "grouping_status": "verified" if group else "unknown", "original_export_split": "train",
    }
    records = [media]
    for index, timestamp in enumerate(timestamps):
        records.append({
            "entity": "observation", "schema_version": 1, "observation_id": f"obs-{index}",
            "media_id": "video-1", "source_id": "video-fixture", "frame_index": index,
            "timestamp_ms": timestamp, "person_id": "person-1", "identity_verified": verified,
            "bbox_format": "xyxy_pixels", "bbox": [1, 2, 20, 30], "mapping_action": "map",
            "canonical_label": "drowning" if index < 3 else "swimming", "annotation_status": "reviewed",
        })
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


class TemporalWindowTests(unittest.TestCase):
    def test_builds_fixed_length_causal_windows_for_one_verified_track(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.jsonl"
            write_video_records(path)
            report = build_temporal_windows(path, sequence_length=3, sample_period_ms=100,
                                            max_track_gap_ms=300, min_valid_ratio=1.0,
                                            split_assignments={"video-1": {"split": "train", "global_group_id": "capture-1"}})
            self.assertEqual(report["status"], "ready_for_review")
            self.assertEqual(report["window_count"], 3)
            window = report["windows"][0]
            self.assertEqual(window["observation_ids"], ["obs-0", "obs-1", "obs-2"])
            self.assertEqual(window["label"], "drowning")
            self.assertTrue(window["causal"])
            self.assertTrue(window["training_eligible"])
            self.assertEqual(window["split"], "train")
            self.assertTrue(report["source_video_disjointness_verified"])

    def test_unknown_video_group_blocks_training_even_when_window_is_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.jsonl"
            write_video_records(path, timestamps=(0, 100, 200), group=False)
            report = build_temporal_windows(path, sequence_length=3, sample_period_ms=100,
                                            max_track_gap_ms=300, min_valid_ratio=1.0,
                                            split_assignments={"video-1": {"split": "train", "global_group_id": ""}})
            self.assertEqual(report["window_count"], 1)
            self.assertFalse(report["windows"][0]["training_eligible"])
            self.assertIn("source_video_group_identity_unverified", report["blockers"])

    def test_unverified_identity_and_image_only_source_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.jsonl"
            write_video_records(path, timestamps=(0, 100, 200), verified=False)
            report = build_temporal_windows(path, sequence_length=3, sample_period_ms=100,
                                            max_track_gap_ms=300, min_valid_ratio=1.0,
                                            split_assignments={"video-1": {"split": "train", "global_group_id": "capture-1"}})
            self.assertEqual(report["status"], "blocked")
            self.assertIn("unverified_person_identity", {issue["code"] for issue in report["issues"]})
            path.write_text(json.dumps({"entity": "media", "media_id": "image-1", "modality": "image"}), encoding="utf-8")
            image_report = build_temporal_windows(path, sequence_length=3, sample_period_ms=100,
                                                  max_track_gap_ms=300, min_valid_ratio=1.0)
            self.assertIn("no_temporal_video_annotations", image_report["blockers"])


if __name__ == "__main__":
    unittest.main()
