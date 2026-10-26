import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from drowning.data.adapters import convert_to_jsonl, scan_source
from drowning.data.audit import audit_jsonl
from drowning.data.registry import SourceConfigError
from drowning.data.windows import build_temporal_windows


def make_video(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    if not writer.isOpened():
        raise RuntimeError("test video codec unavailable")
    for frame in range(5):
        writer.write(np.full((32, 32, 3), frame * 20, dtype=np.uint8))
    writer.release()


def cvat_source():
    return {
        "schema_version": 1, "source_id": "cvat-video-fixture", "source_version": "v1",
        "format": "cvat_video", "modality": "video",
        "paths": {"root_env": "CVAT_ROOT", "annotations": "annotations.xml", "video": "video.avi", "timestamps": "timestamps.json"},
        "mapping": {"Drowning": {"action": "map", "target": "drowning"}},
        "mapping_review": {"status": "approved", "sample_review_required": False},
        "identity_review": {"status": "verified", "sample_review_required": False},
        "timing": {"mode": "explicit_timestamps"},
        "grouping": {"status": "verified", "global_group_id": "capture-1"},
    }


class CVATAdapterTests(unittest.TestCase):
    def test_cvat_tracks_keep_time_identity_outside_and_create_causal_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            root.mkdir()
            make_video(root / "video.avi")
            (root / "timestamps.json").write_text(json.dumps([0, 100, 200, 300, 400]), encoding="utf-8")
            xml = '''<annotations><track id="7" label="person">
            <box frame="0" xtl="2" ytl="2" xbr="20" ybr="28" outside="0" occluded="0" keyframe="1"><attribute name="state">Drowning</attribute></box>
            <box frame="1" xtl="3" ytl="2" xbr="21" ybr="28" outside="0" occluded="1" keyframe="1"><attribute name="state">Drowning</attribute></box>
            <box frame="2" xtl="4" ytl="2" xbr="22" ybr="28" outside="0" occluded="0" keyframe="1"><attribute name="state">Drowning</attribute></box>
            <box frame="3" xtl="4" ytl="2" xbr="22" ybr="28" outside="1" occluded="0" keyframe="1" />
            <box frame="4" xtl="5" ytl="2" xbr="23" ybr="28" outside="0" occluded="0" keyframe="1"><attribute name="state">Drowning</attribute></box>
            </track></annotations>'''
            (root / "annotations.xml").write_text(xml, encoding="utf-8")
            source = cvat_source()
            canonical = base / "canonical.jsonl"
            summary = convert_to_jsonl(source, root, canonical)
            self.assertEqual(summary["media_count"], 1)
            self.assertEqual(summary["observation_count"], 5)
            report = audit_jsonl([canonical])
            self.assertTrue(report["valid"], report["issues"])
            records = [json.loads(line) for line in canonical.read_text(encoding="utf-8").splitlines()]
            track_rows = [row for row in records if row["entity"] == "observation"]
            self.assertEqual(track_rows[1]["timestamp_ms"], 100)
            self.assertTrue(track_rows[1]["identity_verified"])
            self.assertTrue(track_rows[1]["occluded"])
            outside = next(row for row in track_rows if row["outside"])
            self.assertIsNone(outside["bbox"])
            self.assertEqual(outside["annotation_status"], "outside_marker")
            windows = build_temporal_windows(canonical, sequence_length=2, sample_period_ms=100,
                                             max_track_gap_ms=300, min_valid_ratio=1.0,
                                             split_assignments={track_rows[0]["media_id"]: {
                                                 "split": "train", "global_group_id": "capture-1"}})
            self.assertEqual(windows["window_count"], 2)
            self.assertNotIn(track_rows[4]["observation_id"], windows["windows"][-1]["observation_ids"])
            self.assertTrue(all(window["training_eligible"] for window in windows["windows"]))

    def test_cvat_scan_counts_state_attributes_and_refuses_unverified_cfr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            root.mkdir()
            make_video(root / "video.avi")
            xml = '''<annotations><track id="1" label="person"><box frame="0" xtl="1" ytl="1" xbr="20" ybr="30" outside="0"><attribute name="state">Drowning</attribute></box></track></annotations>'''
            (root / "annotations.xml").write_text(xml, encoding="utf-8")
            (root / "timestamps.json").write_text(json.dumps([0, 100, 200, 300, 400]), encoding="utf-8")
            source = cvat_source()
            report = scan_source(source, root)
            self.assertEqual(report["media_count"], 1)
            self.assertEqual(report["annotation_count"], 1)
            self.assertEqual(report["annotation_counts_by_mapping_action"], {"map": 1})
            source["timing"] = {"mode": "constant_frame_rate", "cfr_verified": False}
            source["paths"].pop("timestamps")
            with self.assertRaisesRegex(SourceConfigError, "verified constant-frame-rate"):
                convert_to_jsonl(source, root, Path(directory) / "blocked.jsonl")


if __name__ == "__main__":
    unittest.main()
