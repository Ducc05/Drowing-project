"""Multi-video CVAT sources and the reviewed_jsonl contract, converted from synthetic videos."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from drowning.data.adapters import convert_to_jsonl, scan_source
from drowning.data.audit import audit_jsonl
from drowning.data.quality import build_quality_report
from drowning.data.registry import SourceConfigError
from drowning.data.review import apply_reviews, record_grouping_review
from drowning.data.windows import build_temporal_windows
from drowning.orchestration.kaggle_run import discover_source_root
from tests.test_cvat_adapter import make_video

MAPPING = {"Drowning": {"action": "map", "target": "drowning"}, "Swimming": {"action": "map", "target": "swimming"},
           "out of water": {"action": "map", "target": "out_of_water"}}


def boxes(label, frames=range(5), x=2):
    return "".join(f'<box frame="{f}" xtl="{x}" ytl="2" xbr="{x + 18}" ybr="28" outside="0" occluded="0" keyframe="1">'
                   f'<attribute name="state">{label}</attribute></box>' for f in frames)


def base_source(**extra):
    return {"schema_version": 1, "source_id": "videos-fixture", "source_version": "1", "modality": "video",
            "paths": {"root_env": "X"}, "mapping": MAPPING, "mapping_review": {"status": "approved", "sample_review_required": False},
            "identity_review": {"status": "verified", "sample_review_required": False},
            "timing": {"mode": "explicit_timestamps"}, **extra}


class VideoSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "raw"
        self.root.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def cvat_root(self):
        for name, label in (("a", "Drowning"), ("b", "Swimming")):
            make_video(self.root / "tasks" / name / "video.avi")
            (self.root / "tasks" / name / "ts.json").write_text(json.dumps([0, 100, 200, 300, 400]), encoding="utf-8")
            (self.root / "tasks" / name / "annotations.xml").write_text(
                f'<annotations><track id="1" label="person">{boxes(label)}</track></annotations>', encoding="utf-8")

    def cvat_source(self, declare_groups=True, **extra):
        tasks = [{"annotations": f"tasks/{n}/annotations.xml", "video": f"tasks/{n}/video.avi", "timestamps": f"tasks/{n}/ts.json",
                  **({"global_group_id": f"session-{n}"} if declare_groups else {})} for n in ("a", "b")]
        return base_source(format="cvat_video", tasks=tasks, **extra)

    def records(self, source, name="out.jsonl"):
        output = self.base / name
        convert_to_jsonl(source, self.root, output)
        return [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()], output

    def test_cvat_source_with_several_tasks_yields_one_media_and_track_set_per_video(self):
        self.cvat_root()
        source = self.cvat_source()
        scan = scan_source(source, self.root)
        self.assertEqual((scan["media_count"], scan["annotation_count"]), (2, 10))
        self.assertEqual(scan["annotation_counts_by_source_label"], {"Drowning": 5, "Swimming": 5})
        records, output = self.records(source)
        media = [r for r in records if r["entity"] == "media"]
        self.assertEqual({m["global_group_id"] for m in media}, {"session-a", "session-b"})
        self.assertTrue(all(m["grouping_status"] == "verified" for m in media))
        persons = {r["person_id"] for r in records if r["entity"] == "observation"}
        self.assertEqual(len(persons), 2)  # same CVAT track id in two videos stays two people
        self.assertTrue(audit_jsonl([output])["valid"])
        windows = build_temporal_windows(output, sequence_length=2, sample_period_ms=100, max_track_gap_ms=300, min_valid_ratio=1.0)
        self.assertEqual(sorted({w["label"] for w in windows["windows"]}), ["drowning", "swimming"])
        self.assertEqual(windows["window_count"], 8)

    def test_single_pair_descriptor_still_works_and_bad_task_lists_are_rejected(self):
        self.cvat_root()
        single = base_source(format="cvat_video", paths={"root_env": "X", "annotations": "tasks/a/annotations.xml",
                                                          "video": "tasks/a/video.avi", "timestamps": "tasks/a/ts.json"})
        self.assertEqual(scan_source(single, self.root)["media_count"], 1)
        for bad in ([], [{"annotations": "x"}], "nope"):
            with self.assertRaises(SourceConfigError):
                scan_source(base_source(format="cvat_video", tasks=bad), self.root)

    def test_reviewed_groups_apply_to_videos_by_path_and_quality_proposes_one_group_per_video(self):
        self.cvat_root()
        source = self.cvat_source(declare_groups=False)
        records, output = self.records(source)
        self.assertTrue(all(m["grouping_status"] == "unknown" for m in records if m["entity"] == "media"))
        report, proposed = build_quality_report(source, output, self.root)
        self.assertEqual(report["video_groups"]["proposed_group_count"], 2)
        self.assertEqual(set(proposed), {"tasks/a/video.avi", "tasks/b/video.avi"})
        self.assertEqual(len(set(proposed.values())), 2)
        reviews = self.base / "reviews"
        record_grouping_review(source, reviewer="Dung", by_path=proposed, method="one_group_per_video_file", evidence={},
                               note="Videos come from different recording days and cameras.", reviews_dir=reviews)
        grouped, _ = self.records(apply_reviews(source, reviews), "grouped.jsonl")
        self.assertTrue(all(m["grouping_status"] == "verified" for m in grouped if m["entity"] == "media"))

    def test_discovery_uses_the_first_task_as_marker(self):
        self.cvat_root()
        found = discover_source_root(self.cvat_source(), self.base)
        self.assertEqual((found["status"], found["candidates"]), ("found", ["raw"]))

    # -- reviewed_jsonl ------------------------------------------------------------------
    def jsonl_root(self, annotations=None):
        make_video(self.root / "v" / "pool.avi")
        Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(self.root / "still.png")
        media = [{"path": "v/pool.avi", "modality": "video", "group_id": "pool-1"},
                 {"path": "still.png", "modality": "image"}]
        rows = annotations if annotations is not None else (
            [{"path": "v/pool.avi", "frame": f, "timestamp_ms": f * 100, "person_id": "p1", "label": "Swimming",
              "bbox": [2, 2, 20, 28]} for f in range(3)]
            + [{"path": "v/pool.avi", "frame": 3, "timestamp_ms": 300, "person_id": "p1", "label": "Swimming", "outside": True},
               {"path": "v/pool.avi", "frame": 4, "timestamp_ms": 400, "person_id": "p1", "label": "Drowning", "bbox": [3, 2, 21, 28]},
               {"path": "still.png", "label": "out of water", "bbox": [5, 5, 30, 35]}])
        (self.root / "media.jsonl").write_text("\n".join(json.dumps(m) for m in media), encoding="utf-8")
        (self.root / "ann.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    def jsonl_source(self):
        return base_source(format="reviewed_jsonl", modality="mixed", paths={"root_env": "X", "media": "media.jsonl", "annotations": "ann.jsonl"})

    def test_reviewed_jsonl_converts_video_and_image_rows_with_explicit_time_and_identity(self):
        self.jsonl_root()
        source = self.jsonl_source()
        scan = scan_source(source, self.root)
        self.assertEqual((scan["media_count"], scan["annotation_count"]), (2, 5))  # outside markers are not boxes
        records, output = self.records(source)
        self.assertTrue(audit_jsonl([output])["valid"])
        media = {m["raw_relative_path"]: m for m in records if m["entity"] == "media"}
        self.assertEqual((media["v/pool.avi"]["modality"], media["v/pool.avi"]["global_group_id"]), ("video", "pool-1"))
        self.assertEqual(media["still.png"]["modality"], "image")
        observations = [r for r in records if r["entity"] == "observation"]
        outside = next(o for o in observations if o["outside"])
        self.assertEqual((outside["bbox"], outside["annotation_status"], outside["mapping_action"]), (None, "outside_marker", "review"))
        image_obs = next(o for o in observations if o["original_label"] == "out of water")
        self.assertIsNone(image_obs["timestamp_ms"])
        self.assertFalse(image_obs["identity_verified"])
        video_obs = [o for o in observations if o["media_id"] == media["v/pool.avi"]["media_id"] and not o["outside"]]
        self.assertTrue(all(o["identity_verified"] and o["timestamp_ms"] == o["frame_index"] * 100 for o in video_obs))
        windows = build_temporal_windows(output, sequence_length=2, sample_period_ms=100, max_track_gap_ms=300, min_valid_ratio=1.0)
        self.assertEqual(windows["window_count"], 2)  # the outside marker breaks the track before frame 4

    def test_reviewed_jsonl_refuses_to_guess_missing_time_identity_or_geometry(self):
        good = {"path": "v/pool.avi", "frame": 0, "timestamp_ms": 0, "person_id": "p", "label": "Swimming", "bbox": [2, 2, 20, 28]}
        cases = {
            "timestamp": {k: v for k, v in good.items() if k != "timestamp_ms"},
            "person": {k: v for k, v in good.items() if k != "person_id"},
            "frame range": {**good, "frame": 99},
            "bbox outside": {**good, "bbox": [2, 2, 500, 28]},
            "bbox order": {**good, "bbox": [20, 2, 2, 28]},
            "no label": {**good, "label": ""},
            "unknown path": {**good, "path": "nope.avi"},
        }
        for name, row in cases.items():
            self.jsonl_root([row])
            with self.assertRaises(SourceConfigError, msg=name):
                list(convert_to_jsonl(self.jsonl_source(), self.root, self.base / f"{name.replace(' ', '_')}.jsonl").items())
        with self.assertRaises(SourceConfigError):
            scan_source({**self.jsonl_source(), "paths": {"root_env": "X"}}, self.root)

    def test_unlabelled_identity_is_not_verified_without_the_descriptor_review(self):
        self.jsonl_root()
        source = self.jsonl_source()
        source["identity_review"] = {"status": "pending", "sample_review_required": True}
        records, output = self.records(source)
        self.assertFalse(any(r.get("identity_verified") for r in records if r["entity"] == "observation"))
        windows = build_temporal_windows(output, sequence_length=2, sample_period_ms=100, max_track_gap_ms=300, min_valid_ratio=1.0)
        self.assertEqual(windows["window_count"], 0)
        self.assertIn("no_windows_meet_verified_track_sampling_contract", windows["blockers"])


if __name__ == "__main__":
    unittest.main()
