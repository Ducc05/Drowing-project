import json
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from drowning.data.adapters import convert_to_jsonl
from drowning.data.registry import SourceConfigError, read_class_names, validate_source
from drowning.orchestration.kaggle_run import discover_source_root

ROOT = Path(__file__).resolve().parents[1]
REGEX = r"^train/images/(.+?)-\d+_jpg\.rf\.[0-9a-f]+\.[A-Za-z0-9]+$"


def descriptor(**overrides):
    source = {"schema_version": 1, "source_id": "ds_no_yaml", "source_version": "1", "format": "yolo_detection", "modality": "image",
              "paths": {"root_env": "X", "marker": "train/labels", "input_dir_contains": "dataset-test-1"},
              "source_labels": {"class_names": {0: "Drowning", 1: "Person out of water", 2: "Swimming"}},
              "mapping": {"Drowning": {"action": "map", "target": "drowning"}, "Swimming": {"action": "map", "target": "swimming"},
                          "Person out of water": {"action": "map", "target": "out_of_water"}},
              "mapping_review": {"status": "pending_sample_review", "sample_review_required": True},
              "grouping": {"path_regex": REGEX}}
    source.update(overrides)
    return source


def make_dataset(root: Path, files: dict[str, str]):
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    for stem, label in files.items():
        Image.new("RGB", (40, 40), (len(stem) * 5 % 255, 80, 120)).save(root / "train" / "images" / f"{stem}.jpg")
        (root / "train" / "labels" / f"{stem}.txt").write_text(label, encoding="utf-8")


class DatasetWithoutYamlTests(unittest.TestCase):
    def test_class_names_come_from_the_descriptor_when_there_is_no_class_list(self):
        self.assertEqual(read_class_names(descriptor(), Path(".")), [(0, "Drowning"), (1, "Person out of water"), (2, "Swimming")])
        self.assertEqual(read_class_names(descriptor(source_labels={"class_names": ["a", "b"]}), Path(".")), [(0, "a"), (1, "b")])
        for bad in ({"class_names": []}, {"class_names": {0: " "}}, {"class_names": "text"}):
            with self.assertRaises(SourceConfigError):
                read_class_names(descriptor(source_labels=bad), Path("."))
        self.assertEqual(validate_source(descriptor()), [])

    def test_a_declared_class_list_file_still_wins_over_nothing_and_is_required_without_inline_names(self):
        with self.assertRaises(SourceConfigError):
            read_class_names(descriptor(source_labels={}), Path("."))

    def test_every_copy_and_frame_of_one_source_video_gets_one_group_and_the_records_say_it_is_unconfirmed(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as out_dir:
            root = Path(temp)
            make_dataset(root, {
                "drowning1_mp4-3_jpg.rf.aaaa1111": "2 0.5 0.5 0.4 0.4\n",
                "drowning1_mp4-3_jpg.rf.bbbb2222": "2 0.5 0.5 0.4 0.4\n",      # augmented copy of the same frame
                "drowning1_mp4-7_jpg.rf.cccc3333": "0 0.5 0.5 0.4 0.4\n",      # another frame of the same video
                "12_mp4-1_jpg.rf.dddd4444": "1 0.5 0.5 0.4 0.4\n2 0.3 0.3 0.2 0.2\n",
                "odd name": "2 0.5 0.5 0.4 0.4\n"})
            output = Path(out_dir) / "out.jsonl"
            convert_to_jsonl(descriptor(), root, output)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            media = {Path(r["raw_relative_path"]).stem: r for r in rows if r["entity"] == "media"}
            same = {media[k]["global_group_id"] for k in media if k.startswith("drowning1_mp4")}
            self.assertEqual(same, {"ds_no_yaml:drowning1_mp4"})
            self.assertEqual(media["12_mp4-1_jpg.rf.dddd4444"]["global_group_id"], "ds_no_yaml:12_mp4")
            self.assertTrue(all(m["grouping_assurance"] == "filename_pattern_unconfirmed" for k, m in media.items() if k != "odd name"))
            self.assertIsNone(media["odd name"]["global_group_id"])           # no match: left unknown, not guessed
            self.assertEqual(media["odd name"]["grouping_status"], "unknown")
            observations = [r for r in rows if r["entity"] == "observation"]
            self.assertEqual({o["original_label"] for o in observations}, {"Drowning", "Swimming", "Person out of water"})
            self.assertEqual({o["canonical_label"] for o in observations}, {"drowning", "swimming", "out_of_water"})

    def test_bad_group_patterns_are_refused(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as out_dir:
            root = Path(temp)
            make_dataset(root, {"a-1_jpg.rf.ab12": "2 0.5 0.5 0.4 0.4\n"})
            for pattern in ("([", r"^train/images/.+$"):
                with self.assertRaises(SourceConfigError):
                    convert_to_jsonl(descriptor(grouping={"path_regex": pattern}), root, Path(out_dir) / "x.jsonl")


class DiscoveryTests(unittest.TestCase):
    def test_the_attached_datasets_own_name_separates_two_datasets_with_the_same_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            inputs = Path(temp)
            for name in ("alanoud-pool", "dataset-test-1"):
                (inputs / "datasets" / "vuduc792005" / name / "train" / "labels").mkdir(parents=True)
            found = discover_source_root(descriptor(), inputs)
            self.assertEqual(found["status"], "found")
            self.assertEqual(found["root"], inputs / "datasets" / "vuduc792005" / "dataset-test-1")
            without_slug = descriptor(paths={"root_env": "X", "marker": "train/labels"})
            self.assertEqual(discover_source_root(without_slug, inputs)["status"], "ambiguous")  # never picks one at random
            missing = descriptor(paths={"root_env": "X", "marker": "train/labels", "input_dir_contains": "nope"})
            self.assertEqual(discover_source_root(missing, inputs)["status"], "missing")


class ShippedDescriptorTests(unittest.TestCase):
    def test_the_real_descriptor_is_valid_and_maps_all_three_declared_names(self):
        source = yaml.safe_load((ROOT / "configs" / "datasets" / "dataset_test_1.yaml").read_text(encoding="utf-8"))
        self.assertEqual(validate_source(source), [])
        names = [name for _, name in read_class_names(source, Path("."))]
        self.assertEqual(sorted(names), sorted(source["mapping"]))
        self.assertEqual(source["mapping_review"]["status"], "pending_sample_review")  # never pre-approved


if __name__ == "__main__":
    unittest.main()
