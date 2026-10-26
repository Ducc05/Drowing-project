import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from PIL import Image
import yaml

from drowning.data.adapters import convert_to_jsonl, scan_source
from drowning.data.crops import build_crops
from drowning.cli import main
from drowning.data.registry import SourceConfigError


def write_image(path: Path, size=(100, 80)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(20, 40, 60)).save(path)


class AdapterTests(unittest.TestCase):
    def test_crop_preview_is_deterministic_and_never_training_eligible_when_mapping_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            root.mkdir()
            (root / "data.yaml").write_text("names: [Drowning]\n", encoding="utf-8")
            image_path = root / "images" / "one.png"
            write_image(image_path)
            (root / "labels").mkdir()
            (root / "labels" / "one.txt").write_text("0 0.5 0.5 0.4 0.5\n", encoding="utf-8")
            raw_before = image_path.read_bytes()
            source = {
                "schema_version": 1, "source_id": "crop-fixture", "source_version": "v1",
                "format": "yolo_detection", "modality": "image",
                "paths": {"root_env": "FIXTURE_ROOT", "class_list": "data.yaml:names"},
                "mapping": {"Drowning": {"action": "map", "target": "drowning"}},
                "mapping_review": {"status": "pending_sample_review", "sample_review_required": True},
            }
            canonical = base / "canonical.jsonl"
            convert_to_jsonl(source, root, canonical)
            with self.assertRaisesRegex(SourceConfigError, "not approved"):
                build_crops(source, canonical, root, base / "blocked-crops")
            summary = build_crops(source, canonical, root, base / "preview-crops", preview_only=True, limit=1)
            crop_meta = json.loads((base / "preview-crops" / "crop_manifest.jsonl").read_text(encoding="utf-8"))
            with Image.open(base / "preview-crops" / crop_meta["crop_relative_path"]) as crop:
                self.assertEqual(crop.size, (224, 224))
            self.assertEqual(summary["crop_count"], 1)
            self.assertEqual(summary["training_eligible_count"], 0)
            self.assertTrue(crop_meta["review_required"])
            self.assertFalse(crop_meta["training_eligible"])
            self.assertEqual(image_path.read_bytes(), raw_before)

    def test_training_crop_requires_reviewed_mapping_and_verified_group_split(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            root.mkdir()
            (root / "data.yaml").write_text("names: [Drowning]\n", encoding="utf-8")
            image_path = root / "images" / "one.png"
            write_image(image_path)
            (root / "labels").mkdir()
            (root / "labels" / "one.txt").write_text("0 0.5 0.5 0.4 0.5\n", encoding="utf-8")
            (root / "groups.json").write_text(json.dumps({"by_path": {
                "images/one.png": {"global_group_id": "pool-session-1", "status": "verified"}
            }}), encoding="utf-8")
            source = {
                "schema_version": 1, "source_id": "crop-reviewed-fixture", "source_version": "v1",
                "format": "yolo_detection", "modality": "image",
                "paths": {"root_env": "FIXTURE_ROOT", "class_list": "data.yaml:names"},
                "mapping": {"Drowning": {"action": "map", "target": "drowning"}},
                "mapping_review": {"status": "approved", "sample_review_required": False},
                "grouping": {"status": "verified", "metadata_file": "groups.json"},
            }
            canonical = base / "canonical.jsonl"
            convert_to_jsonl(source, root, canonical)
            media = next(json.loads(line) for line in canonical.read_text(encoding="utf-8").splitlines()
                         if json.loads(line)["entity"] == "media")
            split_manifest = base / "split.jsonl"
            split_manifest.write_text(json.dumps({"media_id": media["media_id"],
                                                 "global_group_id": "pool-session-1", "split": "train"}) + "\n",
                                      encoding="utf-8")
            with self.assertRaisesRegex(SourceConfigError, "split-manifest"):
                build_crops(source, canonical, root, base / "missing-split-crops")
            result = build_crops(source, canonical, root, base / "train-crops", split_manifest=split_manifest)
            crop_meta = json.loads((base / "train-crops" / "crop_manifest.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(result["training_eligible_count"], 1)
            self.assertEqual(crop_meta["split"], "train")
            self.assertFalse(crop_meta["review_required"])

    def test_yolo_scan_counts_by_split_and_mapping_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            root.mkdir()
            (root / "data.yaml").write_text("names: [Drowning, normal]\n", encoding="utf-8")
            write_image(root / "train" / "images" / "one.PNG")
            labels = root / "train" / "labels"
            labels.mkdir(parents=True)
            label_file = labels / "one.txt"
            label_file.write_text("0 0.5 0.5 0.4 0.4\n1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            before = label_file.read_bytes()
            source = {
                "schema_version": 1, "source_id": "scan-fixture", "source_version": "v1",
                "format": "yolo_detection", "modality": "image",
                "paths": {"root_env": "FIXTURE_ROOT", "class_list": "data.yaml:names"},
                "mapping": {"Drowning": {"action": "map", "target": "drowning"}},
            }
            report = scan_source(source, root)
            self.assertEqual(report["media_count"], 1)
            self.assertEqual(report["annotation_count"], 2)
            self.assertEqual(report["media_counts_by_original_split"], {"train": 1})
            self.assertEqual(report["annotation_counts_by_mapping_action"], {"map": 1, "review": 1})
            self.assertFalse(report["source_data_modified"])
            self.assertEqual(label_file.read_bytes(), before)

    def test_yolo_scan_reports_invalid_numeric_bboxes_and_rejects_ambiguous_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.yaml").write_text("names: [Drowning]\n", encoding="utf-8")
            write_image(root / "images" / "one.PNG")
            (root / "labels").mkdir()
            label = root / "labels/one.txt"
            label.write_text("0 nan 0.5 0.2 0.2\n0 0.5 inf 0.2 0.2\n0 0.1 0.5 0.4 0.4\n0 0.5 0.5 -0.2 0.2\n0 bad 0.5 0.2 0.2\n0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
            source = {"schema_version": 1, "source_id": "scan-invalid", "source_version": "v1",
                      "format": "yolo_detection", "modality": "image",
                      "paths": {"root_env": "FIXTURE_ROOT", "class_list": "data.yaml:names"},
                      "mapping": {"Drowning": {"action": "map", "target": "drowning"}}}
            before = label.read_bytes()
            report = scan_source(source, root)
            self.assertEqual(report["malformed_annotation_count"], 5)
            self.assertEqual(report["annotation_count"], 6)
            self.assertEqual(label.read_bytes(), before)
            write_image(root / "images" / "one.jpg")
            with self.assertRaisesRegex(SourceConfigError, "found 2"):
                scan_source(source, root)

    def test_yolo_conversion_keeps_source_label_and_unknown_as_review(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            root.mkdir()
            (root / "data.yaml").write_text("names:\n  0: Drowning\n  3: Not drowning\n", encoding="utf-8")
            write_image(root / "train" / "images" / "swimmer.png")
            labels = root / "train" / "labels" / "swimmer.txt"
            labels.parent.mkdir(parents=True)
            labels.write_text("0 0.5 0.5 0.4 0.5\n3 0.2 0.25 0.2 0.25\n", encoding="utf-8")
            source = {
                "schema_version": 1, "source_id": "fixture-yolo", "source_version": "v1",
                "format": "yolo_detection", "modality": "image",
                "paths": {"root_env": "FIXTURE_ROOT", "class_list": "data.yaml:names"},
                "mapping": {"Drowning": {"action": "map", "target": "drowning"}},
            }
            output = base / "derived" / "records.jsonl"
            summary = convert_to_jsonl(source, root, output)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            media = next(record for record in records if record["entity"] == "media")
            observations = [record for record in records if record["entity"] == "observation"]
            self.assertEqual(summary["media_count"], 1)
            self.assertEqual(summary["observation_count"], 2)
            self.assertEqual(media["raw_relative_path"], "train/images/swimmer.png")
            self.assertEqual(observations[0]["bbox"], [30.0, 20.0, 70.0, 60.0])
            self.assertEqual(observations[0]["canonical_label"], "drowning")
            self.assertEqual(observations[1]["mapping_action"], "review")
            self.assertIsNone(observations[1]["canonical_label"])
            self.assertEqual(observations[1]["original_label"], "Not drowning")
            self.assertFalse(observations[0]["identity_verified"])
            self.assertEqual(observations[0]["grouping_status"], "unknown")

    def test_pascal_voc_conversion_applies_one_based_inclusive_coordinates_and_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            root.mkdir()
            write_image(root / "JPEGImages" / "one.png")
            annotations = root / "Annotations"
            annotations.mkdir()
            xml = '''<annotation><filename>one.png</filename><size><width>100</width><height>80</height></size>
            <object><name>Drowning</name><truncated>1</truncated><difficult>0</difficult>
            <bndbox><xmin>11</xmin><ymin>21</ymin><xmax>40</xmax><ymax>60</ymax></bndbox></object></annotation>'''
            (annotations / "one.xml").write_text(xml, encoding="utf-8")
            source = {
                "schema_version": 1, "source_id": "voc-fixture", "source_version": "v1",
                "format": "pascal_voc", "modality": "image",
                "paths": {"root_env": "FIXTURE_ROOT", "annotations": "Annotations", "images": "JPEGImages"},
                "mapping": {"Drowning": {"action": "map", "target": "drowning"}},
            }
            output = base / "voc.jsonl"
            summary = convert_to_jsonl(source, root, output)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            observation = next(record for record in records if record["entity"] == "observation")
            self.assertEqual(summary["media_count"], 1)
            self.assertEqual(observation["bbox"], [10.0, 20.0, 40.0, 60.0])
            self.assertEqual(observation["original_label_id"], None)
            self.assertTrue(observation["annotation_flags"]["truncated"])
            self.assertFalse(observation["annotation_flags"]["difficult"])
            self.assertEqual(observation["canonical_label"], "drowning")

    def test_voc_scan_and_converter_reject_out_of_image_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "raw"
            root.mkdir()
            write_image(root / "JPEGImages" / "one.png")
            annotations = root / "Annotations"
            annotations.mkdir()
            xml = '''<annotation><filename>one.png</filename><object><name>Drowning</name>
            <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>120</xmax><ymax>30</ymax></bndbox></object></annotation>'''
            (annotations / "one.xml").write_text(xml, encoding="utf-8")
            source = {"schema_version": 1, "source_id": "voc-fixture", "source_version": "v1",
                      "format": "pascal_voc", "modality": "image",
                      "paths": {"root_env": "FIXTURE_ROOT", "annotations": "Annotations", "images": "JPEGImages"},
                      "mapping": {"Drowning": {"action": "map", "target": "drowning"}}}
            with self.assertRaisesRegex(SourceConfigError, "outside"):
                convert_to_jsonl(source, root, Path(directory) / "derived.jsonl")

    def test_yolo_rejects_segmentation_rows_and_bbox_outside_image(self):
        for annotation in ("0 0.5 0.5 0.2 0.2 0.1 0.2", "0 0.05 0.5 0.4 0.2"):
            with self.subTest(annotation=annotation), tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                root = base / "raw"
                root.mkdir()
                (root / "data.yaml").write_text("names: [Drowning]\n", encoding="utf-8")
                write_image(root / "images" / "sample.png")
                (root / "labels").mkdir()
                (root / "labels" / "sample.txt").write_text(annotation + "\n", encoding="utf-8")
                source = {
                    "schema_version": 1, "source_id": "fixture-yolo", "source_version": "v1",
                    "format": "yolo_detection", "modality": "image",
                    "paths": {"root_env": "FIXTURE_ROOT", "class_list": "data.yaml:names"},
                    "mapping": {"Drowning": {"action": "map", "target": "drowning"}},
                }
                with self.assertRaises(SourceConfigError):
                    convert_to_jsonl(source, root, base / "derived.jsonl")

    def test_coco_uses_category_ids_and_preserves_ignore_and_review(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            write_image(root / "images" / "one.png")
            annotation_file = root / "annotations" / "instances.json"
            annotation_file.parent.mkdir(parents=True)
            annotation_file.write_text(json.dumps({
                "categories": [
                    {"id": 2, "name": "drow"},
                    {"id": 9, "name": "boat"},
                    {"id": 20, "name": "normal"},
                ],
                "images": [{"id": 77, "file_name": "one.png"}],
                "annotations": [
                    {"id": 50, "image_id": 77, "category_id": 2, "bbox": [10, 12, 30, 24]},
                    {"id": 51, "image_id": 77, "category_id": 9, "bbox": [1, 1, 5, 5], "iscrowd": 1},
                    {"id": 52, "image_id": 77, "category_id": 20, "bbox": [50, 20, 12, 14]},
                ],
            }), encoding="utf-8")
            source = {
                "schema_version": 1, "source_id": "fixture-coco", "source_version": "v1",
                "format": "coco_detection", "modality": "image",
                "paths": {"root_env": "FIXTURE_ROOT", "class_list": "annotations/instances.json:categories",
                          "annotations": "annotations/instances.json", "images": "images"},
                "mapping": {"drow": {"action": "map", "target": "drowning"},
                            "boat": {"action": "ignore", "reason": "context"}},
            }
            output = base / "derived.jsonl"
            convert_to_jsonl(source, root, output)
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            observations = [record for record in records if record["entity"] == "observation"]
            self.assertEqual(observations[0]["original_label_id"], 2)
            self.assertEqual(observations[0]["bbox"], [10.0, 12.0, 40.0, 36.0])
            self.assertEqual(observations[1]["mapping_action"], "ignore")
            self.assertTrue(observations[1]["iscrowd"])
            self.assertEqual(observations[1]["annotation_status"], "ignored_annotation")
            self.assertEqual(observations[2]["mapping_action"], "review")
            self.assertIsNone(observations[2]["canonical_label"])

    def test_refuses_output_inside_raw_root_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            root.mkdir()
            (root / "data.yaml").write_text("names: [Drowning]\n", encoding="utf-8")
            write_image(root / "images" / "sample.png")
            (root / "labels").mkdir()
            (root / "labels" / "sample.txt").write_text("", encoding="utf-8")
            source = {
                "schema_version": 1, "source_id": "fixture-yolo", "source_version": "v1",
                "format": "yolo_detection", "modality": "image",
                "paths": {"root_env": "FIXTURE_ROOT", "class_list": "data.yaml:names"}, "mapping": {},
            }
            with self.assertRaisesRegex(SourceConfigError, "outside"):
                convert_to_jsonl(source, root, root / "derived.jsonl")
            existing = base / "existing.jsonl"
            existing.write_text("do not overwrite", encoding="utf-8")
            with self.assertRaisesRegex(SourceConfigError, "Refusing to overwrite"):
                convert_to_jsonl(source, root, existing)
            self.assertEqual(existing.read_text(encoding="utf-8"), "do not overwrite")
            with self.assertRaisesRegex(SourceConfigError, "sources/"):
                convert_to_jsonl(source, root, base / "sources" / "new.jsonl")
            with self.assertRaisesRegex(SourceConfigError, "DrowningDemo is protected"):
                convert_to_jsonl(source, root, Path("D:/DrowningDemo/new.jsonl"))

    def test_cli_convert_uses_env_root_and_writes_derived_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "raw"
            root.mkdir()
            (root / "data.yaml").write_text("names: [Drowning]\n", encoding="utf-8")
            write_image(root / "images" / "sample.png")
            (root / "labels").mkdir()
            (root / "labels" / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            config = {
                "schema_version": 1, "source_id": "cli-fixture", "source_version": "v1",
                "format": "yolo_detection", "modality": "image",
                "paths": {"root_env": "FIXTURE_ROOT", "class_list": "data.yaml:names"},
                "mapping": {"Drowning": {"action": "map", "target": "drowning"}},
            }
            config_path = base / "source.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            env_path = base / ".env"
            env_path.write_text(f"FIXTURE_ROOT={root}\n", encoding="utf-8")
            output = base / "derived" / "records.jsonl"
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["--env-file", str(env_path), "source", "convert", str(config_path),
                                  "--output", str(output)])
            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["media_count"], 1)
            self.assertEqual(payload["observation_count"], 1)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
