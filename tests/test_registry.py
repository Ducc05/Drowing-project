import json
import tempfile
import unittest
from pathlib import Path

from drowning.data.labels import resolve_label
from drowning.data.registry import SourceConfigError, inspect_source, read_class_names, validate_source


def source_config():
    return {
        "schema_version": 1,
        "source_id": "fixture",
        "source_version": "test-1",
        "format": "yolo_detection",
        "modality": "image",
        "paths": {"root_env": "FIXTURE_ROOT", "class_list": "data.yaml:names"},
        "mapping": {
            "Drowning": {"action": "map", "target": "drowning"},
            "normal": {"action": "review", "reason": "ambiguous source meaning"},
            "boat": {"action": "ignore", "reason": "not a person"},
        },
    }


class RegistryTests(unittest.TestCase):
    def test_source_contract_and_mapping_report(self):
        config = source_config()
        self.assertEqual(validate_source(config), [])
        report = inspect_source(config, {})
        self.assertTrue(report["valid_config"])
        self.assertFalse(report["root_configured"])
        self.assertEqual(report["unknown_labels_default_to"], "review")
        self.assertFalse(report["conversion_performed"])

    def test_mapping_is_exact_after_typographic_normalization(self):
        mapping = source_config()["mapping"]
        self.assertEqual(resolve_label("  DROWNING ", mapping).canonical_label, "drowning")
        self.assertEqual(resolve_label("normal", mapping).action, "review")
        self.assertEqual(resolve_label("not drowning", mapping).action, "review")
        self.assertEqual(resolve_label("swimming", mapping).action, "review")

    def test_reads_yolo_names_in_declared_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.yaml").write_text("names:\n  0: Drowning\n  1: Swimming\n", encoding="utf-8")
            found = read_class_names(source_config(), root)
            self.assertEqual(found, [(0, "Drowning"), (1, "Swimming")])

    def test_reads_non_contiguous_coco_category_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "annotations.json").write_text(json.dumps({"categories": [
                {"id": 2, "name": "drown"}, {"id": 9, "name": "swim"}
            ]}), encoding="utf-8")
            config = source_config()
            config["format"] = "coco_detection"
            config["paths"]["class_list"] = "annotations.json:categories"
            self.assertEqual(read_class_names(config, root), [(2, "drown"), (9, "swim")])

    def test_rejects_class_list_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            config = source_config()
            config["paths"]["class_list"] = "../outside.yaml:names"
            with self.assertRaisesRegex(SourceConfigError, "escapes"):
                read_class_names(config, Path(directory))

    def test_rejects_invalid_map_target_and_normalized_duplicate(self):
        config = source_config()
        config["mapping"]["drowning"] = {"action": "map", "target": "swimming"}
        self.assertTrue(any("normalized duplicate" in error for error in validate_source(config)))
        config["mapping"] = {"panic": {"action": "map", "target": "swim"}}
        self.assertTrue(any("invalid canonical target" in error for error in validate_source(config)))


if __name__ == "__main__":
    unittest.main()
