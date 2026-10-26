import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from drowning.cli import main
from drowning.data.audit import audit_jsonl


def records_for_source(source_id, media_id, class_name, box, fingerprint):
    media = {
        "entity": "media", "schema_version": 1, "media_id": media_id,
        "source_id": source_id, "content_sha256": fingerprint, "width": 100, "height": 80,
    }
    observation = {
        "entity": "observation", "schema_version": 1, "observation_id": f"obs-{media_id}",
        "media_id": media_id, "source_id": source_id, "mapping_action": "map",
        "canonical_label": class_name, "bbox": box,
    }
    return [media, observation]


class AuditTests(unittest.TestCase):
    def test_detects_cross_source_exact_duplicate_with_conflicting_box_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            all_records = records_for_source("source-a", "media-a", "drowning", [10, 10, 40, 40], "a" * 64)
            all_records += records_for_source("source-b", "media-b", "swimming", [10, 10, 40, 40], "a" * 64)
            path.write_text("\n".join(json.dumps(record) for record in all_records) + "\n", encoding="utf-8")
            report = audit_jsonl([path])
            self.assertTrue(report["valid"])
            self.assertEqual(len(report["exact_duplicate_content_groups"]), 1)
            self.assertEqual(report["same_box_label_conflicts"][0]["labels"], ["drowning", "swimming"])
            self.assertEqual(report["same_box_label_conflicts"][0]["disposition"], "review_all_matching_annotations")
            self.assertEqual(report["cross_split_duplicate_group_count"], 0)
            self.assertFalse(report["automatic_quarantine_performed"])

    def test_reports_duplicates_that_cross_original_export_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            train = records_for_source("source-a", "media-train", "drowning", [5, 5, 25, 25], "c" * 64)
            test = records_for_source("source-a", "media-test", "drowning", [5, 5, 25, 25], "c" * 64)
            train[0]["original_export_split"] = "train"
            test[0]["original_export_split"] = "test"
            path.write_text("\n".join(json.dumps(record) for record in train + test) + "\n", encoding="utf-8")
            report = audit_jsonl([path])
            self.assertEqual(report["cross_split_duplicate_group_count"], 1)
            self.assertTrue(report["exact_duplicate_content_groups"][0]["cross_split_duplicate"])

    def test_reports_dangling_media_reference_invalid_bbox_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            media, _ = records_for_source("source-a", "media-a", "drowning", [0, 0, 1, 1], "b" * 64)
            bad_observation = {
                "entity": "observation", "schema_version": 1, "observation_id": "bad",
                "media_id": "missing-media", "source_id": "source-a", "mapping_action": "review",
                "canonical_label": None, "bbox": [1, 1, 1, 5],
            }
            path.write_text(json.dumps(media) + "\n" + json.dumps(bad_observation) + "\n{invalid\n", encoding="utf-8")
            report = audit_jsonl([path])
            self.assertFalse(report["valid"])
            codes = {issue["code"] for issue in report["issues"]}
            self.assertIn("missing_media_reference", codes)
            self.assertIn("nonpositive_bbox_area", codes)
            self.assertIn("invalid_json", codes)

    def test_cli_audit_returns_nonzero_for_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text("{invalid\n", encoding="utf-8")
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(["--env-file", str(Path(directory) / "missing.env"), "source", "audit", str(path)])
            self.assertEqual(exit_code, 2)
            report = json.loads(stdout.getvalue())
            self.assertFalse(report["valid"])
            self.assertEqual(report["issues"][0]["code"], "invalid_json")

    def test_cli_audit_can_write_report_without_overwriting_input(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "valid.jsonl"
            media, observation = records_for_source("source-a", "media-a", "drowning", [1, 1, 5, 5], "d" * 64)
            input_path.write_text(json.dumps(media) + "\n" + json.dumps(observation) + "\n", encoding="utf-8")
            report_path = Path(directory) / "audit.json"
            output = StringIO()
            with redirect_stdout(output):
                result = main(["--env-file", str(Path(directory) / "missing.env"), "source", "audit",
                               str(input_path), "--output", str(report_path)])
            self.assertEqual(result, 0)
            self.assertTrue(json.loads(report_path.read_text(encoding="utf-8"))["valid"])
            self.assertEqual(json.loads(output.getvalue())["issue_count"], 0)


if __name__ == "__main__":
    unittest.main()
