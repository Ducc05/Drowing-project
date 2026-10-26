"""Library-level and generated-notebook checks for the Kaggle prepare/smoke runs (CPU, synthetic data)."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from drowning.orchestration.kaggle_run import (PACKAGE_NAME, REPORT_NAME, discover_source_root, package_output,
                                               run_prepare, run_smoke)

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["Drowning", "Swimming", "out of water"]
SPEC = importlib.util.spec_from_file_location("notebook_builder", ROOT / "notebooks" / "build_notebook.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def texture(seed, size=64):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).resize((size, size), Image.Resampling.NEAREST)


def make_yolo_export(root: Path, names=NAMES, count=15):
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names}), encoding="utf-8")
    for i in range(count):
        split = "train" if i % 3 else "valid"
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        texture(i).save(root / split / "images" / f"img{i}.png")
        (root / split / "labels" / f"img{i}.txt").write_text(f"{i % len(names)} 0.5 0.5 0.6 0.6\n", encoding="utf-8")


class KaggleRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.inputs = self.base / "kaggle-input"
        self.sources_dir = ROOT / "configs" / "datasets"
        self.reviews = self.base / "reviews"

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, **overrides):
        args = dict(sources_dir=self.sources_dir, reviews_dir=self.reviews, input_root=self.inputs,
                    output_dir=self.base / "out", work_dir=self.base / "work", code_fingerprint="f" * 64)
        args.update(overrides)
        return run_prepare(**args)

    def test_discovery_finds_nested_dataset_and_refuses_to_guess(self):
        source = yaml.safe_load((self.sources_dir / "alanoud_swimming_drowning_v4.yaml").read_text(encoding="utf-8"))
        self.inputs.mkdir()
        self.assertEqual(discover_source_root(source, self.inputs)["status"], "missing")
        make_yolo_export(self.inputs / "datasets" / "someone" / "pool-export")
        found = discover_source_root(source, self.inputs)
        self.assertEqual((found["status"], found["candidates"]), ("found", ["datasets/someone/pool-export"]))
        make_yolo_export(self.inputs / "second-copy")
        self.assertEqual(discover_source_root(source, self.inputs)["status"], "ambiguous")

    def test_discovery_ignores_datasets_whose_class_names_differ(self):
        source = yaml.safe_load((self.sources_dir / "alanoud_swimming_drowning_v4.yaml").read_text(encoding="utf-8"))
        make_yolo_export(self.inputs / "other", names=["person", "boat", "buoy"], count=3)
        result = discover_source_root(source, self.inputs)
        self.assertEqual((result["status"], result["candidates"]), ("missing", []))

    def test_prepare_with_no_input_reports_missing_and_does_not_train(self):
        self.inputs.mkdir()
        report = self.prepare()
        self.assertEqual(report["sources"][0]["status"], "input_missing")
        self.assertIn("alanoud_swimming_drowning:input_missing", report["blockers"])
        self.assertIn("no_source_prepared", report["blockers"])
        self.assertFalse(report["training_started"] or report["model_candidate"])
        self.assertTrue((self.base / "out" / REPORT_NAME).is_file())

    def test_prepare_on_real_layout_writes_review_material_and_blockers(self):
        make_yolo_export(self.inputs / "pool-export")
        report = self.prepare()
        entry = report["sources"][0]
        self.assertEqual(entry["status"], "prepared", entry.get("error"))
        self.assertEqual((entry["mapping_review"], entry["grouping_review"]), ("pending", "pending"))
        self.assertEqual(entry["scan"]["media_count"], 15)
        self.assertTrue(entry["audit"]["valid"])
        source_dir = self.base / "out" / "per-source" / "alanoud_swimming_drowning@4"
        for name in ("scan.json", "audit.json", "quality.json", "mapping-report.json", "proposed-groups.json", "sheets/samples.json"):
            self.assertTrue((source_dir / name).is_file(), name)
        self.assertIn("alanoud_swimming_drowning:mapping_review_not_approved", report["blockers"])
        self.assertIn("alanoud_swimming_drowning:source_groups_not_verified", report["blockers"])
        self.assertIn("no_temporal_video_source_prepared", report["blockers"])
        self.assertNotIn(str(self.inputs), json.dumps(report))
        archive = package_output(self.base / "out", self.base / PACKAGE_NAME)
        with zipfile.ZipFile(archive) as bundle:
            self.assertIn(REPORT_NAME, bundle.namelist())
            self.assertTrue(all("\\" not in name and not name.startswith("/") for name in bundle.namelist()))

    def test_prepare_isolates_a_failing_source(self):
        make_yolo_export(self.inputs / "pool-export")
        (self.inputs / "pool-export" / "valid" / "labels" / "img0.txt").write_text("9 0.5 0.5 0.6 0.6\n", encoding="utf-8")
        report = self.prepare()
        self.assertIn(report["sources"][0]["status"], {"failed"})
        self.assertIn("alanoud_swimming_drowning:prepare_failed", report["blockers"])

    def test_smoke_runs_the_real_model_without_claiming_a_candidate(self):
        report = run_smoke(output_dir=self.base / "smoke", code_fingerprint="f" * 64)
        self.assertEqual(report["image_logits_shape"][-1], 3)
        self.assertTrue(report["all_trainable_parameters_received_gradients"])
        self.assertFalse(report["model_candidate"])
        self.assertEqual(report["gpu_confirmed"], report["environment"]["cuda_available"])

    def test_prepare_counts_temporal_windows_for_a_labeled_video_source_without_a_split(self):
        from tests.test_cvat_adapter import cvat_source, make_video
        project = self.base / "project"
        (project / "configs" / "datasets").mkdir(parents=True)
        (project / "configs" / "recipes").mkdir(parents=True)
        (project / "configs" / "datasets" / "cvat.yaml").write_text(yaml.safe_dump(cvat_source()), encoding="utf-8")
        (project / "configs" / "recipes" / "temporal.yaml").write_text(yaml.safe_dump({"sampling": {
            "sequence_length": 2, "sample_period_ms": 100, "max_track_gap_ms": 300, "min_valid_ratio": 1.0}}), encoding="utf-8")
        data = self.inputs / "video-dataset"
        data.mkdir(parents=True)
        make_video(data / "video.avi")
        (data / "timestamps.json").write_text(json.dumps([0, 100, 200, 300, 400]), encoding="utf-8")
        boxes = "".join(f'<box frame="{f}" xtl="2" ytl="2" xbr="20" ybr="28" outside="0" occluded="0" keyframe="1">'
                        '<attribute name="state">Drowning</attribute></box>' for f in range(5))
        (data / "annotations.xml").write_text(f'<annotations><track id="7" label="person">{boxes}</track></annotations>', encoding="utf-8")
        report = self.prepare(sources_dir=project / "configs" / "datasets")
        entry = report["sources"][0]
        self.assertEqual(entry["status"], "prepared", entry.get("error"))
        self.assertEqual(entry["temporal"]["window_count"], 4)
        self.assertEqual(entry["temporal"]["windows_by_label"], {"drowning": 4})
        self.assertEqual(entry["temporal"]["training_eligible_window_count"], 0)
        self.assertEqual(report["temporal_window_count_before_split"], 4)
        self.assertNotIn("no_temporal_video_source_prepared", report["blockers"])
        self.assertTrue((self.base / "out" / "per-source" / "cvat-video-fixture@v1" / "sheets" / "drowning--random.jpg").is_file())

    def run_notebook(self, mode):
        notebook = BUILDER.build_notebook(mode)
        code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
                         and "train_candidate" not in "".join(cell["source"]).split("\n")[0])
        script = self.base / f"notebook_{mode}.py"
        script.write_text(code, encoding="utf-8")
        env = {**os.environ, "DROWN_CODE_DIR": str(self.base / "code"), "DROWN_INPUT_ROOT": str(self.inputs),
               "DROWN_OUTPUT_DIR": str(self.base / "nb-out"), "DROWN_SCRATCH_DIR": str(self.base / "scratch"),
               "DROWN_ARCHIVE_PATH": str(self.base / "nb-run-output.zip")}
        return subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=300,
                              cwd=str(self.base))

    def test_generated_prepare_notebook_runs_end_to_end_from_its_embedded_source(self):
        make_yolo_export(self.inputs / "datasets" / "pool-export")
        result = self.run_notebook("prepare_only")
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        with zipfile.ZipFile(self.base / "nb-run-output.zip") as bundle:
            report = json.loads(bundle.read(REPORT_NAME))
        self.assertEqual(report["run_mode"], "prepare_only")
        self.assertEqual(report["sources"][0]["status"], "prepared")
        self.assertEqual(report["code_fingerprint"], BUILDER_FINGERPRINT(BUILDER.build_notebook("prepare_only")))
        self.assertFalse(report["training_started"])

    def test_generated_smoke_notebook_runs_and_writes_a_non_candidate_report(self):
        result = self.run_notebook("remote_smoke")
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        with zipfile.ZipFile(self.base / "nb-run-output.zip") as bundle:
            report = json.loads(bundle.read(REPORT_NAME))
        self.assertEqual((report["run_mode"], report["model_candidate"]), ("remote_smoke", False))


def BUILDER_FINGERPRINT(notebook):
    import ast
    import base64
    import hashlib
    setup = "".join(notebook["cells"][1]["source"])
    payload = ast.literal_eval(setup.split("SOURCE_BUNDLE_B64 = ", 1)[1].splitlines()[0])
    return hashlib.sha256(base64.b64decode(payload)).hexdigest()


if __name__ == "__main__":
    unittest.main()
