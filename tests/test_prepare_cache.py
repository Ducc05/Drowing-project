"""Incremental prepare must skip work only for identical, verified source inputs."""
import base64
import json
import shutil
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import yaml

from drowning.data.prepare_cache import MANIFEST, bundle_snapshots, source_fingerprints
from drowning.data import prepare_cache
from drowning.orchestration import kaggle_run
from drowning.orchestration.kaggle_run import package_output
from drowning.orchestration.run_collector import summarize_report, verify_run_archive
from tests.test_kaggle_run import BUILDER, ROOT, make_yolo_export


class PrepareCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.configs = self.base / "configs"
        self.sources = self.configs / "datasets"
        self.sources.mkdir(parents=True)
        shutil.copytree(ROOT / "configs" / "recipes", self.configs / "recipes")
        for name in ("ontology.yaml", "review_policy.yaml"):
            shutil.copyfile(ROOT / "configs" / name, self.configs / name)
        self.inputs = self.base / "inputs"
        self.add_source("a")
        self.counter = 0
        self.environment = patch.object(kaggle_run, "_environment", return_value={"synthetic_fixture": True})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def tearDown(self):
        self.temporary.cleanup()

    def add_source(self, name):
        source = yaml.safe_load((ROOT / "configs" / "datasets" / "alanoud_swimming_drowning_v4.yaml").read_text(encoding="utf-8"))
        source.update(source_id=name, source_version="1")
        source["paths"]["input_dir_contains"] = name + "-input"
        (self.sources / (name + ".yaml")).write_text(yaml.safe_dump(source), encoding="utf-8")
        make_yolo_export(self.inputs / (name + "-input"), count=6)

    def prepare(self, cache=None):
        self.counter += 1
        out = self.base / f"out-{self.counter}"
        report = kaggle_run.run_prepare(sources_dir=self.sources, reviews_dir=self.base / "reviews",
                                       input_root=self.inputs, output_dir=out,
                                       work_dir=self.base / f"work-{self.counter}", code_fingerprint="a" * 64,
                                       cache_dir=cache)
        self.assertTrue(all(s["status"] == "prepared" for s in report["sources"]), report["sources"])
        return report, out

    def test_identical_source_reuses_reports_without_running_preparation(self):
        first, original = self.prepare()
        with patch.object(kaggle_run, "_prepare_source", side_effect=AssertionError("unchanged source was prepared again")):
            second, restored = self.prepare(original)
        self.assertEqual(second["incremental_prepare"], {"reused_sources": ["a"], "prepared_sources": []})
        self.assertEqual(first["sources"][0]["canonical_sha256"], second["sources"][0]["canonical_sha256"])
        for name in ("quality.json", "mapping-report.json", "sheets/samples.json", "sheets/drowning--random.jpg"):
            self.assertEqual((original / "per-source/a@1" / name).read_bytes(),
                             (restored / "per-source/a@1" / name).read_bytes())
        archive = package_output(restored, self.base / "reused.zip")
        checked = verify_run_archive(archive, self.base / "verified", expected_mode="prepare_only")
        summary = summarize_report(checked["report"])
        self.assertEqual(summary["sources"][0]["cache_status"], "reused")
        self.assertEqual(summary["incremental_prepare"]["reused_sources"], ["a"])

    def test_new_source_is_prepared_and_existing_source_is_reused(self):
        _, original = self.prepare()
        self.add_source("b")
        actual = kaggle_run._prepare_source
        with patch.object(kaggle_run, "_prepare_source", wraps=actual) as work:
            second, _ = self.prepare(original)
        self.assertEqual([c.args[0]["source_id"] for c in work.call_args_list], ["b"])
        self.assertEqual(second["incremental_prepare"], {"reused_sources": ["a"], "prepared_sources": ["b"]})

    def test_changed_labels_images_mapping_and_recipe_invalidate_cache(self):
        _, original = self.prepare()
        labels = self.inputs / "a-input/train/labels/img1.txt"
        image = self.inputs / "a-input/train/images/img1.png"
        descriptor = self.sources / "a.yaml"
        recipe = self.configs / "recipes/image.yaml"
        mutations = [
            (labels, lambda: labels.write_text("0 0.5 0.5 0.6 0.6\n", encoding="utf-8")),
            (image, lambda: shutil.copyfile(self.inputs / "a-input/train/images/img2.png", image)),
            (descriptor, lambda: descriptor.write_text(descriptor.read_text(encoding="utf-8").replace(
                "target: swimming", "target: drowning"), encoding="utf-8")),
            (recipe, lambda: recipe.write_text(recipe.read_text(encoding="utf-8").replace(
                "batch_size: 32", "batch_size: 16"), encoding="utf-8")),
        ]
        for path, mutate in mutations:
            with self.subTest(path=path.name):
                before = path.read_bytes()
                mutate()
                with patch.object(kaggle_run, "_prepare_source", wraps=kaggle_run._prepare_source) as work:
                    report, _ = self.prepare(original)
                self.assertEqual(work.call_count, 1)
                self.assertEqual(report["incremental_prepare"]["prepared_sources"], ["a"])
                path.write_bytes(before)

    def test_content_fingerprint_is_independent_of_root_and_file_timestamps(self):
        source = yaml.safe_load((self.sources / "a.yaml").read_text(encoding="utf-8"))
        first = source_fingerprints(source, self.inputs / "a-input", self.configs)
        copy = self.base / "another-location"
        shutil.copytree(self.inputs / "a-input", copy)
        for path in copy.rglob("*"):
            if path.is_file():
                import os
                os.utime(path, (1000000000, 1000000000))
        self.assertEqual(first, source_fingerprints(source, copy, self.configs))

    def test_processor_or_library_change_invalidates_cached_reports(self):
        _, original = self.prepare()
        old_hash = prepare_cache._text_sha256
        old_version = prepare_cache.importlib.metadata.version
        changes = [
            patch.object(prepare_cache, "_text_sha256", side_effect=lambda path:
                         "f" * 64 if path.name == "adapters.py" else old_hash(path)),
            patch.object(prepare_cache.importlib.metadata, "version", side_effect=lambda name:
                         "changed-version" if name == "Pillow" else old_version(name)),
        ]
        for change in changes:
            with change, patch.object(kaggle_run, "_prepare_source", wraps=kaggle_run._prepare_source) as work:
                report, _ = self.prepare(original)
            self.assertEqual(work.call_count, 1)
            self.assertEqual(report["incremental_prepare"]["reused_sources"], [])

    def test_legacy_or_corrupted_cache_falls_back_to_real_preparation(self):
        for damage in ("legacy", "corrupt", "missing", "bad_manifest", "bad_entry"):
            with self.subTest(damage=damage):
                _, original = self.prepare()
                folder = original / "per-source/a@1"
                if damage == "legacy":
                    (folder / MANIFEST).unlink()
                elif damage == "corrupt":
                    (folder / "quality.json").write_text("{}", encoding="utf-8")
                elif damage == "missing":
                    (folder / "audit.json").unlink()
                elif damage == "bad_manifest":
                    (folder / MANIFEST).write_text("[]", encoding="utf-8")
                else:
                    document = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
                    document["entry"] = []
                    (folder / MANIFEST).write_text(json.dumps(document), encoding="utf-8")
                with patch.object(kaggle_run, "_prepare_source", wraps=kaggle_run._prepare_source) as work:
                    report, _ = self.prepare(original)
                self.assertEqual(work.call_count, 1)
                self.assertEqual(report["incremental_prepare"]["reused_sources"], [])

    def test_only_verified_report_members_are_embedded_with_origin_and_budget(self):
        _, original = self.prepare()
        folder = original / "per-source/a@1"
        (folder / "secret.key").write_text("must not travel", encoding="utf-8")
        packed = bundle_snapshots([("job-a", folder), ("older-job", folder)])
        self.assertEqual(packed["source_versions"], 1)
        cache = self.base / "transported-cache"
        with zipfile.ZipFile(BytesIO(base64.b64decode(packed["payload"]))) as archive:
            self.assertFalse(any(name.endswith("secret.key") for name in archive.namelist()))
            archive.extractall(cache)
        with patch.object(kaggle_run, "_prepare_source", side_effect=AssertionError("transported snapshot was ignored")):
            report, _ = self.prepare(cache)
        self.assertEqual(report["sources"][0]["cache_origin_job_id"], "job-a")
        self.assertEqual(bundle_snapshots([("job-a", folder)], max_bytes=1)["source_versions"], 0)
        notebook = BUILDER.build_notebook("prepare_only", prepare_cache=packed)
        self.assertEqual(notebook["metadata"]["drowning_project"]["prepare_cache_source_versions"], 1)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "generated-notebook", "exec")

    def test_generated_notebook_reuses_snapshot_from_another_job(self):
        # Exercise the actual transport: verified cache -> notebook -> embedded
        # source in a fresh Python process, including Windows/LF normalization.
        output = self.base / "baseline-real-descriptor"
        source_id = "alanoud_swimming_drowning"
        kaggle_run.run_prepare(sources_dir=ROOT / "configs/datasets", reviews_dir=ROOT / "reviews",
                              input_root=self.inputs, output_dir=output, work_dir=self.base / "baseline-work",
                              code_fingerprint="b" * 64, only_sources={source_id})
        packed = bundle_snapshots([("baseline-job", output / "per-source/alanoud_swimming_drowning@4")])
        self.assertEqual(packed["source_versions"], 1)
        notebook = BUILDER.build_notebook("prepare_only", prepare_cache=packed)
        setup = "".join(notebook["cells"][1]["source"])
        run = "".join(notebook["cells"][3]["source"])
        reject_work = ('\nfrom drowning.orchestration import kaggle_run\n'
                       'kaggle_run._environment = lambda: {"synthetic_fixture": True}\n'
                       'def unexpected_work(*args, **kwargs):\n'
                       '    raise RuntimeError("unchanged source was prepared in generated notebook")\n'
                       'kaggle_run._prepare_source = unexpected_work\n')
        script = self.base / "generated-prepare.py"
        script.write_text(setup + reject_work + run, encoding="utf-8")
        env = {**os.environ, "DROWN_CODE_DIR": str(self.base / "embedded-code"),
               "DROWN_INPUT_ROOT": str(self.inputs), "DROWN_OUTPUT_DIR": str(self.base / "remote-out"),
               "DROWN_SCRATCH_DIR": str(self.base / "remote-work"),
               "DROWN_PREPARE_CACHE_DIR": str(self.base / "remote-cache"),
               "DROWN_ARCHIVE_PATH": str(self.base / "remote.zip")}
        result = subprocess.run([sys.executable, str(script)], env=env, cwd=self.base,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        report = json.loads((self.base / "remote-out/run-report.json").read_text(encoding="utf-8"))
        entry = next(s for s in report["sources"] if s["source_id"] == source_id)
        self.assertEqual(entry["status"], "prepared", entry.get("error"))
        self.assertEqual(entry["cache_status"], "reused")
        self.assertEqual(entry["cache_origin_job_id"], "baseline-job")


if __name__ == "__main__":
    unittest.main()
