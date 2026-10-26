import ast
import base64
import importlib.util
import io
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("notebook_builder", ROOT / "notebooks" / "build_notebook.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class KaggleNotebookBuilderTests(unittest.TestCase):
    def test_notebook_code_compiles_and_defaults_to_prepare_only(self):
        notebook = BUILDER.build_notebook()
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code_cells), 4)
        for cell in code_cells:
            ast.parse("".join(cell["source"]))
        setup = "".join(code_cells[0]["source"])
        self.assertIn('os.environ.get("DROWN_RUN_MODE",', setup)
        self.assertIn("prepare_only", setup)
        self.assertIn('"train_candidate"', setup)
        self.assertEqual(notebook["metadata"]["drowning_project"]["training_executor"], "kaggle")
        candidate = "".join(code_cells[3]["source"])
        self.assertIn("torch.cuda.is_available()", candidate)
        self.assertIn("no CPU fallback", candidate)
        self.assertIn("run_train_candidate", candidate)
        self.assertIn("write_blocked_report", candidate)
        self.assertIn('"approved": False', candidate)
        library = (ROOT / "src" / "drowning" / "orchestration" / "kaggle_train.py").read_text(encoding="utf-8")
        for needle in ("train_stage_a_epoch", "train_stage_b_epoch", '"test_split_used": False', "no CPU fallback"):
            self.assertIn(needle, library)
        smoke_notebook = BUILDER.build_notebook("remote_smoke")
        smoke_setup = "".join(next(cell for cell in smoke_notebook["cells"] if cell["cell_type"] == "code")["source"])
        self.assertIn('os.environ.get("DROWN_RUN_MODE", \'remote_smoke\')', smoke_setup)

    def test_embedded_source_bundle_excludes_local_data_and_credentials(self):
        notebook = BUILDER.build_notebook()
        setup = "".join(notebook["cells"][1]["source"])
        marker = "SOURCE_BUNDLE_B64 = "
        payload = setup.split(marker, 1)[1].splitlines()[0]
        bundle_text = ast.literal_eval(payload)
        archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(bundle_text)))
        names = archive.namelist()
        self.assertIn("src/drowning/ml/model.py", names)
        self.assertIn("configs/datasets/alanoud_swimming_drowning_v4.yaml", names)
        self.assertFalse(any(name.endswith(".env") or ".pt" in name or "local-cache" in name for name in names))
        self.assertNotIn("D:/DrowningKaggleCheck/data", "\n".join(archive.read(name).decode("utf-8", errors="ignore") for name in names))

    def test_bundle_hash_does_not_depend_on_file_times_or_machine(self):
        import os
        first = BUILDER.source_bundle_sha256()
        target = ROOT / "src" / "drowning" / "__init__.py"
        original = target.stat().st_mtime
        try:
            os.utime(target, (original + 5000, original + 5000))
            self.assertEqual(BUILDER.source_bundle_sha256(), first)
        finally:
            os.utime(target, (original, original))
        self.assertEqual(len(first), 64)

    def test_run_options_are_validated_and_embedded(self):
        notebook = BUILDER.build_notebook("train_candidate", {"pretrained": True, "image_epochs": 3})
        self.assertEqual(notebook["metadata"]["drowning_project"]["options"],
                         {"pretrained": True, "image_epochs": 3, "temporal_epochs": 5, "branch": "image_only",
                          "video_epochs": 3, "frame_stride_ms": 500, "max_frames_per_video": 300,
                          "continue_from_approved": False, "source_ids": None,
                          "augment": True, "patience": 3, "holdout_sources": None})
        self.assertIn("RUN_OPTIONS = " + repr(notebook["metadata"]["drowning_project"]["options"]), "".join(notebook["cells"][1]["source"]))
        for bad in ({"pretrained": "yes"}, {"image_epochs": 0}, {"temporal_epochs": 99}, {"epochs": 3}, {"image_epochs": True},
                    {"frame_stride_ms": 10}, {"max_frames_per_video": 1}, {"video_epochs": 0},
                    {"augment": "yes"}, {"patience": -1}, {"patience": 99}, {"holdout_sources": []}, {"holdout_sources": "a"},
                    {"source_ids": ["a"], "holdout_sources": ["b"]}):
            with self.assertRaises(ValueError):
                BUILDER.build_notebook("train_candidate", bad)

    def test_kernel_template_is_private_and_gpu_enabled(self):
        import json
        metadata = json.loads((ROOT / "notebooks" / "kernel-metadata.template.json").read_text(encoding="utf-8"))
        self.assertTrue(metadata["is_private"])
        self.assertTrue(metadata["enable_gpu"])
        self.assertEqual(metadata["dataset_sources"], [])
        self.assertTrue(metadata["id"].startswith("USERNAME/"))


if __name__ == "__main__":
    unittest.main()
