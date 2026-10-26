"""Checkpoint loading and packaging contracts; no real training or cloud calls."""
import ast
import contextlib
import hashlib
import io
import os
import shutil
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from uuid import uuid4

import torch

from drowning.inference.aggregation import DEFAULT_AGGREGATION
from drowning.ml.bundle import save_bundle
from drowning.ml.model import DrowningTemporalModel
from drowning.orchestration.kaggle_train import initialize_training_model, prepare_training_inputs, run_train_candidate
from notebooks.build_notebook import build_notebook, normalize_options
from drowning.integrations.kaggle import KaggleCLIClient, KaggleError
from tests.test_kaggle_train import TrainingInputFixture


def make_parent_bundle(directory, model_id=None):
    model_id = model_id or str(uuid4())
    model = DrowningTemporalModel(hidden_size=8).eval()
    manifest = save_bundle(model, directory, completion_kind="candidate", temporal_trained=False,
                          model_kind="image_tracked", aggregation=dict(DEFAULT_AGGREGATION),
                          preprocessing={"size": [32, 32], "context_fraction": 0.15,
                                         "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                          dataset_fingerprint="d" * 64, recipe_fingerprint="r" * 64,
                          code_fingerprint="c" * 64, model_version_id=model_id)
    parent = {"id": model_id, "model_kind": "image_tracked",
              "weights_sha256": manifest["files"]["weights.pt"]["sha256"],
              "manifest_sha256": hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()}
    return model, manifest, parent


class CheckpointContinuationTests(unittest.TestCase):
    def test_loaded_weights_equal_parent_and_wrong_contract_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "parent"
            original, manifest, _ = make_parent_bundle(directory)
            cfg = manifest["preprocessing"]
            inputs = {"recipes": ({"input": {"size": cfg["size"], "mean": cfg["mean"], "std": cfg["std"],
                                             "crop_context_fraction": cfg["context_fraction"]}},
                                  {"model": {"hidden_size": 8}})}
            options = {"branch": "image_only", "continue_from_approved": True}
            loaded = initialize_training_model(inputs, device="cpu", options=options, parent_bundle_dir=directory)
            self.assertTrue(all(torch.equal(value, loaded.state_dict()[key]) for key, value in original.state_dict().items()))
            with self.assertRaisesRegex(ValueError, "exactly one"):
                initialize_training_model(inputs, device="cpu", options=options)
            with self.assertRaisesRegex(ValueError, "model kind"):
                initialize_training_model(inputs, device="cpu", options={**options, "branch": "temporal"}, parent_bundle_dir=directory)
            inputs["recipes"][0]["input"]["size"] = [64, 64]
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                initialize_training_model(inputs, device="cpu", options=options, parent_bundle_dir=directory)

    def test_notebook_loads_the_exact_attached_parent_without_cloud_credentials(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, parent = make_parent_bundle(root / "parent")
            options = {"continue_from_approved": True, "source_ids": ["new_dataset"]}
            slug = "drown-parent-" + uuid4().hex
            attached = root / "inputs" / slug
            shutil.copytree(root / "parent", attached)
            notebook = build_notebook("train_candidate", options, parent_dataset_ref="owner/" + slug, parent_model=parent)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code": ast.parse("".join(cell["source"]))
            scope, paths = {}, list(sys.path)
            try:
                with mock.patch.dict(os.environ, {"DROWN_CODE_DIR": str(root / "code"), "DROWN_INPUT_ROOT": str(root / "inputs")}), contextlib.redirect_stdout(io.StringIO()):
                    exec("".join(notebook["cells"][1]["source"]), scope)
            finally:
                sys.path[:] = paths
            self.assertEqual(scope["PARENT_MODEL"], parent)
            self.assertEqual(scope["RUN_OPTIONS"]["source_ids"], ["new_dataset"])
            self.assertEqual((scope["PARENT_MODEL_DIR"] / "weights.pt").read_bytes(), (root / "parent/weights.pt").read_bytes())
            (attached / "weights.pt").write_bytes(b"corrupt checkpoint")
            try:
                with mock.patch.dict(os.environ, {"DROWN_CODE_DIR": str(root / "code"), "DROWN_INPUT_ROOT": str(root / "inputs")}), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "weights checksum"):
                        exec("".join(notebook["cells"][1]["source"]), {})
            finally:
                sys.path[:] = paths

    def test_checkpoint_dataset_upload_is_private_and_waits_for_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "parent"
            _, _, parent = make_parent_bundle(directory)
            ref = "owner/drown-parent-" + uuid4().hex
            (directory / "dataset-metadata.json").write_text(json.dumps({"id": ref}), encoding="utf-8")
            calls = []
            def runner(command, **kwargs):
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, "ready" if "status" in command else "created", "")
            client = KaggleCLIClient(executable="kaggle", runner=runner)
            client.push_private_checkpoint_dataset(directory, ref, parent, sleep=lambda _: None)
            self.assertEqual(calls[0][1:3], ["datasets", "create"])
            self.assertNotIn("--public", calls[0])
            self.assertNotIn("-u", calls[0])
            self.assertEqual(calls[1][1:3], ["datasets", "status"])
            client.runner = lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "processing", "")
            with self.assertRaisesRegex(KaggleError, "did not become ready"):
                client.push_private_checkpoint_dataset(directory, ref, parent, sleep=lambda _: None)
            (directory / ".env").write_text("private fixture", encoding="utf-8")
            with self.assertRaises(ValueError):
                client.push_private_checkpoint_dataset(directory, ref, parent, sleep=lambda _: None)

    def test_options_reject_missing_sources_and_conflicting_initialization(self):
        for options in ({"source_ids": []}, {"continue_from_approved": "yes"},
                        {"continue_from_approved": True, "pretrained": True}):
            with self.assertRaises(ValueError): normalize_options(options)
        with self.assertRaisesRegex(ValueError, "pinned"):
            build_notebook("train_candidate", {"continue_from_approved": True})


class SelectedDatasetTests(TrainingInputFixture):
    def test_training_inputs_use_only_the_selected_registered_dataset(self):
        inputs = prepare_training_inputs(sources_dir=self.sources, reviews_dir=self.reviews, input_root=self.inputs,
                                         work_dir=self.base / "selected", recipes_dir=self.recipes,
                                         branch="image_only", only_sources={"pool_images"})
        self.assertEqual(inputs["blockers"], [])
        self.assertEqual(inputs["report"]["included_sources"], ["pool_images"])

    def test_tiny_synthetic_continuation_records_lineage_and_preserves_parent(self):
        parent_dir = self.base / "parent"
        _, manifest, parent = make_parent_bundle(parent_dir)
        before = (parent_dir / "weights.pt").read_bytes()
        inputs = prepare_training_inputs(sources_dir=self.sources, reviews_dir=self.reviews, input_root=self.inputs,
                                         work_dir=self.base / "selected", recipes_dir=self.recipes,
                                         branch="image_only", only_sources={"pool_images"})
        result = run_train_candidate(inputs, device="cpu", options={"branch": "image_only", "contract_test": True,
                                     "continue_from_approved": True, "image_epochs": 1}, parent_bundle_dir=parent_dir,
                                     output_dir=self.base / "out", archive_path=self.base / "out.zip", code_fingerprint="c" * 64)
        self.assertEqual(result["completion_kind"], "synthetic_smoke")
        self.assertEqual(result["initialization"], "approved_checkpoint")
        self.assertEqual(result["parent_model_version"], parent["id"])
        self.assertEqual(result["parent_weights_sha256"], parent["weights_sha256"])
        self.assertTrue(result["optimizer_reset"])
        import json
        saved = json.loads((self.base / "out/model-bundle/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["parent_model_version"], parent["id"])
        self.assertEqual(saved["initialization"]["weights_source"], "parent_checkpoint")
        self.assertEqual((parent_dir / "weights.pt").read_bytes(), before)
