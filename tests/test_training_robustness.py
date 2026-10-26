import json
import unittest

from drowning.orchestration.kaggle_train import prepare_training_inputs, run_train_candidate
from tests.test_kaggle_train import TrainingInputFixture


class TrainingRobustnessTests(TrainingInputFixture):
    def prepare_image_only(self, name="work", **kwargs):
        return prepare_training_inputs(sources_dir=self.sources, reviews_dir=self.reviews, input_root=self.inputs,
                                       work_dir=self.base / name, recipes_dir=self.recipes, branch="image_only", **kwargs)

    def train(self, inputs, name, **options):
        return run_train_candidate(inputs, device="cpu", options={"contract_test": True, "branch": "image_only", **options},
                                   output_dir=self.base / name, archive_path=self.base / f"{name}.zip", code_fingerprint="c" * 64)

    def test_best_epoch_is_kept_early_stop_works_and_validation_is_reported_per_source(self):
        (self.sources / "pool_videos.yaml").unlink()  # images only
        inputs = self.prepare_image_only("w-sel")
        metrics = self.train(inputs, "run-sel", image_epochs=4, patience=1)
        selection = metrics["epoch_selection"]["images"]
        self.assertEqual(selection["epochs_requested"], 4)
        self.assertTrue(1 <= selection["best_epoch"] <= selection["epochs_run"] <= 4)
        self.assertEqual(selection["early_stopped"], selection["epochs_run"] < 4)
        history = metrics["image_stage_history"]
        self.assertEqual(len(history), selection["epochs_run"])
        self.assertTrue(all("validation_macro_f1" in entry for entry in history))
        self.assertEqual(set(metrics["validation_by_source"]["images"]), {"pool_images"})
        self.assertEqual(len(metrics["validation_by_source"]["images"]["pool_images"]["confusion_matrix"]), 3)
        self.assertEqual(metrics["augmentation"], {"enabled": True, "applied_to": "train split only", "seeded": True})
        self.assertIn("optimistic", metrics["epoch_selection_note"])
        self.assertIsNone(metrics["heldout"])
        samples = metrics["validation_samples"]["images:validation"]
        self.assertTrue(samples and set(samples[0]) == {"s", "p", "f", "b", "l"})  # enough to score another model on the same crops
        self.assertEqual({s["s"] for s in samples}, {"pool_images"})

    def test_never_stopping_early_runs_every_epoch_and_augmentation_can_be_turned_off(self):
        (self.sources / "pool_videos.yaml").unlink()
        inputs = self.prepare_image_only("w-full")
        metrics = self.train(inputs, "run-full", image_epochs=3, patience=0, augment=False)
        self.assertEqual(metrics["epoch_selection"]["images"]["epochs_run"], 3)
        self.assertFalse(metrics["epoch_selection"]["images"]["early_stopped"])
        self.assertFalse(metrics["augmentation"]["enabled"])

    def test_same_seed_and_options_still_give_identical_weights_with_augmentation(self):
        (self.sources / "pool_videos.yaml").unlink()
        inputs = self.prepare_image_only("w-det")
        digests = []
        for name in ("det-a", "det-b"):
            self.train(inputs, name, image_epochs=2)
            manifest = json.loads((self.base / name / "model-bundle" / "manifest.json").read_text(encoding="utf-8"))
            digests.append(manifest["files"]["weights.pt"]["sha256"])
        self.assertEqual(digests[0], digests[1])

    def test_a_held_out_source_is_never_trained_on_and_is_scored_separately(self):
        inputs = self.prepare_image_only("w-hold", holdout_sources=["pool_images"])
        self.assertEqual(inputs["blockers"], [])
        report = inputs["report"]
        self.assertEqual(report["holdout_sources"], ["pool_images"])
        self.assertGreater(report["image_crops_by_split"]["heldout"]["drowning"], 0)
        self.assertEqual(report["image_crops_by_split"]["train"], {})  # every loose image is held out
        self.assertEqual(report["training_phases"], ["video_frames"])
        rows = [json.loads(line) for line in inputs["paths"]["crop_manifest"].read_text(encoding="utf-8").splitlines()]
        self.assertTrue(rows and all(r["split"] == "heldout" for r in rows))
        metrics = self.train(inputs, "run-hold", video_epochs=1, holdout_sources=["pool_images"])
        self.assertEqual(metrics["training_phases"], ["video_frames"])
        self.assertEqual(metrics["samples"]["image_train"], 0)
        held = metrics["heldout"]["images"]
        self.assertGreater(held["samples"], 0)
        self.assertEqual(set(held["by_source"]), {"pool_images"})
        self.assertEqual(len(held["overall"]["confusion_matrix"]), 3)

    def test_unknown_or_total_holdout_is_a_clear_blocker_and_temporal_branch_refuses_it(self):
        self.assertIn("holdout_source_not_found:nope", self.prepare_image_only("w-x1", holdout_sources=["nope"])["blockers"])
        self.assertIn("every_source_is_held_out",
                      self.prepare_image_only("w-x2", holdout_sources=["pool_images", "pool_videos"])["blockers"])
        with self.assertRaises(ValueError):
            prepare_training_inputs(sources_dir=self.sources, reviews_dir=self.reviews, input_root=self.inputs,
                                    work_dir=self.base / "w-x3", recipes_dir=self.recipes, branch="temporal",
                                    holdout_sources=["pool_images"])

    def test_leakage_report_is_present_for_validation_and_never_blocks(self):
        (self.sources / "pool_videos.yaml").unlink()
        inputs = self.prepare_image_only("w-leak")
        check = inputs["report"]["leakage_check"]
        self.assertIn("validation", check)
        self.assertTrue(check["validation"]["lower_bound"])
        self.assertEqual(inputs["blockers"], [])


if __name__ == "__main__":
    unittest.main()
