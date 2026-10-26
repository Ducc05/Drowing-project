import json
import sys
import unittest
from pathlib import Path

from drowning.db.models import ModelVersion  # noqa: E402
from drowning.inference.aggregation import DEFAULT_AGGREGATION  # noqa: E402
from drowning.inference.pipeline import PersonStateInference  # noqa: E402
from drowning.ml.bundle import load_bundle, validate_manifest  # noqa: E402
from drowning.ml.model import DrowningTemporalModel  # noqa: E402
from drowning.orchestration.kaggle_train import run_train_candidate  # noqa: E402
from tests.test_inference_pipeline import FakeClassifier, FixedPersonDetector, Image, approved_test_manifest  # noqa: E402
from tests.test_kaggle_train import TrainingInputFixture  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from notebooks.build_notebook import normalize_options  # noqa: E402


def image_tracked_manifest(**aggregation):
    manifest = approved_test_manifest()
    manifest.update(model_kind="image_tracked", temporal_trained=False, temporal_contract=None,
                    aggregation={**DEFAULT_AGGREGATION, "min_frames": 3, "min_alert_duration_ms": 200, **aggregation})
    return manifest


class ImageOnlyTrainingTests(TrainingInputFixture):
    def setUp(self):
        super().setUp()
        (self.sources / "pool_videos.yaml").unlink()  # like Alanoud: images only, no labeled video

    def prepare_branch(self, branch, name="work"):
        from drowning.orchestration.kaggle_train import prepare_training_inputs
        return prepare_training_inputs(sources_dir=self.sources, reviews_dir=self.reviews, input_root=self.inputs,
                                       work_dir=self.base / name, recipes_dir=self.recipes, branch=branch)

    def test_image_only_branch_needs_no_windows_but_the_temporal_branch_is_still_blocked(self):
        self.assertIn("no_temporal_windows", self.prepare_branch("temporal", "w-t")["blockers"])
        inputs = self.prepare_branch("image_only", "w-i")
        self.assertEqual(inputs["blockers"], [])
        self.assertEqual(inputs["report"]["branch"], "image_only")
        self.assertEqual(inputs["report"]["temporal_windows_by_split"]["train"], {})
        self.assertNotEqual(inputs["dataset_fingerprint"], self.prepare_branch("temporal", "w-t2")["dataset_fingerprint"])

    def test_image_only_run_produces_an_image_tracked_bundle_and_truthful_metrics(self):
        inputs = self.prepare_branch("image_only")
        metrics = run_train_candidate(inputs, device="cpu", options={"contract_test": True, "image_epochs": 1, "branch": "image_only"},
                                      output_dir=self.base / "out", archive_path=self.base / "out.zip", code_fingerprint="c" * 64)
        manifest = json.loads((self.base / "out" / "model-bundle" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["model_kind"], manifest["temporal_trained"], manifest["temporal_contract"]),
                         ("image_tracked", False, None))
        self.assertEqual(manifest["aggregation"], DEFAULT_AGGREGATION)
        self.assertEqual((metrics["model_kind"], metrics["branch"], metrics["temporal_validation"], metrics["temporal_stage_history"]),
                         ("image_tracked", "image_only", None, []))
        self.assertEqual((metrics["samples"]["windows_train"], metrics["samples"]["windows_validation"]), (0, 0))
        self.assertIn("unbenchmarked", metrics["aggregation_note"])
        model, loaded = load_bundle(self.base / "out" / "model-bundle", allow_smoke=True)
        self.assertIsInstance(model, DrowningTemporalModel)
        self.assertEqual(loaded["model_kind"], "image_tracked")

    def test_inputs_prepared_for_another_branch_are_refused(self):
        inputs = self.prepare_branch("image_only")
        with self.assertRaisesRegex(RuntimeError, "different branch"):
            run_train_candidate(inputs, device="cpu", options={"contract_test": True}, output_dir=self.base / "x",
                                archive_path=self.base / "x.zip", code_fingerprint="c" * 64)


class VideoFrameSamplingTests(unittest.TestCase):
    @staticmethod
    def track(media, person, labels, step_ms=100):
        return [{"media_id": media, "person_id": person, "observation_id": f"{media}-{person}-{i}", "frame_index": i,
                 "timestamp_ms": i * step_ms, "canonical_label": label} for i, label in enumerate(labels)]

    def test_first_frame_stride_and_every_label_change_are_kept(self):
        from drowning.data.crops import select_video_frames
        rows = self.track("v", "p1", ["swimming"] * 10 + ["drowning"] * 3 + ["swimming"] * 3)
        kept = [o["frame_index"] for o in select_video_frames(rows, stride_ms=500, max_per_video=100)]
        self.assertEqual(kept, [0, 5, 10, 13])  # first frame, stride frame 5, state changes at 10 and 13; frame 15 is only 200 ms after 13

    def test_tracks_are_sampled_independently_and_a_long_video_is_thinned_but_keeps_changes(self):
        from drowning.data.crops import select_video_frames
        rows = self.track("v", "a", ["swimming"] * 40 + ["drowning"] * 5) + self.track("v", "b", ["swimming"] * 3)
        kept = select_video_frames(rows, stride_ms=100, max_per_video=10)
        self.assertEqual(len(kept), 10)
        self.assertIn(40, [o["frame_index"] for o in kept if o["person_id"] == "a"])  # the swimming->drowning frame survives
        with self.assertRaises(ValueError):
            select_video_frames(rows, stride_ms=0, max_per_video=5)


class ImageOnlyWithVideoTests(TrainingInputFixture):
    def prepare_image_only(self, name="work"):
        from drowning.orchestration.kaggle_train import prepare_training_inputs
        return prepare_training_inputs(sources_dir=self.sources, reviews_dir=self.reviews, input_root=self.inputs,
                                       work_dir=self.base / name, recipes_dir=self.recipes, branch="image_only")

    def test_video_is_cut_into_frames_and_trained_before_the_images(self):
        inputs = self.prepare_image_only()
        self.assertEqual(inputs["blockers"], [])
        report = inputs["report"]
        self.assertEqual(report["training_phases"], ["video_frames", "images"])
        self.assertTrue(report["video_frame_crops_by_split"]["train"])
        self.assertEqual(report["temporal_windows_by_split"]["train"], {})
        rows = [json.loads(line) for line in inputs["paths"]["video_crop_manifest"].read_text(encoding="utf-8").splitlines()]
        self.assertTrue(rows and all(r["origin"] == "video_frame" and r["training_eligible"] for r in rows))
        splits_per_video = {}
        for row in rows:
            splits_per_video.setdefault(row["media_id"], set()).add(row["split"])
        self.assertTrue(all(len(v) == 1 for v in splits_per_video.values()))  # frames never leave their video's split
        metrics = run_train_candidate(inputs, device="cpu", options={"contract_test": True, "image_epochs": 1, "video_epochs": 1,
                                                                     "branch": "image_only"},
                                      output_dir=self.base / "out", archive_path=self.base / "out.zip", code_fingerprint="c" * 64)
        self.assertEqual(metrics["training_phases"], ["video_frames", "images"])
        self.assertEqual((len(metrics["video_frame_stage_history"]), len(metrics["image_stage_history"])), (1, 1))
        self.assertGreater(metrics["samples"]["video_frames_train"], 0)
        self.assertEqual(metrics["temporal_validation"], None)
        for key in ("video_frame_validation", "image_validation", "image_validation_after_stage_a", "video_frame_validation_after_video_phase"):
            self.assertEqual(len(metrics[key]["confusion_matrix"]), 3, key)

    def test_a_video_only_dataset_trains_the_frames_phase_alone(self):
        for path in list(self.sources.glob("pool_images.yaml")):
            path.unlink()
        inputs = self.prepare_image_only("w-video")
        self.assertEqual((inputs["blockers"], inputs["report"]["training_phases"]), ([], ["video_frames"]))
        metrics = run_train_candidate(inputs, device="cpu", options={"contract_test": True, "video_epochs": 1, "branch": "image_only"},
                                      output_dir=self.base / "o2", archive_path=self.base / "o2.zip", code_fingerprint="c" * 64)
        self.assertEqual((metrics["image_validation"], metrics["image_stage_history"], metrics["samples"]["image_train"]), (None, [], 0))

    def test_without_any_labeled_crops_the_run_is_blocked_with_a_clear_reason(self):
        for path in list(self.sources.glob("*.yaml")):
            path.unlink()
        self.assertIn("no_source_ready_for_training", self.prepare_image_only("w-none")["blockers"])


class ImageTrackedContractTests(unittest.TestCase):
    def test_manifest_validation_requires_aggregation_and_forbids_a_temporal_claim(self):
        self.assertEqual(validate_manifest(image_tracked_manifest()), [])
        self.assertTrue(validate_manifest({**image_tracked_manifest(), "aggregation": None}))
        self.assertTrue(validate_manifest({**image_tracked_manifest(), "temporal_trained": True}))
        self.assertTrue(validate_manifest({**image_tracked_manifest(), "model_kind": "mystery"}))
        untrained = {**approved_test_manifest(), "temporal_trained": False}
        self.assertIn("bundle has no trained temporal head", validate_manifest(untrained))

    def test_model_kind_is_derived_without_a_schema_change(self):
        def record(**kwargs):
            return ModelVersion(model_family="f", display_version="v", schema_version=1, class_keys=[], status="candidate", **kwargs)
        self.assertEqual(record(temporal_trained=True, completion_kind="candidate").model_kind, "temporal")
        self.assertEqual(record(temporal_trained=False, completion_kind="candidate", metrics={"model_kind": "image_tracked"}).model_kind,
                         "image_tracked")
        self.assertIsNone(record(temporal_trained=False, completion_kind="candidate", metrics={}).model_kind)
        self.assertIsNone(record(temporal_trained=False, completion_kind="synthetic_smoke", metrics={"model_kind": "image_tracked"}).model_kind)

    def test_branch_option_is_validated(self):
        self.assertEqual(normalize_options({"branch": "image_only"})["branch"], "image_only")
        self.assertEqual(normalize_options(None)["branch"], "image_only")  # the UI trains the image head only
        self.assertEqual(normalize_options({"branch": "temporal"})["branch"], "temporal")
        with self.assertRaises(ValueError):
            normalize_options({"branch": "video_only"})


class ImageTrackedInferenceTests(unittest.TestCase):
    def make(self, **aggregation):
        return PersonStateInference(FakeClassifier(), image_tracked_manifest(**aggregation), FixedPersonDetector(),
                                    model_version="candidate-it", input_size=32)

    def test_per_frame_until_the_track_is_ready_then_aggregated_by_tracker_id(self):
        pipeline = self.make()
        image = Image.new("RGB", (32, 32), "blue")
        records = [pipeline.process_frame(image, 100 * i, i)[0] for i in range(5)]
        self.assertEqual([r["prediction_branch"] for r in records], ["image_frame", "image_frame", "image_tracked", "image_tracked", "image_tracked"])
        self.assertEqual({r["track_id"] for r in records}, {records[0]["track_id"]})  # one person, one tracker ID
        self.assertTrue(all(r["label"] == "swimming" and r["frame_label"] == "swimming" for r in records))
        self.assertFalse(any(r["alert"] or r["temporal_available"] for r in records))
        self.assertEqual([r["track_frames"] for r in records], [1, 2, 3, 4, 5])

    def test_reset_stream_forgets_the_aggregated_history(self):
        pipeline = self.make()
        image = Image.new("RGB", (32, 32), "blue")
        for i in range(4):
            pipeline.process_frame(image, 100 * i, i)
        pipeline.reset_stream()
        self.assertEqual(pipeline.process_frame(image, 500, 0)[0]["track_frames"], 1)

    def test_sustained_drowning_scores_raise_the_alert_only_after_the_hold_time(self):
        class Drowning(FakeClassifier):
            def forward_images(self, images):
                import torch
                scores = torch.zeros((images.shape[0], 3), device=images.device)
                scores[:, 0] = 5.0
                return scores

        pipeline = PersonStateInference(Drowning(), image_tracked_manifest(tau_ms=100), FixedPersonDetector(),
                                        model_version="candidate-it", input_size=32)
        image = Image.new("RGB", (32, 32), "blue")
        records = [pipeline.process_frame(image, 100 * i, i)[0] for i in range(8)]
        self.assertFalse(records[2]["alert"])  # ready, but not yet held for min_alert_duration_ms
        self.assertTrue(records[-1]["alert"])
        self.assertEqual(records[-1]["label"], "drowning")

    def test_temporal_bundles_still_work_unchanged(self):
        pipeline = PersonStateInference(FakeClassifier(), approved_test_manifest(), FixedPersonDetector(),
                                        model_version="v", sequence_length=2, sample_period_ms=100, max_track_gap_ms=500, input_size=32)
        self.assertIsNone(pipeline.aggregator)
        record = pipeline.process_frame(Image.new("RGB", (32, 32)), 0, 0)[0]
        self.assertEqual(record["prediction_branch"], "image_warmup")


if __name__ == "__main__":
    unittest.main()
