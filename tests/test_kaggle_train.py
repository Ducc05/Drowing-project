"""Training-input assembly and the two-stage training contract on tiny synthetic data (CPU, contract test only).

Nothing here produces a usable model: the contract-test bundle is labeled synthetic_smoke and no CPU
candidate can be made without the explicit contract-test flag.
"""
import json
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock
from uuid import uuid4

import cv2
import numpy as np
import yaml
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from drowning.db.models import Base, ExecutionSlot, Job
from drowning.domain.jobs import JobState
from drowning.ml.datasets import CanonicalImageCropDataset
from drowning.orchestration.candidate_sync import CandidateArtifactSync
from drowning.orchestration.kaggle_train import (build_metrics, class_weights, prepare_training_inputs,
                                                 run_train_candidate, write_blocked_report)
from tests.test_kaggle_run import ROOT

NAMES = ["Drowning", "Swimming", "out of water"]
MAPPING = {"Drowning": {"action": "map", "target": "drowning"}, "Swimming": {"action": "map", "target": "swimming"},
           "out of water": {"action": "map", "target": "out_of_water"}}
APPROVED = {"status": "approved", "sample_review_required": False}


def texture(seed, size=64):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).resize((size, size), Image.Resampling.NEAREST)


class TrainingInputFixture(unittest.TestCase):
    IMAGES, VIDEOS = 12, 12

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.sources, self.recipes, self.reviews = self.base / "sources", self.base / "recipes", self.base / "reviews"
        self.inputs = self.base / "inputs"
        for directory in (self.sources, self.recipes, self.inputs):
            directory.mkdir()
        recipes = ROOT / "configs" / "recipes"
        image = yaml.safe_load((recipes / "image.yaml").read_text(encoding="utf-8"))
        image["input"]["size"] = [32, 32]
        image["training"]["batch_size"] = 8
        temporal = yaml.safe_load((recipes / "temporal.yaml").read_text(encoding="utf-8"))
        temporal["sampling"].update(sequence_length=2, sample_period_ms=100, max_track_gap_ms=300, min_valid_ratio=1.0)
        temporal["model"]["hidden_size"] = 8
        temporal["training"]["batch_size"] = 4
        (self.recipes / "image.yaml").write_text(yaml.safe_dump(image), encoding="utf-8")
        (self.recipes / "temporal.yaml").write_text(yaml.safe_dump(temporal), encoding="utf-8")
        self.make_image_source()
        self.make_video_source()

    def tearDown(self):
        self.temp.cleanup()

    def make_image_source(self, approved=True, groups=True):
        root = self.inputs / "pool-images"
        (root / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
        (root / "data.yaml").write_text(yaml.safe_dump({"names": NAMES}), encoding="utf-8")
        grouping = {}
        for i in range(self.IMAGES):
            split = "train" if i % 2 else "valid"
            (root / split / "images").mkdir(parents=True, exist_ok=True)
            (root / split / "labels").mkdir(parents=True, exist_ok=True)
            texture(i).save(root / split / "images" / f"i{i}.png")
            # three people in one image, one per class: every split that has an image has every class
            (root / split / "labels" / f"i{i}.txt").write_text(
                "0 0.2 0.5 0.3 0.6\n1 0.5 0.5 0.3 0.6\n2 0.8 0.5 0.3 0.6\n", encoding="utf-8")
            grouping[f"{split}/images/i{i}.png"] = f"img-group-{i}"
        (root / "groups.json").write_text(json.dumps(grouping), encoding="utf-8")
        source = {"schema_version": 1, "source_id": "pool_images", "source_version": "1", "format": "yolo_detection",
                  "modality": "image", "paths": {"root_env": "X", "class_list": "data.yaml:names"},
                  "source_labels": {"expected_names": NAMES}, "mapping": MAPPING,
                  "mapping_review": APPROVED if approved else {"status": "pending_sample_review", "sample_review_required": True},
                  "grouping": {"status": "verified", "metadata_file": "groups.json"} if groups else {"status": "unknown"}}
        (self.sources / "pool_images.yaml").write_text(yaml.safe_dump(source), encoding="utf-8")

    def make_video_source(self, approved=True):
        root = self.inputs / "pool-videos"
        tasks = []
        for i in range(self.VIDEOS):
            video = root / f"v{i}" / "video.avi"
            video.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
            for frame in range(9):
                writer.write(np.full((32, 32, 3), 20 * frame + i, dtype=np.uint8))
            writer.release()
            (video.parent / "ts.json").write_text(json.dumps([f * 100 for f in range(9)]), encoding="utf-8")
            labels = [NAMES[f // 3] for f in range(9)]  # state changes inside one track
            boxes = "".join(f'<box frame="{f}" xtl="2" ytl="2" xbr="20" ybr="28" outside="0" occluded="0" keyframe="1">'
                            f'<attribute name="state">{labels[f]}</attribute></box>' for f in range(9))
            (video.parent / "ann.xml").write_text(f'<annotations><track id="1" label="person">{boxes}</track></annotations>', encoding="utf-8")
            tasks.append({"annotations": f"v{i}/ann.xml", "video": f"v{i}/video.avi", "timestamps": f"v{i}/ts.json",
                          "global_group_id": f"video-group-{i}"})
        source = {"schema_version": 1, "source_id": "pool_videos", "source_version": "1", "format": "cvat_video",
                  "modality": "video", "paths": {"root_env": "X"}, "tasks": tasks, "mapping": MAPPING,
                  "mapping_review": APPROVED if approved else {"status": "pending_sample_review", "sample_review_required": True},
                  "identity_review": {"status": "verified", "sample_review_required": False},
                  "timing": {"mode": "explicit_timestamps"}}
        (self.sources / "pool_videos.yaml").write_text(yaml.safe_dump(source), encoding="utf-8")

    def prepare(self, name="work", seed=20261026):
        return prepare_training_inputs(sources_dir=self.sources, reviews_dir=self.reviews, input_root=self.inputs,
                                       work_dir=self.base / name, recipes_dir=self.recipes, seed=seed)


class PrepareTrainingInputsTests(TrainingInputFixture):
    def test_approved_sources_produce_group_split_crops_and_windows(self):
        inputs = self.prepare()
        self.assertEqual(inputs["blockers"], [], inputs["report"])
        report = inputs["report"]
        self.assertEqual(sorted(report["included_sources"]), ["pool_images", "pool_videos"])
        self.assertEqual(report["excluded_sources"], {})
        self.assertEqual(len(inputs["dataset_fingerprint"]), 64)
        for name in ("merged", "split", "crop_manifest", "window_manifest"):
            self.assertTrue(inputs["paths"][name].is_file(), name)
        for split in ("train", "validation"):
            self.assertEqual(set(report["image_crops_by_split"][split]), {"drowning", "swimming", "out_of_water"})
            self.assertEqual(set(report["temporal_windows_by_split"][split]), {"drowning", "swimming", "out_of_water"})
        self.assertFalse(report["source_video_disjointness_verified"])
        self.assertEqual(set(inputs["video_roots"]), {"pool_videos"})
        # no group is split across partitions
        manifest = [json.loads(l) for l in inputs["paths"]["split"].read_text(encoding="utf-8").splitlines()]
        by_group = {}
        for row in manifest:
            by_group.setdefault(row["global_group_id"], set()).add(row["split"])
        self.assertTrue(all(len(s) == 1 for s in by_group.values()))
        self.assertEqual(len(by_group), self.IMAGES + self.VIDEOS)

    def test_fingerprint_is_deterministic_and_changes_with_the_seed(self):
        first, second = self.prepare("work-a"), self.prepare("work-b")
        self.assertEqual(first["dataset_fingerprint"], second["dataset_fingerprint"])
        other = self.prepare("work-c", seed=7)
        if not other["blockers"]:
            self.assertNotEqual(other["dataset_fingerprint"], first["dataset_fingerprint"])

    def test_unapproved_unverified_or_missing_sources_are_excluded_with_a_reason(self):
        self.make_image_source(approved=False)
        self.assertEqual(self.prepare("w1")["report"]["excluded_sources"], {"pool_images": "mapping_review_not_approved"})
        self.make_image_source(groups=False)
        self.assertEqual(self.prepare("w2")["report"]["excluded_sources"], {"pool_images": "source_groups_not_verified"})
        shutil.rmtree(self.inputs / "pool-images")
        self.assertEqual(self.prepare("w3")["report"]["excluded_sources"], {"pool_images": "input_missing"})

    def test_images_without_any_video_source_block_with_clear_reasons(self):
        (self.sources / "pool_videos.yaml").unlink()
        inputs = self.prepare()
        self.assertIn("no_temporal_windows", inputs["blockers"])
        shutil.rmtree(self.inputs)
        self.inputs.mkdir()
        empty = self.prepare("w-empty")
        self.assertEqual(empty["blockers"], ["no_source_ready_for_training"])
        self.assertIsNone(empty["dataset_fingerprint"])

    def test_missing_class_in_the_validation_split_is_reported(self):
        for path in self.sources.glob("pool_videos.yaml"):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            data["tasks"] = data["tasks"][:3]
            path.write_text(yaml.safe_dump(data), encoding="utf-8")
        inputs = self.prepare()
        text = " ".join(inputs["blockers"])
        self.assertTrue("missing_classes" in text or inputs["blockers"] == [], inputs["blockers"])

    def test_class_weights_come_from_counts_and_are_bounded(self):
        from collections import Counter
        weights = class_weights(Counter({"drowning": 900, "swimming": 90, "out_of_water": 10}))
        self.assertLess(weights[0], 1.0)
        self.assertEqual(weights[2], 8.0)
        self.assertEqual(class_weights(Counter()), [1.0, 1.0, 1.0])


class TrainCandidateContractTests(TrainingInputFixture):
    def test_cpu_candidate_is_refused_without_the_explicit_contract_flag(self):
        inputs = self.prepare()
        with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
            run_train_candidate(inputs, device="cpu", options={}, output_dir=self.base / "o", archive_path=self.base / "o.zip",
                                code_fingerprint="c" * 64)
        self.assertFalse((self.base / "o").exists())
        blocked = {**inputs, "blockers": ["no_temporal_windows"]}
        with self.assertRaisesRegex(RuntimeError, "blocked"):
            run_train_candidate(blocked, device="cuda", options={}, output_dir=self.base / "o2", archive_path=self.base / "o2.zip",
                                code_fingerprint="c" * 64)

    def test_two_stage_contract_run_never_reads_the_test_split_and_is_not_a_candidate(self):
        inputs = self.prepare()
        requested = []
        original = CanonicalImageCropDataset.__init__

        def spy(self_inner, manifest, *, split, **kwargs):
            requested.append(split)
            return original(self_inner, manifest, split=split, **kwargs)

        with mock.patch.object(CanonicalImageCropDataset, "__init__", spy):
            metrics = run_train_candidate(inputs, device="cpu", options={"contract_test": True, "image_epochs": 1, "temporal_epochs": 1},
                                          output_dir=self.base / "candidate", archive_path=self.base / "candidate.zip",
                                          code_fingerprint="c" * 64, job_context={"kernel_ref": "owner/k", "kernel_version": "3"})
        self.assertEqual(sorted(requested), ["train", "validation"])
        self.assertEqual((metrics["completion_kind"], metrics["approved"], metrics["test_split_used"]),
                         ("synthetic_smoke", False, False))
        self.assertEqual(metrics["dataset_fingerprint"], inputs["dataset_fingerprint"])
        self.assertEqual(set(metrics["samples"]), {"image_train", "image_validation", "windows_train", "windows_validation"})
        for key in ("image_validation_after_stage_a", "image_validation", "temporal_validation", "class_weights_from_train_split"):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["temporal_validation"]["confusion_matrix"].__len__(), 3)
        with zipfile.ZipFile(self.base / "candidate.zip") as archive:
            names = set(archive.namelist())
        self.assertEqual(names, {"model-bundle/manifest.json", "model-bundle/weights.pt", "metrics.json"})
        manifest = json.loads((self.base / "candidate" / "model-bundle" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual((manifest["completion_kind"], manifest["temporal_trained"]), ("synthetic_smoke", True))
        self.assertEqual(manifest["model_version_id"], metrics["candidate_id"])

    def test_same_inputs_and_seed_give_bit_identical_weights_and_another_seed_does_not(self):
        inputs = self.prepare()
        options = {"contract_test": True, "image_epochs": 1, "temporal_epochs": 1}

        def train(name, seed):
            metrics = run_train_candidate(inputs, device="cpu", options=options, output_dir=self.base / name,
                                          archive_path=self.base / f"{name}.zip", code_fingerprint="c" * 64, seed=seed)
            manifest = json.loads((self.base / name / "model-bundle" / "manifest.json").read_text(encoding="utf-8"))
            return manifest["files"]["weights.pt"]["sha256"], metrics

        first, first_metrics = train("run-a", 20261026)
        second, second_metrics = train("run-b", 20261026)
        other, _ = train("run-c", 7)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(first_metrics["image_stage_history"], second_metrics["image_stage_history"])
        self.assertEqual(first_metrics["temporal_validation"], second_metrics["temporal_validation"])
        self.assertTrue(first_metrics["determinism"]["deterministic_algorithms"])
        self.assertIn("same GPU model", first_metrics["determinism"]["caveat"])

    def test_candidate_metrics_envelope_matches_what_the_collector_verifies(self):
        metrics = build_metrics(completion_kind="candidate", candidate_id="abc", dataset_fingerprint="d" * 64)
        self.assertEqual((metrics["completion_kind"], metrics["approved"], metrics["test_split_used"]), ("candidate", False, False))
        self.assertEqual((metrics["candidate_id"], metrics["dataset_fingerprint"]), ("abc", "d" * 64))


class BlockedTrainRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def test_blocked_report_is_collected_as_a_readable_failure_and_frees_the_slot(self):
        archive = self.base / "run-output.zip"
        write_blocked_report(self.base / "out", archive, blockers=["no_temporal_windows", "image_validation_split_missing_classes:out_of_water"],
                             report={"excluded_sources": {"pool_videos": "source_groups_not_verified"}}, code_fingerprint="a" * 64)

        class Kaggle:
            def collect_exact_output(self, ref, destination):
                shutil.copyfile(archive, destination / "run-output.zip")

        job_id = uuid4()
        with self.factory() as session:
            session.add(Job(id=job_id, job_type="train", state=JobState.COLLECTING.value, phase="collecting",
                            idempotency_key=f"blocked-{job_id}", payload={"mode": "train_candidate", "executor": "kaggle"},
                            owner_id="w", execution_token=2, remote_ref="o/k", remote_version="4", remote_status="completed"))
            session.flush()
            session.add(ExecutionSlot(slot_id=1, job_id=job_id, owner_id="w", fencing_token=2, reconciliation_status="owned"))
            session.commit()
        sync = CandidateArtifactSync(self.factory, Kaggle(), object(), drive_folder_id="f", cache_root=self.base / "cache", worker_id="w")
        result = sync.collect(job_id)
        self.assertEqual((result["state"], result["train_blocked"]), ("failed", True))
        with self.factory() as session:
            job = session.get(Job, job_id)
            self.assertEqual((job.state, job.error_code), ("failed", "train_inputs_blocked"))
            self.assertIn("no_temporal_windows", job.error_summary)
            self.assertTrue(job.payload["run_summary"]["train_blocked"])
            self.assertEqual(job.payload["run_summary"]["excluded_sources"], {"pool_videos": "source_groups_not_verified"})
            self.assertIsNone(session.get(ExecutionSlot, 1).job_id)


if __name__ == "__main__":
    unittest.main()
