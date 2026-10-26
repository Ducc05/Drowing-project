"""Mapping/grouping review records, review sheets and quality report on a small synthetic YOLO export."""
import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

import httpx

import numpy as np
import yaml
from PIL import Image

from drowning.api.app import create_app
from drowning.config import Settings
from drowning.data.adapters import convert_to_jsonl
from drowning.data.quality import build_quality_report, cluster_near_duplicates, dhash
from drowning.data.registry import SourceConfigError, load_source
from drowning.data.review import (ReviewError, apply_reviews, load_grouping_review, load_mapping_review,
                                  mapping_fingerprint, record_grouping_review, record_mapping_review, review_status)
from drowning.data.samples import build_review_sheets

NAMES = ["Drowning", "Swimming", "out of water"]


def texture(seed, size=64):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((size, size), Image.Resampling.NEAREST)


def descriptor(root_env="FIXTURE_ROOT", mapping=None):
    return {"schema_version": 1, "source_id": "fixture_pool", "source_version": "1", "format": "yolo_detection",
            "modality": "image", "paths": {"root_env": root_env, "class_list": "data.yaml:names"},
            "mapping": mapping or {"Drowning": {"action": "map", "target": "drowning"},
                                   "Swimming": {"action": "map", "target": "swimming"},
                                   "out of water": {"action": "map", "target": "out_of_water"}},
            "mapping_review": {"status": "pending_sample_review", "sample_review_required": True}}


class ReviewFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.root, self.work, self.reviews = base / "raw", base / "work", base / "reviews"
        self.root.mkdir()
        (self.root / "data.yaml").write_text(yaml.safe_dump({"names": NAMES}), encoding="utf-8")
        # Images 0-11 are distinct textures; train/dup0 is a near-duplicate of valid/img0 (leakage).
        plan = [("train", i, texture(i), i % 3) for i in range(12)]
        plan += [("valid", 100 + i, texture(200 + i), i % 3) for i in range(6)]
        plan.append(("valid", 999, texture(0), 0))
        for split, number, image, class_id in plan:
            (self.root / split / "images").mkdir(parents=True, exist_ok=True)
            (self.root / split / "labels").mkdir(parents=True, exist_ok=True)
            image.save(self.root / split / "images" / f"img{number}.png")
            (self.root / split / "labels" / f"img{number}.txt").write_text(f"{class_id} 0.5 0.5 0.6 0.6\n", encoding="utf-8")
        self.source = descriptor()
        import os
        os.environ["FIXTURE_ROOT"] = str(self.root)
        self.addCleanup(os.environ.pop, "FIXTURE_ROOT", None)
        self.canonical = self.work / "canonical.jsonl"
        convert_to_jsonl(self.source, self.root, self.canonical)

    def tearDown(self):
        self.temp.cleanup()


class NearDuplicateTests(unittest.TestCase):
    def test_clusters_find_close_hashes_and_keep_far_ones_apart(self):
        hashes = {"a": 0b1010 << 40, "b": (0b1010 << 40) ^ 0b11, "c": 0xFFFF_FFFF_0000_0000, "d": 5}
        clusters = cluster_near_duplicates(hashes, 2)
        self.assertIn(["a", "b"], clusters)
        self.assertIn(["c"], clusters)
        self.assertEqual(sum(len(c) for c in clusters), 4)
        with self.assertRaises(ValueError):
            cluster_near_duplicates(hashes, -1)

    def test_pigeonhole_bucketing_matches_brute_force(self):
        rng = np.random.default_rng(1)
        hashes = {f"k{i}": int(rng.integers(0, 2**63)) for i in range(60)}
        for key in list(hashes)[:20]:  # add noisy copies
            hashes[key + "x"] = hashes[key] ^ (1 << int(rng.integers(0, 60)))
        for distance in (0, 1, 3):
            expected = {frozenset(c) for c in _brute_force(hashes, distance)}
            self.assertEqual({frozenset(c) for c in cluster_near_duplicates(hashes, distance)}, expected)


def _brute_force(hashes, distance):
    parent = {k: k for k in hashes}

    def find(k):
        while parent[k] != k:
            k = parent[k]
        return k

    keys = list(hashes)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            if bin(hashes[left] ^ hashes[right]).count("1") <= distance:
                parent[find(left)] = find(right)
    groups = {}
    for k in keys:
        groups.setdefault(find(k), []).append(k)
    return list(groups.values())


class MappingReviewTests(ReviewFixture):
    def approve(self, **overrides):
        args = dict(reviewer="Dung", confirmed_labels=list(NAMES), samples_viewed={n: 24 for n in NAMES},
                    notes={}, reviews_dir=self.reviews)
        args.update(overrides)
        return record_mapping_review(self.source, **args)

    def test_nothing_is_approved_until_a_named_reviewer_confirms_every_mapped_label(self):
        self.assertIsNone(load_mapping_review(self.source, self.reviews))
        self.assertEqual(apply_reviews(self.source, self.reviews)["mapping_review"]["status"], "pending_sample_review")
        for bad, message in ((dict(reviewer=" "), "reviewer"), (dict(confirmed_labels=["Drowning"]), "not confirmed"),
                             (dict(samples_viewed={"Drowning": 24, "Swimming": 3, "out of water": 24}), "at least 10")):
            with self.assertRaisesRegex(ReviewError, message):
                self.approve(**bad)
        self.assertFalse(self.reviews.exists())

    def test_approval_overlays_source_and_marks_observations_reviewed(self):
        self.approve()
        approved = apply_reviews(self.source, self.reviews)
        self.assertEqual(approved["mapping_review"]["status"], "approved")
        self.assertFalse(approved["mapping_review"]["sample_review_required"])
        output = self.work / "reviewed.jsonl"
        convert_to_jsonl(approved, self.root, output)
        statuses = {json.loads(l)["annotation_status"] for l in output.read_text(encoding="utf-8").splitlines()
                    if '"entity": "observation"' in l}
        self.assertEqual(statuses, {"reviewed"})
        pending = {json.loads(l)["annotation_status"] for l in self.canonical.read_text(encoding="utf-8").splitlines()
                   if '"entity": "observation"' in l}
        self.assertEqual(pending, {"source_unreviewed"})

    def test_changing_the_mapping_invalidates_the_approval(self):
        self.approve()
        changed = descriptor(mapping={**self.source["mapping"], "Swimming": {"action": "map", "target": "out_of_water"}})
        self.assertNotEqual(mapping_fingerprint(changed), mapping_fingerprint(self.source))
        self.assertIsNone(load_mapping_review(changed, self.reviews))
        self.assertEqual(review_status(changed, self.reviews)["mapping_review"], "stale")
        self.assertEqual(apply_reviews(changed, self.reviews)["mapping_review"]["status"], "pending_sample_review")

    def test_ambiguous_source_names_need_a_written_definition(self):
        mapping = {"normal": {"action": "map", "target": "swimming"}, "person": {"action": "review"},
                   "Drowning": {"action": "map", "target": "drowning"}}
        source = descriptor(mapping=mapping)
        kwargs = dict(reviewer="Dung", confirmed_labels=["normal", "Drowning"],
                      samples_viewed={"normal": 30, "Drowning": 30}, reviews_dir=self.reviews)
        with self.assertRaisesRegex(ReviewError, "written definition"):
            record_mapping_review(source, **kwargs)
        record = record_mapping_review(source, notes={"normal": "Only people inside the pool water, not on the deck."}, **kwargs)
        self.assertEqual(record["status"], "approved")

    def test_stored_review_file_is_loaded_by_load_source_from_project_layout(self):
        project = Path(self.temp.name) / "project"
        (project / "configs" / "datasets").mkdir(parents=True)
        config = project / "configs" / "datasets" / "fixture.yaml"
        config.write_text(yaml.safe_dump(self.source), encoding="utf-8")
        self.assertEqual(load_source(config)["mapping_review"]["status"], "pending_sample_review")
        record_mapping_review(self.source, reviewer="Dung", confirmed_labels=NAMES,
                              samples_viewed={n: 12 for n in NAMES}, reviews_dir=project / "reviews")
        self.assertEqual(load_source(config)["mapping_review"]["status"], "approved")
        elsewhere = Path(self.temp.name) / "loose.yaml"
        elsewhere.write_text(yaml.safe_dump(self.source), encoding="utf-8")
        self.assertEqual(load_source(elsewhere)["mapping_review"]["status"], "pending_sample_review")


class SamplesAndQualityTests(ReviewFixture):
    def test_review_sheets_cover_each_label_and_are_idempotent_per_mapping(self):
        output = self.work / "sheets"
        manifest = build_review_sheets(self.source, self.canonical, self.root, output, per_label=10)
        keys = {sheet["key"] for sheet in manifest["sheets"]}
        self.assertEqual(keys, {f"{name}--{kind}" for name in ("drowning", "swimming", "out-of-water")
                                for kind in ("random", "smallest")})
        for sheet in manifest["sheets"]:
            self.assertTrue((output / sheet["file"]).is_file())
            self.assertEqual(sheet["sample_count"], min(10, sheet["total_for_label"]))
        self.assertEqual(build_review_sheets(self.source, self.canonical, self.root, output, per_label=10), manifest)
        with self.assertRaises(SourceConfigError):
            build_review_sheets(descriptor(mapping={"Drowning": {"action": "review"}}), self.canonical, self.root,
                                output, per_label=10)
        with self.assertRaises(ValueError):
            build_review_sheets(self.source, self.canonical, self.root, self.work / "x", per_label=3)
        with self.assertRaises(SourceConfigError):
            build_review_sheets(self.source, self.canonical, self.root, self.root / "inside-raw", per_label=10)

    def test_quality_report_counts_labels_and_finds_cross_split_near_duplicates(self):
        report, proposed = build_quality_report(self.source, self.canonical, self.root, reviews_dir=self.reviews)
        self.assertEqual(report["counts"]["media"], 19)
        labels = {l["original_label"]: l for l in report["labels"]}
        self.assertEqual(sum(l["count"] for l in labels.values()), 19)
        self.assertEqual(labels["Drowning"]["by_original_split"], {"train": 4, "valid": 3})
        near = report["near_duplicates"]
        self.assertTrue(near["checked"])
        self.assertGreaterEqual(near["clusters_crossing_original_splits"], 1)
        self.assertEqual(proposed["train/images/img0.png"], proposed["valid/images/img999.png"])
        self.assertEqual(len(proposed), 19)
        readiness = report["image_training_readiness"]
        self.assertFalse(readiness["ready_for_split_and_crop_building"])
        self.assertIn("mapping_review_not_approved", readiness["blockers"])
        self.assertIn("source_groups_not_verified", readiness["blockers"])

    def test_skipping_pixels_reports_unchecked_and_proposes_nothing(self):
        report, proposed = build_quality_report(self.source, self.canonical, None)
        self.assertFalse(report["near_duplicates"]["checked"])
        self.assertEqual(proposed, {})

    def test_verified_groups_flow_into_conversion_and_clear_the_group_blocker(self):
        _, proposed = build_quality_report(self.source, self.canonical, self.root)
        with self.assertRaisesRegex(ReviewError, "written basis"):
            record_grouping_review(self.source, reviewer="Dung", by_path=proposed, method="m", evidence={},
                                   note="short", reviews_dir=self.reviews)
        record_grouping_review(self.source, reviewer="Dung", by_path=proposed, method="near_duplicate_clusters_user_reviewed",
                               evidence={}, note="Checked cross-split example pairs on the review sheet.",
                               reviews_dir=self.reviews)
        self.assertIsNotNone(load_grouping_review(self.source, self.reviews))
        grouped = apply_reviews(self.source, self.reviews)
        output = self.work / "grouped.jsonl"
        convert_to_jsonl(grouped, self.root, output)
        media = [json.loads(l) for l in output.read_text(encoding="utf-8").splitlines() if '"entity": "media"' in l]
        self.assertTrue(all(m["grouping_status"] == "verified" and m["global_group_id"] for m in media))
        by_path = {m["raw_relative_path"]: m["global_group_id"] for m in media}
        self.assertEqual(by_path["train/images/img0.png"], by_path["valid/images/img999.png"])
        report, _ = build_quality_report(grouped, output, None, reviews_dir=self.reviews)
        self.assertTrue(report["image_training_readiness"]["source_groups_verified"])

    def test_tampered_group_record_is_ignored(self):
        _, proposed = build_quality_report(self.source, self.canonical, self.root)
        record_grouping_review(self.source, reviewer="Dung", by_path=proposed, method="m", evidence={},
                               note="Checked cross-split example pairs on the review sheet.", reviews_dir=self.reviews)
        path = next(self.reviews.glob("*grouping-review.json"))
        data = json.loads(path.read_text(encoding="utf-8"))
        data["by_path"]["train/images/img0.png"] = "somewhere-else"
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertIsNone(load_grouping_review(self.source, self.reviews))

    def test_dhash_is_stable_for_a_resized_copy(self):
        path = Path(self.temp.name) / "copy.png"
        texture(0, size=128).save(path)
        distance = bin(dhash(self.root / "train/images/img0.png") ^ dhash(path)).count("1")
        self.assertLessEqual(distance, 8)


def call(app, method, path, **kwargs):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(run())


class ReviewApiTests(ReviewFixture):
    def setUp(self):
        super().setUp()
        project = Path(self.temp.name) / "project"
        (project / "configs" / "datasets").mkdir(parents=True)
        (project / "configs" / "datasets" / "fixture.yaml").write_text(yaml.safe_dump(self.source), encoding="utf-8")
        self.project = project
        settings = Settings.from_env({"INPUT_CACHE_DIR": str(project / "local-cache" / "inputs")})
        self.app = create_app(settings=settings, config_dir=project / "configs" / "datasets",
                              env_file=project / "missing.env")
        self.base = "/api/v1/sources/fixture_pool/review"

    def prepare(self):
        self.assertEqual(call(self.app, "POST", self.base + "/prepare").status_code, 200)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = call(self.app, "GET", self.base).json()
            if status["task"]["state"] in {"done", "failed"}:
                return status
            time.sleep(0.1)
        self.fail("review preparation did not finish")

    def test_prepare_then_approve_mapping_and_grouping_through_http(self):
        initial = call(self.app, "GET", self.base).json()
        self.assertEqual((initial["mapping_review"], initial["grouping_review"], initial["sheets"]), ("pending", "pending", None))
        premature = call(self.app, "POST", self.base + "/approve-mapping", json={
            "reviewer": "Dung", "confirmed_labels": NAMES, "samples_viewed": {n: 24 for n in NAMES}})
        self.assertEqual(premature.status_code, 409)

        status = self.prepare()
        self.assertEqual(status["task"]["state"], "done", status["task"])
        self.assertEqual(len(status["sheets"]), 6)
        self.assertEqual(status["quality"]["counts"]["media"], 19)
        sheet = call(self.app, "GET", self.base + "/sheets/drowning--random")
        self.assertEqual((sheet.status_code, sheet.headers["content-type"]), (200, "image/jpeg"))
        self.assertEqual(call(self.app, "GET", self.base + "/sheets/..%2Fsamples").status_code, 404)

        bad = call(self.app, "POST", self.base + "/approve-mapping", json={
            "reviewer": "Dung", "confirmed_labels": ["Drowning"], "samples_viewed": {"Drowning": 24}})
        self.assertEqual(bad.status_code, 422)
        good = call(self.app, "POST", self.base + "/approve-mapping", json={
            "reviewer": "Dung", "confirmed_labels": NAMES, "samples_viewed": {n: 24 for n in NAMES}})
        self.assertEqual(good.status_code, 200, good.text)
        listed = call(self.app, "GET", "/api/v1/sources").json()["sources"][0]
        self.assertEqual(listed["mapping_review_status"], "approved")

        no_note = call(self.app, "POST", self.base + "/approve-grouping", json={"reviewer": "Dung", "note": "ok"})
        self.assertEqual(no_note.status_code, 422)
        grouped = call(self.app, "POST", self.base + "/approve-grouping", json={
            "reviewer": "Dung", "note": "Looked at the cross-split pairs; they are the same photo."})
        self.assertEqual(grouped.status_code, 200, grouped.text)
        self.assertFalse(grouped.json()["source_video_disjointness_verified"])
        after = self.prepare()
        self.assertEqual(after["grouping_review"], "verified")
        self.assertTrue(after["quality"]["image_training_readiness"]["ready_for_split_and_crop_building"])

    def test_unknown_source_and_missing_root_are_reported(self):
        self.assertEqual(call(self.app, "GET", "/api/v1/sources/nope/review").status_code, 404)
        import os
        os.environ.pop("FIXTURE_ROOT")
        self.assertEqual(call(self.app, "POST", self.base + "/prepare").status_code, 409)


if __name__ == "__main__":
    unittest.main()
