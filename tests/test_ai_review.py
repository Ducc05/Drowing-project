"""Rule-based review: automatic rules, evidence bundle, AI verdict thresholds and the audit trail."""
import asyncio
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import httpx
import yaml

from drowning.api.app import create_app
from drowning.api.review import ReviewConflict, SourceReviewService
from drowning.config import Settings
from drowning.data.review import load_grouping_review, load_mapping_review
from drowning.data.review_rules import load_policy
from drowning.data.registry import load_source
from tests.test_review import NAMES, ReviewFixture

ROOT = Path(__file__).resolve().parents[1]


def relaxed_policy(base: Path, **overrides) -> Path:
    """The shipped thresholds suit a 10k-image export, so tiny fixtures run with smaller count minimums."""
    policy = load_policy()
    policy["auto_rules"]["min_boxes_per_mapped_label"]["min"] = 3
    policy["auto_rules"]["min_class_share"]["min"] = 0.01
    policy["auto_rules"]["largest_cluster_fraction"] = {"severity": "warn", "max": 0.9, "fail_above": 0.99}
    for key, value in overrides.items():
        policy["auto_rules"][key] = value
    path = base / "policy.yaml"
    path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    return path


class AiReviewTests(ReviewFixture):
    def setUp(self):
        super().setUp()
        self.project = Path(self.temp.name) / "project"
        (self.project / "configs" / "datasets").mkdir(parents=True)
        (self.project / "configs" / "datasets" / "fixture.yaml").write_text(yaml.safe_dump(self.source), encoding="utf-8")
        self.reviews = self.project / "reviews"
        self.service = self.make_service(relaxed_policy(self.project))
        self.service.prepare("fixture_pool", background=False)
        self.assertEqual(self.service._tasks["fixture_pool"]["state"], "done", self.service._tasks)

    def make_service(self, policy):
        return SourceReviewService(self.project / "configs" / "datasets", reviews_dir=self.reviews,
                                   work_dir=self.project / "local-cache", policy_path=policy)

    def verdict(self, **changes):
        evidence = self.service.evidence("fixture_pool")
        source = self.service.source("fixture_pool")
        from drowning.data.review import mapping_fingerprint
        good = {"schema_version": 1, "source_id": "fixture_pool", "mapping_hash": mapping_fingerprint(source),
                "evidence_sha256": evidence["evidence_sha256"],
                "reviewer": {"kind": "ai", "name": "Test Reviewer 1", "delegated_by": "user"}, "viewed_images": True,
                "mapping": {"labels": {name: {"samples_viewed": 24, "visible_person_fraction": 0.95,
                                              "label_consistent_fraction": 0.92, "notes": "fine"} for name in NAMES}},
                "grouping": {"cross_split_pairs_viewed": 12, "same_scene_fraction": 1.0,
                             "largest_cluster_same_scene_fraction": 0.95, "notes": "Rows are the same photo recoloured, checked by eye."},
                "decision": {"mapping": "approve", "grouping": "approve"}, "limitations": "none"}
        return copy.deepcopy({**good, **changes})

    # -- automatic rules -------------------------------------------------------------------
    def test_auto_rules_pass_on_clean_data_and_block_on_each_failing_rule(self):
        rules = self.service.auto_rules("fixture_pool")
        self.assertEqual((rules["outcome"], rules["failed"]), ("pass", []))
        self.assertEqual({r["id"] for r in rules["rules"]}, set(load_policy()["auto_rules"]))
        strict = self.make_service(relaxed_policy(self.project, min_boxes_per_mapped_label={"severity": "fail", "min": 1000}))
        self.assertEqual(strict.auto_rules("fixture_pool")["failed"], ["min_boxes_per_mapped_label"])

    def test_ambiguous_label_names_block_until_the_descriptor_defines_them(self):
        path = self.project / "configs" / "datasets" / "fixture.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data["mapping"]["normal"] = {"action": "map", "target": "swimming"}
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.service.prepare("fixture_pool", background=False)  # a new mapping is a new review
        self.assertIn("ambiguous_label_names", self.service.auto_rules("fixture_pool")["failed"])
        data["mapping"]["normal"]["definition"] = "Only people in the pool water, never on the deck or poolside."
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        self.service.prepare("fixture_pool", background=False)
        self.assertNotIn("ambiguous_label_names", self.service.auto_rules("fixture_pool")["failed"])

    # -- evidence bundle -------------------------------------------------------------------
    def test_evidence_bundle_holds_every_image_the_reviewer_needs_and_is_stable(self):
        first = self.service.evidence("fixture_pool")
        folder = Path(first["evidence_dir"])
        for name in ("README_AI_REVIEW.md", "auto-rules.json", "verdict-template.json", "quality.json",
                     "mapping-sheets/drowning--random.jpg", "group-sheets/groups--cross-split.jpg", "group-sheets/groups.json"):
            self.assertTrue((folder / name).is_file(), name)
        readme = (folder / "README_AI_REVIEW.md").read_text(encoding="utf-8")
        for needle in ("must actually LOOK", "uncertain", "0.8", "does NOT prove"):
            self.assertIn(needle, readme)
        self.assertEqual(self.service.evidence("fixture_pool")["evidence_sha256"], first["evidence_sha256"])
        template = json.loads((folder / "verdict-template.json").read_text(encoding="utf-8"))
        self.assertFalse(template["viewed_images"])
        self.assertEqual(template["evidence_sha256"], first["evidence_sha256"])

    # -- verdicts --------------------------------------------------------------------------
    def test_a_good_verdict_records_ai_approvals_with_evidence_and_policy_version(self):
        outcome = self.service.submit_verdict("fixture_pool", self.verdict())
        self.assertEqual(outcome["decisions"], {"mapping": "approve", "grouping": "approve"})
        self.assertEqual(outcome["recorded"], {"mapping": "approved", "grouping": "verified"})
        self.assertEqual(outcome["ask_user"], [])
        source = load_source(self.project / "configs" / "datasets" / "fixture.yaml", self.reviews)
        mapping, grouping = load_mapping_review(source, self.reviews), load_grouping_review(source, self.reviews)
        for record in (mapping, grouping):
            self.assertEqual((record["reviewer"], record["reviewer_kind"], record["delegated_by"]), ("ai:Test Reviewer 1", "ai", "user"))
            self.assertEqual(record["evidence_sha256"], outcome["evidence_sha256"])
            self.assertEqual(record["rules_version"], outcome["rules_version"])
        self.assertIn("source-video disjointness not verified", grouping["note"])
        trail = list((self.reviews / "verdicts").glob("fixture_pool@1.*.ai.json"))
        self.assertEqual(len(trail), 1)
        self.assertEqual(json.loads(trail[0].read_text(encoding="utf-8"))["outcome"]["recorded"]["mapping"], "approved")
        self.assertEqual(self.service.status("fixture_pool")["mapping_reviewer_kind"], "ai")

    def test_numbers_below_the_threshold_override_an_approve_and_record_nothing(self):
        weak = self.verdict()
        weak["mapping"]["labels"]["Drowning"]["label_consistent_fraction"] = 0.5
        weak["grouping"]["same_scene_fraction"] = 0.6
        outcome = self.service.submit_verdict("fixture_pool", weak)
        self.assertEqual(outcome["decisions"], {"mapping": "reject", "grouping": "reject"})
        self.assertTrue(any("Drowning" in p and "50%" in p for p in outcome["problems"]["mapping"]))
        self.assertEqual(outcome["recorded"], {"mapping": "not_recorded", "grouping": "not_recorded"})
        source = load_source(self.project / "configs" / "datasets" / "fixture.yaml", self.reviews)
        self.assertIsNone(load_mapping_review(source, self.reviews))
        self.assertIsNone(load_grouping_review(source, self.reviews))

    def test_uncertain_asks_the_user_and_records_nothing(self):
        outcome = self.service.submit_verdict("fixture_pool", self.verdict(decision={"mapping": "uncertain", "grouping": "approve"}))
        self.assertEqual(outcome["ask_user"], ["mapping"])
        self.assertEqual(outcome["recorded"]["mapping"], "not_recorded")
        self.assertEqual(outcome["recorded"]["grouping"], "verified")

    def test_failing_automatic_rules_block_approval_even_if_the_ai_approves(self):
        strict = self.make_service(relaxed_policy(self.project, min_boxes_per_mapped_label={"severity": "fail", "min": 1000}))
        evidence = strict.evidence("fixture_pool")
        verdict = self.verdict()
        verdict["evidence_sha256"] = evidence["evidence_sha256"]
        outcome = strict.submit_verdict("fixture_pool", verdict)
        self.assertEqual(outcome["decisions"]["mapping"], "reject")
        self.assertIn("automatic rules failed", " ".join(outcome["problems"]["mapping"]))
        self.assertEqual(outcome["recorded"]["mapping"], "not_recorded")

    def test_malformed_unseen_stale_or_human_verdicts_are_refused(self):
        cases = {
            "not viewed": {"viewed_images": False},
            "stale evidence": {"evidence_sha256": "0" * 64},
            "other mapping": {"mapping_hash": "1" * 64},
            "human kind": {"reviewer": {"kind": "human", "name": "x"}},
            "no name": {"reviewer": {"kind": "ai", "name": " "}},
            "bad schema": {"schema_version": 2},
            "bad decision": {"decision": {"mapping": "maybe", "grouping": "approve"}},
        }
        for name, change in cases.items():
            with self.assertRaises(ReviewConflict, msg=name):
                self.service.submit_verdict("fixture_pool", self.verdict(**change))
        out_of_range = self.verdict()
        out_of_range["mapping"]["labels"]["Swimming"]["visible_person_fraction"] = 1.5
        with self.assertRaises(ReviewConflict):
            self.service.submit_verdict("fixture_pool", out_of_range)
        missing = self.verdict()
        del missing["mapping"]["labels"]["out of water"]
        self.assertEqual(self.service.submit_verdict("fixture_pool", missing)["decisions"]["mapping"], "reject")
        too_few = self.verdict()
        too_few["mapping"]["labels"]["Swimming"]["samples_viewed"] = 5
        self.assertEqual(self.service.submit_verdict("fixture_pool", too_few)["decisions"]["mapping"], "reject")

    def test_policy_can_forbid_ai_approvals(self):
        policy = load_policy()
        policy["delegation"]["allow_ai_approval"] = False
        policy["auto_rules"]["min_boxes_per_mapped_label"]["min"] = 3
        path = self.project / "no-ai.yaml"
        path.write_text(yaml.safe_dump(policy), encoding="utf-8")
        service = self.make_service(path)
        verdict = self.verdict()
        verdict["evidence_sha256"] = service.evidence("fixture_pool")["evidence_sha256"]
        with self.assertRaisesRegex(ReviewConflict, "does not allow AI approvals"):
            service.submit_verdict("fixture_pool", verdict)

    # -- HTTP ------------------------------------------------------------------------------
    def test_endpoints_expose_rules_evidence_and_verdict(self):
        os.environ["INPUT_CACHE_DIR"] = str(self.project / "local-cache" / "inputs")
        self.addCleanup(os.environ.pop, "INPUT_CACHE_DIR", None)
        app = create_app(settings=Settings.from_env({"INPUT_CACHE_DIR": str(self.project / "local-cache" / "inputs")}),
                         config_dir=self.project / "configs" / "datasets", env_file=self.project / "missing.env")
        app.state.reviews.policy_path = self.service.policy_path

        def call(method, path, **kwargs):
            async def run():
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
                    return await client.request(method, path, **kwargs)
            return asyncio.run(run())

        base = "/api/v1/sources/fixture_pool/review"
        self.assertEqual(call("GET", base + "/rules").json()["outcome"], "pass")
        evidence = call("POST", base + "/evidence").json()
        self.assertTrue(Path(evidence["evidence_dir"]).is_dir())
        status = call("GET", base).json()
        self.assertEqual(status["auto_rules"]["outcome"], "pass")
        self.assertEqual({s["kind"] for s in status["group_sheets"]}, {"cross-split", "same-split", "largest"} & {s["kind"] for s in status["group_sheets"]})
        self.assertEqual(call("GET", base + "/sheets/groups--cross-split").status_code, 200)
        refused = call("POST", base + "/verdict", json={**self.verdict(), "viewed_images": False})
        self.assertEqual(refused.status_code, 409)
        accepted = call("POST", base + "/verdict", json=self.verdict())
        self.assertEqual(accepted.status_code, 200, accepted.text)
        self.assertEqual(accepted.json()["recorded"]["mapping"], "approved")


if __name__ == "__main__":
    unittest.main()
