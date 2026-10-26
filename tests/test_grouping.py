import unittest

from drowning.data.grouping import SplitBlocked, build_group_split


def media(media_id, group_id, fingerprint):
    return {"media_id": media_id, "global_group_id": group_id, "grouping_status": "verified",
            "content_sha256": fingerprint}


class GroupSplitTests(unittest.TestCase):
    def test_verified_groups_and_exact_duplicates_never_cross_splits(self):
        items = [
            media("a1", "scene-a", "1" * 64),
            media("a2", "scene-a", "2" * 64),
            media("b1", "scene-b", "3" * 64),
            media("b2", "scene-b", "4" * 64),
            media("c1", "scene-c", "5" * 64),
            media("duplicate-copy", "scene-copy", "1" * 64),
        ]
        first = build_group_split(items, seed=10)
        second = build_group_split(items, seed=10)
        self.assertEqual(first["manifest"], second["manifest"])
        assignments = {record["media_id"]: record["split"] for record in first["manifest"]}
        self.assertEqual(assignments["a1"], assignments["a2"])
        self.assertEqual(assignments["a1"], assignments["duplicate-copy"])
        self.assertTrue(first["source_video_disjointness_verified"])
        self.assertEqual(set(first["media_counts_by_split"]), {"train", "validation", "test"})

    def test_manual_holdout_is_preserved_for_entire_group(self):
        items = [media(f"m-{group}", group, str(index) * 64) for index, group in enumerate(("a", "b", "c"), 1)]
        report = build_group_split(items, holdout_group_ids={"b"}, seed=42)
        holdout = next(item for item in report["manifest"] if item["global_group_id"] == "b")
        self.assertEqual(holdout["split"], "test")
        self.assertEqual(holdout["split_reason"], "user_holdout")

    def test_unknown_source_group_and_missing_holdout_block_split(self):
        items = [media("a", "a", "a" * 64), media("b", "b", "b" * 64), media("c", "c", "c" * 64)]
        items[0]["grouping_status"] = "unknown"
        with self.assertRaisesRegex(SplitBlocked, "unverified"):
            build_group_split(items)
        with self.assertRaisesRegex(SplitBlocked, "holdout group"):
            build_group_split(items[1:], holdout_group_ids={"absent"})


if __name__ == "__main__":
    unittest.main()
