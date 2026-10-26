import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from drowning.data.leakage import dihedral_dhashes, near_copy_report
from drowning.ml.augment import augment_crop, sample_rng
from drowning.ml.datasets import CanonicalImageCropDataset


def texture(seed, size=64):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).resize((size, size), Image.Resampling.NEAREST)


class AugmentTests(unittest.TestCase):
    def test_same_seed_epoch_and_index_give_identical_pixels_and_another_epoch_differs(self):
        image = texture(1)
        first = augment_crop(image, sample_rng(7, 0, 3))
        again = augment_crop(image, sample_rng(7, 0, 3))
        self.assertEqual(first.tobytes(), again.tobytes())
        self.assertEqual(first.size, image.size)
        others = {augment_crop(image, sample_rng(7, epoch, 3)).tobytes() for epoch in range(1, 6)}
        self.assertGreater(len(others), 1)

    def test_output_stays_a_valid_square_rgb_crop_for_many_samples(self):
        image = texture(2, 48)
        for index in range(60):
            out = augment_crop(image, sample_rng(1, 0, index))
            self.assertEqual((out.size, out.mode), ((48, 48), "RGB"))


class DatasetAugmentTests(unittest.TestCase):
    def make_manifest(self, directory, count=4):
        rows = []
        for i in range(count):
            relative = f"c{i}.jpg"
            texture(i).save(directory / relative, format="JPEG")
            rows.append({"crop_relative_path": relative, "crop_sha256": hashlib.sha256((directory / relative).read_bytes()).hexdigest(),
                         "split": "train", "training_eligible": True, "canonical_label": ["drowning", "swimming", "out_of_water"][i % 3]})
        manifest = directory / "crop_manifest.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return manifest

    def test_validation_style_dataset_is_never_augmented_and_train_style_is_reproducible(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = self.make_manifest(Path(temp))
            plain_a = CanonicalImageCropDataset(manifest, split="train", output_size=32)
            plain_b = CanonicalImageCropDataset(manifest, split="train", output_size=32)
            self.assertTrue(all(np.array_equal(plain_a[i][0].numpy(), plain_b[i][0].numpy()) for i in range(4)))
            aug = CanonicalImageCropDataset(manifest, split="train", output_size=32, augment=True, augment_seed=5)
            aug2 = CanonicalImageCropDataset(manifest, split="train", output_size=32, augment=True, augment_seed=5)
            self.assertTrue(np.array_equal(aug[1][0].numpy(), aug2[1][0].numpy()))
            self.assertFalse(np.array_equal(aug[1][0].numpy(), plain_a[1][0].numpy()))
            before = aug[1][0].numpy().copy()
            aug.set_epoch(1)
            self.assertFalse(np.array_equal(aug[1][0].numpy(), before))
            self.assertEqual(aug[1][1], plain_a[1][1])  # the label is untouched


class LeakageTests(unittest.TestCase):
    def test_mirrored_and_turned_copies_are_found_and_unrelated_images_are_not(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = []
            for i in range(6):
                path = root / f"train{i}.png"
                texture(100 + i, 96).save(path)
                train.append(path)
            queries = []
            mirrored = root / "mirrored.png"
            ImageOps.mirror(Image.open(train[0])).save(mirrored)
            turned = root / "turned.png"
            Image.open(train[1]).rotate(90, expand=True).save(turned)
            unrelated = root / "unrelated.png"
            texture(999, 96).save(unrelated)
            queries = [mirrored, turned, unrelated]
            report = near_copy_report(train, queries)
            self.assertEqual((report["queries"], report["references"], report["with_near_copy"]), (3, 6, 2))
            self.assertEqual(report["fraction"], round(2 / 3, 4))
            self.assertTrue(report["lower_bound"])
            self.assertEqual(len(dihedral_dhashes(train[0])), 8)

    def test_empty_inputs_report_no_fraction(self):
        self.assertIsNone(near_copy_report([], [Path("x")])["fraction"])


if __name__ == "__main__":
    unittest.main()
