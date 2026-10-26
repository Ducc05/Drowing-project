import json
import tempfile
import unittest
from pathlib import Path

from notebooks.materialize_kernel import materialize_kernel


class KaggleMaterializerTests(unittest.TestCase):
    def test_materializes_private_mode_matched_kernel_without_remote_action(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "kernel"
            result = materialize_kernel("owner/drowning-temporal", output, mode="remote_smoke",
                                        dataset_sources=["owner/temporal-data"])
            metadata = json.loads((result / "kernel-metadata.json").read_text(encoding="utf-8"))
            notebook = json.loads((result / "Drowning_Temporal_Kaggle.ipynb").read_text(encoding="utf-8"))
            self.assertTrue(metadata["is_private"])
            self.assertEqual(metadata["id"], "owner/drowning-temporal")
            self.assertEqual(metadata["dataset_sources"], ["owner/temporal-data"])
            self.assertEqual(notebook["metadata"]["drowning_project"]["run_mode_default"], "remote_smoke")
            self.assertNotIn("kernel-metadata", notebook["metadata"])
            with self.assertRaises(FileExistsError):
                materialize_kernel("owner/drowning-temporal", output)

    def test_refuses_missing_ref_invalid_mode_and_protected_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                materialize_kernel("", Path(directory) / "kernel")
            with self.assertRaises(ValueError):
                materialize_kernel("owner/kernel", Path(directory) / "kernel", mode="local_train")
            with self.assertRaisesRegex(ValueError, "Protected"):
                materialize_kernel("owner/kernel", Path("D:/DrowningDemo/kernel"))


if __name__ == "__main__":
    unittest.main()
