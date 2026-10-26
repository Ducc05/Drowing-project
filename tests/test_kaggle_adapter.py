import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from drowning.integrations.kaggle import KaggleCLIClient, KaggleError, RemoteStateUnconfirmed


class KaggleAdapterTests(unittest.TestCase):
    def test_push_passes_argument_array_and_requires_private_metadata(self):
        seen = []
        def runner(args, **kwargs):
            seen.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, stdout="submitted", stderr="")

        with tempfile.TemporaryDirectory() as directory:
            kernel_dir = Path(directory)
            (kernel_dir / "notebook.ipynb").write_text("{}", encoding="utf-8")
            metadata = {"id": "owner/drowning-temp", "code_file": "notebook.ipynb", "is_private": True}
            (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            client = KaggleCLIClient("kaggle", runner=runner)
            client.push_private_kernel(kernel_dir, timeout_seconds=900)
            self.assertEqual(seen[0][0][:4], ["kaggle", "kernels", "push", "-p"])
            self.assertFalse(seen[0][1]["shell"])
            self.assertEqual(seen[0][1]["timeout"], 930)
            metadata["is_private"] = False
            (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "is_private"):
                client.push_private_kernel(kernel_dir)
            metadata["is_private"] = True
            (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (kernel_dir / ".env").write_text("TOKEN=secret", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "only its notebook"):
                client.push_private_kernel(kernel_dir)

    def test_push_timeout_is_remote_unconfirmed_and_never_retried(self):
        calls = []
        def runner(args, **kwargs):
            calls.append(args)
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])

        with tempfile.TemporaryDirectory() as directory:
            kernel_dir = Path(directory)
            (kernel_dir / "notebook.ipynb").write_text("{}", encoding="utf-8")
            (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
                "id": "owner/drowning-temp", "code_file": "notebook.ipynb", "is_private": True,
            }), encoding="utf-8")
            with self.assertRaises(RemoteStateUnconfirmed):
                KaggleCLIClient("kaggle", runner=runner).push_private_kernel(kernel_dir, timeout_seconds=30)
            self.assertEqual(len(calls), 1)

    def test_exact_version_output_collection_and_path_safety(self):
        seen = []
        def runner(args, **kwargs):
            seen.append(args)
            return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

        client = KaggleCLIClient("kaggle", runner=runner)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "artifacts"
            client.collect_exact_output("owner/drowning-temp/17", target)
            self.assertEqual(seen[0][1:4], ["kernels", "output", "owner/drowning-temp/17"])
            with self.assertRaisesRegex(ValueError, "exact Kaggle version"):
                client.collect_exact_output("owner/drowning-temp", target / "latest")
            (target / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                client.collect_exact_output("owner/drowning-temp/17", target)
            with self.assertRaisesRegex(ValueError, "Protected"):
                client.collect_exact_output("owner/drowning-temp/17", Path("D:/DrowningDemo/out"))

    def test_invalid_ref_and_missing_cli_are_clear(self):
        with self.assertRaises(ValueError):
            KaggleCLIClient.validate_kernel_ref("owner;Remove-Item/slug")
        client = KaggleCLIClient(executable="kaggle-not-installed")
        with self.assertRaisesRegex(KaggleError, "not found"):
            client.version()


if __name__ == "__main__":
    unittest.main()
