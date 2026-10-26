import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from drowning.ml.bundle import load_bundle, save_bundle, validate_manifest
    from drowning.ml.model import DrowningTemporalModel
except ImportError:
    torch = None
    DrowningTemporalModel = None
    load_bundle = save_bundle = validate_manifest = None


@unittest.skipIf(torch is None, "install optional ML dependencies to run model contract tests")
class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_image_and_temporal_shapes_and_causal_prefix(self):
        torch.manual_seed(7)
        model = DrowningTemporalModel(hidden_size=16).eval()
        frames = torch.randn(1, 4, 3, 64, 64)
        with torch.no_grad():
            output = model(frames)
            changed_future = frames.clone()
            changed_future[:, 3] = torch.randn_like(changed_future[:, 3]) * 20
            future_output = model(changed_future)
        self.assertEqual(tuple(output["image_logits"].shape), (1, 4, 3))
        self.assertEqual(tuple(output["temporal_logits"].shape), (1, 4, 3))
        self.assertTrue(torch.allclose(output["temporal_logits"][:, :3], future_output["temporal_logits"][:, :3]))

    def test_invalid_frames_do_not_change_recurrent_state(self):
        torch.manual_seed(11)
        model = DrowningTemporalModel(hidden_size=12).eval()
        frames = torch.randn(1, 3, 3, 64, 64)
        mask = torch.tensor([[True, False, True]])
        changed = frames.clone()
        changed[:, 1] = torch.randn_like(changed[:, 1]) * 50
        with torch.no_grad():
            first = model(frames, mask)["temporal_logits"]
            second = model(changed, mask)["temporal_logits"]
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=1e-6))

    def test_gradient_smoke_does_not_run_optimizer(self):
        model = DrowningTemporalModel(hidden_size=8).train()
        frames = torch.randn(2, 2, 3, 64, 64)
        output = model(frames)
        loss = output["image_logits"].mean() + output["temporal_logits"].mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
        self.assertIsNone(getattr(model, "optimizer", None))

    def test_smoke_bundle_round_trip_is_labeled_and_not_approvable(self):
        model = DrowningTemporalModel(hidden_size=8).eval()
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "smoke-bundle"
            manifest = save_bundle(model, bundle_path, completion_kind="synthetic_smoke",
                                   temporal_trained=False, preprocessing={"size": [64, 64], "color": "RGB",
                                                                          "context_fraction": 0.15,
                                                                          "mean": [0.485, 0.456, 0.406],
                                                                          "std": [0.229, 0.224, 0.225]})
            self.assertEqual(manifest["completion_kind"], "synthetic_smoke")
            self.assertIn("torch", manifest["runtime_versions"])
            self.assertTrue(validate_manifest(manifest, require_temporal=False))
            loaded, loaded_manifest = load_bundle(bundle_path, require_temporal=False, allow_smoke=True)
            self.assertEqual(loaded_manifest["class_keys"], ["drowning", "swimming", "out_of_water"])
            self.assertFalse(loaded.training)
            with self.assertRaisesRegex(ValueError, "cannot be assigned"):
                save_bundle(model, Path(directory) / "approved", completion_kind="approved",
                            temporal_trained=True, preprocessing={})

    def test_bundle_checksum_is_verified_before_loading(self):
        model = DrowningTemporalModel(hidden_size=8).eval()
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "bundle"
            save_bundle(model, bundle_path, completion_kind="synthetic_smoke",
                        temporal_trained=False, preprocessing={"size": 224, "context_fraction": 0.15,
                                                              "mean": [0.485, 0.456, 0.406],
                                                              "std": [0.229, 0.224, 0.225]})
            with (bundle_path / "weights.pt").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_bundle(bundle_path, require_temporal=False, allow_smoke=True)


if __name__ == "__main__":
    unittest.main()
