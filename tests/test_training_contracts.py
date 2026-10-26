import unittest

try:
    import torch
    from drowning.ml.training import classification_metrics, image_loss, temporal_loss
except ImportError:
    torch = None
    image_loss = temporal_loss = classification_metrics = None


@unittest.skipIf(torch is None, "install optional ML dependencies to run model contract tests")
class TrainingContractTests(unittest.TestCase):
    def test_image_and_temporal_losses_ignore_unsupervised_anchors(self):
        image_logits = torch.randn(2, 3, requires_grad=True)
        image_targets = torch.tensor([0, 2])
        self.assertGreater(float(image_loss(image_logits, image_targets).detach()), 0)
        logits = torch.randn(1, 3, 3, requires_grad=True)
        labels = torch.tensor([[0, -100, 2]])
        valid = torch.tensor([[True, True, True]])
        loss = temporal_loss(logits, labels, valid)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_rejects_temporal_batch_without_supervised_anchors(self):
        with self.assertRaisesRegex(ValueError, "no supervised"):
            temporal_loss(torch.randn(1, 2, 3), torch.full((1, 2), -100), torch.ones((1, 2), dtype=torch.bool))

    def test_classification_metrics_report_per_class_recall_and_macro_f1(self):
        report = classification_metrics([0, 0, 1, 2], [0, 1, 1, 2])
        self.assertEqual(report["sample_count"], 4)
        self.assertEqual(report["accuracy"], 0.75)
        self.assertAlmostEqual(report["macro_f1"], (2/3 + 2/3 + 1)/3)
        self.assertEqual(report["per_class_recall"]["swimming"], 1.0)


if __name__ == "__main__":
    unittest.main()
