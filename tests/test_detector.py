"""Detector adapter contracts that do not load or update real weights."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from drowning.inference.detector import UltralyticsPersonDetector


class DetectorColorOrderTests(unittest.TestCase):
    def test_pil_rgb_image_is_converted_to_contiguous_bgr_before_ultralytics(self):
        observed = {}

        class FakeYOLO:
            names = {0: "person"}

            def predict(self, image, **kwargs):
                observed["image"] = image
                observed["kwargs"] = kwargs
                return [SimpleNamespace(boxes=None)]

        with tempfile.TemporaryDirectory() as temp:
            weights = Path(temp) / "fixture.pt"
            weights.write_bytes(b"test fixture only")
            with patch("ultralytics.YOLO", return_value=FakeYOLO()):
                detector = UltralyticsPersonDetector(weights, person_class_id=0)
                result = detector.detect(Image.new("RGB", (2, 2), (10, 20, 30)))

        self.assertEqual(result, [])
        self.assertEqual(observed["image"][0, 0].tolist(), [30, 20, 10])
        self.assertTrue(observed["image"].flags["C_CONTIGUOUS"])
        self.assertEqual(observed["kwargs"]["classes"], [0])
        self.assertEqual(observed["kwargs"]["conf"], 0.25)


if __name__ == "__main__":
    unittest.main()
