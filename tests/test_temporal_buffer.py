import unittest

from drowning.inference.buffer import FrameCrop, TemporalWindowBuffer


class TemporalBufferTests(unittest.TestCase):
    def test_buffer_uses_fixed_sampling_and_waits_for_full_window(self):
        buffer = TemporalWindowBuffer(sequence_length=3, sample_period_ms=250, max_gap_ms=1000)
        self.assertIsNone(buffer.push(FrameCrop("track-1", 0, "a", "v1")))
        self.assertIsNone(buffer.push(FrameCrop("track-1", 100, "skip", "v1")))
        self.assertIsNone(buffer.push(FrameCrop("track-1", 250, "b", "v1")))
        window = buffer.push(FrameCrop("track-1", 500, "c", "v1"))
        self.assertEqual([item.crop for item in window], ["a", "b", "c"])
        self.assertEqual([item.timestamp_ms for item in window], [0, 250, 500])

    def test_resets_on_track_gap_scene_or_model_change(self):
        for next_frame in (
            FrameCrop("track-1", 1500, "gap", "v1", "scene-a"),
            FrameCrop("track-1", 250, "scene", "v1", "scene-b"),
            FrameCrop("track-1", 250, "model", "v2", "scene-a"),
        ):
            with self.subTest(frame=next_frame):
                buffer = TemporalWindowBuffer(sequence_length=2, sample_period_ms=100, max_gap_ms=500)
                buffer.push(FrameCrop("track-1", 0, "first", "v1", "scene-a"))
                self.assertIsNone(buffer.push(next_frame))
                self.assertEqual(len(buffer._frames["track-1"]), 1)

    def test_tracks_have_independent_windows_and_require_monotonic_timestamps(self):
        buffer = TemporalWindowBuffer(sequence_length=2, sample_period_ms=100, max_gap_ms=500)
        buffer.push(FrameCrop("person-a", 0, "a0", "v1"))
        buffer.push(FrameCrop("person-b", 0, "b0", "v1"))
        self.assertEqual(buffer.tracks(), ("person-a", "person-b"))
        with self.assertRaisesRegex(ValueError, "timestamps must increase"):
            buffer.push(FrameCrop("person-a", 0, "a1", "v1"))
        a_window = buffer.push(FrameCrop("person-a", 100, "a1", "v1"))
        b_window = buffer.push(FrameCrop("person-b", 100, "b1", "v1"))
        self.assertEqual([item.crop for item in a_window], ["a0", "a1"])
        self.assertEqual([item.crop for item in b_window], ["b0", "b1"])

    def test_rejects_invalid_sequence_contract(self):
        with self.assertRaises(ValueError):
            TemporalWindowBuffer(sequence_length=1, sample_period_ms=200, max_gap_ms=200)


if __name__ == "__main__":
    unittest.main()
