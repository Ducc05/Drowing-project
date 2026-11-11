"""Motion gate: a track that travels more than a few body lengths is a swimmer, not a drowning person."""
import unittest

from PIL import Image

from drowning.config import Settings
from drowning.inference.h20saver import H20SaverInference
from drowning.inference.motion import MotionGate
from tests.test_cell_gate import DROWNING, SWIMMING, TrackedDetector, alerted, box_at, crossing, stationary


def run(frames, *, motion=1.5, grid=None, step_ms=100, fps=10, motion_window_s=3.0, live=False):
    pipeline = H20SaverInference(TrackedDetector(frames), model_version="motion-test", cell_grid=grid,
                                 motion_max_body_lengths=motion, motion_window_s=motion_window_s)
    pipeline.configure_live_stream(fps) if live else pipeline.configure_stream(fps)
    image = Image.new("RGB", (200, 200))
    return pipeline, [pipeline.process_frame(image, i * step_ms, i) for i in range(len(frames))]


def drifting(count, start=50.0, end=60.0, cy=60, track="trk-a", half=5.0):
    """A victim slowly carried sideways: (end - start) / (2 * half) body lengths over the whole clip."""
    return [[(track, box_at(start + (end - start) * i / (count - 1), cy, half=half), DROWNING)] for i in range(count)]


class MotionGateTests(unittest.TestCase):
    def feed(self, gate, points, size=10.0, step_ms=100, track="a"):
        out = None
        for i, (cx, cy) in enumerate(points):
            out = gate.update(i * step_ms, [(track, box_at(cx, cy, half=size / 2))])
        return out

    def test_a_stationary_track_has_no_motion(self):
        gate = MotionGate(window_ms=3000, max_body_lengths=1.5)
        self.assertEqual(self.feed(gate, [(50, 50)] * 20), {"a": 0.0})
        self.assertFalse(gate.travelling("a", 1900))

    def test_a_fast_swimmer_covers_many_body_lengths(self):
        gate = MotionGate(window_ms=3000, max_body_lengths=1.5)
        self.feed(gate, [(10 + 4 * i, 50) for i in range(20)])  # 76 px with a 10 px body
        self.assertAlmostEqual(gate.body_lengths("a", 1900), 7.6, places=1)
        self.assertTrue(gate.travelling("a", 1900))

    def test_motion_is_measured_in_body_lengths_so_distance_from_the_camera_does_not_matter(self):
        small, large = MotionGate(3000, 1.5), MotionGate(3000, 1.5)
        self.feed(small, [(10 + 2 * i, 50) for i in range(20)], size=10)
        self.feed(large, [(10 + 8 * i, 50) for i in range(20)], size=40)
        self.assertAlmostEqual(small.body_lengths("a", 1900), large.body_lengths("a", 1900), places=2)

    def test_slow_drift_stays_below_the_limit_and_diagonal_movement_counts(self):
        gate = MotionGate(3000, 1.5)
        self.feed(gate, [(50 + 0.3 * i, 50) for i in range(30)])  # 8.7 px = 0.87 body lengths
        self.assertFalse(gate.travelling("a", 2900))
        diagonal = MotionGate(3000, 1.5)
        self.feed(diagonal, [(50 + 0.5 * i, 50 + 0.5 * i) for i in range(30)])  # ~20.5 px = 2.05 body lengths
        self.assertTrue(diagonal.travelling("a", 2900))

    def test_old_movement_is_forgotten_after_the_window(self):
        gate = MotionGate(3000, 1.5)
        for i in range(10):  # sprint during the first second
            gate.update(i * 100, [("a", box_at(10 + 4 * i, 50))])
        self.assertTrue(gate.travelling("a", 900))
        for i in range(10, 50):  # then rests
            gate.update(i * 100, [("a", box_at(46, 50))])
        self.assertFalse(gate.travelling("a", 4900))

    def test_tracks_are_independent_and_unknown_or_expired_tracks_are_not_travelling(self):
        gate = MotionGate(3000, 1.5)
        for i in range(20):
            gate.update(i * 100, [("walker", box_at(10 + 4 * i, 50)), ("still", box_at(50, 80))])
        self.assertTrue(gate.travelling("walker", 1900))
        self.assertFalse(gate.travelling("still", 1900))
        self.assertFalse(gate.travelling("nobody", 1900))
        self.assertFalse(gate.travelling("walker", 30000))  # no samples left in the window
        gate.update(30000, [])
        self.assertEqual(gate.body_lengths("walker", 30000), 0.0)

    def test_reset_and_validation(self):
        gate = MotionGate(3000, 1.5)
        self.feed(gate, [(10 + 4 * i, 50) for i in range(20)])
        gate.reset()
        self.assertFalse(gate.travelling("a", 1900))
        for bad in ({"window_ms": 0, "max_body_lengths": 1.5}, {"window_ms": 3000, "max_body_lengths": 0}):
            with self.assertRaises(ValueError):
                MotionGate(**bad)


class MotionGatedAlertTests(unittest.TestCase):
    def test_a_swimmer_crossing_the_view_is_not_confirmed_but_is_without_the_gate(self):
        _, gated = run(crossing(30))
        self.assertFalse(any(alerted(gated, "trk-b")))
        _, control = run(crossing(30), motion=None)
        self.assertTrue(any(alerted(control, "trk-b")))

    def test_the_suppressed_alarm_is_reported_with_its_motion(self):
        _, records = run(crossing(30))
        flat = [r for frame in records for r in frame]
        self.assertTrue(any(r["alert_suppressed_by_motion"] for r in flat))
        self.assertFalse(any(r["alert_suppressed_by_cell"] for r in flat))  # cell gate is off here
        self.assertGreater(max(r["motion"] for r in flat), 1.5)

    def test_a_stationary_person_alerts_as_soon_as_the_track_alarm_does(self):
        _, records = run(stationary(40))
        flags = [r[0]["alert"] for r in records]
        self.assertEqual(flags.index(True), 13)  # 14th vote of a 2 s window at 10 fps: no extra delay
        self.assertEqual(max(r[0]["motion"] for r in records), 0.0)

    def test_a_slowly_drifting_victim_still_alerts_even_across_cell_borders(self):
        _, records = run(drifting(40, 50, 60))  # 1 body length in 4 s
        self.assertTrue(any(alerted(records, "trk-a")))

    def test_a_flailing_victim_who_stays_in_place_still_alerts(self):
        shaking = [[("trk-a", box_at(50 + (4 if i % 2 else -4), 60 + (3 if i % 3 else -3)), DROWNING)] for i in range(40)]
        _, records = run(shaking)  # extent ~ 10 px with a 10 px body = 1.0 body length
        self.assertTrue(any(alerted(records, "trk-a")))

    def test_victim_alerts_while_the_swimmer_next_to_them_does_not(self):
        frames = [r + v for r, v in zip(crossing(40), stationary(40))]
        _, records = run(frames)
        self.assertTrue(any(alerted(records, "trk-a")))
        self.assertFalse(any(alerted(records, "trk-b")))

    def test_a_swimming_person_never_alerts(self):
        frames = [[("trk-a", box_at(50, 60), SWIMMING)] for _ in range(40)]
        _, records = run(frames)
        self.assertFalse(any(alerted(records, "trk-a")))

    def test_a_victim_who_starts_swimming_away_loses_the_alert(self):
        still = stationary(30)
        away = [[("trk-a", box_at(60 + 5 * i, 60), DROWNING)] for i in range(20)]
        _, records = run(still + away)
        flags = [r[0]["alert"] for r in records]
        self.assertTrue(any(flags[:30]))
        self.assertFalse(flags[-1])

    def test_both_gates_together_still_block_the_swimmer_and_pass_the_victim(self):
        frames = [r + v for r, v in zip(crossing(40), stationary(40))]
        _, records = run(frames, grid=(4, 4))
        self.assertTrue(any(alerted(records, "trk-a")))
        self.assertFalse(any(alerted(records, "trk-b")))

    def test_live_clock_uses_elapsed_milliseconds(self):
        _, records = run(crossing(30), live=True)
        self.assertFalse(any(alerted(records, "trk-b")))
        _, still = run(stationary(40), live=True)
        self.assertTrue(any(alerted(still, "trk-a")))

    def test_motion_window_is_elapsed_time_whatever_the_stride(self):
        """Samples 300 ms apart (stride 3 at 10 fps): the last 3 s hold 10 samples, 9 steps of 24 px."""
        frames = [[("trk-b", box_at(10 + 8 * i * 3, 60), DROWNING)] for i in range(14)]
        pipeline = H20SaverInference(TrackedDetector(frames), model_version="motion-test",
                                     motion_max_body_lengths=1.5)
        pipeline.configure_stream(10, frame_stride=3)
        image = Image.new("RGB", (400, 200))
        records = [pipeline.process_frame(image, i * 300, i * 3)[0] for i in range(14)]
        self.assertAlmostEqual(records[-1]["motion"], 21.6, places=1)

    def test_reset_stream_clears_motion_history_and_metadata_names_the_policy(self):
        pipeline, _ = run(crossing(30))
        meta = pipeline.result_metadata()
        self.assertEqual(meta["motion_gate"], {"max_body_lengths": 1.5, "window_s": 3.0})
        self.assertEqual(meta["inference_policy"], "h20saver_motion_gated_votes_v1")
        self.assertNotIn("cell_gate", meta)
        both, _ = run(stationary(5), grid=(4, 4))
        self.assertEqual(both.result_metadata()["inference_policy"], "h20saver_cell_motion_gated_votes_v1")
        pipeline.reset_stream()
        pipeline.detector.frames = crossing(2)
        record, = pipeline.process_frame(Image.new("RGB", (200, 200)), 0, 0)
        self.assertEqual(record["motion"], 0.0)
        off, _ = run(stationary(5), motion=None)
        self.assertNotIn("motion_gate", off.result_metadata())

    def test_motion_gate_off_adds_no_record_fields_beyond_the_suppression_flag(self):
        _, records = run(stationary(3), motion=None)
        self.assertNotIn("motion", records[0][0])
        self.assertFalse(records[0][0]["alert_suppressed_by_motion"])

    def test_invalid_parameters_are_rejected(self):
        for kwargs in ({"motion_max_body_lengths": 0.0}, {"motion_max_body_lengths": -1}, {"motion_window_s": 0}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                H20SaverInference(TrackedDetector([]), model_version="x", **{"motion_max_body_lengths": 1.5, **kwargs})


class MotionSettingsTests(unittest.TestCase):
    def test_defaults_overrides_and_off(self):
        default = Settings.from_env({})
        self.assertEqual((default.h20saver_motion_max_body_lengths, default.h20saver_motion_window_seconds), (1.5, 3.0))
        custom = Settings.from_env({"H20SAVER_MOTION_MAX_BODY_LENGTHS": "2.5", "H20SAVER_MOTION_WINDOW_SECONDS": "4"})
        self.assertEqual((custom.h20saver_motion_max_body_lengths, custom.h20saver_motion_window_seconds), (2.5, 4.0))
        for off in ("off", "0", "none", "OFF"):
            self.assertIsNone(Settings.from_env({"H20SAVER_MOTION_MAX_BODY_LENGTHS": off}).h20saver_motion_max_body_lengths)

    def test_invalid_values_are_rejected(self):
        for env in ({"H20SAVER_MOTION_MAX_BODY_LENGTHS": "abc"}, {"H20SAVER_MOTION_MAX_BODY_LENGTHS": "-1"},
                    {"H20SAVER_MOTION_MAX_BODY_LENGTHS": "99"}, {"H20SAVER_MOTION_WINDOW_SECONDS": "0"},
                    {"H20SAVER_MOTION_WINDOW_SECONDS": "abc"}, {"H20SAVER_MOTION_WINDOW_SECONDS": "99"}):
            with self.assertRaises(ValueError, msg=str(env)):
                Settings.from_env(env)


if __name__ == "__main__":
    unittest.main()
