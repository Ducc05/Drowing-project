import unittest

import numpy as np

from drowning.inference.tracking import Box, ByteTracker, GreedyIoUTracker, linear_sum_assignment


def person(cx, cy, confidence=0.9, w=40, h=80):
    return Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, confidence)


def run(tracker, frames):
    """Return per-frame ``{truth_name: track_id}`` for frames of ``{truth_name: Box}``."""
    history = []
    for frame in frames:
        names = list(frame)
        result = tracker.update([frame[name] for name in names])
        history.append({name: track_id for name, (track_id, _) in zip(names, result)})
    return history


def id_switches(history):
    last, switches = {}, 0
    for assignment in history:
        for name, track_id in assignment.items():
            if name in last and last[name] != track_id:
                switches += 1
            last[name] = track_id
    return switches


def crossing_scene(frames=40, occlusion=range(18, 22)):
    scene = []
    for t in range(frames):
        frame = {"a": person(40 + 12 * t, 100 + 0.5 * t), "b": person(520 - 12 * t, 100 - 0.5 * t)}
        if t in occlusion:
            del frame["b"]
        scene.append(frame)
    return scene


class AssignmentTests(unittest.TestCase):
    def test_hungarian_matches_known_optimum_for_rectangular_matrices(self):
        cost = np.array([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
        self.assertEqual(sum(cost[r, c] for r, c in linear_sum_assignment(cost)), 5.0)
        wide = np.array([[1.0, 9.0, 9.0, 0.5], [9.0, 2.0, 9.0, 9.0]])
        self.assertEqual(linear_sum_assignment(wide), [(0, 3), (1, 1)])
        self.assertEqual(linear_sum_assignment(wide.T), [(1, 1), (3, 0)])
        self.assertEqual(linear_sum_assignment(np.zeros((0, 3))), [])


class ByteTrackerTests(unittest.TestCase):
    def test_keeps_both_ids_through_a_crossing_and_a_short_occlusion(self):
        history = run(ByteTracker(), crossing_scene())
        self.assertEqual(id_switches(history), 0)
        self.assertEqual(len({a["a"] for a in history}), 1)
        self.assertEqual(len({a["b"] for a in history if "b" in a}), 1)

    def test_every_detection_is_returned_including_low_confidence_unmatched_ones(self):
        tracker = ByteTracker(high_threshold=0.5)
        result = tracker.update([person(50, 50, 0.9), person(300, 300, 0.3)])
        self.assertEqual(len(result), 2)
        self.assertEqual(len({track_id for track_id, _ in result}), 2)

    def test_confidence_dip_does_not_lose_identity(self):
        frames = []
        for t in range(30):
            confidence = 0.3 if 10 <= t < 20 else 0.9
            frames.append({"a": person(100 + 6 * t, 120, confidence)})
        history = run(ByteTracker(high_threshold=0.5), frames)
        self.assertEqual(len({a["a"] for a in history}), 1)

    def test_track_is_dropped_after_max_lost_frames_then_gets_a_new_id(self):
        tracker = ByteTracker(max_lost_frames=2)
        first = tracker.update([person(100, 100)])[0][0]
        for _ in range(4):
            tracker.update([])
        self.assertNotEqual(tracker.update([person(100, 100)])[0][0], first)

    def test_deterministic_and_resettable(self):
        scene = crossing_scene()
        self.assertEqual(run(ByteTracker(), scene), run(ByteTracker(), scene))
        tracker = ByteTracker()
        tracker.update([person(10, 10)])
        tracker.reset()
        self.assertEqual(tracker.update([person(10, 10)])[0][0], "trk-000002")

    def test_not_worse_than_greedy_baseline_on_dense_crossing_scene(self):
        # Four swimmers on a collision course; recorded as synthetic evidence only.
        scene = []
        for t in range(50):
            scene.append({"p0": person(60 + 9 * t, 90 + 8 * t, 0.8), "p1": person(500 - 9 * t, 90 + 8 * t, 0.8),
                          "p2": person(60 + 9 * t, 420 - 8 * t, 0.8), "p3": person(500 - 9 * t, 420 - 8 * t, 0.8)})
        byte_switches = id_switches(run(ByteTracker(), scene))
        greedy_switches = id_switches(run(GreedyIoUTracker(), scene))
        self.assertLessEqual(byte_switches, greedy_switches)
        self.assertEqual(byte_switches, 0)

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            ByteTracker(high_threshold=1.5)
        with self.assertRaises(ValueError):
            ByteTracker(max_lost_frames=-1)


if __name__ == "__main__":
    unittest.main()
