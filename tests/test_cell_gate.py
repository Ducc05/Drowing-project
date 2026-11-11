"""Cell-gated drowning alerts: a rescuer who keeps moving across cells must not look like a drowning person."""
import math
import unittest

from PIL import Image

from drowning.config import Settings
from drowning.inference.cells import CellGrid, parse_cell_grid
from drowning.inference.h20saver import H20SaverInference
from drowning.inference.tracking import Box

DROWNING, SWIMMING = 0, 1


class TrackedDetector:
    """Feeds explicit (track_id, box, ontology index) detections, one list per frame."""

    def __init__(self, frames):
        self.frames = list(frames)

    def track(self, image):
        return self.frames.pop(0)

    def detect(self, image):
        return [(box, index) for _, box, index in self.track(image)]


def box_at(cx, cy, half=5.0, confidence=0.9):
    return Box(cx - half, cy - half, cx + half, cy + half, confidence=confidence)


def run(frames, *, grid=(4, 4), live=False, step_ms=100, fps=10, window_s=3.0, size=(100, 100)):
    pipeline = H20SaverInference(TrackedDetector(frames), model_version="cell-test",
                                 cell_grid=grid, cell_window_s=window_s)
    pipeline.configure_live_stream(fps) if live else pipeline.configure_stream(fps)
    image = Image.new("RGB", size)
    return pipeline, [pipeline.process_frame(image, i * step_ms, i) for i in range(len(frames))]


def stationary(count, cx=60, cy=60, track="trk-a"):
    return [[(track, box_at(cx, cy), DROWNING)] for _ in range(count)]


def crossing(count, start=10, end=90, cy=20, track="trk-b"):
    """A misclassified rescuer who swims straight across the frame."""
    return [[(track, box_at(start + (end - start) * i / (count - 1), cy), DROWNING)] for i in range(count)]


def alerted(records, track):
    return [any(r["alert"] for r in frame if r["track_id"] == track) for frame in records]


class CellGridTests(unittest.TestCase):
    def test_box_centre_selects_its_cell_and_boundary_jitter_credits_the_neighbour(self):
        grid = CellGrid(4, 4)
        self.assertEqual(grid.cells_for_box(box_at(60, 60), 100, 100), frozenset({(2, 2)}))
        near_edge = grid.cells_for_box(box_at(26, 60), 100, 100)  # 1px past the x=25 boundary
        self.assertEqual(near_edge, frozenset({(0, 2), (1, 2)}))

    def test_cells_are_clipped_to_the_frame(self):
        grid = CellGrid(4, 4)
        self.assertEqual(grid.cells_for_box(box_at(0, 0), 100, 100), frozenset({(0, 0)}))
        self.assertEqual(grid.cells_for_box(box_at(100, 100), 100, 100), frozenset({(3, 3)}))

    def test_parse_cell_grid(self):
        self.assertEqual(parse_cell_grid("6x4"), (6, 4))
        self.assertEqual(parse_cell_grid(" 8 X 6 "), (8, 6))
        for off in ("off", "none", "0", "false", "OFF"):
            self.assertIsNone(parse_cell_grid(off))
        for bad in ("abc", "0x3", "4x0", "33x4", "4", "4x4x4"):
            with self.assertRaises(ValueError, msg=bad):
                parse_cell_grid(bad)


class CellGatedAlertTests(unittest.TestCase):
    def test_a_person_staying_in_one_cell_alerts_after_the_cell_window(self):
        _, records = run(stationary(40))
        flags = [r[0]["alert"] for r in records]
        self.assertTrue(any(flags))
        self.assertFalse(flags[0])
        # 3 s window at 10 fps with a 0.7 ratio: the 21st vote, later than the 2 s per-track window alone.
        self.assertEqual(flags.index(True), 20)
        self.assertTrue(all(flags[20:]))

    def test_a_person_moving_across_cells_never_alerts_but_does_without_the_gate(self):
        frames = crossing(30)
        _, gated = run(frames)
        self.assertFalse(any(alerted(gated, "trk-b")))
        _, ungated = run(crossing(30), grid=None)
        self.assertTrue(any(alerted(ungated, "trk-b")))  # control: the old behaviour raised the false alarm

    def test_the_suppressed_alarm_is_reported_instead_of_silently_dropped(self):
        _, records = run(crossing(30))
        flat = [r for frame in records for r in frame]
        self.assertTrue(any(r["alert_suppressed_by_cell"] for r in flat))
        self.assertFalse(any(r["alert"] for r in flat))

    def test_boundary_jitter_does_not_break_a_stationary_alarm(self):
        frames = [[("trk-a", box_at(24 if i % 2 else 26, 60), DROWNING)] for i in range(40)]
        _, records = run(frames)
        self.assertTrue(any(alerted(records, "trk-a")))

    def test_victim_alerts_while_a_rescuer_moving_through_other_cells_does_not(self):
        rescuer, victim = crossing(40), stationary(40)
        frames = [r + v for r, v in zip(rescuer, victim)]
        _, records = run(frames)
        self.assertTrue(any(alerted(records, "trk-a")))
        self.assertFalse(any(alerted(records, "trk-b")))

    def test_a_swimming_person_in_the_cell_never_alerts(self):
        frames = [[("trk-a", box_at(60, 60), SWIMMING)] for _ in range(40)]
        _, records = run(frames)
        self.assertFalse(any(alerted(records, "trk-a")))

    def test_alarm_clears_after_the_person_disappears(self):
        frames = stationary(30) + [[] for _ in range(60)]
        pipeline, records = run(frames)
        self.assertTrue(any(alerted(records[:30], "trk-a")))
        self.assertEqual(pipeline.active_alert_tracks, [])
        interval, = pipeline.result_metadata()["alert_intervals"]
        self.assertGreater(interval["end"], interval["start"])

    def test_active_alert_tracks_and_policy_reflect_the_gate(self):
        pipeline, _ = run(stationary(40))
        self.assertEqual(pipeline.active_alert_tracks, ["trk-a"])
        meta = pipeline.result_metadata()
        self.assertEqual(meta["inference_policy"], "h20saver_cell_gated_votes_v1")
        self.assertEqual(meta["cell_gate"], {"cols": 4, "rows": 4, "window_s": 3.0})
        ungated, _ = run(stationary(5), grid=None)
        self.assertEqual(ungated.result_metadata()["inference_policy"], "h20saver_reference_votes_v1")
        self.assertNotIn("cell_gate", ungated.result_metadata())

    def test_reset_stream_clears_cell_history(self):
        pipeline, _ = run(stationary(40))
        pipeline.reset_stream()
        self.assertEqual(pipeline.active_alert_tracks, [])
        pipeline.detector.frames = stationary(5)
        record, = pipeline.process_frame(Image.new("RGB", (100, 100)), 0, 0)
        self.assertFalse(record["alert"])

    def test_frame_size_comes_from_numpy_frames_too(self):
        import numpy as np
        pipeline = H20SaverInference(TrackedDetector(stationary(40)), model_version="cell-test", cell_grid=(4, 4))
        pipeline.configure_stream(10)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)  # 200 wide: the x=60 centre is in column 1, not 2
        records = [pipeline.process_frame(frame, i * 100, i)[0] for i in range(40)]
        self.assertTrue(any(r["alert"] for r in records))


class LiveCellGateTests(unittest.TestCase):
    def test_live_stationary_alerts_and_moving_does_not(self):
        _, still = run(stationary(60), live=True)
        self.assertTrue(any(alerted(still, "trk-a")))
        _, moving = run(crossing(30), live=True)
        self.assertFalse(any(alerted(moving, "trk-b")))


class CellSettingsTests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        default = Settings.from_env({})
        self.assertEqual((default.h20saver_cell_grid, default.h20saver_cell_window_seconds), ((6, 4), 3.0))
        custom = Settings.from_env({"H20SAVER_CELL_GRID": "8x6", "H20SAVER_CELL_WINDOW_SECONDS": "4.5"})
        self.assertEqual((custom.h20saver_cell_grid, custom.h20saver_cell_window_seconds), ((8, 6), 4.5))
        self.assertIsNone(Settings.from_env({"H20SAVER_CELL_GRID": "off"}).h20saver_cell_grid)

    def test_invalid_values_are_rejected(self):
        for env in ({"H20SAVER_CELL_GRID": "nope"}, {"H20SAVER_CELL_WINDOW_SECONDS": "0"},
                    {"H20SAVER_CELL_WINDOW_SECONDS": "abc"}, {"H20SAVER_CELL_WINDOW_SECONDS": "99"}):
            with self.assertRaises(ValueError, msg=str(env)):
                Settings.from_env(env)


if __name__ == "__main__":
    unittest.main()


class CellOverlayTests(unittest.TestCase):
    def hot_pipeline(self):
        pipeline, records = run(stationary(40))  # person at (60, 60): cell (2, 2) of the 4x4 grid on 100x100
        return pipeline, records

    def test_overlay_lists_the_hot_cells_only_while_they_are_alarmed(self):
        pipeline, _ = self.hot_pipeline()
        self.assertEqual(pipeline.cell_overlay, {"cols": 4, "rows": 4, "hot": [(2, 2)]})
        off, _ = run(stationary(5), grid=None)
        self.assertIsNone(off.cell_overlay)

    def test_hot_cell_intervals_are_reported_for_the_web_player(self):
        pipeline, _ = self.hot_pipeline()
        interval, = pipeline.result_metadata()["cell_intervals"]
        self.assertEqual(interval["cell"], [2, 2])
        self.assertEqual(interval["start"], 2000)  # 21st vote at 100 ms steps
        self.assertGreater(interval["end"], interval["start"])
        self.assertNotIn("cell_intervals", run(stationary(5), grid=None)[0].result_metadata())

    def test_cell_intervals_close_when_the_cell_cools_and_reset_clears_them(self):
        pipeline, _ = run(stationary(30) + [[] for _ in range(60)])
        interval, = pipeline.result_metadata()["cell_intervals"]
        self.assertLess(interval["end"], 9000)
        pipeline.reset_stream()
        self.assertEqual(pipeline.result_metadata()["cell_intervals"], [])

    def test_rendered_frame_tints_hot_cells_and_leaves_others_clear(self):
        import numpy as np
        pipeline, _ = self.hot_pipeline()
        frame = pipeline.draw_annotations(np.zeros((100, 100, 3), dtype=np.uint8), [])
        blue, green, red = frame[72, 72].tolist()  # inside hot cell (2, 2), away from the box
        self.assertGreater(red, 40)
        self.assertLess(max(blue, green), 20)
        self.assertEqual(frame[87, 12].tolist(), [0, 0, 0])  # cell (0, 3): not hot, not on a grid line
        self.assertGreater(max(frame[50, 12].tolist()), 0)  # the y=50 grid line is faintly visible

    def test_rendering_without_the_gate_draws_no_grid(self):
        import numpy as np
        pipeline, _ = run(stationary(40), grid=None)
        frame = pipeline.draw_annotations(np.zeros((100, 100, 3), dtype=np.uint8), [])
        self.assertEqual(frame[50, 12].tolist(), [0, 0, 0])

    def test_preview_jpeg_shows_the_same_overlay(self):
        import cv2
        import numpy as np
        from drowning.inference.preview import preview_jpeg
        pipeline, _ = self.hot_pipeline()
        jpeg = preview_jpeg(np.zeros((100, 100, 3), dtype=np.uint8), [], 4000, [], pipeline.cell_overlay)
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        blue, green, red = frame[72, 72].tolist()
        self.assertGreater(red, 30)
        self.assertLess(max(blue, green), 40)
        plain = preview_jpeg(np.zeros((100, 100, 3), dtype=np.uint8), [], 4000, [])
        self.assertLess(cv2.imdecode(np.frombuffer(plain, dtype=np.uint8), cv2.IMREAD_COLOR)[72, 72].max(), 10)


class ScriptDetector:
    """Detections come from ``script(n)`` for the n-th processed frame (n counts from 0)."""

    def __init__(self, script):
        self.script, self.calls = script, 0

    def track(self, image):
        found = self.script(self.calls)
        self.calls += 1
        return found

    def detect(self, image):
        return [(box, index) for _, box, index in self.track(image)]


class RecordingPipeline(H20SaverInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alert_frames, self.suppressed_frames, self.motion_suppressed_frames = [], 0, 0

    def process_frame(self, image, timestamp_ms, frame_index):
        records = super().process_frame(image, timestamp_ms, frame_index)
        if any(r["alert"] for r in records):
            self.alert_frames.append(frame_index)
        self.suppressed_frames += sum(r["alert_suppressed_by_cell"] for r in records)
        self.motion_suppressed_frames += sum(r["alert_suppressed_by_motion"] for r in records)
        return records


class LiveCameraSessionTests(unittest.TestCase):
    """The whole live path: CameraSession -> reset_stream/configure_live_stream -> process_frame -> draw -> JPEG."""

    def run_camera(self, script, *, grid=(4, 4), frames=50, **gates):
        from tests.test_camera import CameraSession, ScriptedCapture, wait_for

        class TenFpsCapture(ScriptedCapture):
            def get(self, prop):
                return 10.0  # cv2.CAP_PROP_FPS

        detector = ScriptDetector(script)
        pipeline = RecordingPipeline(detector, model_version="camera-test", cell_grid=grid, cell_window_s=3.0, **gates)
        # One processed frame = 100 ms, whatever the machine speed: the camera derives timestamps from this clock.
        session = CameraSession(pipeline, 0, display_source="webcam:0", clock=lambda: detector.calls * 0.1,
                                capture_factory=lambda source: TenFpsCapture(source, frame_delay=0.002))
        session.start()
        try:
            self.assertTrue(wait_for(lambda: session.frames_processed >= frames or session.state == "failed", timeout=30))
            self.assertEqual(session.state, "running", session.error)
            jpeg, alert_tracks = session.latest_jpeg(), session.summary()["alert_track_ids"]
        finally:
            session.stop()
        return pipeline, jpeg, alert_tracks

    @staticmethod
    def stationary_script(n):
        return [("trk-a", box_at(40, 30, half=3), DROWNING)]  # 64x48 camera frame: cell (2, 2) of the 4x4 grid

    @staticmethod
    def crossing_script(n):
        # A misclassified rescuer swims across the 64 px wide frame in 3 s and leaves it.
        return [("trk-b", box_at(8 + 48 * n / 29, 12, half=3), DROWNING)] if n < 30 else []

    def test_stationary_person_raises_the_alert_through_the_camera_session(self):
        pipeline, _, alert_tracks = self.run_camera(self.stationary_script, frames=45)
        self.assertTrue(pipeline.alert_frames)
        # Live clock: first alert once 70% of the 3 s cell window (~2.1 s at 100 ms/frame) has accumulated.
        self.assertGreaterEqual(pipeline.alert_frames[0], 20)
        self.assertLessEqual(pipeline.alert_frames[0], 24)

    def test_the_rescuer_crossing_the_view_never_raises_the_alert_through_the_camera_session(self):
        pipeline, _, alert_tracks = self.run_camera(self.crossing_script, frames=50)
        self.assertEqual(pipeline.alert_frames, [])
        self.assertEqual(alert_tracks, [])
        self.assertGreater(pipeline.suppressed_frames, 0)  # the per-track alarm did fire and the gate stopped it

    def test_motion_condition_alone_stops_the_crossing_swimmer_through_the_camera_session(self):
        pipeline, _, alert_tracks = self.run_camera(self.crossing_script, grid=None, frames=50,
                                                    motion_max_body_lengths=1.5)
        self.assertEqual((pipeline.alert_frames, alert_tracks), ([], []))
        self.assertGreater(pipeline.motion_suppressed_frames, 0)
        self.assertEqual(pipeline.suppressed_frames, 0)  # the cell gate was off

    def test_motion_condition_still_lets_a_stationary_person_alert_through_the_camera_session(self):
        pipeline, _, alert_tracks = self.run_camera(self.stationary_script, grid=None, frames=40,
                                                    motion_max_body_lengths=1.5)
        self.assertEqual(alert_tracks, ["trk-a"])
        self.assertLessEqual(pipeline.alert_frames[0], 15)  # no cell window delay: the 2 s track window alone

    def test_without_the_gate_the_same_crossing_does_raise_the_alert(self):
        pipeline, _, _ = self.run_camera(self.crossing_script, grid=None, frames=50)
        self.assertTrue(pipeline.alert_frames)  # control: this is the false alarm the gate removes

    def test_the_streamed_jpeg_carries_the_tinted_hot_cell(self):
        import cv2
        import numpy as np
        _, jpeg, alert_tracks = self.run_camera(self.stationary_script, frames=45)
        self.assertEqual(alert_tracks, ["trk-a"])
        frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR).astype(int)
        hot_b, _, hot_r = frame[34, 45]    # inside cell (2, 2), below the alert banner, clear of the box
        cold_b, _, cold_r = frame[42, 10]  # cell (0, 3): not hot
        self.assertGreater(hot_r - hot_b, 40)
        self.assertLess(abs(cold_r - cold_b), 15)


class DiagnosticFieldTests(unittest.TestCase):
    """Per-box fields that show *why* a drowning label has not (yet) become an alert."""

    def test_vote_and_track_ratio_are_reported_each_frame(self):
        _, records = run(stationary(30), grid=None)
        first, last = records[0][0], records[-1][0]
        self.assertIs(first["drowning_vote"], True)
        self.assertAlmostEqual(first["drowning_ratio"], 1 / 20, places=2)  # 2 s window of 20 samples at 10 fps
        self.assertGreaterEqual(last["drowning_ratio"], 0.7)
        self.assertTrue(last["alert"])

    def test_a_drowning_label_below_the_vote_confidence_never_builds_a_ratio(self):
        weak = [[("trk-a", box_at(60, 60, confidence=0.4), DROWNING)] for _ in range(40)]
        _, records = run(weak, grid=None)
        self.assertEqual(records[0][0]["label"], "drowning")
        self.assertTrue(all(r[0]["drowning_vote"] is False and r[0]["drowning_ratio"] == 0 for r in records))
        self.assertFalse(any(r[0]["alert"] for r in records))

    def test_a_flickering_label_shows_the_ratio_that_stays_below_the_alarm(self):
        flicker = [[("trk-a", box_at(60, 60), DROWNING if i % 2 else SWIMMING)] for i in range(40)]
        _, records = run(flicker, grid=None)
        self.assertLess(max(r[0]["drowning_ratio"] for r in records), 0.7)
        self.assertFalse(any(r[0]["alert"] for r in records))

    def test_each_new_track_id_restarts_the_ratio(self):
        switching = [[(f"trk-{i // 10}", box_at(60, 60), DROWNING)] for i in range(40)]
        _, records = run(switching, grid=None)
        self.assertLess(max(r[0]["drowning_ratio"] for r in records), 0.7)  # 10 votes per ID never fill 70% of 2 s

    def test_live_ratio_is_time_weighted(self):
        _, records = run(stationary(40), grid=None, live=True)
        self.assertGreaterEqual(records[-1][0]["drowning_ratio"], 0.7)
        self.assertLess(records[3][0]["drowning_ratio"], 0.3)


class LargeBodyCellTests(unittest.TestCase):
    """A person close to the camera covers many cells: moving a little must not look like leaving the cell."""

    def test_a_large_box_credits_every_cell_it_covers(self):
        grid = CellGrid(6, 4)
        big = Box(100, 40, 480, 420, confidence=0.9)  # 380 px body in a 640x480 frame (cells are ~107x120 px)
        cells = grid.cells_for_box(big, 640, 480)
        self.assertEqual({c for c, _ in cells}, {0, 1, 2, 3, 4})
        self.assertEqual({r for _, r in cells}, {0, 1, 2, 3})

    def test_a_small_box_still_only_credits_its_centre_cell_and_edge_neighbours(self):
        grid = CellGrid(4, 4)
        self.assertEqual(grid.cells_for_box(box_at(60, 60), 100, 100), frozenset({(2, 2)}))
        self.assertEqual(grid.cells_for_box(box_at(26, 60), 100, 100), frozenset({(0, 2), (1, 2)}))

    @staticmethod
    def swinging(size, swing, count=280):
        def at(n):
            t = n / 14
            cx = 320 + swing * math.sin(2 * math.pi * 0.6 * t)
            cy = 240 + swing * math.sin(2 * math.pi * 0.45 * t + 1)
            return [("trk-a", Box(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2, confidence=0.8), DROWNING)]
        return [at(n) for n in range(count)]

    def run_live(self, frames, **gates):
        pipeline = H20SaverInference(TrackedDetector(frames), model_version="large", **gates)
        pipeline.configure_live_stream(14)
        image = Image.new("RGB", (640, 480))
        return [pipeline.process_frame(image, round(n * 1000 / 14), n)[0] for n in range(len(frames))]

    def test_a_close_up_person_swinging_a_fraction_of_a_body_length_still_alerts_with_the_grid(self):
        for swing in (30, 60, 90, 130):  # all well under one body length of a 380 px person
            records = self.run_live(self.swinging(380, swing), cell_grid=(6, 4))
            share = sum(r["alert"] for r in records) / len(records)
            self.assertGreater(share, 0.7, f"swing {swing}px: alert only {share:.0%} of the time")

    def test_close_up_person_with_both_gates_alerts_but_a_fast_swimmer_does_not(self):
        both = {"cell_grid": (6, 4), "motion_max_body_lengths": 1.5}
        victim = self.run_live(self.swinging(150, 40), **both)
        self.assertGreater(sum(r["alert"] for r in victim) / len(victim), 0.7)
        swimmer = [[("trk-b", Box(20 + 600 * min(n, 41) / 41 - 75, 165, 20 + 600 * min(n, 41) / 41 + 75, 315, confidence=0.8),
                     DROWNING)] if n < 42 else [] for n in range(120)]  # crosses ~4 body lengths in 3 s, then leaves
        pipeline = H20SaverInference(TrackedDetector(swimmer), model_version="large", **both)
        pipeline.configure_live_stream(14)
        image = Image.new("RGB", (640, 480))
        seen = []
        for n in range(len(swimmer)):
            seen += pipeline.process_frame(image, round(n * 1000 / 14), n)
        self.assertFalse(any(r["alert"] for r in seen))
        self.assertTrue(any(r["alert_suppressed_by_motion"] or r["alert_suppressed_by_cell"] for r in seen))
