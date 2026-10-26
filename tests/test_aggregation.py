import unittest

from drowning.inference.aggregation import DEFAULT_AGGREGATION, TrackAggregator, validate_aggregation

CONFIG = {"method": "ema_probabilities", "tau_ms": 1000, "max_gap_ms": 1500, "min_frames": 3, "alert_label": "drowning",
          "enter_probability": 0.6, "exit_probability": 0.4, "min_alert_duration_ms": 1000}
DROWN, SWIM, OUT = [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]
MIDDLE = [0.5, 0.45, 0.05]  # drowning probability between the exit (0.4) and enter (0.6) levels


def feed(agg, track, scores, start=0, step=250):
    return [agg.update(track, start + i * step, probs) for i, probs in enumerate(scores)]


class AggregationTests(unittest.TestCase):
    def test_no_label_until_enough_frames_then_label_follows_the_smoothed_scores(self):
        results = feed(TrackAggregator(CONFIG), "t1", [SWIM] * 5)
        self.assertIsNone(results[0]["label"])
        self.assertFalse(results[1]["track_ready"])
        self.assertEqual((results[2]["label"], results[-1]["label"]), ("swimming", "swimming"))

    def test_a_single_noisy_frame_does_not_flip_the_label_or_raise_an_alert(self):
        results = feed(TrackAggregator(CONFIG), "t1", [SWIM] * 6 + [DROWN] + [SWIM] * 4)
        self.assertTrue(all(r["label"] == "swimming" for r in results[2:]))
        self.assertFalse(any(r["alert"] for r in results))

    def test_sustained_drowning_raises_the_alert_late_enough_holds_in_the_middle_band_then_clears(self):
        agg = TrackAggregator(CONFIG)
        results = feed(agg, "t1", [SWIM] * 4 + [DROWN] * 12)
        first_alert = next(i for i, r in enumerate(results) if r["alert"])
        self.assertGreater(first_alert, 4)  # not instant: the average must rise and stay above the enter level
        self.assertTrue(all(r["alert"] for r in results[first_alert:]))
        clock = 16 * 250
        middle = feed(agg, "t1", [MIDDLE] * 20, start=clock)
        self.assertTrue(all(r["alert"] for r in middle))  # hysteresis keeps it between the two levels
        self.assertTrue(0.4 <= middle[-1]["smoothed_scores"][0] < 0.6)
        cleared = feed(agg, "t1", [SWIM] * 10, start=clock + 20 * 250)
        self.assertFalse(cleared[-1]["alert"])

    def test_a_dip_between_the_thresholds_cannot_complete_an_alert_that_never_held(self):
        agg = TrackAggregator({**CONFIG, "tau_ms": 200, "min_alert_duration_ms": 750})
        results = feed(agg, "t1", [DROWN] * 2 + [MIDDLE] * 4 + [DROWN] * 2)
        self.assertFalse(any(r["alert"] for r in results))

    def test_long_gap_or_out_of_order_time_starts_a_fresh_observation(self):
        agg = TrackAggregator(CONFIG)
        feed(agg, "t1", [DROWN] * 6)
        after_gap = agg.update("t1", 10_000, SWIM)
        self.assertEqual((after_gap["track_frames"], after_gap["label"]), (1, None))
        self.assertEqual(agg.update("t1", 9_000, SWIM)["track_frames"], 1)

    def test_tracks_are_independent_and_reset_forgets_them(self):
        agg = TrackAggregator(CONFIG)
        a, b = feed(agg, "a", [DROWN] * 6)[-1], feed(agg, "b", [OUT] * 6)[-1]
        self.assertEqual((a["label"], b["label"]), ("drowning", "out_of_water"))
        agg.reset("a")
        self.assertEqual(agg.update("a", 1500, OUT)["track_frames"], 1)
        self.assertEqual(agg.update("b", 1500, OUT)["track_frames"], 7)
        agg.reset()
        self.assertEqual(agg.update("b", 1750, OUT)["track_frames"], 1)

    def test_invalid_contracts_are_rejected_and_the_default_is_valid(self):
        self.assertEqual(validate_aggregation(CONFIG), [])
        self.assertEqual(validate_aggregation(DEFAULT_AGGREGATION), [])
        for bad in ({}, {**CONFIG, "exit_probability": 0.7}, {**CONFIG, "alert_label": "nope"}, {**CONFIG, "tau_ms": 0},
                    {**CONFIG, "method": "median"}, {**CONFIG, "min_frames": 0}, {**CONFIG, "min_frames": True}):
            self.assertTrue(validate_aggregation(bad), bad)
            with self.assertRaises(ValueError):
                TrackAggregator(bad)


if __name__ == "__main__":
    unittest.main()
