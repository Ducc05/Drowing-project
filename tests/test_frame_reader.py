"""Background frame reader: ordering, stride, early stop and error propagation, with a fake capture."""
import threading
import time
import unittest

import numpy as np

from drowning.inference.frame_reader import FrameReader


class FakeCapture:
    def __init__(self, count, fail_at=None, retrieve_fail_at=None, delay=0.0):
        self.count, self.fail_at, self.retrieve_fail_at, self.delay = count, fail_at, retrieve_fail_at, delay
        self.position, self.retrieved, self.released = -1, [], False

    def grab(self):
        if self.fail_at is not None and self.position + 1 == self.fail_at:
            raise RuntimeError("decoder exploded")
        time.sleep(self.delay)
        self.position += 1
        return self.position < self.count

    def retrieve(self):
        if self.position == self.retrieve_fail_at:
            return False, None
        self.retrieved.append(self.position)
        return True, np.full((2, 2, 3), self.position % 255, dtype=np.uint8)

    def get(self, prop):
        return self.position * 10.0  # POS_MSEC stand-in

    def release(self):
        self.released = True


class FrameReaderTests(unittest.TestCase):
    def read(self, capture, **kwargs):
        reader = FrameReader(capture, pos_msec_prop=0, **kwargs)
        with reader:
            items = list(reader)
        return reader, items

    def test_every_frame_in_order_with_container_timestamp(self):
        reader, items = self.read(FakeCapture(5), frame_stride=1)
        self.assertEqual([(i.index, i.process, i.pos_msec) for i in items],
                         [(n, True, n * 10.0) for n in range(5)])
        self.assertEqual(reader.frames_read, 5)

    def test_skipped_frames_are_grabbed_but_not_decoded_unless_requested(self):
        capture = FakeCapture(10)
        reader, items = self.read(capture, frame_stride=3)
        self.assertEqual([i.index for i in items], [0, 3, 6, 9])
        self.assertEqual(capture.retrieved, [0, 3, 6, 9])
        self.assertEqual(reader.frames_read, 10)

    def test_skipped_frames_are_delivered_when_a_video_is_being_rewritten(self):
        capture = FakeCapture(7)
        _, items = self.read(capture, frame_stride=3, keep_skipped=True)
        self.assertEqual([(i.index, i.process) for i in items],
                         [(0, True), (1, False), (2, False), (3, True), (4, False), (5, False), (6, True)])
        self.assertTrue(all(i.frame is not None for i in items))

    def test_max_frames_limits_reading(self):
        reader, items = self.read(FakeCapture(100), frame_stride=1, max_frames=4)
        self.assertEqual(len(items), 4)
        self.assertEqual(reader.frames_read, 4)

    def test_a_failed_retrieve_ends_the_stream_without_counting_that_frame(self):
        reader, items = self.read(FakeCapture(10, retrieve_fail_at=3), frame_stride=1)
        self.assertEqual([i.index for i in items], [0, 1, 2])
        self.assertEqual(reader.frames_read, 3)

    def test_reader_errors_are_raised_in_the_consumer(self):
        with self.assertRaisesRegex(RuntimeError, "decoder exploded"):
            self.read(FakeCapture(10, fail_at=2), frame_stride=1)

    def test_closing_early_stops_the_thread_and_releases_the_capture(self):
        capture = FakeCapture(10_000, delay=0.001)
        before = threading.active_count()
        reader = FrameReader(capture, pos_msec_prop=0, frame_stride=1, queue_size=2)
        with reader:
            next(iter(reader))
        self.assertTrue(capture.released)
        self.assertEqual(threading.active_count(), before)
        self.assertLess(reader.frames_read, 100)  # bounded queue: it never decodes the whole file ahead


if __name__ == "__main__":
    unittest.main()
