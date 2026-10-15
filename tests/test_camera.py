"""Threaded frame grabbing: freshness, failure and shutdown."""

import threading
import time

import pytest

from conftest import FakeCapture
from fatma.camera import DirectCapture, FrameGrabber


class BlockingCapture:
    """Hands out frames only when the test releases them."""

    def __init__(self):
        self.gate = threading.Event()
        self.served = 0
        self.released = False

    def read(self):
        if not self.gate.wait(timeout=2.0):
            return False, None
        self.served += 1
        return True, f"frame-{self.served}"

    def release(self):
        self.released = True


def wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class TestFrameGrabber:
    def test_serves_a_frame(self):
        with FrameGrabber(FakeCapture(["a", "b", "c"], loop=True)) as grab:
            ok, frame = grab.read()
        assert ok
        assert frame in {"a", "b", "c"}

    def test_each_read_waits_for_a_new_frame(self):
        """Returning the same image twice would burn an inference pass on
        a frame the tracker has already seen."""
        cap = BlockingCapture()
        with FrameGrabber(cap) as grab:
            cap.gate.set()
            ok1, first = grab.read()
            ok2, second = grab.read()
        assert ok1 and ok2
        assert first != second

    def test_stale_frames_are_dropped_not_queued(self):
        """The reason this class exists: a slow consumer must not fall
        progressively further behind the camera."""
        cap = FakeCapture([f"f{i}" for i in range(500)], loop=True)
        with FrameGrabber(cap) as grab:
            grab.read()
            assert wait_for(lambda: cap.reads > 50)
            ok, frame = grab.read()
        assert ok
        # The frame handed over is the newest one captured, not the second
        # one the camera produced.
        assert frame != "f1"
        assert grab.frames_dropped > 0

    def test_reports_failure_when_the_device_stops(self):
        with FrameGrabber(FakeCapture([])) as grab:
            ok, frame = grab.read()
        assert ok is False
        assert frame is None
        assert grab.failed

    def test_read_times_out_rather_than_hanging(self):
        cap = BlockingCapture()      # never gated open
        with FrameGrabber(cap) as grab:
            start = time.monotonic()
            ok, _ = grab.read(timeout=0.1)
            elapsed = time.monotonic() - start
        assert ok is False
        assert elapsed < 1.0

    def test_context_manager_releases_the_device(self):
        cap = FakeCapture(["a"], loop=True)
        with FrameGrabber(cap) as grab:
            grab.read()
        assert cap.released

    def test_stop_is_idempotent(self):
        cap = FakeCapture(["a"], loop=True)
        grab = FrameGrabber(cap).start()
        grab.read()
        grab.stop()
        grab.stop()
        assert cap.released

    def test_stop_survives_a_release_that_raises(self):
        class AngryCapture(FakeCapture):
            def release(self):
                raise RuntimeError("device already gone")

        grab = FrameGrabber(AngryCapture(["a"], loop=True)).start()
        grab.read()
        grab.stop()   # must not propagate

    def test_double_start_is_refused(self):
        grab = FrameGrabber(FakeCapture(["a"], loop=True)).start()
        try:
            with pytest.raises(RuntimeError, match="already started"):
                grab.start()
        finally:
            grab.stop()

    def test_thread_does_not_outlive_the_grabber(self):
        cap = FakeCapture(["a"], loop=True)
        grab = FrameGrabber(cap, name="fatma-test-thread").start()
        grab.read()
        grab.stop()
        assert wait_for(
            lambda: not any(t.name == "fatma-test-thread"
                            for t in threading.enumerate())
        )


class TestDirectCapture:
    def test_passes_frames_straight_through(self):
        cap = FakeCapture(["a", "b"])
        with DirectCapture(cap) as src:
            assert src.read() == (True, "a")
            assert src.read() == (True, "b")
            assert src.read() == (False, None)
            assert src.failed

    def test_counts_frames_and_drops_none(self):
        cap = FakeCapture(["a", "b"])
        with DirectCapture(cap) as src:
            src.read()
            src.read()
        assert src.frames_captured == 2
        assert src.frames_dropped == 0
        assert cap.released

    def test_interface_matches_the_threaded_grabber(self):
        """Both are used interchangeably by the runtime loop."""
        for attr in ("read", "start", "stop", "frames_captured",
                     "frames_dropped", "failed"):
            assert hasattr(DirectCapture(FakeCapture([])), attr)
            assert hasattr(FrameGrabber(FakeCapture([])), attr)
