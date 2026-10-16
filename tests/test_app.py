"""Runtime-loop pieces that can be exercised without a camera."""

import numpy as np
import pytest

from conftest import make_face
from fatma.config import VALID_EYES, Settings

cv2 = pytest.importorskip("cv2")

from fatma.app import TrackerApp, make_target_image  # noqa: E402
from fatma.screen import Rect  # noqa: E402


@pytest.fixture
def app():
    try:
        return TrackerApp(Settings(), pointer_backend="null")
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"no display/pointer available: {exc}")


class TestMediaPipeTimestamps:
    def test_timestamps_strictly_increase(self, app):
        """MediaPipe's VIDEO mode rejects a timestamp that does not
        advance. Once the pointer stopped costing 100 ms per frame, two
        frames landing inside the same millisecond became ordinary."""
        t0 = 0.0
        stamps = [app._next_timestamp_ms(0.0005, t0) for _ in range(5)]
        assert stamps == sorted(stamps)
        assert len(set(stamps)) == len(stamps)

    def test_timestamps_track_real_elapsed_time(self, app):
        t0 = 100.0
        assert app._next_timestamp_ms(100.5, t0) == 500
        assert app._next_timestamp_ms(101.0, t0) == 1000

    def test_a_backwards_clock_still_advances_the_stamp(self, app):
        t0 = 0.0
        first = app._next_timestamp_ms(1.0, t0)
        second = app._next_timestamp_ms(0.5, t0)
        assert second > first


class TestTargetImage:
    """The target is drawn into the window's measured rectangle.

    Regression: the canvas used to be sized by subtracting an assumed
    25-pixel menu bar. Measured, the menu bar is 33 and a full-screen
    OpenCV window covers the whole screen anyway, so the subtraction both
    shrank the image — leaving OpenCV to stretch it — and shifted every dot
    away from the position the calibrator was recording for it.
    """

    def test_the_image_is_exactly_the_canvas_size(self):
        img, _ = make_target_image(400, 300, Rect(0, 0, 800, 600), 60)
        assert img.shape == (600, 800, 3)

    def test_a_canvas_offset_down_the_screen_is_honoured(self):
        """A window that starts below the menu bar: a dot at screen y=100
        belongs 67 pixels down its canvas, not 100."""
        img, actual = make_target_image(400, 100, Rect(0, 33, 800, 567), 60)
        assert img.shape == (567, 800, 3)
        assert actual == pytest.approx((400, 100))
        red = np.argwhere((img[:, :, 2] > 200) & (img[:, :, 1] < 60))
        cy, cx = red.mean(axis=0)
        assert cx == pytest.approx(400, abs=2)
        assert cy == pytest.approx(67, abs=2)

    def test_bullseye_lands_on_the_requested_screen_point(self):
        img, actual = make_target_image(400, 300, Rect(0, 0, 800, 600), 60)
        assert actual == pytest.approx((400, 300))
        red = np.argwhere((img[:, :, 2] > 200) & (img[:, :, 1] < 60))
        cy, cx = red.mean(axis=0)
        assert (cx, cy) == pytest.approx((400, 300), abs=2)

    def test_an_unreachable_point_reports_where_it_landed(self):
        """The dot cannot be drawn above the window, so the calibrator has
        to be told the position it actually reached — fitting against the
        position it asked for would pair a gaze reading with a place the
        dot never was."""
        _, actual = make_target_image(400, 5, Rect(0, 33, 800, 567), 60)
        assert actual[1] > 33
        assert actual[1] == pytest.approx(33 + 32, abs=2)

    def test_a_target_at_an_edge_stays_fully_visible(self):
        for x, y in ((799, 300), (1, 300), (400, 599), (400, 1)):
            img, _ = make_target_image(x, y, Rect(0, 0, 800, 600), 60)
            red = np.argwhere((img[:, :, 2] > 200) & (img[:, :, 1] < 60))
            assert len(red) > 0
            assert 0 <= red[:, 0].min() and red[:, 0].max() < 600
            assert 0 <= red[:, 1].min() and red[:, 1].max() < 800


class TestKeyHandling:
    def test_q_quits(self, app):
        assert app._handle_key(ord('q')) is False

    def test_m_toggles_the_mouse(self, app):
        before = app.mouse_on
        app._handle_key(ord('m'))
        assert app.mouse_on is not before

    def test_s_cycles_smoothing(self, app):
        before = app.s.smoothing_index
        app._handle_key(ord('s'))
        assert app.s.smoothing_index != before
        assert app.cursor.filter_x.min_cutoff == app.s.smoothing()[0]

    def test_c_starts_a_calibration_and_clears_drift(self, app):
        app.cursor.drift_dx = 50.0
        app._handle_key(ord('c'))
        assert app.calibrator.active
        assert app.cursor.drift_dx == 0.0

    def test_escape_cancels_a_calibration(self, app):
        app._handle_key(ord('c'))
        app._handle_key(27)
        assert not app.calibrator.active

    def test_anchor_is_ignored_before_calibration(self, app):
        """Anchoring an uncalibrated mapping would bake in a meaningless
        offset that then survives into the next session."""
        app._handle_key(ord('a'))
        assert app.cursor.drift_dx == 0.0

    def test_capital_a_resets_drift(self, app):
        app.cursor.drift_dx = 42.0
        app._handle_key(ord('A'))
        assert app.cursor.drift_dx == 0.0

    def test_unknown_keys_are_ignored(self, app):
        assert app._handle_key(ord('z')) is True


class TestOverlay:
    """The preview overlay, drawn on every visible frame.

    Regression: nothing exercised this path, because every other test runs
    with the preview off. A mode the overlay did not know about therefore
    sailed through the whole suite and crashed on the first frame that had
    a face in it.
    """

    @staticmethod
    def _frame():
        return np.full((480, 640, 3), 40, dtype=np.uint8)

    @pytest.mark.parametrize("eye", VALID_EYES)
    def test_draws_for_every_eye_mode(self, eye):
        try:
            app = TrackerApp(Settings(dominant_eye=eye, show_preview=False),
                             pointer_backend="null")
        except Exception as exc:                  # pragma: no cover
            pytest.skip(f"no display/pointer available: {exc}")
        frame = self._frame()
        app._draw_overlay(frame, make_face(gaze=(0.2, -0.1)),
                          0.3, 0.3, 0.5, 30.0)
        assert not np.array_equal(frame, self._frame()), "nothing was drawn"

    def test_draws_without_a_face(self, app):
        """Looking away must not crash the overlay; the status text still
        has to render."""
        frame = self._frame()
        app._draw_overlay(frame, None, None, None, None, 0.0)
        assert not np.array_equal(frame, self._frame())

    def test_draws_before_any_calibration(self, app):
        app.calibrator.tps = None
        app.calibrator.quality = None
        frame = self._frame()
        app._draw_overlay(frame, make_face(), 0.3, 0.3, 0.5, 30.0)
        assert not np.array_equal(frame, self._frame())

    def test_draws_with_a_calibration_quality_report(self, app):
        from fatma.quality import CalibrationQuality

        app.calibrator.quality = CalibrationQuality(
            per_point=(0.01, 0.03), mean=0.02, worst=0.03, worst_index=1
        )
        frame = self._frame()
        app._draw_overlay(frame, make_face(), 0.3, 0.3, 0.5, 30.0)
        assert not np.array_equal(frame, self._frame())

    def test_draws_while_gestures_are_latched(self, app):
        app.gestures.left_closed = True
        app.gestures.right_closed = True
        app.gestures.brow_raised = True
        frame = self._frame()
        app._draw_overlay(frame, make_face(), 0.05, 0.05, 0.9, 30.0)
        assert not np.array_equal(frame, self._frame())


class TestFps:
    def test_reports_zero_before_any_frames(self, app):
        assert app.fps == 0.0

    def test_computes_the_rate_over_the_window(self, app):
        for i in range(31):
            app._frame_times.append(i / 30)
        assert app.fps == pytest.approx(30.0, abs=0.01)
