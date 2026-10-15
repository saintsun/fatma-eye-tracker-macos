"""The gaze-to-pixel pipeline: deadzone, clamping, freezing and drift."""

import pytest

from fatma.calibration import Calibrator
from fatma.config import Settings
from fatma.cursor import CursorController

SCREEN_W, SCREEN_H = 1000, 800


def gaze_for(target):
    tx, ty = target
    return 0.30 * tx + 0.05, 0.20 * ty - 0.25


def calibrated():
    cal = Calibrator(min_samples=1)
    cal.start()
    for target in cal.POINTS_SCREEN:
        cal.feed(*gaze_for(target))
        cal.confirm()
    assert cal.is_calibrated
    return cal


def make_controller(**overrides):
    settings = Settings(**{
        "median_window": 1,          # keep the median out of the way
        "smoothing_index": 0,
        **overrides,
    })
    return CursorController(settings, calibrated(), SCREEN_W, SCREEN_H)


def settle(ctrl, target, frames=80, t0=0.0):
    """Run one gaze reading until the filters converge."""
    nx, ny = gaze_for(target)
    pos = ctrl.position
    for i in range(frames):
        pos = ctrl.update(nx, ny, t0 + i / 30)
    return pos


class TestPipeline:
    def test_cursor_reaches_the_calibrated_target(self):
        ctrl = make_controller(deadzone_px=0.0)
        x, y = settle(ctrl, (0.5, 0.5))
        assert x == pytest.approx(SCREEN_W * 0.5, abs=5)
        assert y == pytest.approx(SCREEN_H * 0.5, abs=5)

    def test_cursor_stays_inside_the_screen(self):
        ctrl = make_controller(deadzone_px=0.0)
        for nx, ny in [(-99.0, -99.0), (99.0, 99.0)]:
            for i in range(40):
                x, y = ctrl.update(nx, ny, i / 30)
            assert 0 <= x <= SCREEN_W - 1
            assert 0 <= y <= SCREEN_H - 1

    def test_starts_at_the_screen_centre(self):
        ctrl = make_controller()
        assert ctrl.position == (SCREEN_W * 0.5, SCREEN_H * 0.5)


class TestDeadzone:
    def test_small_movements_are_suppressed(self):
        """Breath and landmark tremor must not jiggle the cursor."""
        ctrl = make_controller(deadzone_px=20.0)
        settle(ctrl, (0.5, 0.5))
        before = ctrl.position
        nx, ny = gaze_for((0.505, 0.5))     # ~5 px of travel
        for i in range(40):
            ctrl.update(nx, ny, 10.0 + i / 30)
        assert ctrl.position == before

    def test_real_movements_get_through(self):
        ctrl = make_controller(deadzone_px=20.0)
        settle(ctrl, (0.2, 0.2))
        before = ctrl.position
        after = settle(ctrl, (0.8, 0.8), t0=10.0)
        assert abs(after[0] - before[0]) > 100

    def test_zero_deadzone_tracks_every_change(self):
        ctrl = make_controller(deadzone_px=0.0)
        settle(ctrl, (0.5, 0.5))
        before = ctrl.position
        nx, ny = gaze_for((0.51, 0.5))
        ctrl.update(nx, ny, 10.0)
        assert ctrl.position != before


class TestFreeze:
    def test_frozen_frames_do_not_move_the_cursor(self):
        """While an eyelid is closing the iris reading is garbage, so the
        click must land where the user was last looking."""
        ctrl = make_controller(deadzone_px=0.0)
        settle(ctrl, (0.2, 0.2))
        held = ctrl.position
        nx, ny = gaze_for((0.9, 0.9))
        for i in range(40):
            ctrl.update(nx, ny, 10.0 + i / 30, frozen=True)
        assert ctrl.position == held

    def test_filters_keep_running_while_frozen(self):
        """So that unfreezing does not replay a stale backlog."""
        ctrl = make_controller(deadzone_px=0.0)
        settle(ctrl, (0.2, 0.2))
        nx, ny = gaze_for((0.9, 0.9))
        for i in range(60):
            ctrl.update(nx, ny, 10.0 + i / 30, frozen=True)
        # One unfrozen frame should now land near the new target, because
        # the filter state already caught up during the freeze.
        x, y = ctrl.update(nx, ny, 12.0)
        assert x == pytest.approx(SCREEN_W * 0.9, abs=30)


class TestDrift:
    def test_anchor_centres_the_cursor(self):
        ctrl = make_controller(deadzone_px=0.0)
        settle(ctrl, (0.2, 0.3))
        ctrl.anchor()
        x, y = settle(ctrl, (0.2, 0.3), t0=10.0)
        assert x == pytest.approx(SCREEN_W * 0.5, abs=5)
        assert y == pytest.approx(SCREEN_H * 0.5, abs=5)

    def test_anchor_is_correct_when_the_prediction_was_clamped(self):
        """Regression: the offset used to be recovered by subtracting the
        drift back out of the final position. At a screen edge that
        position has been clamped, so the recovered prediction was wrong
        and anchoring there bent the mapping instead of shifting it."""
        ctrl = make_controller(deadzone_px=0.0)
        # Drive hard into the top-left corner so the output clamps.
        nx, ny = gaze_for((-2.0, -2.0))
        for i in range(60):
            ctrl.update(nx, ny, i / 30)
        assert ctrl.position == (0.0, 0.0), "expected a clamped position"

        ctrl.anchor()
        for i in range(60):
            x, y = ctrl.update(nx, ny, 10.0 + i / 30)
        assert x == pytest.approx(SCREEN_W * 0.5, abs=5)
        assert y == pytest.approx(SCREEN_H * 0.5, abs=5)

    def test_anchor_shifts_without_distorting(self):
        """An anchor is a pure translation: relative geometry is preserved."""
        ctrl = make_controller(deadzone_px=0.0)
        a_before = settle(ctrl, (0.3, 0.3))
        b_before = settle(ctrl, (0.7, 0.7), t0=10.0)

        settle(ctrl, (0.3, 0.3), t0=20.0)
        ctrl.anchor()
        a_after = settle(ctrl, (0.3, 0.3), t0=30.0)
        b_after = settle(ctrl, (0.7, 0.7), t0=40.0)

        assert (b_after[0] - a_after[0]) == pytest.approx(
            b_before[0] - a_before[0], abs=2
        )

    def test_reset_drift_restores_the_raw_mapping(self):
        ctrl = make_controller(deadzone_px=0.0)
        before = settle(ctrl, (0.3, 0.7))
        settle(ctrl, (0.1, 0.1), t0=10.0)
        ctrl.anchor()
        ctrl.reset_drift()
        assert ctrl.drift_dx == 0.0 and ctrl.drift_dy == 0.0
        assert settle(ctrl, (0.3, 0.7), t0=20.0) == pytest.approx(before, abs=2)


class TestConfiguration:
    def test_set_smoothing_installs_the_new_preset(self):
        ctrl = make_controller()
        ctrl.set_smoothing(0.15, 0.15)
        assert ctrl.filter_x.min_cutoff == pytest.approx(0.15)
        assert ctrl.filter_y.min_cutoff == pytest.approx(0.15 * 0.6)

    def test_set_smoothing_clears_filter_history(self):
        ctrl = make_controller()
        settle(ctrl, (0.5, 0.5))
        ctrl.set_smoothing(0.3, 0.3)
        assert ctrl.filter_x.x_prev is None

    def test_reset_filters_clears_everything(self):
        ctrl = make_controller()
        settle(ctrl, (0.5, 0.5))
        ctrl.reset_filters()
        assert ctrl.filter_x.x_prev is None
        assert len(ctrl.median_x) == 0

    def test_rejects_a_degenerate_screen(self):
        with pytest.raises(ValueError, match="positive"):
            CursorController(Settings(), calibrated(), 0, 800)
