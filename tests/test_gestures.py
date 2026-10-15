"""Wink and brow-raise recognition.

Every test drives the detector frame by frame on a synthetic 30 fps clock,
so hold times and cooldowns are exercised at the resolution the real loop
runs at.
"""

import pytest

from fatma.gestures import Gesture, GestureDetector

DT = 1 / 30
OPEN = 0.30          # comfortably above ear_open
SHUT = 0.05          # comfortably below ear_closed
AJAR = 0.19          # inside the hysteresis band


@pytest.fixture
def det():
    return GestureDetector(baseline_warmup=3)


def drive(det, frames, start=100.0, brow=None, enabled=True):
    """Feed (left_ear, right_ear) pairs and collect everything emitted."""
    fired = []
    for i, (left, right) in enumerate(frames):
        fired += det.update(left, right, brow, start + i * DT, enabled=enabled)
    return fired


def hold(left, right, frames):
    return [(left, right)] * frames


class TestWinks:
    def test_left_wink_fires_a_primary_click(self):
        det = GestureDetector(baseline_warmup=3)
        fired = drive(det, hold(SHUT, OPEN, 12))
        assert fired == [Gesture.LEFT_CLICK]

    def test_right_wink_fires_a_secondary_click(self):
        det = GestureDetector(baseline_warmup=3)
        fired = drive(det, hold(OPEN, SHUT, 12))
        assert fired == [Gesture.RIGHT_CLICK]

    def test_a_blink_closes_both_eyes_and_clicks_nothing(self):
        """The single most important negative case: ordinary blinking must
        not scatter clicks across the screen."""
        det = GestureDetector(baseline_warmup=3)
        fired = drive(det, hold(SHUT, SHUT, 30))
        assert fired == []

    def test_a_wink_shorter_than_the_hold_does_nothing(self):
        det = GestureDetector(closed_hold_s=0.20, baseline_warmup=3)
        fired = drive(det, hold(SHUT, OPEN, 4) + hold(OPEN, OPEN, 4))
        assert fired == []

    def test_a_wink_fires_once_not_once_per_frame(self):
        det = GestureDetector(baseline_warmup=3)
        fired = drive(det, hold(SHUT, OPEN, 60))
        assert fired == [Gesture.LEFT_CLICK]

    def test_reopening_allows_a_second_wink(self):
        det = GestureDetector(cooldown_s=0.1, baseline_warmup=3)
        frames = hold(SHUT, OPEN, 10) + hold(OPEN, OPEN, 10) + hold(SHUT, OPEN, 10)
        assert drive(det, frames) == [Gesture.LEFT_CLICK, Gesture.LEFT_CLICK]

    def test_half_closed_eye_does_not_latch(self):
        """Hysteresis: the band between the thresholds is a no-man's land
        so a trembling eyelid cannot chatter."""
        det = GestureDetector(baseline_warmup=3)
        assert drive(det, hold(AJAR, OPEN, 30)) == []
        assert det.left_closed is False

    def test_an_eye_must_reopen_past_the_upper_threshold(self):
        det = GestureDetector(cooldown_s=0.0, baseline_warmup=3)
        # Shut, click, then only come back up into the band: still latched,
        # so no second click.
        frames = hold(SHUT, OPEN, 10) + hold(AJAR, OPEN, 20)
        assert drive(det, frames) == [Gesture.LEFT_CLICK]
        assert det.left_closed is True

    def test_missing_ear_values_are_ignored(self):
        det = GestureDetector(baseline_warmup=3)
        assert drive(det, [(None, None)] * 20) == []
        assert drive(det, [(SHUT, None)] * 20) == []


class TestCooldown:
    def test_cooldown_blocks_a_rapid_second_gesture(self):
        det = GestureDetector(cooldown_s=1.0, baseline_warmup=3)
        frames = hold(SHUT, OPEN, 10) + hold(OPEN, OPEN, 3) + hold(SHUT, OPEN, 10)
        assert drive(det, frames) == [Gesture.LEFT_CLICK]

    def test_cooldown_expires(self):
        det = GestureDetector(cooldown_s=0.2, baseline_warmup=3)
        frames = (hold(SHUT, OPEN, 10) + hold(OPEN, OPEN, 12)
                  + hold(SHUT, OPEN, 10))
        assert drive(det, frames) == [Gesture.LEFT_CLICK, Gesture.LEFT_CLICK]

    def test_a_wink_and_a_brow_raise_on_one_frame_emit_one_gesture(self):
        """Regression: the cooldown used to be evaluated once per frame
        before any gesture fired, so two gestures coming due together both
        got through and the user got a click plus a double-click."""
        det = GestureDetector(cooldown_s=1.0, baseline_warmup=1,
                              brow_enabled=True)
        # Seed a resting baseline, then raise the brows and wink together.
        det.update(OPEN, OPEN, 0.50, 0.0)
        fired = []
        for i in range(1, 20):
            fired += det.update(SHUT, OPEN, 0.80, i * DT)
        assert len(fired) == 1


class TestBrowRaise:
    def test_it_is_off_by_default(self):
        """Looking at the top of the screen raises the eyebrows by itself,
        so the gesture fires exactly when the user is reaching for the menu
        bar. A real session produced five unasked-for double clicks."""
        det = GestureDetector(baseline_warmup=1)
        assert det.brow_enabled is False
        det.update(OPEN, OPEN, 0.50, 0.0)
        fired = []
        for i in range(1, 30):
            fired += det.update(OPEN, OPEN, 0.90, i * DT)
        assert fired == []

    def test_a_raised_brow_does_not_freeze_the_cursor_when_disabled(self):
        det = GestureDetector(baseline_warmup=1)
        det.update(OPEN, OPEN, 0.50, 0.0)
        assert det.is_in_progress(OPEN, OPEN, 0.90) is False

    def test_sustained_raise_fires_a_double_click(self):
        det = GestureDetector(baseline_warmup=1, brow_enabled=True)
        det.update(OPEN, OPEN, 0.50, 0.0)
        fired = []
        for i in range(1, 20):
            fired += det.update(OPEN, OPEN, 0.80, i * DT)
        assert fired == [Gesture.DOUBLE_CLICK]

    def test_a_brief_raise_does_nothing(self):
        det = GestureDetector(brow_hold_s=0.2, baseline_warmup=1,
                              brow_enabled=True)
        det.update(OPEN, OPEN, 0.50, 0.0)
        fired = []
        for i in range(1, 4):
            fired += det.update(OPEN, OPEN, 0.80, i * DT)
        for i in range(4, 10):
            fired += det.update(OPEN, OPEN, 0.50, i * DT)
        assert fired == []

    def test_a_raise_below_the_ratio_does_nothing(self):
        det = GestureDetector(brow_raise_ratio=1.15, baseline_warmup=1,
                              brow_enabled=True)
        det.update(OPEN, OPEN, 0.50, 0.0)
        fired = []
        for i in range(1, 30):
            fired += det.update(OPEN, OPEN, 0.55, i * DT)   # only +10%
        assert fired == []


class TestBrowBaseline:
    def test_baseline_is_seeded_from_a_median_not_the_first_frame(self):
        """A startup frame caught mid-raise must not become the resting
        reference for the whole session."""
        det = GestureDetector(baseline_warmup=5)
        for i, value in enumerate([0.90, 0.50, 0.50, 0.52, 0.48]):
            det.update(OPEN, OPEN, value, i * DT)
        assert det.brow_baseline == pytest.approx(0.50)

    def test_baseline_is_unset_until_warmup_completes(self):
        det = GestureDetector(baseline_warmup=5)
        det.update(OPEN, OPEN, 0.5, 0.0)
        assert det.brow_baseline is None

    def test_a_held_raise_does_not_drag_the_baseline_up(self):
        """Otherwise holding the brows up would re-centre the detector and
        the gesture would stop working."""
        det = GestureDetector(baseline_warmup=1, brow_ema_alpha=0.2)
        det.update(OPEN, OPEN, 0.50, 0.0)
        baseline = det.brow_baseline
        for i in range(1, 100):
            det.update(OPEN, OPEN, 0.90, i * DT)
        assert det.brow_baseline == pytest.approx(baseline)

    def test_slow_posture_drift_is_absorbed(self):
        det = GestureDetector(baseline_warmup=1, brow_ema_alpha=0.2)
        det.update(OPEN, OPEN, 0.50, 0.0)
        for i in range(1, 200):
            det.update(OPEN, OPEN, 0.52, i * DT)
        assert det.brow_baseline == pytest.approx(0.52, abs=1e-3)


class TestEnabledFlag:
    def test_disabled_detector_emits_nothing(self):
        det = GestureDetector(baseline_warmup=3)
        assert drive(det, hold(SHUT, OPEN, 20), enabled=False) == []

    def test_state_still_advances_while_disabled(self):
        """So that re-enabling mid-wink does not fire a stale click."""
        det = GestureDetector(baseline_warmup=3)
        drive(det, hold(SHUT, OPEN, 20), enabled=False)
        assert det.left_closed is True

    def test_disabled_frames_do_not_consume_the_cooldown(self):
        det = GestureDetector(cooldown_s=5.0, baseline_warmup=3)
        drive(det, hold(SHUT, OPEN, 20), enabled=False)
        det.update(OPEN, OPEN, None, 101.0)
        fired = []
        for i in range(30):
            fired += det.update(SHUT, OPEN, None, 102.0 + i * DT)
        assert fired == [Gesture.LEFT_CLICK]


class TestInProgress:
    def test_a_closing_eye_freezes_the_cursor(self):
        """The eyelid covers the iris ring, so those frames' gaze readings
        are meaningless and the cursor must hold still."""
        det = GestureDetector()
        assert det.is_in_progress(SHUT, OPEN, None) is True
        assert det.is_in_progress(AJAR, OPEN, None) is True

    def test_open_eyes_do_not_freeze_the_cursor(self):
        assert GestureDetector().is_in_progress(OPEN, OPEN, None) is False

    def test_a_raised_brow_freezes_the_cursor(self):
        det = GestureDetector(baseline_warmup=1, brow_enabled=True)
        det.update(OPEN, OPEN, 0.50, 0.0)
        assert det.is_in_progress(OPEN, OPEN, 0.80) is True
        assert det.is_in_progress(OPEN, OPEN, 0.50) is False

    def test_missing_values_do_not_freeze_the_cursor(self):
        assert GestureDetector().is_in_progress(None, None, None) is False


class TestLifecycle:
    def test_reset_clears_every_latch(self):
        det = GestureDetector(baseline_warmup=1)
        drive(det, hold(SHUT, OPEN, 20), brow=0.5)
        det.reset()
        assert det.left_closed is False
        assert det.brow_baseline is None
        assert det.last_fired_at == float("-inf")

    def test_state_snapshot_reports_the_latches(self):
        det = GestureDetector(baseline_warmup=3)
        drive(det, hold(SHUT, OPEN, 20))
        state = det.state(SHUT, OPEN, None)
        assert state.left_closed is True
        assert state.right_closed is False
        assert state.in_progress is True

    @pytest.mark.parametrize("kwargs", [
        {"ear_open": 0.1, "ear_closed": 0.2},
        {"brow_guard_ratio": 1.5, "brow_raise_ratio": 1.2},
        {"brow_ema_alpha": 0.0},
        {"brow_ema_alpha": 1.0},
    ])
    def test_rejects_contradictory_thresholds(self, kwargs):
        with pytest.raises(ValueError):
            GestureDetector(**kwargs)
