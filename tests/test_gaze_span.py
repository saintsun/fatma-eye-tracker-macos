"""Gaze span, and the steadiness gate on confirming a calibration point.

Both come out of one measured session. Its calibration passed through every
control point and still put the cursor 200 px from the target, because the
gaze had travelled only 0.079 iris diameters vertically across the whole
screen — so the mapping multiplied every pixel of landmark noise by twelve
thousand. Residuals cannot see that; the span can.
"""

import pytest

from fatma.calibration import Calibrator
from fatma.config import Settings
from fatma.quality import (
    MIN_GOOD_SPAN_X,
    MIN_GOOD_SPAN_Y,
    GazeSpan,
    gaze_span,
)

#: The session that prompted all of this.
MEASURED_BAD = GazeSpan(x=0.1749, y=0.0790)
#: An earlier session on the same machine, three times more accurate.
MEASURED_GOOD = GazeSpan(x=0.4013, y=0.1995)


class TestGazeSpan:
    def test_measures_the_travel_of_a_point_set(self):
        span = gaze_span([(0.1, -0.2), (0.4, -0.1), (0.2, -0.15)])
        assert span.x == pytest.approx(0.3)
        assert span.y == pytest.approx(0.1)

    def test_a_degenerate_set_spans_nothing(self):
        assert gaze_span([(0.1, -0.2)]) == GazeSpan(0.0, 0.0)
        assert gaze_span([]) == GazeSpan(0.0, 0.0)

    def test_the_measured_bad_session_is_flagged(self):
        assert not MEASURED_BAD.ok
        assert "yakin otur" in MEASURED_BAD.advice()

    def test_the_measured_good_session_passes(self):
        assert MEASURED_GOOD.ok
        assert MEASURED_GOOD.advice() == ""

    def test_amplification_explains_the_observed_error(self):
        """The arithmetic that makes the span matter: with this span, a
        third of a pixel of landmark noise is a hundred pixels of cursor."""
        _, per_unit_y = MEASURED_BAD.amplification(1470, 956)
        assert per_unit_y == pytest.approx(12095, rel=0.01)
        # An iris about 36 px across in a 1280-wide frame is 0.028 of the
        # frame; a third of a pixel of noise is this much cursor error.
        iris_norm = 36 / 1280
        noise_px = 0.3 / 1280
        assert (noise_px / iris_norm) * per_unit_y > 80

    def test_a_wider_span_amplifies_less(self):
        bad = MEASURED_BAD.amplification(1470, 956)[1]
        good = MEASURED_GOOD.amplification(1470, 956)[1]
        assert good < bad / 2

    def test_thresholds_sit_between_the_two_measured_sessions(self):
        assert MEASURED_BAD.y < MIN_GOOD_SPAN_Y < MEASURED_GOOD.y
        assert MEASURED_BAD.x < MIN_GOOD_SPAN_X < MEASURED_GOOD.x

    def test_advice_is_renderable_on_the_overlay(self):
        assert MEASURED_BAD.advice().isascii()


def gaze_for(target, scale=(0.30, 0.20)):
    tx, ty = target
    return scale[0] * tx + 0.05, scale[1] * ty - 0.25


def run(cal, scale=(0.30, 0.20), settle=None):
    cal.start()
    for target in cal.POINTS_SCREEN:
        nx, ny = gaze_for(target, scale)
        for _ in range(settle or cal.min_samples + 2):
            cal.feed(nx, ny)
        assert cal.confirm(), "a settled point was refused"


class TestSpanIsMeasuredOnFit:
    def test_a_healthy_calibration_reports_a_healthy_span(self):
        cal = Calibrator(min_samples=1)
        run(cal)
        assert cal.span is not None
        assert cal.span.ok

    def test_a_cramped_calibration_is_flagged(self):
        """The same nine targets, but the eyes barely moved — which is what
        sitting too far from the screen does."""
        cal = Calibrator(min_samples=1)
        run(cal, scale=(0.12, 0.05))
        assert not cal.span.ok
        assert cal.span.advice()

    def test_residuals_alone_do_not_catch_it(self):
        """The point of having the span at all: a cramped calibration fits
        its own control points perfectly."""
        cal = Calibrator(min_samples=1)
        run(cal, scale=(0.12, 0.05))
        assert cal.quality.valid
        assert cal.quality.mean < 0.05, "residuals look fine"
        assert not cal.span.ok, "but the span does not"


class TestSteadinessGate:
    def test_a_point_is_refused_while_the_gaze_is_still_moving(self):
        """Regression: five samples at 30 fps is a sixth of a second, so a
        key pressed as the target appeared captured the gaze mid-flight and
        bent the whole spline."""
        cal = Calibrator(min_samples=3, sample_window=9, stability=0.02)
        cal.start()
        for x in (0.5, 0.4, 0.3, 0.2):      # still travelling
            cal.feed(x, -0.2)
        assert not cal.ready_to_confirm
        assert cal.confirm() is False
        assert cal.idx == 0

    def test_it_is_accepted_once_the_gaze_settles(self):
        cal = Calibrator(min_samples=3, sample_window=9, stability=0.02)
        cal.start()
        for x in (0.5, 0.4, 0.3):
            cal.feed(x, -0.2)
        for _ in range(9):
            cal.feed(0.2, -0.2)
        assert cal.ready_to_confirm
        assert cal.confirm() is True

    def test_steadiness_is_infinite_before_enough_samples(self):
        cal = Calibrator(min_samples=5)
        cal.start()
        cal.feed(0.2, -0.2)
        assert cal.steadiness == float("inf")

    def test_steadiness_reports_the_spread(self):
        cal = Calibrator(min_samples=3, sample_window=5)
        cal.start()
        for x in (0.20, 0.23, 0.21):
            cal.feed(x, -0.2)
        assert cal.steadiness == pytest.approx(0.03)

    def test_a_tremor_inside_tolerance_still_confirms(self):
        cal = Calibrator(min_samples=5, sample_window=9, stability=0.02)
        cal.start()
        for i in range(9):
            cal.feed(0.2 + (0.005 if i % 2 else -0.005), -0.2)
        assert cal.confirm() is True

    def test_the_default_sample_count_is_no_longer_a_sixth_of_a_second(self):
        assert Settings().calib_min_samples >= 10
        assert Settings().calib_stability > 0
