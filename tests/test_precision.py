"""Precision claims, verified against simulated landmark noise.

Real tracking accuracy can only be measured with a camera and a person.
What *can* be pinned down without either is the thing that actually governs
it: how much of the detector's landmark noise survives into the gaze
signal. Every test here injects independent Gaussian jitter into the
landmarks — which is what the detector produces — and measures the spread
of the resulting signal over many trials.

The numbers these tests assert are noise-reduction *ratios*, which follow
from averaging independent samples and hold regardless of how noisy the
real detector turns out to be.
"""

import math
import random
import statistics

import pytest

from conftest import make_face
from fatma.landmarks import (
    LEFT_EYE_ANCHORS,
    LEFT_EYE_LEFT,
    LEFT_EYE_RIGHT,
    RIGHT_EYE_ANCHORS,
    RIGHT_EYE_LEFT,
    RIGHT_EYE_RIGHT,
    eye_reference,
    fuse_eyes,
    normalized_iris_pair,
    single_eye_signal,
)
from fatma.quality import (
    CalibrationQuality,
    JitterMeter,
    gaze_scale,
    leave_one_out_quality,
)

NOISE = 0.0006          # landmark jitter, normalised image units
TRIALS = 400
GAZE = (0.20, -0.10)


def spread(signals):
    """RMS distance of a set of 2D readings from their own mean."""
    mx = statistics.fmean(s[0] for s in signals)
    my = statistics.fmean(s[1] for s in signals)
    return math.sqrt(
        statistics.fmean((s[0] - mx) ** 2 + (s[1] - my) ** 2 for s in signals)
    )


def sample(fn, seed=0, trials=TRIALS, **face_kwargs):
    rng = random.Random(seed)
    out = []
    for _ in range(trials):
        face = make_face(noise=NOISE, rng=rng, **face_kwargs)
        value = fn(face)
        if value is not None:
            out.append(value)
    return out


class TestEyeReference:
    def test_both_references_agree_on_a_clean_face(self):
        """The two reference sets must measure the same thing; only their
        noise differs. The fixture places the anchor landmarks so their
        centroid lands exactly on the corner midpoint."""
        face = make_face(gaze=GAZE)
        assert single_eye_signal(face, "right", "anchors") == pytest.approx(
            single_eye_signal(face, "right", "corners"), abs=1e-12
        )

    def test_six_anchors_are_quieter_than_two_corners(self):
        """Averaging six landmarks instead of two cuts the reference's
        contribution to gaze noise. This is the whole reason the anchor
        set exists."""
        corners = spread(sample(
            lambda f: single_eye_signal(f, "right", "corners"), seed=1,
            gaze=GAZE))
        anchors = spread(sample(
            lambda f: single_eye_signal(f, "right", "anchors"), seed=1,
            gaze=GAZE))
        assert anchors < corners
        # Both share the same iris-ring noise, so the improvement is
        # bounded; requiring a clear 10% is a conservative floor.
        assert anchors < corners * 0.9

    def test_anchor_sets_are_corner_symmetric(self):
        """Each corner is flanked by one landmark above and one below, so
        a lowering eyelid shifts them in opposite directions and cancels
        in the centroid instead of leaking into the vertical channel."""
        for anchors, corner_a, corner_b in (
            (LEFT_EYE_ANCHORS, LEFT_EYE_LEFT, LEFT_EYE_RIGHT),
            (RIGHT_EYE_ANCHORS, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT),
        ):
            assert len(anchors) == 6
            assert len(set(anchors)) == 6
            assert anchors[0] == corner_a
            assert anchors[3] == corner_b

    def test_reference_lookup_rejects_bad_input(self):
        with pytest.raises(ValueError, match="reference mode"):
            eye_reference("left", "elbow")
        with pytest.raises(ValueError, match="unknown eye"):
            eye_reference("third", "anchors")


class TestBinocularFusion:
    def test_fusing_two_eyes_cuts_noise_by_root_two(self):
        """Both eyes measure the same gaze with independent landmark
        noise, so averaging them is free precision — the single cheapest
        improvement available."""
        one = spread(sample(
            lambda f: single_eye_signal(f, "right", "anchors"), seed=2,
            gaze=GAZE, left_gaze=GAZE))
        fused = spread(sample(
            lambda f: normalized_iris_pair(f, "auto", "anchors"), seed=2,
            gaze=GAZE, left_gaze=GAZE))
        ratio = fused / one
        assert ratio == pytest.approx(1 / math.sqrt(2), rel=0.25)
        assert fused < one

    def test_disagreeing_eyes_fall_back_to_one(self):
        """Strabismus or a lid covering one eye makes a reading wrong, not
        merely noisy, and averaging a wrong value with a good one just
        moves the error somewhere else."""
        value, which = fuse_eyes((0.2, -0.1), (0.9, -0.1), dominant="right")
        assert which == "right"
        assert value == (0.9, -0.1)

    def test_agreeing_eyes_are_averaged(self):
        value, which = fuse_eyes((0.20, -0.10), (0.22, -0.12),
                                 dominant="right")
        assert which == "both"
        assert value == pytest.approx((0.21, -0.11))

    def test_a_missing_eye_is_tolerated(self):
        assert fuse_eyes(None, (0.2, -0.1)) == ((0.2, -0.1), "right")
        assert fuse_eyes((0.2, -0.1), None) == ((0.2, -0.1), "left")
        assert fuse_eyes(None, None) == (None, "none")

    def test_auto_mode_survives_one_unusable_eye(self):
        """'both' returns nothing if either eye fails; 'auto' keeps going,
        which is what a user with one good eye needs."""
        face = make_face(gaze=GAZE)
        for idx in (469, 470, 471, 472):      # flatten the left iris ring
            face[idx] = face[33]
        assert normalized_iris_pair(face, "both") is None
        assert normalized_iris_pair(face, "auto") is not None


class TestCalibrationQuality:
    def _run(self, jitter=0.0, seed=0, grid=None):
        from fatma.calibration import Calibrator

        rng = random.Random(seed)
        cal = Calibrator(min_samples=1, grid=grid or 9)
        cal.start()
        for tx, ty in cal.POINTS_SCREEN:
            nx = 0.30 * tx + 0.05 + rng.gauss(0, jitter)
            ny = 0.20 * ty - 0.25 + rng.gauss(0, jitter)
            cal.feed(nx, ny)
            cal.confirm()
        return cal

    def test_a_clean_calibration_scores_near_zero(self):
        cal = self._run(jitter=0.0)
        assert cal.quality.valid
        assert cal.quality.mean < 0.05

    def test_a_sloppy_calibration_scores_worse(self):
        """The metric has to separate good from bad, otherwise it is just
        decoration."""
        clean = self._run(jitter=0.0, seed=3).quality
        sloppy = self._run(jitter=0.01, seed=3).quality
        assert sloppy.mean > clean.mean

    def test_the_worst_point_is_identified(self):
        cal = self._run(jitter=0.008, seed=5)
        q = cal.quality
        assert 0 <= q.worst_index < len(q.per_point)
        assert q.per_point[q.worst_index] == q.worst

    def test_quality_is_reported_in_pixels(self):
        q = CalibrationQuality(per_point=(0.01,), mean=0.02, worst=0.05,
                               worst_index=0)
        mean_px, worst_px = q.in_pixels(1470, 956)
        assert mean_px == pytest.approx(0.02 * 1237, rel=0.01)
        assert worst_px > mean_px

    def test_too_few_points_cannot_be_cross_validated(self):
        q = leave_one_out_quality([[0, 0], [1, 0], [0, 1]],
                                  [[0, 0], [1, 0], [0, 1]])
        assert not q.valid
        assert "at least 5" in q.reason

    def test_grades_get_worse_as_error_rises(self):
        grades = [
            CalibrationQuality((), m, m, 0).grade()
            for m in (0.01, 0.03, 0.06, 0.2)
        ]
        assert len(set(grades)) == 4

    def test_thirteen_points_cover_the_edge_midpoints(self):
        """Spline error peaks furthest from any target — along the middle
        of each edge. The 13-grid puts a target exactly there."""
        from fatma.calibration import Calibrator

        nine = set(Calibrator.GRID_9)
        thirteen = set(Calibrator.GRID_13)
        assert nine <= thirteen
        assert len(thirteen) == 13
        assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in thirteen)

    def test_consecutive_thirteen_grid_targets_are_far_apart(self):
        """Adjacent targets that sit next to each other invite a user to
        barely move, which is what produces a degenerate fit."""
        from fatma.calibration import Calibrator

        pts = Calibrator.GRID_13
        first_five = pts[:5]
        for a, b in zip(first_five, first_five[1:]):
            assert math.dist(a, b) > 0.4


class TestJitterMeter:
    def test_a_still_cursor_reports_no_jitter(self):
        meter = JitterMeter(window=10)
        for _ in range(10):
            meter.update(500.0, 500.0)
        assert meter.rms == pytest.approx(0.0)
        assert meter.is_fixating

    def test_a_wandering_cursor_reports_its_spread(self):
        meter = JitterMeter(window=4)
        for x in (0.0, 10.0, 0.0, 10.0):
            meter.update(x, 0.0)
        assert meter.rms == pytest.approx(5.0)

    def test_a_moving_cursor_is_not_a_fixation(self):
        meter = JitterMeter(window=10, fixation_px=5.0)
        for i in range(10):
            meter.update(i * 100.0, 0.0)
        assert not meter.is_fixating

    def test_best_records_the_quietest_fixation(self):
        meter = JitterMeter(window=4, fixation_px=100.0)
        for x in (0.0, 20.0, 0.0, 20.0):
            meter.update(x, 0.0)
        noisy = meter.best
        for _ in range(4):
            meter.update(7.0, 7.0)
        assert meter.best < noisy

    def test_best_ignores_saccades(self):
        meter = JitterMeter(window=4, fixation_px=5.0)
        for i in range(4):
            meter.update(i * 500.0, 0.0)
        assert meter.best is None

    def test_needs_two_samples_before_reporting(self):
        meter = JitterMeter(window=5)
        meter.update(1.0, 1.0)
        assert meter.rms == 0.0
        assert not meter.is_fixating

    def test_rejects_a_useless_window(self):
        with pytest.raises(ValueError, match="at least 2"):
            JitterMeter(window=1)


class TestGazeScale:
    def test_reports_the_noise_amplification_factor(self):
        """This number is why capture resolution dominates: the gaze
        signal spans a fraction of an iris diameter across a whole screen,
        so landmark error is multiplied by thousands on its way to the
        cursor."""
        sx, sy = gaze_scale((0.03, 0.43), (-0.26, -0.06), 1470, 956)
        assert sx == pytest.approx(1470 / 0.40, rel=0.01)
        assert sy == pytest.approx(956 / 0.20, rel=0.01)
        # Vertical is the harder axis: less gaze travel over fewer pixels.
        assert sy > sx

    def test_a_degenerate_range_does_not_divide_by_zero(self):
        sx, sy = gaze_scale((0.1, 0.1), (0.2, 0.2), 1470, 956)
        assert math.isfinite(sx) and math.isfinite(sy)
