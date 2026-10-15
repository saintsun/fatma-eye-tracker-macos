"""Feature extraction from landmark geometry."""

import math

import pytest

from conftest import EYE_WIDTH, INTER_IRIS, IRIS_R, make_face
from fatma.landmarks import (
    LEFT_EYE_LEFT,
    LEFT_EYE_RIGHT,
    LEFT_IRIS_RING,
    RIGHT_EYE_LEFT,
    RIGHT_EYE_RIGHT,
    RIGHT_IRIS_RING,
    Point,
    brow_height,
    ear,
    ear_pair,
    eye_signal,
    has_iris_landmarks,
    normalized_iris_pair,
    visualisation_indices,
)


class TestEyeSignal:
    def test_centred_iris_gives_zero_offset(self):
        signal = eye_signal(make_face(gaze=(0.0, 0.0)), RIGHT_IRIS_RING,
                            (RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT))
        assert signal == pytest.approx((0.0, 0.0), abs=1e-12)

    @pytest.mark.parametrize("gaze", [
        (0.5, 0.0), (-0.5, 0.0), (0.0, 0.25), (0.0, -0.25), (0.3, -0.2),
    ])
    def test_offset_is_reported_in_iris_diameters(self, gaze):
        signal = eye_signal(make_face(gaze=gaze), RIGHT_IRIS_RING,
                            (RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT))
        assert signal == pytest.approx(gaze, abs=1e-12)

    def test_signal_is_invariant_to_camera_distance(self):
        """Scaling the whole face must not change the reported gaze.

        This is the property that lets the user lean in or sit back without
        recalibrating, so it is worth pinning down explicitly.
        """
        near = make_face(gaze=(0.4, -0.2), iris_r=IRIS_R)
        far = make_face(gaze=(0.4, -0.2), iris_r=IRIS_R / 2)
        a = eye_signal(near, RIGHT_IRIS_RING, (RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT))
        b = eye_signal(far, RIGHT_IRIS_RING, (RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT))
        assert a == pytest.approx(b, abs=1e-12)

    def test_degenerate_iris_returns_none(self):
        face = make_face()
        for idx in RIGHT_IRIS_RING:
            face[idx] = Point(0.65, 0.5)
        assert eye_signal(face, RIGHT_IRIS_RING,
                          (RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)) is None

    def test_model_without_iris_landmarks_returns_none(self):
        """A 468-point model has no iris ring; indexing it would raise."""
        face = make_face(n_points=468)
        assert not has_iris_landmarks(face)
        assert eye_signal(face, RIGHT_IRIS_RING,
                          (RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)) is None


class TestNormalizedIrisPair:
    def test_right_eye_selected(self):
        face = make_face(gaze=(0.4, 0.1), left_gaze=(-0.4, -0.1))
        assert normalized_iris_pair(face, "right") == pytest.approx((0.4, 0.1))

    def test_left_eye_selected(self):
        face = make_face(gaze=(0.4, 0.1), left_gaze=(-0.4, -0.1))
        assert normalized_iris_pair(face, "left") == pytest.approx((-0.4, -0.1))

    def test_both_averages_the_eyes(self):
        face = make_face(gaze=(0.4, 0.2), left_gaze=(0.0, 0.0))
        assert normalized_iris_pair(face, "both") == pytest.approx((0.2, 0.1))

    def test_both_returns_none_when_one_eye_is_degenerate(self):
        face = make_face()
        for idx in LEFT_IRIS_RING:
            face[idx] = Point(0.35, 0.5)
        assert normalized_iris_pair(face, "both") is None
        # The good eye still works on its own — which is the whole point of
        # offering a dominant-eye mode.
        assert normalized_iris_pair(face, "right") is not None

    def test_unknown_eye_rejected(self):
        with pytest.raises(ValueError, match="dominant_eye"):
            normalized_iris_pair(make_face(), "middle")


class TestEar:
    @pytest.mark.parametrize("value", [0.30, 0.22, 0.16, 0.05])
    def test_ear_matches_requested_openness(self, value):
        left, right = ear_pair(make_face(left_ear=value, right_ear=value))
        assert left == pytest.approx(value)
        assert right == pytest.approx(value)

    def test_eyes_are_measured_independently(self):
        left, right = ear_pair(make_face(left_ear=0.05, right_ear=0.30))
        assert left == pytest.approx(0.05)
        assert right == pytest.approx(0.30)

    def test_zero_width_eye_returns_none(self):
        face = make_face()
        face[RIGHT_EYE_LEFT] = Point(0.65, 0.5)
        face[RIGHT_EYE_RIGHT] = Point(0.65, 0.5)
        assert ear(face, 386, 374, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT) is None

    def test_ear_is_invariant_to_face_scale(self):
        """EAR divides one facial distance by another, so scale cancels."""
        base = make_face(left_ear=0.3)
        scaled = [Point(p.x * 2, p.y * 2) for p in base]
        assert ear(scaled, 159, 145, LEFT_EYE_LEFT, LEFT_EYE_RIGHT) == (
            pytest.approx(ear(base, 159, 145, LEFT_EYE_LEFT, LEFT_EYE_RIGHT))
        )


class TestBrowHeight:
    @pytest.mark.parametrize("value", [0.3, 0.5, 0.8])
    def test_matches_requested_height(self, value):
        assert brow_height(make_face(brow=value)) == pytest.approx(value)

    def test_raising_brows_increases_the_measure(self):
        assert brow_height(make_face(brow=0.7)) > brow_height(make_face(brow=0.5))

    def test_zero_inter_iris_returns_none(self):
        face = make_face()
        face[468] = Point(0.5, 0.5)
        face[473] = Point(0.5, 0.5)
        assert brow_height(face) is None

    def test_missing_iris_landmarks_returns_none(self):
        assert brow_height(make_face(n_points=468)) is None


class TestVisualisationIndices:
    @pytest.mark.parametrize("eye,expected_ring", [
        ("left", LEFT_IRIS_RING),
        ("right", RIGHT_IRIS_RING),
        ("both", LEFT_IRIS_RING + RIGHT_IRIS_RING),
    ])
    def test_returns_the_active_eye(self, eye, expected_ring):
        ring, corners = visualisation_indices(eye)
        assert ring == expected_ring
        assert len(corners) == (4 if eye == "both" else 2)

    def test_unknown_eye_rejected(self):
        with pytest.raises(ValueError):
            visualisation_indices("nose")


def test_fixture_geometry_is_what_the_docstring_claims():
    """Guard the fixture itself — every other test rests on these numbers."""
    face = make_face()
    assert math.isclose(
        abs(face[RIGHT_EYE_RIGHT].x - face[RIGHT_EYE_LEFT].x), EYE_WIDTH
    )
    assert math.isclose(abs(face[473].x - face[468].x), INTER_IRIS)
