"""Feature extraction from MediaPipe FaceLandmarker output.

All functions here take a flat sequence of landmarks — any object exposing
float ``.x`` and ``.y`` in normalised image coordinates (0..1, origin at the
top-left, y growing downwards). That is exactly what MediaPipe returns, and
it is also trivial to fake in tests, so nothing in this module imports
mediapipe.

Every function returns ``None`` rather than raising when the geometry is
degenerate (zero-size iris, zero-width eye, landmarks missing), because a
single bad frame is normal and the caller simply skips it.
"""

from __future__ import annotations

from math import hypot
from typing import NamedTuple, Protocol, Sequence


class LandmarkLike(Protocol):
    """Structural type for a MediaPipe normalised landmark."""

    x: float
    y: float


class Point(NamedTuple):
    """Minimal :class:`LandmarkLike` implementation, for tests and fakes."""

    x: float
    y: float


# --- Landmark indices ---------------------------------------------------
# MediaPipe FaceLandmarker with refined (iris) landmarks emits 478 points.
# The four iris ring points per eye form a circle around the pupil;
# averaging them yields a far steadier centre than the single-point
# landmarks 468/473, and the ring diameter is an anatomical constant
# (~11.7 mm in adults) which lets us normalise out camera distance.

LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_IRIS_RING = (469, 470, 471, 472)   # right, top, left, bottom (image axes)
RIGHT_IRIS_RING = (474, 475, 476, 477)
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# The iris offset is measured against a reference point on the eye. Using
# just the two corners makes that reference only as steady as two
# landmarks, and its noise lands straight in the gaze signal.
#
# These sets add, for each corner, the two eye-contour landmarks either
# side of it in the FaceMesh ring. Averaging six points instead of two cuts
# the reference noise by roughly sqrt(3), and the two extra points at each
# corner sit one above and one below it, so the first-order effect of a
# lowering eyelid cancels in the centroid instead of leaking into the
# vertical gaze channel.
#
# Ring order around the 33/133 eye is
#   33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
# so 33 is flanked by 246 and 7, and 133 by 155 and 173.
LEFT_EYE_ANCHORS = (33, 7, 246, 133, 155, 173)
# Ring order around the 362/263 eye is
#   362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
RIGHT_EYE_ANCHORS = (362, 382, 398, 263, 249, 466)

#: Reference modes accepted by :func:`eye_signal`.
REFERENCE_MODES = ("anchors", "corners")

# Upper / lower eyelid mid-points, for the eye-aspect ratio.
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Eyebrow mid-points.
LEFT_BROW_MID = 105
RIGHT_BROW_MID = 334

#: Number of landmarks the refined model emits. Anything shorter means the
#: model was built without iris refinement and cannot drive gaze tracking.
REQUIRED_LANDMARK_COUNT = 478

#: Smallest distance treated as non-degenerate, in normalised image units.
EPS = 1e-6


def has_iris_landmarks(landmarks: Sequence[LandmarkLike]) -> bool:
    """True when the landmark list is long enough to contain iris points."""
    return len(landmarks) >= REQUIRED_LANDMARK_COUNT


def _centroid(
    landmarks: Sequence[LandmarkLike], indices: Sequence[int]
) -> tuple[float, float]:
    n = len(indices)
    sx = sy = 0.0
    for i in indices:
        p = landmarks[i]
        sx += p.x
        sy += p.y
    return sx / n, sy / n


def eye_signal(
    landmarks: Sequence[LandmarkLike],
    ring_idx: Sequence[int],
    reference_idx: Sequence[int],
) -> tuple[float, float] | None:
    """Iris-centre offset from a reference point on the eye, for one eye.

    The result is expressed in *iris diameters*. Scaling by the iris — not
    by eye width — gives the vertical channel a dynamic range comparable to
    the horizontal one, and makes the signal invariant to how far the user
    sits from the camera.

    Args:
        landmarks: Full landmark list.
        ring_idx: The four iris ring landmarks.
        reference_idx: Landmarks whose centroid is the "looking straight
            ahead" reference. Two corners works; the six-point anchor sets
            are steadier, see :data:`LEFT_EYE_ANCHORS`.

    Returns ``None`` if the iris ring is degenerate.
    """
    if not has_iris_landmarks(landmarks):
        return None
    ix, iy = _centroid(landmarks, ring_idx)
    r0, r1, r2, r3 = (landmarks[i] for i in ring_idx)
    # Opposing ring points give two diameter estimates (horizontal and
    # vertical axes); averaging them is robust to the ring being squashed
    # by a partly lowered eyelid.
    diam = (hypot(r0.x - r2.x, r0.y - r2.y)
            + hypot(r1.x - r3.x, r1.y - r3.y)) * 0.5
    if diam < EPS:
        return None
    mid_x, mid_y = _centroid(landmarks, reference_idx)
    return (ix - mid_x) / diam, (iy - mid_y) / diam


def eye_reference(eye: str, mode: str = "anchors") -> tuple[int, ...]:
    """Reference landmark set for one eye.

    Args:
        eye: ``"left"`` or ``"right"``, from the subject's perspective.
        mode: ``"anchors"`` for the steadier six-point set, ``"corners"``
            for the two corner landmarks on their own.
    """
    if mode not in REFERENCE_MODES:
        raise ValueError(f"unknown reference mode: {mode!r}")
    if eye == "left":
        return (LEFT_EYE_ANCHORS if mode == "anchors"
                else (LEFT_EYE_LEFT, LEFT_EYE_RIGHT))
    if eye == "right":
        return (RIGHT_EYE_ANCHORS if mode == "anchors"
                else (RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT))
    raise ValueError(f"unknown eye: {eye!r}")


def single_eye_signal(
    landmarks: Sequence[LandmarkLike], eye: str, mode: str = "anchors"
) -> tuple[float, float] | None:
    """Gaze signal for one named eye."""
    ring = LEFT_IRIS_RING if eye == "left" else RIGHT_IRIS_RING
    return eye_signal(landmarks, ring, eye_reference(eye, mode))


def fuse_eyes(
    left: tuple[float, float] | None,
    right: tuple[float, float] | None,
    dominant: str = "right",
    max_disagreement: float = 0.06,
) -> tuple[tuple[float, float] | None, str]:
    """Combine the two eyes when they agree, fall back when they do not.

    Both eyes measure the same gaze direction with independent landmark
    noise, so averaging them cuts that noise by a factor of sqrt(2) — the
    cheapest precision there is, and the reason not to simply pick one eye
    and discard the other.

    It only works while the two agree. Strabismus, a drooping lid over one
    eye, or an eye partly out of frame make one reading wrong rather than
    merely noisy, and averaging a wrong reading with a good one just moves
    the error around. When the two disagree by more than
    ``max_disagreement`` iris diameters, the dominant eye is used alone.

    Returns:
        ``((nx, ny), which)`` where ``which`` is ``"both"``, ``"left"`` or
        ``"right"``, or ``(None, "none")`` if neither eye is usable.
    """
    if left is None and right is None:
        return None, "none"
    if left is None:
        return right, "right"
    if right is None:
        return left, "left"
    if (abs(left[0] - right[0]) > max_disagreement
            or abs(left[1] - right[1]) > max_disagreement):
        chosen = left if dominant == "left" else right
        return chosen, dominant
    return ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5), "both"


def normalized_iris_pair(
    landmarks: Sequence[LandmarkLike],
    dominant_eye: str = "right",
    mode: str = "anchors",
    max_disagreement: float = 0.06,
) -> tuple[float, float] | None:
    """Gaze signal for the active eye(s), in iris-diameter units.

    ``dominant_eye="auto"`` is the precise option: it averages both eyes
    while they agree and drops to the right eye when they do not. ``"both"``
    averages unconditionally, ``"left"``/``"right"`` use one eye only.
    Sides are given from the subject's perspective.
    """
    if dominant_eye in ("auto", "both"):
        left = single_eye_signal(landmarks, "left", mode)
        right = single_eye_signal(landmarks, "right", mode)
        if dominant_eye == "both":
            if left is None or right is None:
                return None
            return (left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5
        fused, _ = fuse_eyes(left, right, "right", max_disagreement)
        return fused
    if dominant_eye in ("left", "right"):
        return single_eye_signal(landmarks, dominant_eye, mode)
    raise ValueError(f"unknown dominant_eye: {dominant_eye!r}")


def ear(
    landmarks: Sequence[LandmarkLike],
    top_idx: int,
    bottom_idx: int,
    left_idx: int,
    right_idx: int,
) -> float | None:
    """Eye Aspect Ratio — vertical lid gap over horizontal eye width.

    Roughly 0.30 for an open eye and below 0.15 when fully closed. Both
    terms scale with face size, so the ratio is invariant to camera
    distance. Used to detect winks.
    """
    top = landmarks[top_idx]
    bot = landmarks[bottom_idx]
    left = landmarks[left_idx]
    right = landmarks[right_idx]
    vertical = hypot(top.x - bot.x, top.y - bot.y)
    horizontal = hypot(left.x - right.x, left.y - right.y)
    if horizontal < EPS:
        return None
    return vertical / horizontal


def ear_pair(
    landmarks: Sequence[LandmarkLike],
) -> tuple[float | None, float | None]:
    """(left EAR, right EAR) in one call."""
    return (
        ear(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
            LEFT_EYE_LEFT, LEFT_EYE_RIGHT),
        ear(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
            RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT),
    )


def brow_height(landmarks: Sequence[LandmarkLike]) -> float | None:
    """Mean brow-to-eyelid gap, normalised by inter-iris distance.

    Inter-iris distance is the standard anatomical scale reference — the
    eyes do not move relative to each other — so dividing by it cancels
    camera distance. A raise of both brows pushes this well above its
    resting value, which the gesture layer uses as a double-click trigger.
    """
    if not has_iris_landmarks(landmarks):
        return None
    lb = landmarks[LEFT_BROW_MID]
    rb = landmarks[RIGHT_BROW_MID]
    le_top = landmarks[LEFT_EYE_TOP]
    re_top = landmarks[RIGHT_EYE_TOP]
    li = landmarks[LEFT_IRIS_CENTER]
    ri = landmarks[RIGHT_IRIS_CENTER]
    # Image y grows downwards, so a brow sitting above the lid gives a
    # positive gap here.
    left_gap = le_top.y - lb.y
    right_gap = re_top.y - rb.y
    inter_iris = hypot(li.x - ri.x, li.y - ri.y)
    if inter_iris < EPS:
        return None
    return ((left_gap + right_gap) * 0.5) / inter_iris


def visualisation_indices(
    dominant_eye: str = "right",
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(iris ring indices, eye corner indices) to draw for the active eye."""
    if dominant_eye == "left":
        return LEFT_IRIS_RING, (LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
    if dominant_eye == "right":
        return RIGHT_IRIS_RING, (RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
    if dominant_eye in ("both", "auto"):
        # "auto" reads both eyes and only drops one when they disagree, so
        # showing both is what matches what the tracker is actually doing.
        return (
            LEFT_IRIS_RING + RIGHT_IRIS_RING,
            (LEFT_EYE_LEFT, LEFT_EYE_RIGHT, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT),
        )
    raise ValueError(f"unknown dominant_eye: {dominant_eye!r}")
