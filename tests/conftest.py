"""Shared fixtures: a synthetic face whose extracted features are exact.

The builder lays out landmarks so that every quantity the tracker derives
comes out at a value the test asked for:

* eye corners are 0.1 apart, so ``ear`` is ``lid_gap / 0.1``
* the iris ring has radius ``r``, so its diameter is ``2r`` and placing the
  centre at ``corner_mid + gaze * 2r`` makes ``eye_signal`` return ``gaze``
* the two iris centres are 0.3 apart, so ``brow_height`` is
  ``brow_gap / 0.3``

That lets tests assert on exact numbers instead of on tolerances pulled out
of thin air.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fatma.landmarks import (  # noqa: E402
    LEFT_BROW_MID,
    LEFT_EYE_ANCHORS,
    LEFT_EYE_BOTTOM,
    LEFT_EYE_LEFT,
    LEFT_EYE_RIGHT,
    LEFT_EYE_TOP,
    LEFT_IRIS_CENTER,
    LEFT_IRIS_RING,
    REQUIRED_LANDMARK_COUNT,
    RIGHT_BROW_MID,
    RIGHT_EYE_ANCHORS,
    RIGHT_EYE_BOTTOM,
    RIGHT_EYE_LEFT,
    RIGHT_EYE_RIGHT,
    RIGHT_EYE_TOP,
    RIGHT_IRIS_CENTER,
    RIGHT_IRIS_RING,
    Point,
)

#: Eye-corner separation; the EAR denominator.
EYE_WIDTH = 0.1
#: How far the corner-flanking anchor landmarks sit above and below their
#: corner. They are placed at the corner's own x and symmetrically in y, so
#: the six-point anchor centroid lands exactly on the two-corner midpoint —
#: which lets a test compare the two references on equal terms.
ANCHOR_V = 0.004
#: Distance between the two iris centres; the brow-height denominator.
INTER_IRIS = 0.3
#: Iris ring radius, so the iris diameter is 2 * IRIS_R.
IRIS_R = 0.01

LEFT_EYE_CX, RIGHT_EYE_CX = 0.35, 0.65
EYE_CY = 0.5


def _place_ring(points, ring_idx, cx, cy, r):
    """Ring order is (right, top, left, bottom) in image axes."""
    right, top, left, bottom = ring_idx
    points[right] = Point(cx + r, cy)
    points[top] = Point(cx, cy - r)
    points[left] = Point(cx - r, cy)
    points[bottom] = Point(cx, cy + r)


def make_face(
    gaze: tuple[float, float] = (0.0, 0.0),
    left_gaze: tuple[float, float] | None = None,
    left_ear: float = 0.30,
    right_ear: float = 0.30,
    brow: float = 0.50,
    iris_r: float = IRIS_R,
    n_points: int = REQUIRED_LANDMARK_COUNT,
    noise: float = 0.0,
    rng=None,
) -> list[Point]:
    """Build a landmark list with exactly the features requested.

    Args:
        gaze: Iris offset for the right eye, in iris diameters.
        left_gaze: Iris offset for the left eye; defaults to ``gaze``.
        left_ear: Target eye-aspect ratio for the left eye.
        right_ear: Target eye-aspect ratio for the right eye.
        brow: Target normalised brow height.
        iris_r: Iris ring radius in normalised image units.
        n_points: Landmark count; pass fewer to simulate a model without
            iris refinement.
        noise: Standard deviation of independent Gaussian jitter added to
            every landmark, in normalised image units. This is what the
            real detector produces, and it is the thing the signal design
            has to survive.
        rng: Source of randomness for ``noise``.
    """
    if left_gaze is None:
        left_gaze = gaze
    points = [Point(0.0, 0.0)] * n_points

    def put(idx, p):
        if idx < n_points:
            points[idx] = p

    diam = 2 * iris_r

    # Eye corners, 0.1 apart and centred on each eye.
    put(LEFT_EYE_LEFT, Point(LEFT_EYE_CX - EYE_WIDTH / 2, EYE_CY))
    put(LEFT_EYE_RIGHT, Point(LEFT_EYE_CX + EYE_WIDTH / 2, EYE_CY))
    put(RIGHT_EYE_LEFT, Point(RIGHT_EYE_CX - EYE_WIDTH / 2, EYE_CY))
    put(RIGHT_EYE_RIGHT, Point(RIGHT_EYE_CX + EYE_WIDTH / 2, EYE_CY))

    # Corner-flanking anchors: same x as their corner, symmetric in y.
    for anchors, eye_cx in ((LEFT_EYE_ANCHORS, LEFT_EYE_CX),
                            (RIGHT_EYE_ANCHORS, RIGHT_EYE_CX)):
        outer, o_lo, o_hi, inner, i_lo, i_hi = anchors
        for idx, cx, dy in (
            (o_lo, eye_cx - EYE_WIDTH / 2, -ANCHOR_V),
            (o_hi, eye_cx - EYE_WIDTH / 2, +ANCHOR_V),
            (i_lo, eye_cx + EYE_WIDTH / 2, -ANCHOR_V),
            (i_hi, eye_cx + EYE_WIDTH / 2, +ANCHOR_V),
        ):
            put(idx, Point(cx, EYE_CY + dy))

    # Iris rings, offset from the corner midpoint by gaze * diameter.
    if n_points > max(LEFT_IRIS_RING):
        _place_ring(points, LEFT_IRIS_RING,
                    LEFT_EYE_CX + left_gaze[0] * diam,
                    EYE_CY + left_gaze[1] * diam, iris_r)
    if n_points > max(RIGHT_IRIS_RING):
        _place_ring(points, RIGHT_IRIS_RING,
                    RIGHT_EYE_CX + gaze[0] * diam,
                    EYE_CY + gaze[1] * diam, iris_r)

    # Iris centres, fixed 0.3 apart regardless of gaze, so that the brow
    # normaliser stays constant.
    put(LEFT_IRIS_CENTER, Point(RIGHT_EYE_CX - INTER_IRIS, EYE_CY))
    put(RIGHT_IRIS_CENTER, Point(RIGHT_EYE_CX, EYE_CY))

    # Eyelids: EAR = vertical gap / EYE_WIDTH.
    left_gap = left_ear * EYE_WIDTH
    right_gap = right_ear * EYE_WIDTH
    put(LEFT_EYE_TOP, Point(LEFT_EYE_CX, EYE_CY - left_gap / 2))
    put(LEFT_EYE_BOTTOM, Point(LEFT_EYE_CX, EYE_CY + left_gap / 2))
    put(RIGHT_EYE_TOP, Point(RIGHT_EYE_CX, EYE_CY - right_gap / 2))
    put(RIGHT_EYE_BOTTOM, Point(RIGHT_EYE_CX, EYE_CY + right_gap / 2))

    # Brows: height = mean(lid_top.y - brow.y) / INTER_IRIS.
    brow_gap = brow * INTER_IRIS
    put(LEFT_BROW_MID, Point(LEFT_EYE_CX, EYE_CY - left_gap / 2 - brow_gap))
    put(RIGHT_BROW_MID, Point(RIGHT_EYE_CX, EYE_CY - right_gap / 2 - brow_gap))

    if noise:
        source = rng if rng is not None else random.Random(0)
        points = [
            Point(p.x + source.gauss(0.0, noise), p.y + source.gauss(0.0, noise))
            for p in points
        ]
    return points


@pytest.fixture
def face():
    """The face builder, as a fixture."""
    return make_face


class FakeCapture:
    """Stands in for ``cv2.VideoCapture``.

    Hands out a scripted list of frames, then either blocks (``loop=True``)
    or reports failure.
    """

    def __init__(self, frames, loop: bool = False):
        self.frames = list(frames)
        self.loop = loop
        self.index = 0
        self.released = False
        self.reads = 0

    def read(self):
        self.reads += 1
        if self.index >= len(self.frames):
            if not self.loop:
                return False, None
            self.index = 0
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.released = True

    def isOpened(self):  # noqa: N802 - mirrors the OpenCV name
        return not self.released
