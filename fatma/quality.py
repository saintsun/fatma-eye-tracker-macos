"""Measuring precision, rather than guessing at it.

Two different things get called "accuracy" in a gaze tracker and they have
different causes and different fixes:

* **Calibration error** — how well the fitted mapping generalises. Measured
  here by leave-one-out cross-validation: refit without one calibration
  point, predict it, and see how far off the prediction lands. A bad number
  means the calibration itself was sloppy (the user blinked, looked away,
  or moved their head between targets) and no amount of filtering will
  rescue it.

* **Runtime jitter** — how much the cursor wanders while the user holds
  their gaze still. Measured by :class:`JitterMeter`. This one is driven by
  landmark noise, and the lever on it is capture resolution and averaging,
  not calibration.

Both are reported in screen pixels, because that is the unit the user
actually experiences.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .i18n import DEFAULT_LANGUAGE, t

from .tps import TPSFitError, TPSMapping


#: Gaze span below which a calibration is amplifying noise badly. The span
#: is how far the iris travels, in its own diameters, across the whole
#: screen — and the mapping multiplies landmark noise by (screen / span).
#: A session measured at 0.079 vertical produced 208 px of mean error;
#: an earlier one at 0.200 on the same machine was three times better.
MIN_GOOD_SPAN_Y = 0.12
MIN_GOOD_SPAN_X = 0.25


@dataclass(frozen=True)
class GazeSpan:
    """How far the gaze travelled during calibration, in iris diameters.

    This is the single most important thing about a calibration and it is
    not visible in the fit residuals at all: a mapping can pass exactly
    through every control point and still be useless, because a small span
    means every pixel of landmark noise is multiplied by a huge factor on
    its way to the cursor.

    The span is mostly set by how far the user sits from the *screen*. Far
    away, the screen subtends a small visual angle, the eyes barely rotate
    to cover it, and the signal collapses into the noise.
    """

    x: float
    y: float

    @property
    def ok(self) -> bool:
        return self.x >= MIN_GOOD_SPAN_X and self.y >= MIN_GOOD_SPAN_Y

    def amplification(self, screen_w: int, screen_h: int) -> tuple[float, float]:
        """Screen pixels per unit of gaze signal, per axis."""
        return gaze_scale((0.0, self.x), (0.0, self.y), screen_w, screen_h)

    def advice(self, lang: str = DEFAULT_LANGUAGE) -> str:
        """What to tell the user, or an empty string when it is fine."""
        if self.ok:
            return ""
        return t(lang, "span.narrow", x=self.x, y=self.y,
                 min_x=MIN_GOOD_SPAN_X, min_y=MIN_GOOD_SPAN_Y)


def gaze_span(points_eye) -> GazeSpan:
    """Measure the gaze span of a set of calibration readings."""
    arr = np.asarray(points_eye, dtype=float)
    if arr.ndim != 2 or len(arr) < 2:
        return GazeSpan(0.0, 0.0)
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    return GazeSpan(float(hi[0] - lo[0]), float(hi[1] - lo[1]))


@dataclass(frozen=True)
class CalibrationQuality:
    """Leave-one-out accuracy of a fitted calibration.

    Errors are in screen fractions; multiply by the display size for
    pixels. ``per_point`` follows the order of the calibration targets, so
    a single bad target is identifiable rather than just dragging the mean.
    """

    per_point: tuple[float, ...]
    mean: float
    worst: float
    worst_index: int
    valid: bool = True
    reason: str = ""

    def in_pixels(self, screen_w: int, screen_h: int) -> tuple[float, float]:
        """(mean, worst) error in pixels on a given display.

        A screen fraction is converted with the display diagonal, since the
        error is a 2D distance in a normalised square.
        """
        diag = math.hypot(screen_w, screen_h) / math.sqrt(2)
        return self.mean * diag, self.worst * diag

    def grade(self, lang: str = DEFAULT_LANGUAGE) -> str:
        """A one-word verdict, for the preview overlay and the panel."""
        if not self.valid:
            return t(lang, "grade.unknown")
        if self.mean < 0.02:
            return t(lang, "grade.great")
        if self.mean < 0.04:
            return t(lang, "grade.good")
        if self.mean < 0.08:
            return t(lang, "grade.fair")
        return t(lang, "grade.poor")


def leave_one_out_quality(points_eye, targets) -> CalibrationQuality:
    """Cross-validate a calibration without needing new measurements.

    Each control point is held out in turn, the spline is refitted on the
    rest, and the held-out point is predicted. This is the honest estimate
    of how the mapping will behave *between* the targets the user looked
    at, which is where the cursor spends all of its time.

    Args:
        points_eye: ``(n, 2)`` gaze readings, one per target.
        targets: ``(n, 2)`` target positions in screen fractions.

    Returns:
        A :class:`CalibrationQuality`. ``valid`` is False when there are too
        few points to cross-validate, or when a refit failed.
    """
    eye = np.asarray(points_eye, dtype=float)
    scr = np.asarray(targets, dtype=float)
    n = len(eye)
    if n < 5:
        return CalibrationQuality(
            (), 0.0, 0.0, -1, valid=False,
            reason=f"need at least 5 points to cross-validate, got {n}",
        )

    errors: list[float] = []
    for i in range(n):
        keep = np.delete(np.arange(n), i)
        try:
            fit = TPSMapping(eye[keep], scr[keep, 0], scr[keep, 1])
        except TPSFitError as exc:
            return CalibrationQuality(
                (), 0.0, 0.0, -1, valid=False,
                reason=f"refit without point {i} failed: {exc}",
            )
        px, py = fit(eye[i, 0], eye[i, 1])
        errors.append(math.hypot(px - scr[i, 0], py - scr[i, 1]))

    worst = max(errors)
    return CalibrationQuality(
        per_point=tuple(errors),
        mean=float(np.mean(errors)),
        worst=worst,
        worst_index=errors.index(worst),
    )


class JitterMeter:
    """Runtime precision: how far the cursor wanders while the gaze rests.

    Reports the RMS deviation of recent cursor positions about their own
    mean. During a saccade that number is large and meaningless; during a
    fixation it is the noise floor the user is fighting, in pixels.

    Args:
        window: Frames to average over. One second at 30 fps by default.
        fixation_px: RMS below which the window counts as a fixation.
    """

    def __init__(self, window: int = 30, fixation_px: float = 40.0) -> None:
        if window < 2:
            raise ValueError("window must be at least 2")
        self.window = window
        self.fixation_px = fixation_px
        self._xs: deque[float] = deque(maxlen=window)
        self._ys: deque[float] = deque(maxlen=window)
        self.best: float | None = None

    def reset(self) -> None:
        self._xs.clear()
        self._ys.clear()
        self.best = None

    def update(self, x: float, y: float) -> float:
        """Add a cursor position and return the current RMS deviation."""
        self._xs.append(float(x))
        self._ys.append(float(y))
        rms = self.rms
        if self.is_fixating and (self.best is None or rms < self.best):
            self.best = rms
        return rms

    @property
    def rms(self) -> float:
        """RMS distance from the window mean, in pixels."""
        n = len(self._xs)
        if n < 2:
            return 0.0
        mx = sum(self._xs) / n
        my = sum(self._ys) / n
        total = sum((x - mx) ** 2 + (y - my) ** 2
                    for x, y in zip(self._xs, self._ys))
        return math.sqrt(total / n)

    @property
    def is_fixating(self) -> bool:
        """True when the window looks like a held gaze rather than a move."""
        return len(self._xs) >= self.window and self.rms < self.fixation_px

    @property
    def samples(self) -> int:
        return len(self._xs)


def gaze_scale(
    nx_range: tuple[float, float],
    ny_range: tuple[float, float],
    screen_w: int,
    screen_h: int,
) -> tuple[float, float]:
    """Screen pixels per unit of gaze signal, per axis.

    This is the amplification factor between landmark noise and cursor
    noise, and it is brutal: with a horizontal gaze span of 0.4 iris
    diameters across a 1470 px screen, one hundredth of an iris diameter of
    landmark error becomes 37 px of cursor error. It is the number that
    explains why capture resolution matters more than filtering.
    """
    span_x = max(nx_range[1] - nx_range[0], 1e-9)
    span_y = max(ny_range[1] - ny_range[0], 1e-9)
    return screen_w / span_x, screen_h / span_y
