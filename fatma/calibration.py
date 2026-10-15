"""Nine-point gaze calibration and its on-disk format.

The calibrator collects a gaze reading for each of nine screen targets,
fits a thin-plate spline from gaze space to normalised screen space, and
maps every later reading through it.

Screen *pixels* deliberately never appear here: the mapping produces
fractions of the screen, and the caller multiplies by the real resolution.
That keeps a saved calibration valid across display changes and lets the
whole module be tested without a display.
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np

from .quality import (
    CalibrationQuality,
    GazeSpan,
    gaze_span,
    leave_one_out_quality,
)
from .tps import TPSFitError, TPSMapping

#: Current on-disk schema. ``load`` refuses anything else rather than
#: silently mapping gaze through a mapping it does not understand.
CALIB_KIND = "tps"

#: Fallback linear mapping used before any calibration exists. Centres a
#: gaze range of roughly +-0.2 iris diameters on the screen.
FALLBACK_SPAN = 0.4
FALLBACK_OFFSET = 0.2


class CalibrationError(Exception):
    """Raised when a calibration cannot be fitted or loaded."""


def _quality_from_payload(payload) -> CalibrationQuality | None:
    """Rebuild the accuracy figures stored alongside a saved calibration.

    Per-point errors are not written to the file, so the restored report
    carries the summary only; ``per_point`` stays empty and callers that
    want to name the worst target have to re-calibrate.
    """
    if not isinstance(payload, dict):
        return None
    try:
        mean = float(payload["mean"])
        worst = float(payload["worst"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (mean >= 0 and worst >= mean):
        return None
    return CalibrationQuality(per_point=(), mean=mean, worst=worst,
                              worst_index=-1)


class Calibrator:
    """Collects calibration samples and maps gaze to normalised screen space.

    The nine targets are pushed close to the screen edges so that the
    reachable cursor area covers the menu bar and the dock, not just the
    safe middle of the display.

    Args:
        min_samples: Reject a ``confirm()`` backed by fewer samples than
            this — it means the user pressed the key before settling on the
            target, which would poison that control point.
        sample_window: How many trailing samples the per-point median is
            taken over (roughly half a second at 30 fps).
        clamp_margin: Fraction of the observed gaze range added as headroom
            around the clamping box.
    """

    GRID_9: tuple[tuple[float, float], ...] = (
        (0.03, 0.03), (0.5, 0.03), (0.97, 0.03),
        (0.03, 0.5),  (0.5, 0.5),  (0.97, 0.5),
        (0.03, 0.97), (0.5, 0.97), (0.97, 0.97),
    )

    #: The 9 grid plus the four mid-edge points. The spline is exact at
    #: every target and merely interpolates between them, so error peaks
    #: furthest from any target — along the middle of each edge. These four
    #: extra points sit precisely there. Ordered so that consecutive
    #: targets are far apart, which keeps the user's gaze moving and makes
    #: a "never moved" calibration impossible to produce by accident.
    GRID_13: tuple[tuple[float, float], ...] = (
        (0.03, 0.03), (0.97, 0.03), (0.03, 0.97), (0.97, 0.97),
        (0.5, 0.5),
        (0.5, 0.03), (0.5, 0.97), (0.03, 0.5), (0.97, 0.5),
        (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
    )

    POINTS_SCREEN = GRID_9

    def __init__(
        self,
        min_samples: int = 5,
        sample_window: int = 15,
        clamp_margin: float = 0.05,
        grid: int = 9,
        stability: float = 0.02,
    ) -> None:
        if grid == 13:
            self.POINTS_SCREEN = self.GRID_13
        elif grid == 9:
            self.POINTS_SCREEN = self.GRID_9
        else:
            raise ValueError(f"grid must be 9 or 13, got {grid}")
        self.min_samples = min_samples
        self.sample_window = sample_window
        self.clamp_margin = clamp_margin
        #: How tightly the trailing samples must cluster before a point
        #: may be confirmed, in iris diameters.
        self.stability = stability
        self.points_eye: list[np.ndarray] = []
        self.samples: list[tuple[float, float]] = []
        self.idx = 0
        self.active = False
        self.tps: TPSMapping | None = None
        self.nx_min = self.nx_max = 0.0
        self.ny_min = self.ny_max = 0.0
        #: Message from the last failed fit, for the UI to show.
        self.fit_error: str | None = None
        #: Mapping in force before the current run started, so that an
        #: abandoned calibration does not cost the user the working one.
        self._previous: tuple | None = None
        #: Cross-validated accuracy of the current fit, once measured.
        self.quality: CalibrationQuality | None = None
        #: Gaze travel of the current fit; the dominant term in accuracy.
        self.span: GazeSpan | None = None

    # --- state ---------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """True once a mapping has been fitted or loaded."""
        return self.tps is not None

    @property
    def total_points(self) -> int:
        return len(self.POINTS_SCREEN)

    @property
    def current_target(self) -> tuple[float, float]:
        """Normalised (x, y) of the target the user should look at now."""
        return self.POINTS_SCREEN[min(self.idx, self.total_points - 1)]

    @property
    def sample_count(self) -> int:
        """Samples gathered for the current target."""
        return len(self.samples)

    @property
    def steadiness(self) -> float:
        """Spread of the trailing samples, in iris diameters.

        Large while the gaze is still travelling to the target, small once
        it has settled. ``inf`` until enough samples have arrived.
        """
        window = self.samples[-self.sample_window:]
        if len(window) < self.min_samples:
            return float("inf")
        xs = [s[0] for s in window]
        ys = [s[1] for s in window]
        return max(max(xs) - min(xs), max(ys) - min(ys))

    @property
    def ready_to_confirm(self) -> bool:
        """True when the gaze has actually settled on the target.

        A sample count alone is not enough: at 30 fps the old threshold of
        five samples was a sixth of a second, so a key pressed as the target
        appeared captured the gaze still in flight across the screen. The
        median then averaged that sweep into the control point, and a single
        such point bends the whole spline.
        """
        return (self.active
                and len(self.samples) >= self.min_samples
                and self.steadiness <= self.stability)

    # --- collection ----------------------------------------------------

    def start(self) -> None:
        """Begin a fresh calibration.

        The existing mapping is set aside rather than thrown away: during
        the run the tracker should behave as uncalibrated, but abandoning
        the run must not cost the user a calibration that was working.
        """
        self._previous = (
            self.tps, self.nx_min, self.nx_max, self.ny_min, self.ny_max
        )
        self.points_eye.clear()
        self.samples.clear()
        self.idx = 0
        self.active = True
        self.tps = None
        self.fit_error = None

    def cancel(self) -> None:
        """Abort an in-progress calibration, restoring the previous one."""
        self.active = False
        self.points_eye.clear()
        self.samples.clear()
        self.idx = 0
        if self._previous is not None:
            (self.tps, self.nx_min, self.nx_max,
             self.ny_min, self.ny_max) = self._previous
            self._previous = None

    def feed(self, nx: float, ny: float) -> None:
        """Record a gaze sample for the current target."""
        if self.active:
            self.samples.append((nx, ny))

    def confirm(self) -> bool:
        """Lock in the current target from the samples collected so far.

        The control point is the *median* of the trailing window, not the
        mean: while the user was moving onto the target their gaze swept
        across the screen, and a mean would drag the point along that path.

        Returns:
            True if a control point was recorded. False if the calibration
            is not running or too few samples have arrived yet.
        """
        if not self.ready_to_confirm:
            return False
        arr = np.array(self.samples[-self.sample_window:])
        self.points_eye.append(np.median(arr, axis=0))
        self.samples.clear()
        self.idx += 1
        if self.idx >= self.total_points:
            self._fit()
            self.active = False
        return True

    def _fit(self) -> None:
        eye = np.array(self.points_eye)
        scr = np.array(self.POINTS_SCREEN, dtype=float)
        try:
            fitted = TPSMapping(eye, scr[:, 0], scr[:, 1])
        except TPSFitError as exc:
            # A failed fit must not cost the user the mapping they already
            # had — they can retry, or carry on with the old one.
            self.fit_error = str(exc)
            if self._previous is not None:
                (self.tps, self.nx_min, self.nx_max,
                 self.ny_min, self.ny_max) = self._previous
                self._previous = None
            return
        self.tps = fitted
        self._previous = None
        self.fit_error = None
        # Cross-validate before the user starts relying on the mapping: a
        # calibration spoiled by a blink or a head shift looks identical to
        # a good one until the cursor misbehaves, and this turns that into
        # a number they can act on.
        self.quality = leave_one_out_quality(eye, scr)
        self.span = gaze_span(eye)
        # Remember the gaze range the fit actually covers. Outside its
        # convex hull a TPS extrapolates smoothly but without bound, so a
        # glance away from the screen would otherwise fling the cursor.
        self.nx_min, self.ny_min = eye.min(axis=0)
        self.nx_max, self.ny_max = eye.max(axis=0)
        margin_x = (self.nx_max - self.nx_min) * self.clamp_margin
        margin_y = (self.ny_max - self.ny_min) * self.clamp_margin
        self.nx_min -= margin_x
        self.nx_max += margin_x
        self.ny_min -= margin_y
        self.ny_max += margin_y

    # --- mapping -------------------------------------------------------

    def clamp_input(self, nx: float, ny: float) -> tuple[float, float]:
        """Clip a gaze reading into the box the calibration covers.

        Clamping the *input* keeps the spline inside the region it was
        fitted on. Clipping only the output would still let an unbounded
        extrapolation swing the cursor hard against an edge before being
        cut off, which reads as a jump rather than as "stopped at the edge".
        """
        if self.nx_max > self.nx_min:
            nx = min(max(nx, self.nx_min), self.nx_max)
        if self.ny_max > self.ny_min:
            ny = min(max(ny, self.ny_min), self.ny_max)
        return nx, ny

    def map_normalized(self, nx: float, ny: float) -> tuple[float, float]:
        """Map a gaze reading to screen fractions in ``[0, 1]``."""
        if self.tps is None:
            sx = (nx + FALLBACK_OFFSET) / FALLBACK_SPAN
            sy = (ny + FALLBACK_OFFSET) / FALLBACK_SPAN
        else:
            cnx, cny = self.clamp_input(nx, ny)
            sx, sy = self.tps(cnx, cny)
        return (
            min(max(sx, 0.0), 1.0),
            min(max(sy, 0.0), 1.0),
        )

    def map_to_screen(
        self, nx: float, ny: float, screen_w: int, screen_h: int
    ) -> tuple[float, float]:
        """Map a gaze reading to pixels on a ``screen_w x screen_h`` display."""
        sx, sy = self.map_normalized(nx, ny)
        return sx * screen_w, sy * screen_h

    # --- persistence ---------------------------------------------------

    def to_dict(self, screen: Sequence[int] | None = None) -> dict:
        """Serialise the fitted mapping."""
        if self.tps is None:
            raise CalibrationError("nothing to save: no calibration fitted")
        return {
            "kind": CALIB_KIND,
            "tps": self.tps.to_dict(),
            "nx_range": [float(self.nx_min), float(self.nx_max)],
            "ny_range": [float(self.ny_min), float(self.ny_max)],
            "screen": list(screen) if screen is not None else None,
            "quality": (
                {"mean": self.quality.mean, "worst": self.quality.worst}
                if self.quality is not None and self.quality.valid else None
            ),
        }

    def save(self, path: str, screen: Sequence[int] | None = None) -> None:
        """Write the calibration to ``path`` atomically.

        Writing to a temporary file and renaming it means a crash partway
        through cannot leave a half-written calibration that fails to load
        on the next run.
        """
        payload = self.to_dict(screen)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)

    def load_dict(self, data: dict) -> None:
        """Restore from a parsed payload.

        Raises:
            CalibrationError: if the payload is not a current-format
                calibration.
        """
        if not isinstance(data, dict):
            raise CalibrationError("calibration payload is not an object")
        if data.get("kind") != CALIB_KIND or "tps" not in data:
            raise CalibrationError(
                "unrecognised calibration format — please re-calibrate"
            )
        try:
            tps = TPSMapping.from_dict(data["tps"])
        except TPSFitError as exc:
            raise CalibrationError(str(exc)) from exc
        try:
            nx_min, nx_max = (float(v) for v in data["nx_range"])
            ny_min, ny_max = (float(v) for v in data["ny_range"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"malformed gaze range: {exc}") from exc
        if not (nx_max > nx_min and ny_max > ny_min):
            raise CalibrationError("gaze range is empty or inverted")
        self.tps = tps
        self.nx_min, self.nx_max = nx_min, nx_max
        self.ny_min, self.ny_max = ny_min, ny_max
        self.fit_error = None
        # Cross-validation needs the original control points, which are not
        # in the file — only the fitted mapping is. Restoring the figures
        # recorded at fit time means a user who restarts still sees how good
        # their calibration was, instead of an unhelpful "calibrated: yes".
        self.quality = _quality_from_payload(data.get("quality"))

    def load(self, path: str) -> bool:
        """Load a calibration from ``path``.

        Returns:
            True on success. False if the file is absent, unreadable or in
            an unusable format — tracking then falls back to the linear
            approximation until the user calibrates.
        """
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self.load_dict(data)
        except (OSError, json.JSONDecodeError, CalibrationError):
            return False
        return True
