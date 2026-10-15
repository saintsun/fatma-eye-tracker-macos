"""Gaze-to-cursor pipeline: filtering, mapping, deadzone and drift.

This is the whole path from a raw iris reading to the pixel the cursor
should sit on, with no I/O in it, so each stage can be driven directly from
tests:

    raw gaze -> temporal median -> One-Euro -> calibration -> drift offset
             -> clamp to screen -> deadzone -> cursor position

The deadzone is the last stage deliberately. Applying it in screen space
means its size is a constant number of pixels regardless of where on the
calibration surface the user is looking; in gaze space the same threshold
would cover a different distance at the centre than at the edges.
"""

from __future__ import annotations

from .calibration import Calibrator
from .config import Settings
from .filters import TemporalMedian, make_axis_filters


class CursorController:
    """Turns filtered gaze readings into a cursor position.

    Args:
        settings: Tunables; only the tracking and cursor fields are read.
        calibrator: Supplies the gaze-to-screen mapping.
        screen_w: Display width in pixels.
        screen_h: Display height in pixels.
    """

    def __init__(
        self,
        settings: Settings,
        calibrator: Calibrator,
        screen_w: int,
        screen_h: int,
    ) -> None:
        if screen_w <= 0 or screen_h <= 0:
            raise ValueError("screen dimensions must be positive")
        self.settings = settings
        self.calibrator = calibrator
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.median_x = TemporalMedian(settings.median_window)
        self.median_y = TemporalMedian(settings.median_window)
        self.filter_x, self.filter_y = make_axis_filters(
            *settings.smoothing(),
            settings.vertical_cutoff_scale,
            settings.vertical_beta_scale,
        )

        self.drift_dx = 0.0
        self.drift_dy = 0.0
        #: Last mapping output before drift and clamping were applied.
        #: Anchoring needs the unmodified prediction — deriving it by
        #: subtracting the drift back out of the final position is wrong
        #: whenever that position was clamped at a screen edge.
        self._raw_sx = screen_w * 0.5
        self._raw_sy = screen_h * 0.5
        self.x = screen_w * 0.5
        self.y = screen_h * 0.5

    # --- configuration -------------------------------------------------

    def set_smoothing(self, min_cutoff: float, beta: float) -> None:
        """Swap in a new One-Euro preset, discarding filter history."""
        self.filter_x, self.filter_y = make_axis_filters(
            min_cutoff,
            beta,
            self.settings.vertical_cutoff_scale,
            self.settings.vertical_beta_scale,
        )

    def reset_filters(self) -> None:
        """Clear median and One-Euro history, e.g. when recalibrating."""
        self.median_x.reset()
        self.median_y.reset()
        self.filter_x.reset()
        self.filter_y.reset()

    # --- drift ---------------------------------------------------------

    def anchor(self) -> tuple[float, float]:
        """Treat the current gaze as "looking at the screen centre".

        Head-pose changes shift the whole gaze-to-screen relationship by a
        near-constant offset. Recording that offset cancels it without
        making the user sit through a fresh nine-point calibration.

        Returns the new (dx, dy) offset.
        """
        self.drift_dx = self.screen_w * 0.5 - self._raw_sx
        self.drift_dy = self.screen_h * 0.5 - self._raw_sy
        return self.drift_dx, self.drift_dy

    def reset_drift(self) -> None:
        """Drop the drift offset."""
        self.drift_dx = 0.0
        self.drift_dy = 0.0

    # --- main pipeline -------------------------------------------------

    def update(
        self, nx: float, ny: float, now: float, frozen: bool = False
    ) -> tuple[float, float]:
        """Advance the pipeline one frame and return the cursor position.

        Args:
            nx: Raw horizontal gaze reading, in iris diameters.
            ny: Raw vertical gaze reading, in iris diameters.
            now: Monotonic timestamp in seconds.
            frozen: When True the filters and the mapping still advance,
                but the cursor is left where it is. Set while a gesture is
                underway, because a closing eyelid distorts the iris ring
                and the mapped position for those frames is meaningless.

        Returns:
            The cursor position in screen pixels.
        """
        nx_f = self.filter_x(self.median_x.update(nx), now)
        ny_f = self.filter_y(self.median_y.update(ny), now)

        sx, sy = self.calibrator.map_to_screen(
            nx_f, ny_f, self.screen_w, self.screen_h
        )
        self._raw_sx, self._raw_sy = sx, sy

        sx = min(max(sx + self.drift_dx, 0.0), self.screen_w - 1.0)
        sy = min(max(sy + self.drift_dy, 0.0), self.screen_h - 1.0)

        if not frozen:
            dx = sx - self.x
            dy = sy - self.y
            dead = self.settings.deadzone_px
            # Squared comparison avoids a sqrt per frame.
            if dx * dx + dy * dy > dead * dead:
                self.x = sx
                self.y = sy
        return self.x, self.y

    @property
    def position(self) -> tuple[float, float]:
        """Where the cursor currently is, per this controller."""
        return self.x, self.y

    @property
    def raw_prediction(self) -> tuple[float, float]:
        """Last mapping output, before drift and clamping."""
        return self._raw_sx, self._raw_sy
