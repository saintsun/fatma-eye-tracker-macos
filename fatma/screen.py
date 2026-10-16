"""Where a window actually is on screen, measured rather than assumed.

The calibration target has to appear at a known point on the *screen*, but
it is drawn into a window's canvas, and on macOS that canvas is not where
the naive arithmetic says it is: the menu bar occupies the top, a window
may carry a title bar, and a "fullscreen" OpenCV window may not have gone
fullscreen at all.

The tracker used to paper over this with a constant — 25 pixels of assumed
menu bar. Measured on one machine the menu bar is 33, and a window placed
at the top starts below it. Any error here is not cosmetic: the calibrator
records "the user looked at 3% down the screen" while the dot was somewhere
else entirely, and that wrong pairing is fitted into the mapping.

So nothing here guesses. :func:`window_rect` asks the window server where
our window ended up, and :func:`place_target` works out where to draw so
the dot lands on the requested screen point — reporting back the point it
could actually reach, which is what the fit must be told.
"""

from __future__ import annotations

import os
from typing import NamedTuple


class Rect(NamedTuple):
    """A screen rectangle in pixels, origin top-left."""

    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.right and self.y <= py <= self.bottom


def _quartz():
    try:
        import Quartz
    except Exception:
        return None
    return Quartz


def main_display_size() -> tuple[int, int]:
    """Main display size in pixels."""
    q = _quartz()
    if q is None:                                  # pragma: no cover
        from .pointer import screen_size

        return screen_size()
    bounds = q.CGDisplayBounds(q.CGMainDisplayID())
    return int(bounds.size.width), int(bounds.size.height)


def menu_bar_height(default: int = 25) -> int:
    """Height of the macOS menu bar, measured from the window server.

    Returns ``default`` where it cannot be determined, which is the old
    hardcoded guess — wrong by eight pixels on the machine this was written
    on, which is rather the point.
    """
    q = _quartz()
    if q is None:
        return default
    try:
        windows = q.CGWindowListCopyWindowInfo(
            q.kCGWindowListOptionOnScreenOnly, q.kCGNullWindowID
        ) or []
        for window in windows:
            if window.get("kCGWindowName") == "Menubar":
                return int(window.get("kCGWindowBounds", {}).get("Height",
                                                                 default))
    except Exception:
        pass
    return default


def window_rect(title: str, pid: int | None = None) -> Rect | None:
    """Screen rectangle of one of our own windows, by title.

    Args:
        title: The window title, as passed to ``cv2.namedWindow``.
        pid: Process to match; defaults to this one. Matching on the pid
            as well as the title keeps another application's window of the
            same name from being measured instead.

    Returns None when the window is not on screen yet, or when the window
    server cannot be reached.
    """
    q = _quartz()
    if q is None:
        return None
    want_pid = os.getpid() if pid is None else pid
    try:
        windows = q.CGWindowListCopyWindowInfo(
            q.kCGWindowListOptionOnScreenOnly, q.kCGNullWindowID
        ) or []
    except Exception:
        return None
    for window in windows:
        if window.get("kCGWindowOwnerPID") != want_pid:
            continue
        if window.get("kCGWindowName") != title:
            continue
        bounds = window.get("kCGWindowBounds") or {}
        rect = Rect(float(bounds.get("X", 0)), float(bounds.get("Y", 0)),
                    float(bounds.get("Width", 0)), float(bounds.get("Height", 0)))
        if rect.w > 0 and rect.h > 0:
            return rect
    return None


def place_target(
    screen_x: float,
    screen_y: float,
    canvas: Rect,
    margin: float = 0.0,
) -> tuple[tuple[int, int], tuple[float, float]]:
    """Work out where to draw a dot so it lands on a screen point.

    Args:
        screen_x: Desired screen x in pixels.
        screen_y: Desired screen y in pixels.
        canvas: Where the drawing surface sits on the screen.
        margin: Keep the dot at least this far inside the canvas, so a
            target is never clipped in half by an edge.

    Returns:
        ``((canvas_px, canvas_py), (actual_screen_x, actual_screen_y))``.
        The two differ whenever the requested point lies outside the
        canvas — under the menu bar, say. The caller must fit against the
        *actual* point: pairing a gaze reading with a position the dot
        never occupied is how a calibration ends up subtly wrong
        everywhere rather than obviously wrong in one place.
    """
    lo_x, hi_x = canvas.x + margin, canvas.right - margin
    lo_y, hi_y = canvas.y + margin, canvas.bottom - margin
    actual_x = min(max(screen_x, lo_x), max(lo_x, hi_x))
    actual_y = min(max(screen_y, lo_y), max(lo_y, hi_y))
    return (
        (int(round(actual_x - canvas.x)), int(round(actual_y - canvas.y))),
        (actual_x, actual_y),
    )


def coverage(canvas: Rect, screen_w: int, screen_h: int) -> tuple[float, float]:
    """Fraction of the screen the canvas can actually reach, per axis.

    Well below 1.0 means the calibration cannot place targets near the
    screen edges, so the mapping will be extrapolating there.
    """
    if screen_w <= 0 or screen_h <= 0:
        return 0.0, 0.0
    return (min(canvas.w / screen_w, 1.0), min(canvas.h / screen_h, 1.0))
