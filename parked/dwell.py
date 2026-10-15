"""Dwell: selecting something by holding your gaze on it.

For someone who can only move their eyes, every control has to be reachable
with the eyes alone. Dwell is the primitive that makes that possible and is
the standard one in eye-gaze AAC: rest your gaze somewhere, a progress ring
fills, and when it completes the action fires.

Two variants live here, because the tracker needs dwell on both sides of
calibration:

* :class:`DwellTracker` works in screen pixels, for clicking and for menu
  targets, and is used once a mapping exists.
* :class:`GazeStability` works on the raw iris signal, for calibration
  itself — at that point there is no gaze-to-screen mapping yet, so "is the
  user holding still?" is the only question that can be asked. It is also
  the question that matters: a calibration point should be captured exactly
  when the gaze has settled on the target.

Neither reads a clock; the caller passes timestamps, so both are fully
deterministic under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import NamedTuple


@dataclass(frozen=True)
class DwellState:
    """What the dwell looked like on one frame."""

    #: True while a dwell is accumulating.
    active: bool = False
    #: 0.0 to 1.0, for drawing a progress ring.
    progress: float = 0.0
    #: True on the single frame the dwell completed.
    fired: bool = False


class _DwellBase:
    """Shared hold-still-to-select logic.

    Args:
        hold_s: How long the point must stay put before firing.
        radius: How far it may wander and still count as held.
        rearm_outside: When True (the default) the point must leave the
            radius before the same spot can fire again. Without it, resting
            your gaze on a button would retrigger it forever.
    """

    def __init__(
        self,
        hold_s: float = 1.0,
        radius: float = 60.0,
        rearm_outside: bool = True,
    ) -> None:
        if hold_s <= 0:
            raise ValueError("hold_s must be positive")
        if radius <= 0:
            raise ValueError("radius must be positive")
        self.hold_s = hold_s
        self.radius = radius
        self.rearm_outside = rearm_outside
        self._ax: float | None = None
        self._ay: float | None = None
        self._since: float = 0.0
        self._armed = True

    def reset(self) -> None:
        """Forget the current dwell and re-arm."""
        self._ax = self._ay = None
        self._since = 0.0
        self._armed = True

    @property
    def anchor(self) -> tuple[float, float] | None:
        """Where the current dwell is centred, if one is running."""
        if self._ax is None or self._ay is None:
            return None
        return self._ax, self._ay

    def _step(self, x: float, y: float, now: float) -> DwellState:
        if self._ax is None or hypot(x - self._ax, y - self._ay) > self.radius:
            # Moved away: start a new dwell here, and allow firing again.
            self._ax, self._ay = x, y
            self._since = now
            self._armed = True
            return DwellState(active=True, progress=0.0)

        elapsed = now - self._since
        progress = min(elapsed / self.hold_s, 1.0)
        if progress >= 1.0 and self._armed and self._may_fire():
            if self.rearm_outside:
                self._armed = False
            else:
                self._since = now
            self._on_fire()
            return DwellState(active=True, progress=1.0, fired=True)
        return DwellState(active=True, progress=progress)

    def _may_fire(self) -> bool:
        return True

    def _on_fire(self) -> None:
        pass


class DwellTracker(_DwellBase):
    """Dwell in screen pixels — for clicking and for menu targets.

    The radius has to be generous: webcam gaze tracking lands within
    roughly 50-150 px of the intended point, so a tight radius would mean
    the dwell never completes because the estimate itself drifts.
    """

    def update(self, x: float, y: float, now: float) -> DwellState:
        """Advance one frame at cursor position ``(x, y)``."""
        return self._step(x, y, now)


class GazeStability(_DwellBase):
    """Dwell on the raw gaze signal — for calibration without a mapping.

    Args:
        hold_s: How long the gaze must be steady before the point is taken.
        tolerance: Allowed wander, in iris diameters. A whole screen spans
            roughly 0.3 units horizontally, so fixation jitter is an order
            of magnitude smaller than this.
        min_travel: How far the gaze must have moved from the previously
            captured point before another capture is allowed. Without it a
            user staring straight ahead would "capture" nine identical
            points and the spline fit would be impossible.
    """

    def __init__(
        self,
        hold_s: float = 1.2,
        tolerance: float = 0.030,
        min_travel: float = 0.015,
    ) -> None:
        super().__init__(hold_s=hold_s, radius=tolerance, rearm_outside=True)
        self.min_travel = min_travel
        self._last_captured: tuple[float, float] | None = None

    @property
    def tolerance(self) -> float:
        return self.radius

    def update(self, nx: float, ny: float, now: float) -> DwellState:
        """Advance one frame at raw gaze reading ``(nx, ny)``."""
        return self._step(nx, ny, now)

    def _may_fire(self) -> bool:
        if self._last_captured is None:
            return True
        lx, ly = self._last_captured
        return hypot(self._ax - lx, self._ay - ly) >= self.min_travel

    def _on_fire(self) -> None:
        self._last_captured = (self._ax, self._ay)

    def start_point(self) -> None:
        """Begin collecting the next calibration point."""
        self._ax = self._ay = None
        self._since = 0.0
        self._armed = True

    def reset(self) -> None:
        """Forget everything, including which point was captured last."""
        super().reset()
        self._last_captured = None


class Rect(NamedTuple):
    """An axis-aligned screen rectangle, in pixels."""

    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass(frozen=True)
class Zone:
    """A dwell target: a rectangle that fires an action when held."""

    key: str
    rect: Rect
    label: str = ""
    hold_s: float | None = None


@dataclass(frozen=True)
class ZoneState:
    """Which zone is being dwelled on, and how far along it is."""

    focused: str | None = None
    progress: float = 0.0
    fired: str | None = None


class ZoneSet:
    """A set of dwell targets, at most one of which can be active.

    Unlike :class:`DwellTracker` the anchor is the *zone*, not the point:
    the gaze may wander anywhere inside a large button without resetting
    progress, which is what makes big targets forgiving of tracking error.
    """

    def __init__(self, zones: list[Zone], hold_s: float = 1.2) -> None:
        self.zones = zones
        self.hold_s = hold_s
        self._focused: str | None = None
        self._since: float = 0.0
        self._armed = True

    def reset(self) -> None:
        """Clear focus and re-arm."""
        self._focused = None
        self._since = 0.0
        self._armed = True

    def set_zones(self, zones: list[Zone]) -> None:
        """Swap the targets, e.g. when the menu changes page."""
        self.zones = zones
        self.reset()

    def zone_for(self, x: float, y: float) -> Zone | None:
        """The first zone containing the point, if any."""
        for zone in self.zones:
            if zone.rect.contains(x, y):
                return zone
        return None

    def update(self, x: float, y: float, now: float) -> ZoneState:
        """Advance one frame at cursor position ``(x, y)``."""
        zone = self.zone_for(x, y)
        if zone is None:
            # Looking away from every target re-arms the set, so a button
            # fires once per visit rather than once per frame.
            self._focused = None
            self._armed = True
            return ZoneState()

        if zone.key != self._focused:
            self._focused = zone.key
            self._since = now
            self._armed = True
            return ZoneState(focused=zone.key, progress=0.0)

        hold = zone.hold_s if zone.hold_s is not None else self.hold_s
        progress = min((now - self._since) / hold, 1.0)
        if progress >= 1.0 and self._armed:
            self._armed = False
            return ZoneState(focused=zone.key, progress=1.0, fired=zone.key)
        return ZoneState(focused=zone.key, progress=progress)
