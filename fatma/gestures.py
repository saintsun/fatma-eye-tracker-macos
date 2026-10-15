"""Wink and brow-raise gesture recognition.

The detector is a pure state machine: it consumes per-frame eye-aspect
ratios and brow height plus a timestamp, and returns the click events that
became due. It never touches the mouse, so the whole gesture vocabulary can
be exercised in tests by feeding synthetic sequences.

Gestures:
    left wink   -> primary click
    right wink  -> secondary click
    brow raise  -> double click

Each is debounced with hysteresis (distinct enter/exit thresholds) and a
minimum hold time, and all three share one cooldown so a single facial
movement cannot emit two different clicks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from statistics import median


class Gesture(str, Enum):
    """A recognised gesture, ready to be turned into a mouse action."""

    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"


@dataclass
class GestureState:
    """Per-frame snapshot, for the preview overlay."""

    left_closed: bool = False
    right_closed: bool = False
    brow_raised: bool = False
    brow_baseline: float | None = None
    #: True while any facial movement is distorting the iris landmarks.
    in_progress: bool = False


class GestureDetector:
    """Detects winks and brow raises from eyelid and brow geometry.

    Args:
        ear_closed: EAR below which an eye counts as shut.
        ear_open: EAR above which a shut eye counts as reopened. Must
            exceed ``ear_closed``; the gap is the hysteresis band that
            stops a trembling eyelid from chattering.
        closed_hold_s: How long an eye must stay shut before it clicks.
        brow_raise_ratio: ``brow / baseline`` above which the brows count
            as raised.
        brow_guard_ratio: ``brow / baseline`` below which the baseline is
            allowed to keep adapting. Holding a raise above this does not
            drag the baseline up with it.
        brow_hold_s: How long a raise must be held before it clicks.
        cooldown_s: Minimum gap between any two emitted gestures.
        brow_ema_alpha: Baseline adaptation rate. At 30 fps, 0.001 gives a
            time constant of roughly 30 s: posture drift is absorbed, a
            deliberate brow pulse is not.
        baseline_warmup: Frames used to seed the brow baseline. Seeding
            from a median instead of the very first frame keeps a startup
            blink or a raised brow out of the resting reference.
    """

    def __init__(
        self,
        ear_closed: float = 0.16,
        ear_open: float = 0.22,
        closed_hold_s: float = 0.20,
        brow_raise_ratio: float = 1.15,
        brow_guard_ratio: float = 1.10,
        brow_hold_s: float = 0.20,
        cooldown_s: float = 0.45,
        brow_ema_alpha: float = 0.001,
        baseline_warmup: int = 15,
        brow_enabled: bool = False,
    ) -> None:
        if ear_open <= ear_closed:
            raise ValueError("ear_open must exceed ear_closed")
        if brow_guard_ratio > brow_raise_ratio:
            raise ValueError("brow_guard_ratio must not exceed brow_raise_ratio")
        if not 0.0 < brow_ema_alpha < 1.0:
            raise ValueError("brow_ema_alpha must be in (0, 1)")
        self.ear_closed = ear_closed
        self.ear_open = ear_open
        self.closed_hold_s = closed_hold_s
        self.brow_raise_ratio = brow_raise_ratio
        self.brow_guard_ratio = brow_guard_ratio
        self.brow_hold_s = brow_hold_s
        self.cooldown_s = cooldown_s
        self.brow_ema_alpha = brow_ema_alpha
        self.baseline_warmup = baseline_warmup
        #: Looking up raises the eyebrows by itself, so this gesture
        #: fires exactly when the user is reaching for the top of the
        #: screen. Opt-in rather than on by default.
        self.brow_enabled = brow_enabled

        self.left_closed = False
        self.right_closed = False
        self._left_since: float | None = None
        self._right_since: float | None = None
        self.brow_raised = False
        self._brow_since: float | None = None
        self.brow_baseline: float | None = None
        self._warmup: deque[float] = deque(maxlen=max(baseline_warmup, 1))
        self.last_fired_at = float("-inf")

    def reset(self) -> None:
        """Clear all latched state, including the brow baseline."""
        self.left_closed = False
        self.right_closed = False
        self._left_since = None
        self._right_since = None
        self.brow_raised = False
        self._brow_since = None
        self.brow_baseline = None
        self._warmup.clear()
        self.last_fired_at = float("-inf")

    # --- brow baseline -------------------------------------------------

    def _update_baseline(self, brow: float) -> None:
        if self.brow_baseline is None:
            self._warmup.append(brow)
            if len(self._warmup) >= self.baseline_warmup:
                self.brow_baseline = float(median(self._warmup))
            return
        # Only adapt while the brows are clearly down, so a held raise
        # cannot drag the reference up behind it.
        if brow <= self.brow_baseline * self.brow_guard_ratio:
            a = self.brow_ema_alpha
            self.brow_baseline = (1 - a) * self.brow_baseline + a * brow

    # --- main entry point ----------------------------------------------

    def update(
        self,
        ear_left: float | None,
        ear_right: float | None,
        brow: float | None,
        now: float,
        enabled: bool = True,
    ) -> list[Gesture]:
        """Advance the state machine one frame.

        Args:
            ear_left: Left eye aspect ratio, or None if unavailable.
            ear_right: Right eye aspect ratio, or None if unavailable.
            brow: Normalised brow height, or None if unavailable.
            now: Monotonic timestamp in seconds.
            enabled: When False, state still advances but no gesture is
                emitted. Used to suppress clicks while the mouse is off or
                before calibration, without desynchronising the latches.

        Returns:
            The gestures that became due on this frame, in priority order.
            At most one gesture is ever returned, because all three share
            a cooldown.
        """
        if brow is not None:
            self._update_baseline(brow)

        fired: list[Gesture] = []

        # Winks. A natural blink shuts both eyes, so a click only counts
        # when the other eye is plainly open — that is what makes it a
        # wink rather than a blink.
        if ear_left is not None and ear_right is not None:
            right_is_open = ear_right > self.ear_open
            left_is_open = ear_left > self.ear_open

            if right_is_open:
                self.left_closed, self._left_since, hit = self._advance_eye(
                    ear_left, self.left_closed, self._left_since, now
                )
                if hit and enabled and self._fire(now):
                    fired.append(Gesture.LEFT_CLICK)
                    self._left_since = None
            else:
                self.left_closed = ear_left < self.ear_closed
                self._left_since = None

            if left_is_open:
                self.right_closed, self._right_since, hit = self._advance_eye(
                    ear_right, self.right_closed, self._right_since, now
                )
                if hit and enabled and self._fire(now):
                    fired.append(Gesture.RIGHT_CLICK)
                    self._right_since = None
            else:
                self.right_closed = ear_right < self.ear_closed
                self._right_since = None

        # Brow raise.
        if (self.brow_enabled and brow is not None
                and self.brow_baseline is not None):
            raised_now = brow > self.brow_baseline * self.brow_raise_ratio
            if raised_now and not self.brow_raised:
                self.brow_raised = True
                self._brow_since = now
            elif not raised_now and self.brow_raised:
                self.brow_raised = False
                self._brow_since = None
            if (
                self.brow_raised
                and self._brow_since is not None
                and now - self._brow_since >= self.brow_hold_s
            ):
                if enabled and self._fire(now):
                    fired.append(Gesture.DOUBLE_CLICK)
                    self._brow_since = None

        return fired

    def _advance_eye(
        self,
        ear: float,
        is_closed: bool,
        closed_since: float | None,
        now: float,
    ) -> tuple[bool, float | None, bool]:
        """Advance one eye's latch. Returns (closed, since, hold_elapsed)."""
        if not is_closed and ear < self.ear_closed:
            is_closed = True
            closed_since = now
        elif is_closed and ear > self.ear_open:
            is_closed = False
            closed_since = None
        hit = (
            is_closed
            and closed_since is not None
            and now - closed_since >= self.closed_hold_s
        )
        return is_closed, closed_since, hit

    def _fire(self, now: float) -> bool:
        """Claim the shared cooldown. True if this gesture may be emitted.

        Checked and claimed per gesture rather than once per frame, so a
        wink and a brow raise that come due on the same frame cannot both
        get through.
        """
        if now - self.last_fired_at < self.cooldown_s:
            return False
        self.last_fired_at = now
        return True

    # --- queries -------------------------------------------------------

    def is_in_progress(
        self,
        ear_left: float | None,
        ear_right: float | None,
        brow: float | None,
    ) -> bool:
        """True while a facial movement is corrupting the gaze signal.

        A closing eyelid covers and distorts the iris ring, and a brow
        raise shifts the whole face slightly. The cursor is frozen for the
        duration so the click lands where the user was last looking rather
        than wherever the distorted landmarks point.
        """
        if ear_left is not None and ear_left < self.ear_open:
            return True
        if ear_right is not None and ear_right < self.ear_open:
            return True
        if (
            self.brow_enabled
            and brow is not None
            and self.brow_baseline is not None
            and brow > self.brow_baseline * self.brow_guard_ratio
        ):
            return True
        return False

    def state(
        self,
        ear_left: float | None = None,
        ear_right: float | None = None,
        brow: float | None = None,
    ) -> GestureState:
        """Snapshot of the latched state, for the preview overlay."""
        return GestureState(
            left_closed=self.left_closed,
            right_closed=self.right_closed,
            brow_raised=self.brow_raised,
            brow_baseline=self.brow_baseline,
            in_progress=self.is_in_progress(ear_left, ear_right, brow),
        )
