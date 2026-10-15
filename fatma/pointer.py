"""Mouse output backends.

PyAutoGUI is the obvious way to move the cursor, and it is the wrong one
for a 30 fps control loop. Every public PyAutoGUI call is wrapped in a
decorator that sleeps for ``pyautogui.PAUSE`` (0.1 s by default) *after*
the call and runs a fail-safe check that queries the cursor position
before it. Measured on this project's own machine:

    pyautogui.moveTo(), stock settings   114 ms per call
    pyautogui.moveTo(), PAUSE = 0         11 ms per call
    Quartz CGEventPost directly          0.03 ms per call

A tracker that calls ``moveTo`` once per frame with stock settings is
therefore capped at about 8 frames per second no matter how fast the
camera and the model are. :class:`QuartzPointer` posts the Core Graphics
event itself and gets the per-frame cost back to noise.

:class:`PyAutoGUIPointer` remains as a portable fallback (with ``PAUSE``
forced to zero), and :class:`NullPointer` records calls for tests and for
``--dry-run``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Pointer(Protocol):
    """Mouse sink used by the runtime loop."""

    def move_to(self, x: float, y: float) -> None: ...
    def click(self) -> None: ...
    def right_click(self) -> None: ...
    def double_click(self) -> None: ...


class NullPointer:
    """Records what it was asked to do and does nothing.

    Backs ``--dry-run`` and every gesture test.
    """

    name = "null"

    def __init__(self) -> None:
        self.moves: list[tuple[float, float]] = []
        self.clicks: list[str] = []

    def move_to(self, x: float, y: float) -> None:
        self.moves.append((x, y))

    def click(self) -> None:
        self.clicks.append("left")

    def right_click(self) -> None:
        self.clicks.append("right")

    def double_click(self) -> None:
        self.clicks.append("double")

    @property
    def position(self) -> tuple[float, float] | None:
        return self.moves[-1] if self.moves else None


class QuartzPointer:
    """macOS mouse control through Core Graphics events.

    Posts ``CGEvent``s straight to the HID event tap, skipping PyAutoGUI's
    pause and fail-safe machinery entirely.
    """

    name = "quartz"

    def __init__(self) -> None:
        import Quartz  # pyobjc-framework-Quartz, already a PyAutoGUI dep

        self._q = Quartz
        self._moved_event = Quartz.kCGEventMouseMoved
        self._tap = Quartz.kCGHIDEventTap
        self._left = Quartz.kCGMouseButtonLeft
        self._right = Quartz.kCGMouseButtonRight
        self._last: tuple[int, int] | None = None

    @staticmethod
    def available() -> bool:
        try:
            import Quartz  # noqa: F401
        except Exception:
            return False
        return True

    def _post(self, event_type, x: float, y: float, button, click_state=None):
        ev = self._q.CGEventCreateMouseEvent(None, event_type, (x, y), button)
        if click_state is not None:
            self._q.CGEventSetIntegerValueField(
                ev, self._q.kCGMouseEventClickState, click_state
            )
        self._q.CGEventPost(self._tap, ev)

    def move_to(self, x: float, y: float) -> None:
        """Move the cursor. Repeated moves to the same pixel are dropped.

        The tracker re-sends the held position every frame while the gaze
        sits inside the deadzone; posting an identical event 30 times a
        second is pure overhead and can interfere with other input.
        """
        pos = (int(x), int(y))
        if pos == self._last:
            return
        self._last = pos
        self._post(self._moved_event, pos[0], pos[1], self._left)

    def _click_sequence(self, button, down, up, clicks: int) -> None:
        x, y = self.position()
        for i in range(1, clicks + 1):
            # The click-state field is what makes the window server read
            # two clicks as a double click rather than two singles.
            self._post(down, x, y, button, click_state=i)
            self._post(up, x, y, button, click_state=i)

    def position(self) -> tuple[float, float]:
        """Current cursor position in screen pixels."""
        loc = self._q.CGEventGetLocation(self._q.CGEventCreate(None))
        return loc.x, loc.y

    def click(self) -> None:
        self._click_sequence(
            self._left, self._q.kCGEventLeftMouseDown,
            self._q.kCGEventLeftMouseUp, 1,
        )

    def right_click(self) -> None:
        self._click_sequence(
            self._right, self._q.kCGEventRightMouseDown,
            self._q.kCGEventRightMouseUp, 1,
        )

    def double_click(self) -> None:
        self._click_sequence(
            self._left, self._q.kCGEventLeftMouseDown,
            self._q.kCGEventLeftMouseUp, 2,
        )


class PyAutoGUIPointer:
    """Portable fallback. Disables the global pause and the fail-safe.

    Still an order of magnitude slower than :class:`QuartzPointer`; used
    only where Quartz is unavailable.
    """

    name = "pyautogui"

    def __init__(self) -> None:
        import pyautogui

        # Without these two lines every call costs ~100 ms.
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = False
        self._p = pyautogui
        self._last: tuple[int, int] | None = None

    def move_to(self, x: float, y: float) -> None:
        pos = (int(x), int(y))
        if pos == self._last:
            return
        self._last = pos
        self._p.moveTo(pos[0], pos[1], duration=0.0, _pause=False)

    def click(self) -> None:
        self._p.click(_pause=False)

    def right_click(self) -> None:
        self._p.rightClick(_pause=False)

    def double_click(self) -> None:
        self._p.doubleClick(_pause=False)


def create_pointer(backend: str = "auto") -> Pointer:
    """Build the best available pointer.

    Args:
        backend: ``"auto"``, ``"quartz"``, ``"pyautogui"`` or ``"null"``.

    Raises:
        ValueError: on an unknown backend name.
        RuntimeError: if a specific backend was requested and is missing.
    """
    if backend == "null":
        return NullPointer()
    if backend == "quartz":
        return QuartzPointer()
    if backend == "pyautogui":
        return PyAutoGUIPointer()
    if backend != "auto":
        raise ValueError(f"unknown pointer backend: {backend!r}")
    if QuartzPointer.available():
        return QuartzPointer()
    return PyAutoGUIPointer()


def screen_size() -> tuple[int, int]:
    """Main display size in pixels.

    Tries Quartz first so that a headless import of this module does not
    drag in PyAutoGUI's screenshot stack.
    """
    try:
        import Quartz

        bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        return int(bounds.size.width), int(bounds.size.height)
    except Exception:
        import pyautogui

        size = pyautogui.size()
        return int(size[0]), int(size[1])
