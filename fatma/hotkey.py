"""A global keyboard shortcut, for stopping the tracker from anywhere.

This is a safety control more than a convenience. When the pointer
misbehaves — a stale calibration, a camera knocked out of position — the
usual ways out are gone: the preview window may not have focus, and
clicking a button is exactly the thing that has stopped working. A shortcut
that is caught system-wide stops the pointer whatever is in front.

It is built on a Core Graphics event tap, which needs the Accessibility
permission the tracker already requires to move the pointer at all. The tap
is created in listen-only mode, so the key still reaches whatever
application is in front: silently swallowing a combination in every other
program is not this tool's business.
"""

from __future__ import annotations

import threading
from typing import Callable

#: Virtual key codes (kVK_ANSI_*) for the keys offered. Escape is
#: deliberately absent: option+command+escape is Force Quit Applications,
#: and shadowing the escape hatch of last resort is a poor trade.
KEYCODES = {
    "q": 12, "w": 13, "e": 14, "r": 15, "t": 17, "y": 16, "u": 32,
    "s": 1, "d": 2, "f": 3, "g": 5, "h": 4, "j": 38, "k": 40, "l": 37,
    "x": 7, "c": 8, "period": 47,
}

#: Known clashes with option+command, shown next to the key picker.
#:
#: These come from documentation and experience, not from this machine: a
#: Mac application's key equivalents are compiled into it, not readable
#: from a preference file. What *is* checkable — the system shortcut table
#: and user overrides — was checked and is clear. For anything else the
#: panel's "last pressed" readout settles it: press the combination and
#: see whether it arrives.
#:
#: The tap is listen-only, so a clash means both things happen rather than
#: the shortcut failing. That is a nuisance for a console, and worse for
#: something like Close All Windows.
CONFLICTS = {
    "c": "Safari: JavaScript console",
    "d": "macOS: turn Dock hiding on/off",
    "f": "Safari: search with Google",
    "h": "macOS: Hide Others (every application)",
    "j": "Chrome / Edge / Firefox: developer console",
    "k": "Firefox: web console",
    "l": "Safari: show downloads",
    "t": "several apps: show/hide toolbar",
    "u": "Safari / Chrome: view page source",
    "w": "macOS: Close All Windows (every application)",
    "y": "Finder: Quick Look slideshow",
}

#: Offered keys with no clash known to this project.
CLEAN_KEYS = tuple(k for k in KEYCODES if k not in CONFLICTS)


class HotkeyError(RuntimeError):
    """Raised when the shortcut cannot be registered."""


def describe(key: str, cmd: bool = True, alt: bool = True,
             shift: bool = False, ctrl: bool = False) -> str:
    """Human-readable form, in the order macOS writes modifiers."""
    parts = []
    if ctrl:
        parts.append("ctrl")
    if alt:
        parts.append("option")
    if shift:
        parts.append("shift")
    if cmd:
        parts.append("command")
    parts.append(key.upper() if len(key) == 1 else key)
    return " + ".join(parts)


class GlobalHotkey:
    """Watches for one key combination across the whole session.

    Args:
        on_fire: Called from the tap's thread when the combination is
            pressed. Keep it short and thread-safe — the tracker's handler
            only sets a flag that the capture loop reads.
        key: Key name from :data:`KEYCODES`.
        cmd, alt, shift, ctrl: Required modifiers. Any *other* modifier
            being held means no match, so Option+Command+J does not also
            fire on Shift+Option+Command+J.
    """

    def __init__(
        self,
        on_fire: Callable[[], None],
        key: str = "j",
        cmd: bool = True,
        alt: bool = True,
        shift: bool = False,
        ctrl: bool = False,
    ) -> None:
        if key not in KEYCODES:
            raise HotkeyError(f"unknown key: {key!r}")
        self.on_fire = on_fire
        self.key = key
        self.keycode = KEYCODES[key]
        self.cmd, self.alt, self.shift, self.ctrl = cmd, alt, shift, ctrl
        self._thread: threading.Thread | None = None
        self._loop = None
        self._tap = None
        self._started = threading.Event()
        self.error: str | None = None

    @property
    def description(self) -> str:
        return describe(self.key, self.cmd, self.alt, self.shift, self.ctrl)

    # --- matching ------------------------------------------------------

    def matches(self, keycode: int, flags: int, masks: dict) -> bool:
        """True when this event is exactly our combination.

        Separated from the tap so the logic can be tested without one.
        """
        if keycode != self.keycode:
            return False
        for name, wanted in (("cmd", self.cmd), ("alt", self.alt),
                             ("shift", self.shift), ("ctrl", self.ctrl)):
            held = bool(flags & masks[name])
            if held != wanted:
                return False
        return True

    # --- lifecycle -----------------------------------------------------

    def start(self, timeout: float = 2.0) -> bool:
        """Install the tap on a background thread.

        Returns False and sets :attr:`error` when it cannot be installed —
        almost always a missing Accessibility permission. That is not fatal:
        every control the shortcut offers also exists on the panel.
        """
        if self._thread is not None:
            raise HotkeyError("already started")
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="fatma-hotkey")
        self._thread.start()
        self._started.wait(timeout)
        return self.error is None and self._tap is not None

    def _run(self) -> None:
        try:
            import Quartz
        except Exception as exc:                   # pragma: no cover
            self.error = f"Quartz unavailable: {exc}"
            self._started.set()
            return

        masks = {
            "cmd": Quartz.kCGEventFlagMaskCommand,
            "alt": Quartz.kCGEventFlagMaskAlternate,
            "shift": Quartz.kCGEventFlagMaskShift,
            "ctrl": Quartz.kCGEventFlagMaskControl,
        }

        def callback(proxy, event_type, event, refcon):
            try:
                if event_type == Quartz.kCGEventKeyDown:
                    keycode = Quartz.CGEventGetIntegerValueField(
                        event, Quartz.kCGKeyboardEventKeycode)
                    flags = Quartz.CGEventGetFlags(event)
                    if self.matches(int(keycode), int(flags), masks):
                        self.on_fire()
                elif event_type in (Quartz.kCGEventTapDisabledByTimeout,
                                    Quartz.kCGEventTapDisabledByUserInput):
                    # The system disables a tap that was slow to respond;
                    # without re-enabling it the shortcut silently dies.
                    Quartz.CGEventTapEnable(self._tap, True)
            except Exception:
                # A raise here would tear down the tap and take the
                # shortcut with it, so failures stay contained.
                pass
            return event

        try:
            tap = Quartz.CGEventTapCreate(
                Quartz.kCGSessionEventTap,
                Quartz.kCGHeadInsertEventTap,
                # Listen-only: the key still reaches the front application.
                Quartz.kCGEventTapOptionListenOnly,
                Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown),
                callback,
                None,
            )
        except Exception as exc:                   # pragma: no cover
            self.error = f"could not create the event tap: {exc}"
            self._started.set()
            return
        if not tap:
            self.error = (
                "could not watch the keyboard. Grant Accessibility access "
                "in System Settings -> Privacy & Security -> Accessibility."
            )
            self._started.set()
            return

        self._tap = tap
        source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
        self._loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(self._loop, source,
                                  Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(tap, True)
        self._started.set()
        Quartz.CFRunLoopRun()

    def stop(self) -> None:
        """Remove the tap. Safe to call when it never started."""
        if self._loop is not None:
            try:
                import Quartz

                if self._tap is not None:
                    Quartz.CGEventTapEnable(self._tap, False)
                Quartz.CFRunLoopStop(self._loop)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._loop = None
        self._tap = None
        self._started.clear()
