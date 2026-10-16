"""Make the process look like Fatma rather than like Python.

Launched from a bundle, the Dock still shows "Python" with the Python
rocket: the launcher hands off to the interpreter in the virtualenv, which
lives outside the bundle, so macOS reads the application identity from
*that* executable instead. The thorough fix is to link libpython and call
``Py_BytesMain`` in-process, keeping the executable inside the bundle.

This is the cheap fix, and it is enough: the running application's name and
icon can both be set from AppKit once Python is up. The Dock may show
Python for the instant before that happens.

Everything here is best-effort. A wrong icon is cosmetic, and nothing in it
is worth failing a launch over.
"""

from __future__ import annotations

import pathlib
import sys

ICON = pathlib.Path(__file__).resolve().parent.parent / "assets" / "Fatma.icns"


def adopt_identity(name: str = "Fatma", icon: pathlib.Path | None = None,
                   register: bool = True) -> bool:
    """Set the Dock name and icon, and check in with Launch Services.

    The check-in matters as much as the name. Launched from a bundle, an
    application has to tell the window server it has finished launching;
    this one spends its first seconds serving a web panel and opens no
    window at all, so Launch Services concluded it had hung and killed it
    with error -1712 before anything appeared. Creating the NSApplication
    and calling ``finishLaunching`` is that check-in.

    Returns True if anything was applied. Safe to call anywhere: off macOS,
    or without AppKit, it does nothing.
    """
    if sys.platform != "darwin":
        return False
    try:
        from AppKit import NSApplication, NSImage
        from Foundation import NSBundle
    except Exception:
        return False

    applied = False
    try:
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None:
            # This is what the Dock and the menu bar read for the name.
            info["CFBundleName"] = name
            applied = True
    except Exception:
        pass

    if register:
        try:
            app = NSApplication.sharedApplication()
            # Regular, so it gets a Dock tile and a menu bar entry rather
            # than running as a faceless agent.
            app.setActivationPolicy_(0)
            app.finishLaunching()
            applied = True
        except Exception:
            pass

    path = icon or ICON
    try:
        if path.exists():
            image = NSImage.alloc().initWithContentsOfFile_(str(path))
            if image is not None:
                NSApplication.sharedApplication().setApplicationIconImage_(image)
                applied = True
    except Exception:
        pass
    return applied


def pump_events(limit: int = 12) -> int:
    """Service the Cocoa event queue, and return how many events ran.

    An application with a Dock tile is expected to drain its event queue.
    This one spends its idle time serving HTTP and opens no window at all,
    so launched from Finder it never drained anything, macOS decided it had
    hung, and killed it — silently, with nothing in the log, which is a
    miserable thing to debug. During tracking OpenCV's waitKey does this
    for us; while idle, nobody else will.

    Best-effort, like everything else here: a failure to pump is not a
    reason to stop tracking.
    """
    if sys.platform != "darwin":
        return 0
    try:
        from AppKit import (
            NSAnyEventMask,
            NSApplication,
            NSDefaultRunLoopMode,
            NSDate,
        )
    except Exception:
        return 0
    try:
        app = NSApplication.sharedApplication()
    except Exception:
        return 0
    served = 0
    for _ in range(limit):
        try:
            event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                NSAnyEventMask, NSDate.distantPast(), NSDefaultRunLoopMode, True)
        except Exception:
            break
        if event is None:
            break
        try:
            app.sendEvent_(event)
        except Exception:
            pass
        served += 1
    return served
