#!/usr/bin/env python3
"""Attach assets/Fatma.icns to the launcher, so Finder shows the nazar.

macOS stores a custom file icon in the file's resource fork rather than
anywhere visible, so it has to be set through AppKit; there is no way to
do it by copying a file into place.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent


def main() -> int:
    icns = ROOT / "assets" / "Fatma.icns"
    target = ROOT / "Fatma.command"
    if not icns.exists() or not target.exists():
        print("nothing to do")
        return 0
    try:
        from AppKit import NSImage, NSWorkspace
    except Exception as exc:
        print(f"AppKit unavailable: {exc}")
        return 0
    image = NSImage.alloc().initWithContentsOfFile_(str(icns))
    if image is None:
        print("could not read the icon")
        return 1
    ok = NSWorkspace.sharedWorkspace().setIcon_forFile_options_(
        image, str(target), 0)
    print("icon set" if ok else "icon not set")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
