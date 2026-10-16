#!/usr/bin/env python3
"""assets/nazar.svg -> PNG set + assets/Fatma.icns

Run it after editing the SVG. The PNGs are also what the control panel's
favicon is built from, so the icon in the browser tab and the icon in the
Dock cannot drift apart.

    .venv/bin/python make_icon.py
"""

import base64
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
SVG = ASSETS / "nazar.svg"
SIZES = (16, 32, 64, 128, 256, 512, 1024)


def render(svg: pathlib.Path, out: pathlib.Path, size: int) -> None:
    """Rasterise the SVG at one size, through AppKit.

    AppKit is used rather than a drawing library because it is already
    present on every machine this runs on — pyobjc comes in with PyAutoGUI
    — and it renders the SVG the same way the system will.
    """
    from AppKit import (
        NSBitmapImageRep,
        NSGraphicsContext,
        NSImage,
        NSMakeSize,
        NSPNGFileType,
    )

    img = NSImage.alloc().initWithContentsOfFile_(str(svg))
    if img is None:
        sys.exit(f"could not read {svg}")
    img.setSize_(NSMakeSize(size, size))
    rep = NSBitmapImageRep.alloc().\
        initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
            None, size, size, 8, 4, True, False,
            "NSCalibratedRGBColorSpace", 0, 0)
    ctx = NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(ctx)
    img.drawInRect_(((0, 0), (size, size)))
    NSGraphicsContext.restoreGraphicsState()
    out.write_bytes(bytes(rep.representationUsingType_properties_(
        NSPNGFileType, {})))


def main() -> int:
    if not SVG.exists():
        sys.exit(f"missing {SVG}")
    iconset = ASSETS / "Fatma.iconset"
    iconset.mkdir(parents=True, exist_ok=True)
    for size in SIZES:
        render(SVG, iconset / f"icon_{size}x{size}.png", size)
        half = size // 2
        if half in SIZES:
            render(SVG, iconset / f"icon_{half}x{half}@2x.png", size)

    icns = ASSETS / "Fatma.icns"
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)],
                   check=True)

    # The panel embeds the favicon rather than serving a file, so the page
    # stays a single self-contained document.
    png = (iconset / "icon_64x64.png").read_bytes()
    (ASSETS / "favicon.b64").write_text(base64.b64encode(png).decode())

    print(f"wrote {icns}")
    print(f"wrote {ASSETS / 'favicon.b64'} ({len(png)} bytes of PNG)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
