#!/usr/bin/env python3
"""Build Fatma.app — double-click it, no terminal.

A ``.command`` file opens Terminal and leaves it sitting there. An ``.app``
does not, which is the whole point of this.

Getting there took two dead ends worth recording, because both look like
they should work:

* A shell script as the bundle's main executable. macOS will not run one
  any more. Launch Services starts the bundle, waits for the reply to the
  "open application" Apple event that a script cannot send, and kills it
  with error -1712 — before the script has run at all, so there is nothing
  in any log to explain it.
* A small compiled launcher that ``execv``s the virtualenv's Python. Same
  -1712: the handshake is the application's job, and replacing the process
  image with a plain interpreter means nobody answers.

So the bundle is built with ``osacompile``, whose applet binary is Apple's
own and speaks that protocol correctly. It launches the tracker detached
and exits; the tracker then claims the Dock name and icon for itself
through ``fatma.macapp``.

    .venv/bin/python make_app.py
"""

import pathlib
import plistlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
APP = ROOT / "Fatma.app"
ICNS = ROOT / "assets" / "Fatma.icns"

sys.path.insert(0, str(ROOT))
from fatma import LONG_NAME, __version__  # noqa: E402

SCRIPT = '''on run
	set projectPath to "__PROJECT__"
	-- Keep this handler fast. "open" waits for it to return, and anything
	-- slow here makes the launch look like a hang: importing mediapipe to
	-- check the install took two seconds and earned a -1712 every time,
	-- even though the tracker had started perfectly.
	try
		do shell script "test -x " & quoted form of (projectPath & "/.venv/bin/python") & " -a -f " & quoted form of (projectPath & "/face_landmarker.task")
	on error
		display dialog "Fatma kurulumu tamamlanmamis. Terminal acilacak." buttons {"Tamam"} default button 1 with title "Fatma"
		do shell script "open -a Terminal " & quoted form of (projectPath & "/install.sh")
		return
	end try
	-- Detached, with the log as the only record: launched from Finder
	-- there is no terminal for anything to print to.
	do shell script "cd " & quoted form of projectPath & " && PYTHONUNBUFFERED=1 nohup ./.venv/bin/python -m fatma >> fatma.log 2>&1 &"
end run
'''

#: Keys osacompile does not set, or sets wrongly for our purposes.
OVERRIDES = {
    "CFBundleName": "Fatma",
    "CFBundleDisplayName": "Fatma",
    "CFBundleIdentifier": "com.saintsun.fatma",
    "CFBundleShortVersionString": __version__,
    "CFBundleVersion": __version__,
    "NSHighResolutionCapable": True,
    # Required, or the camera prompt never appears and the process is
    # killed the moment it touches the camera.
    "NSCameraUsageDescription":
        "Fatma tracks your eyes through the camera to move the pointer. "
        "Video never leaves this machine.",
    "NSAppleEventsUsageDescription":
        "Fatma starts the eye tracker and its control panel.",
}


def main() -> int:
    if not ICNS.exists():
        sys.exit(f"missing {ICNS} — run make_icon.py first")
    if shutil.which("osacompile") is None:
        sys.exit("osacompile not found; it ships with macOS")

    if APP.exists():
        shutil.rmtree(APP)

    with tempfile.TemporaryDirectory() as tmp:
        source = pathlib.Path(tmp) / "fatma.applescript"
        source.write_text(SCRIPT.replace("__PROJECT__", str(ROOT)))
        build = subprocess.run(
            ["osacompile", "-o", str(APP), str(source)],
            capture_output=True, text=True,
        )
        if build.returncode != 0:
            sys.exit(f"osacompile failed:\n{build.stderr}")

    # The applet carries its icon as Resources/applet.icns, so replacing
    # that file is all it takes — no Info.plist change needed for it.
    for name in ("applet.icns", "droplet.icns"):
        target = APP / "Contents" / "Resources" / name
        if target.exists():
            shutil.copy2(ICNS, target)

    plist_path = APP / "Contents" / "Info.plist"
    info = plistlib.loads(plist_path.read_bytes())
    info.update(OVERRIDES)
    plist_path.write_bytes(plistlib.dumps(info))

    # Editing the bundle invalidates osacompile's signature, and an invalid
    # one is refused where no signature at all would have been accepted.
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(APP)],
                   check=False, capture_output=True)

    # Nudge Launch Services so Finder shows the new icon and name without
    # waiting for it to notice by itself.
    subprocess.run(
        ["/System/Library/Frameworks/CoreServices.framework/Frameworks/"
         "LaunchServices.framework/Support/lsregister", "-f", str(APP)],
        check=False, capture_output=True,
    )

    print(f"built {APP}")
    print(f"  {LONG_NAME} {__version__}")
    print("  double-click it: no terminal, and the panel opens by itself")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
