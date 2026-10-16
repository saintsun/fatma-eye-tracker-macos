<div align="center">

# Fatma: Eye Tracker

**Move the mouse pointer with your eyes.**

No headset, no infrared, no dedicated hardware — just the webcam your laptop
already has. Everything runs on your machine, and no video ever leaves it.

![platform](https://img.shields.io/badge/platform-macOS%2013%2B-111?style=flat-square)
![python](https://img.shields.io/badge/python-3.12-111?style=flat-square)
![cost](https://img.shields.io/badge/cost-free-00c853?style=flat-square)
![license](https://img.shields.io/badge/license-MIT-111?style=flat-square)

***English** · [Türkçe](README.tr.md)*

*v0.2.0*

</div>

---

> **Not a medical device, and not something to depend on.** Accuracy is
> roughly 50–150 px — enough to steer toward a region of the screen, not to
> hit a small button reliably. Do not rely on it to call for help, to
> communicate in an emergency, or as anyone's only way of operating a
> computer. If you need dependable eye control, look at purpose-built
> infrared trackers and at [OptiKey](https://github.com/OptiKey/OptiKey).

---

## Install: one command

Open Terminal (⌘ + Space → type "Terminal" → Enter), paste the line below and
press Enter:

```bash
curl -fsSL https://raw.githubusercontent.com/saintsun/fatma-eye-tracker-macos/main/install.sh | bash
```

That is all. It takes 2–3 minutes, mostly downloading.

---

## Buy Fatma something

The project is free and stays free. If it earns its keep, you can feed Fatma.

[![Payoneer](https://img.shields.io/badge/Payoneer-Buy%20Fatma%20something-FF4800?style=for-the-badge&logo=payoneer&logoColor=white&labelColor=0D1117)](https://link.payoneer.com/Token?t=B8694A2C126D470EAE34E7CC7A0817D9&src=pl)
[![GitHub Sponsors](https://img.shields.io/badge/GitHub-Sponsor-39FF14?style=for-the-badge&logo=githubsponsors&logoColor=white&labelColor=0D1117)](https://github.com/sponsors/saintsun)

| | |
|---|---|
| $5 | A Turkish coffee |
| $10 | Döner wrap |
| $25 | İskender |
| $50 | Hookah session |
| $100 | A night at the pavyon |

---

## Install: the details

The command downloads and installs everything it needs:

| | |
|---|---|
| Project files | into `~/Fatma` |
| Homebrew and Python 3.12 | installed if missing |
| MediaPipe, OpenCV, NumPy | into a virtualenv inside the folder |
| Face landmark model | 3.6 MB |

**Requirements:** a Mac running macOS 13 or newer, with a webcam. Tested on
Apple Silicon; Intel should work but is untested.

MediaPipe is stable on Python 3.12. Newer interpreters often have no wheel,
which is why the installer insists on that version.

<details>
<summary>I would rather not use the Terminal</summary>

Download the ZIP from the repository, double-click to unpack it, then open
Terminal in that folder and run `./run.sh`. A double-clickable installer is
not shipped yet.
</details>

### After installing: two permissions

macOS asks for two permissions; both are required.

**Camera:** click **Allow** the first time it starts.

**Accessibility:** needed to move the pointer. If the pointer never moves,
this is almost always why:

```bash
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
```

Find **Terminal** (or iTerm) in the list and turn the switch **on**, then quit
and reopen the terminal.

### Your first session

**Double-click `Fatma.command`** in the `~/Fatma` folder. The control panel
opens in your browser; everything happens from there and you never need the
terminal again. The camera stays off until you press start, so nothing comes
on and nothing asks for permission just because you opened the page.

1. Press **Takibi başlat**. macOS asks for Camera access the first time; if
   anything goes wrong it is shown on the page, not in a terminal.
2. Press **Kalibrasyonu başlat**. The screen goes dark and a ring appears.
3. Look straight at the ring and **wait** — the hint turns green and says
   HAZIR when your gaze has settled. Only then press **SPACE**. Repeat for
   all 13 rings. Pressing early is the commonest way to ruin a calibration.
4. Read the two numbers it prints: the accuracy, and the gaze span.
5. Press **İmleci aç** to hand the pointer over to your eyes.

**Sit close to the screen**, keep your head still, and have light on your
face. The first of those matters more than every setting in the app put
together — see Accuracy below.

---

## What it does

- **You look, the pointer follows.** Iris position is tracked through the
  webcam and mapped to the screen by a calibration you do once per sitting.
- **Winks click.** Left wink is a click, right wink is a right click, raising
  both brows is a double click. An ordinary two-eyed blink does nothing.
- **It tells you how accurate it is.** After calibration it cross-validates
  itself and prints the error in pixels, so you know whether to trust it or
  do it again.
- **Free and fully local.** No API key, no account, no subscription, no
  internet connection after the install. Video never leaves the machine.

---

## Everyday use

| Key | What happens |
|---|---|
| **`c`** | start calibration — look at each ring, SPACE to confirm |
| **SPACE** | confirm the ring you are looking at |
| **ESC** | cancel calibration, keeping the previous one |
| **`m`** | pointer control on / off |
| **`s`** | change smoothing (4 levels) |
| **`a`** | drift anchor — look at the centre of the screen and press |
| **`A`** | clear the drift anchor |
| **`q`** | quit |

```bash
./run.sh --dry-run        # track and preview, never touch the mouse
./run.sh --benchmark      # 200 frames, per-stage timings, then exit
./run.sh --no-gestures    # turn off wink and brow clicks
./run.sh --help           # everything
```

---

## The control panel

A browser panel opens with the tracker, on `127.0.0.1` only. It prints its
address at startup:

```
panel: http://127.0.0.1:8766
```

Everything the keyboard does is there — start calibration, confirm a target,
turn the pointer on and off, anchor the drift — plus the live precision
readouts: frame rate, calibration accuracy in pixels, and jitter while your
gaze is held. The **Hassasiyet** tab says what to change when a number looks
wrong.

Useful because the preview window has to keep focus for the keys to work; the
panel does not. A carer can drive the calibration from a phone on the same
machine's browser while the person being calibrated looks at the targets.

`--no-panel` turns it off, `--panel-port N` pins the port.

---

## If something does not work

**The calibration ring never appears.** Use `./run.sh --calib-window normal`.
The full-screen window is prettier but some window managers eat its first
frame.

**The pointer does not move.** Is the pointer switched on — press `m` and
check the preview. If it still will not move, Accessibility permission is
missing; see above.

**It is wildly inaccurate.** Look at the accuracy line printed after
calibration. Above ~120 px means the calibration itself went wrong, not the
tracking: recalibrate, looking squarely at each ring and keeping your head
still. The message names the ring that spoiled it.

**It drifts over time.** Head position has shifted. Look at the centre of the
screen and press `a`. If that is not enough, recalibrate.

**The pointer shakes.** Press `s` a few times for heavier smoothing, or raise
the tremor threshold: `./run.sh --deadzone 15`.

**The camera does not open.** Camera permission, or another application is
holding the camera. Close Zoom, Teams, Photo Booth.

**Update:**

```bash
cd ~/Fatma && git pull && ./install.sh
```

Your calibration is preserved.

---

## Cost and privacy

- **Nothing to pay.** No API key, no account, no subscription.
- **Video never leaves the machine.** Face tracking runs locally. The only
  download is the model, during install.
- The app touches only the camera and Accessibility (to move the pointer).
- Your `calibration.json` describes your own eyes and seating position. It is
  excluded from the repository; do not paste it into issues.

---

## Uninstall

```bash
rm -rf ~/Fatma
```

Nothing is installed outside that folder. Homebrew and Python are left alone
because other programs use them. Remove the Camera and Accessibility entries
for your terminal in System Settings if you want them gone.

---

## For developers

Everything below is technical: how it works, what was measured, and the traps
found along the way.

### The flow

```
webcam → threaded grabber (newest frame only) → MediaPipe FaceLandmarker
      → iris offset in iris diameters → temporal median → One-Euro filter
      → thin-plate-spline calibration → drift offset → deadzone → CGEventPost

eyelid + brow geometry → gesture state machine → click events
```

The maths and the state machinery import no camera, screen or mouse, which is
what makes them testable.

```
fatma/
  config.py       Settings — every tunable, validated
  filters.py      OneEuroFilter, TemporalMedian
  landmarks.py    iris / eyelid / brow geometry
  tps.py          thin-plate-spline interpolation
  calibration.py  Calibrator and the calibration file format
  quality.py      accuracy measurement: cross-validation, jitter
  gestures.py     wink + brow state machine (emits events, never clicks)
  cursor.py       gaze → pixel pipeline: filter, map, deadzone, drift
  camera.py       threaded frame grabber
  panel.py        the browser control panel (127.0.0.1 only)
  pointer.py      mouse backends (Quartz / PyAutoGUI / Null)
  app.py          OpenCV + MediaPipe runtime loop and CLI
legacy/           the pre-refactor single-file version, for reference
parked/           gaze-only control, written but not wired in
```

Only `app.py` imports cv2, mediapipe or pyautogui.

### What governs accuracy (measured)

One number governs everything, and it is unforgiving: **the gaze signal
travels less than half an iris diameter across the entire screen.** From a
real calibration on a 1470×956 display:

| axis | gaze range | screen | scale |
|---|---|---|---|
| horizontal | 0.401 iris diameters | 1470 px | 1 unit = 3663 px |
| vertical | 0.200 iris diameters | 956 px | 1 unit = 4791 px |

So 50 px of cursor accuracy demands 0.014 iris diameters of landmark
precision. At 640×480 the iris is about 18 px across, which makes **one pixel
of landmark error worth ~206 screen pixels**. The whole system runs in the
sub-pixel regime, and no amount of filter tuning changes that.

What does change it, in order of effect:

1. **Capture resolution.** 640 → 1280 halves the error, → 1920 thirds it. It
   used to be too expensive; now that the pointer call costs 0.03 ms instead
   of 114 ms, it is free. Default is 1280×720, and a warning is printed if the
   camera refuses that mode — silently dropping to 640 would halve precision
   invisibly.
2. **Both eyes.** Two eyes measure the same gaze with independent noise, so
   averaging cuts it by **√2**. `--eye auto` (default) averages while they
   agree and falls back to one eye when they do not: strabismus, a drooping
   lid, an eye out of frame.
3. **A steadier reference.** The iris offset is measured against a reference
   point, which used to be two eye corners and is now **six landmarks** —
   each corner plus the contour point above and below it. Averaging cuts
   reference noise by about √3, and the vertical symmetry means a lowering
   eyelid cancels to first order instead of leaking into the vertical channel.
   `--reference corners` restores the old behaviour.
4. **13 targets.** Spline error peaks furthest from any target, along the
   middle of each edge. The 13-point grid puts a target exactly there.

### Resource usage (measured, M3)

| | |
|---|---|
| Per-frame tracking cost | 2–5 ms, so the loop is camera-bound |
| MediaPipe inference | 1.5–6 ms at 640×480 through 1920×1080 |
| Pointer move (Quartz) | 0.03 ms |
| 30 fps budget | 33.3 ms |

### Measuring accuracy

- **Calibration accuracy** is cross-validated the moment calibration ends.
  Each target is held out in turn, the spline refitted on the rest, and the
  held-out target predicted. That is the honest estimate of behaviour
  *between* the targets, which is where the cursor spends all its time.
- **Jitter** is live in the preview: how far the cursor wanders in pixels
  while the gaze is held. The "best" figure is the quietest fixation seen,
  which is the noise floor the setup can reach.

They measure different things and have different cures: a poor calibration
score means recalibrate; high jitter means resolution and averaging.

### Traps found along the way

**PyAutoGUI cost 114 ms per frame.** Every public call is wrapped in a
decorator that sleeps for `pyautogui.PAUSE` afterwards, 0.1 s by default, and
runs a fail-safe check that queries the cursor position before it. One
`moveTo` per frame caps the loop near 8 fps regardless of how fast the camera
and the model are. Setting `PAUSE = 0` brings it to 11 ms; posting the Core
Graphics event directly brings it to 0.03 ms. This one bug was the difference
between "the tracker is slow" and "the tracker is camera-bound".

**Latency crept upward through the session.** `VideoCapture.read()` returns
frames in order, so a loop slower than the camera falls progressively further
behind and the lag grows the longer you use it — which no filter tuning can
fix. Capture now runs on its own thread and keeps only the newest frame.

**Calibration could not be completed at all.** The target image was pushed to
the window only when the target index changed, and the window was recreated
and re-asserted full-screen each time, so the window-server's full-screen
transition swallowed that single frame. The window stayed blank — and because
the preview had already been destroyed, no OpenCV window was left to receive
keystrokes, so SPACE never reached the application. The window is now created
once per run and repainted every frame.

**MediaPipe deadlocked on exit.** The landmarker holds native resources behind
a serial dispatcher. Left to the garbage collector, its finaliser waits on an
executor that is already winding down and the process hangs instead of
quitting. It is closed explicitly now; `close()` is idempotent, so the
finaliser becomes a no-op.

**`np.eye(n) * np.inf` is NaN off the diagonal.** It was masking the diagonal
of a distance matrix to find coincident calibration points. `0 * inf` is NaN,
so every off-diagonal cell became NaN, the comparison never fired, and a
degenerate calibration reached the solver and failed there with a misleading
message instead of being caught.

**Timestamps collided once the loop got fast.** MediaPipe's video mode rejects
a timestamp that does not advance, and with 114 ms gone from every frame, two
frames landing inside the same millisecond stopped being hypothetical.

**Abandoning a calibration destroyed the working one.** Pressing `c` cleared
the mapping immediately, so backing out with ESC left the user with nothing.
The previous mapping is now set aside and restored on cancel, and on a failed
fit.

### Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

383 tests. Coverage is 98–100% on the pure-logic modules; what is uncovered is
the I/O that needs a camera and a real mouse.

```bash
python -m pytest -m "not slow"     # skip the tests that run real inference
python -m pytest --cov=fatma       # coverage report
```

Tests build a synthetic face (`tests/conftest.py`) with eye corners exactly
0.1 apart and iris centres exactly 0.3 apart, so every derived measure is
asserted against an exact number rather than a tolerance.
`tests/test_precision.py` injects simulated landmark noise and measures the
spread of the signal, which is how the √2 binocular gain and the six-landmark
reference are established without a camera.

### Sit close to the screen

The one thing that governs accuracy more than any setting, and it is not in
the software: **how far you sit from the screen.** Further away, the screen
subtends a smaller visual angle, your eyes rotate less to cover it, and the
whole gaze signal shrinks into the noise.

Two sessions measured on the same machine:

| | horizontal | vertical | vertical amplification |
|---|---|---|---|
| sitting close | 0.401 | 0.200 | 4 791 px per unit |
| sitting back | 0.175 | 0.079 | 12 095 px per unit |

The second calibration passed through every control point exactly and still
put the cursor 208 px from the target on average, because a third of a pixel
of landmark noise had become a hundred pixels of cursor. Residuals cannot see
this at all — the span can, so it is measured and reported after every
calibration, with a warning when it is too small.

### Known limits

- **Head pose is not compensated.** Tracking uses eye movement only, so you
  have to keep your head still. This is the largest remaining source of error,
  and MediaPipe already exposes the head transform — it is simply not wired up
  yet.
- macOS only in practice. The tracking core is portable; the pointer backend
  and the window handling are not.
- **Every control is a key press,** so a person who can only move their eyes
  cannot start, calibrate or stop it without someone else present. A dwell
  engine and a gaze menu are written and waiting in `parked/`.
- Calibration is per sitting. It survives a resolution change, but not a
  change of camera angle or seating position.
- Not verified on real faces: how much of the simulated precision gain
  survives, whether a given camera really delivers 720p at 30 fps, and whether
  the anchor landmark indices are truly the corner neighbours in your model
  (they come from the documented FaceMesh topology).
  `--reference corners --eye right --width 640` restores the old behaviour for
  an A/B.

### Where to help

1. **Head-pose compensation.** Enable
   `output_facial_transformation_matrixes`, use yaw and pitch to correct the
   iris offset. Biggest remaining accuracy win.
2. **Linux and Windows pointer backends.** `fatma/pointer.py` already has the
   seam; `QuartzPointer` is the model to copy.
3. **Gaze-only control.** See `parked/README.md`.

Keep new logic out of `app.py` where you can — it is the only module that
cannot be tested without hardware.

---

## License

[MIT](LICENSE) © Eren Güneş

Built on [MediaPipe](https://github.com/google-ai-edge/mediapipe) and
[OpenCV](https://opencv.org/), both Apache 2.0. An independent project, not
affiliated with or endorsed by Google.
