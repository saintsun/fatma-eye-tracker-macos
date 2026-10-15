# Fatma: deviceless eye tracker

Control the mouse pointer with your eyes, using nothing but the webcam your
laptop already has. No headset, no infrared illuminator, no dedicated
hardware, no subscription.

Iris landmarks come from MediaPipe's FaceLandmarker. A thin-plate-spline
calibration maps them to screen coordinates. Winks and brow raises become
clicks.

*Türkçe: [README.tr.md](README.tr.md)*

> **This is not a medical device, and not an assistive device you should
> depend on.** Accuracy is roughly 50–150 px — good enough to steer toward
> a region of the screen, not to hit a small button reliably. Do not rely
> on it to call for help, to communicate in an emergency, or as anyone's
> only means of operating a computer. If you need dependable eye control,
> look at purpose-built infrared trackers and at [OptiKey](https://github.com/OptiKey/OptiKey).

## Status

Working and tested on macOS (Apple Silicon, Python 3.12). Linux and Windows
are untested — the tracking core is portable, the pointer backend and the
window handling are not.

Every control is a key press today, which means a person who can only move
their eyes cannot start, calibrate or stop it without someone else present.
Gaze-only control is written but parked, see [`parked/`](parked/).

## Install

```bash
git clone <your-fork-url> Fatma
cd Fatma
./run.sh
```

First run creates the virtualenv, installs dependencies, downloads the
landmark model (3.6 MB) and starts. Arguments pass straight through:
`./run.sh --eye left`.

MediaPipe is stable on Python 3.12; 3.13 and newer often have no wheel.

### macOS permissions

Two permissions are needed on first run. Grant them, then quit and reopen
your terminal.

- **Camera** — System Settings → Privacy & Security → Camera
- **Accessibility** (to move the pointer) — System Settings → Privacy &
  Security → Accessibility

## Use

Press `c` and look at each target in turn, pressing SPACE on each. Then `m`
to hand the pointer over to your eyes.

| Key | Action |
| --- | --- |
| `c` | Start calibration — look at each target, SPACE to confirm |
| `SPACE` | Confirm the current calibration target |
| `ESC` | Cancel calibration, keeping the previous one |
| `m` | Toggle pointer control |
| `s` | Cycle smoothing (4 presets) |
| `a` | Anchor drift — look at the centre of the screen and press |
| `A` | Clear the drift anchor |
| `q` | Quit |

Gestures: **left wink** = click, **right wink** = right click, **raise both
brows** = double click. An ordinary two-eyed blink does nothing.

```bash
python -m fatma --dry-run        # track and preview, never touch the mouse
python -m fatma --benchmark      # 200 frames, per-stage timings, then exit
python -m fatma --eye left       # left / right / both / auto
python -m fatma --grid 9         # 9 calibration targets instead of 13
python -m fatma --no-gestures    # disable wink and brow clicks
python -m fatma --help           # everything
```

If the full-screen calibration target never appears, use
`--calib-window normal`.

## Accuracy

One number governs everything here, and it is unforgiving: **the gaze
signal travels less than half an iris diameter across the entire screen.**
Measured from a real calibration on a 1470×956 display:

| axis | gaze range | screen | scale |
| --- | --- | --- | --- |
| horizontal | 0.401 iris diameters | 1470 px | 1 unit = 3663 px |
| vertical | 0.200 iris diameters | 956 px | 1 unit = 4791 px |

So 50 px of cursor accuracy demands 0.014 iris diameters of landmark
precision. At 640×480 the iris is about 18 px across, which makes **one
pixel of landmark error worth ~206 screen pixels**. The whole system runs
in the sub-pixel regime, and no amount of filter tuning changes that.

What does change it, in order of effect:

1. **Capture resolution.** The largest lever. 640 → 1280 halves the error,
   → 1920 thirds it. It used to be too expensive; now that the pointer call
   costs 0.03 ms instead of 114 ms, it is free. Default is **1280×720**. If
   the camera refuses that mode, a warning is printed at startup — silently
   dropping to 640 would halve precision invisibly.
2. **Both eyes.** Two eyes measure the same gaze with independent noise, so
   averaging cuts it by **√2**. `--eye auto` (default) averages while they
   agree and falls back to one eye when they do not — strabismus, a drooping
   lid, an eye out of frame.
3. **A steadier reference.** The iris offset is measured against a reference
   point. That used to be two eye corners; it is now **six landmarks**,
   each corner plus the contour point above and below it. Averaging cuts
   reference noise by about √3, and the vertical symmetry means a lowering
   eyelid cancels to first order instead of leaking into the vertical
   channel. `--reference corners` restores the old behaviour.
4. **13 targets.** Spline error peaks furthest from any target — along the
   middle of each edge. The 13-point grid puts a target exactly there.

### Measuring it

So you can check rather than guess:

- **Calibration accuracy** is cross-validated the moment calibration ends.
  Each target is held out in turn, the spline is refitted on the rest, and
  the held-out target predicted. That is the honest estimate of behaviour
  *between* the targets, which is where the cursor spends all its time.
  Mean and worst error are printed in pixels, naming the target that spoiled
  it.
- **Jitter** is live in the preview: how far the cursor wanders in pixels
  while your gaze is held. The "best" figure is the quietest fixation seen,
  which is the noise floor your setup can reach.

They measure different things and have different cures: a poor calibration
score means recalibrate; high jitter means resolution and averaging.

### What is verified and what is not

`tests/test_precision.py` injects simulated landmark noise and measures the
spread of the resulting signal. The √2 gain from binocular fusion and the
six-landmark reference being quieter than two corners are both established
that way — they are ratios that follow from averaging independent samples,
so they hold whatever the real noise turns out to be.

Not verified, stated plainly: how much of that survives on real faces,
whether a given camera really delivers 720p at 30 fps, and whether the
anchor landmark indices are truly the corner neighbours in your model (they
are taken from the documented FaceMesh topology, not compared against live
output). `--reference corners --eye right --width 640` restores the old
behaviour for an A/B.

### Known limits

- **Keep your head still.** Tracking uses eye movement only; head pose is
  not compensated. This is the largest remaining source of error, and
  MediaPipe already exposes the head transform — it is simply not wired up
  yet.
- Light on your face helps a lot.
- Calibration is saved to `calibration.json` and restored at startup. It is
  stored in screen fractions, so it survives a resolution change, but not a
  change of camera angle or seating position.

## Performance

Per-frame tracking cost is ~2–5 ms, so the loop is bound by the camera, not
by the code. Getting there meant fixing PyAutoGUI:

| method | per frame |
| --- | --- |
| `pyautogui.moveTo()`, stock settings | **114 ms** |
| `pyautogui.moveTo()`, `PAUSE = 0` | 11 ms |
| Quartz `CGEventPost` directly | **0.03 ms** |

Every public PyAutoGUI call is wrapped in a decorator that sleeps for
`pyautogui.PAUSE` afterwards, 0.1 s by default. One `moveTo` per frame caps
the loop at about 8 fps no matter how fast the camera and the model are.
`fatma.pointer.QuartzPointer` posts the Core Graphics event itself.

The second problem was latency. `VideoCapture.read()` returns frames in
order, so a loop slower than the camera falls progressively further behind
and the lag grows through the session. `fatma.camera.FrameGrabber` captures
on its own thread and keeps only the newest frame.

Measure it on your own machine with `--benchmark`.

## Layout

The maths and the state machinery import no camera, screen or mouse, which
is what makes them testable.

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
  pointer.py      mouse backends (Quartz / PyAutoGUI / Null)
  app.py          OpenCV + MediaPipe runtime loop and CLI
legacy/           the pre-refactor single-file version, for reference
parked/           gaze-only control, written but not wired in
```

Only `app.py` imports cv2, mediapipe or pyautogui.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

254 tests. Coverage is 98–100% on the pure-logic modules; what is uncovered
is the I/O that needs a camera and a real mouse.

```bash
python -m pytest -m "not slow"     # skip the tests that run real inference
python -m pytest --cov=fatma       # coverage report
```

Tests build a synthetic face (`tests/conftest.py`) with eye corners exactly
0.1 apart and iris centres exactly 0.3 apart, so every derived measure can
be asserted against an exact number rather than a tolerance.

## Contributing

The most useful things to pick up, roughly in order:

1. **Head-pose compensation.** Enable
   `output_facial_transformation_matrixes`, use yaw and pitch to correct
   the iris offset. Biggest remaining accuracy win.
2. **Linux and Windows pointer backends.** `fatma/pointer.py` already has
   the seam; `QuartzPointer` is the model to copy.
3. **Gaze-only control.** `parked/` has a dwell engine and a gaze menu
   written and ready to wire in. See `parked/README.md`.

Keep new logic out of `app.py` where you can — that is the only module that
cannot be tested without hardware.

## License

MIT. See [LICENSE](LICENSE).

Built on [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache
2.0) and [OpenCV](https://opencv.org/) (Apache 2.0).
