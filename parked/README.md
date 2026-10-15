# Parked: gaze-only control

Written for the "user can only move their eyes" scenario, then parked when
the scope moved to precision. Not imported by the `fatma` package and not
covered by tests.

- `dwell.py` — dwell selection (hold your gaze to click). `DwellTracker`
  works in screen pixels; `GazeStability` works on the raw gaze signal, for
  calibration before a mapping exists. `ZoneSet` handles large targets.
- `menu.py` — full-screen gaze menu: pause, recalibrate, toggle clicking,
  quit, opened by dwelling in a screen corner.

Why it matters when that scenario returns: today every control is a key
press, including the nine SPACE presses that calibration needs. A user who
can only move their eyes cannot start the application, calibrate it, or
stop it without someone else present.

To revive: move the files back into `fatma/`, wire `GazeStability` into the
calibration loop in `app.py` in place of the SPACE handler, and give the
menu's cursor position the same value the tracker feeds the pointer.
