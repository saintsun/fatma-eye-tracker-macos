"""Tunable parameters for Fatma, in one place.

Everything the runtime loop reads lives in :class:`Settings`, so tests can
build a variant without touching module globals and the CLI can override
values without editing source.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "face_landmarker.task")
DEFAULT_CALIB_PATH = os.path.join(PROJECT_ROOT, "calibration.json")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# One-Euro presets: (min_cutoff, beta).
#   min_cutoff — smoothing while the gaze is still (lower = steadier fixation)
#   beta       — how fast the filter relaxes during motion (higher = snappier)
SMOOTHING_LEVELS: tuple[tuple[float, float], ...] = (
    (1.5, 0.80),   # snappy  — minimal lag, some jitter at rest
    (0.6, 0.50),   # default — smooth fixation, fast saccades
    (0.3, 0.30),   # smooth  — heavy idle smoothing, still responsive
    (0.15, 0.15),  # max     — for users with limited motion
)

VALID_EYES = ("auto", "left", "right", "both")
VALID_REFERENCES = ("anchors", "corners")
VALID_GRIDS = (9, 13)
VALID_LANGUAGES = ("en", "tr")
VALID_CALIB_WINDOWS = ("fullscreen", "normal")


@dataclass
class Settings:
    """Runtime configuration.

    Attributes are grouped by subsystem. All durations are seconds, all
    pixel values are screen pixels, and all landmark-derived thresholds are
    dimensionless ratios.
    """

    # --- capture -------------------------------------------------------
    camera_index: int = 0
    #: Capture resolution is the dominant lever on precision. The gaze
    #: signal spans well under half an iris diameter across the whole
    #: screen, so at 640x480 a single pixel of landmark error becomes
    #: roughly 200 screen pixels of cursor error. Doubling the capture
    #: width halves that. It used to cost too much CPU to afford; with the
    #: pointer backend no longer burning 114 ms a frame, it no longer does.
    cam_width: int = 1280
    cam_height: int = 720
    cam_fps: int = 30
    threaded_capture: bool = True

    # --- tracking ------------------------------------------------------
    #: "auto" averages both eyes while they agree (sqrt(2) less noise) and
    #: falls back to one eye when they do not.
    dominant_eye: str = "auto"
    #: Which landmarks form the "looking straight ahead" reference.
    eye_reference: str = "anchors"
    #: Iris-diameter gap above which the two eyes are treated as
    #: disagreeing rather than merely noisy.
    max_eye_disagreement: float = 0.06
    median_window: int = 5
    smoothing_index: int = 1
    #: Vertical channel carries roughly half the dynamic range of the
    #: horizontal one, so it gets a proportionally heavier filter.
    vertical_cutoff_scale: float = 0.6
    vertical_beta_scale: float = 0.8

    # --- cursor --------------------------------------------------------
    deadzone_px: float = 8.0
    mouse_enabled_at_start: bool = False

    # --- gestures ------------------------------------------------------
    ear_closed: float = 0.16
    ear_open: float = 0.22
    closed_hold_s: float = 0.20
    brow_raise_ratio: float = 1.15
    brow_guard_ratio: float = 1.10
    brow_hold_s: float = 0.20
    #: Brow-raise double-click, off by default: looking at the top of
    #: the screen raises the eyebrows on its own, so the gesture fires
    #: by itself exactly when the user is trying to reach a menu bar.
    brow_enabled: bool = False
    click_cooldown_s: float = 0.45
    brow_ema_alpha: float = 0.001
    gestures_enabled: bool = True

    # --- calibration ---------------------------------------------------
    #: Minimum samples behind a confirmed point. At 30 fps the old
    #: value of 5 was a sixth of a second — short enough to capture
    #: the gaze still travelling to the target.
    calib_min_samples: int = 12
    calib_sample_window: int = 15
    calib_clamp_margin: float = 0.05
    #: 9 or 13 targets. The spline is exact at each target and interpolates
    #: between them, so adding the four mid-edge points of the 13-grid cuts
    #: the error in the regions furthest from any target.
    calib_grid: int = 13
    #: How tightly samples must cluster before a point is accepted.
    calib_stability: float = 0.02

    # --- model / files -------------------------------------------------
    model_path: str = DEFAULT_MODEL_PATH
    calibration_path: str = DEFAULT_CALIB_PATH

    # --- ui ------------------------------------------------------------
    #: Applies to the camera overlay and the browser panel alike.
    language: str = "en"
    show_preview: bool = True
    #: The preview frame is scaled to this width before the overlay is
    #: drawn, which is what actually determines the window size.
    preview_width: int = 640
    preview_height: int = 360
    #: macOS keeps the menu bar on top of "fullscreen" OpenCV windows, so
    #: the canvas origin sits this many pixels below screen y=0.
    macos_menu_bar_h: int = 25
    target_size: int = 60
    #: Global shortcut that stops the pointer from anywhere. A safety
    #: control: when the pointer misbehaves, clicking a button is the
    #: very thing that has stopped working.
    hotkey_enabled: bool = True
    #: G is clear of macOS, Safari, Chrome and Edge, it sits on the home
    #: row where it can be found without looking, and the nearest
    #: mis-press is harmless: command+G alone is Find Next. X and S are
    #: equally unbound but sit next to Cut and Save.
    hotkey_key: str = "g"
    #: The browser control panel: every keyboard action, plus live
    #: precision readouts, on 127.0.0.1 only.
    panel_enabled: bool = True
    #: None lets the panel pick the first free port in its range.
    panel_port: int | None = None
    #: How the calibration target window is shown. "fullscreen" hides
    #: every distraction; "normal" keeps a title bar and does not cover the
    #: menu bar, but paints reliably on window managers where the
    #: full-screen transition eats the first frame.
    calib_window: str = "fullscreen"

    smoothing_levels: tuple[tuple[float, float], ...] = field(
        default=SMOOTHING_LEVELS
    )

    def __post_init__(self) -> None:
        if self.dominant_eye not in VALID_EYES:
            raise ValueError(
                f"dominant_eye must be one of {VALID_EYES}, "
                f"got {self.dominant_eye!r}"
            )
        if self.eye_reference not in VALID_REFERENCES:
            raise ValueError(
                f"eye_reference must be one of {VALID_REFERENCES}, "
                f"got {self.eye_reference!r}"
            )
        if self.calib_grid not in VALID_GRIDS:
            raise ValueError(
                f"calib_grid must be one of {VALID_GRIDS}, "
                f"got {self.calib_grid}"
            )
        if self.max_eye_disagreement <= 0:
            raise ValueError("max_eye_disagreement must be positive")
        if self.panel_port is not None and not (1024 <= self.panel_port <= 65535):
            raise ValueError("panel_port must be between 1024 and 65535")
        if self.language not in VALID_LANGUAGES:
            raise ValueError(
                f"language must be one of {VALID_LANGUAGES}, "
                f"got {self.language!r}"
            )
        if self.calib_window not in VALID_CALIB_WINDOWS:
            raise ValueError(
                f"calib_window must be one of {VALID_CALIB_WINDOWS}, "
                f"got {self.calib_window!r}"
            )
        if not 0 <= self.smoothing_index < len(self.smoothing_levels):
            raise ValueError(
                f"smoothing_index out of range: {self.smoothing_index}"
            )
        if self.median_window < 1:
            raise ValueError("median_window must be >= 1")
        if self.ear_open <= self.ear_closed:
            raise ValueError(
                "ear_open must exceed ear_closed for hysteresis to work"
            )
        if self.brow_raise_ratio <= 1.0:
            raise ValueError("brow_raise_ratio must be > 1.0")
        if self.brow_guard_ratio > self.brow_raise_ratio:
            raise ValueError(
                "brow_guard_ratio must not exceed brow_raise_ratio, "
                "otherwise the baseline absorbs real raises"
            )
        if not 0.0 < self.brow_ema_alpha < 1.0:
            raise ValueError("brow_ema_alpha must be in (0, 1)")

    def smoothing(self) -> tuple[float, float]:
        """(min_cutoff, beta) for the currently selected preset."""
        return self.smoothing_levels[self.smoothing_index]

    def cycle_smoothing(self) -> tuple[float, float]:
        """Advance to the next preset and return it."""
        self.smoothing_index = (self.smoothing_index + 1) % len(
            self.smoothing_levels
        )
        return self.smoothing()
