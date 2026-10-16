"""Fatma runtime: camera, landmark model, preview window and key handling.

This is the only module that touches OpenCV, MediaPipe and the mouse. All
of the maths and state machinery lives in sibling modules that import none
of those, which is what makes the rest of the package testable.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import numpy as np

from . import LONG_NAME, __version__
from .calibration import Calibrator
from .camera import DirectCapture, FrameGrabber, negotiated_size, open_capture
from .config import (
    MODEL_URL,
    VALID_CALIB_WINDOWS,
    VALID_EYES,
    VALID_GRIDS,
    VALID_REFERENCES,
    Settings,
)
from .cursor import CursorController
from .gestures import Gesture, GestureDetector
from .i18n import DEFAULT_LANGUAGE, LANGUAGES, t
from .landmarks import (
    LEFT_BROW_MID,
    LEFT_EYE_BOTTOM,
    LEFT_EYE_TOP,
    RIGHT_BROW_MID,
    RIGHT_EYE_BOTTOM,
    RIGHT_EYE_TOP,
    brow_height,
    ear_pair,
    normalized_iris_pair,
    visualisation_indices,
)
from .panel import Panel, PanelBridge
from .pointer import create_pointer, screen_size
from .quality import JitterMeter

# macOS will not shrink a window below the width of its title bar text, so
# a long title silently sets a floor on the preview size. The key list it
# used to carry is in the overlay and in the panel.
PREVIEW_WINDOW = "Fatma"
TARGET_WINDOW = "Fatma calibration target"

#: OpenCV's Hershey fonts are ASCII-only: cv2.putText draws "?" for every
#: other code point, so "sabit bakış" reaches the screen as "sabit bak????".
#: The overlay is a terse debug readout and the panel carries the real
#: Turkish text, so the drawing layer transliterates rather than taking on a
#: font-rendering dependency. Applying it at the one place that draws means
#: text added later cannot quietly break either.
_ASCII_FOLD = str.maketrans({
    "ı": "i", "İ": "I", "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G",
    "ü": "u", "Ü": "U", "ö": "o", "Ö": "O", "ç": "c", "Ç": "C",
    "â": "a", "î": "i", "û": "u", "—": "-", "–": "-", "’": "'",
    "“": '"', "”": '"', "…": "...", "·": "-", "×": "x",
})


def ascii_fold(text: str) -> str:
    """Make a string safe for cv2.putText, which cannot draw non-ASCII."""
    folded = text.translate(_ASCII_FOLD)
    return folded.encode("ascii", "replace").decode("ascii")


GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)


def make_target_image(
    screen_cx: int,
    screen_cy: int,
    screen_w: int,
    screen_h: int,
    menu_bar_h: int,
    target_size: int,
    step: int = 0,
    total: int = 0,
    ready: bool | None = None,
    lang: str = DEFAULT_LANGUAGE,
):
    """Dark fullscreen canvas with a bullseye at a given screen position.

    macOS keeps its menu bar above "fullscreen" OpenCV windows, so the
    canvas origin sits ``menu_bar_h`` pixels below screen y=0 and the
    bullseye is shifted up to compensate.

    The preview window is hidden during calibration, so the progress and
    the instruction are drawn here — otherwise the only place that says
    what to do is the terminal, which is behind a full-screen window.
    """
    import cv2

    canvas_h = screen_h - menu_bar_h
    img = np.full((canvas_h, screen_w, 3), 20, dtype=np.uint8)
    r = target_size // 2
    cx = min(max(screen_cx, r), screen_w - r)
    cy = min(max(screen_cy - menu_bar_h, r), canvas_h - r)
    # The ring itself carries the readiness, because that is the one place
    # the user is already looking. Reading a hint somewhere else means
    # moving the eyes off the target at the exact moment they are about to
    # confirm it, which is what the hint was trying to prevent.
    ring = GREEN if ready else YELLOW
    cv2.circle(img, (cx, cy), max(r - 2, 6), ring, 2 if not ready else 3)
    cv2.circle(img, (cx, cy), max(r // 2, 4), GREEN if ready else ORANGE, 1)
    cv2.circle(img, (cx, cy), 3, RED, -1)
    if ready:
        cv2.circle(img, (cx, cy), r + 10, GREEN, 1)

    if total:
        if ready is None:
            hint = f"{t(lang, 'calib.hint')}    ({t(lang, 'calib.cancel')})"
        elif ready:
            hint = t(lang, "calib.ready")
        else:
            # Pressing early is the single commonest way to ruin a
            # calibration, so the screen says plainly when not to.
            hint = t(lang, "calib.wait")
        _draw_edge_hints(img, hint, GREEN if ready else WHITE,
                         screen_w, canvas_h, cx, cy, r)
        _draw_progress(img, step, total, screen_w, canvas_h, cx, cy)
    return img


def _draw_progress(img, step, total, screen_w, canvas_h, cx, cy):
    """Step counter and one dot per target, in the corner furthest away.

    Pinned to a fixed corner it eventually sits under the ring — the
    targets reach into all four corners by design. The dots matter beyond
    counting: a window that paints is obviously alive, and the failure this
    replaced looked exactly like a frozen or missing one.
    """
    import cv2

    banner = f"{step}/{total}"
    (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
    dots_w = total * 18
    block_w = max(bw, dots_w)
    margin = 40
    corners = (
        (margin, margin + bh),
        (screen_w - block_w - margin, margin + bh),
        (margin, canvas_h - margin - 26),
        (screen_w - block_w - margin, canvas_h - margin - 26),
    )
    x, y = max(corners, key=lambda c: (c[0] + block_w / 2 - cx) ** 2
               + (c[1] - bh / 2 - cy) ** 2)
    cv2.putText(img, ascii_fold(banner), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                1.6, WHITE, 3)
    for i in range(total):
        colour = YELLOW if i < step else (70, 70, 70)
        cv2.circle(img, (x + 8 + i * 18, y + 22), 5, colour, -1)


def _draw_edge_hints(img, hint, colour, screen_w, canvas_h, cx, cy, r):
    """Repeat the hint on all four edges, skipping any the target covers.

    A single hint at the bottom is useless when the ring is at the top: the
    user has to look away from the target to read it. Repeating it on every
    edge means one copy is always in peripheral vision, wherever the ring
    happens to be, and none of them is worth looking directly at.
    """
    import cv2

    text = ascii_fold(hint)
    scale, thickness = 0.8, 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                  scale, thickness)
    margin = 46
    placements = (
        ((screen_w - tw) // 2, margin + th),                  # top
        ((screen_w - tw) // 2, canvas_h - margin),            # bottom
        (margin, (canvas_h + th) // 2),                       # left
        (screen_w - tw - margin, (canvas_h + th) // 2),       # right
    )
    # Clearance that keeps a hint from colliding with the ring it describes.
    keep_out = r + th + 28
    for x, y in placements:
        box_cx, box_cy = x + tw / 2, y - th / 2
        if (abs(box_cx - cx) < tw / 2 + keep_out
                and abs(box_cy - cy) < th / 2 + keep_out):
            continue
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, colour, thickness)


class StageTimer:
    """Accumulates per-stage wall time, for ``--benchmark``."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, stage: str, seconds: float) -> None:
        self.totals[stage] = self.totals.get(stage, 0.0) + seconds
        self.counts[stage] = self.counts.get(stage, 0) + 1

    def report(self) -> list[str]:
        return [
            f"  {stage:<12} {self.totals[stage] / self.counts[stage] * 1000:7.2f} ms/frame"
            for stage in sorted(self.totals, key=self.totals.get, reverse=True)
        ]


class TrackerApp:
    """Owns the capture loop and all the device-facing objects."""

    def __init__(self, settings: Settings, pointer_backend: str = "auto") -> None:
        self.s = settings
        self.screen_w, self.screen_h = screen_size()
        self.pointer = create_pointer(pointer_backend)
        self.calibrator = Calibrator(
            min_samples=settings.calib_min_samples,
            sample_window=settings.calib_sample_window,
            clamp_margin=settings.calib_clamp_margin,
            grid=settings.calib_grid,
            stability=settings.calib_stability,
        )
        self.cursor = CursorController(
            settings, self.calibrator, self.screen_w, self.screen_h
        )
        self.gestures = GestureDetector(
            ear_closed=settings.ear_closed,
            ear_open=settings.ear_open,
            closed_hold_s=settings.closed_hold_s,
            brow_raise_ratio=settings.brow_raise_ratio,
            brow_guard_ratio=settings.brow_guard_ratio,
            brow_hold_s=settings.brow_hold_s,
            cooldown_s=settings.click_cooldown_s,
            brow_ema_alpha=settings.brow_ema_alpha,
            brow_enabled=settings.brow_enabled,
        )
        self.mouse_on = settings.mouse_enabled_at_start
        self.preview_visible = False
        self._last_calib_idx = -1
        self._calib_window_open = False
        self._last_calib_ready: bool | None = None
        self._target_img: np.ndarray | None = None
        self._rgb: np.ndarray | None = None
        self._frame_times: deque[float] = deque(maxlen=60)
        self._mp_ts = -1
        self.jitter = JitterMeter()
        self.bridge = PanelBridge()
        self.panel = Panel(self.bridge, settings.panel_port) \
            if settings.panel_enabled else None
        self._quit = False
        #: True while the camera is open and the loop is running.
        self._tracking = False
        #: Last failure, surfaced on the panel instead of killing
        #: the process — a camera permission that has not been
        #: granted is the common case, and the user is looking at
        #: the browser, not at a terminal.
        self.last_error: str | None = None
        self._face_seen = False
        self._preview_sized = False
        self.cam_w, self.cam_h = settings.cam_width, settings.cam_height

    # --- setup ---------------------------------------------------------

    def _build_detector(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        if not os.path.exists(self.s.model_path):
            raise FileNotFoundError(
                f"landmark model missing: {self.s.model_path}\n"
                f"Download it with:\n  curl -fL -o "
                f"{os.path.basename(self.s.model_path)} {MODEL_URL}"
            )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=self.s.model_path
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return mp_vision.FaceLandmarker.create_from_options(options)

    def _next_timestamp_ms(self, now: float, t0: float) -> int:
        """Strictly increasing millisecond timestamp for MediaPipe.

        ``RunningMode.VIDEO`` rejects a timestamp that does not advance,
        and two frames can easily land inside the same millisecond once the
        pointer backend stops costing 100 ms per frame.
        """
        ts = int((now - t0) * 1000)
        if ts <= self._mp_ts:
            ts = self._mp_ts + 1
        self._mp_ts = ts
        return ts

    # --- preview window ------------------------------------------------

    def _show_preview_window(self) -> None:
        import cv2

        cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PREVIEW_WINDOW, self.s.preview_width,
                         self.s.preview_height)
        cv2.moveWindow(
            PREVIEW_WINDOW,
            max(0, self.screen_w - self.s.preview_width - 20),
            40,
        )
        self.preview_visible = True
        self._preview_sized = False

    def _preview_frame(self, frame):
        """Scale a capture frame down to the preview's own size.

        The window is sized by the image, not by ``resizeWindow``, so
        showing the capture frame directly makes the preview as large as the
        capture resolution — which is how raising capture to 720p turned a
        small corner window into one covering half the screen.

        Drawing the overlay on the scaled copy also keeps the text legible:
        the font size is fixed in pixels, so on a 1280-wide frame it would
        be half as large relative to the window.
        """
        import cv2

        height, width = frame.shape[:2]
        target = self.s.preview_width
        if width <= target:
            return frame.copy()
        scale = target / width
        return cv2.resize(frame, (target, max(1, round(height * scale))),
                          interpolation=cv2.INTER_AREA)

    def _hide_preview_window(self) -> None:
        import cv2

        try:
            cv2.destroyWindow(PREVIEW_WINDOW)
        except cv2.error:
            pass
        self.preview_visible = False

    # --- per-frame drawing ---------------------------------------------

    def _draw_overlay(self, frame, landmarks, ear_l, ear_r, brow, fps) -> None:
        """Annotate the preview frame. Only called when it will be shown."""
        import cv2

        h, w = frame.shape[:2]
        if landmarks is not None:
            ring, corners = visualisation_indices(self.s.dominant_eye)
            for idx in ring:
                p = landmarks[idx]
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, GREEN, -1)
            for idx in corners:
                p = landmarks[idx]
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, BLUE, -1)
            if self.gestures.left_closed:
                self._mark_eye(frame, landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM)
            if self.gestures.right_closed:
                self._mark_eye(frame, landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)
            if self.gestures.brow_raised:
                for idx in (LEFT_BROW_MID, RIGHT_BROW_MID):
                    p = landmarks[idx]
                    cv2.circle(frame, (int(p.x * w), int(p.y * h)), 5, YELLOW, -1)

        span = self.calibrator.span
        mc, beta = self.s.smoothing()
        baseline = self.gestures.brow_baseline
        ear_str = (
            f"L={ear_l:.2f} R={ear_r:.2f}"
            if ear_l is not None and ear_r is not None else "L=- R=-"
        )
        brow_str = (
            f"{brow:.3f}/{baseline:.3f}"
            if brow is not None and baseline is not None else "-"
        )
        # Precision readouts first — they are the numbers worth watching.
        # Jitter is how far the cursor wanders while the gaze is held, in
        # pixels; the "best" figure is the quietest fixation seen so far,
        # which is the noise floor this setup can reach.
        lang = self.s.language
        jitter = (
            f"{t(lang, 'hud.jitter')}: {self.jitter.rms:5.0f} px"
            + (f" ({t(lang, 'hud.best')} {self.jitter.best:.0f})"
               if self.jitter.best is not None else "")
            + (f"  [{t(lang, 'hud.holding')}]" if self.jitter.is_fixating else "")
        )
        quality = self.calibrator.quality
        if quality is not None and quality.valid:
            mean_px, worst_px = quality.in_pixels(self.screen_w, self.screen_h)
            calib = (f"{t(lang, 'hud.calib')}: {quality.grade(lang)} "
                     f"({t(lang, 'hud.calib_mean')} {mean_px:.0f} px, "
                     f"{t(lang, 'hud.calib_worst')} {worst_px:.0f} px)")
        else:
            calib = (f"{t(lang, 'hud.calib')}: "
                     f"{'ok' if self.calibrator.is_calibrated else t(lang, 'hud.calib_none')} (c)")
        on, off = t(lang, "hud.on"), t(lang, "hud.off")
        lines = [
            f"{fps:.0f} fps | {self.cam_w}x{self.cam_h} | "
            f"pointer: {getattr(self.pointer, 'name', '?')}",
            jitter,
            calib,
            f"{t(lang, 'hud.mouse')}: {on if self.mouse_on else off} (m)",
            f"{t(lang, 'hud.filter')}: cutoff={mc} beta={beta} (s)",
            f"{t(lang, 'hud.drift')}: dx={self.cursor.drift_dx:+.0f} "
            f"dy={self.cursor.drift_dy:+.0f} (a/A)",
            f"EAR {ear_str} | Brow {brow_str}",
            f"{t(lang, 'hud.eye')}: {self.s.dominant_eye}/{self.s.eye_reference}",
        ]
        self._draw_status_block(frame, lines)

    @staticmethod
    def _draw_status_block(frame, lines) -> None:
        """Draw the readout bottom-left: black on a solid yellow block.

        Nothing adaptive, because nothing needs to be. The camera sees
        whatever is behind the user — a wall, a window, a dark shirt can
        all pass under the same line within a second — and any text drawn
        straight onto that is readable only some of the time. Covering the
        area outright removes the problem, and that corner of the preview
        shows the user's shoulder, not anything worth seeing.

        Black on yellow is the pairing road signs and warning labels use:
        the highest practical contrast after black on white, and it stays
        legible at small sizes and off-angle.
        """
        import cv2

        h, w = frame.shape[:2]
        scale, step, pad = 0.5, 19, 8
        widths = [cv2.getTextSize(ascii_fold(line), cv2.FONT_HERSHEY_SIMPLEX,
                                  scale, 1)[0][0] for line in lines]
        block_w = min(max(widths) + pad * 2, w)
        block_h = step * len(lines) + pad
        top = max(0, h - block_h)
        cv2.rectangle(frame, (0, top), (block_w, h), YELLOW, -1)
        for i, line in enumerate(lines):
            y = top + pad + step * i + 9
            cv2.putText(frame, ascii_fold(line), (pad, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1,
                        cv2.LINE_AA)

    @staticmethod
    def _mark_eye(frame, landmarks, top_idx, bottom_idx) -> None:
        import cv2

        h, w = frame.shape[:2]
        p, q = landmarks[top_idx], landmarks[bottom_idx]
        cv2.circle(
            frame,
            (int((p.x + q.x) * 0.5 * w), int((p.y + q.y) * 0.5 * h)),
            12, RED, 2,
        )

    def _open_calibration_window(self) -> None:
        """Create the target window once, at the start of a run.

        Creating it and re-asserting full-screen on every target — which is
        what this used to do — restarts the window-server's full-screen
        transition each time, and the transition swallows the frame that
        was supposed to be painted. The window then stays blank, and since
        the preview has already been destroyed there is no OpenCV window
        left to receive keystrokes, so SPACE never reaches the app and
        calibration cannot be completed at all.
        """
        import cv2

        if self.s.calib_window == "fullscreen":
            cv2.namedWindow(TARGET_WINDOW, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(TARGET_WINDOW, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        else:
            # Escape hatch: a plain window keeps its title bar and does not
            # cover the menu bar, but it paints reliably everywhere.
            cv2.namedWindow(TARGET_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(TARGET_WINDOW, self.screen_w,
                             self.screen_h - self.s.macos_menu_bar_h)
        cv2.moveWindow(TARGET_WINDOW, 0, 0)
        self._calib_window_open = True

    def _close_calibration_window(self) -> None:
        import cv2

        try:
            cv2.destroyWindow(TARGET_WINDOW)
        except cv2.error:
            pass
        self._calib_window_open = False
        self._target_img = None
        self._last_calib_idx = -1

    def _update_calibration_window(self) -> None:
        import cv2

        if self.calibrator.active:
            if self.preview_visible:
                self._hide_preview_window()
            if not self._calib_window_open:
                self._open_calibration_window()
            ready = self.calibrator.ready_to_confirm
            if (self.calibrator.idx != self._last_calib_idx
                    or ready != self._last_calib_ready):
                tx, ty = self.calibrator.current_target
                self._target_img = make_target_image(
                    int(tx * self.screen_w), int(ty * self.screen_h),
                    self.screen_w, self.screen_h,
                    self.s.macos_menu_bar_h, self.s.target_size,
                    step=self.calibrator.idx,
                    total=self.calibrator.total_points,
                    ready=ready,
                    lang=self.s.language,
                )
                self._last_calib_ready = ready
                if self.calibrator.idx != self._last_calib_idx:
                    print(f"[calib] {self.calibrator.idx + 1}/"
                          f"{self.calibrator.total_points} — halkaya bak, "
                          f"bakışın oturunca SPACE")
                self._last_calib_idx = self.calibrator.idx
            # Repaint every frame. One imshow per target is not reliably
            # painted: the window needs the event loop to come round
            # several times before the first frame appears.
            if self._target_img is not None:
                cv2.imshow(TARGET_WINDOW, self._target_img)
        else:
            if self._calib_window_open:
                self._close_calibration_window()
            if not self.preview_visible and self.s.show_preview:
                self._show_preview_window()

    # --- input ---------------------------------------------------------

    def _handle_key(self, key: int) -> bool:
        """Process one keypress. Returns False to quit."""
        if key in (ord('q'), ord('Q')):
            return False
        if key == ord('c'):
            self._start_calibration()
        elif key == ord(' '):
            # SPACE outside a run means nothing; without this guard a stray
            # press would re-save the loaded calibration on every keystroke.
            self._confirm_calibration_point()
        elif key == 27:  # ESC
            if self.calibrator.active:
                self.calibrator.cancel()
                print("[calib] cancelled")
        elif key == ord('m'):
            self.mouse_on = not self.mouse_on
            print(f"[mouse] {'ON' if self.mouse_on else 'OFF'}")
        elif key == ord('s'):
            self.cursor.set_smoothing(*self.s.cycle_smoothing())
        elif key == ord('a'):
            if self.calibrator.is_calibrated:
                dx, dy = self.cursor.anchor()
                print(f"[anchor] drift set to dx={dx:+.0f} dy={dy:+.0f}")
        elif key == ord('A'):
            self.cursor.reset_drift()
            print("[anchor] drift reset")
        return True

    # --- the loop ------------------------------------------------------

    def run(self, max_frames: int | None = None, benchmark: bool = False) -> int:
        """Run the tracker.

        Args:
            max_frames: Stop after this many frames instead of waiting for
                'q'. Used by ``--benchmark`` and by smoke tests.
            benchmark: Collect per-stage timings and print them on exit.
        """
        import cv2
        import mediapipe as mp

        timer = StageTimer() if benchmark else None
        frames = 0
        detector = self._build_detector()
        capture = open_capture(
            self.s.camera_index, self.s.cam_width,
            self.s.cam_height, self.s.cam_fps,
        )
        try:
            self.cam_w, self.cam_h = negotiated_size(capture)
        except Exception:
            # A stand-in capture may not implement get(); keep the request.
            self.cam_w, self.cam_h = self.s.cam_width, self.s.cam_height
        source = (
            FrameGrabber(capture) if self.s.threaded_capture
            else DirectCapture(capture)
        )

        print(f"{LONG_NAME} v{__version__}")
        print(f"screen {self.screen_w}x{self.screen_h} | "
              f"pointer {getattr(self.pointer, 'name', '?')} | "
              f"eye {self.s.dominant_eye}/{self.s.eye_reference}")
        print(f"camera {self.cam_w}x{self.cam_h}"
              + (f" (istenen {self.s.cam_width}x{self.s.cam_height})"
                 if (self.cam_w, self.cam_h)
                 != (self.s.cam_width, self.s.cam_height) else ""))
        if self.cam_w < self.s.cam_width:
            print("UYARI: kamera istenen çözünürlüğü vermedi. Hassasiyet "
                  "doğrudan çözünürlüğe bağlı — --width/--height ile "
                  "kameranın desteklediği bir mod deneyin.")
        if self.calibrator.load(self.s.calibration_path):
            print(f"loaded calibration from {self.s.calibration_path}")
            self._report_quality()
        else:
            print("no usable calibration — press 'c' to calibrate")
        print("keys: c=calibrate  m=mouse  s=smoothing  a=anchor  "
              "A=reset drift  q=quit")
        print("keep the preview window SMALL so it never covers a target")

        t0 = time.monotonic()
        try:
            with source:
                if self.s.show_preview:
                    self._show_preview_window()
                while True:
                    t_read = time.perf_counter()
                    ok, frame = source.read()
                    if not ok:
                        if source.failed:
                            print("camera stopped returning frames")
                            break
                        continue
                    if timer:
                        timer.add("capture", time.perf_counter() - t_read)

                    now = time.monotonic()
                    self._frame_times.append(now)
                    frames += 1
                    t_stage = time.perf_counter()
                    frame = cv2.flip(frame, 1)
                    # Reuse one RGB buffer instead of allocating a new
                    # 640x480x3 array every frame.
                    if (self._rgb is None
                            or self._rgb.shape != frame.shape):
                        self._rgb = np.empty_like(frame)
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._rgb)

                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=self._rgb
                    )
                    if timer:
                        timer.add("convert", time.perf_counter() - t_stage)
                        t_stage = time.perf_counter()
                    result = detector.detect_for_video(
                        mp_image, self._next_timestamp_ms(now, t0)
                    )
                    if timer:
                        timer.add("inference", time.perf_counter() - t_stage)
                        t_stage = time.perf_counter()

                    landmarks = (
                        result.face_landmarks[0]
                        if result.face_landmarks else None
                    )
                    ear_l = ear_r = brow = None
                    gaze = None
                    if landmarks is not None:
                        gaze = normalized_iris_pair(
                            landmarks,
                            self.s.dominant_eye,
                            self.s.eye_reference,
                            self.s.max_eye_disagreement,
                        )
                        ear_l, ear_r = ear_pair(landmarks)
                        brow = brow_height(landmarks)

                    frozen = self.gestures.is_in_progress(ear_l, ear_r, brow)

                    if gaze is not None:
                        if self.calibrator.active:
                            self.calibrator.feed(*gaze)
                        else:
                            x, y = self.cursor.update(
                                gaze[0], gaze[1], now, frozen=frozen
                            )
                            self.jitter.update(*self.cursor.raw_prediction)
                            if self.mouse_on:
                                self.pointer.move_to(x, y)

                    if self.s.gestures_enabled:
                        can_click = (
                            self.mouse_on
                            and self.calibrator.is_calibrated
                            and not self.calibrator.active
                        )
                        for gesture in self.gestures.update(
                            ear_l, ear_r, brow, now, enabled=can_click
                        ):
                            self._dispatch(gesture)

                    self._face_seen = landmarks is not None
                    self._apply_panel_actions()
                    self._publish_state()
                    if self._quit:
                        break

                    if timer:
                        timer.add("track", time.perf_counter() - t_stage)
                        t_stage = time.perf_counter()

                    self._update_calibration_window()

                    if self.preview_visible:
                        shown = self._preview_frame(frame)
                        self._draw_overlay(
                            shown, landmarks, ear_l, ear_r, brow, self.fps
                        )
                        cv2.imshow(PREVIEW_WINDOW, shown)
                        if not self._preview_sized:
                            # resizeWindow before the first imshow does not
                            # stick; re-assert once the window exists.
                            cv2.resizeWindow(PREVIEW_WINDOW, shown.shape[1],
                                             shown.shape[0])
                            self._preview_sized = True
                    if timer:
                        timer.add("draw", time.perf_counter() - t_stage)

                    key = cv2.waitKey(1) & 0xFF
                    if key != 255 and not self._handle_key(key):
                        self._quit = True
                        break
                    if max_frames is not None and frames >= max_frames:
                        break
                    if not self._tracking:
                        # The panel asked for tracking to stop; the camera
                        # and the model are released on the way out and the
                        # panel keeps serving.
                        break
        finally:
            self._tracking = False
            self._hide_preview_window()
            self._close_calibration_window()
            # The landmarker holds native resources behind a serial
            # dispatcher. Left to the garbage collector, its finaliser can
            # deadlock waiting on an executor that is already winding down,
            # and the process hangs on exit instead of quitting. close() is
            # idempotent, so doing it here makes the finaliser a no-op.
            detector.close()
            cv2.destroyAllWindows()
            dropped = getattr(source, "frames_dropped", 0)
            print(f"{frames} frames at {self.fps:.1f} fps"
                  + (f", {dropped} stale frames dropped" if dropped else ""))
            if timer:
                print("per-stage cost:")
                for line in timer.report():
                    print(line)
        return 0

    # --- serving -------------------------------------------------------

    def serve(self, open_browser: bool = True) -> int:
        """Serve the panel, tracking only when it asks.

        The camera, the landmark model and the preview window are expensive
        and intrusive — the camera light comes on, macOS asks for
        permission, and a stale calibration can send the cursor flying. So
        the process starts idle: it serves the panel, and opens the camera
        only when the user presses start.

        It also means a failure has somewhere to go. Started from a
        terminal, a camera permission error printed a line and exited;
        here it lands on the page the user is actually looking at, and the
        process stays up so they can fix it and press start again.
        """
        if self.panel is None:
            print("error: panel kapalı; --no-panel ile serve modu çalışmaz",
                  file=sys.stderr)
            return 2
        if not self.panel.start():
            print("error: panel için boş port bulunamadı", file=sys.stderr)
            return 1

        self.calibrator.load(self.s.calibration_path)
        url = self.panel.url
        print(f"{LONG_NAME} v{__version__}")
        print(f"panel: {url}")
        print("takibi panelden başlat. Bu pencereyi kapatma.")
        if open_browser:
            import webbrowser

            webbrowser.open(url)

        try:
            while not self._quit:
                self._apply_panel_actions()
                if self._quit:
                    break
                if self._tracking:
                    self._track_once()
                else:
                    self._publish_state()
                    # Idle costs nothing: no camera, no model, no inference.
                    time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            self._tracking = False
            self.panel.stop()
        return 0

    def _track_once(self) -> None:
        """One tracking session, from camera open to camera closed."""
        self.last_error = None
        try:
            self.run()
        except FileNotFoundError as exc:
            self.last_error = str(exc).splitlines()[0]
        except RuntimeError as exc:
            self.last_error = str(exc)
        except Exception as exc:                  # pragma: no cover
            self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._tracking = False
            self.mouse_on = False
            if self.last_error:
                print(f"[hata] {self.last_error}", file=sys.stderr)
            self._publish_state()

    # --- control panel -------------------------------------------------

    def _publish_state(self) -> None:
        """Hand the panel a snapshot of everything it displays."""
        q = self.calibrator.quality
        quality = None
        if q is not None and q.valid:
            quality_grade = q.grade(self.s.language)
            mean_px, worst_px = q.in_pixels(self.screen_w, self.screen_h)
            quality = {
                "mean_px": mean_px, "worst_px": worst_px,
                "grade": quality_grade, "worst_index": q.worst_index,
                "points": len(q.per_point),
            }
        span = self.calibrator.span
        mc, beta = self.s.smoothing()
        self.bridge.publish({
            "running": True,
            "tracking": self._tracking,
            "error": self.last_error,
            "version": __version__,
            "mouse_on": self.mouse_on,
            "calibrating": self.calibrator.active,
            "calib_step": self.calibrator.idx,
            "calib_total": self.calibrator.total_points,
            "calibrated": self.calibrator.is_calibrated,
            "quality": quality,
            "span": ({"x": span.x, "y": span.y, "ok": span.ok,
                      "advice": span.advice(self.s.language)}
                     if span is not None else None),
            "fps": self.fps if self._tracking else 0.0,
            "jitter": {
                "rms": self.jitter.rms,
                "best": self.jitter.best,
                "fixating": self.jitter.is_fixating,
            },
            "camera": [self.cam_w, self.cam_h],
            "screen": [self.screen_w, self.screen_h],
            "pointer": getattr(self.pointer, "name", "?"),
            "eye": self.s.dominant_eye,
            "reference": self.s.eye_reference,
            "smoothing": {
                "index": self.s.smoothing_index, "cutoff": mc, "beta": beta,
                "total": len(self.s.smoothing_levels),
            },
            "deadzone": self.s.deadzone_px,
            "grid": self.s.calib_grid,
            "gestures": self.s.gestures_enabled,
            "lang": self.s.language,
            "drift": [self.cursor.drift_dx, self.cursor.drift_dy],
            "face": self._face_seen and self._tracking,
        })

    def _apply_panel_actions(self) -> None:
        """Run whatever the panel queued, on the loop's own thread.

        Doing this here rather than in the HTTP handler is what keeps the
        tracker single-threaded: OpenCV, MediaPipe and the filters are only
        ever touched from the capture loop.
        """
        for action, value in self.bridge.drain():
            if action == "start":
                self._tracking = True
                self.last_error = None
            elif action == "stop":
                self._tracking = False
            elif action == "mouse":
                self.mouse_on = bool(value)
            elif action == "calibrate":
                self._start_calibration()
            elif action == "confirm":
                self._confirm_calibration_point()
            elif action == "cancel":
                if self.calibrator.active:
                    self.calibrator.cancel()
            elif action == "anchor":
                if self.calibrator.is_calibrated:
                    self.cursor.anchor()
            elif action == "reset_drift":
                self.cursor.reset_drift()
            elif action == "smoothing":
                if 0 <= value < len(self.s.smoothing_levels):
                    self.s.smoothing_index = value
                    self.cursor.set_smoothing(*self.s.smoothing())
            elif action == "eye":
                if value in VALID_EYES:
                    self.s.dominant_eye = value
            elif action == "reference":
                if value in VALID_REFERENCES:
                    self.s.eye_reference = value
            elif action == "grid":
                if value in VALID_GRIDS:
                    self.s.calib_grid = value
                    # Takes effect on the next run; swapping the target list
                    # mid-calibration would strand the collected points.
                    if not self.calibrator.active:
                        self.calibrator = Calibrator(
                            min_samples=self.s.calib_min_samples,
                            sample_window=self.s.calib_sample_window,
                            clamp_margin=self.s.calib_clamp_margin,
                            grid=value,
                        )
                        self.calibrator.load(self.s.calibration_path)
                        self.cursor.calibrator = self.calibrator
            elif action == "deadzone":
                self.s.deadzone_px = max(0.0, float(value))
            elif action == "language":
                if value in LANGUAGES:
                    self.s.language = value
            elif action == "gestures":
                self.s.gestures_enabled = bool(value)
            elif action == "quit":
                self._quit = True

    def _start_calibration(self) -> None:
        self.calibrator.start()
        # A stale anchor from an earlier session would add a constant bias
        # on top of the new mapping.
        self.cursor.reset_drift()
        self.cursor.reset_filters()

    def _confirm_calibration_point(self) -> None:
        if not self.calibrator.active:
            return
        if not self.calibrator.confirm():
            print(f"[calib] hold still — only "
                  f"{self.calibrator.sample_count} samples so far "
                  f"(need {self.calibrator.min_samples})")
        elif self.calibrator.fit_error:
            print(f"[calib] FAILED: {self.calibrator.fit_error}")
            print("[calib] press 'c' and look at each bullseye squarely.")
        elif not self.calibrator.active:
            self.calibrator.save(
                self.s.calibration_path, (self.screen_w, self.screen_h)
            )
            print(f"[calib] done — saved to {self.s.calibration_path}")
            self._report_quality()

    def _report_quality(self) -> None:
        """Print the cross-validated accuracy of the current calibration.

        Leave-one-out error is the only accuracy figure available without
        asking the user to look at fresh targets, and it catches the common
        failure — a calibration spoiled by a blink or a head shift — which
        otherwise stays invisible until the cursor misbehaves.
        """
        q = self.calibrator.quality
        if q is None or not q.valid:
            return
        mean_px, worst_px = q.in_pixels(self.screen_w, self.screen_h)
        print(f"[calib] doğruluk: {q.grade()} — ortalama {mean_px:.0f} px, "
              f"en kötü {worst_px:.0f} px "
              f"(nokta {q.worst_index + 1}/{len(q.per_point)})")
        span = self.calibrator.span
        if span is not None:
            px_x, px_y = span.amplification(self.screen_w, self.screen_h)
            print(f"[calib] bakış aralığı: yatay {span.x:.3f} dikey "
                  f"{span.y:.3f} iris çapı "
                  f"(1 birim = {px_x:.0f}/{px_y:.0f} px)")
            if not span.ok:
                print("[calib] " + span.advice())
        if q.mean >= 0.08:
            print("[calib] bu kalibrasyon zayıf. Yeniden deneyin: her "
                  "hedefe tam bakın, kafanızı oynatmayın, bakış oturmadan "
                  "onaylamayın.")

    def _dispatch(self, gesture: Gesture) -> None:
        x, y = self.cursor.position
        if gesture is Gesture.LEFT_CLICK:
            self.pointer.click()
        elif gesture is Gesture.RIGHT_CLICK:
            self.pointer.right_click()
        elif gesture is Gesture.DOUBLE_CLICK:
            self.pointer.double_click()
        print(f"[gesture] {gesture.value} at ({x:.0f}, {y:.0f})")

    @property
    def fps(self) -> float:
        """Frame rate over the last second or so of frames."""
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        if span <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / span


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fatma",
        description=LONG_NAME,
    )
    p.add_argument("--version", action="version",
                   version=f"{LONG_NAME} {__version__}")
    p.add_argument("--eye", choices=VALID_EYES, default="auto",
                   help="which eye drives the cursor; 'auto' averages both "
                        "while they agree (default: auto)")
    p.add_argument("--reference", choices=VALID_REFERENCES, default="anchors",
                   help="landmarks used as the straight-ahead reference; "
                        "'anchors' averages six, 'corners' uses two "
                        "(default: anchors)")
    p.add_argument("--grid", type=int, choices=VALID_GRIDS, default=13,
                   help="number of calibration targets (default: 13)")
    p.add_argument("--calib-window", choices=VALID_CALIB_WINDOWS,
                   default="fullscreen",
                   help="calibration target window style; use 'normal' if "
                        "the full-screen target never appears "
                        "(default: fullscreen)")
    p.add_argument("--camera", type=int, default=0, metavar="N",
                   help="camera index (default: 0)")
    p.add_argument("--width", type=int, default=1280,
                   help="capture width; precision scales with it "
                        "(default: 1280)")
    p.add_argument("--height", type=int, default=720,
                   help="capture height (default: 720)")
    p.add_argument("--fps", type=int, default=30,
                   help="requested capture rate (default: 30)")
    p.add_argument("--pointer", choices=["auto", "quartz", "pyautogui", "null"],
                   default="auto",
                   help="mouse backend; 'null' moves nothing (default: auto)")
    p.add_argument("--dry-run", action="store_true",
                   help="track and preview without touching the mouse")
    p.add_argument("--mouse", action="store_true",
                   help="start with mouse control already on")
    p.add_argument("--no-preview", action="store_true",
                   help="run without the preview window")
    p.add_argument("--no-gestures", action="store_true",
                   help="disable wink clicks")
    p.add_argument("--brow", action="store_true",
                   help="enable the brow-raise double click; off by default "
                        "because looking up raises the brows on its own")
    p.add_argument("--no-threaded-capture", action="store_true",
                   help="read frames inline instead of in a capture thread")
    p.add_argument("--deadzone", type=float, default=8.0, metavar="PX",
                   help="ignore cursor moves below this size (default: 8)")
    p.add_argument("--smoothing", type=int, default=1, metavar="N",
                   help="smoothing preset 0-3, higher is smoother (default: 1)")
    p.add_argument("--calibration", metavar="PATH",
                   help="calibration file to load and save")
    p.add_argument("--model", metavar="PATH",
                   help="path to face_landmarker.task")
    p.add_argument("--no-panel", action="store_true",
                   help="do not start the browser control panel")
    p.add_argument("--panel-port", type=int, metavar="PORT",
                   help="port for the control panel (default: first free "
                        "from 8766)")
    p.add_argument("--lang", choices=LANGUAGES, default="en",
                   help="interface language for the overlay and the panel "
                        "(default: en)")
    p.add_argument("--track", action="store_true",
                   help="open the camera straight away instead of waiting "
                        "for the panel to ask")
    p.add_argument("--no-browser", action="store_true",
                   help="do not open the panel in a browser on startup")
    p.add_argument("--benchmark", type=int, nargs="?", const=200,
                   metavar="FRAMES",
                   help="run FRAMES frames (default 200), print per-stage "
                        "timings and exit")
    return p


def settings_from_args(args: argparse.Namespace) -> Settings:
    """Translate parsed CLI arguments into a :class:`Settings`."""
    kwargs = dict(
        camera_index=args.camera,
        cam_width=args.width,
        cam_height=args.height,
        cam_fps=args.fps,
        dominant_eye=args.eye,
        eye_reference=args.reference,
        calib_grid=args.grid,
        calib_window=args.calib_window,
        smoothing_index=args.smoothing,
        deadzone_px=args.deadzone,
        mouse_enabled_at_start=args.mouse,
        show_preview=not args.no_preview,
        gestures_enabled=not args.no_gestures,
        brow_enabled=args.brow,
        threaded_capture=not args.no_threaded_capture,
        panel_enabled=not args.no_panel,
        language=args.lang,
    )
    if args.panel_port:
        kwargs["panel_port"] = args.panel_port
    if args.calibration:
        kwargs["calibration_path"] = args.calibration
    if args.model:
        kwargs["model_path"] = args.model
    return Settings(**kwargs)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = settings_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.benchmark:
        settings.panel_enabled = False
    backend = "null" if args.dry_run or args.benchmark else args.pointer
    try:
        app = TrackerApp(settings, pointer_backend=backend)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Benchmarking and a panel-less run are one-shot: open the camera, do
    # the work, exit. Otherwise the process serves the panel and tracks
    # only when asked, so the user never has to touch a terminal again.
    direct = args.benchmark or not settings.panel_enabled
    try:
        if direct:
            app._tracking = True
            return app.run(max_frames=args.benchmark,
                           benchmark=bool(args.benchmark))
        if args.track:
            app._tracking = True
        return app.serve(open_browser=not args.no_browser)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
