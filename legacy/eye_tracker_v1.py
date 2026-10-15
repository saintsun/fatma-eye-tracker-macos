"""
Eye-controlled mouse demo (MediaPipe Tasks API).

Uses the new MediaPipe FaceLandmarker (face_landmarker.task) which includes
iris landmarks. Tracks iris position inside the eye and maps it to screen
coordinates to move the cursor.

Controls (in the preview window):
  q  - quit
  c  - start/restart calibration (look at the 5 dots in order, press SPACE)
  SPACE - confirm a calibration point (during calibration)
  m  - toggle mouse control on/off
  s  - cycle smoothing strength
"""

import json
import math
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

pyautogui.FAILSAFE = False

SCREEN_W, SCREEN_H = pyautogui.size()
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "face_landmarker.task")
CALIB_FILE = os.path.join(HERE, "calibration.json")
PREVIEW_WINDOW = "Eye Tracker (q=quit, c=calibrate, m=mouse, s=smoothing)"
TARGET_WINDOW = "Calibration target"
TARGET_SIZE = 60

CAM_W, CAM_H = 640, 480  # lower res = much less CPU
# macOS reserves the top of the screen for the system menu bar even when
# an OpenCV window is "fullscreen", so the OpenCV canvas starts ~25 px
# below screen y=0. Bullseye drawing compensates with this offset.
MACOS_MENU_BAR_H = 25


class OneEuroFilter:
    """Adaptive low-pass: smooth when slow, snappy when moving fast."""

    def __init__(self, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(dt, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


def make_fullscreen_target_image(screen_cx, screen_cy):
    """Dark canvas sized to the visible OpenCV fullscreen area (below the
    macOS menu bar), with a bullseye centred at screen-coords (cx, cy).

    The OpenCV canvas top edge lives at screen y = MACOS_MENU_BAR_H, so we
    shift the bullseye up by that offset to make it land at the requested
    absolute screen position.
    """
    canvas_h = SCREEN_H - MACOS_MENU_BAR_H
    canvas_w = SCREEN_W
    img = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)
    cx = screen_cx
    cy = screen_cy - MACOS_MENU_BAR_H
    # Bullseye centre must remain visible — if a calibration point lands
    # inside the menu-bar region, clamp it into the canvas.
    cy = max(TARGET_SIZE // 2, min(cy, canvas_h - TARGET_SIZE // 2))
    r = TARGET_SIZE // 2
    cv2.circle(img, (cx, cy), max(r - 2, 6), (0, 255, 255), 2)
    cv2.circle(img, (cx, cy), max(r // 2, 4), (0, 165, 255), 1)
    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
    return img

# Landmark indices (MediaPipe FaceLandmarker, refined landmarks).
# Iris ring landmarks form a 4-point circle around each pupil. Averaging
# them gives a more stable iris CENTER than the single-point landmarks
# 468/473, and the ring diameter is an anatomical constant (~11.7 mm)
# which lets us normalise out the user's distance from the camera.
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_IRIS_RING = (469, 470, 471, 472)   # right, top, left, bottom (image axes)
RIGHT_IRIS_RING = (474, 475, 476, 477)
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# Eye-aspect-ratio (EAR) landmarks — upper / lower eyelid mid-points.
# EAR = vertical eyelid gap / horizontal eye width. Drops from ~0.30
# (open) to <0.15 (closed); we use this to detect winks → clicks.
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Eyebrow midpoints. Distance from brow to eye top, normalised by
# inter-iris distance, rises when both brows lift — used as a
# camera-distance-invariant trigger for double-click.
LEFT_BROW_MID = 105
RIGHT_BROW_MID = 334


def _eye_signal(landmarks, ring_idx, left_idx, right_idx):
    """Compute iris-center offset from eye-corner midpoint for one eye.

    Returns (nx, ny) where the unit is "iris diameters" — anatomical scale.
    Using the iris ring (not the single iris center landmark) averages out
    per-frame jitter. Using iris diameter (not eye width) as the scale gives
    the vertical channel comparable resolution to the horizontal channel,
    AND makes the signal invariant to camera distance.
    """
    ring = np.array([(landmarks[i].x, landmarks[i].y) for i in ring_idx])
    iris_xy = ring.mean(axis=0)                                 # robust center
    # Diameter from opposing ring points (horizontal + vertical axes averaged).
    diam = (np.linalg.norm(ring[0] - ring[2])
            + np.linalg.norm(ring[1] - ring[3])) * 0.5
    if diam < 1e-6:
        return None
    l = np.array([landmarks[left_idx].x, landmarks[left_idx].y])
    r = np.array([landmarks[right_idx].x, landmarks[right_idx].y])
    eye_mid = (l + r) * 0.5
    offset = (iris_xy - eye_mid) / diam                         # iris-diameter units
    return offset[0], offset[1]


# Which eye drives the cursor. Set to "right" to track only the user's
# right eye (helpful for users with mild strabismus or unequal gaze
# alignment, where binocular averaging would smear the signal). Use "left"
# for the user's left eye, or "both" for the previous binocular average.
DOMINANT_EYE = "right"


def normalized_iris_pair(landmarks):
    """Iris offset for the active eye(s), scaled by iris diameter.

    Returns (nx, ny) in iris-diameter units. With DOMINANT_EYE="both" this
    averages both eyes (cancels per-eye landmark noise but breaks down if
    the user has strabismus). With "left" or "right" it uses just that
    eye, which is preferable when the two eyes disagree.
    """
    if DOMINANT_EYE == "both":
        left = _eye_signal(landmarks, LEFT_IRIS_RING, LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
        right = _eye_signal(landmarks, RIGHT_IRIS_RING, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
        if left is None or right is None:
            return None
        return (left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5
    elif DOMINANT_EYE == "left":
        return _eye_signal(landmarks, LEFT_IRIS_RING, LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
    else:   # right (subject's perspective; MediaPipe indices 473/362/263)
        return _eye_signal(landmarks, RIGHT_IRIS_RING, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)


def _ear(landmarks, top_idx, bottom_idx, left_idx, right_idx):
    """Eye Aspect Ratio for one eye — vertical lid gap / horizontal width.

    Open eye → ~0.30; fully closed → <0.15. Camera-distance invariant
    because both the numerator and denominator scale together with face
    size. Used to detect winks for click gestures.
    """
    top = landmarks[top_idx]
    bot = landmarks[bottom_idx]
    l = landmarks[left_idx]
    r = landmarks[right_idx]
    vertical = ((top.x - bot.x) ** 2 + (top.y - bot.y) ** 2) ** 0.5
    horizontal = ((l.x - r.x) ** 2 + (l.y - r.y) ** 2) ** 0.5
    if horizontal < 1e-6:
        return None
    return vertical / horizontal


def _brow_height(landmarks):
    """Average brow-to-eye distance, normalised by inter-iris distance.

    Inter-iris distance is the standard anatomical scale reference —
    eyes don't move closer/farther on the face, so dividing by it
    cancels out camera distance. Returns None if landmarks invalid.
    """
    lb = landmarks[LEFT_BROW_MID]
    rb = landmarks[RIGHT_BROW_MID]
    le_top = landmarks[LEFT_EYE_TOP]
    re_top = landmarks[RIGHT_EYE_TOP]
    li = landmarks[LEFT_IRIS_CENTER]
    ri = landmarks[RIGHT_IRIS_CENTER]
    # Vertical gap in image coords (smaller y = higher on screen, so brow
    # above eye means eye_top.y > brow.y → positive gap).
    left_gap = le_top.y - lb.y
    right_gap = re_top.y - rb.y
    inter_iris = ((li.x - ri.x) ** 2 + (li.y - ri.y) ** 2) ** 0.5
    if inter_iris < 1e-6:
        return None
    return ((left_gap + right_gap) * 0.5) / inter_iris


class TemporalMedian:
    """Sliding-window median filter — kills isolated landmark outliers
    (single-frame jumps) without adding much latency. Applied to the raw
    iris signal BEFORE One-Euro smoothing, so One-Euro's adaptive cutoff
    isn't disturbed by spikes."""
    def __init__(self, window=5):
        self.window = window
        self.values = []

    def update(self, x):
        self.values.append(x)
        if len(self.values) > self.window:
            self.values.pop(0)
        return float(np.median(self.values))

    def reset(self):
        self.values.clear()


class TPSMapping:
    """Thin Plate Spline interpolation in 2D.

    Builds a smooth surface that passes exactly through every control point
    and is C² continuous between them. Solved per output axis via the
    standard linear system (no scipy dependency, just numpy).

    Reference: Bookstein 1989. The interpolant is
        f(p) = a₀ + a₁·x + a₂·y + Σᵢ wᵢ · φ(‖p − pᵢ‖)
    where φ(r) = r² · log(r) is the thin-plate kernel.
    """

    @staticmethod
    def _phi(r):
        out = np.zeros_like(r)
        mask = r > 1e-12
        out[mask] = (r[mask] ** 2) * np.log(r[mask])
        return out

    @classmethod
    def _kernel(cls, A, B):
        # A: (M, 2), B: (N, 2) → (M, N) of pairwise φ(‖a − b‖)
        diff = A[:, None, :] - B[None, :, :]
        r = np.sqrt((diff ** 2).sum(axis=2))
        return cls._phi(r)

    def __init__(self, src=None, tx=None, ty=None):
        if src is None:
            return
        src = np.asarray(src, dtype=float)
        n = len(src)
        K = self._kernel(src, src)                    # (n, n)
        P = np.hstack([np.ones((n, 1)), src])         # (n, 3): [1, x, y]
        L = np.zeros((n + 3, n + 3))
        L[:n, :n] = K
        L[:n, n:] = P
        L[n:, :n] = P.T
        # Right-hand sides for x and y outputs (zero-padded at the constraint rows).
        Yx = np.concatenate([np.asarray(tx, dtype=float), np.zeros(3)])
        Yy = np.concatenate([np.asarray(ty, dtype=float), np.zeros(3)])
        # Tiny ridge regularisation for numerical stability on near-duplicate src points.
        L += np.eye(n + 3) * 1e-9
        self.src = src
        self.cx = np.linalg.solve(L, Yx)
        self.cy = np.linalg.solve(L, Yy)

    def __call__(self, x, y):
        pt = np.array([[x, y]], dtype=float)
        kr = self._kernel(pt, self.src)[0]            # (n,)
        feats = np.concatenate([kr, [1.0, x, y]])
        return float(feats @ self.cx), float(feats @ self.cy)

    def to_dict(self):
        return {
            "kind": "tps",
            "src": self.src.tolist(),
            "cx": self.cx.tolist(),
            "cy": self.cy.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.src = np.array(d["src"], dtype=float)
        obj.cx = np.array(d["cx"], dtype=float)
        obj.cy = np.array(d["cy"], dtype=float)
        return obj


class Calibrator:
    """9-point calibration with Thin Plate Spline mapping.

    TPS goes through every calibration point exactly and stays smooth
    between them — better than affine + bilinear at corners and especially
    important for accessibility (ALS) use where the user can't physically
    re-anchor between sessions and needs predictable behaviour everywhere
    on the screen.
    """

    # Calibration grid stretched closer to the screen edges so the
    # reachable cursor area covers the menu bar / dock regions too.
    POINTS_SCREEN = [
        (0.03, 0.03), (0.5, 0.03), (0.97, 0.03),
        (0.03, 0.5),  (0.5, 0.5),  (0.97, 0.5),
        (0.03, 0.97), (0.5, 0.97), (0.97, 0.97),
    ]

    def __init__(self):
        self.points_eye = []
        self.idx = 0
        self.active = False
        self.tps = None
        self.samples = []
        self.nx_min = self.nx_max = 0.0
        self.ny_min = self.ny_max = 0.0

    @property
    def is_calibrated(self):
        return self.tps is not None

    def start(self):
        self.points_eye.clear()
        self.samples.clear()
        self.idx = 0
        self.active = True
        self.tps = None

    def feed(self, nx, ny):
        if self.active:
            self.samples.append((nx, ny))

    def confirm(self):
        if not self.active or not self.samples:
            return
        # Median over the last ~0.5 s of samples — rejects gaze-transition
        # frames (when the user was settling onto the target). Mean would
        # bias the calibration point toward the transition path.
        arr = np.array(self.samples[-15:])
        self.points_eye.append(np.median(arr, axis=0))
        self.samples.clear()
        self.idx += 1
        if self.idx >= len(self.POINTS_SCREEN):
            self._fit()
            self.active = False
            self.save()

    def _fit(self):
        eye = np.array(self.points_eye)
        scr = np.array(self.POINTS_SCREEN)
        self.tps = TPSMapping(eye, scr[:, 0], scr[:, 1])
        # Remember calibration input range so we can clamp inputs — TPS
        # extrapolates smoothly but unbounded outside the convex hull, so
        # clamping prevents wild jumps when the user briefly looks away
        # from the screen or blinks.
        self.nx_min, self.ny_min = eye.min(axis=0)
        self.nx_max, self.ny_max = eye.max(axis=0)
        margin_x = (self.nx_max - self.nx_min) * 0.05
        margin_y = (self.ny_max - self.ny_min) * 0.05
        self.nx_min -= margin_x
        self.nx_max += margin_x
        self.ny_min -= margin_y
        self.ny_max += margin_y

    def map_to_screen(self, nx, ny):
        if self.tps is None:
            sx = (nx + 0.2) / 0.4
            sy = (ny + 0.2) / 0.4
        else:
            # TPS used to extrapolate beyond the convex hull — without
            # input clamping, looking past a calibration corner lets the
            # cursor reach the actual screen edge. We protect against
            # blow-up by output-clamping after the prediction.
            sx, sy = self.tps(nx, ny)
        sx = float(np.clip(sx, 0.0, 1.0))
        sy = float(np.clip(sy, 0.0, 1.0))
        return sx * SCREEN_W, sy * SCREEN_H

    def save(self):
        if self.tps is None:
            return
        with open(CALIB_FILE, "w") as f:
            json.dump({
                "kind": "tps",
                "tps": self.tps.to_dict(),
                "nx_range": [self.nx_min, self.nx_max],
                "ny_range": [self.ny_min, self.ny_max],
                "screen": [SCREEN_W, SCREEN_H],
            }, f)
        print(f"Calibration saved to {CALIB_FILE}")

    def load(self):
        if not os.path.exists(CALIB_FILE):
            return False
        try:
            with open(CALIB_FILE) as f:
                data = json.load(f)
            if data.get("kind") != "tps" or "tps" not in data:
                print("Old calibration format detected — please re-calibrate (press 'c').")
                return False
            self.tps = TPSMapping.from_dict(data["tps"])
            self.nx_min, self.nx_max = data["nx_range"]
            self.ny_min, self.ny_max = data["ny_range"]
            print(f"Loaded calibration from {CALIB_FILE}")
            return True
        except Exception as e:
            print(f"Could not load calibration: {e}")
            return False


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model file missing: {MODEL_PATH}")
        print("Download it with:")
        print("  curl -fL -o face_landmarker.task https://storage.googleapis.com/"
              "mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Grant Camera permission to your terminal in"
              " System Settings -> Privacy & Security -> Camera.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    detector = mp_vision.FaceLandmarker.create_from_options(options)

    calibrator = Calibrator()
    calibrator.load()  # auto-restore calibration if file exists

    # One-Euro filter levels: (min_cutoff, beta).
    #   min_cutoff = smoothing applied when the gaze is still
    #                (lower = smoother fixation, less jitter)
    #   beta       = how fast the filter relaxes during motion
    #                (higher = snappier on big jumps / saccades)
    # The combo gives "smooth when slow, fast when moving fast" — desired
    # behaviour for cursor control: stable on small targets, no lag during
    # screen-spanning saccades.
    smoothing_levels = [
        (1.5, 0.80),   # snappy   — minimal lag, some jitter at rest
        (0.6, 0.50),   # default  — smooth fixation, fast saccades
        (0.3, 0.30),   # smooth   — heavy idle smoothing, still responsive
        (0.15, 0.15),  # max      — for users with limited motion
    ]
    smoothing_idx = 1
    # Vertical channel has ~half the dynamic range of horizontal even with
    # iris-diameter scaling, so it benefits from a slightly heavier filter
    # (lower cutoff) to keep noise from dominating the small Y signal.
    def make_filters(mc, beta):
        return (OneEuroFilter(mc, beta),
                OneEuroFilter(mc * 0.6, beta * 0.8))
    filter_x, filter_y = make_filters(*smoothing_levels[smoothing_idx])
    mouse_on = False
    last_calib_idx = -1
    # Screen-space drift correction set by the 'a' key. When the user looks
    # at the screen center and presses 'a' we record how far the predicted
    # cursor is from (W/2, H/2) and add that delta to every future output.
    # This fixes slow drift from head pose changes without forcing a full
    # 9-point recalibration.
    drift_dx = 0.0
    drift_dy = 0.0
    last_predicted_sx = SCREEN_W * 0.5
    last_predicted_sy = SCREEN_H * 0.5

    # --- Stability layer ---------------------------------------------------
    # Two filters stacked: temporal median absorbs single-frame landmark
    # spikes; the screen-space "snap" lerp below kills sub-threshold breath
    # tremor while still tracking real saccades instantly.
    median_x = TemporalMedian(window=5)
    median_y = TemporalMedian(window=5)
    last_cursor_sx = SCREEN_W * 0.5
    last_cursor_sy = SCREEN_H * 0.5
    # Pure deadzone: ignore movements below this. One-Euro already does
    # velocity-adaptive smoothing (smooth at rest, snappy on saccades) so
    # we don't add another smoothing layer here — only a tremor killer.
    # Tuned big enough to absorb laptop-on-lap breath shake without
    # noticeably suppressing real micro-adjustments.
    DEADZONE_PX = 8

    # --- Gesture detection -------------------------------------------------
    # Eye-closure click: EAR has to be below CLOSED for HOLD seconds, with
    # COOLDOWN before the next click can fire.
    EAR_CLOSED = 0.16
    EAR_OPEN = 0.22                    # hysteresis — must reopen above this
    CLOSED_HOLD_S = 0.20
    BROW_RAISE_RATIO = 1.15            # current / baseline → raised
    BROW_HOLD_S = 0.20
    CLICK_COOLDOWN_S = 0.45            # global gesture cooldown
    left_closed_since = None
    right_closed_since = None
    left_is_closed = False             # latched (hysteresis state)
    right_is_closed = False
    brow_raised_since = None
    brow_is_raised = False
    # Slow EMA of brow distance gives a self-adapting "neutral" baseline.
    # Time constant ~30 s (alpha ≈ 0.001 at 30 fps); short gesture pulses
    # don't shift it, but slow posture changes do.
    brow_baseline = None
    BROW_EMA_ALPHA = 0.001
    last_click_at = 0.0

    def apply_anchor(reason="manual"):
        nonlocal drift_dx, drift_dy
        if not calibrator.is_calibrated:
            return
        drift_dx = SCREEN_W * 0.5 - (last_predicted_sx - drift_dx)
        drift_dy = SCREEN_H * 0.5 - (last_predicted_sy - drift_dy)
        print(f"[anchor:{reason}] drift set to dx={drift_dx:+.0f}px dy={drift_dy:+.0f}px")

    print("Started. Press 'c' to calibrate, 'a' to anchor drift,"
          " 'm' to toggle mouse, 'q' to quit.")
    print("IMPORTANT: keep this preview window SMALL — do NOT make it fullscreen,")
    print("otherwise the mouse cursor (the calibration target) will be hidden behind it.")

    # Preview window — small, top-right of the screen (out of the way of the
    # 9-point calibration grid which uses the four corners + edges).
    PREV_W, PREV_H = 480, 320

    def show_preview_window():
        cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(PREVIEW_WINDOW, PREV_W, PREV_H)
        cv2.moveWindow(PREVIEW_WINDOW,
                       max(0, SCREEN_W - PREV_W - 20),
                       max(0, 40))

    show_preview_window()
    preview_visible = True

    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - t0) * 1000)
        result = detector.detect_for_video(mp_image, ts_ms)

        nx_s = ny_s = None
        ear_left = ear_right = brow_h = None
        now = time.time()
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            pair = normalized_iris_pair(lm)
            if pair is not None:
                # Temporal median first: removes single-frame landmark
                # outliers without adding noticeable lag.
                nx_med = median_x.update(pair[0])
                ny_med = median_y.update(pair[1])
                nx_s = filter_x(nx_med, now)
                ny_s = filter_y(ny_med, now)

            # Gesture features.
            ear_left = _ear(lm, LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                            LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
            ear_right = _ear(lm, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM,
                             RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
            brow_h = _brow_height(lm)
            if brow_h is not None:
                if brow_baseline is None:
                    brow_baseline = brow_h
                else:
                    # Only update baseline when brows are clearly NOT raised
                    # (current ≤ baseline * 1.10). Holding a raise wouldn't
                    # poison the baseline.
                    if brow_h <= brow_baseline * 1.10:
                        brow_baseline = ((1 - BROW_EMA_ALPHA) * brow_baseline
                                         + BROW_EMA_ALPHA * brow_h)

            # Visualise.
            if DOMINANT_EYE == "left":
                ring_pts = LEFT_IRIS_RING
                corner_pts = (LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
            elif DOMINANT_EYE == "right":
                ring_pts = RIGHT_IRIS_RING
                corner_pts = (RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
            else:
                ring_pts = LEFT_IRIS_RING + RIGHT_IRIS_RING
                corner_pts = (LEFT_EYE_LEFT, LEFT_EYE_RIGHT,
                              RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
            for idx in ring_pts:
                p = lm[idx]
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, (0, 255, 0), -1)
            for idx in corner_pts:
                p = lm[idx]
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, (255, 0, 0), -1)
            # Closed-eye indicator: red ring around the eye that's closed.
            if left_is_closed:
                p = lm[LEFT_EYE_TOP]; q = lm[LEFT_EYE_BOTTOM]
                cv2.circle(frame, (int((p.x + q.x) * 0.5 * w),
                                   int((p.y + q.y) * 0.5 * h)),
                           12, (0, 0, 255), 2)
            if right_is_closed:
                p = lm[RIGHT_EYE_TOP]; q = lm[RIGHT_EYE_BOTTOM]
                cv2.circle(frame, (int((p.x + q.x) * 0.5 * w),
                                   int((p.y + q.y) * 0.5 * h)),
                           12, (0, 0, 255), 2)
            # Brow indicator: yellow on raised.
            if brow_is_raised:
                for idx in (LEFT_BROW_MID, RIGHT_BROW_MID):
                    p = lm[idx]
                    cv2.circle(frame, (int(p.x * w), int(p.y * h)),
                               5, (0, 255, 255), -1)

        # Freeze cursor during any gesture in progress:
        #   - Either eye closing past the open threshold disturbs iris
        #     landmarks (eyelid covers/distorts the iris ring), making
        #     nx/ny garbage. The cursor would otherwise jerk around.
        #   - A brow raise can shift the whole face slightly; same idea.
        # The click then fires at the LAST stable cursor position — which
        # is what the user intended (click where I was just looking).
        gesture_in_progress = False
        if ear_left is not None and ear_left < EAR_OPEN:
            gesture_in_progress = True
        if ear_right is not None and ear_right < EAR_OPEN:
            gesture_in_progress = True
        if (brow_h is not None and brow_baseline is not None
                and brow_h > brow_baseline * 1.10):
            gesture_in_progress = True

        if nx_s is not None:
            if calibrator.active:
                calibrator.feed(nx_s, ny_s)
            else:
                sx, sy = calibrator.map_to_screen(nx_s, ny_s)
                # Apply drift correction (set by the 'a' anchor key)
                sx = float(np.clip(sx + drift_dx, 0.0, SCREEN_W - 1))
                sy = float(np.clip(sy + drift_dy, 0.0, SCREEN_H - 1))
                last_predicted_sx, last_predicted_sy = sx, sy
                # Simple deadzone: skip updates smaller than a few pixels
                # (kills breathing/landmark tremor). Real saccades cross
                # the threshold immediately and follow at full speed.
                # During a gesture, also skip — landmark noise is huge.
                if not gesture_in_progress:
                    dx = sx - last_cursor_sx
                    dy = sy - last_cursor_sy
                    if (dx * dx + dy * dy) > DEADZONE_PX * DEADZONE_PX:
                        last_cursor_sx = sx
                        last_cursor_sy = sy
                if mouse_on:
                    pyautogui.moveTo(last_cursor_sx, last_cursor_sy, duration=0.0)

        # --- Click gestures -------------------------------------------
        # State machine per eye with hysteresis (separate thresholds for
        # close vs. open) avoids flicker. A click fires when the eye has
        # been continuously closed for CLOSED_HOLD_S and the global
        # cooldown has elapsed.
        in_cooldown = now - last_click_at < CLICK_COOLDOWN_S

        def update_eye(ear, is_closed, closed_since, click_fn, label):
            """Returns (new_is_closed, new_closed_since). Fires click as a
            side-effect when the hold threshold + cooldown are satisfied."""
            nonlocal last_click_at
            if ear is None:
                return is_closed, closed_since
            if not is_closed and ear < EAR_CLOSED:
                is_closed = True
                closed_since = now
            elif is_closed and ear > EAR_OPEN:
                is_closed = False
                closed_since = None
            if (is_closed and closed_since is not None
                    and now - closed_since >= CLOSED_HOLD_S
                    and not in_cooldown
                    and mouse_on and calibrator.is_calibrated):
                click_fn()
                last_click_at = now
                print(f"[gesture] {label} click at "
                      f"({last_cursor_sx:.0f}, {last_cursor_sy:.0f})")
                closed_since = None  # require re-open before next fire
            return is_closed, closed_since

        # Run state updates. A natural blink closes BOTH eyes → no click
        # fires only if the OTHER eye stays open (the wink test). The
        # state machine above treats each eye independently, but since
        # the user can't keep the dominant eye open while closing it,
        # we additionally require the OPPOSITE eye to stay open. This is
        # implicit: closing both eyes for >CLOSED_HOLD_S would still fire
        # both clicks — to prevent that, gate on "other eye open".
        if ear_left is not None and ear_right is not None:
            other_open_for_left = ear_right > EAR_OPEN
            other_open_for_right = ear_left > EAR_OPEN
            if other_open_for_left:
                left_is_closed, left_closed_since = update_eye(
                    ear_left, left_is_closed, left_closed_since,
                    pyautogui.click, "LEFT")
            else:
                # If both eyes are closed, treat as a blink, not a wink.
                left_is_closed = ear_left < EAR_CLOSED
                left_closed_since = None
            if other_open_for_right:
                right_is_closed, right_closed_since = update_eye(
                    ear_right, right_is_closed, right_closed_since,
                    pyautogui.rightClick, "RIGHT")
            else:
                right_is_closed = ear_right < EAR_CLOSED
                right_closed_since = None

        # Brow-raise → double-click
        if brow_h is not None and brow_baseline is not None:
            raised_now = brow_h > brow_baseline * BROW_RAISE_RATIO
            if raised_now and not brow_is_raised:
                brow_is_raised = True
                brow_raised_since = now
            elif not raised_now and brow_is_raised:
                brow_is_raised = False
                brow_raised_since = None
            if (brow_is_raised and brow_raised_since is not None
                    and now - brow_raised_since >= BROW_HOLD_S
                    and not in_cooldown
                    and mouse_on and calibrator.is_calibrated):
                pyautogui.doubleClick()
                last_click_at = now
                print(f"[gesture] DOUBLE-CLICK at "
                      f"({last_cursor_sx:.0f}, {last_cursor_sy:.0f})")
                brow_raised_since = None

        # During calibration: hide preview, show only a bullseye at each
        # target screen position. This avoids the preview window covering
        # any of the 9 calibration corners.
        if calibrator.active:
            if preview_visible:
                try:
                    cv2.destroyWindow(PREVIEW_WINDOW)
                except cv2.error:
                    pass
                preview_visible = False

            tx, ty = calibrator.POINTS_SCREEN[calibrator.idx]
            target_x_px = int(tx * SCREEN_W)
            target_y_px = int(ty * SCREEN_H)
            if calibrator.idx != last_calib_idx:
                cv2.namedWindow(TARGET_WINDOW, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(TARGET_WINDOW,
                                      cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
                cv2.imshow(TARGET_WINDOW,
                           make_fullscreen_target_image(target_x_px, target_y_px))
                cv2.moveWindow(TARGET_WINDOW, 0, 0)
                total = len(calibrator.POINTS_SCREEN)
                print(f"[calib] {calibrator.idx + 1}/{total} - look at the bullseye, press SPACE")
                last_calib_idx = calibrator.idx
        else:
            if last_calib_idx != -1:
                try:
                    cv2.destroyWindow(TARGET_WINDOW)
                except cv2.error:
                    pass
            last_calib_idx = -1
            if not preview_visible:
                show_preview_window()
                preview_visible = True

        mc, beta = smoothing_levels[smoothing_idx]
        ear_str = (f"L={ear_left:.2f} R={ear_right:.2f}"
                   if ear_left is not None and ear_right is not None else "L=- R=-")
        brow_str = (f"{brow_h:.3f}/{brow_baseline:.3f}"
                    if brow_h is not None and brow_baseline is not None else "-")
        status = [
            f"Mouse: {'ON' if mouse_on else 'OFF'} (m)",
            f"Filter: cutoff={mc} beta={beta} (s)",
            f"Calibrated: {'YES' if calibrator.is_calibrated else 'NO'} (c)",
            f"Drift: dx={drift_dx:+.0f} dy={drift_dy:+.0f} (a=anchor, A=reset)",
            f"EAR {ear_str} | Brow {brow_str}",
            f"Eye: {DOMINANT_EYE}  (wink L/R = click; brow up = dblclick)",
        ]
        for i, line in enumerate(status):
            cv2.putText(frame, line, (10, h - 10 - 22 * (len(status) - 1 - i)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if preview_visible:
            cv2.imshow(PREVIEW_WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            calibrator.start()
            # Fresh calibration → fresh drift. A stale anchor offset from
            # an earlier session would otherwise add a constant bias on
            # top of the new mapping.
            drift_dx = drift_dy = 0.0
            median_x.reset()
            median_y.reset()
        elif key == ord(' '):
            calibrator.confirm()
        elif key == 27:  # ESC -> cancel calibration
            if calibrator.active:
                calibrator.active = False
                calibrator.points_eye.clear()
                calibrator.samples.clear()
                calibrator.idx = 0
        elif key == ord('m'):
            mouse_on = not mouse_on
        elif key == ord('s'):
            smoothing_idx = (smoothing_idx + 1) % len(smoothing_levels)
            filter_x, filter_y = make_filters(*smoothing_levels[smoothing_idx])
        elif key == ord('a'):
            # Anchor: assume the user is looking at the screen center right
            # now; the offset between that and the model's prediction is
            # the accumulated drift. Recording it cancels the drift for all
            # subsequent frames until the user re-anchors or resets.
            apply_anchor(reason="manual")
        elif key == ord('A'):
            drift_dx = drift_dy = 0.0
            print("[anchor] drift reset")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
