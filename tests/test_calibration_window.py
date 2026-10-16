"""The calibration target window.

A regression suite for the bug that made calibration impossible: the
target image was pushed to the window only when the target index changed,
and the window was recreated and re-asserted full-screen each time. The
window-server's full-screen transition swallowed that single frame, so the
window stayed blank — and because the preview had already been destroyed,
no OpenCV window was left to receive keystrokes, so SPACE never reached the
application and the run could never be completed.

Two properties keep that from coming back: the window is created exactly
once per run, and the image is pushed on every frame.
"""

import numpy as np
import pytest

from fatma.config import Settings

cv2 = pytest.importorskip("cv2")

from fatma import app as app_module  # noqa: E402
from fatma.app import TrackerApp, make_target_image  # noqa: E402
from fatma.screen import Rect  # noqa: E402


@pytest.fixture
def app():
    try:
        return TrackerApp(Settings(show_preview=False), pointer_backend="null")
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"no display/pointer available: {exc}")


class FakeCv2:
    """Records the window calls the app makes, instead of drawing."""

    def __init__(self):
        self.created: list[str] = []
        self.shown: list[str] = []
        self.destroyed: list[str] = []
        self.fullscreen: list[str] = []
        self.error = cv2.error
        self.WINDOW_NORMAL = cv2.WINDOW_NORMAL
        self.WND_PROP_FULLSCREEN = cv2.WND_PROP_FULLSCREEN
        self.WINDOW_FULLSCREEN = cv2.WINDOW_FULLSCREEN

    def namedWindow(self, name, flags=0):  # noqa: N802
        self.created.append(name)

    def setWindowProperty(self, name, prop, value):  # noqa: N802
        self.fullscreen.append(name)

    def resizeWindow(self, name, w, h):  # noqa: N802
        pass

    def moveWindow(self, name, x, y):  # noqa: N802
        pass

    def imshow(self, name, image):
        self.shown.append(name)

    def destroyWindow(self, name):  # noqa: N802
        self.destroyed.append(name)

    def __getattr__(self, name):
        # Only the window calls are intercepted; drawing still has to work,
        # so everything else falls through to the real OpenCV.
        return getattr(cv2, name)


@pytest.fixture
def fake_cv2(monkeypatch):
    fake = FakeCv2()
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            return fake
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    return fake


class TestWindowLifecycle:
    def test_target_is_repainted_on_every_frame(self, app, fake_cv2):
        """The actual fix. One imshow per target is not reliably painted:
        the window needs the event loop to come round several times before
        anything appears on screen."""
        app.calibrator.start()
        app._update_calibration_window()          # opens and probes
        before = fake_cv2.shown.count(app_module.TARGET_WINDOW)
        for _ in range(5):
            app._update_calibration_window()
        after = fake_cv2.shown.count(app_module.TARGET_WINDOW)
        assert after - before == 5

    def test_window_is_created_once_per_run(self, app, fake_cv2):
        """Recreating it per target restarts the full-screen transition,
        which is what ate the frame."""
        app.calibrator.start()
        for _ in range(3):
            app._update_calibration_window()
        app.calibrator.idx = 1          # advance to the next target
        for _ in range(3):
            app._update_calibration_window()
        assert fake_cv2.created.count(app_module.TARGET_WINDOW) == 1
        # Full-screen is re-asserted while the window is appearing, because
        # the window server can drop the request mid-transition — which is
        # how a target window ends up wearing a title bar. What must not
        # repeat is the window's creation.
        before = fake_cv2.fullscreen.count(app_module.TARGET_WINDOW)
        app.calibrator.idx = 2
        for _ in range(3):
            app._update_calibration_window()
        assert fake_cv2.fullscreen.count(app_module.TARGET_WINDOW) == before

    def test_advancing_a_target_redraws_the_image(self, app, fake_cv2):
        app.calibrator.start()
        app._update_calibration_window()
        first = app._target_img
        app.calibrator.idx = 1
        app._update_calibration_window()
        assert app._target_img is not first

    def test_window_is_destroyed_when_the_run_ends(self, app, fake_cv2):
        app.calibrator.start()
        app._update_calibration_window()
        app.calibrator.cancel()
        app._update_calibration_window()
        assert app_module.TARGET_WINDOW in fake_cv2.destroyed
        assert app._calib_window_open is False
        assert app._target_img is None

    def test_a_second_run_opens_a_fresh_window(self, app, fake_cv2):
        for _ in range(2):
            app.calibrator.start()
            app._update_calibration_window()
            app.calibrator.cancel()
            app._update_calibration_window()
        assert fake_cv2.created.count(app_module.TARGET_WINDOW) == 2

    def test_normal_window_mode_skips_the_fullscreen_transition(self, fake_cv2):
        """Escape hatch for window managers where full-screen misbehaves."""
        try:
            app = TrackerApp(Settings(show_preview=False,
                                      calib_window="normal"),
                             pointer_backend="null")
        except Exception as exc:                  # pragma: no cover
            pytest.skip(f"no display/pointer available: {exc}")
        app.calibrator.start()
        app._update_calibration_window()
        assert fake_cv2.created == [app_module.TARGET_WINDOW]
        assert fake_cv2.fullscreen == []

    def test_nothing_is_drawn_when_not_calibrating(self, app, fake_cv2):
        app._update_calibration_window()
        assert fake_cv2.shown == []


class TestTargetImage:
    def test_progress_is_drawn_on_the_target(self):
        """The preview is hidden during calibration, so the canvas is the
        only place the user can see where they are."""
        plain = make_target_image(400, 300, Rect(0, 0, 800, 600), 60)[0]
        annotated = make_target_image(400, 300, Rect(0, 0, 800, 600), 60,
                                      step=2, total=13)[0]
        assert not np.array_equal(plain, annotated)
        # Text and progress dots add light pixels the plain canvas lacks.
        assert (annotated > 200).sum() > (plain > 200).sum()

    def test_progress_dots_track_the_step(self):
        early = make_target_image(400, 300, Rect(0, 0, 800, 600), 60, step=1, total=13)[0]
        late = make_target_image(400, 300, Rect(0, 0, 800, 600), 60, step=11, total=13)[0]
        band_early = early[50:70]
        band_late = late[50:70]
        yellow = lambda b: ((b[:, :, 1] > 200) & (b[:, :, 2] > 200)).sum()
        assert yellow(band_late) > yellow(band_early)

    def test_bullseye_is_still_placed_correctly_with_annotations(self):
        img, _ = make_target_image(400, 300, Rect(0, 0, 800, 600), 60, step=3, total=13)
        red = np.argwhere((img[:, :, 2] > 200) & (img[:, :, 1] < 60))
        cy, cx = red.mean(axis=0)
        assert (cx, cy) == pytest.approx((400, 300), abs=3)

    def test_annotations_do_not_overflow_the_canvas(self):
        img, _ = make_target_image(40, 40, Rect(0, 0, 800, 600), 60,
                                   step=0, total=13)
        assert img.shape == (600, 800, 3)
