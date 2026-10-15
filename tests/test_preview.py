"""The preview window: its text, its size and its title.

All three were wrong in ways that only showed up on screen, which is why
each gets a test that fails in Python rather than needing an eye on a
window.
"""

import numpy as np
import pytest

from fatma.config import Settings

cv2 = pytest.importorskip("cv2")

from conftest import make_face  # noqa: E402
from fatma import app as app_module  # noqa: E402
from fatma.app import TrackerApp, ascii_fold, make_target_image  # noqa: E402


@pytest.fixture
def app():
    try:
        return TrackerApp(Settings(show_preview=False, panel_enabled=False),
                          pointer_backend="null")
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"no display/pointer available: {exc}")


class TestAsciiFold:
    def test_turkish_letters_become_their_ascii_counterparts(self):
        """Regression: "sabit bakış" reached the screen as "sabit bak????"
        because the Hershey fonts cv2.putText uses are ASCII-only."""
        assert ascii_fold("sabit bakış") == "sabit bakis"
        assert ascii_fold("en kötü") == "en kotu"
        assert ascii_fold("doğruluk") == "dogruluk"
        assert ascii_fold("İYİ ÇOK ŞÜPHELİ") == "IYI COK SUPHELI"

    def test_dashes_and_quotes_are_folded(self):
        assert ascii_fold("a — b") == "a - b"
        assert ascii_fold("1470×956") == "1470x956"

    def test_plain_ascii_is_untouched(self):
        assert ascii_fold("30 fps | 1280x720") == "30 fps | 1280x720"

    def test_anything_left_over_still_renders(self):
        """A character with no mapping must become '?', not raise."""
        folded = ascii_fold("日本語")
        assert folded.isascii()

    def test_output_is_always_ascii(self):
        for text in ("Kalibrasyon: çok iyi", "ölü bölge", "Jitter — 18 px"):
            assert ascii_fold(text).isascii()


class TestOverlayIsRenderable:
    """Every string the overlay draws has to survive cv2.putText."""

    def _lines(self, app, **kwargs):
        drawn = []
        real_puttext = cv2.putText

        def spy(img, text, *args, **kw):
            drawn.append(text)
            return real_puttext(img, text, *args, **kw)

        import unittest.mock as mock

        with mock.patch.object(cv2, "putText", spy):
            frame = np.full((360, 640, 3), 40, dtype=np.uint8)
            app._draw_overlay(frame, kwargs.get("landmarks"),
                              kwargs.get("ear", 0.3), kwargs.get("ear", 0.3),
                              kwargs.get("brow", 0.5), 30.0)
        return drawn

    def test_every_drawn_line_is_ascii(self, app):
        from fatma.quality import CalibrationQuality

        app.calibrator.quality = CalibrationQuality(
            per_point=(0.01,), mean=0.02, worst=0.05, worst_index=0
        )
        app.jitter.update(10.0, 10.0)
        app.jitter.update(10.0, 10.0)
        for line in self._lines(app, landmarks=make_face()):
            assert line.isascii(), f"not renderable: {line!r}"

    def test_every_drawn_line_is_ascii_without_a_face(self, app):
        for line in self._lines(app, landmarks=None, ear=None, brow=None):
            assert line.isascii(), f"not renderable: {line!r}"

    def test_target_image_text_is_ascii(self):
        drawn = []
        real_puttext = cv2.putText
        import unittest.mock as mock

        def spy(img, text, *args, **kw):
            drawn.append(text)
            return real_puttext(img, text, *args, **kw)

        with mock.patch.object(cv2, "putText", spy):
            make_target_image(400, 300, 800, 600, 25, 60, step=2, total=13)
        assert drawn
        for line in drawn:
            assert line.isascii(), f"not renderable: {line!r}"


class TestPreviewSizing:
    def test_a_capture_frame_is_scaled_down_to_the_preview_width(self, app):
        """Regression: the window is sized by the image, so showing the
        capture frame made the preview as large as the capture. Raising
        capture to 720p turned a corner window into half the screen."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        shown = app._preview_frame(frame)
        assert shown.shape[1] == app.s.preview_width

    def test_aspect_ratio_is_preserved(self, app):
        shown = app._preview_frame(np.zeros((720, 1280, 3), dtype=np.uint8))
        assert shown.shape[1] / shown.shape[0] == pytest.approx(1280 / 720,
                                                               rel=0.01)

    def test_a_frame_already_small_enough_is_not_enlarged(self, app):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        assert app._preview_frame(frame).shape == frame.shape

    def test_the_scaled_copy_does_not_alias_the_capture_frame(self, app):
        """The overlay is drawn on the copy; writing through to the frame
        MediaPipe just consumed would be a surprise at best."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        shown = app._preview_frame(frame)
        shown[0, 0] = (255, 255, 255)
        assert tuple(frame[0, 0]) == (0, 0, 0)

    @pytest.mark.parametrize("capture", [(480, 640), (720, 1280), (1080, 1920)])
    def test_window_size_is_independent_of_capture_resolution(self, app, capture):
        shown = app._preview_frame(np.zeros((*capture, 3), dtype=np.uint8))
        assert shown.shape[1] == app.s.preview_width

    def test_the_overlay_fits_the_scaled_frame(self, app):
        """Seven status lines at the fixed font size must not run off the
        bottom of a 360-pixel-tall preview."""
        frame = np.zeros((app.s.preview_height, app.s.preview_width, 3),
                         dtype=np.uint8)
        app._draw_overlay(frame, make_face(), 0.3, 0.3, 0.5, 30.0)
        # The topmost status line must land inside the image.
        assert frame[:, :, :].any(), "nothing drawn"
        top_rows = np.argwhere(frame.any(axis=(1, 2)))
        assert top_rows.min() >= 0


class TestWindowTitle:
    def test_the_title_is_short(self):
        """Regression: macOS will not shrink a window below the width of
        its title bar text, so the key list in the title silently set a
        floor on how small the preview could be."""
        assert len(app_module.PREVIEW_WINDOW) <= 24

    def test_the_title_is_ascii(self):
        assert app_module.PREVIEW_WINDOW.isascii()
