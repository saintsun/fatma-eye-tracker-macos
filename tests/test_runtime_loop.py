"""End-to-end run of the capture loop against a synthetic camera.

This is the only test that exercises ``TrackerApp.run`` — MediaPipe, the
OpenCV conversions, the timestamp guard and the shutdown path all run for
real. It needs the landmark model on disk and takes a second or two, so it
is marked ``slow``; run the rest with ``-m 'not slow'``.

The synthetic frames contain no face, which is deliberate: the no-face
branch is the one a real session hits whenever the user looks away, and a
crash there would end the session.
"""

import os

import numpy as np
import pytest

from fatma.config import Settings

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

from fatma import app as app_module  # noqa: E402
from fatma.app import TrackerApp  # noqa: E402

pytestmark = pytest.mark.slow


class SyntheticCamera:
    """Emits deterministic frames as fast as it is asked."""

    def __init__(self, width=640, height=480, limit=10_000):
        self.frame = np.full((height, width, 3), 90, dtype=np.uint8)
        cv2.circle(self.frame, (width // 2, height // 2), 80, (160, 160, 160), -1)
        self.served = 0
        self.limit = limit
        self.released = False

    def read(self):
        if self.served >= self.limit:
            return False, None
        self.served += 1
        return True, self.frame.copy()

    def release(self):
        self.released = True

    def isOpened(self):  # noqa: N802
        return True


@pytest.fixture
def settings():
    s = Settings(show_preview=False)
    if not os.path.exists(s.model_path):
        pytest.skip(f"landmark model not present at {s.model_path}")
    return s


@pytest.fixture
def camera(monkeypatch):
    cam = SyntheticCamera()
    monkeypatch.setattr(app_module, "open_capture",
                        lambda *a, **k: cam)
    return cam


def build(settings):
    try:
        return TrackerApp(settings, pointer_backend="null")
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"no display/pointer available: {exc}")


class TestRuntimeLoop:
    def test_runs_and_shuts_down_cleanly(self, settings, camera):
        app = build(settings)
        assert app.run(max_frames=30) == 0
        assert camera.released
        assert camera.served >= 30

    def test_frameless_camera_ends_the_loop_instead_of_spinning(
        self, settings, monkeypatch
    ):
        cam = SyntheticCamera(limit=0)
        monkeypatch.setattr(app_module, "open_capture", lambda *a, **k: cam)
        app = build(settings)
        assert app.run(max_frames=100) == 0
        assert cam.released

    def test_a_face_free_frame_does_not_move_the_mouse(self, settings, camera):
        """No face means no gaze reading; the cursor must simply hold."""
        app = build(settings)
        app.mouse_on = True
        app.run(max_frames=20)
        assert app.pointer.moves == []

    def test_mediapipe_accepts_every_timestamp(self, settings, camera):
        """The guard against two frames sharing a millisecond. Without it
        MediaPipe raises once the loop gets fast enough, which is exactly
        what the pointer fix made it do."""
        app = build(settings)
        assert app.run(max_frames=60) == 0

    def test_benchmark_mode_reports_every_stage(self, settings, camera, capsys):
        app = build(settings)
        app.run(max_frames=20, benchmark=True)
        out = capsys.readouterr().out
        assert "per-stage cost" in out
        for stage in ("capture", "convert", "inference", "track"):
            assert stage in out

    def test_threaded_and_direct_capture_agree(self, settings, monkeypatch):
        results = {}
        for threaded in (True, False):
            cam = SyntheticCamera()
            monkeypatch.setattr(app_module, "open_capture",
                                lambda *a, **k: cam)
            s = Settings(show_preview=False, threaded_capture=threaded)
            app = build(s)
            results[threaded] = app.run(max_frames=15)
            assert cam.released
        assert results == {True: 0, False: 0}

    def test_a_missing_model_is_reported_clearly(self, camera, tmp_path):
        s = Settings(show_preview=False,
                     model_path=str(tmp_path / "absent.task"))
        app = build(s)
        with pytest.raises(FileNotFoundError, match="model missing"):
            app.run(max_frames=1)
