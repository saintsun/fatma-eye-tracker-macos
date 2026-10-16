"""Serve mode: the panel is up before the camera ever is.

Starting from a terminal meant the camera opened immediately, macOS asked
for permission at once, a stale calibration could fling the cursor, and any
failure printed a line to a window nobody was reading before the process
died. Serve mode inverts that: the process comes up idle with the panel
served, and the camera opens only when the page asks.
"""

import threading
import time

import pytest

from fatma.config import Settings

pytest.importorskip("cv2")

from fatma import app as app_module  # noqa: E402
from fatma.app import TrackerApp  # noqa: E402


def build(**overrides):
    settings = Settings(show_preview=False, **overrides)
    try:
        return TrackerApp(settings, pointer_backend="null")
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"no display/pointer available: {exc}")


@pytest.fixture
def app():
    return build()


def serve_in_thread(app):
    """Run serve() until the test asks it to quit."""
    result = {}

    def run():
        result["code"] = app.serve(open_browser=False)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    for _ in range(200):
        if app.panel.port:
            break
        time.sleep(0.01)
    return thread, result


def stop(app, thread):
    app.bridge.submit("quit")
    thread.join(timeout=5.0)


class TestIdleFirst:
    def test_the_camera_is_not_opened_until_asked(self, app, monkeypatch):
        """The whole point: the camera light does not come on, and macOS
        does not prompt, just because the panel was opened."""
        opened = []
        monkeypatch.setattr(app_module, "open_capture",
                            lambda *a, **k: opened.append(a))
        thread, result = serve_in_thread(app)
        try:
            time.sleep(0.2)
            assert opened == []
            assert app._tracking is False
        finally:
            stop(app, thread)
        assert result["code"] == 0

    def test_the_panel_is_serving_while_idle(self, app):
        import json
        import urllib.request

        thread, _ = serve_in_thread(app)
        try:
            with urllib.request.urlopen(app.panel.url + "/api/state",
                                        timeout=3) as response:
                state = json.loads(response.read())
            assert state["running"] is True
            assert state["tracking"] is False
        finally:
            stop(app, thread)

    def test_idle_state_does_not_claim_a_live_camera(self, app):
        app._tracking = False
        app._face_seen = True
        app._publish_state()
        state = app.bridge.snapshot()
        assert state["fps"] == 0.0
        assert state["face"] is False

    def test_a_calibration_is_loaded_before_any_tracking(self, app, tmp_path,
                                                         monkeypatch):
        loaded = []
        monkeypatch.setattr(app.calibrator, "load",
                            lambda path: loaded.append(path) or False)
        thread, _ = serve_in_thread(app)
        try:
            time.sleep(0.1)
            assert loaded == [app.s.calibration_path]
        finally:
            stop(app, thread)


class TestStartStop:
    def test_start_opens_the_camera(self, app, monkeypatch):
        calls = []

        def fake_run(*a, **k):
            calls.append("ran")
            app._tracking = False
            return 0

        monkeypatch.setattr(app, "run", fake_run)
        thread, _ = serve_in_thread(app)
        try:
            app.bridge.submit("start")
            for _ in range(200):
                if calls:
                    break
                time.sleep(0.01)
            assert calls == ["ran"]
        finally:
            stop(app, thread)

    def test_stop_ends_the_tracking_loop(self, app):
        """The loop has to notice the request between frames, since the
        camera and the model are released on the way out."""
        app._tracking = True
        app.bridge.submit("stop")
        app._apply_panel_actions()
        assert app._tracking is False

    def test_stopping_also_releases_the_mouse(self, app, monkeypatch):
        monkeypatch.setattr(app, "run", lambda *a, **k: 0)
        app.mouse_on = True
        app._tracking = True
        app._track_once()
        assert app.mouse_on is False
        assert app._tracking is False

    def test_quit_leaves_serve(self, app):
        thread, result = serve_in_thread(app)
        app.bridge.submit("quit")
        thread.join(timeout=5.0)
        assert not thread.is_alive()
        assert result["code"] == 0

    def test_the_panel_port_is_released_on_quit(self, app):
        from fatma.panel import _free_port

        thread, _ = serve_in_thread(app)
        port = app.panel.port
        stop(app, thread)
        assert _free_port((port,)) == port


class TestErrorsReachThePanel:
    def test_a_camera_failure_is_shown_not_fatal(self, app, monkeypatch):
        """A missing Camera permission is the common first run. It used to
        print one line and exit; the user is looking at the browser."""
        def boom(*a, **k):
            raise RuntimeError("cannot open camera 0")

        monkeypatch.setattr(app, "run", boom)
        app._tracking = True
        app._track_once()
        assert app._tracking is False
        assert "cannot open camera" in app.last_error
        app._publish_state()
        assert "cannot open camera" in app.bridge.snapshot()["error"]

    def test_a_missing_model_is_shown(self, app, monkeypatch):
        def boom(*a, **k):
            raise FileNotFoundError("landmark model missing: x.task\nmore")

        monkeypatch.setattr(app, "run", boom)
        app._tracking = True
        app._track_once()
        assert app.last_error == "landmark model missing: x.task"

    def test_an_unexpected_failure_is_still_caught(self, app, monkeypatch):
        """Serve mode must survive anything the tracking loop throws, or a
        user with no terminal has no way back."""
        monkeypatch.setattr(app, "run",
                            lambda *a, **k: (_ for _ in ()).throw(
                                ZeroDivisionError("boom")))
        app._tracking = True
        app._track_once()
        assert "ZeroDivisionError" in app.last_error

    def test_starting_again_clears_the_error(self, app):
        app.last_error = "eski hata"
        app.bridge.submit("start")
        app._apply_panel_actions()
        assert app.last_error is None
        assert app._tracking is True

    def test_serve_refuses_without_a_panel(self):
        app = build(panel_enabled=False)
        assert app.serve(open_browser=False) == 2


class TestCli:
    def test_serve_is_the_default_mode(self):
        from fatma.app import build_parser

        args = build_parser().parse_args([])
        assert args.track is False
        assert args.no_panel is False

    def test_track_starts_immediately(self):
        from fatma.app import build_parser

        assert build_parser().parse_args(["--track"]).track is True

    def test_browser_opening_can_be_suppressed(self):
        from fatma.app import build_parser

        assert build_parser().parse_args(["--no-browser"]).no_browser is True
