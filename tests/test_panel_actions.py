"""Panel actions applied to a real tracker.

The bridge only proves a request was queued. These tests take the other
half: that draining a queued action actually changes the tracker, and that
the snapshot the panel renders matches what the tracker believes.
"""

import pytest

from fatma.config import Settings

pytest.importorskip("cv2")

from fatma.app import TrackerApp  # noqa: E402


@pytest.fixture
def app():
    try:
        return TrackerApp(Settings(show_preview=False, panel_enabled=False),
                          pointer_backend="null")
    except Exception as exc:                      # pragma: no cover
        pytest.skip(f"no display/pointer available: {exc}")


def send(app, action, value=None):
    assert app.bridge.submit(action, value)[0], f"{action} was refused"
    app._apply_panel_actions()


class TestActionsChangeTheTracker:
    def test_mouse_toggles(self, app):
        send(app, "mouse", True)
        assert app.mouse_on is True
        send(app, "mouse", False)
        assert app.mouse_on is False

    def test_calibrate_starts_a_run_and_clears_drift(self, app):
        app.cursor.drift_dx = 99.0
        send(app, "calibrate")
        assert app.calibrator.active
        assert app.cursor.drift_dx == 0.0

    def test_cancel_ends_the_run(self, app):
        send(app, "calibrate")
        send(app, "cancel")
        assert not app.calibrator.active

    def test_confirm_advances_a_point(self, app):
        send(app, "calibrate")
        for _ in range(app.calibrator.min_samples):
            app.calibrator.feed(0.1, -0.2)
        send(app, "confirm")
        assert app.calibrator.idx == 1

    def test_confirm_outside_a_run_does_nothing(self, app):
        send(app, "confirm")
        assert app.calibrator.idx == 0

    def test_smoothing_switches_the_filters(self, app):
        send(app, "smoothing", 3)
        assert app.s.smoothing_index == 3
        assert app.cursor.filter_x.min_cutoff == app.s.smoothing()[0]

    def test_an_out_of_range_smoothing_index_is_ignored(self, app):
        before = app.s.smoothing_index
        send(app, "smoothing", 99)
        assert app.s.smoothing_index == before

    def test_eye_and_reference_switch(self, app):
        send(app, "eye", "left")
        send(app, "reference", "corners")
        assert app.s.dominant_eye == "left"
        assert app.s.eye_reference == "corners"

    def test_a_bogus_eye_is_ignored(self, app):
        before = app.s.dominant_eye
        send(app, "eye", "nose")
        assert app.s.dominant_eye == before

    def test_deadzone_changes_and_cannot_go_negative(self, app):
        send(app, "deadzone", 15)
        assert app.s.deadzone_px == 15.0
        send(app, "deadzone", -5)
        assert app.s.deadzone_px == 0.0

    def test_gestures_toggle(self, app):
        send(app, "gestures", False)
        assert app.s.gestures_enabled is False

    def test_anchor_needs_a_calibration(self, app):
        send(app, "anchor")
        assert app.cursor.drift_dx == 0.0

    def test_reset_drift_clears_it(self, app):
        app.cursor.drift_dx = 42.0
        send(app, "reset_drift")
        assert app.cursor.drift_dx == 0.0

    def test_quit_asks_the_loop_to_stop(self, app):
        assert app._quit is False
        send(app, "quit")
        assert app._quit is True

    def test_grid_rebuilds_the_calibrator(self, app):
        send(app, "grid", 9)
        assert app.s.calib_grid == 9
        assert app.calibrator.total_points == 9
        # The cursor must follow the new calibrator, or it would keep
        # mapping through the old one.
        assert app.cursor.calibrator is app.calibrator

    def test_grid_is_not_swapped_mid_run(self, app):
        """Changing the target list while points are being collected would
        strand the ones already taken."""
        send(app, "calibrate")
        before = app.calibrator
        send(app, "grid", 9)
        assert app.calibrator is before

    def test_actions_are_consumed_once(self, app):
        send(app, "mouse", True)
        app.mouse_on = False
        app._apply_panel_actions()
        assert app.mouse_on is False


class TestPublishedState:
    def test_snapshot_reports_the_tracker(self, app):
        app.mouse_on = True
        app._publish_state()
        state = app.bridge.snapshot()
        assert state["running"] is True
        assert state["mouse_on"] is True
        assert state["eye"] == app.s.dominant_eye
        assert state["camera"] == [app.cam_w, app.cam_h]
        assert state["screen"] == [app.screen_w, app.screen_h]

    def test_snapshot_is_json_serialisable(self, app):
        """It is sent straight to the browser; a stray numpy scalar there
        would fail the request rather than the test suite."""
        import json

        app._publish_state()
        json.dumps(app.bridge.snapshot())

    def test_quality_is_absent_before_calibration(self, app):
        app.calibrator.quality = None
        app._publish_state()
        assert app.bridge.snapshot()["quality"] is None

    def test_quality_is_reported_in_pixels(self, app):
        from fatma.quality import CalibrationQuality

        app.calibrator.quality = CalibrationQuality(
            per_point=(0.01, 0.05), mean=0.02, worst=0.05, worst_index=1
        )
        app._publish_state()
        quality = app.bridge.snapshot()["quality"]
        assert quality["mean_px"] > 1
        assert quality["worst_index"] == 1
        assert quality["points"] == 2

    def test_calibration_progress_is_reported(self, app):
        send(app, "calibrate")
        app._publish_state()
        state = app.bridge.snapshot()
        assert state["calibrating"] is True
        assert state["calib_total"] == app.calibrator.total_points

    def test_every_key_the_page_reads_is_published(self, app):
        """A key the page reads but the tracker never sets renders as
        "undefined" in the browser and is invisible from Python."""
        app._publish_state()
        published = set(app.bridge.snapshot())
        expected = {
            "running", "version", "mouse_on", "calibrating", "calib_step",
            "calib_total", "calibrated", "quality", "fps", "jitter", "camera",
            "screen", "pointer", "eye", "reference", "smoothing", "deadzone",
            "grid", "gestures", "drift", "face",
        }
        assert expected <= published
