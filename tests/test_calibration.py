"""Calibration collection, mapping, clamping and persistence."""

import json

import pytest

from fatma.calibration import CALIB_KIND, CalibrationError, Calibrator


def gaze_for(target):
    """A plausible gaze reading for a screen target.

    Roughly matches the ranges seen in real sessions: about 0.3 iris
    diameters of horizontal travel and 0.2 of vertical travel across the
    whole screen, with the vertical channel sitting below zero.
    """
    tx, ty = target
    return 0.30 * tx + 0.05, 0.20 * ty - 0.25


def run_full_calibration(cal: Calibrator, noise: float = 0.0) -> None:
    """Drive a complete nine-point run with clean samples."""
    cal.start()
    for i, target in enumerate(cal.POINTS_SCREEN):
        nx, ny = gaze_for(target)
        for k in range(cal.min_samples + 2):
            jitter = noise * (1 if k % 2 else -1)
            cal.feed(nx + jitter, ny + jitter)
        assert cal.confirm(), f"point {i} was rejected"


class TestCollection:
    def test_starts_uncalibrated(self):
        cal = Calibrator()
        assert not cal.is_calibrated
        assert not cal.active

    def test_full_run_produces_a_calibration(self):
        cal = Calibrator()
        run_full_calibration(cal)
        assert cal.is_calibrated
        assert not cal.active
        assert cal.fit_error is None

    def test_confirm_is_rejected_without_enough_samples(self):
        """Pressing SPACE before settling would poison a control point."""
        cal = Calibrator(min_samples=5)
        cal.start()
        cal.feed(0.1, -0.2)
        assert cal.confirm() is False
        assert cal.idx == 0

    def test_confirm_outside_a_run_is_rejected(self):
        assert Calibrator().confirm() is False

    def test_ready_to_confirm_tracks_sample_count(self):
        cal = Calibrator(min_samples=3)
        cal.start()
        assert not cal.ready_to_confirm
        for _ in range(3):
            cal.feed(0.1, -0.2)
        assert cal.ready_to_confirm

    def test_feed_is_ignored_when_not_calibrating(self):
        cal = Calibrator()
        cal.feed(0.1, -0.2)
        assert cal.sample_count == 0

    def test_median_rejects_the_approach_to_the_target(self):
        """While the user moves onto a target their gaze sweeps the screen.
        A mean would drag the control point along that path; the median
        must not."""
        cal = Calibrator(min_samples=3, sample_window=9)
        cal.start()
        for _ in range(4):           # transition frames, far away
            cal.feed(5.0, 5.0)
        for _ in range(5):           # settled on the target
            cal.feed(0.2, -0.1)
        cal.confirm()
        assert tuple(cal.points_eye[0]) == pytest.approx((0.2, -0.1))

    def test_cancel_abandons_the_run(self):
        cal = Calibrator()
        cal.start()
        cal.feed(0.1, -0.2)
        cal.cancel()
        assert not cal.active
        assert cal.sample_count == 0
        assert cal.idx == 0

    def test_abandoning_a_run_restores_the_previous_calibration(self):
        """Pressing 'c' by mistake and backing out with ESC must not cost
        the user a calibration that was working."""
        cal = Calibrator()
        run_full_calibration(cal)
        probe = gaze_for((0.25, 0.75))
        before = cal.map_normalized(*probe)

        cal.start()
        assert not cal.is_calibrated, "the run should behave as uncalibrated"
        cal.feed(*probe)
        cal.cancel()

        assert cal.is_calibrated
        assert cal.map_normalized(*probe) == pytest.approx(before, abs=1e-12)

    def test_progress_reporting(self):
        cal = Calibrator(min_samples=1)
        cal.start()
        assert cal.current_target == cal.POINTS_SCREEN[0]
        cal.feed(0.1, -0.2)
        cal.confirm()
        assert cal.current_target == cal.POINTS_SCREEN[1]
        assert cal.total_points == 9


class TestMapping:
    def test_maps_each_target_back_to_itself(self):
        """The round trip the whole calibration exists for."""
        cal = Calibrator()
        run_full_calibration(cal)
        for target in cal.POINTS_SCREEN:
            assert cal.map_normalized(*gaze_for(target)) == pytest.approx(
                target, abs=1e-4
            )

    def test_interpolates_between_targets(self):
        cal = Calibrator()
        run_full_calibration(cal)
        assert cal.map_normalized(*gaze_for((0.25, 0.75))) == pytest.approx(
            (0.25, 0.75), abs=0.02
        )

    def test_output_is_always_on_screen(self):
        cal = Calibrator()
        run_full_calibration(cal)
        for nx, ny in [(-50.0, -50.0), (50.0, 50.0), (0.0, 0.0)]:
            sx, sy = cal.map_normalized(nx, ny)
            assert 0.0 <= sx <= 1.0
            assert 0.0 <= sy <= 1.0

    def test_input_is_clamped_to_the_calibrated_range(self):
        """Outside its convex hull a TPS extrapolates without bound. Input
        clamping is what stops a glance away from the screen flinging the
        cursor; clipping only the output would still produce a jump."""
        cal = Calibrator()
        run_full_calibration(cal)
        far = cal.clamp_input(1000.0, -1000.0)
        assert far == (cal.nx_max, cal.ny_min)
        # A reading far outside the hull maps to the same place as one just
        # outside it, instead of to something wilder.
        a = cal.map_normalized(1000.0, -1000.0)
        b = cal.map_normalized(cal.nx_max, cal.ny_min)
        assert a == pytest.approx(b)

    def test_clamp_is_a_no_op_before_calibration(self):
        assert Calibrator().clamp_input(7.0, -7.0) == (7.0, -7.0)

    def test_uncalibrated_fallback_is_linear_and_bounded(self):
        cal = Calibrator()
        assert cal.map_normalized(0.0, 0.0) == pytest.approx((0.5, 0.5))
        assert cal.map_normalized(-0.2, -0.2) == pytest.approx((0.0, 0.0))
        assert cal.map_normalized(99.0, 99.0) == (1.0, 1.0)

    def test_map_to_screen_scales_by_resolution(self):
        cal = Calibrator()
        run_full_calibration(cal)
        nx, ny = gaze_for((0.5, 0.5))
        x, y = cal.map_to_screen(nx, ny, 1710, 1112)
        assert x == pytest.approx(855, abs=10)
        assert y == pytest.approx(556, abs=10)

    def test_a_saved_calibration_survives_a_resolution_change(self):
        """Mapping is stored in screen fractions, so a different display
        just rescales it."""
        cal = Calibrator()
        run_full_calibration(cal)
        nx, ny = gaze_for((0.25, 0.75))
        small = cal.map_to_screen(nx, ny, 1280, 800)
        large = cal.map_to_screen(nx, ny, 2560, 1600)
        assert large[0] == pytest.approx(small[0] * 2)
        assert large[1] == pytest.approx(small[1] * 2)


class TestDegenerateCalibration:
    def test_identical_readings_report_an_error_instead_of_raising(self):
        """If the user stares straight ahead through all nine targets the
        fit is impossible — the app must say so, not crash."""
        cal = Calibrator(min_samples=1)
        cal.start()
        for _ in cal.POINTS_SCREEN:
            cal.feed(0.1, -0.2)
            cal.confirm()
        assert not cal.is_calibrated
        assert cal.fit_error is not None
        assert "distinct" in cal.fit_error

    def test_a_failed_fit_leaves_the_mapping_unusable_not_half_set(self):
        cal = Calibrator(min_samples=1)
        cal.start()
        for _ in cal.POINTS_SCREEN:
            cal.feed(0.1, -0.2)
            cal.confirm()
        # Falls back to the linear approximation rather than a broken spline.
        assert cal.map_normalized(0.0, 0.0) == pytest.approx((0.5, 0.5))

    def test_a_failed_fit_preserves_an_earlier_working_calibration(self):
        cal = Calibrator()
        run_full_calibration(cal)
        probe = gaze_for((0.5, 0.5))
        before = cal.map_normalized(*probe)

        cal.min_samples = 1
        cal.start()
        for _ in cal.POINTS_SCREEN:   # a botched run: never moved the gaze
            cal.feed(0.1, -0.2)
            cal.confirm()

        assert cal.fit_error is not None
        assert cal.is_calibrated, "the old mapping should still be in force"
        assert cal.map_normalized(*probe) == pytest.approx(before, abs=1e-12)


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        cal = Calibrator()
        run_full_calibration(cal)
        path = tmp_path / "calib.json"
        cal.save(str(path), (1710, 1112))

        restored = Calibrator()
        assert restored.load(str(path))
        for target in cal.POINTS_SCREEN:
            assert restored.map_normalized(*gaze_for(target)) == pytest.approx(
                cal.map_normalized(*gaze_for(target)), abs=1e-9
            )

    def test_saved_payload_has_the_documented_shape(self, tmp_path):
        cal = Calibrator()
        run_full_calibration(cal)
        path = tmp_path / "calib.json"
        cal.save(str(path), (1710, 1112))
        data = json.loads(path.read_text())
        assert data["kind"] == CALIB_KIND
        assert data["screen"] == [1710, 1112]
        assert len(data["nx_range"]) == 2
        assert data["tps"]["kind"] == "tps"

    def test_save_is_atomic(self, tmp_path):
        """A crash mid-write must not leave an unloadable calibration."""
        cal = Calibrator()
        run_full_calibration(cal)
        path = tmp_path / "calib.json"
        cal.save(str(path), (100, 100))
        assert not (tmp_path / "calib.json.tmp").exists()
        assert list(tmp_path.iterdir()) == [path]

    def test_saving_without_a_calibration_is_an_error(self, tmp_path):
        with pytest.raises(CalibrationError, match="nothing to save"):
            Calibrator().save(str(tmp_path / "x.json"))

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert Calibrator().load(str(tmp_path / "absent.json")) is False

    def test_corrupt_json_is_rejected(self, tmp_path):
        path = tmp_path / "calib.json"
        path.write_text("{not json")
        assert Calibrator().load(str(path)) is False

    def test_old_format_is_rejected(self, tmp_path):
        """The pre-TPS format stored affine coefficients under other keys;
        loading it would silently mis-map every gaze reading."""
        path = tmp_path / "calib.json"
        path.write_text(json.dumps({"kind": "affine", "a": [1, 2, 3]}))
        cal = Calibrator()
        assert cal.load(str(path)) is False
        assert not cal.is_calibrated

    def test_inverted_gaze_range_is_rejected(self, tmp_path):
        cal = Calibrator()
        run_full_calibration(cal)
        payload = cal.to_dict((100, 100))
        payload["nx_range"] = [1.0, 0.0]
        path = tmp_path / "calib.json"
        path.write_text(json.dumps(payload))
        assert Calibrator().load(str(path)) is False

    def test_truncated_coefficients_are_rejected(self, tmp_path):
        cal = Calibrator()
        run_full_calibration(cal)
        payload = cal.to_dict((100, 100))
        payload["tps"]["cx"] = payload["tps"]["cx"][:3]
        path = tmp_path / "calib.json"
        path.write_text(json.dumps(payload))
        assert Calibrator().load(str(path)) is False

    def test_load_dict_raises_with_a_reason(self):
        with pytest.raises(CalibrationError, match="unrecognised"):
            Calibrator().load_dict({"kind": "nope"})
