"""Settings validation and the CLI that builds them."""

import pytest

from fatma.config import SMOOTHING_LEVELS, Settings


class TestValidation:
    def test_defaults_are_valid(self):
        Settings()

    @pytest.mark.parametrize("kwargs,message", [
        ({"dominant_eye": "nose"}, "dominant_eye"),
        ({"smoothing_index": 99}, "smoothing_index"),
        ({"smoothing_index": -1}, "smoothing_index"),
        ({"median_window": 0}, "median_window"),
        ({"ear_open": 0.1, "ear_closed": 0.2}, "hysteresis"),
        ({"brow_raise_ratio": 0.9}, "brow_raise_ratio"),
        ({"brow_guard_ratio": 1.9}, "brow_guard_ratio"),
        ({"brow_ema_alpha": 0.0}, "brow_ema_alpha"),
        ({"brow_ema_alpha": 1.0}, "brow_ema_alpha"),
    ])
    def test_rejects_bad_values(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            Settings(**kwargs)

    def test_guard_ratio_may_equal_the_raise_ratio(self):
        Settings(brow_guard_ratio=1.15, brow_raise_ratio=1.15)


class TestSmoothing:
    def test_reports_the_selected_preset(self):
        assert Settings(smoothing_index=0).smoothing() == SMOOTHING_LEVELS[0]

    def test_cycles_and_wraps(self):
        s = Settings(smoothing_index=len(SMOOTHING_LEVELS) - 1)
        assert s.cycle_smoothing() == SMOOTHING_LEVELS[0]
        assert s.smoothing_index == 0

    def test_presets_get_smoother_as_the_index_rises(self):
        cutoffs = [level[0] for level in SMOOTHING_LEVELS]
        assert cutoffs == sorted(cutoffs, reverse=True)


class TestCli:
    def test_defaults_round_trip_into_settings(self):
        from fatma.app import build_parser, settings_from_args

        args = build_parser().parse_args([])
        settings = settings_from_args(args)
        assert settings.dominant_eye == "auto"
        assert settings.eye_reference == "anchors"
        assert settings.calib_grid == 13
        assert (settings.cam_width, settings.cam_height) == (1280, 720)
        assert settings.camera_index == 0
        assert settings.threaded_capture is True
        assert settings.show_preview is True

    def test_flags_are_applied(self):
        from fatma.app import build_parser, settings_from_args

        args = build_parser().parse_args([
            "--eye", "left", "--camera", "2", "--deadzone", "15",
            "--no-preview", "--no-gestures", "--no-threaded-capture",
            "--mouse", "--smoothing", "3",
        ])
        settings = settings_from_args(args)
        assert settings.dominant_eye == "left"
        assert settings.camera_index == 2
        assert settings.deadzone_px == 15
        assert settings.show_preview is False
        assert settings.gestures_enabled is False
        assert settings.threaded_capture is False
        assert settings.mouse_enabled_at_start is True
        assert settings.smoothing_index == 3

    def test_paths_are_overridable(self, tmp_path):
        from fatma.app import build_parser, settings_from_args

        args = build_parser().parse_args([
            "--calibration", str(tmp_path / "c.json"),
            "--model", str(tmp_path / "m.task"),
        ])
        settings = settings_from_args(args)
        assert settings.calibration_path == str(tmp_path / "c.json")
        assert settings.model_path == str(tmp_path / "m.task")

    def test_rejects_an_unknown_eye(self):
        from fatma.app import build_parser

        with pytest.raises(SystemExit):
            build_parser().parse_args(["--eye", "nose"])

    def test_an_out_of_range_preset_is_reported_not_crashed(self, capsys):
        from fatma.app import main

        assert main(["--smoothing", "99"]) == 2
        assert "smoothing_index" in capsys.readouterr().err
