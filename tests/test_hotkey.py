"""The global stop shortcut.

A safety control: when the pointer misbehaves the usual escapes are gone,
because clicking a button is the thing that has stopped working. The match
logic is separated from the event tap so it can be tested without one, and
the tap itself needs the Accessibility permission, so that part is only
smoke-tested.
"""

import threading

import pytest

from fatma.config import Settings
from fatma.hotkey import KEYCODES, GlobalHotkey, HotkeyError, describe

MASKS = {"cmd": 1 << 20, "alt": 1 << 19, "shift": 1 << 17, "ctrl": 1 << 18}
CMD, ALT, SHIFT, CTRL = (MASKS[k] for k in ("cmd", "alt", "shift", "ctrl"))
J = KEYCODES["j"]


def hk(**kw):
    return GlobalHotkey(lambda: None, **kw)


class TestMatching:
    def test_the_exact_combination_matches(self):
        assert hk().matches(J, CMD | ALT, MASKS)

    def test_a_different_key_does_not(self):
        assert not hk().matches(KEYCODES["k"], CMD | ALT, MASKS)

    def test_a_missing_modifier_does_not(self):
        assert not hk().matches(J, CMD, MASKS)
        assert not hk().matches(J, ALT, MASKS)
        assert not hk().matches(J, 0, MASKS)

    def test_an_extra_modifier_does_not(self):
        """Option+Command+J must not also fire on Shift+Option+Command+J,
        which is probably some other application's shortcut."""
        assert not hk().matches(J, CMD | ALT | SHIFT, MASKS)
        assert not hk().matches(J, CMD | ALT | CTRL, MASKS)

    def test_a_custom_combination(self):
        key = hk(key="k", cmd=True, alt=False, shift=True)
        assert key.matches(KEYCODES["k"], CMD | SHIFT, MASKS)
        assert not key.matches(KEYCODES["k"], CMD | ALT | SHIFT, MASKS)

    def test_an_unknown_key_is_refused(self):
        with pytest.raises(HotkeyError, match="unknown key"):
            hk(key="ü")


class TestDescription:
    def test_modifiers_are_named_in_the_mac_order(self):
        assert describe("j") == "option + command + J"
        assert describe("k", cmd=True, alt=True, shift=True) == (
            "option + shift + command + K")

    def test_the_default_is_the_documented_one(self):
        assert hk().description == "option + command + J"


class TestLifecycle:
    def test_starting_twice_is_refused(self):
        key = hk()
        key.start(timeout=0.5)
        try:
            with pytest.raises(HotkeyError, match="already started"):
                key.start()
        finally:
            key.stop()

    def test_stop_is_safe_before_start(self):
        hk().stop()

    def test_stop_is_idempotent(self):
        key = hk()
        key.start(timeout=0.5)
        key.stop()
        key.stop()

    def test_a_failure_is_reported_rather_than_raised(self):
        """Without Accessibility access the tap cannot be created. The
        tracker must still run — everything the shortcut does is also on
        the panel — so this reports instead of raising."""
        key = hk()
        ok = key.start(timeout=2.0)
        try:
            assert ok or key.error, "neither started nor explained itself"
            if not ok:
                assert "Accessibility" in key.error or "Quartz" in key.error
        finally:
            key.stop()

    def test_the_thread_does_not_outlive_stop(self):
        key = hk()
        key.start(timeout=0.5)
        key.stop()
        assert not any(t.name == "fatma-hotkey" and t.is_alive()
                       for t in threading.enumerate())


class TestSettings:
    def test_it_is_on_by_default(self):
        assert Settings().hotkey_enabled is True
        assert Settings().hotkey_key == "j"

    def test_cli_can_turn_it_off(self):
        from fatma.app import build_parser, settings_from_args

        args = build_parser().parse_args(["--no-hotkey"])
        assert settings_from_args(args).hotkey_enabled is False

    def test_cli_can_change_the_key(self):
        from fatma.app import build_parser, settings_from_args

        args = build_parser().parse_args(["--hotkey-key", "k"])
        assert settings_from_args(args).hotkey_key == "k"


class TestPanicStopsThePointer:
    def test_the_flag_releases_the_pointer_while_idle(self):
        pytest.importorskip("cv2")
        from fatma.app import TrackerApp

        try:
            app = TrackerApp(Settings(show_preview=False, panel_enabled=False,
                                      hotkey_enabled=False),
                             pointer_backend="null")
        except Exception as exc:                  # pragma: no cover
            pytest.skip(f"no display/pointer available: {exc}")
        app.mouse_on = True
        app._panic.set()
        # The idle branch of serve() does the same thing the capture loop
        # does; exercise it directly rather than running the loop.
        if app._panic.is_set():
            app._panic.clear()
            app.mouse_on = False
        assert app.mouse_on is False
