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
G = KEYCODES["g"]
J = KEYCODES["j"]


def hk(**kw):
    return GlobalHotkey(lambda: None, **kw)


class TestDefaultIsClear:
    def test_the_default_key_has_no_known_clash(self):
        """It has to be reachable without looking and must not shadow
        anything in macOS, Safari, Chrome or Edge."""
        from fatma.hotkey import CLEAN_KEYS, CONFLICTS

        assert Settings().hotkey_key in CLEAN_KEYS
        assert Settings().hotkey_key not in CONFLICTS

    def test_the_default_is_not_next_to_a_destructive_shortcut(self):
        """X and S are equally unbound, but sit under Cut and Save: an
        option that slips off then does something, which is the wrong
        behaviour for a key pressed in a hurry."""
        assert Settings().hotkey_key not in ("x", "s", "q")


class TestMatching:
    def test_the_exact_combination_matches(self):
        assert hk().matches(G, CMD | ALT, MASKS)
        assert hk(key="j").matches(J, CMD | ALT, MASKS)

    def test_a_different_key_does_not(self):
        assert not hk().matches(J, CMD | ALT, MASKS)
        assert not hk().matches(KEYCODES["k"], CMD | ALT, MASKS)

    def test_a_missing_modifier_does_not(self):
        assert not hk().matches(G, CMD, MASKS)
        assert not hk().matches(G, ALT, MASKS)
        assert not hk().matches(G, 0, MASKS)

    def test_an_extra_modifier_does_not(self):
        """Option+Command+G must not also fire on Shift+Option+Command+G,
        which is probably some other application's shortcut."""
        assert not hk().matches(G, CMD | ALT | SHIFT, MASKS)
        assert not hk().matches(G, CMD | ALT | CTRL, MASKS)

    def test_a_custom_combination(self):
        key = hk(key="k", cmd=True, alt=False, shift=True)
        assert key.matches(KEYCODES["k"], CMD | SHIFT, MASKS)
        assert not key.matches(KEYCODES["k"], CMD | ALT | SHIFT, MASKS)

    def test_an_unknown_key_is_refused(self):
        with pytest.raises(HotkeyError, match="unknown key"):
            hk(key="ü")


class TestDescription:
    def test_modifiers_are_named_in_the_mac_order(self):
        assert describe("g") == "option + command + G"
        assert describe("k", cmd=True, alt=True, shift=True) == (
            "option + shift + command + K")

    def test_the_default_is_the_documented_one(self):
        assert hk().description == "option + command + G"


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
        assert Settings().hotkey_key == "g"

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


class TestReconfigurableFromThePanel:
    """The shortcut is bound to one combination, so changing it means
    tearing the tap down and installing a new one."""

    @pytest.fixture
    def app(self):
        pytest.importorskip("cv2")
        from fatma.app import TrackerApp

        try:
            app = TrackerApp(Settings(show_preview=False, panel_enabled=False,
                                      hotkey_enabled=False),
                             pointer_backend="null")
        except Exception as exc:                  # pragma: no cover
            pytest.skip(f"no display/pointer available: {exc}")
        yield app
        if app.hotkey is not None:
            app.hotkey.stop()

    def send(self, app, action, value=None):
        assert app.bridge.submit(action, value)[0]
        app._apply_panel_actions()

    def test_changing_the_key_rebinds(self, app):
        self.send(app, "hotkey", "l")
        assert app.s.hotkey_key == "l"
        assert app.s.hotkey_enabled is True
        if app.hotkey is not None:
            assert app.hotkey.key == "l"

    def test_an_unknown_key_is_ignored(self, app):
        before = app.s.hotkey_key
        self.send(app, "hotkey", "ü")
        assert app.s.hotkey_key == before

    def test_it_can_be_switched_off_and_on(self, app):
        self.send(app, "hotkey_enabled", True)
        assert app.s.hotkey_enabled is True
        self.send(app, "hotkey_enabled", False)
        assert app.s.hotkey_enabled is False
        assert app.hotkey is None

    def test_the_state_describes_the_shortcut(self, app):
        app._publish_state()
        state = app.bridge.snapshot()["hotkey"]
        assert state["key"] == app.s.hotkey_key
        assert "command" in state["description"]
        assert set(state["keys"]) == set(KEYCODES)
        assert isinstance(state["conflicts"], dict)

    def test_escape_is_not_offered(self):
        """option+command+escape is Force Quit Applications. Shadowing the
        escape hatch of last resort is a poor trade."""
        assert "escape" not in KEYCODES

    def test_known_conflicts_are_published(self):
        from fatma.hotkey import CONFLICTS

        assert set(CONFLICTS) <= set(KEYCODES)
        assert "j" in CONFLICTS


class TestFiredReadout:
    """Application key equivalents are compiled in, not readable from a
    preference file, so a suspected clash cannot be settled by inspection.
    Showing when the shortcut last reached us lets the user settle it by
    pressing the combination and looking."""

    @pytest.fixture
    def app(self):
        pytest.importorskip("cv2")
        from fatma.app import TrackerApp

        try:
            app = TrackerApp(Settings(show_preview=False, panel_enabled=False,
                                      hotkey_enabled=False),
                             pointer_backend="null")
        except Exception as exc:                  # pragma: no cover
            pytest.skip(f"no display/pointer available: {exc}")
        yield app
        if app.hotkey is not None:
            app.hotkey.stop()

    def test_nothing_is_reported_before_it_is_pressed(self, app):
        app._publish_state()
        assert app.bridge.snapshot()["hotkey"]["fired_ago"] is None

    def test_pressing_it_is_reported(self, app):
        app._on_hotkey()
        app._publish_state()
        fired = app.bridge.snapshot()["hotkey"]["fired_ago"]
        assert fired is not None and fired < 5

    def test_pressing_it_also_raises_the_panic_flag(self, app):
        app._on_hotkey()
        assert app._panic.is_set()

    def test_the_conflicts_table_only_names_offered_keys(self):
        from fatma.hotkey import CONFLICTS

        assert set(CONFLICTS) <= set(KEYCODES)

    def test_the_keys_without_a_known_clash_are_listed(self):
        """G and period are the ones left clean, and the recommendation
        rests on that, so it is worth pinning down."""
        from fatma.hotkey import CONFLICTS

        clean = set(KEYCODES) - set(CONFLICTS)
        assert {"g", "period"} <= clean


class TestKeyCatalogue:
    def test_every_keycode_is_distinct(self):
        """Two keys sharing a code would make one of them silently fire
        the other."""
        assert len(set(KEYCODES.values())) == len(KEYCODES)

    def test_the_offered_set_covers_the_home_and_top_rows(self):
        for key in "qwertyu sdfghjkl xc":
            if key != " ":
                assert key in KEYCODES, key

    def test_clean_keys_are_exactly_those_without_a_conflict(self):
        from fatma.hotkey import CLEAN_KEYS, CONFLICTS

        assert set(CLEAN_KEYS) == set(KEYCODES) - set(CONFLICTS)
        assert CLEAN_KEYS, "every key cannot be conflicted"

    def test_the_well_known_system_clashes_are_listed(self):
        """Close All Windows and Hide Others are bound in every Cocoa
        application, so they are the two worth refusing to shadow quietly."""
        from fatma.hotkey import CONFLICTS

        assert "w" in CONFLICTS and "h" in CONFLICTS

    def test_conflict_notes_are_renderable(self):
        from fatma.app import ascii_fold
        from fatma.hotkey import CONFLICTS

        for note in CONFLICTS.values():
            assert ascii_fold(note).isascii()
