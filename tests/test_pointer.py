"""Mouse backends, including the performance regression that mattered most."""

import pytest

from fatma.pointer import (
    NullPointer,
    Pointer,
    PyAutoGUIPointer,
    QuartzPointer,
    create_pointer,
)


class TestNullPointer:
    def test_records_moves_and_clicks(self):
        p = NullPointer()
        p.move_to(10, 20)
        p.click()
        p.right_click()
        p.double_click()
        assert p.moves == [(10, 20)]
        assert p.clicks == ["left", "right", "double"]

    def test_position_reports_the_last_move(self):
        p = NullPointer()
        assert p.position is None
        p.move_to(1, 2)
        p.move_to(3, 4)
        assert p.position == (3, 4)

    def test_satisfies_the_pointer_protocol(self):
        assert isinstance(NullPointer(), Pointer)


class TestCreatePointer:
    def test_null_backend(self):
        assert isinstance(create_pointer("null"), NullPointer)

    def test_auto_picks_something_usable(self):
        assert isinstance(create_pointer("auto"), Pointer)

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="unknown pointer backend"):
            create_pointer("telepathy")


class TestPyAutoGUIPointer:
    def test_disables_the_global_pause(self):
        """Regression, and the single biggest win in this project.

        Every public PyAutoGUI call is wrapped in a decorator that sleeps
        for ``pyautogui.PAUSE`` afterwards. At the stock 0.1 s, calling
        moveTo once per frame caps the tracker at under 10 fps no matter
        how fast the camera and the model are.
        """
        pyautogui = pytest.importorskip("pyautogui")
        pyautogui.PAUSE = 0.1          # pretend it was never touched
        PyAutoGUIPointer()
        assert pyautogui.PAUSE == 0

    def test_disables_the_failsafe(self):
        """The fail-safe queries the cursor position before every call and
        aborts the whole program if the cursor is in a corner — which is
        exactly where calibration sends it."""
        pyautogui = pytest.importorskip("pyautogui")
        pyautogui.FAILSAFE = True
        PyAutoGUIPointer()
        assert pyautogui.FAILSAFE is False


@pytest.mark.skipif(not QuartzPointer.available(), reason="Quartz unavailable")
class TestQuartzPointer:
    def test_reports_a_plausible_cursor_position(self):
        x, y = QuartzPointer().position()
        assert x >= 0 and y >= 0

    def test_repeated_moves_to_the_same_pixel_are_dropped(self, monkeypatch):
        """The tracker re-sends the held position every frame while the
        gaze sits inside the deadzone; posting an identical event 30 times
        a second is pure overhead."""
        p = QuartzPointer()
        posted = []
        monkeypatch.setattr(
            p, "_post", lambda *a, **k: posted.append(a)
        )
        p.move_to(100, 100)
        p.move_to(100, 100)
        p.move_to(100.4, 100.4)     # same pixel after truncation
        assert len(posted) == 1
        p.move_to(101, 100)
        assert len(posted) == 2

    def test_double_click_posts_two_down_up_pairs(self, monkeypatch):
        p = QuartzPointer()
        monkeypatch.setattr(p, "position", lambda: (0.0, 0.0))
        posted = []
        monkeypatch.setattr(
            p, "_post",
            lambda ev, x, y, button, click_state=None: posted.append(click_state),
        )
        p.double_click()
        # The escalating click-state field is what the window server reads
        # as a double click rather than two singles.
        assert posted == [1, 1, 2, 2]

    def test_single_click_posts_one_pair(self, monkeypatch):
        p = QuartzPointer()
        monkeypatch.setattr(p, "position", lambda: (0.0, 0.0))
        posted = []
        monkeypatch.setattr(
            p, "_post",
            lambda ev, x, y, button, click_state=None: posted.append(click_state),
        )
        p.click()
        assert posted == [1, 1]
