"""The gaze menu: every control, reachable with the eyes alone.

A user who can only move their eyes cannot press a key, so a keyboard
shortcut is not a control — it is a control for their carer. Everything the
tracker can do is therefore also on this menu, which opens by dwelling in a
screen corner and is operated entirely by dwelling on large targets.

Design constraints that drive the layout:

* Webcam gaze lands within roughly 50-150 px of the intended point, so
  targets are a third of the screen across rather than button-sized.
* Opening the menu must not be easy to do by accident, so the corner
  trigger holds for longer than a menu choice does.
* Quitting must not be a single mis-dwell: a user who quits by accident
  may have no way to start the application again.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .dwell import Rect, Zone, ZoneSet


class MenuAction(str, Enum):
    """What a completed dwell on a menu target asks the app to do."""

    TOGGLE_CLICK = "toggle_click"
    TOGGLE_CURSOR = "toggle_cursor"
    RECALIBRATE = "recalibrate"
    QUIT = "quit"
    CLOSE = "close"


class MenuMode(str, Enum):
    CLOSED = "closed"
    MAIN = "main"
    CONFIRM_QUIT = "confirm_quit"


@dataclass(frozen=True)
class MenuTile:
    """One rendered target."""

    key: str
    rect: Rect
    label: str
    sublabel: str = ""
    progress: float = 0.0
    focused: bool = False
    danger: bool = False


@dataclass(frozen=True)
class MenuView:
    """Everything the renderer needs for one frame."""

    mode: MenuMode
    tiles: tuple[MenuTile, ...] = ()
    title: str = ""
    #: Progress of the corner trigger, drawn while the menu is closed.
    trigger_progress: float = 0.0


class GazeMenu:
    """Dwell-operated menu overlaying the whole screen.

    Args:
        screen_w: Display width in pixels.
        screen_h: Display height in pixels.
        hold_s: Dwell time for a menu choice.
        trigger_hold_s: Dwell time for the corner that opens the menu.
            Longer than ``hold_s`` so that glancing at the corner while
            working does not open it.
        trigger_size: Side length of the square corner trigger.
        auto_close_s: Close again if nothing is chosen for this long, so an
            accidental opening is not a trap.
    """

    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        hold_s: float = 1.2,
        trigger_hold_s: float = 1.8,
        trigger_size: int = 150,
        auto_close_s: float = 20.0,
    ) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.hold_s = hold_s
        self.auto_close_s = auto_close_s
        self.mode = MenuMode.CLOSED
        self._opened_at = 0.0

        self.trigger_rect = Rect(0, 0, trigger_size, trigger_size)
        self._trigger = ZoneSet(
            [Zone("open", self.trigger_rect, "Menü")], hold_s=trigger_hold_s
        )
        self._zones = ZoneSet([], hold_s=hold_s)
        self._progress: dict[str, float] = {}
        self._focused: str | None = None

    # --- state ---------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self.mode is not MenuMode.CLOSED

    def open(self, now: float) -> None:
        self.mode = MenuMode.MAIN
        self._opened_at = now
        self._zones.set_zones(self._main_zones())
        self._trigger.reset()
        self._focused = None
        self._progress = {}

    def close(self) -> None:
        self.mode = MenuMode.CLOSED
        self._zones.set_zones([])
        self._trigger.reset()
        self._focused = None
        self._progress = {}

    # --- layout --------------------------------------------------------

    def _grid(self, cols: int, rows: int, margin: int = 24) -> list[Rect]:
        """Split the screen below the title bar into equal tiles."""
        top = int(self.screen_h * 0.14)
        avail_h = self.screen_h - top - margin
        cell_w = (self.screen_w - margin * (cols + 1)) / cols
        cell_h = (avail_h - margin * (rows - 1)) / rows
        return [
            Rect(
                margin + c * (cell_w + margin),
                top + r * (cell_h + margin),
                cell_w,
                cell_h,
            )
            for r in range(rows)
            for c in range(cols)
        ]

    def _main_zones(self) -> list[Zone]:
        cells = self._grid(2, 3)
        keys = [
            MenuAction.TOGGLE_CLICK.value,
            MenuAction.TOGGLE_CURSOR.value,
            MenuAction.RECALIBRATE.value,
            MenuAction.CLOSE.value,
            MenuAction.QUIT.value,
        ]
        # Five actions in a 2x3 grid; the sixth cell stays empty so the
        # Quit target is not adjacent to Close on both sides.
        return [Zone(key, cells[i]) for i, key in enumerate(keys)]

    def _confirm_zones(self) -> list[Zone]:
        cells = self._grid(2, 1)
        return [
            Zone("quit_yes", cells[0]),
            Zone("quit_no", cells[1]),
        ]

    # --- per-frame -----------------------------------------------------

    def update(
        self,
        x: float,
        y: float,
        now: float,
        click_on: bool = False,
        cursor_on: bool = True,
    ) -> MenuAction | None:
        """Advance one frame.

        Args:
            x: Cursor x in screen pixels.
            y: Cursor y in screen pixels.
            now: Monotonic timestamp.
            click_on: Whether dwell clicking is currently enabled; only
                affects the label shown.
            cursor_on: Whether cursor output is currently enabled.

        Returns:
            The action chosen on this frame, if any.
        """
        self._click_on = click_on
        self._cursor_on = cursor_on

        if self.mode is MenuMode.CLOSED:
            state = self._trigger.update(x, y, now)
            self._progress = {"open": state.progress}
            if state.fired:
                self.open(now)
            return None

        if now - self._opened_at > self.auto_close_s:
            self.close()
            return None

        state = self._zones.update(x, y, now)
        self._focused = state.focused
        self._progress = {state.focused: state.progress} if state.focused else {}
        if not state.fired:
            return None

        return self._handle(state.fired, now)

    def _handle(self, key: str, now: float) -> MenuAction | None:
        if self.mode is MenuMode.MAIN:
            if key == MenuAction.QUIT.value:
                # Quitting strands a user who cannot restart the app, so it
                # takes a second deliberate dwell.
                self.mode = MenuMode.CONFIRM_QUIT
                self._opened_at = now
                self._zones.set_zones(self._confirm_zones())
                return None
            if key == MenuAction.CLOSE.value:
                self.close()
                return MenuAction.CLOSE
            self.close()
            return MenuAction(key)

        if self.mode is MenuMode.CONFIRM_QUIT:
            if key == "quit_yes":
                self.close()
                return MenuAction.QUIT
            self.mode = MenuMode.MAIN
            self._opened_at = now
            self._zones.set_zones(self._main_zones())
        return None

    # --- rendering -----------------------------------------------------

    def view(self) -> MenuView:
        """Describe what should be drawn this frame."""
        if self.mode is MenuMode.CLOSED:
            return MenuView(
                mode=self.mode,
                trigger_progress=self._progress.get("open", 0.0),
            )
        if self.mode is MenuMode.CONFIRM_QUIT:
            cells = self._confirm_zones()
            labels = [("quit_yes", "EVET, ÇIK", "", True),
                      ("quit_no", "HAYIR", "geri dön", False)]
            tiles = tuple(
                MenuTile(
                    key=key, rect=zone.rect, label=label, sublabel=sub,
                    progress=self._progress.get(key, 0.0),
                    focused=self._focused == key, danger=danger,
                )
                for zone, (key, label, sub, danger) in zip(cells, labels)
            )
            return MenuView(mode=self.mode, tiles=tiles,
                            title="Çıkmak istediğine emin misin?")

        click_on = getattr(self, "_click_on", False)
        cursor_on = getattr(self, "_cursor_on", True)
        labels = {
            MenuAction.TOGGLE_CLICK.value: (
                "TIKLAMA", "açık — kapatmak için bak" if click_on
                else "kapalı — açmak için bak", False,
            ),
            MenuAction.TOGGLE_CURSOR.value: (
                "İMLEÇ", "açık — duraklatmak için bak" if cursor_on
                else "duraklatıldı — sürdürmek için bak", False,
            ),
            MenuAction.RECALIBRATE.value: (
                "YENİDEN KALİBRE ET", "9 noktaya sırayla bak", False,
            ),
            MenuAction.CLOSE.value: ("MENÜYÜ KAPAT", "", False),
            MenuAction.QUIT.value: ("ÇIKIŞ", "onay ister", True),
        }
        tiles = tuple(
            MenuTile(
                key=zone.key, rect=zone.rect,
                label=labels[zone.key][0], sublabel=labels[zone.key][1],
                progress=self._progress.get(zone.key, 0.0),
                focused=self._focused == zone.key,
                danger=labels[zone.key][2],
            )
            for zone in self._zones.zones
        )
        return MenuView(mode=self.mode, tiles=tiles, title="FATMA — MENÜ")
