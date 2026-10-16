"""The browser control panel: its bridge, its server and its actions.

The panel runs on HTTP threads while the tracker runs on the capture
thread, and the whole design rests on them never touching the same object.
:class:`PanelBridge` is that boundary, so most of this file is about it:
what crosses, what is rejected, and that nothing else gets through.
"""

import json
import threading
import urllib.error
import urllib.request

import pytest

from fatma.config import Settings
from fatma.panel import PAGE, Panel, PanelBridge, _free_port


@pytest.fixture
def bridge():
    return PanelBridge()


@pytest.fixture
def server(bridge):
    panel = Panel(bridge)
    if not panel.start():
        pytest.skip("no free port for the panel")
    yield panel
    panel.stop()


def get(url, timeout=3.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.read()


def post(url, payload, timeout=3.0):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class TestBridge:
    def test_published_state_is_what_comes_back(self, bridge):
        bridge.publish({"fps": 30, "mouse_on": True})
        assert bridge.snapshot() == {"fps": 30, "mouse_on": True}

    def test_snapshot_is_a_copy(self, bridge):
        """The HTTP thread must not be able to mutate what the loop owns."""
        bridge.publish({"fps": 30})
        bridge.snapshot()["fps"] = 0
        assert bridge.snapshot()["fps"] == 30

    def test_actions_queue_and_drain_once(self, bridge):
        bridge.submit("calibrate")
        bridge.submit("mouse", True)
        assert bridge.drain() == [("calibrate", None), ("mouse", True)]
        assert bridge.drain() == []

    def test_unknown_actions_are_refused(self, bridge):
        ok, message = bridge.submit("rm -rf")
        assert not ok
        assert "bilinmeyen" in message
        assert bridge.drain() == []

    def test_values_are_coerced_to_the_declared_type(self, bridge):
        bridge.submit("smoothing", "2")
        bridge.submit("deadzone", 15)
        bridge.submit("mouse", 1)
        assert bridge.drain() == [
            ("smoothing", 2), ("deadzone", 15.0), ("mouse", True)
        ]

    def test_a_value_that_cannot_be_coerced_is_refused(self, bridge):
        ok, message = bridge.submit("smoothing", "çok")
        assert not ok
        assert "geçersiz" in message

    def test_an_action_needing_a_value_is_refused_without_one(self, bridge):
        ok, message = bridge.submit("eye")
        assert not ok
        assert "değer" in message

    def test_valueless_actions_take_no_value(self, bridge):
        assert bridge.submit("anchor")[0]
        assert bridge.drain() == [("anchor", None)]

    def test_the_queue_is_bounded(self, bridge):
        """A panel left open in a background tab must not grow the queue
        without limit if the loop ever stalls."""
        small = PanelBridge(max_queue=4)
        for _ in range(50):
            small.submit("anchor")
        assert len(small.drain()) == 4

    def test_concurrent_submits_do_not_lose_the_lock(self, bridge):
        big = PanelBridge(max_queue=1000)

        def worker():
            for _ in range(100):
                big.submit("anchor")

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(big.drain()) == 400


class TestServer:
    def test_serves_the_page(self, server):
        status, body = get(server.url)
        assert status == 200
        assert b"<title>Fatma" in body

    def test_serves_the_state_as_json(self, server, bridge):
        bridge.publish({"fps": 29.5, "mouse_on": False})
        status, body = get(server.url + "/api/state")
        assert status == 200
        assert json.loads(body)["fps"] == 29.5

    def test_an_action_reaches_the_queue(self, server, bridge):
        status, payload = post(server.url + "/api/action",
                               {"action": "calibrate"})
        assert status == 200 and payload["ok"]
        assert bridge.drain() == [("calibrate", None)]

    def test_a_bad_action_is_rejected_without_queueing(self, server, bridge):
        status, payload = post(server.url + "/api/action",
                               {"action": "explode"})
        assert payload["ok"] is False
        assert bridge.drain() == []

    def test_malformed_json_is_rejected(self, server):
        request = urllib.request.Request(
            server.url + "/api/action", data=b"{not json",
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=3.0) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        assert status == 400

    def test_unknown_paths_are_not_found(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(server.url + "/etc/passwd")
        assert exc.value.code == 404

    def test_binds_to_loopback_only(self, server):
        """The panel drives the user's mouse; it has no business being
        reachable from the network."""
        assert server.server.server_address[0] == "127.0.0.1"
        assert server.url.startswith("http://127.0.0.1:")

    def test_stop_releases_the_port(self, bridge):
        panel = Panel(bridge)
        assert panel.start()
        port = panel.port
        panel.stop()
        assert panel.port is None
        assert _free_port((port,)) == port

    def test_stop_is_safe_before_start(self, bridge):
        Panel(bridge).stop()

    def test_a_second_panel_takes_another_port(self, bridge):
        first, second = Panel(bridge), Panel(PanelBridge())
        try:
            assert first.start()
            assert second.start()
            assert first.port != second.port
        finally:
            first.stop()
            second.stop()

    def test_a_taken_requested_port_fails_rather_than_wandering(self, bridge):
        first = Panel(bridge)
        assert first.start()
        try:
            second = Panel(PanelBridge(), port=first.port)
            assert second.start() is False
        finally:
            first.stop()


class TestPage:
    def test_every_action_the_page_sends_is_one_the_bridge_accepts(self):
        """The page and the bridge are edited separately; a typo in either
        would be a button that silently does nothing."""
        import re

        sent = set(re.findall(r"act\('([a-z_]+)'", PAGE))
        assert sent, "no actions found in the page"
        unknown = sent - set(PanelBridge.ACTIONS)
        assert not unknown, f"page sends unknown actions: {unknown}"

    def test_the_page_polls_the_state_endpoint(self):
        assert "/api/state" in PAGE
        assert "/api/action" in PAGE

    def test_the_page_names_no_external_host_but_fonts(self):
        import re

        hosts = set(re.findall(r"https?://([a-zA-Z0-9.-]+)", PAGE))
        assert hosts <= {"fonts.googleapis.com", "fonts.gstatic.com"}, hosts


class TestSettings:
    def test_panel_is_on_by_default(self):
        assert Settings().panel_enabled is True
        assert Settings().panel_port is None

    def test_port_can_be_pinned(self):
        assert Settings(panel_port=9000).panel_port == 9000

    @pytest.mark.parametrize("port", [80, 0, 70000])
    def test_a_silly_port_is_rejected(self, port):
        with pytest.raises(ValueError, match="panel_port"):
            Settings(panel_port=port)

    def test_cli_can_turn_it_off(self):
        from fatma.app import build_parser, settings_from_args

        args = build_parser().parse_args(["--no-panel"])
        assert settings_from_args(args).panel_enabled is False

    def test_cli_can_pin_the_port(self):
        from fatma.app import build_parser, settings_from_args

        args = build_parser().parse_args(["--panel-port", "9100"])
        assert settings_from_args(args).panel_port == 9100


class TestIconAndBrand:
    def test_the_favicon_is_embedded_not_served(self):
        """The page stays one self-contained document, so the tab icon
        travels with it rather than needing a second request."""
        from fatma.panel import FAVICON, render_page

        page = render_page("en")
        assert "__FAVICON__" not in page
        if FAVICON:
            assert "data:image/png;base64," in page
            assert FAVICON[:16] in page

    def test_a_missing_icon_does_not_break_the_page(self, monkeypatch):
        """A tab with no icon is a cosmetic loss, not a reason to refuse to
        serve the panel."""
        import fatma.panel as panel

        monkeypatch.setattr(panel, "FAVICON", "")
        page = panel.render_page("en")
        assert "__FAVICON__" not in page
        assert "<title>Fatma" in page

    def test_the_wordmark_returns_to_the_first_tab(self):
        """A logo going home is the convention; the panel had no way back
        to Control except the tab itself."""
        from fatma.panel import PAGE_TEMPLATE

        assert 'id="brand"' in PAGE_TEMPLATE
        assert "#brand').onclick" in PAGE_TEMPLATE
        assert "function showTab(" in PAGE_TEMPLATE

    def test_the_icon_source_exists(self):
        import pathlib

        svg = pathlib.Path(__file__).resolve().parent.parent / "assets" / "nazar.svg"
        assert svg.exists(), "assets/nazar.svg is the source make_icon.py renders"
        assert "<svg" in svg.read_text()
