"""The shared string table.

The overlay drawn on camera frames and the browser panel used to hold their
own copies of the same wording. Both read from one table now, so these
tests are mostly about that table staying complete and usable from both
sides — including the one constraint the panel does not have, that OpenCV
can only draw ASCII.
"""

import json
import re

import pytest

from fatma.app import ascii_fold
from fatma.config import VALID_LANGUAGES, Settings
from fatma.i18n import (
    DEFAULT_LANGUAGE,
    LANGUAGES,
    TEXT,
    missing_keys,
    t,
)
from fatma.panel import PAGE_TEMPLATE, render_page


class TestTable:
    def test_every_language_is_complete(self):
        assert missing_keys() == {}

    def test_no_language_has_keys_english_lacks(self):
        base = set(TEXT[DEFAULT_LANGUAGE])
        for lang, table in TEXT.items():
            assert set(table) <= base, f"{lang} has orphan keys"

    def test_languages_match_the_settings(self):
        assert set(LANGUAGES) == set(VALID_LANGUAGES)

    def test_default_is_english(self):
        assert DEFAULT_LANGUAGE == "en"
        assert Settings().language == "en"

    def test_placeholders_match_across_languages(self):
        """A key whose Turkish text expects different fields than its
        English one formats into a half-filled sentence."""
        fields = lambda s: set(re.findall(r"\{(\w+)", s))
        for key, english in TEXT["en"].items():
            for lang, table in TEXT.items():
                if key in table:
                    assert fields(table[key]) == fields(english), \
                        f"{lang}:{key}"


class TestLookup:
    def test_returns_the_requested_language(self):
        assert t("tr", "p.power.start") == "Takibi başlat"
        assert t("en", "p.power.start") == "Start tracking"

    def test_falls_back_to_english(self):
        assert t("de", "p.power.start") == "Start tracking"

    def test_an_unknown_key_returns_itself(self):
        """Better a visible wrong label than an exception mid-frame."""
        assert t("en", "p.nope.missing") == "p.nope.missing"

    def test_formats_fields(self):
        text = t("en", "p.calib.running", step=3, total=13)
        assert "3/13" in text

    def test_a_missing_field_does_not_raise(self):
        assert t("en", "p.calib.running", step=3)

    def test_numeric_formats_are_applied(self):
        text = t("en", "span.narrow", x=0.1749, y=0.079,
                 min_x=0.25, min_y=0.12)
        assert "0.175" in text and "0.079" in text


class TestOverlayCanDrawIt:
    @pytest.mark.parametrize("lang", LANGUAGES)
    def test_every_overlay_string_survives_the_ascii_fold(self, lang):
        """OpenCV's Hershey fonts are ASCII-only. A string that folds to
        "?" is unreadable on the camera overlay even though it looks fine
        in the panel."""
        for key, text in TEXT[lang].items():
            if not (key.startswith("hud.") or key.startswith("calib.")
                    or key.startswith("grade.") or key.startswith("span.")):
                continue
            folded = ascii_fold(text)
            assert folded.isascii()
            # Folding must not *introduce* a "?"; a key whose text is
            # deliberately "?" is fine.
            assert folded.count("?") == text.count("?"), \
                f"{lang}:{key} -> {folded}"


class TestPageRendering:
    def test_the_template_has_no_leftover_placeholders(self):
        page = render_page("en")
        assert "__STRINGS__" not in page
        assert "__LANG__" not in page

    def test_the_whole_table_is_shipped_to_the_browser(self):
        """Switching language in the panel must not need a round trip."""
        page = render_page("en")
        payload = re.search(r"const STR=(\{.*?\});\n", page, re.S)
        assert payload
        table = json.loads(payload.group(1))
        assert set(table) == set(LANGUAGES)
        assert table["tr"]["p.power.start"] == "Takibi başlat"

    def test_the_initial_language_is_set(self):
        assert "let LANG='tr'" in render_page("tr")
        assert "let LANG='en'" in render_page("en")

    def test_an_unknown_language_falls_back(self):
        assert "let LANG='en'" in render_page("de")

    def test_the_document_language_follows_the_selection(self):
        """CSS text-transform follows it: Turkish uppercases "i" to a
        dotted capital, so an English page marked Turkish renders
        "POINTER" with a dotted I."""
        assert '<html lang="en">' in render_page("en")
        assert '<html lang="tr">' in render_page("tr")

    def test_every_static_slot_has_a_string(self):
        """A data-i18n attribute naming a key that does not exist renders
        as the raw key on screen."""
        keys = set(re.findall(r'data-i18n="([^"]+)"', PAGE_TEMPLATE))
        assert keys
        unknown = keys - set(TEXT[DEFAULT_LANGUAGE])
        assert not unknown, f"page references unknown keys: {unknown}"

    def test_every_js_lookup_has_a_string(self):
        keys = set(re.findall(r"T\('([a-z][\w.]+)'", PAGE_TEMPLATE))
        assert keys
        unknown = keys - set(TEXT[DEFAULT_LANGUAGE])
        assert not unknown, f"page looks up unknown keys: {unknown}"

    def test_the_page_carries_no_stray_turkish(self):
        """Everything the user reads should come from the table, so the
        template itself should be free of Turkish text."""
        body = PAGE_TEMPLATE.replace("__STRINGS__", "")
        turkish = set("ışğüöçİŞĞÜÖÇ") & set(body)
        assert not turkish, f"untranslated text in the template: {turkish}"


class TestLanguageAction:
    def test_the_panel_accepts_a_language_change(self):
        from fatma.panel import PanelBridge

        bridge = PanelBridge()
        assert bridge.submit("language", "tr")[0]
        assert bridge.drain() == [("language", "tr")]

    def test_settings_reject_an_unknown_language(self):
        with pytest.raises(ValueError, match="language"):
            Settings(language="de")

    def test_cli_sets_the_language(self):
        from fatma.app import build_parser, settings_from_args

        args = build_parser().parse_args(["--lang", "tr"])
        assert settings_from_args(args).language == "tr"
