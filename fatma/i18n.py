"""User-facing strings, in one table.

The overlay drawn on the camera frames and the browser panel used to carry
their own hardcoded Turkish, which meant the same idea was worded twice and
could drift. Both now read from here, so switching the language switches
everything the user sees.

The whole table is also handed to the page as JSON, so the panel can change
language instantly without a round trip.

Note that the overlay is drawn with OpenCV's Hershey fonts, which are
ASCII-only; ``fatma.app.ascii_fold`` transliterates on the way to the
screen, so Turkish text here is written normally and folded at the edge.
"""

from __future__ import annotations

LANGUAGES = ("en", "tr")
DEFAULT_LANGUAGE = "en"

TEXT: dict[str, dict[str, str]] = {
    "en": {
        # --- calibration overlay ---
        "calib.ready": "READY - press SPACE",
        "calib.wait": "LOOK AT THE RING AND WAIT",
        "calib.hint": "Look at the ring, then press SPACE",
        "calib.cancel": "ESC = cancel",
        # --- preview overlay ---
        "hud.jitter": "Jitter",
        "hud.best": "best",
        "hud.holding": "holding still",
        "hud.calib": "Calibration",
        "hud.calib_none": "none",
        "hud.calib_mean": "avg",
        "hud.calib_worst": "worst",
        "hud.mouse": "Mouse",
        "hud.on": "ON",
        "hud.off": "OFF",
        "hud.filter": "Filter",
        "hud.drift": "Drift",
        "hud.eye": "Eye",
        # --- calibration grades ---
        "grade.great": "very good",
        "grade.good": "good",
        "grade.fair": "fair",
        "grade.poor": "poor - recalibrate",
        "grade.unknown": "?",
        # --- span advice ---
        "span.narrow": (
            "Gaze range is too small (horizontal {x:.3f}, vertical {y:.3f}; "
            "aim for at least {min_x:.2f} / {min_y:.2f}). Sit closer to the "
            "screen and move only your eyes, not your head. A small range "
            "turns landmark noise straight into cursor error."
        ),
        # --- panel: nav + chips ---
        "p.tab.control": "Control",
        "p.tab.precision": "Precision",
        "p.tab.settings": "Settings",
        "p.chip.camera": "camera",
        "p.chip.screen": "screen",
        "p.chip.pointer": "pointer",
        "p.chip.face_yes": "face visible",
        "p.chip.face_no": "no face",
        "p.chip.camera_off": "camera off",
        "p.chip.calibrated": "calibrated",
        "p.chip.uncalibrated": "not calibrated",
        # --- panel: power ---
        "p.power.start": "Start tracking",
        "p.power.stop": "Stop tracking",
        "p.state.off": "Tracking is off",
        "p.state.off_sub": (
            "Camera is closed. Starting may ask for macOS camera permission."
        ),
        "p.state.on": "Tracking",
        "p.state.calibrating": "Calibrating",
        "p.state.face": "Camera open, face visible.",
        "p.state.noface": "Camera open, looking for a face.",
        "p.error.title": "Could not start",
        # --- panel: pointer card ---
        "p.pointer.title": "Pointer",
        "p.pointer.on": "Turn pointer on",
        "p.pointer.off": "Stop pointer",
        "p.pointer.anchor": "Look at centre, anchor",
        "p.pointer.reset": "Clear anchor",
        "p.pointer.sub_free": "The pointer is free.",
        "p.pointer.sub_tracking": "The pointer is following your eyes.",
        "p.pointer.sub_needcal": (
            "Calibrate first — without it the pointer goes nowhere useful."
        ),
        # --- panel: calibration card ---
        "p.calib.title": "Calibration",
        "p.calib.start": "Start calibration",
        "p.calib.confirm": "Confirm this point",
        "p.calib.cancel": "Cancel",
        "p.calib.running": "{step}/{total} — look at the ring, then confirm",
        "p.calib.done": (
            "Calibrated. Redo it if your camera angle or seating changed."
        ),
        "p.calib.none": "Not calibrated yet. Look at each of the rings.",
        # --- panel: status card ---
        "p.status.title": "Status",
        "p.status.pointer": "Pointer",
        "p.status.drift": "Drift",
        "p.status.eye": "Eye / reference",
        "p.status.smoothing": "Smoothing",
        "p.status.deadzone": "Dead zone",
        "p.status.gestures": "Gestures",
        # --- panel: precision tab ---
        "p.acc.title": "Calibration accuracy",
        "p.acc.sub": (
            "Cross-validated: each target is held out in turn and predicted. "
            "An honest estimate of behaviour between the targets."
        ),
        "p.acc.mean": "px average",
        "p.acc.worst": "Worst target",
        "p.acc.points": "Targets",
        "p.acc.none": "not calibrated yet",
        "p.acc.pending": "Appears once you calibrate",
        "p.jitter.title": "Jitter",
        "p.jitter.sub": (
            "How far the pointer wanders while your gaze is held. The best "
            "figure is the quietest fixation seen — this setup's noise floor."
        ),
        "p.jitter.best": "Best (noise floor)",
        "p.jitter.holding": "holding still",
        "p.span.title": "Gaze range",
        "p.span.sub": (
            "How far your eyes travelled to cover the whole screen, in iris "
            "diameters. The most important number in a calibration and "
            "invisible in the residuals: a small range turns every pixel of "
            "landmark noise into hundreds of cursor pixels. Mostly set by "
            "how far you sit from the screen."
        ),
        "p.span.axes": "horizontal / vertical",
        "p.span.narrow": "TOO SMALL",
        "p.span.state": "State",
        "p.span.ok": "Good enough",
        "p.span.bad": "Too small — sit closer",
        "p.span.what": "What to do",
        "p.advice.title": "What to change",
        "p.advice.sub": (
            "A poor accuracy score means recalibrate — the problem is the "
            "mapping. High jitter means resolution and averaging — the "
            "problem is noise."
        ),
        "p.advice.ok": "All good",
        "p.advice.ok_sub": "Settings look right for precision.",
        "p.advice.lowres": "Camera below 1280",
        "p.advice.lowres_sub": (
            "Precision scales directly with resolution. Start with --width 1280."
        ),
        "p.advice.span": "Gaze range is small",
        "p.advice.span_sub": (
            "Sit closer to the screen. This matters more than any setting."
        ),
        "p.advice.calib": "Calibration is weak",
        "p.advice.calib_sub": (
            "Recalibrate: look squarely at each ring, keep your head still."
        ),
        "p.advice.jitter": "Jitter is high",
        "p.advice.jitter_sub": "Raise the smoothing or widen the dead zone.",
        "p.advice.oneeye": "Using one eye",
        "p.advice.oneeye_sub": (
            "Switching to auto cuts noise by a factor of the square root of two."
        ),
        # --- panel: settings tab ---
        "p.set.tracking": "Tracking",
        "p.set.which_eye": "Which eye",
        "p.set.eye_auto": "auto",
        "p.set.eye_left": "left",
        "p.set.eye_right": "right",
        "p.set.eye_both": "both",
        "p.set.eye_sub": (
            "auto averages both eyes while they agree, which cuts noise, and "
            "falls back to one when they do not."
        ),
        "p.set.reference": "Reference points",
        "p.set.ref_sub": "anchors: six landmarks, steadier than two corners.",
        "p.set.cursor": "Pointer behaviour",
        "p.set.smoothing": "Smoothing",
        "p.set.sm0": "fast",
        "p.set.sm1": "normal",
        "p.set.sm2": "smooth",
        "p.set.sm3": "max",
        "p.set.deadzone": "Dead zone (tremor threshold)",
        "p.set.gestures": "Gestures",
        "p.set.gest_on": "Gestures on",
        "p.set.gest_off": "Gestures off",
        "p.set.gest_sub": "Left wink clicks, right wink right-clicks.",
        "p.set.nextcal": "Next calibration",
        "p.set.targets": "Number of targets",
        "p.set.t9": "9 targets",
        "p.set.t13": "13 targets",
        "p.set.grid_sub": (
            "13 targets add the edge midpoints, where the error is largest. "
            "Takes effect on the next calibration."
        ),
        "p.set.app": "Application",
        "p.set.quit": "Quit Fatma",
        "p.set.quit_confirm": "Quit Fatma?",
        "p.set.language": "Language",
        # --- toasts ---
        "p.toast.failed": "that did not work",
        "p.toast.offline": "lost the connection",
    },
    "tr": {
        "calib.ready": "HAZIR - SPACE'e bas",
        "calib.wait": "HALKAYA BAK VE BEKLE",
        "calib.hint": "Halkaya bak, sonra SPACE'e bas",
        "calib.cancel": "ESC = iptal",
        "hud.jitter": "Jitter",
        "hud.best": "en iyi",
        "hud.holding": "sabit bakis",
        "hud.calib": "Kalibrasyon",
        "hud.calib_none": "yok",
        "hud.calib_mean": "ort",
        "hud.calib_worst": "en kotu",
        "hud.mouse": "Imlec",
        "hud.on": "ACIK",
        "hud.off": "KAPALI",
        "hud.filter": "Filtre",
        "hud.drift": "Suruklenme",
        "hud.eye": "Goz",
        "grade.great": "cok iyi",
        "grade.good": "iyi",
        "grade.fair": "idare eder",
        "grade.poor": "kotu - yeniden kalibre et",
        "grade.unknown": "?",
        "span.narrow": (
            "Bakis araligi cok dar (yatay {x:.3f}, dikey {y:.3f}; hedef en az "
            "{min_x:.2f} / {min_y:.2f}). Ekrana daha yakin otur ve kafani "
            "degil sadece gozlerini oynat. Dar aralik, landmark gurultusunu "
            "dogrudan imlec hatasina cevirir."
        ),
        "p.tab.control": "Kontrol",
        "p.tab.precision": "Hassasiyet",
        "p.tab.settings": "Ayarlar",
        "p.chip.camera": "kamera",
        "p.chip.screen": "ekran",
        "p.chip.pointer": "pointer",
        "p.chip.face_yes": "yüz görünüyor",
        "p.chip.face_no": "yüz yok",
        "p.chip.camera_off": "kamera kapalı",
        "p.chip.calibrated": "kalibre",
        "p.chip.uncalibrated": "kalibre değil",
        "p.power.start": "Takibi başlat",
        "p.power.stop": "Takibi durdur",
        "p.state.off": "Takip kapalı",
        "p.state.off_sub": (
            "Kamera kapalı. Başlatınca macOS kamera izni isteyebilir."
        ),
        "p.state.on": "Takip çalışıyor",
        "p.state.calibrating": "Kalibrasyon sürüyor",
        "p.state.face": "Kamera açık, yüz görünüyor.",
        "p.state.noface": "Kamera açık, yüz aranıyor.",
        "p.error.title": "Başlatılamadı",
        "p.pointer.title": "İmleç",
        "p.pointer.on": "İmleci aç",
        "p.pointer.off": "İmleci durdur",
        "p.pointer.anchor": "Ortaya bak, çıpala",
        "p.pointer.reset": "Çıpayı sıfırla",
        "p.pointer.sub_free": "İmleç şu an serbest.",
        "p.pointer.sub_tracking": "İmleç gözlerini takip ediyor.",
        "p.pointer.sub_needcal": (
            "Önce kalibre et — kalibrasyon olmadan imleç anlamlı bir yere gitmez."
        ),
        "p.calib.title": "Kalibrasyon",
        "p.calib.start": "Kalibrasyonu başlat",
        "p.calib.confirm": "Bu noktayı onayla",
        "p.calib.cancel": "İptal",
        "p.calib.running": "{step}/{total} — halkaya bak, sonra onayla",
        "p.calib.done": (
            "Kalibre edilmiş. Kamera açısı ya da oturuşun değiştiyse yenile."
        ),
        "p.calib.none": "Henüz kalibre edilmedi. Halkalara sırayla bak.",
        "p.status.title": "Durum",
        "p.status.pointer": "İmleç",
        "p.status.drift": "Sürüklenme",
        "p.status.eye": "Göz / referans",
        "p.status.smoothing": "Yumuşatma",
        "p.status.deadzone": "Ölü bölge",
        "p.status.gestures": "Jestler",
        "p.acc.title": "Kalibrasyon doğruluğu",
        "p.acc.sub": (
            "Çapraz doğrulama: her hedef sırayla dışarıda bırakılıp tahmin "
            "ediliyor. Hedefler arasındaki davranışın dürüst tahmini."
        ),
        "p.acc.mean": "px ortalama",
        "p.acc.worst": "En kötü hedef",
        "p.acc.points": "Hedef sayısı",
        "p.acc.none": "henüz kalibre edilmedi",
        "p.acc.pending": "Kalibrasyon yapınca burada çıkar",
        "p.jitter.title": "Jitter",
        "p.jitter.sub": (
            "Bakışın sabitken imlecin ne kadar gezindiği. En iyi değer "
            "gördüğün en sessiz sabitleme — bu kurulumun gürültü tabanı."
        ),
        "p.jitter.best": "En iyi (gürültü tabanı)",
        "p.jitter.holding": "sabit bakış",
        "p.span.title": "Bakış aralığı",
        "p.span.sub": (
            "Gözlerin tüm ekranı taramak için ne kadar yol kat etti, iris "
            "çapı cinsinden. Kalibrasyonun en önemli sayısı ve artıklarda "
            "görünmüyor: dar bir aralık, her landmark pikselini yüzlerce "
            "ekran pikseline çevirir. Çoğunlukla ekrana uzaklığın belirler."
        ),
        "p.span.axes": "yatay / dikey",
        "p.span.narrow": "ÇOK DAR",
        "p.span.state": "Durum",
        "p.span.ok": "Yeterli",
        "p.span.bad": "Dar — ekrana yaklaş",
        "p.span.what": "Ne yapmalı",
        "p.advice.title": "Neyi değiştirmeli",
        "p.advice.sub": (
            "Doğruluk skoru kötüyse yeniden kalibre et — sorun haritada. "
            "Jitter yüksekse çözünürlük ve ortalama — sorun gürültüde."
        ),
        "p.advice.ok": "Her şey yolunda",
        "p.advice.ok_sub": "Ayarlar hassasiyet için uygun.",
        "p.advice.lowres": "Kamera 1280 altında",
        "p.advice.lowres_sub": (
            "Hassasiyet doğrudan çözünürlüğe bağlı. --width 1280 ile başlat."
        ),
        "p.advice.span": "Bakış aralığı dar",
        "p.advice.span_sub": (
            "Ekrana daha yakın otur. Doğruluğu en çok bu belirliyor."
        ),
        "p.advice.calib": "Kalibrasyon zayıf",
        "p.advice.calib_sub": (
            "Yeniden kalibre et: her halkaya tam bak, kafanı oynatma."
        ),
        "p.advice.jitter": "Jitter yüksek",
        "p.advice.jitter_sub": "Yumuşatmayı artır ya da ölü bölgeyi büyüt.",
        "p.advice.oneeye": "Tek göz kullanılıyor",
        "p.advice.oneeye_sub": "auto moda geçmek gürültüyü √2 azaltır.",
        "p.set.tracking": "Takip",
        "p.set.which_eye": "Hangi göz",
        "p.set.eye_auto": "auto",
        "p.set.eye_left": "sol",
        "p.set.eye_right": "sağ",
        "p.set.eye_both": "ikisi",
        "p.set.eye_sub": (
            "auto, gözler uyuştuğu sürece ortalar — gürültüyü azaltır — "
            "ayrıştıklarında tek göze düşer."
        ),
        "p.set.reference": "Referans noktaları",
        "p.set.ref_sub": "anchors: altı landmark, iki göz köşesinden sağlam.",
        "p.set.cursor": "İmleç davranışı",
        "p.set.smoothing": "Yumuşatma",
        "p.set.sm0": "hızlı",
        "p.set.sm1": "normal",
        "p.set.sm2": "yumuşak",
        "p.set.sm3": "maks",
        "p.set.deadzone": "Ölü bölge (titreme eşiği)",
        "p.set.gestures": "Jestler",
        "p.set.gest_on": "Jestler açık",
        "p.set.gest_off": "Jestler kapalı",
        "p.set.gest_sub": "Sol kırpma tık, sağ kırpma sağ tık.",
        "p.set.nextcal": "Sonraki kalibrasyon",
        "p.set.targets": "Hedef sayısı",
        "p.set.t9": "9 hedef",
        "p.set.t13": "13 hedef",
        "p.set.grid_sub": (
            "13 hedef, kenar ortalarına da nokta koyar — hatanın en büyük "
            "olduğu yer orası. Bir sonraki kalibrasyonda geçerli olur."
        ),
        "p.set.app": "Uygulama",
        "p.set.quit": "Fatma'yı kapat",
        "p.set.quit_confirm": "Fatma kapatılsın mı?",
        "p.set.language": "Dil",
        "p.toast.failed": "olmadı",
        "p.toast.offline": "bağlantı koptu",
    },
}


def t(lang: str, key: str, **fmt) -> str:
    """Look a string up, falling back to English and then to the key.

    A missing key returns the key itself rather than raising: a label that
    reads ``p.set.app`` on screen is a bug report, while an exception in
    the middle of drawing a frame would take the tracker down.
    """
    table = TEXT.get(lang) or TEXT[DEFAULT_LANGUAGE]
    text = table.get(key) or TEXT[DEFAULT_LANGUAGE].get(key) or key
    if fmt:
        try:
            return text.format(**fmt)
        except (KeyError, IndexError, ValueError):
            return text
    return text


def missing_keys() -> dict[str, set[str]]:
    """Keys present in English but absent from another language."""
    base = set(TEXT[DEFAULT_LANGUAGE])
    return {
        lang: base - set(table)
        for lang, table in TEXT.items()
        if lang != DEFAULT_LANGUAGE and base - set(table)
    }
