# Fatma: deviceless eye tracker

Fare imlecini gözlerinle kontrol et — dizüstünde zaten olan kameradan
başka hiçbir şey gerekmeden. Başlık yok, kızılötesi aydınlatıcı yok, özel
donanım yok, abonelik yok.

İris işaret noktaları MediaPipe FaceLandmarker'dan geliyor. Thin-plate-spline
kalibrasyonu bunları ekran koordinatına çeviriyor. Göz kırpma ve kaş
kaldırma tıklamaya dönüşüyor.

*English: [README.md](README.md)*

> **Bu bir tıbbi cihaz değil, güvenip bağlanacağın bir yardımcı cihaz da
> değil.** Hata payı yaklaşık 50–150 piksel — ekranın bir *bölgesine*
> yönelmek için yeterli, küçük bir düğmeye güvenilir şekilde basmak için
> değil. Yardım çağırmak, acil durumda iletişim kurmak ya da birinin
> bilgisayarı kullanmasının tek yolu olarak buna güvenme. Güvenilir göz
> kontrolü gerekiyorsa kızılötesi takip cihazlarına ve
> [OptiKey](https://github.com/OptiKey/OptiKey)'e bak.

## Durum

macOS'ta (Apple Silicon, Python 3.12) çalışıyor ve test edildi. Linux ve
Windows denenmedi — takip çekirdeği taşınabilir, imleç arka ucu ve pencere
yönetimi değil.

Şu an her kontrol bir tuşa basmaktan ibaret; yani sadece gözünü
oynatabilen biri yanında birisi olmadan uygulamayı başlatamaz, kalibre
edemez, durduramaz. Sadece-bakışla kontrol yazıldı ama bağlanmadı:
[`parked/`](parked/).

## Kurulum

```bash
git clone <fork-adresin> Fatma
cd Fatma
./run.sh
```

İlk çalıştırma venv'i kurar, bağımlılıkları yükler, landmark modelini
(3.6 MB) indirir ve başlatır. Argümanlar aynen aktarılır:
`./run.sh --eye left`.

MediaPipe Python 3.12'de stabil; 3.13 ve sonrası için çoğu zaman wheel yok.

### macOS izinleri

İlk çalıştırmada iki izin gerekir. Verdikten sonra terminali kapatıp aç.

- **Kamera** — System Settings → Privacy & Security → Camera
- **Erişilebilirlik** (imleci oynatmak için) — System Settings → Privacy &
  Security → Accessibility

## Kullanım

`c`'ye bas, her hedefe sırayla bak, her birinde SPACE. Sonra `m` ile imleci
gözlerine devret.

| Tuş | Ne yapar |
| --- | --- |
| `c` | Kalibrasyonu başlatır — her hedefe bak, SPACE ile onayla |
| `SPACE` | Sıradaki kalibrasyon hedefini onaylar |
| `ESC` | Kalibrasyonu iptal eder, öncekini geri yükler |
| `m` | İmleç kontrolünü açar/kapatır |
| `s` | Yumuşatmayı değiştirir (4 kademe) |
| `a` | Sürüklenme çıpası — ekranın ortasına bakarken bas |
| `A` | Çıpayı sıfırlar |
| `q` | Çıkar |

Jestler: **sol göz kırpma** = tıklama, **sağ göz kırpma** = sağ tık,
**iki kaşı birden kaldırma** = çift tık. Normal iki gözlü kırpma hiçbir şey
yapmaz.

```bash
python -m fatma --dry-run        # takip et ve göster, mouse'a dokunma
python -m fatma --benchmark      # 200 kare, aşama maliyetleri, çık
python -m fatma --eye left       # left / right / both / auto
python -m fatma --grid 9         # 13 yerine 9 kalibrasyon hedefi
python -m fatma --no-gestures    # kırpma ve kaş tıklamalarını kapat
python -m fatma --help           # tamamı
```

Tam ekran kalibrasyon hedefi hiç görünmüyorsa `--calib-window normal`.

## Hassasiyet

Burada her şeyi yöneten tek bir sayı var ve acımasız: **bakış sinyali tüm
ekran boyunca yarım iris çapından az yol kat ediyor.** 1470×956 bir ekranda
gerçek bir kalibrasyondan ölçülen hali:

| eksen | bakış aralığı | ekran | ölçek |
| --- | --- | --- | --- |
| yatay | 0.401 iris çapı | 1470 px | 1 birim = 3663 px |
| dikey | 0.200 iris çapı | 956 px | 1 birim = 4791 px |

Yani 50 piksel imleç doğruluğu için landmark'tan 0.014 iris çapı hassasiyet
gerekiyor. 640×480'de iris çapı ~18 piksel, dolayısıyla **1 piksellik
landmark hatası ≈ 206 ekran pikseli**. Sistem tamamen alt-piksel bölgesinde
çalışıyor ve bunu hiçbir filtre ayarı değiştirmiyor.

Değiştirenler, etki sırasıyla:

1. **Çözünürlük.** En büyük kaldıraç. 640 → 1280 hatayı yarıya, → 1920
   üçte bire indirir. Eskiden pahalıydı; imleç çağrısı 114 ms yerine
   0.03 ms sürünce bedava hale geldi. Varsayılan **1280×720**. Kamera bu
   modu vermezse açılışta uyarı basılır — sessizce 640'a düşmek
   hassasiyeti görünmeden yarıya böler.
2. **İki göz birden.** İki göz aynı bakışı bağımsız gürültüyle ölçer,
   ortalaması gürültüyü **√2** azaltır. `--eye auto` (varsayılan) gözler
   uyuştuğu sürece ortalar, ayrıştıklarında tek göze düşer — şaşılık,
   düşük göz kapağı, kadraj dışında kalan göz.
3. **Daha sağlam referans.** İris ofseti bir referans noktasına göre
   ölçülüyor. Eskiden iki göz köşesiydi, artık **altı landmark**: her köşe
   artı onun bir üstündeki ve bir altındaki kontur noktası. Ortalama
   referans gürültüsünü ~√3 azaltır, dikey simetri sayesinde inen göz
   kapağı dikey kanala sızmak yerine birinci mertebeden sönümlenir.
   `--reference corners` eski davranışa döner.
4. **13 hedef.** Spline hatası hedeflerden en uzak yerde — kenar
   ortalarında — tepe yapar. 13'lük ızgara tam oraya birer hedef koyar.

### Ölçüm

Tahmin etmek yerine bakabilmen için:

- **Kalibrasyon doğruluğu** kalibrasyon biter bitmez çapraz doğrulanır.
  Her hedef sırayla dışarıda bırakılır, spline kalanlarla yeniden fit
  edilir ve dışarıdaki hedef tahmin edilir. Bu, haritanın *hedefler
  arasında* nasıl davranacağının dürüst tahminidir — imlecin zamanının
  tamamını geçirdiği yer orası. Ortalama ve en kötü hata piksel olarak
  yazılır, bozan hedef adıyla söylenir.
- **Jitter** önizlemede canlı: bakışın sabitken imlecin kaç piksel
  gezindiği. "en iyi" değeri gördüğün en sessiz sabitlemedir, bu kurulumun
  ulaşabildiği gürültü tabanı.

İkisi farklı şey ölçer ve farklı ilacı vardır: kötü kalibrasyon skoru →
yeniden kalibre et; yüksek jitter → çözünürlük ve ortalama.

### Neyi doğruladım, neyi doğrulamadım

`tests/test_precision.py` simüle landmark gürültüsü enjekte edip sinyalin
yayılımını ölçüyor. İki göz birleştirmenin √2 kazancı ve altı-landmark
referansın iki köşeden sessiz olduğu böyle kanıtlandı — bunlar bağımsız
örnek ortalamasından çıkan oranlar, gerçek gürültü ne çıkarsa çıksın
geçerli.

Doğrulanmadı, açıkça: gerçek yüzlerde bunun ne kadarının hayatta kaldığı,
bir kameranın 720p'yi gerçekten 30 fps verip vermediği, ve anchor landmark
indislerinin senin modelinde gerçekten köşe komşusu olduğu (FaceMesh
topolojisinden alındı, canlı çıktıyla karşılaştırılmadı).
`--reference corners --eye right --width 640` ile eski davranışa dönüp A/B
yapabilirsin.

### Bilinen sınırlar

- **Kafanı sabit tut.** Takip yalnızca göz hareketinden yapılıyor, kafa
  pozu telafi edilmiyor. Kalan en büyük hata kaynağı bu, ve MediaPipe kafa
  dönüşümünü zaten veriyor — sadece bağlanmadı.
- Yüzüne gelen ışık çok yardımcı olur.
- Kalibrasyon `calibration.json`'a yazılır ve açılışta geri yüklenir. Ekran
  kesri olarak saklandığı için çözünürlük değişiminden etkilenmez, ama
  kamera açısı veya oturuş pozisyonu değişirse yenilenmesi gerekir.

## Performans

Kare başına takip maliyeti ~2–5 ms; yani döngü koda değil kameraya bağlı.
Buraya gelmek PyAutoGUI'yi düzeltmekten geçti:

| yöntem | kare başına |
| --- | --- |
| `pyautogui.moveTo()`, varsayılan ayarlar | **114 ms** |
| `pyautogui.moveTo()`, `PAUSE = 0` | 11 ms |
| Doğrudan Quartz `CGEventPost` | **0.03 ms** |

PyAutoGUI'nin her public çağrısı, çağrıdan sonra `pyautogui.PAUSE` kadar
uyuyan bir dekoratörle sarılı; varsayılanı 0.1 saniye. Kare başına bir
`moveTo`, kamera ve model ne kadar hızlı olursa olsun döngüyü ~8 fps'e
kilitliyor. `fatma.pointer.QuartzPointer` Core Graphics olayını kendisi
gönderiyor.

İkinci sorun gecikmeydi. `VideoCapture.read()` kareleri sırayla verir, yani
kameradan yavaş bir döngü giderek geriye düşer ve gecikme oturum boyunca
büyür. `fatma.camera.FrameGrabber` yakalamayı kendi thread'inde yapıp
yalnızca en yeni kareyi tutar.

Kendi makinende `--benchmark` ile ölç.

## Yapı

Matematik ve durum mantığı kamera, ekran veya mouse import etmiyor — test
edilebilir olmasının sebebi bu.

```
fatma/
  config.py       Settings — tüm ayarlanabilir değerler, doğrulamalı
  filters.py      OneEuroFilter, TemporalMedian
  landmarks.py    iris / göz kapağı / kaş geometrisi
  tps.py          thin-plate-spline interpolasyon
  calibration.py  Calibrator ve kalibrasyon dosya formatı
  quality.py      doğruluk ölçümü: çapraz doğrulama, jitter
  gestures.py     kırpma + kaş durum makinesi (olay üretir, tıklamaz)
  cursor.py       bakış → piksel hattı: filtre, eşleme, ölü bölge, çıpa
  camera.py       thread'li kare yakalayıcı
  pointer.py      mouse arka uçları (Quartz / PyAutoGUI / Null)
  app.py          OpenCV + MediaPipe çalışma döngüsü ve CLI
legacy/           refaktör öncesi tek dosyalık sürüm, referans için
parked/           sadece-bakışla kontrol, yazıldı ama bağlanmadı
```

Yalnızca `app.py` cv2, mediapipe veya pyautogui import eder.

## Testler

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

254 test. Saf mantık modüllerinde kapsam %98–100; kapsanmayan kısım kamera
ve gerçek mouse gerektiren G/Ç.

```bash
python -m pytest -m "not slow"     # gerçek çıkarım yapan testleri atla
python -m pytest --cov=fatma       # kapsam raporu
```

Testler sentetik bir yüz üretir (`tests/conftest.py`): göz köşeleri tam
0.1, iris merkezleri tam 0.3 birim arayla, böylece türetilen her ölçüt
tolerans yerine kesin sayıyla doğrulanabilir.

## Katkı

En faydalı işler, kabaca sırayla:

1. **Kafa pozu telafisi.** `output_facial_transformation_matrixes`'i aç,
   yaw ve pitch ile iris ofsetini düzelt. Kalan en büyük doğruluk kazancı.
2. **Linux ve Windows imleç arka uçları.** `fatma/pointer.py`'da arayüz
   hazır; `QuartzPointer` kopyalanacak örnek.
3. **Sadece-bakışla kontrol.** `parked/` içinde dwell motoru ve bakış
   menüsü yazılı, bağlanmayı bekliyor. `parked/README.md`'ye bak.

Yeni mantığı mümkün olduğunca `app.py` dışında tut — donanımsız test
edilemeyen tek modül o.

## Lisans

MIT. [LICENSE](LICENSE) dosyasına bak.

[MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache 2.0) ve
[OpenCV](https://opencv.org/) (Apache 2.0) üzerine kurulu.
