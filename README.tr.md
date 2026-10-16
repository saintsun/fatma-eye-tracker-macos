<div align="center">

# Fatma: Göz Takibi

**Fare imlecini gözlerinle oynat.**

Başlık yok, kızılötesi yok, özel donanım yok — dizüstünde zaten olan kamera
yetiyor. Her şey kendi makinende çalışıyor, hiçbir görüntü dışarı çıkmıyor.

![platform](https://img.shields.io/badge/platform-macOS%2013%2B-111?style=flat-square)
![python](https://img.shields.io/badge/python-3.12-111?style=flat-square)
![cost](https://img.shields.io/badge/ücret-bedava-00c853?style=flat-square)
![license](https://img.shields.io/badge/lisans-MIT-111?style=flat-square)

*[English](README.md) · **Türkçe***

*v0.3.0*

</div>

---

> **Tıbbi cihaz değil, güvenip bağlanacağın bir şey de değil.** Hata payı
> yaklaşık 50–150 piksel — ekranın bir *bölgesine* yönelmek için yeterli,
> küçük bir düğmeye güvenilir şekilde basmak için değil. Yardım çağırmak,
> acil durumda iletişim kurmak ya da birinin bilgisayarı kullanmasının tek
> yolu olarak buna güvenme. Güvenilir göz kontrolü gerekiyorsa kızılötesi
> takip cihazlarına ve [OptiKey](https://github.com/OptiKey/OptiKey)'e bak.

---

## Kurulum: tek komut

Terminal'i aç (⌘ + Boşluk → "Terminal" yaz → Enter), aşağıdaki satırı
yapıştır ve Enter'a bas:

```bash
curl -fsSL https://raw.githubusercontent.com/saintsun/fatma-eye-tracker-macos/main/install.sh | bash
```

Hepsi bu. 2–3 dakika sürer, çoğu indirme.

---

## Fatma'ya bir şeyler ısmarla

Proje bedava ve bedava kalacak. İşini görüyorsa Fatma'yı besleyebilirsin.

[![Payoneer](https://img.shields.io/badge/Payoneer-Fatma'ya%20ısmarla-FF4800?style=for-the-badge&logo=payoneer&logoColor=white&labelColor=0D1117)](https://link.payoneer.com/Token?t=B8694A2C126D470EAE34E7CC7A0817D9&src=pl)
[![GitHub Sponsors](https://img.shields.io/badge/GitHub-Sponsor-39FF14?style=for-the-badge&logo=githubsponsors&logoColor=white&labelColor=0D1117)](https://github.com/sponsors/saintsun)

| | |
|---|---|
| $5 | Bir Türk kahvesi |
| $10 | Dürüm |
| $25 | İskender |
| $50 | Nargile |
| $100 | Pavyonda bir gece |

---

## Kurulum: detaylar

Komut gereken her şeyi indirip kuruyor:

| | |
|---|---|
| Proje dosyaları | `~/Fatma` içine |
| Homebrew ve Python 3.12 | yoksa kurulur |
| MediaPipe, OpenCV, NumPy | klasör içindeki sanal ortama |
| Yüz işaret modeli | 3.6 MB |

**Gerekenler:** macOS 13 veya üstü bir Mac ve bir kamera. Apple Silicon'da
test edildi; Intel'de çalışması beklenir ama denenmedi.

MediaPipe Python 3.12'de stabil. Daha yeni sürümlerde çoğu zaman wheel yok,
kurulum da bu yüzden o sürümde ısrar ediyor.

<details>
<summary>Terminal kullanmak istemiyorum</summary>

Depodan ZIP'i indir, çift tıklayıp aç, sonra o klasörde Terminal açıp
`./run.sh` çalıştır. Çift tıklanabilir kurulum dosyası henüz yok.
</details>

### Kurulumdan sonra: iki izin

macOS iki izin isteyecek; ikisi de şart.

**Kamera:** ilk açılışta **İzin Ver**'e tıkla.

**Erişilebilirlik:** imleci oynatmak için gerekli. İmleç hiç kıpırdamıyorsa
sebebi neredeyse her zaman budur:

```bash
open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
```

Listede **Terminal**'i (ya da iTerm'i) bul, anahtarı **aç**, sonra terminali
kapatıp aç.

### İlk oturumun

`~/Fatma` klasöründeki **`Fatma.command`'a çift tıkla**. Kontrol paneli
tarayıcıda açılır; her şey oradan yürür ve bir daha terminale dokunman
gerekmez. Sen başlat demeden kamera açılmaz — sayfayı açtın diye ne ışık
yanar ne izin sorulur.

1. **Takibi başlat**'a bas. macOS ilk seferinde kamera izni ister; bir şey
   ters giderse terminalde değil, sayfada görürsün.
2. **Kalibrasyonu başlat**'a bas. Ekran kararır ve bir halka belirir.
3. Halkaya dosdoğru bak ve **bekle** — bakışın oturunca alttaki yazı
   yeşile dönüp HAZIR der. Ancak o zaman **SPACE**'e bas. 13 halkanın hepsi
   için tekrarla. Erken basmak, kalibrasyonu bozmanın en yaygın yolu.
4. Yazdığı iki rakamı oku: doğruluk ve bakış aralığı.
5. **İmleci aç**'a basıp imleci gözlerine devret.

**Ekrana yakın otur**, kafanı sabit tut, yüzüne ışık gelsin. Bunlardan
ilki, uygulamadaki bütün ayarların toplamından daha çok fark eder —
aşağıdaki Hassasiyet bölümüne bak.

---

## Ne yapıyor

- **Sen bakıyorsun, imleç takip ediyor.** İris konumu kameradan izlenip, bir
  kere yaptığın kalibrasyonla ekrana eşleniyor.
- **Göz kırpmak tıklıyor.** Sol kırpma tıklama, sağ kırpma sağ tık, iki kaşı
  birden kaldırmak çift tık. Normal iki gözlü kırpma hiçbir şey yapmaz.
- **Ne kadar doğru olduğunu söylüyor.** Kalibrasyondan sonra kendini çapraz
  doğrulayıp hatayı piksel olarak yazıyor; güvenip güvenmeyeceğini ya da
  tekrar yapman gerektiğini böylece biliyorsun.
- **Bedava ve tamamen yerel.** API anahtarı yok, hesap yok, abonelik yok,
  kurulumdan sonra internet bile gerekmiyor. Görüntü makineden çıkmıyor.

---

## Günlük kullanım

| Tuş | Ne olur |
|---|---|
| **`c`** | kalibrasyonu başlatır — her halkaya bak, SPACE ile onayla |
| **SPACE** | baktığın halkayı onaylar |
| **ESC** | kalibrasyonu iptal eder, öncekini geri yükler |
| **`m`** | imleç kontrolü açık / kapalı |
| **`s`** | yumuşatmayı değiştirir (4 kademe) |
| **`a`** | sürüklenme çıpası — ekranın ortasına bakarken bas |
| **`A`** | çıpayı sıfırlar |
| **`q`** | çıkar |

```bash
./run.sh --dry-run        # takip et ve göster, mouse'a dokunma
./run.sh --benchmark      # 200 kare, aşama maliyetleri, çık
./run.sh --no-gestures    # kırpma ve kaş tıklamalarını kapat
./run.sh --help           # tamamı
```

---

## Kontrol paneli

Takip başlarken bir tarayıcı paneli açılıyor, yalnızca `127.0.0.1` üzerinde.
Adresini açılışta yazıyor:

```
panel: http://127.0.0.1:8766
```

Klavyenin yaptığı her şey orada — kalibrasyonu başlat, bir hedefi onayla,
imleci aç kapa, sürüklenmeyi çıpala — artı canlı hassasiyet göstergeleri:
kare hızı, piksel cinsinden kalibrasyon doğruluğu, ve bakışın sabitken
jitter. **Hassasiyet** sekmesi bir sayı kötü göründüğünde neyi
değiştireceğini söylüyor.

İşe yarıyor çünkü tuşların çalışması için önizleme penceresinin odakta
kalması gerekiyor; panelin gerekmiyor. Kalibrasyonu başka biri telefondan
yürütebilir, kalibre olan kişi hedeflere bakarken.

`--no-panel` kapatır, `--panel-port N` portu sabitler.

---

## Bir şey çalışmıyorsa

**Kalibrasyon halkası hiç görünmüyor.** `./run.sh --calib-window normal`
kullan. Tam ekran pencere daha şık ama bazı pencere yöneticileri ilk karesini
yutuyor.

**İmleç oynamıyor.** İmleç açık mı — `m`'ye basıp önizlemeden kontrol et.
Hâlâ oynamıyorsa Erişilebilirlik izni eksiktir, yukarı bak.

**Berbat şekilde isabetsiz.** Kalibrasyondan sonra yazılan doğruluk satırına
bak. ~120 pikselin üstü, takibin değil kalibrasyonun bozuk olduğu anlamına
gelir: her halkaya tam bakarak ve kafanı oynatmadan yeniden kalibre et. Mesaj
hangi halkanın bozduğunu söylüyor.

**Zamanla kayıyor.** Kafa pozisyonun değişmiştir. Ekranın ortasına bakıp
`a`'ya bas. Yetmezse yeniden kalibre et.

**İmleç titriyor.** `s`'ye birkaç kez basıp yumuşatmayı artır, ya da titreme
eşiğini büyüt: `./run.sh --deadzone 15`.

**Kamera açılmıyor.** Kamera izni, ya da başka bir uygulama kamerayı tutuyor.
Zoom, Teams, Photo Booth'u kapat.

**Güncelleme:**

```bash
cd ~/Fatma && git pull && ./install.sh
```

Kalibrasyonun korunur.

---

## Ücret ve gizlilik

- **Ödenecek bir şey yok.** API anahtarı yok, hesap yok, abonelik yok.
- **Görüntü makineden çıkmıyor.** Yüz takibi yerel çalışıyor. Tek indirme
  kurulumdaki model.
- Uygulama yalnızca kameraya ve Erişilebilirlik'e (imleci oynatmak için)
  dokunuyor.
- `calibration.json` senin gözlerini ve oturuş pozisyonunu tarif ediyor.
  Depoya dahil edilmiyor; issue'lara yapıştırma.

---

## Kaldırma

```bash
rm -rf ~/Fatma
```

O klasörün dışına hiçbir şey kurulmuyor. Homebrew ve Python'a dokunulmuyor
çünkü başka programlar kullanıyor. İstersen terminalinin Kamera ve
Erişilebilirlik kayıtlarını System Settings'ten sil.

---

## Geliştiriciler için

Buradan aşağısı teknik: nasıl çalıştığı, nelerin ölçüldüğü ve yol boyunca
çıkan tuzaklar.

### Akış

```
kamera → thread'li yakalayıcı (sadece en yeni kare) → MediaPipe FaceLandmarker
      → iris çapı biriminde iris ofseti → zamansal medyan → One-Euro filtre
      → thin-plate-spline kalibrasyon → sürüklenme → ölü bölge → CGEventPost

göz kapağı + kaş geometrisi → jest durum makinesi → tıklama olayları
```

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
  panel.py        tarayıcı kontrol paneli (yalnızca 127.0.0.1)
  pointer.py      mouse arka uçları (Quartz / PyAutoGUI / Null)
  app.py          OpenCV + MediaPipe çalışma döngüsü ve CLI
legacy/           refaktör öncesi tek dosyalık sürüm, referans için
parked/           sadece-bakışla kontrol, yazıldı ama bağlanmadı
```

Yalnızca `app.py` cv2, mediapipe veya pyautogui import eder.

### Hassasiyeti ne belirliyor (ölçüldü)

Her şeyi yöneten tek bir sayı var ve acımasız: **bakış sinyali tüm ekran
boyunca yarım iris çapından az yol kat ediyor.** 1470×956 bir ekranda gerçek
bir kalibrasyondan:

| eksen | bakış aralığı | ekran | ölçek |
|---|---|---|---|
| yatay | 0.401 iris çapı | 1470 px | 1 birim = 3663 px |
| dikey | 0.200 iris çapı | 956 px | 1 birim = 4791 px |

Yani 50 piksel imleç doğruluğu için landmark'tan 0.014 iris çapı hassasiyet
gerekiyor. 640×480'de iris çapı ~18 piksel, dolayısıyla **1 piksellik
landmark hatası ≈ 206 ekran pikseli**. Sistem tamamen alt-piksel bölgesinde
çalışıyor ve bunu hiçbir filtre ayarı değiştirmiyor.

Değiştirenler, etki sırasıyla:

1. **Çözünürlük.** 640 → 1280 hatayı yarıya, → 1920 üçte bire indirir.
   Eskiden pahalıydı; imleç çağrısı 114 ms yerine 0.03 ms sürünce bedava
   hale geldi. Varsayılan 1280×720, ve kamera bu modu vermezse uyarı
   basılıyor — sessizce 640'a düşmek hassasiyeti görünmeden yarıya böler.
2. **İki göz birden.** İki göz aynı bakışı bağımsız gürültüyle ölçer,
   ortalaması gürültüyü **√2** azaltır. `--eye auto` (varsayılan) gözler
   uyuştuğu sürece ortalar, ayrıştıklarında tek göze düşer: şaşılık, düşük
   göz kapağı, kadraj dışında kalan göz.
3. **Daha sağlam referans.** İris ofseti bir referans noktasına göre
   ölçülüyor; eskiden iki göz köşesiydi, artık **altı landmark** — her köşe
   artı onun bir üstündeki ve bir altındaki kontur noktası. Ortalama referans
   gürültüsünü ~√3 azaltır, dikey simetri sayesinde inen göz kapağı dikey
   kanala sızmak yerine birinci mertebeden sönümlenir. `--reference corners`
   eski davranışa döner.
4. **13 hedef.** Spline hatası hedeflerden en uzak yerde, kenar ortalarında
   tepe yapar. 13'lük ızgara tam oraya birer hedef koyar.

### Kaynak kullanımı (ölçüldü, M3)

| | |
|---|---|
| Kare başına takip maliyeti | 2–5 ms, yani döngü kameraya bağlı |
| MediaPipe çıkarımı | 640×480 ile 1920×1080 arası 1.5–6 ms |
| İmleç hareketi (Quartz) | 0.03 ms |
| 30 fps bütçesi | 33.3 ms |

### Doğruluğu ölçmek

- **Kalibrasyon doğruluğu** kalibrasyon biter bitmez çapraz doğrulanıyor. Her
  hedef sırayla dışarıda bırakılıp spline kalanlarla yeniden fit ediliyor ve
  dışarıdaki hedef tahmin ediliyor. Bu, haritanın *hedefler arasında* nasıl
  davranacağının dürüst tahmini — imlecin zamanının tamamını geçirdiği yer
  orası.
- **Jitter** önizlemede canlı: bakış sabitken imlecin kaç piksel gezindiği.
  "en iyi" değeri gördüğün en sessiz sabitleme, yani bu kurulumun
  ulaşabildiği gürültü tabanı.

İkisi farklı şey ölçer ve farklı ilacı vardır: kötü kalibrasyon skoru →
yeniden kalibre et; yüksek jitter → çözünürlük ve ortalama.

### Yol boyunca çıkan tuzaklar

**PyAutoGUI kare başına 114 ms yiyordu.** Her public çağrı, çağrıdan sonra
`pyautogui.PAUSE` kadar uyuyan (varsayılanı 0.1 s) ve öncesinde imleç
konumunu sorgulayan bir dekoratörle sarılı. Kare başına bir `moveTo`, kamera
ve model ne kadar hızlı olursa olsun döngüyü ~8 fps'e kilitliyor. `PAUSE = 0`
bunu 11 ms'ye, Core Graphics olayını doğrudan göndermek 0.03 ms'ye indiriyor.
Tek başına bu hata, "takip yavaş" ile "takip kameraya bağlı" arasındaki
farktı.

**Gecikme oturum boyunca birikiyordu.** `VideoCapture.read()` kareleri
sırayla verir, yani kameradan yavaş bir döngü giderek geriye düşer ve
kullandıkça gecikme büyür — hiçbir filtre ayarı bunu düzeltmez. Yakalama
artık kendi thread'inde çalışıyor ve sadece en yeni kareyi tutuyor.

**Kalibrasyon hiç tamamlanamıyordu.** Hedef görüntüsü pencereye yalnızca
hedef indeksi değiştiğinde gönderiliyordu, üstelik pencere her hedefte
yeniden yaratılıp tam ekran özelliği tekrar atanıyordu; pencere yöneticisinin
tam ekran geçişi de o tek kareyi yutuyordu. Pencere boş kalıyor, önizleme de
çoktan kapatılmış olduğu için tuş alacak hiçbir OpenCV penceresi kalmıyor ve
SPACE uygulamaya hiç ulaşmıyordu. Pencere artık tur başına bir kez açılıp her
karede yeniden boyanıyor.

**Kalibrasyon penceresi tahminle ölçülüyordu.** Tuval yüksekliği, ekran
yüksekliğinden varsayılan 25 piksellik menü çubuğu çıkarılarak
hesaplanıyordu. Ölçtüm: menü çubuğu 33 — üstelik tam ekran bir OpenCV
penceresi menü çubuğu dahil tüm ekranı kaplıyor. Yani çıkarma işlemi hem
görüntüyü küçültüp OpenCV'ye %2.7 gerdiriyor, hem de her noktayı
kalibratörün onun için kaydettiği konumdan kaydırıyordu. Pencerenin
dikdörtgeni artık pencere sunucusundan geliyor, görüntü tam o boyutta
çiziliyor, ve bir nokta istenen yere ulaşamadığında *ulaştığı* konum
kaydediliyor. Bakış ölçümünü noktanın hiç bulunmadığı bir konumla
eşleştirmek, kalibrasyonun tek bir yerde bariz yanlış olmak yerine her
yerde hafifçe yanlış olmasına yol açar.

**MediaPipe çıkışta kilitleniyordu.** Landmarker, seri bir dispatcher
arkasında yerel kaynak tutuyor. Çöp toplayıcıya bırakılınca finalizer'ı
çoktan kapanmakta olan bir executor'ı bekliyor ve süreç çıkmak yerine
asılıyor. Artık açıkça kapatılıyor; `close()` idempotent olduğu için
finalizer boşa dönüyor.

**`np.eye(n) * np.inf` köşegen dışında NaN.** Çakışan kalibrasyon noktalarını
bulmak için mesafe matrisinin köşegenini maskelemekte kullanılıyordu. `0 * inf`
NaN olduğu için köşegen dışındaki her hücre NaN oluyor, karşılaştırma hiç
tetiklenmiyor ve bozuk kalibrasyon çözücüye kadar gidip orada yanıltıcı bir
mesajla patlıyordu.

**Döngü hızlanınca zaman damgaları çakıştı.** MediaPipe'ın video modu
ilerlemeyen zaman damgasını reddediyor, ve her kareden 114 ms kalkınca iki
karenin aynı milisaniyeye düşmesi varsayımsal olmaktan çıktı.

**Kalibrasyonu yarıda bırakmak çalışanı siliyordu.** `c`'ye basmak haritayı
anında temizlediği için ESC ile geri çıkan kullanıcı elinde hiçbir şey
kalmadan kalıyordu. Önceki harita artık kenara alınıp iptalde ve başarısız
fit'te geri yükleniyor.

### Testler

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

443 test. Saf mantık modüllerinde kapsam %98–100; kapsanmayan kısım kamera ve
gerçek mouse gerektiren G/Ç.

```bash
python -m pytest -m "not slow"     # gerçek çıkarım yapan testleri atla
python -m pytest --cov=fatma       # kapsam raporu
```

Testler sentetik bir yüz üretiyor (`tests/conftest.py`): göz köşeleri tam
0.1, iris merkezleri tam 0.3 birim arayla, böylece türetilen her ölçüt
tolerans yerine kesin sayıyla doğrulanıyor. `tests/test_precision.py` simüle
landmark gürültüsü enjekte edip sinyalin yayılımını ölçüyor; √2 kazancı ve
altı-landmark referans kamerasız böyle kanıtlandı.

### Ekrana yakın otur

Doğruluğu her ayardan daha çok belirleyen tek şey ve yazılımda değil:
**ekrana ne kadar uzak oturduğun.** Uzaklaştıkça ekran daha küçük bir görüş
açısı kaplar, gözlerin onu taramak için daha az döner, ve bakış sinyalinin
tamamı gürültünün içine çöker.

Aynı makinede ölçülen iki oturum:

| | yatay | dikey | dikey amplifikasyon |
|---|---|---|---|
| yakın oturunca | 0.401 | 0.200 | birim başına 4 791 px |
| geride oturunca | 0.175 | 0.079 | birim başına 12 095 px |

İkinci kalibrasyon her kontrol noktasından tam geçiyordu ve yine de imleci
hedeften ortalama 208 px uzağa koydu — çünkü üçte bir piksellik landmark
gürültüsü yüz piksellik imleç hatasına dönüşmüştü. Artıklar bunu hiç
göremez; bakış aralığı görür, o yüzden her kalibrasyondan sonra ölçülüp
raporlanıyor ve dar kalırsa uyarı veriyor.

### Bilinen sınırlar

- **Kafa pozu telafi edilmiyor.** Takip yalnızca göz hareketinden yapılıyor,
  o yüzden kafanı sabit tutman gerekiyor. Kalan en büyük hata kaynağı bu, ve
  MediaPipe kafa dönüşümünü zaten veriyor — sadece bağlanmadı.
- Pratikte yalnızca macOS. Takip çekirdeği taşınabilir, imleç arka ucu ve
  pencere yönetimi değil.
- **Her kontrol bir tuşa basmak,** yani sadece gözünü oynatabilen biri yanında
  birisi olmadan başlatamaz, kalibre edemez, durduramaz. `parked/` içinde
  dwell motoru ve bakış menüsü yazılı, bağlanmayı bekliyor.
- Kalibrasyon her oturuma özel. Çözünürlük değişiminden etkilenmiyor ama
  kamera açısı veya oturuş pozisyonu değişirse yenilenmesi gerekiyor.
- Gerçek yüzlerde doğrulanmadı: simüle hassasiyet kazancının ne kadarının
  hayatta kaldığı, bir kameranın 720p'yi gerçekten 30 fps verip vermediği, ve
  anchor landmark indislerinin senin modelinde gerçekten köşe komşusu olduğu
  (FaceMesh topolojisinden alındı). `--reference corners --eye right
  --width 640` eski davranışa döndürüp A/B yapmanı sağlar.

### Nereye yardım lazım

1. **Kafa pozu telafisi.** `output_facial_transformation_matrixes`'i aç, yaw
   ve pitch ile iris ofsetini düzelt. Kalan en büyük doğruluk kazancı.
2. **Linux ve Windows imleç arka uçları.** `fatma/pointer.py`'da arayüz
   hazır; `QuartzPointer` kopyalanacak örnek.
3. **Sadece-bakışla kontrol.** `parked/README.md`'ye bak.

Yeni mantığı mümkün olduğunca `app.py` dışında tut — donanımsız test
edilemeyen tek modül o.

---

## Lisans

[MIT](LICENSE) © Eren Güneş

[MediaPipe](https://github.com/google-ai-edge/mediapipe) ve
[OpenCV](https://opencv.org/) üzerine kurulu, ikisi de Apache 2.0. Bağımsız
bir proje; Google ile bağlantılı değil, Google tarafından desteklenmiyor.
