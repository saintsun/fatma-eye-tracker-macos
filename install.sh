#!/usr/bin/env bash
# Fatma: deviceless eye tracker — one-command installer.
#
#   curl -fsSL https://raw.githubusercontent.com/saintsun/fatma-eye-tracker-macos/main/install.sh | bash
#
# Installs into ~/Fatma. Safe to re-run: it updates in place and keeps your
# calibration.
set -euo pipefail

REPO="${FATMA_REPO:-https://github.com/saintsun/fatma-eye-tracker-macos}"
DEST="${FATMA_DEST:-$HOME/Fatma}"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
die() { printf '\n\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || die "Fatma su an yalnizca macOS'ta calisiyor."

say "1/6  Proje dosyalari"
if [ -d "$DEST/.git" ]; then
  git -C "$DEST" pull --ff-only
elif [ -d "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ]; then
  die "$DEST dolu ama git deposu degil. Tasiyin ya da silin, tekrar deneyin."
else
  command -v git >/dev/null 2>&1 || die "git bulunamadi. Xcode komut satiri
araclarini kurun:  xcode-select --install"
  git clone --depth 1 "$REPO" "$DEST"
fi
cd "$DEST"

say "2/6  Python 3.12"
PYTHON=""
for c in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
  command -v "$c" >/dev/null 2>&1 && { PYTHON="$c"; break; }
done
if [ -z "$PYTHON" ]; then
  command -v brew >/dev/null 2>&1 || {
    say "     Homebrew kuruluyor"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null || true)"
  }
  say "     Python 3.12 kuruluyor"
  brew install python@3.12
  PYTHON="$(brew --prefix)/opt/python@3.12/bin/python3.12"
fi
# MediaPipe ships stable wheels for 3.12; newer interpreters often have none.
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)' \
  || die "Python 3.12 gerekiyor, bulunan: $("$PYTHON" -V)"

say "3/6  Bagimliliklar"
[ -d .venv ] || "$PYTHON" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

say "4/6  Yuz modeli (3.6 MB)"
[ -f face_landmarker.task ] || curl -fL --progress-bar -o face_landmarker.task "$MODEL_URL"

say "5/6  Uygulama paketi"
# Fatma.app is what makes the double-click open no terminal, and the icon
# is what stops the Dock showing a Python rocket.
.venv/bin/python make_icon.py >/dev/null 2>&1 || true
.venv/bin/python make_app.py  >/dev/null 2>&1 || true
[ -d Fatma.app ] && echo "     Fatma.app hazir" || echo "     Fatma.app kurulamadi (run.sh ile calisir)"

say "6/6  Kontrol"
.venv/bin/python -m fatma --version

cat <<EOF

Kurulum bitti.  Klasor: $DEST

Baslatmak icin: Fatma.app'e cift tiklayin.
    open "$DEST/Fatma.app"

Terminalden gormek isterseniz: cd $DEST && ./run.sh

Ilk calistirmada macOS iki izin isteyecek; ikisi de sart:
  • Kamera
  • Erisilebilirlik  (imleci oynatabilmek icin)

Izinler sorulunca izin verin. Hersey panelden yonetiliyor; terminal gerekmez.

Kaldirmak icin:
    $DEST/uninstall.sh
EOF
