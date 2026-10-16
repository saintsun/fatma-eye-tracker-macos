#!/usr/bin/env bash
# Fatma: deviceless eye tracker — remover.
#
#   curl -fsSL https://raw.githubusercontent.com/saintsun/fatma-eye-tracker-macos/main/uninstall.sh | bash
#
# Takes out everything Fatma installed and nothing else. Homebrew and
# Python are left alone: other programs use them.
set -uo pipefail

DEST="${FATMA_DEST:-$HOME/Fatma}"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "Fatma kaldiriliyor"
echo "  klasor: $DEST"
echo

if [ "${1:-}" != "--yes" ] && [ -t 0 ]; then
  printf 'Silinsin mi? Kalibrasyonunuz da gider. [e/H] '
  read -r answer
  case "$answer" in
    e|E|y|Y) ;;
    *) echo "Vazgecildi."; exit 0 ;;
  esac
fi

# Stop it first, or the camera light stays on after the files are gone.
pkill -f '[-]m fatma' 2>/dev/null && echo "  calisan surec durduruldu"

if [ -d "$DEST/Fatma.app" ]; then
  /usr/bin/osascript -e 'tell application "Fatma" to quit' >/dev/null 2>&1 || true
fi

if [ -d "$DEST" ]; then
  rm -rf "$DEST"
  echo "  $DEST silindi"
else
  echo "  $DEST zaten yok"
fi

# Forget the bundle, so Finder and Spotlight stop offering a dead app.
LSREG="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
[ -x "$LSREG" ] && "$LSREG" -u "$DEST/Fatma.app" >/dev/null 2>&1

cat <<MSG

Bitti.

Dokunulmayanlar: Homebrew, Python ve paketleri (baska programlar kullaniyor).
System Settings > Privacy & Security altindaki Kamera ve Erisilebilirlik
kayitlarini elle silebilirsiniz; macOS onlari bir uygulama kaldirilinca
kendiliginden temizlemiyor.
MSG
