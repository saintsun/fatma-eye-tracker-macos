#!/usr/bin/env bash
# Double-click this file to start Fatma.
#
# It sets the virtualenv up on first run, then opens the control panel in
# your browser. Tracking starts from that page — the camera stays off until
# you press the button.
#
# macOS blocks a downloaded .command on first open: right-click the file,
# choose Open, then Open again.
cd "$(dirname "$0")" || exit 1

printf '\033]0;Fatma\007'
echo "Fatma: deviceless eye tracker"
echo

if [ ! -d .venv ] || [ ! -f face_landmarker.task ]; then
  echo "Ilk calistirma: kurulum yapiliyor, birkac dakika surebilir..."
  echo
  if ! ./install.sh; then
    echo
    echo "Kurulum basarisiz. Yukaridaki hataya bakin."
    echo "Kapatmak icin bu pencereyi kapatin."
    read -r _
    exit 1
  fi
fi

echo "Panel tarayicida acilacak. Takibi oradan baslatin."
echo "Durdurmak icin bu pencereyi kapatin ya da Ctrl+C."
echo
.venv/bin/python -m fatma "$@"
status=$?

if [ $status -ne 0 ]; then
  echo
  echo "Fatma hata ile kapandi (kod $status)."
  echo "Kapatmak icin Enter'a basin."
  read -r _
fi
