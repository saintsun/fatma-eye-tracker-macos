#!/usr/bin/env bash
# Fatma: deviceless eye tracker — launcher.
# Creates the virtualenv on first run, then starts the tracker.
# Any arguments are forwarded, e.g.:  ./run.sh --eye left --dry-run
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${FATMA_PYTHON:-}"
if [ -z "$PYTHON_BIN" ]; then
  # MediaPipe ships stable wheels for 3.12; newer interpreters often have none.
  for candidate in python3.12 /opt/homebrew/bin/python3.12 \
                   /usr/local/bin/python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [ ! -d ".venv" ]; then
  if [ -z "$PYTHON_BIN" ]; then
    echo "No suitable Python found. Install Python 3.12:" >&2
    echo "  brew install python@3.12" >&2
    exit 1
  fi
  echo "Creating venv with $PYTHON_BIN ..."
  "$PYTHON_BIN" -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

if [ ! -f "face_landmarker.task" ]; then
  echo "Downloading the face landmark model ..."
  curl -fL -o face_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
fi

exec .venv/bin/python -m fatma "$@"
