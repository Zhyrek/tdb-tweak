#!/bin/sh
# Double-click to start tdb-tweak on macOS (the first run installs it).
cd "$(dirname "$0")" || exit 1
for p in python3.13 python3.12 python3.11 python3; do
    if command -v "$p" >/dev/null 2>&1; then exec "$p" launch.py "$@"; fi
done
echo "Python 3.11 or newer is needed: https://www.python.org/downloads/"; read -r _
