#!/bin/sh
# Start tdb-tweak on Linux (the first run installs it). Run once with --desktop to add a menu entry.
cd "$(dirname "$0")" || exit 1
if [ "$1" = "--desktop" ]; then
    mkdir -p "$HOME/.local/share/applications"
    cat > "$HOME/.local/share/applications/tdb-tweak.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=tdb-tweak
Comment=Hand-tune CALPHAD TDB parameters against a live phase diagram
Exec=$(pwd)/tdb-tweak.sh
Terminal=false
Categories=Science;
DESK
    echo "Added tdb-tweak to the applications menu."; exit 0
fi
for p in python3.13 python3.12 python3.11 python3; do
    if command -v "$p" >/dev/null 2>&1; then exec "$p" launch.py "$@"; fi
done
echo "Python 3.11 or newer is needed (e.g. sudo apt install python3 python3-venv)"
