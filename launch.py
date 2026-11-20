"""One-click start of tdb-tweak on Windows, macOS and Linux.

The first run creates a private environment (.venv next to this file) and installs tdb-tweak into it (needs an
internet connection, a few minutes); later runs start the program directly.  ``--update`` pulls the latest version
first (git clones), ``--reinstall`` repeats the installation.  Double-click the launcher for your
system: tdb-tweak.bat (Windows), tdb-tweak.command (macOS) or tdb-tweak.sh (Linux).
"""
import os
import subprocess
import sys
import venv

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
WIN = os.name == "nt"
PY = os.path.join(VENV, "Scripts" if WIN else "bin", "python.exe" if WIN else "python")
PYW = os.path.join(VENV, "Scripts", "pythonw.exe") if WIN else PY
STAMP = os.path.join(VENV, ".tdbtweak-installed")


def installed() -> bool:
    if not (os.path.exists(PY) and os.path.exists(STAMP)):
        return False
    with open(STAMP) as f:
        return f.read().strip() == str(os.path.getmtime(os.path.join(HERE, "pyproject.toml")))


def install():
    if sys.version_info < (3, 11):
        sys.exit(f"tdb-tweak needs Python 3.11 or newer (this is {sys.version.split()[0]}).")
    print("Setting up tdb-tweak (first run only; this takes a few minutes)…", flush=True)
    if not os.path.exists(PY):
        venv.EnvBuilder(with_pip=True).create(VENV)
    subprocess.check_call([PY, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([PY, "-m", "pip", "install", "-e", HERE])
    with open(STAMP, "w") as f:
        f.write(str(os.path.getmtime(os.path.join(HERE, "pyproject.toml"))))
    print("Done.", flush=True)


def update():
    """Pull the latest version (for a git clone); the next start reinstalls if the dependencies changed."""
    if not os.path.isdir(os.path.join(HERE, ".git")):
        print("This copy is not a git clone: download the latest version from "
              "https://github.com/Zhyrek/tdb-tweak and replace this folder (keep .venv to skip reinstalling).")
        return
    try:
        subprocess.check_call(["git", "-C", HERE, "pull", "--ff-only"])
    except (OSError, subprocess.CalledProcessError) as ex:
        print(f"Update failed ({ex}). Is git installed, and are there local changes?")


def main():
    if "--update" in sys.argv:
        update()
    if "--reinstall" in sys.argv or not installed():
        install()
    args = [a for a in sys.argv[1:] if a not in ("--reinstall", "--update")]
    if WIN and os.path.exists(PYW):
        # no console window for the program itself
        subprocess.Popen([PYW, "-m", "tdb_tweak", *args], cwd=HERE,
                         creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
    else:
        os.execv(PY, [PY, "-m", "tdb_tweak", *args])


if __name__ == "__main__":
    main()
