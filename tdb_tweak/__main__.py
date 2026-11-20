import sys

from .app import main

if __name__ == "__main__":       # worker processes (Windows/macOS "spawn") re-import this module: no second window
    sys.exit(main())
