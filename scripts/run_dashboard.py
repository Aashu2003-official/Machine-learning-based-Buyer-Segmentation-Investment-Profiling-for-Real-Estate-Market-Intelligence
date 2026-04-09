from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = PROJECT_ROOT / '.packages'
SRC_DIR = PROJECT_ROOT / 'src'

if PACKAGES_DIR.exists():
    sys.path.insert(0, str(PACKAGES_DIR))
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from streamlit.web.cli import main as streamlit_main


if __name__ == '__main__':
    sys.argv = ['streamlit', 'run', str(PROJECT_ROOT / 'app.py'), *sys.argv[1:]]
    raise SystemExit(streamlit_main())
