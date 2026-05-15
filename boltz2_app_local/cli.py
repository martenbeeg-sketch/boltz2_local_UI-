from __future__ import annotations

import os
import sys
from pathlib import Path

from streamlit.web.cli import main as streamlit_main


def main() -> int:
    repo_dir = Path(__file__).resolve().parent.parent
    app_path = repo_dir / "app.py"
    os.chdir(repo_dir)
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=false",
        "--browser.serverAddress=localhost",
    ]
    return streamlit_main()
