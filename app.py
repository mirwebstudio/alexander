"""Compatibility entry point for running Streamlit from the repository root."""
import os
import runpy
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent / "ALEXANDER_AI_FINAL"
if not (PROJECT_DIR / "app.py").is_file():
    raise FileNotFoundError(
        f"Alexander app entry point not found: {PROJECT_DIR / 'app.py'}"
    )

sys.path.insert(0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)
runpy.run_path(str(PROJECT_DIR / "app.py"), run_name="__main__")
