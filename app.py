"""
FALCON - Application Launcher & Entrypoint
Runs both:
  1. streamlit run app.py  (Streamlit Cloud & Procfile)
  2. python app.py         (Local CLI direct launch)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import streamlit as st
    from src.dashboard import render_dashboard

    # Check if executed inside Streamlit runtime
    render_dashboard()
except Exception as e:
    # If launched as a standard python script, launch Streamlit process
    import subprocess
    print("Launching FALCON Dashboard via Streamlit runtime...")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")],
        cwd=ROOT
    )
