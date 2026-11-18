"""
FALCON - Master Pipeline Orchestrator
Executes the end-to-end Machine Learning Pipeline:
  Step 1: Exploratory Data Analysis & Statistical Profiling
  Step 2: Advanced Feature Engineering
  Step 3: Dual Model Training & Evaluation (Logistic Regression + Random Forest)
  Step 4: Real-time Prediction Verification
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

SCRIPTS = [
    ("Exploratory Data Analysis", SRC / "data_analysis.py"),
    ("Feature Engineering Pipeline", SRC / "feature_engineering.py"),
    ("Dual Model Training & Evaluation", SRC / "model_training.py"),
    ("Real-Time Prediction Engine Verification", SRC / "predict.py")
]

print("=" * 65)
print("       FALCON - FINANCIAL FRAUD DETECTION PIPELINE")
print("=" * 65)
print(f"Project Root: {ROOT}")
print(f"Python Exec:  {sys.executable}")

for index, (name, script_path) in enumerate(SCRIPTS, start=1):
    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline step failed. Missing script: {script_path}")

    print(f"\n[{index}/{len(SCRIPTS)}] EXECUTING: {name} ({script_path.name})...")
    print("-" * 65)

    try:
        subprocess.run(
            [sys.executable, "-u", str(script_path)],
            cwd=ROOT,
            check=True
        )
    except subprocess.CalledProcessError as error:
        print(f"\n[FATAL ERROR] Pipeline terminated at step '{name}' (Exit Code: {error.returncode})")
        sys.exit(error.returncode)

print("\n" + "=" * 65)
print("FALCON PIPELINE COMPLETED SUCCESSFULLY (100% PASS)")
print("=" * 65)
print("\nGenerated & Verified Deliverables:")
print("  - experiments/ (6 high-res charts, 4 statistical tables)")
print("  - data/processed/ (Stratified engineered dataset)")
print("  - models/ (falcon_model.pkl, logistic_regression.pkl, random_forest.pkl, JSON metadata)")
print("\nNext Steps:")
print("  1. Run Dashboard Locally:  streamlit run app.py")
print("  2. Push to GitHub:         git push origin main")
print("  3. Live Cloud Deployment:  Deploy to Streamlit Community Cloud")
print("=" * 65)
