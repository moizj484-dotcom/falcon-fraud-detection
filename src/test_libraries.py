"""
FALCON - Library Compatibility & Installation Verifier
"""

import sys
import numpy
import pandas
import matplotlib
import sklearn
import joblib
import streamlit
import plotly

print("=" * 55)
print("       FALCON ENVIRONMENT & DEPENDENCY AUDIT")
print("=" * 55)
print(f"Python:       {sys.version.split()[0]}")
print(f"NumPy:        {numpy.__version__}")
print(f"Pandas:       {pandas.__version__}")
print(f"Matplotlib:   {matplotlib.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"Joblib:       {joblib.__version__}")
print(f"Streamlit:    {streamlit.__version__}")
print(f"Plotly:       {plotly.__version__}")
print("-" * 55)
print("STATUS: ALL LIBRARIES VERIFIED & READY FOR FALCON PIPELINE")
print("=" * 55)
